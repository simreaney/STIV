//  PoseResidualTests.swift
//  Does the drift detector detect drift?
//
//  The scene is textured static ground -- the bank -- with a LiDAR depth map,
//  rendered from a swaying handheld pose.  The checker is then told either the
//  true pose, in which case there is nothing to find, or the true pose with a
//  small error added, which is what VIO drift looks like over open water.
//  The error is known, so the checker's measurement of it can be checked.

import CoreVideo
import StivKit
import XCTest
@testable import STIV

final class PoseResidualTests: XCTestCase {
    private let intrinsics = PinholeIntrinsics(horizontalFovDegrees: 60, width: 480, height: 360)
    /// Same 4:3 aspect as the image, as on shipped devices; 3 image px per depth px.
    private let depthSize = (width: 160, height: 120)
    private let ground = WaterPlane(y: 0)
    /// Features of a few centimetres: 5-10 px at this range, so a 21 px patch
    /// holds enough texture to align.
    private let texture = SurfaceTexture(spacings: [0.08, 0.03], seed: 11)
    private let sway = handheld(fromHeight: 1.5, downDegrees: 40, scale: 0.5)

    /// A yaw error that moves the image about two pixels at the centre.
    private var twoPixelYaw: Double { 2 / intrinsics.fx }

    private struct Run {
        let checker: PoseResidualChecker
        let corrections: [PoseCorrection]
        /// Per frame, the pose the checker was told and the true one.
        let told: [CameraPose]
        let truth: [CameraPose]
    }

    private func frameBuffer(at t: Double) -> CVPixelBuffer {
        let buffer = makeYCbCrBuffer(width: intrinsics.width, height: intrinsics.height)
        renderPlane(into: buffer, intrinsics: intrinsics, pose: sway(t), planeY: ground.y,
                    texture: texture)
        return buffer
    }

    /// A featureless frame: every alignment fails on it, as when the bank
    /// leaves the shot.
    private lazy var greyFrame: CVPixelBuffer = {
        let buffer = makeYCbCrBuffer(width: intrinsics.width, height: intrinsics.height)
        writeLuma(buffer) { _, _ in 128 }
        return buffer
    }()

    /// Arm `checker` on the frame at t = 0.
    private func arm(_ checker: PoseResidualChecker, avoid: [ImageSegment] = [],
                     avoidRadiusPx: Double = 90, waterPolygon: [PixelPoint]? = nil) {
        let reference = sway(0)
        let referenceProjection = WaterPlaneProjection(intrinsics: intrinsics, pose: reference,
                                                       plane: ground)
        let (depth, confidence) = makeDepthBuffers(width: depthSize.width,
                                                   height: depthSize.height) { x, y in
            planeDepth(x: x, y: y, depthSize: depthSize, projection: referenceProjection)
        }
        let armed = withDepthPlane(depthMap: depth, confidenceMap: confidence) { depthPlane in
            withLumaPlane(of: frameBuffer(at: 0)) { luma in
                checker.arm(depth: depthPlane, luma: luma, intrinsics: intrinsics,
                            pose: reference, avoid: avoid, avoidRadiusPx: avoidRadiusPx,
                            waterPolygon: waterPolygon)
            } != nil
        }
        XCTAssertEqual(armed, true, "could not lock the reference frame's buffers")
    }

    /// Check one frame at time `t`, rendered from the true pose (or grey) but
    /// checked against `told`.
    private func check(_ checker: PoseResidualChecker, at t: Double, told: CameraPose,
                       grey: Bool = false) throws -> PoseCorrection {
        try XCTUnwrap(withLumaPlane(of: grey ? greyFrame : frameBuffer(at: t)) { luma in
            checker.check(luma: luma, intrinsics: intrinsics, pose: told,
                          trackingNormal: true, sectionCentreNow: nil)
        })
    }

    /// Arm on the frame at t = 0, then check `frames` more, each rendered from
    /// the true pose but checked against `error(truePose)`.
    private func replay(
        frames: Int = 8,
        maxCorrectionPx: Double = 6,
        frameInterval: Int = 1,
        waterPolygon: [PixelPoint]? = nil,
        checker existing: PoseResidualChecker? = nil,
        error: (CameraPose) -> CameraPose = { $0 }
    ) throws -> Run {
        let checker = existing ?? PoseResidualChecker(maxCorrectionPx: maxCorrectionPx,
                                                      frameInterval: frameInterval)
        arm(checker, waterPolygon: waterPolygon)
        var corrections: [PoseCorrection] = []
        var told: [CameraPose] = []
        var truth: [CameraPose] = []
        for i in 1...frames {
            let t = Double(i) / 30
            let real = sway(t)
            let reported = error(real)
            corrections.append(try check(checker, at: t, told: reported))
            told.append(reported)
            truth.append(real)
        }
        return Run(checker: checker, corrections: corrections, told: told, truth: truth)
    }

    /// Where `world` lands through `pose`, corrected by `correction`.
    private func pixel(_ world: (x: Double, y: Double, z: Double), pose: CameraPose,
                       correction: PoseCorrection = .identity) throws -> PixelPoint {
        try XCTUnwrap(projectToImage(world, intrinsics: intrinsics,
                                     pose: correction.corrected(pose)))
    }

    /// A ground point at the centre of the reference view.
    private func centreAnchor() throws -> (x: Double, y: Double, z: Double) {
        let projection = WaterPlaneProjection(intrinsics: intrinsics, pose: sway(0), plane: ground)
        let p = try XCTUnwrap(projection.imageToWorld(PixelPoint(u: intrinsics.cx,
                                                                 v: intrinsics.cy)))
        return (p.x, ground.y, p.z)
    }

    private func distance(_ a: PixelPoint, _ b: PixelPoint) -> Double {
        hypot(a.u - b.u, a.v - b.v)
    }

    // MARK: - measuring and correcting

    func testAnAccuratePoseLeavesNothingToCorrect() throws {
        let result = try replay()
        let report = result.checker.report
        XCTAssertTrue(report.enabled)
        XCTAssertGreaterThanOrEqual(report.patchCount, 12, "the scene must give the check bank to use")
        XCTAssertEqual(report.failedCount, 0, report.summary())
        let residuals = report.residualPx.sorted()
        XCTAssertLessThan(residuals[residuals.count / 2], 0.3, report.summary())
        XCTAssertLessThan(report.maxCorrectionPx, 0.3,
                          "a correct pose was 'corrected' by \(report.maxCorrectionPx) px")
    }

    func testAPoseErrorIsMeasuredAndUndone() throws {
        let result = try replay { yawedInCameraFrame($0, radians: self.twoPixelYaw) }
        let report = result.checker.report
        XCTAssertEqual(report.failedCount, 0, report.summary())
        // Two pixels at the centre; a yaw moves the edges of a 60-degree view a
        // third as much again, and the reported size is the worst anywhere.
        let injected = PoseCorrection(rotation: SIMD3(0, twoPixelYaw, 0))
            .farFieldShiftPx(intrinsics: intrinsics)
        XCTAssertEqual(injected, 2 * (1 + pow(tan(Double.pi / 6), 2)), accuracy: 0.1)
        XCTAssertEqual(report.maxCorrectionPx, injected, accuracy: 0.75, report.summary())
        XCTAssertEqual(result.checker.status.state, .applied)

        // The point of the correction: composed onto the wrong pose, it puts a
        // point where it really is.
        let anchor = try centreAnchor()
        for i in result.corrections.indices {
            let before = try pixel(anchor, pose: result.told[i])
            let truth = try pixel(anchor, pose: result.truth[i])
            let after = try pixel(anchor, pose: result.told[i], correction: result.corrections[i])
            XCTAssertGreaterThan(distance(before, truth), 1.5,
                                 "frame \(i): the injected error is too small to test with")
            XCTAssertLessThan(distance(after, truth), 0.4,
                              "frame \(i): corrected point is still off")
        }
    }

    /// The correction is fitted to bank a couple of metres away and applied to
    /// water tens of metres away.  A sideways slip of the pose moves the bank
    /// by `f t / Z` -- a few pixels -- and the far water by a tenth of that.
    /// A correction that moved the far field by the bank's amount would put
    /// the river samples *further* from the truth than doing nothing.
    func testATranslationErrorIsNotCarriedIntoTheFarField() throws {
        let slip = 0.01   // metres, sideways in the world
        let result = try replay {
            var m = $0.m
            m[12] += slip
            return CameraPose(columnMajor: m)
        }
        XCTAssertEqual(result.checker.report.failedCount, 0, result.checker.report.summary())

        let near = try centreAnchor()
        // On the water plane, 25 m out along the view.
        let far = (x: near.x * 25 / -near.z, y: ground.y, z: -25.0)
        for i in result.corrections.indices {
            let told = result.told[i], truth = result.truth[i], c = result.corrections[i]
            let nearError = distance(try pixel(near, pose: told), try pixel(near, pose: truth))
            XCTAssertGreaterThan(nearError, 1.0,
                                 "frame \(i): the slip is too small to test with")
            XCTAssertLessThan(distance(try pixel(near, pose: told, correction: c),
                                       try pixel(near, pose: truth)), 0.5,
                              "frame \(i): the near field is not corrected")

            let farBefore = distance(try pixel(far, pose: told), try pixel(far, pose: truth))
            let farAfter = distance(try pixel(far, pose: told, correction: c),
                                    try pixel(far, pose: truth))
            XCTAssertLessThan(farAfter, 0.4,
                              "frame \(i): the far field was moved by the near field's error "
                              + "(\(farBefore) px before, \(farAfter) px after)")
        }
    }

    /// A residual above the cap means the geometry is wrong rather than
    /// shifted; warping the samples further would only hide that.  And the
    /// read-out must say what was *applied*: nothing, so the whole error
    /// remains -- not the near-zero residual of the fit that was refused.
    func testAResidualBeyondTheCapIsReportedNotCorrected() throws {
        let result = try replay(maxCorrectionPx: 1) {
            yawedInCameraFrame($0, radians: self.twoPixelYaw)
        }
        let report = result.checker.report
        XCTAssertEqual(report.failedCount, report.frameCount)
        XCTAssertTrue(result.corrections.allSatisfy(\.isIdentity))
        XCTAssertEqual(result.checker.status.state, .rejected)
        let residuals = report.residualPx.filter(\.isFinite).sorted()
        XCTAssertGreaterThan(residuals[residuals.count / 2], 1.5,
                             "the residual must be the one left by what was applied: \(residuals)")
        XCTAssertGreaterThan(result.checker.status.residualPx, 1.5)
    }

    // MARK: - never a step

    /// A correction that arrives all at once moves every sample in one STI
    /// row; it has to be ramped in over the frames until the next check.
    func testANewCorrectionIsRampedInRatherThanStepped() throws {
        let checker = PoseResidualChecker(frameInterval: 6)
        let run = try replay(frames: 11, checker: checker) {
            yawedInCameraFrame($0, radians: self.twoPixelYaw)
        }
        let shifts = run.corrections.map { $0.farFieldShiftPx(intrinsics: intrinsics) }
        XCTAssertTrue(shifts.prefix(5).allSatisfy { $0 == 0 }, "nothing before the first check")
        let full = try XCTUnwrap(shifts.last)
        XCTAssertGreaterThan(full, 1.5, "the ramp must arrive: \(shifts)")
        for i in 1..<shifts.count {
            // A sixth of the way per frame.  The far-field figure is the
            // largest of nine projected shifts, so it is linear in the ramp
            // only to within a fraction of a per cent.
            XCTAssertLessThanOrEqual(shifts[i] - shifts[i - 1], 1.02 * full / 6,
                                     "a step at frame \(i): \(shifts)")
        }
    }

    /// A check that fails -- the bank left the shot -- holds the last good
    /// correction rather than dropping it, and only after a second of failures
    /// ramps it out.
    func testAFailedCheckHoldsTheLastCorrectionThenRampsItOut() throws {
        let interval = 3
        let checker = PoseResidualChecker(frameInterval: interval)
        arm(checker)
        let error = { (pose: CameraPose) in yawedInCameraFrame(pose, radians: self.twoPixelYaw) }

        var frame = 0
        func next(grey: Bool) throws -> PoseCorrection {
            frame += 1
            let t = Double(frame) / 30
            return try check(checker, at: t, told: error(sway(t)), grey: grey)
        }

        // Three good checks, and the ramp to the last of them completed.
        var good = PoseCorrection.identity
        for _ in 0..<(4 * interval - 1) { good = try next(grey: false) }
        XCTAssertEqual(checker.status.state, .applied)
        let full = good.farFieldShiftPx(intrinsics: intrinsics)
        XCTAssertGreaterThan(full, 1.5)

        // The bank leaves the shot: the correction is held, exactly.
        let failedBefore = checker.report.failedCount
        for _ in 0..<(PoseResidualChecker.holdLimitChecks * interval) {
            XCTAssertEqual(try next(grey: true), good, "frame \(frame): not held")
        }
        XCTAssertEqual(checker.status.state, .holding)
        XCTAssertEqual(checker.report.failedCount - failedBefore,
                       PoseResidualChecker.holdLimitChecks, "every failed check is counted")

        // One failure more and it is withdrawn -- gradually.
        var shifts = [full]
        for _ in 0..<(2 * interval) {
            shifts.append(try next(grey: true).farFieldShiftPx(intrinsics: intrinsics))
        }
        XCTAssertEqual(checker.status.state, .released)
        XCTAssertEqual(shifts.last ?? .nan, 0, accuracy: 1e-12, "released: \(shifts)")
        for i in 1..<shifts.count {
            XCTAssertLessThanOrEqual(shifts[i - 1] - shifts[i], 1.02 * full / Double(interval),
                                     "a step while releasing: \(shifts)")
        }
    }

    // MARK: - the patches and the search

    /// Patches on the water would report the water's motion as the camera's.
    func testNoPatchIsTakenFromInsideTheWaterOutline() throws {
        let everywhere = [PixelPoint(u: -10, v: -10), PixelPoint(u: 500, v: -10),
                          PixelPoint(u: 500, v: 400), PixelPoint(u: -10, v: 400)]
        let result = try replay(frames: 2, waterPolygon: everywhere)
        XCTAssertEqual(result.checker.report.patchCount, 0)
        XCTAssertFalse(result.checker.report.enabled, "an unchecked pose must say so")
    }

    /// A search line is water from end to end.  At close range it spans
    /// hundreds of pixels, and a disc round its centre leaves most of it open
    /// to a patch that will lock onto the moving surface.
    func testPatchesStayClearOfTheWholeSearchLineNotJustItsCentre() {
        let line = ImageSegment(a: PixelPoint(u: 20, v: 200), b: PixelPoint(u: 460, v: 200))
        let radius = 30.0

        let checker = PoseResidualChecker()
        arm(checker, avoid: [line], avoidRadiusPx: radius)
        XCTAssertGreaterThan(checker.patchPixels.count, 0, "nothing left to test with")
        for p in checker.patchPixels {
            XCTAssertGreaterThanOrEqual(line.distance(to: p), radius,
                                        "patch at \(p) sits on the search line")
        }

        // The control: the old rule, a disc at the line's centre, admits
        // patches on the line well away from it.
        let centreOnly = PoseResidualChecker()
        let centre = PixelPoint(u: 240, v: 200)
        arm(centreOnly, avoid: [ImageSegment(a: centre, b: centre)], avoidRadiusPx: radius)
        XCTAssertTrue(centreOnly.patchPixels.contains { line.distance(to: $0) < radius },
                      "the control found no patch on the line; the test proves nothing")
    }

    func testSegmentDistanceIsToTheNearestPointOfTheSegment() {
        let s = ImageSegment(a: PixelPoint(u: 0, v: 0), b: PixelPoint(u: 10, v: 0))
        XCTAssertEqual(s.distance(to: PixelPoint(u: 5, v: 3)), 3, accuracy: 1e-12)
        XCTAssertEqual(s.distance(to: PixelPoint(u: -4, v: 3)), 5, accuracy: 1e-12,
                       "beyond an end, the distance is to the end")
        XCTAssertEqual(s.distance(to: PixelPoint(u: 13, v: -4)), 5, accuracy: 1e-12)
        let point = ImageSegment(a: PixelPoint(u: 1, v: 1), b: PixelPoint(u: 1, v: 1))
        XCTAssertEqual(point.distance(to: PixelPoint(u: 4, v: 5)), 5, accuracy: 1e-12)
    }

    /// The two-stage search is documented as a coarse grid over the full
    /// radius and a fine grid over the coarse winner's neighbours: 33 integer
    /// positions and the parabola's four, where scanning the full radius at
    /// both steps took 109.
    func testThePatchSearchIsTheSizeItIsDocumentedToBe() throws {
        let result = try replay(frames: 1)
        let checker = result.checker
        let patches = checker.report.patchCount
        XCTAssertGreaterThan(patches, 0)
        let perPatch = Double(checker.costEvaluations) / Double(patches)
        XCTAssertLessThanOrEqual(perPatch,
                                 Double(PoseResidualChecker.maxCostEvaluationsPerAlignment),
                                 "\(checker.costEvaluations) evaluations for \(patches) patches")
        XCTAssertEqual(PoseResidualChecker.maxCostEvaluationsPerAlignment, 37)
        XCTAssertGreaterThan(perPatch, 30, "the search must actually run")
    }

    /// One checker serves every recording in a session.  Re-arming must drop
    /// the last recording's correction, or it is applied to the first frames of
    /// the next one before the decimated check catches up.
    func testRearmingForgetsThePreviousRecordingsCorrection() throws {
        let checker = PoseResidualChecker(frameInterval: 6)
        // Eleven frames: the check at the sixth, and its ramp complete.
        let drifted = try replay(frames: 11, checker: checker) {
            yawedInCameraFrame($0, radians: self.twoPixelYaw)
        }
        XCTAssertGreaterThan(drifted.corrections.last!.farFieldShiftPx(intrinsics: intrinsics), 1,
                             "the first recording must end with a correction in hand")

        let next = try replay(frames: 1, checker: checker)
        let first = try XCTUnwrap(next.corrections.first)
        XCTAssertTrue(first.isIdentity, "stale correction carried into a new recording")
        XCTAssertEqual(checker.status.state, .unchecked)
    }

    // MARK: - when there is nothing to correct

    /// The fit itself, on synthetic patches: the phone's sensor, two dozen
    /// patches spread over the frame at `depths`, found where `truth` puts
    /// them plus `noise` pixels of alignment error.  The pose told to the fit
    /// is the uncorrected one.
    private func estimate(depths: (Int) -> Double, truth: PoseCorrection, noise: Double,
                          seed: UInt64) throws -> PoseResidualChecker.CorrectionEstimate {
        let sensor = PinholeIntrinsics(fx: 1450, fy: 1450, cx: 960, cy: 720,
                                       width: 1920, height: 1440)
        var rng = GaussianNoise(seed: seed)
        var observations: [PoseResidualChecker.PatchObservation] = []
        for k in 0..<24 {
            let u = 150 + Double(k % 6) / 5 * (1920 - 300)
            let v = 150 + Double(k / 6) / 3 * (1440 - 300)
            let d = depths(k)
            let world = (x: (u - sensor.cx) / sensor.fx * d,
                         y: -(v - sensor.cy) / sensor.fy * d, z: -d)
            let predicted = try XCTUnwrap(projectToImage(world, intrinsics: sensor, pose: .still))
            let found = try XCTUnwrap(projectToImage(world, intrinsics: sensor,
                                                     pose: truth.corrected(.still)))
            observations.append(.init(world: world, predicted: predicted,
                                      measured: PixelPoint(u: found.u + noise * rng.next(),
                                                           v: found.v + noise * rng.next())))
        }
        return try XCTUnwrap(PoseResidualChecker.estimateCorrection(
            observations, intrinsics: sensor, pose: .still))
    }

    /// How far apart two corrections put the river, 20 m out, pixels.
    private func farFieldDisagreement(_ a: PoseCorrection, _ b: PoseCorrection) -> Double {
        let sensor = PinholeIntrinsics(fx: 1450, fy: 1450, cx: 960, cy: 720,
                                       width: 1920, height: 1440)
        var worst = 0.0
        for ray in PoseCorrection.farRays(intrinsics: sensor, depthM: 20) {
            guard let p = projectToImage(ray, intrinsics: sensor, pose: a.corrected(.still)),
                  let q = projectToImage(ray, intrinsics: sensor, pose: b.corrected(.still))
            else { continue }
            worst = max(worst, hypot(p.u - q.u, p.v - q.v))
        }
        return worst
    }

    private let oneDepth: (Int) -> Double = { _ in 2.0 }
    /// From half a metre to five, in no particular order across the frame.
    private let manyDepths: (Int) -> Double = { k in 0.5 + 4.5 * Double((k * 7) % 24) / 23 }

    /// With every patch at one depth, a yaw and a sideways slip look the same
    /// on the bank but not on the river, and a six-parameter fit turns 0.3 px
    /// of alignment noise into a pixel at twenty metres.  The fit must fall
    /// back to rotations, and -- the pose being right -- apply nothing.
    func testACorrectPoseWithPatchesAtOneDepthIsLeftAlone() throws {
        var applied = 0
        for seed in 0..<40 {
            let e = try estimate(depths: oneDepth, truth: .identity, noise: 0.3,
                                 seed: 100 + UInt64(seed))
            XCTAssertEqual(e.model, .rotationOnly, "seed \(seed): one depth cannot separate them")
            XCTAssertLessThan(e.farFieldUncertaintyPx, PoseResidualChecker.maximumFarFieldUncertaintyPx)
            if e.significant { applied += 1 }
        }
        XCTAssertLessThanOrEqual(applied, 1, "noise was applied as a correction \(applied) times in 40")
    }

    /// The same with the patches spread over depth, where the full fit is
    /// well conditioned: its own noise is still not a correction.
    func testACorrectPoseIsLeftAloneWhateverTheDepths() throws {
        var applied = 0
        for seed in 0..<40 {
            let e = try estimate(depths: manyDepths, truth: .identity, noise: 0.3,
                                 seed: 200 + UInt64(seed))
            if e.significant { applied += 1 }
        }
        XCTAssertLessThanOrEqual(applied, 1, "noise was applied as a correction \(applied) times in 40")
    }

    /// And a real error is still found and undone, at one depth or many.
    func testATwoPixelYawIsCorrectedAtOneDepthOrMany() throws {
        let yaw = PoseCorrection(rotation: SIMD3(0, 2.0 / 1450, 0))
        for (name, depths) in [("one depth", oneDepth), ("many depths", manyDepths)] {
            for seed in 0..<10 {
                let e = try estimate(depths: depths, truth: yaw, noise: 0.3,
                                     seed: 300 + UInt64(seed))
                XCTAssertTrue(e.significant, "\(name), seed \(seed): a 2 px yaw went uncorrected")
                XCTAssertLessThan(farFieldDisagreement(e.correction, yaw), 0.5,
                                  "\(name), seed \(seed): corrected to the wrong place")
            }
        }
    }

    /// Patches over a range of depths can tell a slip from a turn, and do.
    func testWithPatchesOverManyDepthsTheFullFitIsUsed() throws {
        let slip = PoseCorrection(translation: SIMD3(0.01, 0, 0))
        let e = try estimate(depths: manyDepths, truth: slip, noise: 0.1, seed: 400)
        XCTAssertEqual(e.model, .full)
        XCTAssertTrue(e.significant)
        XCTAssertLessThan(farFieldDisagreement(e.correction, slip), 0.25)
    }
}
