//  PoseResidual.swift
//  Cross-checking ARKit's pose against what the image actually did.
//
//  The desktop pipeline compensates for camera motion by tracking static
//  features and fitting a per-frame image transform (`stiv/stabilise.py`).  This
//  app does not need to: the search-line sample points are fixed *world* points
//  on the water plane, and every frame projects them through its own
//  `ARCamera.transform`.  That is full 6-DoF compensation, for free, and
//  strictly better than a 2-D image warp -- it is also the natural extension of
//  the choice `stiv/geometry.py` already made, to space samples in world metres
//  rather than in pixels.
//
//  But "for free" is doing some work in that sentence.  Visual-inertial odometry
//  drifts, and it drifts worst looking at open water, which offers little
//  parallax and a surface that moves.  If the pose is wrong, the STI is sampling
//  the wrong patch of river and the velocity is wrong with it, silently.  So
//  this file answers one question: **did the pose fully explain the image
//  motion?**  If it did, the residual is zero.
//
//  Five decisions worth defending:
//
//  * **Patches are anchored in 3-D, not in 2-D.**  Predicting where a bank
//    feature moves under a known camera motion requires its depth -- parallax is
//    depth-dependent, and a 2-D tracker sidesteps that only by fitting a global
//    warp a translating camera does not have.  LiDAR supplies the depth, so each
//    patch is a genuine world point and its predicted image position is exact.
//    The cost: only patches inside the sensor's ~5 m reach qualify, so the check
//    validates the *near field* and the far field is assumed to follow.  That is
//    a real limitation, and `MotionReport` says so rather than implying a clean
//    bill of health.
//
//  * **The correction is a pose, not an image warp.**  It is fitted to patches
//    0.3-5 m away and applied to water ten to fifty metres away, and the two
//    halves of a pose error behave differently over that range: a rotation
//    error moves every point in the image by the same amount whatever its
//    depth, while a translation error moves a point by `f t / Z` -- ten times
//    less at 20 m than at 2 m.  A 2-D similarity fitted on the bank cannot tell
//    them apart, so it would carry a near-field translation error to the far
//    field at full size and over-correct the river.  Fitting a small rigid
//    correction (three rotations, three translations) to patches whose depths
//    are known separates the two, and applying it to the sample points' own 3-D
//    positions -- by correcting the pose they are projected through -- gives
//    each depth exactly its own share.  Where the patches' depths are too
//    alike to separate them, the fit's own covariance says so, and it falls
//    back to rotations alone, which is how VIO actually drifts (yaw, from the
//    gyro).  And a fitted correction is applied only if it stands out from the
//    alignment noise: a correct pose is left alone, not jittered by its own
//    noise.  See `estimateCorrection`.
//
//  * **Translation-only Lucas-Kanade** for each patch, not the pyramidal
//    six-parameter tracker of `stabilise.py`.  The pose has already removed the
//    bulk motion, so the leftover is sub-pixel to a few pixels; a small integer
//    search plus a parabolic sub-pixel step resolves that without a pyramid, a
//    library, or a Fourier transform.
//
//  * **Every nth frame.**  Aligning 24 patches over a full search grid is tens
//    of millions of operations; at 60 Hz that would compete with the STI
//    sampling that is the actual measurement.  Drift is a slow phenomenon and
//    10 Hz resolves it completely, so the check is decimated and the STI is not.
//
//  * **Never a step.**  A correction that appears, changes or is withdrawn
//    between one frame and the next moves every sample by that much in one STI
//    row, and the structure tensor reads the step as a streak.  So a new
//    correction is ramped in over the frames until the next check, and a check
//    that fails -- too few patches in view, or a correction over the cap --
//    *holds* the last good correction rather than snapping to none.  A held
//    correction is evidence about the recent past, not about now, so after a
//    second of failed checks it is ramped out too.
//
//  Capped, because a large residual means the geometry is wrong rather than
//  shifted, and shifting it further would only hide that.

import ARKit
import Foundation
import StivKit

/// One patch of static ground with a known world position.
private struct ReferencePatch {
    /// World metres, from the reference frame's depth.  Fixed for the recording.
    let world: (x: Double, y: Double, z: Double)
    /// Where it sat in the reference frame, captured-image pixels.
    let pixel: PixelPoint
    /// Zero-mean, unit-variance luma template, `(2 * half + 1)` square.
    ///
    /// Normalised once, here, so that per-frame alignment is invariant to the
    /// exposure and white-balance changes the camera makes without being asked
    /// -- on a river in changing light, most of them.
    let template: [Float]
}

/// A straight line in captured-image pixels, for keeping patches off the water.
struct ImageSegment: Equatable {
    var a: PixelPoint
    var b: PixelPoint

    /// Shortest distance from `p` to any point of the segment.
    func distance(to p: PixelPoint) -> Double {
        let du = b.u - a.u, dv = b.v - a.v
        let lengthSquared = du * du + dv * dv
        guard lengthSquared > 1e-12 else { return hypot(p.u - a.u, p.v - a.v) }
        let t = min(max(((p.u - a.u) * du + (p.v - a.v) * dv) / lengthSquared, 0), 1)
        return hypot(p.u - (a.u + t * du), p.v - (a.v + t * dv))
    }
}

/// A small rigid correction to a camera pose: the part of the camera's motion
/// ARKit's pose missed.
///
/// Expressed in the camera's own frame.  Where the uncorrected pose puts a world
/// point at camera-space position `P`, the corrected pose puts it at
/// `R(rotation) P + translation`.  Applied by correcting the pose, so every
/// point -- a bank patch two metres away or a sample on the far bank -- is
/// projected through the same corrected camera, and gets the shift its own
/// depth implies.
struct PoseCorrection: Equatable {
    /// Rotation vector, radians: axis times angle, in camera axes.
    var rotation = SIMD3<Double>(0, 0, 0)
    /// Metres, in camera axes.
    var translation = SIMD3<Double>(0, 0, 0)

    static let identity = PoseCorrection()

    var isIdentity: Bool { rotation == .zero && translation == .zero }

    /// The corrected world-from-camera pose.
    ///
    /// With `T = [R | t]` the pose and `D = [Rd | d]` this correction, the
    /// corrected camera-from-world map is `D T^-1`, whose inverse is
    /// `R' = R Rd^T` and `t' = t - R' d`.
    func corrected(_ pose: CameraPose) -> CameraPose {
        guard !isIdentity else { return pose }
        let rd = rotationMatrix(rotation)
        var m = pose.m
        // R' = R * Rd^T, column by column: R'[i][j] = sum_k R[i][k] Rd[j][k].
        var rPrime = [[Double]](repeating: [0, 0, 0], count: 3)
        for i in 0..<3 {
            for j in 0..<3 {
                var s = 0.0
                for k in 0..<3 { s += pose.at(i, k) * rd[j][k] }
                rPrime[i][j] = s
            }
        }
        let t = pose.position
        let d = translation
        for i in 0..<3 {
            for j in 0..<3 { m[4 * j + i] = rPrime[i][j] }
        }
        m[12] = t.x - (rPrime[0][0] * d.x + rPrime[0][1] * d.y + rPrime[0][2] * d.z)
        m[13] = t.y - (rPrime[1][0] * d.x + rPrime[1][1] * d.y + rPrime[1][2] * d.z)
        m[14] = t.z - (rPrime[2][0] * d.x + rPrime[2][1] * d.y + rPrime[2][2] * d.z)
        return CameraPose(columnMajor: m)
    }

    /// Straight-line interpolation, for ramping one correction into the next.
    /// The rotations involved are milliradians, where interpolating the rotation
    /// vector is indistinguishable from interpolating the rotation.
    static func interpolate(_ a: PoseCorrection, _ b: PoseCorrection,
                            _ f: Double) -> PoseCorrection {
        PoseCorrection(rotation: a.rotation + (b.rotation - a.rotation) * f,
                       translation: a.translation + (b.translation - a.translation) * f)
    }

    /// The most this correction moves a distant point anywhere in the image,
    /// pixels: the rotation's share, which is the same at every depth and so
    /// reaches the far bank undiminished.
    ///
    /// Measured, not bounded: points a kilometre out along the rays through
    /// the centre, the edge midpoints and the corners, projected with and
    /// without the correction.  The pose is irrelevant -- the correction lives
    /// in camera axes -- so it is the identity, and the rays are written in
    /// those axes with the convention `worldPoint(depthPixel:...)` uses.
    func farFieldShiftPx(intrinsics: PinholeIntrinsics) -> Double {
        guard rotation != .zero else { return 0 }
        return PoseCorrection(rotation: rotation).shiftPx(intrinsics: intrinsics, depthM: 1000)
    }

    /// The most this correction moves a point `depthM` metres out anywhere in
    /// the frame, pixels: rotation and translation together.
    func shiftPx(intrinsics: PinholeIntrinsics, depthM: Double) -> Double {
        guard !isIdentity else { return 0 }
        let moved = corrected(.still)
        var worst = 0.0
        for ray in Self.farRays(intrinsics: intrinsics, depthM: depthM) {
            guard let a = projectToImage(ray, intrinsics: intrinsics, pose: .still),
                  let b = projectToImage(ray, intrinsics: intrinsics, pose: moved) else { continue }
            worst = max(worst, hypot(b.u - a.u, b.v - a.v))
        }
        return worst
    }

    /// Camera-space points `depthM` out along the rays through the centre,
    /// the edge midpoints and the corners, in the axes `CameraPose.still`
    /// gives them.
    static func farRays(intrinsics: PinholeIntrinsics, depthM: Double)
        -> [(x: Double, y: Double, z: Double)] {
        let w = Double(intrinsics.width), h = Double(intrinsics.height)
        var rays: [(x: Double, y: Double, z: Double)] = []
        for u in [0, 0.5 * w, w] {
            for v in [0, 0.5 * h, h] {
                rays.append((x: (u - intrinsics.cx) / intrinsics.fx * depthM,
                             y: -(v - intrinsics.cy) / intrinsics.fy * depthM,
                             z: -depthM))
            }
        }
        return rays
    }
}

extension CameraPose {
    /// The camera at the origin, looking down -Z: where a correction, which
    /// lives in camera axes, can be measured without a real pose.
    static let still = CameraPose(columnMajor: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
}

/// Rodrigues: the rotation matrix for a rotation vector, as rows.
private func rotationMatrix(_ w: SIMD3<Double>) -> [[Double]] {
    let theta = (w * w).sum().squareRoot()
    guard theta > 1e-15 else { return [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }
    let k = w / theta
    let c = cos(theta), s = sin(theta), v = 1 - c
    return [
        [c + k.x * k.x * v, k.x * k.y * v - k.z * s, k.x * k.z * v + k.y * s],
        [k.y * k.x * v + k.z * s, c + k.y * k.y * v, k.y * k.z * v - k.x * s],
        [k.z * k.x * v - k.y * s, k.z * k.y * v + k.x * s, c + k.z * k.z * v],
    ]
}

/// What the most recent check concluded, for the live read-out.
///
/// Distinct from `MotionReport`, which is the record of the whole run: this is
/// the state *now*, and it says what was applied rather than what was fitted.
struct PoseCheckStatus: Equatable {
    enum State: Equatable {
        /// No check has run yet.
        case unchecked
        /// The newest check's correction is being applied.
        case applied
        /// The newest check found nothing the alignment noise could not
        /// explain: the pose is right, as far as can be seen, and no
        /// correction is applied.
        case negligible
        /// The newest check failed -- too few patches in view -- and the last
        /// good correction is being held.
        case holding
        /// The newest check wanted a correction over the cap: the pose or the
        /// plane is wrong.  The last good correction is being held.
        case rejected
        /// Checks have failed for long enough that the held correction no
        /// longer describes the camera; it is being ramped out.
        case released
    }

    var state: State = .unchecked
    /// RMS distance, pixels, between where the patches were found and where
    /// the pose *with the correction actually being applied* puts them.  Not
    /// the fit's own residual: a rejected correction leaves its whole error in
    /// place, and this says so.
    var residualPx: Double = .nan
    /// How far the applied correction moves a far-field point, pixels.
    var correctionPx: Double = 0
}

/// Cross-checks the pose and reports how much it missed by.
final class PoseResidualChecker {
    /// Patch half-width in captured-image pixels: 21 px square.
    private static let half = 10
    /// Integer search radius of the coarse stage.  The pose should leave far
    /// less than this; the margin exists so a genuine failure is *measured*
    /// rather than clipped.
    private static let searchRadius = 4
    private static let side = 2 * half + 1
    /// Consecutive failed checks after which a held correction is withdrawn:
    /// a second, at the default decimation and 60 fps.
    static let holdLimitChecks = 10
    /// The depth at which a correction's effect on the river is judged: the
    /// far half of a typical cross-section.
    static let farFieldDepthM = 20.0
    /// The most a correction may be uncertain at `farFieldDepthM`, pixels,
    /// before the six-parameter fit gives way to rotations alone.
    static let maximumFarFieldUncertaintyPx = 0.25
    /// Alignment is never trusted to better than this, pixels, whatever the
    /// fit's own residuals claim.
    static let alignmentNoiseFloorPx = 0.1

    /// Check one frame in this many.  See the note on decimation above.
    let frameInterval: Int
    /// Above this correction, the geometry is wrong rather than shifted: report,
    /// do not correct.
    private let maxCorrectionPx: Double

    private var patches: [ReferencePatch] = []
    /// Where the section centre sat in the reference frame, so the reported
    /// frame shift is a displacement rather than a distance from the lens axis.
    private var referenceSectionCentre: PixelPoint?
    /// Reused so per-frame alignment allocates nothing.
    private var scratch = [Float](repeating: 0,
                                  count: PoseResidualChecker.side * PoseResidualChecker.side)
    private var seen = 0

    /// The correction being ramped towards, and where the ramp started.
    private var target = PoseCorrection.identity
    private var rampFrom = PoseCorrection.identity
    private var rampFrames = 0
    private var consecutiveFailures = 0

    private(set) var report = MotionReport()
    /// The correction handed back for the current frame.
    private(set) var latest = PoseCorrection.identity
    private(set) var status = PoseCheckStatus()
    /// Cost-function evaluations so far.  For the tests: it is how the search
    /// pattern's size is checked against its documentation.
    private(set) var costEvaluations = 0

    /// Where the chosen patches sat in the reference frame.
    var patchPixels: [PixelPoint] { patches.map(\.pixel) }

    init(maxCorrectionPx: Double = 6.0, frameInterval: Int = 6) {
        self.maxCorrectionPx = maxCorrectionPx
        self.frameInterval = max(1, frameInterval)
        report.enabled = false
    }

    // MARK: - setup

    /// Choose patches on static ground from the reference frame.
    ///
    /// - Parameter avoid: image segments the patches must stay clear of -- the
    ///   projected search lines, end to end, which are water by construction.
    ///   Same reasoning as `stabilise.build_mask`, which excludes a corridor
    ///   around the search lines: a tracker that locks onto the moving water
    ///   surface reports the water's motion as the camera's, and the velocity
    ///   collapses toward zero.  At close range a search line spans hundreds of
    ///   pixels, so a disc around its centre alone would leave most of it open.
    /// - Parameter waterPolygon: an optional outline of the water in
    ///   captured-image pixels.  Worth asking the user for -- without it only
    ///   the search-line corridors are excluded, and a wide river fills the
    ///   frame.
    func arm(
        frame: ARFrame,
        avoid: [ImageSegment],
        avoidRadiusPx: Double = 90,
        waterPolygon: [PixelPoint]? = nil,
        sectionCentre: PixelPoint? = nil,
        maxPatches: Int = 24
    ) {
        disarm(sectionCentre: sectionCentre)

        // No depth: no 3-D anchor, so no honest prediction.  The pose still
        // does the compensating; it simply goes unverified, and the report
        // states that rather than implying a clean check.
        guard let sceneDepth = frame.sceneDepth ?? frame.smoothedSceneDepth else { return }

        let intrinsics = frame.camera.stivIntrinsics
        let pose = frame.camera.stivPose
        _ = withDepthPlane(depthMap: sceneDepth.depthMap,
                           confidenceMap: sceneDepth.confidenceMap) { depthPlane in
            withLumaPlane(of: frame.capturedImage) { luma in
                arm(depth: depthPlane, luma: luma, intrinsics: intrinsics, pose: pose,
                    avoid: avoid, avoidRadiusPx: avoidRadiusPx, waterPolygon: waterPolygon,
                    sectionCentre: sectionCentre, maxPatches: maxPatches)
            }
        }
    }

    /// The same, from planes already in hand -- which, unlike an `ARFrame`, a
    /// test can construct.
    func arm(
        depth depthPlane: DepthPlane,
        luma: LumaPlane,
        intrinsics: PinholeIntrinsics,
        pose: CameraPose,
        avoid: [ImageSegment],
        avoidRadiusPx: Double = 90,
        waterPolygon: [PixelPoint]? = nil,
        sectionCentre: PixelPoint? = nil,
        maxPatches: Int = 24
    ) {
        disarm(sectionCentre: sectionCentre)
        var candidates: [(patch: ReferencePatch, variance: Double)] = []

        let scaleU = Double(intrinsics.width) / Double(depthPlane.width)
        let scaleV = Double(intrinsics.height) / Double(depthPlane.height)
        let margin = Double(Self.half + Self.searchRadius + 2)

        // A coarse grid over the depth map: one candidate per cell, so patches
        // spread over the frame instead of clustering on the one high-contrast
        // rock.
        let step = 6
        for dy in Swift.stride(from: 2, to: depthPlane.height - 2, by: step) {
            for dx in Swift.stride(from: 2, to: depthPlane.width - 2, by: step) {
                guard let depth = depthPlane.highConfidenceDepth(x: dx, y: dy) else {
                    continue
                }
                let range = Double(depth)
                guard range >= WaterPlaneFitter.minRangeM,
                      range <= WaterPlaneFitter.maxRangeM else { continue }

                let centre = PixelPoint(u: (Double(dx) + 0.5) * scaleU,
                                        v: (Double(dy) + 0.5) * scaleV)
                guard centre.u >= margin, centre.v >= margin,
                      centre.u <= Double(luma.width) - margin,
                      centre.v <= Double(luma.height) - margin else { continue }

                // Keep clear of the water.
                if avoid.contains(where: { $0.distance(to: centre) < avoidRadiusPx }) {
                    continue
                }
                if let waterPolygon, contains(polygon: waterPolygon, point: centre) {
                    continue
                }

                let u = Int(centre.u.rounded())
                let v = Int(centre.v.rounded())
                // A textureless patch cannot be aligned, and one that cannot be
                // aligned reports a spurious *zero* residual -- which is worse
                // than reporting nothing.
                let variance = luma.patchVariance(centreU: u, centreV: v, half: Self.half)
                guard variance > 25 else { continue }

                let world = worldPoint(
                    depthPixel: (dx, dy),
                    depthSize: (depthPlane.width, depthPlane.height),
                    depth: depth,
                    intrinsics: intrinsics,
                    pose: pose)

                guard let template = normalisedTemplate(luma: luma, centre: centre) else {
                    continue
                }
                candidates.append((ReferencePatch(world: world, pixel: centre,
                                                  template: template), variance))
            }
        }

        // Strongest texture first: that is where sub-pixel alignment is sharpest.
        patches = candidates.sorted { $0.variance > $1.variance }
            .prefix(maxPatches).map(\.patch)
        report.patchCount = patches.count
        report.enabled = !patches.isEmpty
    }

    /// Forget the previous recording entirely -- including its last accepted
    /// correction, which would otherwise be handed back on the first frames of
    /// the next one.
    func disarm(sectionCentre: PixelPoint? = nil) {
        patches.removeAll()
        report = MotionReport()
        report.enabled = false
        referenceSectionCentre = sectionCentre
        seen = 0
        target = .identity
        rampFrom = .identity
        rampFrames = 0
        consecutiveFailures = 0
        latest = .identity
        status = PoseCheckStatus()
    }

    /// Cut at the patch's exact sub-pixel centre, not the nearest pixel.  The
    /// world anchor is back-projected from that exact point, so a template
    /// rounded to whole pixels would sit up to half a pixel from where its
    /// anchor projects -- a residual in every frame, with a perfect pose.
    private func normalisedTemplate(luma: LumaPlane, centre: PixelPoint) -> [Float]? {
        var values = [Float](repeating: 0, count: Self.side * Self.side)
        var sum: Float = 0
        var sumSquares: Float = 0
        for j in 0..<Self.side {
            for i in 0..<Self.side {
                guard let value = luma.sample(u: centre.u - Double(Self.half) + Double(i),
                                              v: centre.v - Double(Self.half) + Double(j))
                else { return nil }
                values[j * Self.side + i] = value
                sum += value
                sumSquares += value * value
            }
        }
        let n = Float(values.count)
        let mean = sum / n
        let std = max(sumSquares / n - mean * mean, 0).squareRoot()
        guard std > 1 else { return nil }
        for k in values.indices { values[k] = (values[k] - mean) / std }
        return values
    }

    // MARK: - per-frame check

    /// Measure this frame's residual, when it is a frame that gets checked, and
    /// return the correction to apply to this frame's pose.
    ///
    /// Called every frame, decimated or not, so the caller can apply the result
    /// unconditionally: on the frames between checks it is the ramp from the
    /// previous correction towards the newest.
    @discardableResult
    func check(frame: ARFrame, sectionCentreNow: PixelPoint?) -> PoseCorrection {
        guard report.enabled, !patches.isEmpty else { return .identity }
        let camera = frame.camera
        return withLumaPlane(of: frame.capturedImage) { luma in
            check(luma: luma, intrinsics: camera.stivIntrinsics, pose: camera.stivPose,
                  trackingNormal: camera.trackingState.isNormal,
                  sectionCentreNow: sectionCentreNow)
        } ?? latest
    }

    /// The same, from a luma plane already in hand -- which, unlike an
    /// `ARFrame`, a test can construct.
    @discardableResult
    func check(luma: LumaPlane, intrinsics: PinholeIntrinsics, pose: CameraPose,
               trackingNormal: Bool, sectionCentreNow: PixelPoint?) -> PoseCorrection {
        guard report.enabled, !patches.isEmpty else { return .identity }

        if !trackingNormal { report.trackingDegraded = true }

        seen += 1
        if seen % frameInterval == 0 {
            measure(luma: luma, intrinsics: intrinsics, pose: pose,
                    sectionCentreNow: sectionCentreNow)
        }

        // Advance the ramp: the whole change lands over one check interval.
        rampFrames += 1
        let f = min(1, Double(rampFrames) / Double(frameInterval))
        latest = f >= 1 ? target : .interpolate(rampFrom, target, f)
        return latest
    }

    private func retarget(_ next: PoseCorrection) {
        rampFrom = latest
        target = next
        rampFrames = 0
    }

    /// One decimated check: align every patch, fit, and accept or hold.
    private func measure(luma: LumaPlane, intrinsics: PinholeIntrinsics, pose: CameraPose,
                         sectionCentreNow: PixelPoint?) {
        report.frameCount += 1

        var observations: [PatchObservation] = []
        observations.reserveCapacity(patches.count)
        for patch in patches {
            guard let expected = projectToImage(patch.world, intrinsics: intrinsics, pose: pose),
                  let found = align(patch: patch, near: expected, in: luma) else { continue }
            observations.append(PatchObservation(world: patch.world, predicted: expected,
                                            measured: found))
        }

        // How far the section moved in the image.  Not an error -- the pose
        // removes it -- but it is the thing the user can act on: if the section
        // has slid off the water, no amount of correct geometry helps.
        if let sectionCentreNow, let reference = referenceSectionCentre {
            report.shiftPx.append(hypot(sectionCentreNow.u - reference.u,
                                        sectionCentreNow.v - reference.v))
        }

        guard observations.count >= 6,
              let fit = Self.estimateCorrection(observations, intrinsics: intrinsics, pose: pose)
        else {
            report.failedCount += 1
            report.residualPx.append(.nan)
            report.inlierFraction.append(0)
            hold(as: .holding, intrinsics: intrinsics)
            status.residualPx = .nan
            return
        }

        // The size of the correction, measured where it matters most: the
        // far field for its rotation, the nearest patch for its translation.
        let corrected = fit.correction.corrected(pose)
        var nearShift = 0.0
        for i in fit.inliers {
            if let p = projectToImage(observations[i].world, intrinsics: intrinsics,
                                      pose: corrected) {
                nearShift = max(nearShift, hypot(p.u - observations[i].predicted.u,
                                                 p.v - observations[i].predicted.v))
            }
        }
        let magnitude = max(nearShift, fit.correction.farFieldShiftPx(intrinsics: intrinsics))

        if !fit.significant {
            // Nothing the alignment noise cannot explain.  Applying the fit
            // anyway would ramp that noise into the river samples -- jitter
            // added to a pose that needed nothing -- so the correction is
            // none, and any earlier one is ramped out.
            consecutiveFailures = 0
            if !target.isIdentity { retarget(.identity) }
            status.state = .negligible
            status.correctionPx = 0
        } else if magnitude <= maxCorrectionPx {
            consecutiveFailures = 0
            report.maxCorrectionPx = max(report.maxCorrectionPx, magnitude)
            retarget(fit.correction)
            status.state = .applied
            status.correctionPx = fit.correction.farFieldShiftPx(intrinsics: intrinsics)
        } else {
            // A residual this large is not a shift to be undone: the pose or the
            // plane is wrong, and warping the samples on top of that would hide
            // exactly what the user needs to know.
            report.failedCount += 1
            hold(as: .rejected, intrinsics: intrinsics)
        }

        // What the samples are left with: the residual under the correction
        // actually being applied, not under the one that was fitted.
        let applied = target.corrected(pose)
        var sumSquares = 0.0
        var n = 0
        for i in fit.inliers {
            guard let p = projectToImage(observations[i].world, intrinsics: intrinsics,
                                         pose: applied) else { continue }
            sumSquares += pow(p.u - observations[i].measured.u, 2)
                        + pow(p.v - observations[i].measured.v, 2)
            n += 1
        }
        let residual = n > 0 ? (sumSquares / Double(n)).squareRoot() : .nan
        report.residualPx.append(residual)
        report.inlierFraction.append(fit.inlierFraction)
        status.residualPx = residual
    }

    /// A failed check: keep the last good correction, unless checks have been
    /// failing for long enough that it no longer describes the camera.
    private func hold(as state: PoseCheckStatus.State, intrinsics: PinholeIntrinsics) {
        consecutiveFailures += 1
        if consecutiveFailures > Self.holdLimitChecks {
            if !target.isIdentity { retarget(.identity) }
            status.state = .released
        } else {
            status.state = state
        }
        status.correctionPx = target.farFieldShiftPx(intrinsics: intrinsics)
    }

    // MARK: - alignment

    /// Sum of squared differences between the normalised template and a
    /// normalised patch of `luma` centred on `centre`.
    private func cost(_ centre: PixelPoint, template: [Float], luma: LumaPlane) -> Double? {
        costEvaluations += 1
        var sum: Float = 0
        var sumSquares: Float = 0
        let side = Self.side
        for j in 0..<side {
            for i in 0..<side {
                guard let value = luma.sample(u: centre.u - Double(Self.half) + Double(i),
                                              v: centre.v - Double(Self.half) + Double(j))
                else { return nil }
                scratch[j * side + i] = value
                sum += value
                sumSquares += value * value
            }
        }
        let n = Float(side * side)
        let mean = sum / n
        let std = max(sumSquares / n - mean * mean, 0).squareRoot()
        guard std > 1 else { return nil }

        var total: Float = 0
        for k in 0..<template.count {
            let d = template[k] - (scratch[k] - mean) / std
            total += d * d
        }
        return Double(total)
    }

    /// The most cost evaluations one alignment can make: the start, a coarse
    /// grid at step 2 over +-4 (24 more), a fine grid at step 1 over +-1 round
    /// the coarse winner (8 more) and two per axis for the parabola.  33
    /// integer positions where an exhaustive +-4 search would take 81, and the
    /// reach is +-5 px: the coarse radius plus the fine step.
    static let maxCostEvaluationsPerAlignment = 1 + 24 + 8 + 4

    private func align(patch: ReferencePatch, near guess: PixelPoint,
                       in luma: LumaPlane) -> PixelPoint? {
        let margin = Double(Self.half + Self.searchRadius + 2)
        guard guess.u >= margin, guess.v >= margin,
              guess.u <= Double(luma.width) - margin,
              guess.v <= Double(luma.height) - margin else { return nil }

        // Two-stage integer search: step 2 over the full radius, then step 1
        // over the eight neighbours of the coarse winner.  On a cost surface
        // as smooth as a 21 px normalised template makes it, the fine stage
        // finds the same minimum the exhaustive grid would.
        var best = guess
        var bestCost = cost(guess, template: patch.template, luma: luma) ?? .infinity
        guard bestCost.isFinite else { return nil }

        for (stride, radius) in [(2, Self.searchRadius), (1, 1)] {
            let centre = best
            for dv in Swift.stride(from: -radius, through: radius, by: stride) {
                for du in Swift.stride(from: -radius, through: radius, by: stride)
                    where du != 0 || dv != 0 {
                    let candidate = PixelPoint(u: centre.u + Double(du),
                                               v: centre.v + Double(dv))
                    if let c = cost(candidate, template: patch.template, luma: luma),
                       c < bestCost {
                        bestCost = c
                        best = candidate
                    }
                }
            }
        }

        // Sub-pixel refinement by a 3-point parabola per axis.  Cheaper than
        // assembling the Gauss-Newton normal equations, and for a cost surface
        // this smooth near its minimum, indistinguishable from them.
        for axis in 0..<2 {
            let minus = axis == 0 ? PixelPoint(u: best.u - 1, v: best.v)
                                  : PixelPoint(u: best.u, v: best.v - 1)
            let plus = axis == 0 ? PixelPoint(u: best.u + 1, v: best.v)
                                 : PixelPoint(u: best.u, v: best.v + 1)
            guard let cMinus = cost(minus, template: patch.template, luma: luma),
                  let cPlus = cost(plus, template: patch.template, luma: luma) else { continue }
            let denominator = cMinus - 2 * bestCost + cPlus
            guard abs(denominator) > 1e-9 else { continue }
            let delta = min(max(0.5 * (cMinus - cPlus) / denominator, -1), 1)
            if axis == 0 { best.u += delta } else { best.v += delta }
        }
        return best
    }

    // MARK: - robust pose-correction fit

    /// One patch in one frame.
    struct PatchObservation {
        let world: (x: Double, y: Double, z: Double)
        /// Where the uncorrected pose puts the patch.
        let predicted: PixelPoint
        /// Where alignment found it.
        let measured: PixelPoint
    }

    /// Which correction the patches can support.
    enum CorrectionModel: Equatable {
        /// Three rotations and three translations: the patches span enough
        /// depth to tell the two apart.
        case full
        /// Rotations alone: they do not.
        case rotationOnly
    }

    struct CorrectionEstimate {
        var correction: PoseCorrection
        var model: CorrectionModel
        var inliers: [Int]
        var inlierFraction: Double
        /// Alignment noise, pixels, from the fit's own residuals, floored.
        var noisePx: Double
        /// One sigma of the fitted correction's shift at `farFieldDepthM`,
        /// pixels: how far the alignment noise alone could move the river.
        var farFieldUncertaintyPx: Double
        /// The fitted correction's own shift there, pixels.
        var farFieldShiftPx: Double
        /// Whether the correction is distinguishable from that noise.
        var significant: Bool
    }

    /// Least-squares pose correction, with outlier rejection, a choice of
    /// model, and a test of whether there is anything to correct.
    ///
    /// **Outliers.**  A patch that latched onto a moving duck, or onto the
    /// water despite the mask, leaves a residual unlike the rest.  It is
    /// removed twice: once on its raw displacement, so it cannot drag the first
    /// fit, and once on its residual after that fit.  The surviving fraction is
    /// reported -- a low inlier fraction is precisely the signal that the
    /// patches are on the water, the same diagnosis
    /// `StabilisationReport.warnings` makes on the desktop.
    ///
    /// **The model.**  A rotation error moves every depth alike, a translation
    /// error moves near points more than far ones, and only patches at
    /// different depths can tell the two apart.  With the patches all at one
    /// depth, a small yaw and a small sideways slip look identical on the bank
    /// but not on the river, and the six-parameter fit turns half a pixel of
    /// alignment noise into a pixel or more at twenty metres.  So the fit's
    /// covariance is propagated to `farFieldDepthM`; above
    /// `maximumFarFieldUncertaintyPx`, the fit falls back to rotations alone --
    /// how VIO mostly drifts, from the gyro -- whose far-field uncertainty is a
    /// fraction of the noise.
    ///
    /// **The deadband.**  A correct pose still yields a fitted correction:
    /// the noise's own.  Applied, it would ramp jitter into the river samples.
    /// So a correction is applied only if it exceeds its own propagated
    /// uncertainty: its chi-square against the fit's covariance beyond the
    /// 99.9% point.  That is the test in every direction the patches measure
    /// at once, not at one depth: a real centimetre's slip moves the bank by
    /// pixels and the river at twenty metres by a fraction of one, and is
    /// still worth correcting for the water near the bank.  The far field's own
    /// exposure to noise is the model's job, above.
    static func estimateCorrection(_ observations: [PatchObservation],
                                   intrinsics: PinholeIntrinsics,
                                   pose: CameraPose) -> CorrectionEstimate? {
        let lengths = observations.map {
            hypot($0.measured.u - $0.predicted.u, $0.measured.v - $0.predicted.v)
        }
        let threshold = max(2.5 * median(lengths), 1.0)
        var inliers = observations.indices.filter { lengths[$0] <= threshold }
        guard inliers.count >= 6,
              var full = solveCorrection(observations, inliers, parameters: 6,
                                         intrinsics: intrinsics, pose: pose) else { return nil }

        let residuals = inliers.map { i -> Double in
            residual(observations[i], full.correction, intrinsics: intrinsics, pose: pose)
        }
        let cut = max(3 * median(residuals.filter(\.isFinite)), 0.75)
        let kept = zip(inliers, residuals).filter { $0.1 <= cut }.map(\.0)
        if kept.count < inliers.count, kept.count >= 6,
           let refit = solveCorrection(observations, kept, parameters: 6,
                                       intrinsics: intrinsics, pose: pose) {
            full = refit
            inliers = kept
        }

        var chosen = full
        var model = CorrectionModel.full
        var noise = noisePx(observations, inliers, full.correction, parameters: 6,
                            intrinsics: intrinsics, pose: pose)
        var uncertainty = farFieldUncertainty(normal: full.normal, noisePx: noise,
                                              intrinsics: intrinsics)
        if !(uncertainty <= maximumFarFieldUncertaintyPx) {
            guard let rotation = solveCorrection(observations, inliers, parameters: 3,
                                                 intrinsics: intrinsics, pose: pose) else {
                return nil
            }
            chosen = rotation
            model = .rotationOnly
            noise = noisePx(observations, inliers, rotation.correction, parameters: 3,
                            intrinsics: intrinsics, pose: pose)
            uncertainty = farFieldUncertainty(normal: rotation.normal, noisePx: noise,
                                              intrinsics: intrinsics)
        }

        // Significance: chi-square of the parameters against their covariance,
        // and the far-field shift against its own uncertainty.
        let p = chosen.normal.count
        let delta = (0..<p).map { k in
            k < 3 ? chosen.correction.rotation[k] : chosen.correction.translation[k - 3]
        }
        var chiSquare = 0.0
        for r in 0..<p {
            for c in 0..<p { chiSquare += delta[r] * chosen.normal[r][c] * delta[c] }
        }
        chiSquare /= noise * noise
        let critical = p == 6 ? 22.46 : 16.27   // chi-square, 99.9%, 6 and 3 dof
        let shift = chosen.correction.shiftPx(intrinsics: intrinsics, depthM: farFieldDepthM)

        return CorrectionEstimate(
            correction: chosen.correction, model: model, inliers: inliers,
            inlierFraction: Double(inliers.count) / Double(observations.count),
            noisePx: noise, farFieldUncertaintyPx: uncertainty, farFieldShiftPx: shift,
            significant: chiSquare >= critical)
    }

    private static func residual(_ o: PatchObservation, _ c: PoseCorrection,
                                 intrinsics: PinholeIntrinsics, pose: CameraPose) -> Double {
        guard let p = projectToImage(o.world, intrinsics: intrinsics, pose: c.corrected(pose))
        else { return .infinity }
        return hypot(p.u - o.measured.u, p.v - o.measured.v)
    }

    /// Per-coordinate RMS of the fit's residuals, with the degrees of freedom
    /// the fit used taken off, and floored.
    private static func noisePx(_ observations: [PatchObservation], _ indices: [Int],
                                _ c: PoseCorrection, parameters: Int,
                                intrinsics: PinholeIntrinsics, pose: CameraPose) -> Double {
        var sumSquares = 0.0
        for i in indices {
            let r = residual(observations[i], c, intrinsics: intrinsics, pose: pose)
            if r.isFinite { sumSquares += r * r }
        }
        let dof = max(2 * indices.count - parameters, 1)
        return max((sumSquares / Double(dof)).squareRoot(), alignmentNoiseFloorPx)
    }

    /// One sigma of a fitted correction's shift at `farFieldDepthM`, worst
    /// across the frame: the covariance `noise^2 (J^T J)^-1` pushed through
    /// the far-field points' own Jacobian, taken through `projectToImage`.
    private static func farFieldUncertainty(normal a: [[Double]], noisePx: Double,
                                            intrinsics: PinholeIntrinsics) -> Double {
        let p = a.count
        var inverse = [[Double]](repeating: [Double](repeating: 0, count: p), count: p)
        for k in 0..<p {
            var e = [Double](repeating: 0, count: p)
            e[k] = 1
            guard let column = solve(a, e) else { return .infinity }
            for r in 0..<p { inverse[r][k] = column[r] }
        }
        let epsilon = 1e-5
        var worst = 0.0
        for ray in PoseCorrection.farRays(intrinsics: intrinsics, depthM: farFieldDepthM) {
            guard let p0 = projectToImage(ray, intrinsics: intrinsics, pose: .still) else {
                continue
            }
            var gu = [Double](repeating: 0, count: p), gv = gu
            for k in 0..<p {
                var nudged = PoseCorrection.identity
                if k < 3 { nudged.rotation[k] = epsilon } else { nudged.translation[k - 3] = epsilon }
                guard let pk = projectToImage(ray, intrinsics: intrinsics,
                                              pose: nudged.corrected(.still)) else { continue }
                gu[k] = (pk.u - p0.u) / epsilon
                gv[k] = (pk.v - p0.v) / epsilon
            }
            var variance = 0.0
            for r in 0..<p {
                for c in 0..<p {
                    variance += (gu[r] * gu[c] + gv[r] * gv[c]) * inverse[r][c]
                }
            }
            worst = max(worst, noisePx * max(variance, 0).squareRoot())
        }
        return worst
    }

    /// Two Gauss-Newton steps on the correction's parameters: the three
    /// rotations, and the three translations unless `parameters` is 3.
    ///
    /// The Jacobian is taken by finite differences *through
    /// `projectToImage`*, so the fit and the application share one projection
    /// and no convention is restated here.  Seven projections per patch per
    /// step, at 10 Hz, on two dozen patches: nothing.  Returns the normal
    /// matrix with the estimate, for the covariance.
    private static func solveCorrection(_ observations: [PatchObservation], _ indices: [Int],
                                        parameters p: Int, intrinsics: PinholeIntrinsics,
                                        pose: CameraPose)
        -> (correction: PoseCorrection, normal: [[Double]])? {
        let epsilon = 1e-5   // radians or metres: ~0.01 px at phone focal lengths
        var estimate = PoseCorrection.identity
        var normal: [[Double]] = []

        for _ in 0..<2 {
            // The pose at the estimate, and at the estimate nudged along each
            // parameter.
            var poses = [estimate.corrected(pose)]
            for k in 0..<p {
                var nudged = estimate
                if k < 3 { nudged.rotation[k] += epsilon } else { nudged.translation[k - 3] += epsilon }
                poses.append(nudged.corrected(pose))
            }

            var a = [[Double]](repeating: [Double](repeating: 0, count: p), count: p)
            var b = [Double](repeating: 0, count: p)
            for i in indices {
                let world = observations[i].world
                guard let p0 = projectToImage(world, intrinsics: intrinsics, pose: poses[0])
                else { continue }
                var ju = [Double](repeating: 0, count: p)
                var jv = [Double](repeating: 0, count: p)
                var usable = true
                for k in 0..<p {
                    guard let pk = projectToImage(world, intrinsics: intrinsics,
                                                  pose: poses[k + 1]) else {
                        usable = false
                        break
                    }
                    ju[k] = (pk.u - p0.u) / epsilon
                    jv[k] = (pk.v - p0.v) / epsilon
                }
                guard usable else { continue }
                let ru = observations[i].measured.u - p0.u
                let rv = observations[i].measured.v - p0.v
                for r in 0..<p {
                    b[r] += ju[r] * ru + jv[r] * rv
                    for c in 0..<p { a[r][c] += ju[r] * ju[c] + jv[r] * jv[c] }
                }
            }
            normal = a
            // A whisper of damping, so a degenerate patch layout cannot make
            // the solve blow up.  Nothing else: what the data cannot decide is
            // decided by the choice of model, not by a prior.
            let trace = (0..<p).reduce(0.0) { $0 + a[$1][$1] }
            for k in 0..<p { a[k][k] += 1e-9 * trace }
            guard let step = solve(a, b) else { return nil }
            estimate.rotation += SIMD3(step[0], step[1], step[2])
            if p > 3 { estimate.translation += SIMD3(step[3], step[4], step[5]) }
        }
        guard (estimate.rotation * estimate.rotation).sum().isFinite,
              (estimate.translation * estimate.translation).sum().isFinite else { return nil }
        return (estimate, normal)
    }
}

/// Gaussian elimination with partial pivoting, for the 6x6 normal equations.
private func solve(_ a: [[Double]], _ b: [Double]) -> [Double]? {
    var m = a
    var y = b
    let n = y.count
    for col in 0..<n {
        guard let pivot = (col..<n).max(by: { abs(m[$0][col]) < abs(m[$1][col]) }),
              abs(m[pivot][col]) > 1e-300 else { return nil }
        if pivot != col { m.swapAt(pivot, col); y.swapAt(pivot, col) }
        for row in (col + 1)..<n {
            let f = m[row][col] / m[col][col]
            guard f != 0 else { continue }
            for k in col..<n { m[row][k] -= f * m[col][k] }
            y[row] -= f * y[col]
        }
    }
    var x = [Double](repeating: 0, count: n)
    for row in stride(from: n - 1, through: 0, by: -1) {
        var s = y[row]
        for k in (row + 1)..<n { s -= m[row][k] * x[k] }
        x[row] = s / m[row][row]
    }
    return x
}

private func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    return sorted[sorted.count / 2]
}

/// Even-odd point-in-polygon, for the water outline.
func contains(polygon: [PixelPoint], point: PixelPoint) -> Bool {
    guard polygon.count >= 3 else { return false }
    var inside = false
    var j = polygon.count - 1
    for i in polygon.indices {
        let a = polygon[i], b = polygon[j]
        if (a.v > point.v) != (b.v > point.v) {
            let t = (point.v - a.v) / (b.v - a.v)
            if point.u < a.u + t * (b.u - a.u) { inside.toggle() }
        }
        j = i
    }
    return inside
}