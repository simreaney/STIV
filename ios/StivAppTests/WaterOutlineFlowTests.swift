//  WaterOutlineFlowTests.swift
//  The automatic water outline in the app: the burst of frames it is found
//  from, and what the cross-section screen does with it.
//
//  StivKit's `WaterOutlineTests` prove the detector on rendered rivers.  These
//  prove the plumbing: that frames are copied out of the camera's buffers at
//  the right size and spacing, that the downstream tap sets it off, that the
//  user's own taps win over it, and that a late answer is dropped.

import CoreVideo
import StivKit
import XCTest
@testable import STIV

// MARK: - a river to point the phone at

/// The scene `SurveyRecordingTests` taps: a section across world X at z = -8,
/// from x = -3 to x = 3, with the water flowing away from the phone along -Z.
private struct RiverAhead {
    var velocity = 0.8
    let water = SurfaceTexture(spacings: [0.5, 0.2, 0.08], seed: 3)
    let bank = SurfaceTexture(spacings: [1.5, 0.6, 0.25], seed: 4)

    func luma(_ u: Double, _ v: Double, k: PinholeIntrinsics, pose: CameraPose, t: Double) -> Float {
        let ray = pose.rotateToWorld((x: (u - k.cx) / k.fx, y: -(v - k.cy) / k.fy, z: -1))
        guard ray.y < -1e-9 else { return 128 }
        let s = -pose.position.y / ray.y
        guard s < 40 else { return 128 }
        let x = pose.position.x + s * ray.x, z = pose.position.z + s * ray.z
        if abs(x) < 3 { return Float(128 + 50 * water.value(x: x, z: z + velocity * t)) }
        return Float(110 + 45 * bank.value(x: x, z: z))
    }

    /// Five reduced frames 0.3 s apart from a hand-held phone 3 m up.
    func burst() -> [LumaFrame] {
        let k = PinholeIntrinsics(horizontalFovDegrees: 60, width: 240, height: 180)
        let sway = handheld(fromHeight: 3, downDegrees: 25)
        return (0..<5).map { i in
            let t = 0.3 * Double(i)
            let pose = sway(t)
            var pixels = [Float](repeating: 0, count: k.width * k.height)
            for v in 0..<k.height {
                for u in 0..<k.width {
                    var sum: Float = 0
                    for (du, dv) in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] {
                        sum += luma(Double(u) + du, Double(v) + dv, k: k, pose: pose, t: t)
                    }
                    pixels[v * k.width + u] = sum / 4
                }
            }
            return LumaFrame(width: k.width, height: k.height, pixels: pixels,
                             projection: WaterPlaneProjection(intrinsics: k, pose: pose,
                                                              plane: WaterPlane(y: 0)),
                             timestamp: t)
        }
    }
}

// MARK: - the burst

final class WaterProbeTests: XCTestCase {
    private let full = PinholeIntrinsics(horizontalFovDegrees: 60, width: 960, height: 720)

    /// A luma buffer whose value is `u + v`, so a block average is known.
    private func gradient() -> CVPixelBuffer {
        let buffer = makeYCbCrBuffer(width: full.width, height: full.height)
        writeLuma(buffer) { u, v in UInt8((u / 8 + v / 8) % 256) }
        return buffer
    }

    func testFramesAreReducedByBlockAveragesTheIntrinsicsDescribe() throws {
        let buffer = makeYCbCrBuffer(width: 64, height: 48)
        writeLuma(buffer) { u, v in UInt8(u + v) }
        let reduced = try XCTUnwrap(withLumaPlane(of: buffer) { reduce($0, by: 4, width: 16, height: 12) })
        // Block (i, j) holds u = 4i..4i+3 and v = 4j..4j+3: a mean of 4i + 4j + 3.
        for j in 0..<12 {
            for i in 0..<16 {
                XCTAssertEqual(reduced[j * 16 + i], Float(4 * i + 4 * j + 3), accuracy: 1e-4)
            }
        }
    }

    func testABurstIsFiveFramesAThirdOfASecondApart() throws {
        let probe = WaterProbe(plane: WaterPlane(y: 0))
        let buffer = gradient()
        let pose = CameraPose.looking(fromHeight: 2, downDegrees: 30)
        var finishedAt: Double?
        for frame in 0..<120 {                        // two seconds at 60 fps
            let t = 10 + Double(frame) / 60
            let done = try XCTUnwrap(withLumaPlane(of: buffer) {
                probe.offer(luma: $0, intrinsics: full, pose: pose, timestamp: t,
                            trackingNormal: frame % 7 != 3)
            })
            if done { finishedAt = t; break }
        }
        XCTAssertEqual(probe.frames.count, WaterProbe.frameCount)
        XCTAssertNotNil(finishedAt)
        let times = probe.frames.map(\.timestamp)
        for (a, b) in zip(times, times.dropFirst()) {
            XCTAssertGreaterThanOrEqual(b - a, WaterProbe.interval - 1e-9)
            XCTAssertLessThan(b - a, WaterProbe.interval + 0.05)
        }
        let first = try XCTUnwrap(probe.frames.first)
        XCTAssertEqual(first.width, 320)               // 960 / 3
        XCTAssertEqual(first.projection.intrinsics, full.reduced(by: 3))
        XCTAssertEqual(first.projection.plane, WaterPlane(y: 0))
    }

    func testABurstGivesUpWhenTrackingNeverRecovers() throws {
        let probe = WaterProbe(plane: WaterPlane(y: 0))
        let buffer = gradient()
        var done = false
        var t = 0.0
        while !done, t < 10 {
            done = try XCTUnwrap(withLumaPlane(of: buffer) {
                probe.offer(luma: $0, intrinsics: full, pose: .still, timestamp: t, trackingNormal: false)
            })
            t += 1.0 / 60
        }
        XCTAssertTrue(done)
        XCTAssertLessThan(t, WaterProbe.deadline + 0.1)
        XCTAssertTrue(probe.frames.isEmpty, "no frame from a pose ARKit did not vouch for")
    }
}

// MARK: - the cross-section screen

final class WaterOutlineFlowTests: XCTestCase {
    private let lowRes = PinholeIntrinsics(horizontalFovDegrees: 60, width: 640, height: 480)
    private var view: CameraSnapshot {
        CameraSnapshot(intrinsics: lowRes, pose: .looking(fromHeight: 3, downDegrees: 25))
    }

    /// The water level measured and the section tapped; `deliver` decides
    /// what the camera hands the detector, and when.
    private func model(deliver: @escaping (@escaping ([LumaFrame]) -> Void) -> Void) -> SurveyModel {
        let model = SurveyModel()
        model.collectWaterFrames = { _, done in
            deliver(done)
            return true
        }
        model.adopt(WaterPlane(y: 0, fitResidualM: 0.02, sampleCount: 80, measuredOnWaterSurface: false))
        model.handleTap(world: PlanePoint(x: -3, z: -8), seenFrom: view)
        model.handleTap(world: PlanePoint(x: 3, z: -8), seenFrom: view)
        return model
    }

    private func waitUntil(_ what: String, _ condition: @escaping () -> Bool) {
        let met = expectation(for: NSPredicate { _, _ in condition() }, evaluatedWith: nil)
        met.expectationDescription = what
        wait(for: [met], timeout: 30)
    }

    override func tearDown() {
        UIApplication.shared.isIdleTimerDisabled = false
        super.tearDown()
    }

    func testTheDownstreamTapOutlinesTheWater() {
        let frames = RiverAhead().burst()
        let model = model { done in DispatchQueue.main.async { done(frames) } }
        model.handleTap(world: PlanePoint(x: 0, z: -20), seenFrom: view)
        XCTAssertEqual(model.pickTarget, .waterOutline)
        XCTAssertEqual(model.waterDetection, .looking)

        waitUntil("outlined") { model.waterSource == .automatic }
        guard case .found(let weak, _) = model.waterDetection else {
            return XCTFail("\(model.waterDetection)")
        }
        XCTAssertFalse(weak)
        XCTAssertFalse(model.flowLooksReversed)
        let outline = model.waterPolygon
        XCTAssertGreaterThanOrEqual(outline.count, 3)
        for z in [-6.0, -8.0, -11.0] {
            XCTAssertTrue(contains(polygon: outline, point: PlanePoint(x: 0, z: z)), "water at z = \(z)")
        }
        for p in [PlanePoint(x: -5.5, z: -8), PlanePoint(x: 5.5, z: -8)] {
            XCTAssertFalse(contains(polygon: outline, point: p), "bank at \(p)")
        }
    }

    func testADownstreamPointOnTheWrongSideIsOfferedAFix() throws {
        let frames = RiverAhead().burst()
        let model = model { done in DispatchQueue.main.async { done(frames) } }
        model.handleTap(world: PlanePoint(x: 0, z: -3), seenFrom: view)   // upstream, by mistake
        waitUntil("outlined") { model.waterSource == .automatic }
        XCTAssertTrue(model.flowLooksReversed)

        model.moveDownstreamAcross()
        let moved = try XCTUnwrap(model.downstream)
        XCTAssertEqual(moved.x, 0, accuracy: 1e-9)
        XCTAssertEqual(moved.z, -13, accuracy: 1e-9, "mirrored across z = -8")
        XCTAssertFalse(model.flowLooksReversed)
        XCTAssertEqual(model.waterSource, .automatic, "the outline itself still stands")
    }

    func testTheUsersOwnTapsWinOverALateOutline() {
        var handOver: (([LumaFrame]) -> Void)?
        let model = model { done in handOver = done }
        model.handleTap(world: PlanePoint(x: 0, z: -20), seenFrom: view)
        XCTAssertEqual(model.waterDetection, .looking)

        model.handleTap(world: PlanePoint(x: -2, z: -7), seenFrom: view)
        XCTAssertEqual(model.waterSource, .manual)
        XCTAssertEqual(model.waterDetection, .idle)

        handOver?(RiverAhead().burst())
        let settled = expectation(description: "the late outline had its chance")
        DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
            DispatchQueue.main.async { settled.fulfill() }
        }
        wait(for: [settled], timeout: 10)
        XCTAssertEqual(model.waterPolygon, [PlanePoint(x: -2, z: -7)], "the tapped outline was replaced")
        XCTAssertEqual(model.waterSource, .manual)
    }

    func testUndoTakesTheWholeAutomaticOutline() {
        let frames = RiverAhead().burst()
        let model = model { done in DispatchQueue.main.async { done(frames) } }
        model.handleTap(world: PlanePoint(x: 0, z: -20), seenFrom: view)
        waitUntil("outlined") { model.waterSource == .automatic }

        model.undoLastTap()
        XCTAssertTrue(model.waterPolygon.isEmpty)
        XCTAssertNil(model.waterSource)
        XCTAssertEqual(model.pickTarget, .waterOutline)
        model.undoLastTap()
        XCTAssertNil(model.downstream)
        XCTAssertEqual(model.pickTarget, .downstream)
    }

    func testStillWaterIsNotOutlined() {
        var still = RiverAhead()
        still.velocity = 0
        let frames = still.burst()
        let model = model { done in DispatchQueue.main.async { done(frames) } }
        model.handleTap(world: PlanePoint(x: 0, z: -20), seenFrom: view)
        waitUntil("given up") {
            if case .notFound = model.waterDetection { return true }
            return false
        }
        XCTAssertTrue(model.waterPolygon.isEmpty)
        XCTAssertNil(model.waterSource)
    }

    /// With no camera running, nothing is waited for.
    func testWithNoCameraNothingIsAttempted() {
        let model = SurveyModel()
        model.adopt(WaterPlane(y: 0, fitResidualM: 0.02, sampleCount: 80, measuredOnWaterSurface: false))
        for p in [PlanePoint(x: -3, z: -8), PlanePoint(x: 3, z: -8), PlanePoint(x: 0, z: -20)] {
            model.handleTap(world: p, seenFrom: view)
        }
        XCTAssertEqual(model.pickTarget, .waterOutline)
        XCTAssertEqual(model.waterDetection, .idle)
    }
}
