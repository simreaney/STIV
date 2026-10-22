//  SyntheticRiverTests.swift
//  End-to-end without a river: rendered camera frames of a surface moving at a
//  known velocity, through the app's own per-frame sampling, through the whole
//  analysis, and the velocity checked at the end.
//
//  This is the test StivKit cannot have.  Its pipeline tests synthesise the
//  STIs directly, so they never exercise the claim this app's design rests on:
//  that sampling fixed *world* points through each frame's own pose removes
//  camera motion completely.  Here the camera sways like a hand-held phone and
//  the river still has to come back at the right speed -- and, as a control,
//  the same sway without the pose has to ruin it, or the test proves nothing.
//
//  What it does not model: rolling shutter, motion blur, exposure changes,
//  lens distortion, or a pose that is itself wrong.  `PoseResidualTests` covers
//  the last; the rest are what the field checks in TESTING.md are for.

import CoreVideo
import StivKit
import XCTest
@testable import STIV

final class SyntheticRiverTests: XCTestCase {
    /// A quarter of the sensor's pixels per side would do; half keeps the
    /// sampling comparable to the phone's at this range while rendering fast.
    private let intrinsics = PinholeIntrinsics(horizontalFovDegrees: 60, width: 640, height: 480)
    private let plane = WaterPlane(y: 0, fitResidualM: 0.02, sampleCount: 60,
                                   measuredOnWaterSurface: false)
    private let framesPerSecond = 30.0
    private let seconds = 8.0
    /// Features of 0.2-0.5 m: two pixels or more even at the far end of the
    /// search lines, so the rendered pattern is not aliased.
    private let texture = SurfaceTexture(spacings: [0.45, 0.18], seed: 7)

    private let height = 3.0
    private let pitch = 25.0
    private var steady: (Double) -> CameraPose {
        { [height, pitch] _ in .looking(fromHeight: height, downDegrees: pitch) }
    }
    private var swaying: (Double) -> CameraPose { handheld(fromHeight: height, downDegrees: pitch) }

    /// A 6 m section 8 m out from a bank 3 m above the water, flow running
    /// straight away from the camera -- the geometry `PipelineTests` uses.
    private func survey() throws -> Survey {
        var s = Survey()
        s.section.stationCount = 7
        s.section.start = PlanePoint(x: -3, z: -8)
        s.section.end = PlanePoint(x: 3, z: -8)
        s.flow.direction = .auto
        s.flow.downstream = PlanePoint(x: 0, z: -20)
        s.sti.searchLineLengthM = 4.0
        s.sti.sampleCount = 96
        s.capture.framesPerSecond = framesPerSecond
        s.capture.frameStep = 1
        s.capture.durationSeconds = seconds
        return try s.validated()
    }

    /// Record and analyse a river flowing at `velocity` towards `downstream`.
    ///
    /// - Parameter camera: where the camera really is at time `t`; the frames
    ///   are rendered from it.
    /// - Parameter sampledWith: the pose the sampler is given.  Defaults to the
    ///   truth, which is what ARKit supplies when tracking is healthy.
    private func record(
        velocity: Double,
        camera: (Double) -> CameraPose,
        sampledWith reported: ((Double) -> CameraPose)? = nil
    ) throws -> StivResult {
        let s = try survey()
        let reference = WaterPlaneProjection(intrinsics: intrinsics, pose: camera(0), plane: plane)
        let layout = try buildSection(survey: s, rectifier: reference)
        XCTAssertTrue(layout.stationsOutsideFrame.isEmpty,
                      "stations \(layout.stationsOutsideFrame) are out of frame; fix the geometry")
        let plan = RecordingPlan(survey: s, plane: plane, layout: layout)
        let recorder = StiRecorder(sampleCounts: plan.sampleCounts, sxPerStation: plan.sxPerStation)
        let buffer = makeYCbCrBuffer(width: intrinsics.width, height: intrinsics.height)
        let flow = layout.flowDirection

        for i in 0..<Int(seconds * framesPerSecond) {
            let t = Double(i) / framesPerSecond
            let pose = camera(t)
            // Render only around the samples; see `renderPlane` on why the grey
            // elsewhere makes this stricter rather than looser.
            let projection = WaterPlaneProjection(intrinsics: intrinsics, pose: pose, plane: plane)
            let footprint = plan.worldPoints.compactMap {
                projection.worldToImage(PlanePoint(x: $0.x, z: $0.z))
            }
            renderPlane(into: buffer, intrinsics: intrinsics, pose: pose, planeY: plane.y,
                        texture: texture,
                        drift: (velocity * t * flow.x, velocity * t * flow.z),
                        region: PixelBox(around: footprint, margin: 3,
                                         width: intrinsics.width, height: intrinsics.height))

            let sampledPose = reported?(t) ?? pose
            let row = withLumaPlane(of: buffer) { luma in
                plan.sampleRow(luma: luma, intrinsics: intrinsics, pose: sampledPose,
                               correction: .identity)
            }
            recorder.append(row: try XCTUnwrap(row), timestamp: t)
        }

        return analyse(survey: s, layout: layout,
                       stis: recorder.finish(frameStep: s.capture.frameStep))
    }

    /// RMS error over every station, counting a station with no velocity at
    /// all as wrong by the whole of the truth.
    private func rmse(_ result: StivResult, truth: Double) -> Double {
        let errors = result.stations.map { $0.velocityMs.isFinite ? $0.velocityMs - truth : truth }
        return (errors.map { $0 * $0 }.reduce(0, +) / Double(errors.count)).squareRoot()
    }

    private func describe(_ result: StivResult) -> String {
        result.stations.map { s in
            "  station \(s.index): \(decimals(s.velocityMs, 3)) m/s \(s.flags.map(\.rawValue))"
        }.joined(separator: "\n") + "\n" + result.summary()
    }

    // MARK: - tests

    func testASteadyCameraRecoversTheFlow() throws {
        let truth = 0.8
        let result = try record(velocity: truth, camera: steady)
        XCTAssertLessThan(rmse(result, truth: truth), 0.03 * truth, describe(result))
        XCTAssertTrue(result.stations.allSatisfy(\.ok), describe(result))
    }

    func testHandShakeIsRemovedByThePose() throws {
        let truth = 0.8
        let result = try record(velocity: truth, camera: swaying)
        XCTAssertLessThan(rmse(result, truth: truth), 0.03 * truth, describe(result))
        XCTAssertTrue(result.stations.allSatisfy(\.ok), describe(result))
    }

    /// The control for the test above.  The same sway, sampled through the
    /// pose of the first frame -- a fixed pixel grid, as if there were no
    /// tracking -- must fail badly, or the sway is too gentle to have tested
    /// anything.
    func testWithoutThePoseTheSameShakeRuinsTheProfile() throws {
        let truth = 0.8
        let first = swaying(0)
        let result = try record(velocity: truth, camera: swaying, sampledWith: { _ in first })
        XCTAssertGreaterThan(rmse(result, truth: truth), 0.25 * truth, describe(result))
    }

    /// The simulated version of the field null test: still water, swaying
    /// camera.  Whatever the stations say, none may confidently report flow.
    func testStillWaterUnderHandShakeInventsNoFlow() throws {
        let result = try record(velocity: 0, camera: swaying)
        for station in result.stations where station.ok {
            XCTAssertLessThan(abs(station.velocityMs), 0.1,
                              "station \(station.index) invents flow\n\(describe(result))")
        }
    }

    /// The other direction: flow towards the camera, against the assumed
    /// downstream.  The analysis must notice and say so, not report a
    /// confident wrong speed.
    func testFlowTowardsTheCameraIsDetectedAsUpstream() throws {
        let result = try record(velocity: -0.8, camera: swaying)
        XCTAssertTrue(result.warnings.contains { $0.contains("opposite to the assumed") },
                      "no direction warning in:\n\(result.warnings)")
        XCTAssertLessThan(rmse(result, truth: 0.8), 0.03 * 0.8, describe(result))
    }
}
