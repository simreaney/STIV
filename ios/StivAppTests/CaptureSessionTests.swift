//  CaptureSessionTests.swift
//  The capture session's bookkeeping, without a camera.
//
//  No frames arrive in the simulator, so these drive the frame-queue entry
//  points directly -- `recordRow` and the session-delegate callbacks -- in the
//  order ARKit would.

import ARKit
import StivKit
import XCTest
@testable import STIV

final class CaptureSessionTests: XCTestCase {
    private let intrinsics = PinholeIntrinsics(horizontalFovDegrees: 60, width: 640, height: 480)
    private let plane = WaterPlane(y: 0, fitResidualM: 0.02, sampleCount: 80,
                                   measuredOnWaterSurface: false)

    private func plan(durationSeconds: Double, framesPerSecond: Double = 30) throws -> RecordingPlan {
        var s = Survey()
        s.section.stationCount = 3
        s.section.start = PlanePoint(x: -3, z: -8)
        s.section.end = PlanePoint(x: 3, z: -8)
        s.flow.downstream = PlanePoint(x: 0, z: -20)
        s.sti.sampleCount = 16
        s.capture.durationSeconds = durationSeconds
        s.capture.framesPerSecond = framesPerSecond
        let projection = WaterPlaneProjection(intrinsics: intrinsics,
                                              pose: .looking(fromHeight: 3, downDegrees: 25),
                                              plane: plane)
        let layout = try buildSection(survey: s.validated(), rectifier: projection)
        return RecordingPlan(survey: s, plane: plane, layout: layout)
    }

    private func row(_ plan: RecordingPlan) -> [Float] {
        [Float](repeating: 100, count: plan.worldPoints.count)
    }

    private func finish(_ capture: ARCaptureSession) -> CompletedRecording? {
        var handed: CompletedRecording??
        capture.finishRecording { handed = .some($0) }
        let done = expectation(for: NSPredicate { _, _ in handed != nil }, evaluatedWith: nil)
        wait(for: [done], timeout: 5)
        return handed ?? nil
    }

    // MARK: - ending a recording

    /// The frame queue itself ends the recording at its duration.  Leaving it
    /// to the main thread to notice would let a busy main thread record past
    /// the duration -- and past the memory budget the duration was fitted to.
    func testTheFrameQueueStopsAtTheDurationWhateverTheMainThreadIsDoing() throws {
        let capture = ARCaptureSession()
        let plan = try plan(durationSeconds: 1)
        capture.startRecording(plan: plan, waterPolygon: nil)
        capture.syncOnFrameQueue {
            // Two seconds of frames, delivered without the main queue getting
            // a look in.
            for i in 0..<60 { capture.recordRow(row(plan), timestamp: Double(i) / 30) }
        }

        let complete = expectation(for: NSPredicate { _, _ in capture.recordingComplete },
                                   evaluatedWith: nil)
        wait(for: [complete], timeout: 5)
        let recording = try XCTUnwrap(finish(capture))
        XCTAssertEqual(recording.recording.rowCount, 31, "one second at 30 fps, inclusive")
        XCTAssertEqual(recording.recording.recordedSeconds, 1, accuracy: 1e-9)
    }

    /// The UI is refreshed at a fraction of the frame rate; the end of the
    /// recording is never one of the refreshes skipped.
    func testProgressIsPublishedOnlyWhenAsked() throws {
        let capture = ARCaptureSession()
        let plan = try plan(durationSeconds: 1)
        capture.startRecording(plan: plan, waterPolygon: nil)
        capture.syncOnFrameQueue {
            for i in 0..<10 {
                capture.recordRow(row(plan), timestamp: Double(i) / 30, refreshUI: false)
            }
        }
        // Let anything that was published arrive.
        let settled = expectation(description: "main queue drained")
        DispatchQueue.main.async { settled.fulfill() }
        wait(for: [settled], timeout: 5)
        XCTAssertEqual(capture.recordedRows, 0, "progress published on a frame that was not a refresh")

        capture.syncOnFrameQueue {
            for i in 10..<40 {
                capture.recordRow(row(plan), timestamp: Double(i) / 30, refreshUI: false)
            }
        }
        let complete = expectation(for: NSPredicate { _, _ in capture.recordingComplete },
                                   evaluatedWith: nil)
        wait(for: [complete], timeout: 5)
        XCTAssertEqual(capture.recordedRows, 31, "the final frame must always be published")
    }

    // MARK: - interruptions and failures

    /// An interruption breaks the world frame: the recording in progress is
    /// dropped, not handed over as if it were good, and the survey is told.
    func testAnInterruptionDropsTheRecordingAndSaysSo() throws {
        let capture = ARCaptureSession()
        var events: [CaptureEvent] = []
        capture.onEvent = { events.append($0) }
        let plan = try plan(durationSeconds: 5)
        capture.startRecording(plan: plan, waterPolygon: nil)
        capture.syncOnFrameQueue {
            for i in 0..<10 { capture.recordRow(row(plan), timestamp: Double(i) / 30) }
        }

        capture.sessionWasInterrupted(capture.session)
        let told = expectation(for: NSPredicate { _, _ in !events.isEmpty }, evaluatedWith: nil)
        wait(for: [told], timeout: 5)
        XCTAssertEqual(events, [.interrupted(wasRecording: true)])
        XCTAssertEqual(capture.phase, .idle)
        XCTAssertNil(finish(capture), "a recording across an interruption was handed over")

        capture.sessionInterruptionEnded(capture.session)
        let ended = expectation(for: NSPredicate { _, _ in events.count == 2 }, evaluatedWith: nil)
        wait(for: [ended], timeout: 5)
        XCTAssertEqual(events.last, .interruptionEnded)
    }

    /// The end of an interruption must never reach the survey before its
    /// start.  ARKit can deliver both while the frame queue is still busy with
    /// a frame; the start is handled on that queue, so the end must queue
    /// behind it, or the survey -- told it was over before it began -- would
    /// wait on the water-level screen for an end that had already come.
    func testTheEndOfAnInterruptionNeverOvertakesItsStart() {
        let capture = ARCaptureSession()
        var events: [CaptureEvent] = []
        capture.onEvent = { events.append($0) }

        // Hold the frame queue, as a long frame would.
        let busy = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            capture.syncOnFrameQueue {
                busy.signal()
                release.wait()
            }
        }
        busy.wait()

        capture.sessionWasInterrupted(capture.session)
        capture.sessionInterruptionEnded(capture.session)
        let drained = expectation(description: "main queue drained")
        DispatchQueue.main.async { drained.fulfill() }
        wait(for: [drained], timeout: 5)
        XCTAssertTrue(events.isEmpty, "the end arrived while the start was still queued: \(events)")

        release.signal()
        let both = expectation(for: NSPredicate { _, _ in events.count == 2 }, evaluatedWith: nil)
        wait(for: [both], timeout: 5)
        XCTAssertEqual(events, [.interrupted(wasRecording: false), .interruptionEnded])
    }

    /// Camera access refused: say so, and that the fix is in Settings.
    func testARefusedCameraIsReportedAsAPermissionProblem() throws {
        let capture = ARCaptureSession()
        var events: [CaptureEvent] = []
        capture.onEvent = { events.append($0) }
        capture.session(capture.session, didFailWithError: ARError(.cameraUnauthorized))
        let told = expectation(for: NSPredicate { _, _ in !events.isEmpty }, evaluatedWith: nil)
        wait(for: [told], timeout: 5)
        XCTAssertEqual(events, [.failed(wasRecording: false)])
        let failure = try XCTUnwrap(capture.failure)
        XCTAssertTrue(failure.needsCameraPermission)
        XCTAssertTrue(failure.message.contains("Settings"), failure.message)
    }

    func testOtherFailuresSayWhatHappenedWithoutBlamingPermissions() {
        for code in [ARError.Code.sensorFailed, .sensorUnavailable, .worldTrackingFailed,
                     .unsupportedConfiguration] {
            let failure = CaptureFailure(ARError(code))
            XCTAssertFalse(failure.needsCameraPermission, "\(code)")
            XCTAssertFalse(failure.message.isEmpty, "\(code)")
        }
        let other = CaptureFailure(NSError(domain: "test", code: 1,
                                           userInfo: [NSLocalizedDescriptionKey: "boom"]))
        XCTAssertTrue(other.message.contains("boom"))
    }

    /// The simulator has no world tracking.  Starting there must say so
    /// rather than run a configuration ARKit cannot honour.
    func testStartingWithoutWorldTrackingFailsPlainly() throws {
        try XCTSkipIf(ARWorldTrackingConfiguration.isSupported, "this device has world tracking")
        let capture = ARCaptureSession()
        capture.start()
        XCTAssertEqual(capture.failure, .unsupported)
    }

    // MARK: - the plan

    /// The corridors the pose check keeps its patches out of run the whole
    /// length of each search line.
    func testSearchLineCorridorsRunEndToEnd() throws {
        let plan = try plan(durationSeconds: 1)
        let projection = WaterPlaneProjection(intrinsics: intrinsics,
                                              pose: .looking(fromHeight: 3, downDegrees: 25),
                                              plane: plane)
        let segments = plan.searchLineSegments(projection: projection)
        XCTAssertEqual(segments.count, plan.layout.searchLines.count)
        for (segment, line) in zip(segments, plan.layout.searchLines) {
            let a = try XCTUnwrap(projection.worldToImage(try XCTUnwrap(line.samples.first)))
            let b = try XCTUnwrap(projection.worldToImage(try XCTUnwrap(line.samples.last)))
            XCTAssertEqual(segment, ImageSegment(a: a, b: b))
            XCTAssertGreaterThan(hypot(a.u - b.u, a.v - b.v), 20, "a corridor, not a dot")
        }
    }
}