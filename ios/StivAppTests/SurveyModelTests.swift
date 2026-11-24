//  SurveyModelTests.swift
//  The survey flow's bookkeeping: which tap means what, how undo walks back
//  through them, and what a recording does from Start to the results.
//
//  No camera is running, so `capture.camera` stays nil.  `SurveyModelTests`
//  never builds a layout -- it is about the state machine, not the geometry.
//  `SurveyRecordingTests` supplies the camera views a phone would, and the
//  frames: it measures, taps, records and analyses without a camera.

import ARKit
import StivKit
import UIKit
import XCTest
@testable import STIV

final class SurveyModelTests: XCTestCase {
    private let near = PlanePoint(x: -2, z: -6)
    private let far = PlanePoint(x: 4, z: -9)
    private let below = PlanePoint(x: 1, z: -15)

    func testTapsAreTakenInTheOrderTheyBecomeMeaningful() {
        let model = SurveyModel()
        XCTAssertEqual(model.pickTarget, .sectionStart)

        model.handleTap(world: near)
        XCTAssertEqual(model.sectionStart, near)
        XCTAssertEqual(model.pickTarget, .sectionEnd)

        model.handleTap(world: far)
        XCTAssertEqual(model.sectionEnd, far)
        XCTAssertEqual(model.pickTarget, .downstream)
        XCTAssertEqual(model.sectionWidthM ?? .nan, hypot(6, 3), accuracy: 1e-12)

        model.handleTap(world: below)
        XCTAssertEqual(model.downstream, below)
        XCTAssertEqual(model.pickTarget, .waterOutline)

        // From here on every tap is another vertex of the optional outline.
        for k in 0..<4 { model.handleTap(world: PlanePoint(x: Double(k), z: -7)) }
        XCTAssertEqual(model.waterPolygon.count, 4)
        XCTAssertEqual(model.pickTarget, .waterOutline)
    }

    func testUndoRetracesTheTapsOneAtATime() {
        let model = SurveyModel()
        for p in [near, far, below, PlanePoint(x: 0, z: -7), PlanePoint(x: 1, z: -7)] {
            model.handleTap(world: p)
        }

        model.undoLastTap()
        XCTAssertEqual(model.waterPolygon.count, 1, "outline vertices go first, one by one")
        model.undoLastTap()
        XCTAssertTrue(model.waterPolygon.isEmpty)
        XCTAssertEqual(model.pickTarget, .waterOutline)

        model.undoLastTap()
        XCTAssertNil(model.downstream)
        XCTAssertEqual(model.pickTarget, .downstream)
        model.undoLastTap()
        XCTAssertNil(model.sectionEnd)
        XCTAssertEqual(model.pickTarget, .sectionEnd)
        model.undoLastTap()
        XCTAssertNil(model.sectionStart)
        XCTAssertEqual(model.pickTarget, .sectionStart)

        model.undoLastTap()   // nothing left: must be harmless
        XCTAssertEqual(model.pickTarget, .sectionStart)
    }

    func testAMissedTapSaysWhy() {
        let model = SurveyModel()
        model.tapMissed()
        XCTAssertEqual(model.tapHint, "Measure the water level first.")
        model.handleTap(world: near)
        XCTAssertNil(model.tapHint, "a tap that lands clears the hint")
    }

    func testWithNoLayoutTheFrameStepFallsBackToTheDefault() {
        let model = SurveyModel()
        XCTAssertNil(model.layout)
        XCTAssertEqual(model.suggestedFrameStep, 2)
        XCTAssertFalse(model.canRetune, "nothing recorded, nothing to re-analyse")
    }

    /// The Stop button and the duration timer both end a recording; whichever
    /// comes second must not overwrite the first's outcome.  With nothing
    /// recorded the outcome is an error, reported once.
    func testStoppingTwiceIsHarmless() {
        let model = SurveyModel()
        model.stopRecordingAndAnalyse()
        model.stopRecordingAndAnalyse()
        let reported = expectation(for: NSPredicate { _, _ in model.analysisError != nil },
                                   evaluatedWith: nil)
        wait(for: [reported], timeout: 5)
        XCTAssertEqual(model.analysisError, "Nothing was recorded.")
        XCTAssertNotEqual(model.step, .results)
    }
}

final class SurveyRecordingTests: XCTestCase {
    private let lowRes = PinholeIntrinsics(horizontalFovDegrees: 60, width: 640, height: 480)

    /// The phone 3 m above the water, aimed 25 degrees down across it.
    private func aimed(_ intrinsics: PinholeIntrinsics? = nil, pitch: Double = 25) -> CameraSnapshot {
        CameraSnapshot(intrinsics: intrinsics ?? lowRes,
                       pose: .looking(fromHeight: 3, downDegrees: pitch))
    }

    /// What Documents held before the test.  Every result is saved there;
    /// only what appeared during the test is removed afterwards, since on a
    /// phone the folder also holds real measurements.
    private var documentsBefore: Set<String> = []

    private func documents() -> [String] {
        (try? FileManager.default.contentsOfDirectory(atPath: URL.documentsDirectory.path)) ?? []
    }

    override func setUp() {
        super.setUp()
        documentsBefore = Set(documents())
    }

    override func tearDown() {
        for name in documents() where !documentsBefore.contains(name) {
            try? FileManager.default.removeItem(at: URL.documentsDirectory.appendingPathComponent(name))
        }
        // The models here move through the camera steps, which switch the
        // idle timer off for the whole app.
        UIApplication.shared.isIdleTimerDisabled = false
        super.tearDown()
    }

    /// A survey with the water level measured and a 6 m section tapped 8 m out,
    /// all from `view`.
    private func surveyed(view: CameraSnapshot? = nil, stations: Int = 5,
                          seconds: Double = 5, fps: Double = 30) -> SurveyModel {
        let model = SurveyModel()
        model.survey.section.stationCount = stations
        model.survey.sti.searchLineLengthM = 4
        model.survey.capture.framesPerSecond = fps
        model.survey.capture.durationSeconds = seconds
        model.adopt(WaterPlane(y: 0, fitResidualM: 0.02, sampleCount: 80,
                               measuredOnWaterSurface: false))
        for p in [PlanePoint(x: -3, z: -8), PlanePoint(x: 3, z: -8), PlanePoint(x: 0, z: -20)] {
            model.handleTap(world: p, seenFrom: view ?? aimed())
        }
        return model
    }

    /// Deliver `count` frames of texture to the recording in progress, at
    /// the model's frame rate.
    private func record(_ model: SurveyModel, rows count: Int) throws {
        let fps = model.survey.capture.framesPerSecond
        try record(model, at: (0..<count).map { Double($0) / fps })
    }

    /// Deliver one frame of texture at each of `timestamps`.
    private func record(_ model: SurveyModel, at timestamps: [Double]) throws {
        let width = try XCTUnwrap(model.layout).searchLines.reduce(0) { $0 + $1.sampleCount }
        var noise = DeterministicNoise(seed: 9)
        model.capture.syncOnFrameQueue {
            for t in timestamps {
                let row = (0..<width).map { _ in Float(128 + 60 * noise.next()) }
                model.capture.recordRow(row, timestamp: t)
            }
        }
    }

    private func waitUntil(_ what: String, timeout: TimeInterval = 20,
                           _ condition: @escaping () -> Bool) {
        let met = expectation(for: NSPredicate { _, _ in condition() }, evaluatedWith: nil)
        met.expectationDescription = what
        wait(for: [met], timeout: timeout)
    }

    private func sampleCounts(_ layout: SectionLayout?) -> [Int] {
        layout?.searchLines.map(\.sampleCount) ?? []
    }

    // MARK: - the view the layout is built from

    /// The settings screen shows no camera feed, so the pose the phone has
    /// there means nothing; Start rebuilds the layout from the view the user is
    /// aiming with, and refuses a view that does not see the section.
    func testTheLayoutIsBuiltFromTheViewTheRecordingStartsFrom() throws {
        let model = surveyed()
        XCTAssertEqual(model.sectionView, aimed(), "the tapping view is kept")
        let fromTaps = sampleCounts(model.layout)

        // On the settings screen, with no feed to aim by, the phone is
        // pointing straight down.  Its settings changes rebuild the layout;
        // they must rebuild it from the tapping view, not from this.
        let flat = CameraSnapshot(intrinsics: lowRes,
                                  pose: .looking(fromHeight: 0.4, downDegrees: 89.9, atZ: -8))
        model.currentView = { flat }
        model.rebuildLayout()
        XCTAssertEqual(sampleCounts(model.layout), fromTaps,
                       "the settings screen rebuilt from the phone's current pose")
        let fromFlat = try buildSection(survey: model.survey.validated(),
                                        rectifier: flat.projection(onto: try XCTUnwrap(model.plane)))
        XCTAssertNotEqual(sampleCounts(fromFlat), fromTaps,
                          "the flat view is too like the tapping view to tell which was used")
        model.go(to: .record)

        // Start, by default, uses the view the phone has now -- here still
        // flat -- and refuses it.  Facing away is refused too.
        model.startRecording()
        XCTAssertTrue(model.recordingError?.contains("straight down") ?? false,
                      model.recordingError ?? "no error")
        model.currentView = {
            CameraSnapshot(intrinsics: self.lowRes,
                           pose: .looking(fromHeight: 3, downDegrees: 25, yawDegrees: 180))
        }
        model.startRecording()
        XCTAssertTrue(model.recordingError?.contains("behind") ?? false,
                      model.recordingError ?? "no error")
        XCTAssertNotEqual(model.capture.phase, .recording)

        // Aimed again, a little steeper than when tapping: accepted, and the
        // layout is this view's.
        let now = aimed(pitch: 30)
        model.startRecording(from: now)
        XCTAssertNil(model.recordingError)
        XCTAssertEqual(model.capture.phase, .recording)
        var expected = model.survey
        expected.sti.sampleCount = nil
        let fromNow = try buildSection(survey: expected.validated(),
                                       rectifier: now.projection(onto: try XCTUnwrap(model.plane)))
        XCTAssertEqual(sampleCounts(model.layout), sampleCounts(fromNow))
        XCTAssertNotEqual(sampleCounts(model.layout), fromTaps,
                          "the two views are too alike to tell which was used")
    }

    /// Two minutes of a full-resolution section would hold well over a
    /// gigabyte.  Start shortens it to fit, and says so.
    func testAnOverlongRecordingIsShortenedToFitMemoryAndSaysSo() throws {
        let sensor = PinholeIntrinsics(horizontalFovDegrees: 60, width: 1920, height: 1440)
        let model = surveyed(view: aimed(sensor), stations: 60, seconds: 120, fps: 60)
        model.go(to: .record)
        model.startRecording(from: aimed(sensor))
        XCTAssertNil(model.recordingError, model.recordingError ?? "")
        XCTAssertEqual(model.capture.phase, .recording)
        let seconds = try XCTUnwrap(model.recordingSeconds)
        XCTAssertLessThan(seconds, 120)
        XCTAssertGreaterThanOrEqual(seconds, RecordingBudget.minimumSeconds)
        let perRow = sampleCounts(model.layout).reduce(0, +)
        XCTAssertLessThanOrEqual(Double(perRow * 4) * 60 * seconds, Double(RecordingBudget.bytes))
        XCTAssertNotNil(model.recordingNotice, "the user must be told the recording was cut")
        XCTAssertEqual(model.survey.capture.durationSeconds, 120,
                       "the user's own setting is left alone")
    }

    // MARK: - stopping

    /// Stop, then Start before the recording is handed over: the Start is
    /// refused, and the run that was stopped is the one analysed -- once,
    /// however many times Stop was pressed.
    func testAStartWhileTheLastRunIsBeingHandedOverIsRefused() throws {
        let model = surveyed()
        model.go(to: .record)
        model.startRecording(from: aimed())
        try record(model, rows: 90)

        model.stopRecordingAndAnalyse()
        model.stopRecordingAndAnalyse()
        XCTAssertTrue(model.isFinishing)
        model.startRecording(from: aimed())
        XCTAssertEqual(model.capture.phase, .recording, "still the old run, being handed over")

        waitUntil("results") { model.result != nil || model.analysisError != nil }
        XCTAssertNil(model.analysisError)
        XCTAssertFalse(model.isFinishing)
        XCTAssertEqual(model.step, .results)
        XCTAssertEqual(model.capture.phase, .idle)
        XCTAssertTrue(model.canRetune)
        XCTAssertEqual(model.result?.durationSeconds ?? 0, 3, accuracy: 0.1,
                       "the analysed recording is the one that was stopped")
    }

    /// A recording handed back after its run was abandoned -- here, by an
    /// interruption that arrived while it was being handed over -- belongs to
    /// no survey any more, and must not reach the results screen.
    func testARecordingHandedOverAfterItsRunWasAbandonedIsDropped() throws {
        let model = surveyed()
        model.go(to: .record)
        model.startRecording(from: aimed())
        try record(model, rows: 60)

        model.stopRecordingAndAnalyse()
        model.capture.onEvent?(.interrupted(wasRecording: true))
        waitUntil("hand-over finished") { model.capture.phase == .idle }
        let settled = expectation(description: "main queue drained")
        DispatchQueue.main.async { settled.fulfill() }
        wait(for: [settled], timeout: 5)

        XCTAssertEqual(model.step, .waterPlane)
        XCTAssertFalse(model.canRetune)
        XCTAssertNil(model.result)
        XCTAssertFalse(model.isFinishing)
    }

    // MARK: - analysis

    /// Every retune is a full pass over the recording.  Taps made while one is
    /// running must not start more beside it, and the answer shown at the end
    /// must be the last one asked for.
    func testRetuningRunsOneAnalysisAtATimeAndKeepsTheNewest() throws {
        // Twelve seconds: enough rows for the structure-tensor window at every
        // frame step tried below.
        let model = surveyed(seconds: 20)
        model.go(to: .record)
        model.startRecording(from: aimed())
        try record(model, rows: 360)
        model.stopRecordingAndAnalyse()
        waitUntil("first result") { model.result != nil }

        for step in [3, 4, 1, 6, 5] { model.setFrameStep(step) }
        XCTAssertTrue(model.isAnalysing)
        waitUntil("retuned") { !model.isAnalysing }
        XCTAssertEqual(model.result?.survey.capture.frameStep, 5, "not the newest request")
        XCTAssertEqual(model.peakConcurrentAnalyses, 1, "analyses overlapped")
        XCTAssertNil(model.analysisError)
    }

    /// StivKit refuses an STI shorter than one analysis window, so the
    /// retune control stops at the largest step the recording can take.
    func testTheFrameStepStopsWhereTheSTIsWouldBeTooShort() throws {
        let model = surveyed()
        model.go(to: .record)
        model.startRecording(from: aimed())
        try record(model, rows: 150)
        model.stopRecordingAndAnalyse()
        waitUntil("first result") { model.result != nil }

        let window = minStiSize(windowPx: model.survey.sti.windowPx,
                                windowType: model.survey.sti.windowType,
                                gradientOrder: model.survey.sti.gradientOrder)
        XCTAssertEqual(model.minimumStiRows, window)
        XCTAssertEqual(model.maxFrameStep, 149 / window)
        XCTAssertLessThan(model.maxFrameStep, 30, "the cap must bite for this test to mean anything")

        model.setFrameStep(30)
        XCTAssertEqual(model.survey.capture.frameStep, model.maxFrameStep)
        waitUntil("retuned") { !model.isAnalysing }
        XCTAssertNil(model.analysisError, model.analysisError ?? "")
        XCTAssertEqual(model.result?.survey.capture.frameStep, model.maxFrameStep)
        XCTAssertGreaterThanOrEqual(model.result?.stiPreviews.first?.rows ?? 0, window)
    }

    /// StivKit grids an STI at the *median* frame period.  A phone dropping
    /// frames has a median above its average, so a cap worked out from the
    /// row count promises steps the analysis then refuses: here 600 intervals,
    /// three in five of 33 ms and the rest 17 ms.
    func testTheFrameStepCapFollowsTheMedianFramePeriod() throws {
        let model = surveyed(seconds: 60)
        model.go(to: .record)
        model.startRecording(from: aimed())
        var t = 0.0
        var times = [t]
        for i in 0..<600 {
            t += i % 5 < 3 ? 0.033 : 0.017
            times.append(t)
        }
        try record(model, at: times)
        model.stopRecordingAndAnalyse()
        waitUntil("first result") { model.result != nil || model.analysisError != nil }

        let window = model.minimumStiRows
        XCTAssertLessThan(model.maxFrameStep, (times.count - 1) / window,
                          "this recording does not reproduce the row-count cap's mistake")
        model.setFrameStep(30)
        XCTAssertEqual(model.survey.capture.frameStep, model.maxFrameStep)
        waitUntil("retuned") { !model.isAnalysing }
        XCTAssertNil(model.analysisError, model.analysisError ?? "")
        XCTAssertEqual(model.result?.survey.capture.frameStep, model.maxFrameStep)
        XCTAssertGreaterThanOrEqual(model.result?.stiPreviews.first?.rows ?? 0, window)
    }

    /// A recording too short for even one window -- stopped at once -- is
    /// refused with the reason, and stays in hand to be retuned.
    func testARecordingTooShortForTheWindowSaysSo() throws {
        let model = surveyed()
        model.go(to: .record)
        model.startRecording(from: aimed())
        try record(model, rows: 20)
        model.stopRecordingAndAnalyse()
        waitUntil("refused") { model.analysisError != nil }
        XCTAssertNil(model.result)
        XCTAssertTrue(model.analysisError?.contains("refused") ?? false, model.analysisError ?? "")
        XCTAssertTrue(model.canRetune)
        XCTAssertEqual(model.maxFrameStep, 1)
    }

    // MARK: - saving

    /// A result is saved the moment it exists, not when Share is tapped: an
    /// app in the background can be ended by iOS at any time.  It goes into
    /// its recording's own folder in Documents, which a retune overwrites, and
    /// Share shares those same files.
    func testEveryResultIsSavedInItsRecordingsFolder() throws {
        let model = surveyed()
        model.go(to: .record)
        model.startRecording(from: aimed())
        try record(model, rows: 150)
        model.stopRecordingAndAnalyse()
        waitUntil("result") { model.result != nil }

        let folder = try XCTUnwrap(model.exportDirectory)
        XCTAssertEqual(folder.deletingLastPathComponent().standardizedFileURL,
                       URL.documentsDirectory.standardizedFileURL)
        let names = ["report.pdf", "report.txt", "summary.json", "survey.yaml", "velocity.csv",
                     "velocity_map.png"]
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted(),
                       names, "saved without Share being tapped")
        let saved = documents()
        // The flow map, drawn in plan as the results screen shows it.
        let png = try Data(contentsOf: folder.appendingPathComponent("velocity_map.png"))
        let picture = try XCTUnwrap(UIImage(data: png))
        XCTAssertEqual(picture.size.width * picture.scale, 1800, accuracy: 1)
        // And the report: A4, titled, summary + figures + table + survey.
        let report = try XCTUnwrap(pdfDocument(folder.appendingPathComponent("report.pdf")))
        XCTAssertGreaterThanOrEqual(report.numberOfPages, 6)
        XCTAssertEqual(report.page(at: 1)?.getBoxRect(.mediaBox).width ?? 0, 595.28, accuracy: 0.5)
        XCTAssertEqual(pdfTitle(report), "STIV report - \(model.survey.section.name)")
        for number in 1...report.numberOfPages {
            XCTAssertTrue(pdfHasMarks(try XCTUnwrap(report.page(at: number))), "page \(number) is blank")
        }

        XCTAssertEqual(model.survey.capture.frameStep, 2)
        model.setFrameStep(1)
        waitUntil("retuned") { !model.isAnalysing }
        XCTAssertEqual(model.result?.survey.capture.frameStep, 1)
        XCTAssertEqual(Set(documents()), Set(saved), "a retune must not add a folder")
        let summary = try JSONSerialization.jsonObject(
            with: Data(contentsOf: folder.appendingPathComponent("summary.json"))) as? [String: Any]
        XCTAssertEqual((summary?["run"] as? [String: Any])?["frame_step"] as? Int, 1,
                       "the retune's result replaces the first")

        let shared = model.exportFiles()
        XCTAssertEqual(shared.map(\.lastPathComponent).sorted(), names)
        XCTAssertTrue(shared.allSatisfy {
            $0.deletingLastPathComponent().standardizedFileURL == folder.standardizedFileURL
        })
    }

    // MARK: - starting again

    /// Results -> New starts from scratch: nothing tapped on the old world
    /// frame survives into the new one, and the old recording is released.
    func testResultsNewStartsFromScratch() throws {
        let model = surveyed()
        model.go(to: .record)
        model.startRecording(from: aimed())
        try record(model, rows: 150)
        model.stopRecordingAndAnalyse()
        waitUntil("result") { model.result != nil }

        model.startNewSurvey()
        XCTAssertEqual(model.step, .waterPlane)
        XCTAssertNil(model.plane)
        XCTAssertNil(model.layout, "Settings would be enabled on stale geometry")
        XCTAssertNil(model.sectionStart)
        XCTAssertNil(model.sectionEnd)
        XCTAssertNil(model.downstream)
        XCTAssertTrue(model.waterPolygon.isEmpty)
        XCTAssertEqual(model.pickTarget, .sectionStart)
        XCTAssertNil(model.sectionView)
        XCTAssertNil(model.result)
        XCTAssertFalse(model.canRetune)
    }

    /// The way back to the water level from a later screen.  The section goes
    /// with it: each tap was a ray that met the old level.
    func testTheWaterLevelCanBeRemeasuredFromLaterScreens() {
        for screen in [SurveyStep.section, .settings, .record] {
            let model = surveyed()
            model.go(to: screen)
            model.remeasureWaterLevel()
            XCTAssertEqual(model.step, .waterPlane, "from \(screen)")
            XCTAssertNil(model.plane)
            XCTAssertNil(model.layout)
            XCTAssertNil(model.sectionStart)
            XCTAssertEqual(model.pickTarget, .sectionStart)
        }

        // Not in the middle of a recording, though.
        let model = surveyed()
        model.go(to: .record)
        model.startRecording(from: aimed())
        model.remeasureWaterLevel()
        XCTAssertEqual(model.step, .record)
        XCTAssertNotNil(model.plane)
    }

    // MARK: - interruptions and failures

    func testAnInterruptionWhileRecordingSendsTheUserBackToMeasure() throws {
        let model = surveyed()
        model.go(to: .record)
        model.startRecording(from: aimed())
        try record(model, rows: 30)

        model.capture.sessionWasInterrupted(model.capture.session)
        waitUntil("notice") { model.notice != nil }
        XCTAssertEqual(model.step, .waterPlane)
        XCTAssertTrue(model.notice?.contains("recording was stopped") ?? false, model.notice ?? "")
        XCTAssertNil(model.plane, "the plane was measured in a world frame that is gone")
        XCTAssertNil(model.layout)
        XCTAssertNil(model.sectionStart)
        XCTAssertEqual(model.capture.phase, .idle)
        XCTAssertFalse(model.canRetune)

        // The end restarts the camera on a fresh world frame and sets the
        // water-level fit going again: the frame queue goes from idle to
        // fitting.  (In the simulator the restart itself reports that there
        // is no world tracking; on a phone it runs.)
        model.capture.sessionInterruptionEnded(model.capture.session)
        waitUntil("restarted") { model.capture.phase == .fittingPlane }
        XCTAssertEqual(model.step, .waterPlane)
        if !ARWorldTrackingConfiguration.isSupported {
            XCTAssertEqual(model.capture.failure, .unsupported, "the restart was never attempted")
        }
    }

    func testARefusedCameraSaysWhyAndOffersTheWayOut() throws {
        let model = surveyed()
        model.capture.session(model.capture.session,
                              didFailWithError: ARError(.cameraUnauthorized))
        waitUntil("failure") { model.capture.failure != nil }
        XCTAssertEqual(model.step, .waterPlane)
        XCTAssertTrue(model.capture.failure?.needsCameraPermission ?? false)

        model.retryCapture()
        XCTAssertEqual(model.step, .waterPlane)
        XCTAssertFalse(model.capture.failure?.needsCameraPermission ?? false,
                       "a retry must start the camera again, not leave the old failure")
    }

    // MARK: - the phone itself

    /// A thirty-second recording is as long as iOS waits before dimming and
    /// locking an untouched phone, and a locked phone interrupts the session.
    func testTheScreenStaysAwakeWhileTheCameraIsInUse() {
        let model = SurveyModel()
        for step in SurveyStep.allCases {
            model.go(to: step)
            XCTAssertEqual(UIApplication.shared.isIdleTimerDisabled, step.usesCamera, "\(step)")
        }
        model.go(to: .results)
        XCTAssertFalse(UIApplication.shared.isIdleTimerDisabled,
                       "the results screen may sleep")
    }

    func testAHotPhoneIsWarnedAbout() {
        let model = SurveyModel()
        model.updateThermalState(.nominal)
        XCTAssertNil(model.thermalWarning)
        model.updateThermalState(.fair)
        XCTAssertNil(model.thermalWarning)
        model.updateThermalState(.serious)
        let serious = model.thermalWarning
        XCTAssertNotNil(serious)
        model.updateThermalState(.critical)
        XCTAssertNotNil(model.thermalWarning)
        XCTAssertNotEqual(model.thermalWarning, serious)
        model.updateThermalState(.nominal)
        XCTAssertNil(model.thermalWarning)
    }
}

// MARK: - reading back a PDF

func pdfDocument(_ url: URL) -> CGPDFDocument? {
    guard let data = try? Data(contentsOf: url) as CFData,
          let provider = CGDataProvider(data: data) else { return nil }
    return CGPDFDocument(provider)
}

func pdfTitle(_ document: CGPDFDocument) -> String? {
    guard let info = document.info else { return nil }
    var title: CGPDFStringRef?
    guard CGPDFDictionaryGetString(info, "Title", &title), let title else { return nil }
    return CGPDFStringCopyTextString(title) as String?
}

/// The text a PDF page draws, as far as `CGPDFScanner` can tell: enough to
/// check a page is not blank, not to read it back word for word.
func pdfHasMarks(_ page: CGPDFPage) -> Bool {
    var operators = 0
    let table = CGPDFOperatorTableCreate()!
    for name in ["Tj", "TJ", "f", "S", "Do", "B"] {
        CGPDFOperatorTableSetCallback(table, name) { _, info in
            info!.assumingMemoryBound(to: Int.self).pointee += 1
        }
    }
    withUnsafeMutablePointer(to: &operators) { counter in
        let stream = CGPDFContentStreamCreateWithPage(page)
        let scanner = CGPDFScannerCreate(stream, table, counter)
        CGPDFScannerScan(scanner)
        CGPDFScannerRelease(scanner)
        CGPDFContentStreamRelease(stream)
    }
    CGPDFOperatorTableRelease(table)
    return operators > 0
}
