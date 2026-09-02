//  SurveyModel.swift
//  The survey as a sequence of decisions, one per screen.
//
//  Modelled on `stiv/notebook_ui.py`, which exists because a config file with
//  forty fields is not how a person sets up their first measurement.  The order
//  is not arbitrary: each step needs the one before it.  Scale must exist before
//  a tap can mean a distance; the section must exist before a search line can be
//  laid along the flow; the section's geometry sets the frame step that keeps
//  `tan phi` in range.

import ARKit
import Combine
import Foundation
import StivKit

enum SurveyStep: Int, CaseIterable, Comparable {
    case welcome
    case waterPlane
    case section
    case settings
    case record
    case results

    static func < (a: SurveyStep, b: SurveyStep) -> Bool { a.rawValue < b.rawValue }

    var title: String {
        switch self {
        case .welcome:    return "Start"
        case .waterPlane: return "Water level"
        case .section:    return "Cross-section"
        case .settings:   return "Settings"
        case .record:     return "Record"
        case .results:    return "Results"
        }
    }
}

/// What the user is currently tapping for on the cross-section screen.
enum PickTarget: Equatable {
    case sectionStart
    case sectionEnd
    case downstream
    case waterOutline
    case done

    var prompt: String {
        switch self {
        case .sectionStart:
            return "Tap the near bank, at the water's edge, where the cross-section starts."
        case .sectionEnd:
            return "Tap the far bank, at the water's edge, where the cross-section ends."
        case .downstream:
            return "Tap a point downstream, so the search lines point the way the water goes."
        case .waterOutline:
            return "Optional: tap around the water to outline it. This keeps the "
                 + "camera-motion check off the moving surface."
        case .done:
            return "Section set."
        }
    }
}

/// Not actor-isolated: every entry point is either a SwiftUI action (already
/// on the main queue) or a completion the capture session dispatches there.
/// Marking it `@MainActor` would only make those completions illegal.
final class SurveyModel: ObservableObject {
    @Published var step: SurveyStep = .welcome
    @Published var survey = Survey()

    // Geometry the user picks, all in world metres on the water plane.  Storing
    // world points rather than screen points is what lets the overlay survive
    // the phone moving between one tap and the next -- and what lets the water
    // outline be reprojected into captured-image pixels later, since the
    // outline is, by definition, on the water plane.
    @Published var sectionStart: PlanePoint?
    @Published var sectionEnd: PlanePoint?
    @Published var downstream: PlanePoint?
    @Published var waterPolygon: [PlanePoint] = []
    @Published var pickTarget: PickTarget = .sectionStart
    /// Why the last tap did nothing, when it did nothing.
    @Published var tapHint: String?

    @Published private(set) var plane: WaterPlane?
    @Published private(set) var layout: SectionLayout?
    @Published private(set) var layoutError: String?

    @Published private(set) var result: StivResult?
    @Published private(set) var isAnalysing = false
    @Published private(set) var analysisError: String?

    /// Manual water-level entry, for devices with no LiDAR.
    @Published var manualCameraHeightM: Double = 1.5

    let capture = ARCaptureSession()
    let availability = LidarAvailability.current

    /// Kept after recording so the frame step can be changed and the whole
    /// analysis re-run without asking the user to record again.  This is the
    /// single biggest usability difference from the desktop pipeline, where
    /// retuning `frame_step` means another pass over the video.
    private var recording: CompletedRecording?
    private var bridge: DesktopBridge?
    /// Guards `stopRecordingAndAnalyse` against its two concurrent callers.
    private var isFinishing = false
    /// Increments per analysis so an overtaken run cannot publish a stale result.
    private var analysisGeneration = 0

    // MARK: - navigation

    func begin() {
        capture.start()
        survey.capture.framesPerSecond = capture.nominalFrameRate
        step = .waterPlane
        capture.setPhase(.fittingPlane)
    }

    func go(to next: SurveyStep) {
        step = next
        switch next {
        case .waterPlane: capture.setPhase(.fittingPlane)
        case .section, .settings: capture.setPhase(.picking)
        case .record: capture.setPhase(.picking)   // recording starts on the button
        case .welcome, .results: capture.setPhase(.idle)
        }
    }

    // MARK: - water plane

    func acceptMeasuredPlane(onWaterSurface: Bool) {
        capture.acceptPlane(onWaterSurface: onWaterSurface) { [weak self] plane in
            guard let self, let plane else { return }
            self.adopt(plane)
        }
    }

    func acceptManualPlane() {
        guard let plane = capture.acceptManualPlane(
            cameraHeightAboveWater: manualCameraHeightM) else { return }
        adopt(plane)
    }

    private func adopt(_ plane: WaterPlane) {
        self.plane = plane
        // The plane's own fit residual *is* the height term of the scale error
        // budget, so it is wired straight into the uncertainty model rather than
        // left as a display value.  An untidy sweep therefore widens the
        // reported sigma instead of silently biasing the velocities.
        survey.scale.planeHeightSigmaM = plane.fitResidualM
        go(to: .section)
    }

    // MARK: - picking

    /// A tap whose ray never met the water plane.
    ///
    /// With a sound plane fit this only happens above the horizon.  If it
    /// happens *everywhere*, the plane is wrong -- fitted above the camera, or
    /// with a typed height of the wrong sign -- and the fix is to measure the
    /// water level again, not to keep tapping.
    func tapMissed() {
        tapHint = plane == nil
            ? "Measure the water level first."
            : "That tap did not meet the water surface -- it is above the horizon. "
            + "If no tap works, the water level is wrong: measure it again."
    }

    func handleTap(world: PlanePoint) {
        tapHint = nil
        switch pickTarget {
        case .sectionStart:
            sectionStart = world
            pickTarget = .sectionEnd
        case .sectionEnd:
            sectionEnd = world
            pickTarget = .downstream
        case .downstream:
            downstream = world
            pickTarget = .waterOutline
            rebuildLayout()
        case .waterOutline:
            waterPolygon.append(world)
        case .done:
            break
        }
    }

    func undoLastTap() {
        switch pickTarget {
        case .sectionStart:
            break
        case .sectionEnd:
            sectionStart = nil
            pickTarget = .sectionStart
        case .downstream:
            sectionEnd = nil
            pickTarget = .sectionEnd
        case .waterOutline where !waterPolygon.isEmpty:
            waterPolygon.removeLast()
        case .waterOutline, .done:
            downstream = nil
            waterPolygon.removeAll()
            pickTarget = .downstream
        }
        rebuildLayout()
    }

    var sectionWidthM: Double? {
        guard let sectionStart, let sectionEnd else { return nil }
        return distance(sectionStart, sectionEnd)
    }

    /// Recompute the search lines whenever the geometry or a setting changes.
    ///
    /// Cheap, and worth doing eagerly: the overlay shows the lines the analysis
    /// will actually use, so a search line hanging over the bank is visible
    /// before thirty seconds are spent recording it.
    func rebuildLayout() {
        layoutError = nil
        guard let sectionStart, let sectionEnd, let plane, let camera = capture.camera else {
            layout = nil
            return
        }
        survey.section.start = sectionStart
        survey.section.end = sectionEnd
        survey.flow.direction = .auto
        survey.flow.downstream = downstream

        do {
            let validated = try survey.validated()
            layout = try buildSection(survey: validated,
                                      rectifier: camera.projection(onto: plane))
        } catch {
            layout = nil
            layoutError = "\(error)"
        }
    }

    /// A frame step that puts the median `tan phi` near 1 for the velocity the
    /// user expects, given the geometry they just picked.
    ///
    /// `tan phi = U St / Sx`, so with `Sx` fixed by the section and `St` set by
    /// the step, this is the one knob -- and picking it well before recording
    /// beats discovering it afterwards, even though the results screen can also
    /// change it after the fact.
    var suggestedFrameStep: Int {
        guard let layout, let line = layout.searchLines.first else { return 2 }
        let expected = 0.5 * (survey.sti.velocityRangeMs.lowerBound
                              + survey.sti.velocityRangeMs.upperBound)
        let velocity = max(abs(expected), 0.05)
        let step = line.sxMetresPerColumn / (velocity / survey.capture.framesPerSecond)
        return min(max(Int(step.rounded()), 1), 30)
    }

    // MARK: - recording

    func startRecording() {
        guard let plane, let layout, let camera = capture.camera else { return }
        let plan = RecordingPlan(survey: survey, plane: plane, layout: layout,
                                 referenceIntrinsics: camera.stivIntrinsics)

        // The water outline is on the water plane, so it reprojects into
        // captured-image pixels through the same rectifier everything else uses.
        let projection = camera.projection(onto: plane)
        let polygon = waterPolygon.compactMap { projection.worldToImage($0) }
        bridge = makeBridge(projection: projection, layout: layout)

        // A new run invalidates the previous one, both so the finish guard will
        // fire again and so the results screen cannot show the old profile
        // alongside a recording in progress.
        analysisGeneration += 1
        isAnalysing = false
        recording = nil
        result = nil
        analysisError = nil

        capture.startRecording(plan: plan,
                               waterPolygon: polygon.count >= 3 ? polygon : nil)
    }

    /// Stop recording and analyse, at most once per recording.
    ///
    /// Two things race to call this: the Stop button, and the `.onChange` that
    /// fires when the recording reaches its requested duration.  Both are
    /// legitimate -- whichever happens first should end the run -- so the method
    /// has to be idempotent rather than exclusive.  Without the guard the second
    /// caller reaches `finishRecording` after the first has already taken the
    /// recorder, gets nil back, and reports "Nothing was recorded" over a
    /// perfectly good measurement.
    func stopRecordingAndAnalyse() {
        guard !isFinishing, recording == nil else { return }
        isFinishing = true
        capture.finishRecording { [weak self] completed in
            guard let self else { return }
            self.isFinishing = false
            guard let completed else {
                self.analysisError = "Nothing was recorded."
                return
            }
            self.recording = completed
            self.step = .results
            self.analyse()
        }
    }

    /// Run (or re-run) the analysis over the recording already in hand.
    ///
    /// The frame-step control can start a second analysis while the first is
    /// still running, and a shorter step is *faster* to analyse -- fewer rows --
    /// so the results can genuinely arrive out of order.  Each run therefore
    /// carries a generation, and only the newest is allowed to publish; an
    /// overtaken run is discarded rather than left to overwrite the answer the
    /// user is looking at.
    func analyse() {
        guard let recording else { return }
        analysisGeneration += 1
        let generation = analysisGeneration
        isAnalysing = true
        analysisError = nil
        var survey = self.survey
        survey.capture.framesPerSecond = recording.plan.survey.capture.framesPerSecond

        let recorder = recording.recorder
        let layout = recording.plan.layout
        let motion = recording.motion

        // Thirty STIs through the structure tensor is a second or two of work:
        // enough to freeze the UI, not enough to need progress reporting.
        DispatchQueue.global(qos: .userInitiated).async {
            let stis = recorder.finish(frameStep: survey.capture.frameStep)
            let result = StivKit.analyse(survey: survey, layout: layout,
                                         stis: stis, motion: motion)
            DispatchQueue.main.async {
                guard generation == self.analysisGeneration else { return }
                self.result = result
                self.isAnalysing = false
            }
        }
    }

    /// Change the frame step and re-analyse in place.
    func setFrameStep(_ step: Int) {
        guard step != survey.capture.frameStep else { return }
        survey.capture.frameStep = step
        analyse()
    }

    var canRetune: Bool { recording != nil }

    // MARK: - export

    private func makeBridge(projection: WaterPlaneProjection,
                            layout: SectionLayout) -> DesktopBridge? {
        guard let start = projection.worldToImage(layout.start),
              let end = projection.worldToImage(layout.end) else { return nil }
        return DesktopBridge(
            imageWidth: projection.intrinsics.width,
            imageHeight: projection.intrinsics.height,
            sectionStartPx: start,
            sectionEndPx: end,
            downstreamPx: downstream.flatMap { projection.worldToImage($0) },
            controlPoints: projection.controlPoints())
    }

    /// Write the three export artefacts to a temporary directory and return
    /// their URLs for the share sheet.
    func exportFiles() -> [URL] {
        guard let result else { return [] }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("stiv-\(Int(Date().timeIntervalSince1970))",
                                    isDirectory: true)
        var urls: [URL] = []
        do {
            try FileManager.default.createDirectory(at: directory,
                                                    withIntermediateDirectories: true)
            let csv = directory.appendingPathComponent("velocity.csv")
            try velocityCSV(result).write(to: csv, atomically: true, encoding: .utf8)
            urls.append(csv)

            let json = directory.appendingPathComponent("summary.json")
            try summaryJSON(result, bridge: bridge).write(to: json)
            urls.append(json)

            let yaml = directory.appendingPathComponent("survey.yaml")
            try surveyYAML(result: result, bridge: bridge)
                .write(to: yaml, atomically: true, encoding: .utf8)
            urls.append(yaml)

            let report = directory.appendingPathComponent("report.txt")
            try result.summary().write(to: report, atomically: true, encoding: .utf8)
            urls.append(report)
        } catch {
            analysisError = "Could not write the export: \(error.localizedDescription)"
        }
        return urls
    }
}
