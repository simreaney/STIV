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
import UIKit

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

    /// The steps with the camera running and the world frame in use.
    var usesCamera: Bool {
        switch self {
        case .waterPlane, .section, .settings, .record: return true
        case .welcome, .results:                        return false
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
    /// The screen on show.  Setting it also decides whether the screen may
    /// sleep: see `keepsScreenAwake`.
    @Published var step: SurveyStep = .welcome {
        didSet { applyIdleTimer() }
    }
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

    /// Why Start was refused.  Cleared by the next attempt.
    @Published private(set) var recordingError: String?
    /// What Start had to give up to stay within memory, while that recording
    /// lasts.
    @Published private(set) var recordingNotice: String?
    /// How long the recording in progress will run: the requested duration,
    /// unless the memory budget shortened it.
    @Published private(set) var recordingSeconds: Double?
    /// True from Stop until the recording has been handed over: the record
    /// button is disabled meanwhile, so a fast Stop-Start cannot begin a run
    /// before the last one is safely out of the frame queue.
    @Published private(set) var isFinishing = false

    @Published private(set) var result: StivResult?
    @Published private(set) var isAnalysing = false
    @Published private(set) var analysisError: String?

    /// Something that happened to the survey that the user did not ask for --
    /// an interruption, typically -- and what to do about it.
    @Published private(set) var notice: String?
    /// Set while the phone is hot enough for iOS to throttle it.
    @Published private(set) var thermalWarning: String?

    /// Manual water-level entry, for devices with no LiDAR.
    @Published var manualCameraHeightM: Double = 1.5

    let capture = ARCaptureSession()
    let availability = LidarAvailability.current

    /// The camera's view right now: `capture.camera`, frozen.  A seam for the
    /// tests, which have no camera and stand in views of their own.
    var currentView: () -> CameraSnapshot? = { nil }

    /// The camera view at the last cross-section tap, while the user was aiming
    /// at the section.  Layouts are built from it until recording starts, when
    /// the live view on the record screen takes over; the settings screen shows
    /// no camera feed, so its own pose means nothing.
    private(set) var sectionView: CameraSnapshot?

    /// Kept after recording so the frame step can be changed and the whole
    /// analysis re-run without asking the user to record again.  This is the
    /// single biggest usability difference from the desktop pipeline, where
    /// retuning `frame_step` means another pass over the video.
    private var recording: CompletedRecording?
    private var bridge: DesktopBridge?
    /// Increments per recording, and when an interruption abandons one, so a
    /// recording handed back after its run was abandoned is dropped.
    private var recordingRun = 0
    /// Increments per analysis so an overtaken run cannot publish a stale result.
    private var analysisGeneration = 0
    /// The same number, readable from the analysis thread.
    private let analysisTicket = AnalysisTicket()
    /// Whether an analysis is running.  At most one is: see `analyse`.
    private var analysisInFlight = false
    /// Set by an interruption: the session must be restarted, from a new world
    /// frame, when it ends.
    private var awaitingRestart = false
    private var cancellables: Set<AnyCancellable> = []

    init() {
        currentView = { [capture] in capture.camera.map(CameraSnapshot.init) }
        capture.onEvent = { [weak self] event in self?.handle(event) }
        NotificationCenter.default
            .publisher(for: ProcessInfo.thermalStateDidChangeNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.updateThermalState(ProcessInfo.processInfo.thermalState)
            }
            .store(in: &cancellables)
        updateThermalState(ProcessInfo.processInfo.thermalState)
    }

    // MARK: - navigation

    func begin() {
        capture.start()
        survey.capture.framesPerSecond = capture.nominalFrameRate
        go(to: .waterPlane)
    }

    func go(to next: SurveyStep) {
        step = next
        switch next {
        case .waterPlane: capture.setPhase(.fittingPlane)
        case .section, .settings: capture.setPhase(.picking)
        case .record: capture.setPhase(.picking)   // recording starts on the button
        case .welcome: capture.setPhase(.idle)
        case .results:
            // Nothing on the results screen needs the camera: the recording is
            // in hand.  Leaving LiDAR and 60 fps tracking running through the
            // analysis would only heat the phone and slow the analysis down.
            capture.setPhase(.idle)
            capture.pause()
        }
    }

    /// Results -> New: a new survey from scratch, on a fresh world frame.
    ///
    /// Everything world-anchored goes -- the plane, the section, the layout
    /// built on them -- because the session restarts with its tracking reset,
    /// and none of it would mean anything in the new frame.  The recording and
    /// its result go too, so their memory is released now rather than when the
    /// next recording replaces them.
    func startNewSurvey() {
        forgetWorld()
        invalidateAnalyses()
        recording = nil
        result = nil
        analysisError = nil
        notice = nil
        capture.start()
        survey.capture.framesPerSecond = capture.nominalFrameRate
        go(to: .waterPlane)
    }

    /// Back to the water level from a later screen.
    ///
    /// The cross-section goes with it: each tap was a ray that met the *old*
    /// level, and the same ray meets a different level somewhere else.
    func remeasureWaterLevel() {
        guard capture.phase != .recording, !isFinishing else { return }
        forgetWorld()
        notice = nil
        go(to: .waterPlane)
    }

    /// Forget everything anchored to the current world frame.
    private func forgetWorld() {
        plane = nil
        layout = nil
        layoutError = nil
        sectionStart = nil
        sectionEnd = nil
        downstream = nil
        waterPolygon = []
        pickTarget = .sectionStart
        tapHint = nil
        sectionView = nil
        bridge = nil
        recordingError = nil
        recordingNotice = nil
        recordingSeconds = nil
    }

    // MARK: - screen and heat

    /// Whether the screen must stay on.
    ///
    /// iOS dims and then locks an idle phone after thirty seconds or so, which
    /// is exactly how long a recording asks the user to keep still for -- and a
    /// locked phone interrupts the AR session.  So the idle timer is off
    /// whenever the camera is in use, and back on for the welcome and results
    /// screens, where sleeping costs nothing.
    var keepsScreenAwake: Bool { step.usesCamera }

    private func applyIdleTimer() {
        UIApplication.shared.isIdleTimerDisabled = keepsScreenAwake
    }

    func updateThermalState(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .serious:
            thermalWarning = "The phone is hot, so iOS may slow the camera and drop frames. "
                + "Shade it, and let it cool between recordings."
        case .critical:
            thermalWarning = "The phone is overheating: iOS is throttling it and may stop the "
                + "camera. Let it cool in the shade before recording."
        default:
            thermalWarning = nil
        }
    }

    // MARK: - interruptions and failures

    private func handle(_ event: CaptureEvent) {
        switch event {
        case .interrupted(let wasRecording):
            guard step.usesCamera else { return }
            awaitingRestart = true
            abandonRecording()
            forgetWorld()
            notice = (wasRecording ? "The recording was stopped: the camera was interrupted, "
                                   + "and nothing recorded across an interruption can be "
                                   + "trusted. "
                                   : "The camera was interrupted. ")
                + "The world frame the water level and the cross-section were measured in "
                + "did not survive it, so measure the water level again."
            // Show the water-level screen now; the session restarts when the
            // interruption ends.
            step = .waterPlane

        case .interruptionEnded:
            guard awaitingRestart else { return }
            awaitingRestart = false
            capture.start()
            go(to: .waterPlane)

        case .failed(let wasRecording):
            guard step.usesCamera else { return }
            abandonRecording()
            forgetWorld()
            if wasRecording {
                notice = "The recording was stopped: the camera failed. Nothing from it was kept."
            }
            step = .waterPlane
        }
    }

    /// Drop the recording in progress, or the one being handed over.
    private func abandonRecording() {
        recordingRun += 1
        isFinishing = false
    }

    /// Start the camera again after a failure, from a fresh world frame.
    func retryCapture() {
        forgetWorld()
        notice = nil
        capture.start()
        survey.capture.framesPerSecond = capture.nominalFrameRate
        go(to: .waterPlane)
    }

    func dismissNotice() { notice = nil }

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

    func adopt(_ plane: WaterPlane) {
        self.plane = plane
        notice = nil
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
            + "If no tap works, the water level is wrong: use Re-measure."
    }

    /// A tap that landed on the water plane.
    ///
    /// - Parameter view: the camera view the tap was made in; the live camera
    ///   when omitted.  Kept, because it is the view the user was aiming at
    ///   the section from -- see `sectionView`.
    func handleTap(world: PlanePoint, seenFrom view: CameraSnapshot? = nil) {
        tapHint = nil
        if let view = view ?? currentView() { sectionView = view }
        switch pickTarget {
        case .sectionStart:
            sectionStart = world
            pickTarget = .sectionEnd
        case .sectionEnd:
            sectionEnd = world
            pickTarget = .downstream
            rebuildLayout()
        case .downstream:
            downstream = world
            pickTarget = .waterOutline
            rebuildLayout()
        case .waterOutline:
            // The outline does not touch the layout: nothing to rebuild.
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
    ///
    /// Built from `sectionView`, the view the section was tapped in, not from
    /// whatever the phone points at now.  Recording rebuilds it once more from
    /// the live view; see `startRecording`.
    func rebuildLayout() {
        if sectionStart != nil, sectionEnd != nil { survey = picked(survey) }
        layout = buildLayout(survey: survey,
                             view: sectionView ?? currentView())
    }

    /// `survey` with the picked geometry written into it.
    private func picked(_ survey: Survey) -> Survey {
        var s = survey
        if let sectionStart { s.section.start = sectionStart }
        if let sectionEnd { s.section.end = sectionEnd }
        s.flow.direction = .auto
        s.flow.downstream = downstream
        return s
    }

    /// The layout for `survey` from `view`, or `nil` with `layoutError` set.
    private func buildLayout(survey: Survey, view: CameraSnapshot?) -> SectionLayout? {
        layoutError = nil
        guard sectionStart != nil, sectionEnd != nil, let plane, let view else { return nil }
        do {
            return try buildSection(survey: picked(survey).validated(),
                                    rectifier: view.projection(onto: plane))
        } catch {
            layoutError = "\(error)"
            return nil
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
        return min(max(Int(step.rounded()), 1), maxFrameStep)
    }

    /// The fewest STI rows -- and samples along a line -- that hold one
    /// structure-tensor window at the analysis settings in force: those of the
    /// recording in hand, or the current ones before there is one.
    var minimumStiRows: Int {
        let sti = recording?.plan.survey.sti ?? survey.sti
        return minStiSize(windowPx: sti.windowPx, windowType: sti.windowType,
                          gradientOrder: sti.gradientOrder)
    }

    /// The largest frame step the analysis can accept: the recording in hand,
    /// or the one planned, divided into STIs of at least `minimumStiRows`.
    ///
    /// The frame-step controls stop here, because StivKit refuses a shorter
    /// STI outright.  The bound keeps a row spare against the rounding of the
    /// time grid, so a step at the limit is never refused.
    var maxFrameStep: Int {
        // StivKit grids the STI at the *median* frame period, not at the
        // average one the row count implies: `finish(frameStep: s)` gives
        // floor(T / (P s)) + 1 rows over a recording of T seconds.  A phone
        // that drops frames has a median above its average, so counting rows
        // would advertise steps the analysis then refuses.
        let seconds: Double
        let period: Double
        if let recording {
            seconds = recording.recording.recordedSeconds
            period = recording.recording.medianFramePeriod ?? 1 / survey.capture.framesPerSecond
        } else {
            seconds = recordingEstimate?.seconds ?? survey.capture.durationSeconds
            period = 1 / survey.capture.framesPerSecond
        }
        // At most T / (P m): a row spare against the grid's rounding.
        let limit = seconds / (period * Double(max(minimumStiRows, 1)))
        guard limit.isFinite else { return 1 }
        return min(max(Int(limit.rounded(.down)), 1), 30)
    }

    /// What the current layout would cost to record at the current settings,
    /// for the settings and record screens.  Provisional: Start rebuilds the
    /// layout from the live view and decides again.
    var recordingEstimate: RecordingBudget.Decision? {
        guard let layout else { return nil }
        return RecordingBudget.decide(sampleCounts: layout.searchLines.map(\.sampleCount),
                                      framesPerSecond: survey.capture.framesPerSecond,
                                      requestedSeconds: survey.capture.durationSeconds,
                                      minimumSamplesPerLine: thinningFloor)
    }

    /// Thinning never takes a line below one analysis window across, or the
    /// station could not be analysed at all.
    private var thinningFloor: Int {
        max(RecordingBudget.minimumSamplesPerLine,
            minStiSize(windowPx: survey.sti.windowPx, windowType: survey.sti.windowType,
                       gradientOrder: survey.sti.gradientOrder))
    }

    // MARK: - recording

    /// Start a recording, if the phone is pointing at the section and the
    /// recording will fit in memory.
    ///
    /// - Parameter view: the camera view to build the layout from; the live
    ///   camera when omitted.  On the record screen that is the view the user
    ///   is aiming with, which is the only one whose pixels the search lines
    ///   will actually be sampled from.
    func startRecording(from view: CameraSnapshot? = nil) {
        guard !isFinishing, capture.phase != .recording else { return }
        recordingError = nil
        recordingNotice = nil
        recordingSeconds = nil
        guard let plane else {
            recordingError = "Measure the water level first."
            return
        }
        guard let view = view ?? currentView() else {
            recordingError = "The camera has not delivered a frame yet. Wait a moment, then "
                + "start again."
            return
        }

        // Rebuild from this view, and check that it actually sees the section.
        var planned = picked(survey)
        guard let aimed = buildLayout(survey: planned, view: view) else {
            recordingError = layoutError ?? "The cross-section is not set."
            return
        }
        let projection = view.projection(onto: plane)
        if let problem = RecordingCheck.problem(layout: aimed, projection: projection) {
            recordingError = problem
            layout = aimed   // so the overlay shows what was checked
            return
        }

        // Then fit it to memory: shorter first, thinner only if it must be.
        let decision = RecordingBudget.decide(
            sampleCounts: aimed.searchLines.map(\.sampleCount),
            framesPerSecond: planned.capture.framesPerSecond,
            requestedSeconds: planned.capture.durationSeconds,
            minimumSamplesPerLine: thinningFloor)
        var recorded = aimed
        if let perLine = decision.samplesPerLine {
            planned.sti.sampleCount = perLine
            guard let thinned = buildLayout(survey: planned, view: view) else {
                recordingError = layoutError
                return
            }
            recorded = thinned
        }
        planned.capture.durationSeconds = decision.seconds
        recordingNotice = decision.notice
        recordingSeconds = decision.seconds
        layout = recorded

        let plan = RecordingPlan(survey: planned, plane: plane, layout: recorded)

        // The water outline is on the water plane, so it reprojects into
        // captured-image pixels through the same rectifier everything else uses.
        let polygon = waterPolygon.compactMap { projection.worldToImage($0) }
        bridge = makeBridge(projection: projection, layout: recorded)

        // A new run invalidates the previous one, both so the finish guard will
        // fire again and so the results screen cannot show the old profile
        // alongside a recording in progress.  Dropping the old recording here
        // also returns its memory before the new one starts to grow.
        recordingRun += 1
        invalidateAnalyses()
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
    ///
    /// The hand-over is tied to the run that asked for it: if the run was
    /// abandoned meanwhile -- an interruption -- whatever comes back is not
    /// this survey's any more, and is dropped.
    func stopRecordingAndAnalyse() {
        guard !isFinishing, recording == nil else { return }
        isFinishing = true
        let run = recordingRun
        capture.finishRecording { [weak self] completed in
            guard let self, run == self.recordingRun else { return }
            self.isFinishing = false
            guard let completed else {
                self.analysisError = "Nothing was recorded."
                return
            }
            self.recording = completed
            // A recording stopped early may be too short for the step chosen
            // before it: start from the largest step it can be analysed at.
            self.survey.capture.frameStep = min(self.survey.capture.frameStep,
                                                self.maxFrameStep)
            self.go(to: .results)
            self.analyse()
        }
    }

    // MARK: - analysis

    /// Run (or re-run) the analysis over the recording already in hand.
    ///
    /// The frame-step control can ask for a new analysis while one is still
    /// running, and every request costs a full pass over a recording that may
    /// hold hundreds of megabytes.  So requests do not overlap: at most one
    /// analysis runs, and a request made meanwhile only marks itself as the
    /// newest.  When the running one ends it publishes only if nothing newer
    /// was asked for, and otherwise starts the newest -- skipping every request
    /// in between.  StivKit's analysis cannot be stopped part-way, so a
    /// superseded run is abandoned at the first point it can be: before
    /// resampling, and again before the structure tensor.
    func analyse() {
        guard recording != nil else { return }
        analysisGeneration += 1
        analysisTicket.set(analysisGeneration)
        isAnalysing = true
        analysisError = nil
        if !analysisInFlight { runNewestAnalysis() }
    }

    private func runNewestAnalysis() {
        guard let recording else {
            isAnalysing = false
            return
        }
        analysisInFlight = true
        let generation = analysisGeneration
        let ticket = analysisTicket
        // The recording's own settings -- its duration, and any thinning the
        // memory budget imposed -- with the one knob the results screen turns.
        var survey = recording.plan.survey
        survey.capture.frameStep = self.survey.capture.frameStep
        let samples = recording.recording
        let layout = recording.plan.layout
        let motion = recording.motion

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            ticket.enter()
            var outcome: Result<StivResult, Error>?
            if ticket.isCurrent(generation) {
                let stis = samples.finish(frameStep: survey.capture.frameStep)
                if ticket.isCurrent(generation) {
                    // The validating entry point: a recording that does not
                    // match its layout is refused with a reason, rather than
                    // analysed into numbers that carry only a warning.
                    outcome = Result {
                        try StivKit.analyse(validating: survey, layout: layout,
                                            stis: stis, motion: motion)
                    }
                }
            }
            ticket.leave()
            DispatchQueue.main.async {
                guard let self else { return }
                self.analysisInFlight = false
                if generation == self.analysisGeneration, let outcome {
                    switch outcome {
                    case .success(let result):
                        self.result = result
                    case .failure(let error):
                        // Keep the last good result on screen: it is still a
                        // true answer at its own frame step, and the retune
                        // control lives beside it.
                        self.analysisError = "The analysis refused this recording at frame "
                            + "step \(survey.capture.frameStep): \(error)"
                    }
                    self.isAnalysing = false
                } else if self.isAnalysing, self.recording != nil {
                    self.runNewestAnalysis()
                } else {
                    self.isAnalysing = false
                }
            }
        }
    }

    /// Make every analysis requested so far stale.
    private func invalidateAnalyses() {
        analysisGeneration += 1
        analysisTicket.set(analysisGeneration)
        isAnalysing = false
    }

    /// The most analyses ever running at once.  One, if `analyse` does what it
    /// says; for the tests.
    var peakConcurrentAnalyses: Int { analysisTicket.peak }

    /// Change the frame step and re-analyse in place, within what the
    /// recording can be analysed at.
    func setFrameStep(_ step: Int) {
        let step = min(max(step, 1), maxFrameStep)
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

/// The newest analysis generation, shared with the analysis thread so a
/// superseded run can notice and stop early; and a count of runs in progress.
private final class AnalysisTicket {
    private let lock = NSLock()
    private var current = 0
    private var running = 0
    private var highWater = 0

    func set(_ generation: Int) {
        lock.lock(); defer { lock.unlock() }
        current = generation
    }

    func isCurrent(_ generation: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return current == generation
    }

    func enter() {
        lock.lock(); defer { lock.unlock() }
        running += 1
        highWater = max(highWater, running)
    }

    func leave() {
        lock.lock(); defer { lock.unlock() }
        running -= 1
    }

    var peak: Int {
        lock.lock(); defer { lock.unlock() }
        return highWater
    }
}