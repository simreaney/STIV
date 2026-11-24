//  ARCaptureSession.swift
//  The one ARSession, and what each phase of the survey does with its frames.
//
//  This replaces `stiv/video.py`.  The desktop reads a file and derives
//  `St = frame_step / fps` from the container's metadata; here frames arrive
//  live, each carrying its own timestamp and its own camera pose, and both are
//  recorded.  A phone cannot assume uniform frame timing -- a warm device
//  quietly delivers fewer frames than the video format advertises -- so the
//  timestamps are kept and `StiRecorder` resamples onto a uniform grid.
//
//  Camera motion is compensated by construction rather than corrected after the
//  fact: the sample points are fixed world points on the water plane, and each
//  frame projects them through its own pose.  That is the whole of
//  `stiv/stabilise.py`, replaced by the arithmetic in `sampleRow` -- and better,
//  because a 2-D image warp cannot represent what a translating camera does to a
//  scene with depth, whereas a 6-DoF pose can.
//
//  Threading: ARKit delivers frames on `frameQueue`, and every mutable thing
//  the per-frame path touches -- the phase, the plan, the recorder, the residual
//  checker -- is owned by that queue.  The `@Published` properties exist only to
//  drive the UI and are written on the main queue.  Nothing derived from a
//  frame's pixel buffers is allowed to escape the delegate call; ARKit recycles
//  those buffers.

import ARKit
import Combine
import Foundation
import StivKit

/// What the session is currently doing with its frames.
enum CapturePhase: Equatable {
    /// Running, but nothing is being accumulated.
    case idle
    /// Collecting LiDAR samples of the water surface.
    case fittingPlane
    /// Waiting for taps: the cross-section and the flow direction.
    case picking
    /// Reading one STI row per frame.
    case recording
}

/// Something the survey has to react to, rather than merely display.
enum CaptureEvent: Equatable {
    /// The session was interrupted -- a call, the app sent to the background,
    /// the camera taken by another app.  Any recording in progress has been
    /// dropped.
    case interrupted(wasRecording: Bool)
    /// The interruption is over and frames can arrive again.
    case interruptionEnded
    /// The session stopped with an error; see `ARCaptureSession.failure`.
    case failed(wasRecording: Bool)
}

/// Why the camera is not running, phrased for the person holding the phone.
struct CaptureFailure: Equatable {
    let message: String
    /// The fix is in the Settings app: camera access was refused.
    let needsCameraPermission: Bool

    init(message: String, needsCameraPermission: Bool = false) {
        self.message = message
        self.needsCameraPermission = needsCameraPermission
    }

    init(_ error: Error) {
        guard let error = error as? ARError else {
            self.init(message: "The camera stopped: \(error.localizedDescription)")
            return
        }
        switch error.code {
        case .cameraUnauthorized:
            self.init(message: "STIV needs the camera to film the river, and access to it "
                             + "is turned off. Allow camera access for STIV in Settings, then "
                             + "come back and try again.",
                      needsCameraPermission: true)
        case .unsupportedConfiguration:
            self.init(message: LidarAvailability.noWorldTracking.explanation)
        case .sensorUnavailable, .sensorFailed:
            self.init(message: "The camera or motion sensors stopped working. Close other "
                             + "apps that use the camera, then try again.")
        case .worldTrackingFailed:
            self.init(message: "World tracking failed, so distances can no longer be "
                             + "trusted. Try again, and move the phone slowly at first.")
        default:
            self.init(message: "The camera stopped: \(error.localizedDescription)")
        }
    }

    static let unsupported = CaptureFailure(message: LidarAvailability.noWorldTracking.explanation)
}

/// Everything fixed at the moment recording starts.
///
/// Precomputed deliberately: the per-frame path should do arithmetic and nothing
/// else, because it runs sixty times a second while a person holds a phone still
/// and hopes.
struct RecordingPlan {
    let survey: Survey
    let plane: WaterPlane
    let layout: SectionLayout
    /// Every search line's samples, flattened in station order, as 3-D world
    /// points on the water plane.
    let worldPoints: [(x: Double, y: Double, z: Double)]
    let sampleCounts: [Int]
    let sxPerStation: [Double]
    /// The section's midpoint, for the frame-shift diagnostic.
    let sectionCentre: PlanePoint

    init(survey: Survey, plane: WaterPlane, layout: SectionLayout) {
        self.survey = survey
        self.plane = plane
        self.layout = layout
        self.sampleCounts = layout.searchLines.map(\.sampleCount)
        self.sxPerStation = layout.searchLines.map(\.sxMetresPerColumn)
        self.sectionCentre = PlanePoint(x: 0.5 * (layout.start.x + layout.end.x),
                                        z: 0.5 * (layout.start.z + layout.end.z))
        var points: [(x: Double, y: Double, z: Double)] = []
        points.reserveCapacity(layout.searchLines.reduce(0) { $0 + $1.sampleCount })
        for line in layout.searchLines {
            for sample in line.samples {
                points.append((sample.x, plane.y, sample.z))
            }
        }
        self.worldPoints = points
    }

    /// Rows the recorder should expect: the duration at the nominal rate, and a
    /// few spare for a frame that arrives early.
    var expectedRows: Int {
        Int(survey.capture.durationSeconds * survey.capture.framesPerSecond) + 8
    }

    /// Read one STI row: project every world sample point through *this*
    /// frame's pose and sample the luma plane there.
    ///
    /// This is the whole of the camera-motion compensation.  Because the points
    /// are anchored in the world rather than on the sensor, a hand that drifts
    /// ten pixels between frames changes *where in the image* the samples are
    /// taken, not *which patch of river* they describe.
    ///
    /// `correction` is what the pose cross-check found the pose missed.  It is
    /// a correction to the *pose*, so each sample gets the shift its own depth
    /// implies; see `PoseCorrection`.  The projection itself is StivKit's.
    ///
    /// Takes the luma plane, intrinsics and pose rather than an `ARFrame` so
    /// that `StivAppTests` can drive it with rendered frames of a river whose
    /// velocity is known -- an `ARFrame` cannot be constructed outside ARKit.
    func sampleRow(luma: LumaPlane, intrinsics: PinholeIntrinsics, pose: CameraPose,
                   correction: PoseCorrection) -> [Float] {
        let projection = WaterPlaneProjection(intrinsics: intrinsics,
                                              pose: correction.corrected(pose),
                                              plane: plane)
        var row = [Float](repeating: .nan, count: worldPoints.count)
        for i in worldPoints.indices {
            let p = worldPoints[i]
            // Behind the camera, there is no pixel: leave it missing.
            guard let px = projection.worldToImage(PlanePoint(x: p.x, z: p.z)) else { continue }
            if let value = luma.sample(u: px.u, v: px.v) { row[i] = value }
            // Otherwise the point left the frame, and NaN is the honest
            // answer: `SpaceTimeImage.outOfFrameFraction` counts it and
            // `preprocess` fills it from the column's time-mean.
        }
        return row
    }

    /// Each search line as an image segment, end to end, through `projection`:
    /// the corridors the pose cross-check must keep its patches out of.
    ///
    /// A straight line on the plane projects to a straight line in the image,
    /// so the first and last samples that land in front of the camera are its
    /// ends.
    func searchLineSegments(projection: WaterPlaneProjection) -> [ImageSegment] {
        func firstProjected<S: Sequence>(_ samples: S) -> PixelPoint?
            where S.Element == PlanePoint {
            for sample in samples {
                if let p = projection.worldToImage(sample) { return p }
            }
            return nil
        }
        return layout.searchLines.compactMap { line in
            guard let first = firstProjected(line.samples),
                  let last = firstProjected(line.samples.reversed()) else { return nil }
            return ImageSegment(a: first, b: last)
        }
    }
}

/// A finished recording, ready for `StivKit.analyse`.
///
/// Holds the recorder's `snapshot`, not the recorder: an immutable value that
/// can go to the analysis thread safely, taken on the frame queue that did the
/// appending.  With the recorder itself released at the same moment, the
/// snapshot is the only owner of the samples and nothing is copied.
struct CompletedRecording {
    let recording: StiRecording
    let plan: RecordingPlan
    let motion: MotionReport
}

/// Owns the `ARSession` and routes its frames according to the current phase.
final class ARCaptureSession: NSObject, ObservableObject {
    let session = ARSession()

    // Published state, for the UI.  Deliberately small: an `ARFrame` retains its
    // pixel buffers and ARKit recycles them, so nothing publishes a frame.
    // `ARCamera` is an immutable snapshot and safe to hold.
    @Published private(set) var phase: CapturePhase = .idle
    @Published private(set) var camera: ARCamera?
    @Published private(set) var trackingState: ARCamera.TrackingState = .notAvailable
    @Published private(set) var depthAvailable = false
    /// From 0 to 1, how far the water-level sweep has got.
    @Published private(set) var planeProgress = 0.0
    @Published private(set) var planeReady = false
    @Published private(set) var planeGuidance = ""
    @Published private(set) var recordedRows = 0
    @Published private(set) var recordedSeconds: Double = 0
    @Published private(set) var motionReport = MotionReport()
    @Published private(set) var poseCheck = PoseCheckStatus()
    /// Set when recording reaches the requested duration.
    @Published private(set) var recordingComplete = false
    /// Why the session is not running, until it is started again.
    @Published private(set) var failure: CaptureFailure?

    /// Interruptions and failures, delivered on the main queue.  The survey
    /// has to *act* on these -- drop a recording, forget a world frame that no
    /// longer holds -- so they are an event rather than a published value.
    var onEvent: ((CaptureEvent) -> Void)?

    /// Owned by `frameQueue`.
    private let fitter = WaterPlaneFitter()
    private let residualChecker = PoseResidualChecker()
    private var queuePhase: CapturePhase = .idle
    private var plan: RecordingPlan?
    private var recorder: StiRecorder?
    /// The burst being collected for the automatic water outline, and who
    /// asked for it.
    private var waterProbe: WaterProbe?
    private var waterProbeDone: (([LumaFrame]) -> Void)?

    /// Counts frames, so the published state can be refreshed at a fraction of
    /// the frame rate.  Owned by `frameQueue`.
    private var uiTick = 0
    /// Refresh the UI one frame in this many: 20 Hz at 60 fps.  The eye cannot
    /// tell 20 from 60 in a progress bar or an overlay, and every publish
    /// re-renders the screen on the main thread while the measurement runs.
    private static let uiInterval = 3
    #if DEBUG
    private var parityChecked = false
    #endif

    private let frameQueue = DispatchQueue(label: "stiv.frames", qos: .userInitiated)

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = frameQueue
    }

    // MARK: - lifecycle

    /// Start world tracking, with depth when the device has it, from a fresh
    /// world frame.
    ///
    /// The video format is chosen for **frame rate**, not resolution.  `St` is
    /// set by the capture period, and a shorter `St` both lowers `tan phi` --
    /// toward the sweet spot of Fig. 7 for a fast river -- and gives the
    /// after-the-fact `frameStep` slider more room to work with.  Extra pixels
    /// only make `Sx` finer than the water's own texture, which buys nothing.
    func start() {
        // Running an unsupported configuration is an error ARKit reports
        // asynchronously, if at all; say so plainly and do not try.
        guard ARWorldTrackingConfiguration.isSupported else {
            publish { $0.failure = .unsupported }
            return
        }
        publish { $0.failure = nil }

        let configuration = ARWorldTrackingConfiguration()
        // Gravity-aligned is the default, and it is load-bearing here: it is
        // what makes the water surface a one-parameter plane.
        configuration.worldAlignment = .gravity
        configuration.planeDetection = []
        configuration.environmentTexturing = .none
        configuration.isLightEstimationEnabled = false

        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
        }
        if let fastest = ARWorldTrackingConfiguration.supportedVideoFormats
            .max(by: { $0.framesPerSecond < $1.framesPerSecond }) {
            configuration.videoFormat = fastest
        }

        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])

        // The published camera and tracking state describe the world frame
        // just reset, and a manual water level taken from them would sit in
        // the wrong one.  Forget them until the new session delivers a frame:
        // through the frame queue, behind any frame of the old session still
        // waiting there, so none of those can publish after this.
        frameQueue.async {
            self.publish {
                $0.camera = nil
                $0.trackingState = .notAvailable
                $0.depthAvailable = false
            }
        }
    }

    /// Stop the camera, LiDAR and tracking: nothing needs them while the
    /// results are on screen, and sixty frames a second of world tracking is
    /// most of what makes a phone hot on a riverbank.  `start` resumes, from a
    /// new world frame.
    func pause() {
        session.pause()
        frameQueue.async { self.abandonWork() }
        publish { $0.phase = .idle }
    }

    /// The capture rate the chosen video format advertises, for `St` and for the
    /// duration estimate on the settings screen.
    var nominalFrameRate: Double {
        guard let format = (session.configuration as? ARWorldTrackingConfiguration)?.videoFormat
        else { return 60 }
        return Double(format.framesPerSecond)
    }

    func setPhase(_ next: CapturePhase) {
        publish { $0.phase = next }
        frameQueue.async {
            self.queuePhase = next
            if next == .fittingPlane { self.fitter.reset() }
        }
        if next == .fittingPlane {
            publish {
                $0.planeProgress = 0
                $0.planeReady = false
                $0.planeGuidance = ""
            }
        }
    }

    /// Drop everything the frame queue was accumulating.  Frame queue only.
    private func abandonWork() {
        queuePhase = .idle
        recorder = nil
        plan = nil
        fitter.reset()
        residualChecker.disarm()
        finishWaterProbe()
    }

    // MARK: - water outline

    /// Collect a short burst of reduced frames for `detectWaterOutline`, and
    /// hand them to `completion` on the main queue: all of them, or whatever
    /// there was when the session stopped or a newer burst was asked for.
    func collectWaterFrames(plane: WaterPlane, completion: @escaping ([LumaFrame]) -> Void) {
        frameQueue.async {
            self.finishWaterProbe()
            self.waterProbe = WaterProbe(plane: plane)
            self.waterProbeDone = completion
        }
    }

    /// Hand over the burst in progress, if any.  Frame queue only.
    private func finishWaterProbe() {
        guard let done = waterProbeDone else { return }
        let frames = waterProbe?.frames ?? []
        waterProbe = nil
        waterProbeDone = nil
        DispatchQueue.main.async { done(frames) }
    }

    // MARK: - water plane

    /// Accept the current fit.
    ///
    /// `onWaterSurface` records whether the samples came from the water itself
    /// (close range) or from the waterline on the bank, which is the usual case
    /// beyond about five metres.  The report should not conflate the two: one is
    /// a direct measurement of the surface, the other infers it from a contour.
    func acceptPlane(onWaterSurface: Bool, completion: @escaping (WaterPlane?) -> Void) {
        frameQueue.async {
            let plane = self.fitter.fit(measuredOnWaterSurface: onWaterSurface)
            self.queuePhase = plane == nil ? .fittingPlane : .picking
            DispatchQueue.main.async {
                if plane != nil { self.phase = .picking }
                completion(plane)
            }
        }
    }

    /// The fallback for devices with no depth sensor: the user's own estimate of
    /// how far the camera is above the water.  Same geometry, worse provenance.
    func acceptManualPlane(cameraHeightAboveWater: Double) -> WaterPlane? {
        // A limited pose's height is not to be trusted with every metre that
        // follows; the button says the same, but this is the decision.
        guard let camera, trackingState.isNormal else { return nil }
        let plane = WaterPlaneFitter.manual(
            cameraHeightAboveWater: cameraHeightAboveWater,
            cameraWorldY: Double(camera.transform.columns.3.y))
        phase = .picking
        frameQueue.async { self.queuePhase = .picking }
        return plane
    }

    // MARK: - recording

    /// Arm the recorder and the pose cross-check, then start reading rows.
    ///
    /// The recorder reserves exactly the plan's rows: `SurveyModel` has already
    /// fitted the plan's duration and sample counts to `RecordingBudget`, so the
    /// reservation is the budget at most, not whatever the settings asked for.
    func startRecording(plan: RecordingPlan, waterPolygon: [PixelPoint]?) {
        // Until the checker is armed, it has checked nothing: say so, rather
        // than "cross-checking against 0 patches".
        var unarmed = MotionReport()
        unarmed.enabled = false
        publish {
            $0.recordedRows = 0
            $0.recordedSeconds = 0
            $0.recordingComplete = false
            $0.motionReport = unarmed
            $0.poseCheck = PoseCheckStatus()
            $0.phase = .recording
        }

        frameQueue.async {
            // Arm the residual check against the frame that starts the
            // recording, so its patches and the STI share one reference.
            if let frame = self.session.currentFrame {
                let projection = frame.camera.projection(onto: plan.plane)
                self.residualChecker.arm(
                    frame: frame,
                    avoid: plan.searchLineSegments(projection: projection),
                    waterPolygon: waterPolygon,
                    sectionCentre: projection.worldToImage(plan.sectionCentre))
            } else {
                self.residualChecker.disarm()
            }
            self.plan = plan
            self.recorder = StiRecorder(sampleCounts: plan.sampleCounts,
                                        sxPerStation: plan.sxPerStation,
                                        expectedRows: plan.expectedRows)
            self.queuePhase = .recording
        }
    }

    /// Stop recording and hand back everything the analysis needs.
    ///
    /// Hands back whatever the frame queue holds when the request reaches it.
    /// Requests are served in order, so a recording started after this call
    /// can never be the one handed back.
    func finishRecording(completion: @escaping (CompletedRecording?) -> Void) {
        frameQueue.async {
            self.queuePhase = .idle
            guard let recorder = self.recorder, let plan = self.plan else {
                DispatchQueue.main.async {
                    self.phase = .idle
                    completion(nil)
                }
                return
            }
            let report = self.residualChecker.report
            let recording = recorder.snapshot
            // Both, together: the plan carries one world point per STI column
            // per station, and a plan without a recorder is a state nothing
            // else here knows how to interpret.
            self.recorder = nil
            self.plan = nil
            DispatchQueue.main.async {
                self.phase = .idle
                completion(CompletedRecording(recording: recording, plan: plan, motion: report))
            }
        }
    }

    // MARK: - per-frame work

    /// Read one STI row from this frame.  See `RecordingPlan.sampleRow`.
    private func sampleRow(frame: ARFrame, plan: RecordingPlan,
                           correction: PoseCorrection) -> [Float]? {
        let intrinsics = frame.camera.stivIntrinsics
        let pose = frame.camera.stivPose
        return withLumaPlane(of: frame.capturedImage) { luma in
            plan.sampleRow(luma: luma, intrinsics: intrinsics, pose: pose,
                           correction: correction)
        }
    }

    /// Append one row to the recording in progress, and end it at the
    /// requested duration.  Frame queue only.
    ///
    /// The end is decided *here*, on the queue that appends, rather than by the
    /// main thread noticing `recordingComplete`: a busy main thread would
    /// otherwise let the recording run past its duration, and past the memory
    /// budget the duration was chosen to fit.
    func recordRow(_ row: [Float], timestamp: Double, refreshUI: Bool = true) {
        guard queuePhase == .recording, let plan, let recorder else { return }
        recorder.append(row: row, timestamp: timestamp)
        let seconds = recorder.recordedSeconds
        let done = seconds >= plan.survey.capture.durationSeconds
        if done { queuePhase = .idle }
        guard refreshUI || done else { return }

        let rows = recorder.rowCount
        let report = residualChecker.report
        let status = residualChecker.status
        publish {
            $0.recordedRows = rows
            $0.recordedSeconds = seconds
            $0.motionReport = report
            $0.poseCheck = status
            if done { $0.recordingComplete = true }
        }
    }

    /// Run `body` on the frame queue and wait for it.  For the tests, which
    /// have no frames to deliver and so drive `recordRow` directly.
    func syncOnFrameQueue<T>(_ body: () -> T) -> T {
        frameQueue.sync(execute: body)
    }

    /// Apply a batch of `@Published` mutations on the main queue.
    private func publish(_ body: @escaping (ARCaptureSession) -> Void) {
        if Thread.isMainThread {
            body(self)
        } else {
            DispatchQueue.main.async { body(self) }
        }
    }
}

// MARK: - ARSessionDelegate

extension ARCaptureSession: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Runs on `frameQueue`, which owns everything mutated below.
        let camera = frame.camera
        let trackingState = camera.trackingState
        let hasDepth = frame.sceneDepth != nil || frame.smoothedSceneDepth != nil
        uiTick += 1
        let refreshUI = uiTick % Self.uiInterval == 0

        #if DEBUG
        // Once per session: check that the hand-written projection agrees with
        // ARKit's own.  Every metre this app reports depends on it.
        if !parityChecked, trackingState.isNormal {
            parityChecked = true
            verifyProjectionParity(frame: frame)
        }
        #endif

        if let probe = waterProbe {
            let pose = camera.stivPose
            let intrinsics = camera.stivIntrinsics
            let finished = withLumaPlane(of: frame.capturedImage) { luma in
                probe.offer(luma: luma, intrinsics: intrinsics, pose: pose,
                            timestamp: frame.timestamp, trackingNormal: trackingState.isNormal)
            } ?? false
            if finished { finishWaterProbe() }
        }

        switch queuePhase {
        case .fittingPlane:
            // Only frames whose pose ARKit vouches for: a limited or
            // relocalising pose puts every return at the wrong height, and the
            // fit would keep them.  The tracking banner says why it paused.
            if trackingState.isNormal { fitter.add(frame: frame) }
            if refreshUI {
                let progress = fitter.progress
                let ready = fitter.isReady
                let guidance = trackingState.isNormal ? fitter.guidance()
                    : "Waiting for tracking: the water level is measured only while "
                    + "ARKit trusts its pose."
                publish {
                    $0.planeProgress = progress
                    $0.planeReady = ready
                    $0.planeGuidance = guidance
                }
            }

        case .recording:
            guard let plan else { break }
            let projection = frame.camera.projection(onto: plan.plane)
            let correction = residualChecker.check(
                frame: frame,
                sectionCentreNow: projection.worldToImage(plan.sectionCentre))
            if let row = sampleRow(frame: frame, plan: plan, correction: correction) {
                recordRow(row, timestamp: frame.timestamp, refreshUI: refreshUI)
            }

        case .idle, .picking:
            break
        }

        // Republishing the camera sixty times a second would invalidate the
        // whole overlay sixty times a second for a pose the eye cannot tell
        // apart.  Twenty is plenty for drawing a cross-section, and it leaves
        // the CPU to the measurement.
        if refreshUI {
            publish {
                $0.camera = camera
                $0.trackingState = trackingState
                $0.depthAvailable = hasDepth
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        let failure = CaptureFailure(error)
        frameQueue.async {
            let wasRecording = self.recorder != nil
            self.abandonWork()
            DispatchQueue.main.async {
                self.phase = .idle
                self.recordingComplete = false
                self.failure = failure
                self.onEvent?(.failed(wasRecording: wasRecording))
            }
        }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        // An interruption breaks the world frame, and with it the water plane's
        // height and every metre derived from it.  Recording through one would
        // produce numbers that look fine and are not, so the recording is
        // dropped -- and the survey is told, so it can say so and send the user
        // back to measure again rather than carrying on in a stale frame.
        frameQueue.async {
            let wasRecording = self.recorder != nil
            self.abandonWork()
            DispatchQueue.main.async {
                self.phase = .idle
                self.recordingComplete = false
                self.onEvent?(.interrupted(wasRecording: wasRecording))
            }
        }
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        // The same route as the interruption itself -- through the frame queue,
        // then the main queue -- so the end can never overtake the start.  A
        // busy frame queue would otherwise hold back `interrupted` while
        // `ended` went straight to the main queue, and the survey, told the
        // interruption was over before it began, would wait for an end that
        // had already come.
        frameQueue.async {
            DispatchQueue.main.async { self.onEvent?(.interruptionEnded) }
        }
    }
}

extension ARCamera.TrackingState {
    /// What to tell the user, phrased as what they should do about it.
    var advice: String? {
        switch self {
        case .normal:
            return nil
        case .notAvailable:
            return "Tracking not available yet -- move the phone slowly to let ARKit start."
        case .limited(.initializing):
            return "Starting up: move the phone slowly for a few seconds."
        case .limited(.excessiveMotion):
            return "Too much motion. Slow down, or brace the phone against something."
        case .limited(.insufficientFeatures):
            return "Not enough visual detail. Include some bank or structure in the frame."
        case .limited(.relocalizing):
            return "Relocalising after an interruption. Point at the scene you started from."
        case .limited:
            return "Tracking is limited; scale and motion compensation may be degraded."
        }
    }

    var isNormal: Bool {
        if case .normal = self { return true }
        return false
    }
}