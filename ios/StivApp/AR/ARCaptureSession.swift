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
    /// Intrinsics of the reference frame, kept for the desktop export bridge.
    let referenceIntrinsics: PinholeIntrinsics
    /// The section's midpoint, for the frame-shift diagnostic.
    let sectionCentre: PlanePoint

    init(survey: Survey, plane: WaterPlane, layout: SectionLayout,
         referenceIntrinsics: PinholeIntrinsics) {
        self.survey = survey
        self.plane = plane
        self.layout = layout
        self.referenceIntrinsics = referenceIntrinsics
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
}

/// A finished recording, ready for `StivKit.analyse`.
struct CompletedRecording {
    let recorder: StiRecorder
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
    @Published private(set) var planeSampleCount = 0
    @Published private(set) var planeGuidance = ""
    @Published private(set) var fittedPlane: WaterPlane?
    @Published private(set) var recordedRows = 0
    @Published private(set) var recordedSeconds: Double = 0
    @Published private(set) var motionReport = MotionReport()
    /// Set when recording reaches the requested duration.
    @Published private(set) var recordingComplete = false

    /// Owned by `frameQueue`.
    private let fitter = WaterPlaneFitter()
    private let residualChecker = PoseResidualChecker()
    private var queuePhase: CapturePhase = .idle
    private var plan: RecordingPlan?
    private var recorder: StiRecorder?

    /// Throttles the pose published for overlay drawing.  Owned by `frameQueue`.
    private var uiTick = 0
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

    /// Start world tracking, with depth when the device has it.
    ///
    /// The video format is chosen for **frame rate**, not resolution.  `St` is
    /// set by the capture period, and a shorter `St` both lowers `tan phi` --
    /// toward the sweet spot of Fig. 7 for a fast river -- and gives the
    /// after-the-fact `frameStep` slider more room to work with.  Extra pixels
    /// only make `Sx` finer than the water's own texture, which buys nothing.
    func start() {
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
    }

    func pause() { session.pause() }

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
                $0.planeSampleCount = 0
                $0.fittedPlane = nil
                $0.planeGuidance = ""
            }
        }
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
                if let plane {
                    self.fittedPlane = plane
                    self.phase = .picking
                }
                completion(plane)
            }
        }
    }

    /// The fallback for devices with no depth sensor: the user's own estimate of
    /// how far the camera is above the water.  Same geometry, worse provenance.
    func acceptManualPlane(cameraHeightAboveWater: Double) -> WaterPlane? {
        guard let camera else { return nil }
        let plane = WaterPlaneFitter.manual(
            cameraHeightAboveWater: cameraHeightAboveWater,
            cameraWorldY: Double(camera.transform.columns.3.y))
        fittedPlane = plane
        phase = .picking
        frameQueue.async { self.queuePhase = .picking }
        return plane
    }

    // MARK: - recording

    /// Arm the recorder and the pose cross-check, then start reading rows.
    func startRecording(plan: RecordingPlan, waterPolygon: [PixelPoint]?) {
        let expectedRows = Int(plan.survey.capture.durationSeconds
                               * plan.survey.capture.framesPerSecond) + 8

        publish {
            $0.recordedRows = 0
            $0.recordedSeconds = 0
            $0.recordingComplete = false
            $0.phase = .recording
        }

        frameQueue.async {
            // Arm the residual check against the frame that starts the
            // recording, so its patches and the STI share one reference.
            if let frame = self.session.currentFrame {
                let projection = frame.camera.projection(onto: plan.plane)
                let avoid = plan.layout.searchLines.compactMap {
                    projection.worldToImage($0.centre)
                }
                self.residualChecker.arm(
                    frame: frame,
                    avoid: avoid,
                    waterPolygon: waterPolygon,
                    sectionCentre: projection.worldToImage(plan.sectionCentre))
            }
            self.plan = plan
            self.recorder = StiRecorder(sampleCounts: plan.sampleCounts,
                                        sxPerStation: plan.sxPerStation,
                                        expectedRows: expectedRows)
            self.queuePhase = .recording
        }
    }

    /// Stop recording and hand back everything the analysis needs.
    func finishRecording(completion: @escaping (CompletedRecording?) -> Void) {
        publish { $0.phase = .idle }
        frameQueue.async {
            self.queuePhase = .idle
            guard let recorder = self.recorder, let plan = self.plan else {
                DispatchQueue.main.async { completion(nil) }
                return
            }
            let report = self.residualChecker.report
            self.recorder = nil
            DispatchQueue.main.async {
                completion(CompletedRecording(recorder: recorder, plan: plan, motion: report))
            }
        }
    }

    // MARK: - per-frame work

    /// Read one STI row: project every world sample point through *this* frame's
    /// pose and sample the luma plane there.
    ///
    /// This is the whole of the camera-motion compensation.  Because the points
    /// are anchored in the world rather than on the sensor, a hand that drifts
    /// ten pixels between frames changes *where in the image* the samples are
    /// taken, not *which patch of river* they describe.
    private func sampleRow(frame: ARFrame, plan: RecordingPlan,
                           correction: ResidualSimilarity) -> [Float]? {
        let intrinsics = frame.camera.stivIntrinsics
        let pose = frame.camera.stivPose
        let isIdentity = correction.scaleCos == 1 && correction.scaleSin == 0
            && correction.tx == 0 && correction.ty == 0

        return withLumaPlane(of: frame.capturedImage) { luma in
            var row = [Float](repeating: .nan, count: plan.worldPoints.count)
            for i in plan.worldPoints.indices {
                let camera = pose.toCamera(plan.worldPoints[i])
                let depth = -camera.z
                guard depth > 1e-6 else { continue }   // behind the camera
                var px = PixelPoint(u: intrinsics.fx * (camera.x / depth) + intrinsics.cx,
                                    v: intrinsics.fy * (-camera.y / depth) + intrinsics.cy)
                if !isIdentity { px = correction.apply(px) }
                if let value = luma.sample(u: px.u, v: px.v) { row[i] = value }
                // Otherwise the point left the frame, and NaN is the honest
                // answer: `SpaceTimeImage.outOfFrameFraction` counts it and
                // `preprocess` fills it from the column's time-mean.
            }
            return row
        }
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

        #if DEBUG
        // Once per session: check that the hand-written projection agrees with
        // ARKit's own.  Every metre this app reports depends on it.
        if !parityChecked, trackingState.isNormal {
            parityChecked = true
            verifyProjectionParity(frame: frame)
        }
        #endif

        switch queuePhase {
        case .fittingPlane:
            fitter.add(frame: frame)
            let count = fitter.sampleCount
            let guidance = fitter.guidance()
            publish {
                $0.planeSampleCount = count
                $0.planeGuidance = guidance
            }

        case .recording:
            guard let plan, let recorder else { break }
            let projection = frame.camera.projection(onto: plan.plane)
            let correction = residualChecker.check(
                frame: frame,
                sectionCentreNow: projection.worldToImage(plan.sectionCentre))
            if let row = sampleRow(frame: frame, plan: plan, correction: correction) {
                recorder.append(row: row, timestamp: frame.timestamp)
                let rows = recorder.rowCount
                let seconds = recorder.recordedSeconds
                let report = residualChecker.report
                let done = seconds >= plan.survey.capture.durationSeconds
                publish {
                    $0.recordedRows = rows
                    $0.recordedSeconds = seconds
                    $0.motionReport = report
                    if done { $0.recordingComplete = true }
                }
            }

        case .idle, .picking:
            break
        }

        // Republishing the camera sixty times a second would invalidate the
        // whole overlay sixty times a second for a pose the eye cannot tell
        // apart.  Twenty is plenty for drawing a cross-section, and it leaves
        // the CPU to the measurement.
        uiTick += 1
        if uiTick % 3 == 0 {
            publish {
                $0.camera = camera
                $0.trackingState = trackingState
                $0.depthAvailable = hasDepth
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        publish { $0.phase = .idle }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        // An interruption breaks the world frame, and with it the water plane's
        // height and every metre derived from it.  Recording through one would
        // produce numbers that look fine and are not.
        frameQueue.async {
            if self.queuePhase == .recording { self.queuePhase = .idle }
        }
        publish { if $0.phase == .recording { $0.phase = .idle } }
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
