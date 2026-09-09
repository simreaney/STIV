//  Survey.swift
//  What a measurement is: everything the analysis needs, and nothing about how
//  it was captured.
//
//  This follows `stiv/config.py` closely: the same sections, the same names and
//  the same defaults wherever the two describe the same quantity, so a parameter
//  tuned on the desktop means what it says here.  Where a value could be carried
//  between the two, it uses the Python's spelling rather than whichever reads
//  better in Swift -- `flow.direction` is `angle`, not `bearing`.
//
//  It is not a field-for-field mirror, and the divergences are not arbitrary:
//
//  * `rectify`, `stabilise` and `output` are gone.  ARKit supplies the
//    rectification and the camera pose, so there is nothing left to configure.
//  * Picked geometry is in ARKit world metres, not image pixels: on a phone the
//    world frame exists from the first frame and pixel coordinates are not
//    stable across frames.  So `section.start` is a `PlanePoint` where the
//    Python has `section.start_px` -- genuinely different quantities, sharing
//    no spelling on purpose.
//  * `capture` replaces `video`: there is no file, and `St` comes from measured
//    frame timestamps rather than a container's declared frame rate.
//
//  `Export.surveyYAML` is what speaks the Python's schema exactly, converting
//  the world geometry back into pixels and ground control points.

import Foundation

// MARK: - capture

public struct CaptureSettings: Codable, Sendable, Equatable {
    /// How long to record, seconds.
    public var durationSeconds: Double = 30.0

    /// Stride through the recorded rows.  This is the `video.frame_step` of the
    /// Python and the single most important tuning knob in STIV: it sets the
    /// time scale `St = frameStep / fps` of the space-time image and therefore
    /// where the measured velocities land on the `tan phi` axis, which section
    /// 2.3 of the paper shows is accurate between 0.1 and 7.
    ///
    /// Unlike the desktop pipeline, the app records at the full frame rate and
    /// applies the step at *analysis* time, so this can be changed on the
    /// results screen without re-recording.
    public var frameStep: Int = 2

    /// Nominal capture rate, used to derive `St`.  The recorder also stores
    /// real per-row timestamps and resamples onto this grid, so a dropped frame
    /// does not tilt the streaks.
    public var framesPerSecond: Double = 60.0

    /// Per-image-row readout delay of the rolling shutter, seconds.
    ///
    /// In an oblique view a search line spans image rows, and each row is read
    /// at a slightly different time, which tilts the streaks and biases `phi`.
    /// iOS exposes no public readout-time API, so this is off by default and
    /// only useful once calibrated for a specific device and video format.
    public var rollingShutterRowSeconds: Double = 0.0

    public var stSecondsPerRow: Double { Double(frameStep) / framesPerSecond }

    public init() {}
}

// MARK: - geometry

/// The measurement cross-section, in ARKit world metres on the water plane.
///
/// Stations are spaced evenly in world space, so the oblique view a handheld
/// phone produces still yields evenly spaced verticals.
public struct SectionSettings: Codable, Sendable, Equatable {
    /// Near-bank end, world `(x, z)` on the water plane.
    public var start: PlanePoint = .init(x: 0, z: 0)
    /// Far-bank end.
    public var end: PlanePoint = .init(x: 10, z: 0)
    public var stationCount: Int = 30
    /// Skip this fraction of the section at each end (banks are usually noisy).
    public var edgeTrim: Double = 0.0
    public var name: String = "section"

    public init() {}
}

/// A point on the horizontal water plane: ARKit world X and Z, in metres.
/// (`y` is the plane's own height and is carried by `WaterPlane`.)
public struct PlanePoint: Codable, Sendable, Equatable, Hashable {
    public var x: Double
    public var z: Double
    public init(x: Double, z: Double) { self.x = x; self.z = z }
}

/// Orientation of the streamwise search lines.
///
/// STIV is a 1-D method: it measures the velocity component along the search
/// line, so the line must be parallel to the main flow (paper, section 2.1).
public struct FlowSettings: Codable, Sendable, Equatable {
    public enum Direction: String, Codable, Sendable {
        /// Perpendicular to the cross-section, in world space.
        case auto
        /// A fixed bearing, degrees counter-clockwise from world +X.
        ///
        /// Spelled `angle` because that is what `stiv/config.py` calls it, and
        /// a value that can cross between the two should not change its name on
        /// the way.
        case angle
        /// Parallel to `from` -> `to`, which point downstream.
        case points
    }

    public var direction: Direction = .auto
    /// `flow.angle_deg` in the Python.
    public var angleDegrees: Double?
    public var from: PlanePoint?
    public var to: PlanePoint?
    /// A point downstream of the section; disambiguates the sign of the
    /// `auto` normal.
    public var downstream: PlanePoint?
    /// Flip the assumed downstream direction.
    public var reverse: Bool = false
    /// If the median velocity comes out negative, mirror the Eq. (14) search
    /// range and re-estimate rather than reporting negative flow.
    public var autoFlip: Bool = true

    public init() {}
}

// MARK: - the analysis itself

/// Space-time image construction and the orientation analysis on it.
public struct StiSettings: Codable, Sendable, Equatable {
    // --- search line ---------------------------------------------------- //

    /// Streamwise length of each search line, metres.  Long enough to hold
    /// several surface features, short enough that the velocity is uniform
    /// along it.  A few metres suits most rivers.
    public var searchLineLengthM: Double = 4.0
    /// Samples along the search line.  `nil` picks roughly one sample per image
    /// pixel, which is the natural resolution limit.
    public var sampleCount: Int?

    // --- STI preprocessing ---------------------------------------------- //

    /// Subtract the time-mean of each spatial column.  Removes static texture
    /// (rocks, glare, bank) that would otherwise draw a vertical -- i.e. zero
    /// velocity -- stripe through the STI.  Keep this on.
    public var removeStaticBackground: Bool = true
    /// Normalise each STI to zero mean / unit variance.
    public var normalise: Bool = true
    /// Gaussian smoothing sigma (pixels) applied to the STI.  Section 2.3 shows
    /// this materially improves accuracy on smooth patterns.
    public var smoothSigma: Double = 1.5

    // --- structure tensor ----------------------------------------------- //

    public var gradientOrder: GradientOrder = .four
    /// Side length of the local integration area `A` of Eqs. (7)-(9).
    public var windowPx: Int = 31
    public var windowType: WindowType = .box
    /// Overlap between neighbouring windows (the paper uses 0.5).
    public var windowOverlap: Double = 0.5

    // --- angle averaging (Eq. 14) --------------------------------------- //

    /// Velocity search range, m/s.  Sets `phiMin` and `phiMax`, which the paper
    /// uses to discard vectors corrupted by obstacles (Fig. 12).  A small
    /// negative bound keeps a genuinely still reach from being biased upward.
    public var velocityRangeMs: ClosedRange<Double> = -0.5 ... 8.0
    /// Coherency thresholds `C` of Eq. (14); the cumulative histograms of
    /// orientation angle for `C_i >= C` are summed (Figs. 10 and 11).
    public var coherencyLevels: [Double] = [0.0, 0.2, 0.4, 0.6, 0.8]
    public var angleBins: Int = 180
    /// Discard windows below this coherency before histogramming at all.
    public var minCoherency: Double = 0.0
    public var estimator: AngleEstimator = .eq14

    public init() {}

    // ClosedRange is Codable, but a YAML round-trip reads better as a pair.
    private enum CodingKeys: String, CodingKey {
        case searchLineLengthM, sampleCount, removeStaticBackground, normalise
        case smoothSigma, gradientOrder, windowPx, windowType, windowOverlap
        case velocityMinMs, velocityMaxMs, coherencyLevels, angleBins
        case minCoherency, estimator
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        var s = StiSettings()
        s.searchLineLengthM = try c.decodeIfPresent(Double.self, forKey: .searchLineLengthM) ?? s.searchLineLengthM
        s.sampleCount = try c.decodeIfPresent(Int.self, forKey: .sampleCount)
        s.removeStaticBackground = try c.decodeIfPresent(Bool.self, forKey: .removeStaticBackground) ?? s.removeStaticBackground
        s.normalise = try c.decodeIfPresent(Bool.self, forKey: .normalise) ?? s.normalise
        s.smoothSigma = try c.decodeIfPresent(Double.self, forKey: .smoothSigma) ?? s.smoothSigma
        s.gradientOrder = try c.decodeIfPresent(GradientOrder.self, forKey: .gradientOrder) ?? s.gradientOrder
        s.windowPx = try c.decodeIfPresent(Int.self, forKey: .windowPx) ?? s.windowPx
        s.windowType = try c.decodeIfPresent(WindowType.self, forKey: .windowType) ?? s.windowType
        s.windowOverlap = try c.decodeIfPresent(Double.self, forKey: .windowOverlap) ?? s.windowOverlap
        let lo = try c.decodeIfPresent(Double.self, forKey: .velocityMinMs) ?? s.velocityRangeMs.lowerBound
        let hi = try c.decodeIfPresent(Double.self, forKey: .velocityMaxMs) ?? s.velocityRangeMs.upperBound
        s.velocityRangeMs = min(lo, hi) ... max(lo, hi)
        s.coherencyLevels = try c.decodeIfPresent([Double].self, forKey: .coherencyLevels) ?? s.coherencyLevels
        s.angleBins = try c.decodeIfPresent(Int.self, forKey: .angleBins) ?? s.angleBins
        s.minCoherency = try c.decodeIfPresent(Double.self, forKey: .minCoherency) ?? s.minCoherency
        s.estimator = try c.decodeIfPresent(AngleEstimator.self, forKey: .estimator) ?? s.estimator
        self = s
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(searchLineLengthM, forKey: .searchLineLengthM)
        try c.encodeIfPresent(sampleCount, forKey: .sampleCount)
        try c.encode(removeStaticBackground, forKey: .removeStaticBackground)
        try c.encode(normalise, forKey: .normalise)
        try c.encode(smoothSigma, forKey: .smoothSigma)
        try c.encode(gradientOrder, forKey: .gradientOrder)
        try c.encode(windowPx, forKey: .windowPx)
        try c.encode(windowType, forKey: .windowType)
        try c.encode(windowOverlap, forKey: .windowOverlap)
        try c.encode(velocityRangeMs.lowerBound, forKey: .velocityMinMs)
        try c.encode(velocityRangeMs.upperBound, forKey: .velocityMaxMs)
        try c.encode(coherencyLevels, forKey: .coherencyLevels)
        try c.encode(angleBins, forKey: .angleBins)
        try c.encode(minCoherency, forKey: .minCoherency)
        try c.encode(estimator, forKey: .estimator)
    }
}

/// Thresholds used to flag, and optionally drop, unreliable stations.
public struct QualitySettings: Codable, Sendable, Equatable {
    /// Minimum fraction of windows falling inside `[phiMin, phiMax]`.
    public var minValidFraction: Double = 0.15
    /// Minimum mean coherency for a station.
    public var minMeanCoherency: Double = 0.10
    /// Maximum velocity uncertainty as a fraction of the station's velocity.
    /// This is usually the criterion that catches a search line straddling the
    /// bank: the angle histogram goes flat, so sigma blows up even though mean
    /// coherency may still scrape past its own threshold.  The comparison is
    /// made against the section's own velocity scale, so it transfers between a
    /// 0.3 m/s brook and a 4 m/s flood without retuning.
    public var maxRelativeUncertainty: Double = 0.5
    /// Flag stations outside the accuracy range demonstrated in the paper.
    public var tanPhiRange: ClosedRange<Double> = 0.05 ... 8.0
    /// Replace failed stations by an interpolation of their neighbours.
    public var interpolateFlagged: Bool = true
    /// Median-filter window (stations) applied across the profile; 0 disables.
    public var medianFilterStations: Int = 0
    /// Flag a station whose *scale* uncertainty exceeds this fraction.  New on
    /// the phone: far-field scale is extrapolated from the water plane, so it
    /// degrades with range in a way a drone's ground sample distance does not.
    public var maxRelativeScaleUncertainty: Double = 0.15

    public init() {}

    private enum CodingKeys: String, CodingKey {
        case minValidFraction, minMeanCoherency, maxRelativeUncertainty
        case tanPhiMin, tanPhiMax, interpolateFlagged, medianFilterStations
        case maxRelativeScaleUncertainty
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        var q = QualitySettings()
        q.minValidFraction = try c.decodeIfPresent(Double.self, forKey: .minValidFraction) ?? q.minValidFraction
        q.minMeanCoherency = try c.decodeIfPresent(Double.self, forKey: .minMeanCoherency) ?? q.minMeanCoherency
        q.maxRelativeUncertainty = try c.decodeIfPresent(Double.self, forKey: .maxRelativeUncertainty) ?? q.maxRelativeUncertainty
        let lo = try c.decodeIfPresent(Double.self, forKey: .tanPhiMin) ?? q.tanPhiRange.lowerBound
        let hi = try c.decodeIfPresent(Double.self, forKey: .tanPhiMax) ?? q.tanPhiRange.upperBound
        q.tanPhiRange = min(lo, hi) ... max(lo, hi)
        q.interpolateFlagged = try c.decodeIfPresent(Bool.self, forKey: .interpolateFlagged) ?? q.interpolateFlagged
        q.medianFilterStations = try c.decodeIfPresent(Int.self, forKey: .medianFilterStations) ?? q.medianFilterStations
        q.maxRelativeScaleUncertainty = try c.decodeIfPresent(Double.self, forKey: .maxRelativeScaleUncertainty) ?? q.maxRelativeScaleUncertainty
        self = q
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(minValidFraction, forKey: .minValidFraction)
        try c.encode(minMeanCoherency, forKey: .minMeanCoherency)
        try c.encode(maxRelativeUncertainty, forKey: .maxRelativeUncertainty)
        try c.encode(tanPhiRange.lowerBound, forKey: .tanPhiMin)
        try c.encode(tanPhiRange.upperBound, forKey: .tanPhiMax)
        try c.encode(interpolateFlagged, forKey: .interpolateFlagged)
        try c.encode(medianFilterStations, forKey: .medianFilterStations)
        try c.encode(maxRelativeScaleUncertainty, forKey: .maxRelativeScaleUncertainty)
    }
}

// MARK: - the whole survey

public struct Survey: Codable, Sendable, Equatable {
    public var capture = CaptureSettings()
    public var section = SectionSettings()
    public var flow = FlowSettings()
    public var sti = StiSettings()
    public var quality = QualitySettings()
    public var scale = ScaleSettings()

    public init() {}

    /// Sanity checks that would otherwise surface as a confusing failure deep
    /// in the analysis.  Mirrors `Config.validate`.
    public func validated() throws -> Survey {
        if section.stationCount < 1 {
            throw SurveyError.invalid("section.stationCount must be >= 1, got \(section.stationCount)")
        }
        if !(0..<0.5).contains(section.edgeTrim) {
            throw SurveyError.invalid("section.edgeTrim must be in [0, 0.5), got \(section.edgeTrim)")
        }
        if sti.windowPx < 5 || sti.windowPx % 2 == 0 {
            throw SurveyError.invalid("sti.windowPx must be odd and >= 5, got \(sti.windowPx)")
        }
        if !(0..<1).contains(sti.windowOverlap) {
            throw SurveyError.invalid("sti.windowOverlap must be in [0, 1), got \(sti.windowOverlap)")
        }
        if sti.searchLineLengthM <= 0 {
            throw SurveyError.invalid("sti.searchLineLengthM must be positive")
        }
        if let n = sti.sampleCount, n < 16 {
            throw SurveyError.invalid("sti.sampleCount must be >= 16, got \(n)")
        }
        if capture.frameStep < 1 {
            throw SurveyError.invalid("capture.frameStep must be >= 1, got \(capture.frameStep)")
        }
        if capture.framesPerSecond <= 0 {
            throw SurveyError.invalid("capture.framesPerSecond must be positive")
        }
        if sti.velocityRangeMs.lowerBound >= sti.velocityRangeMs.upperBound {
            throw SurveyError.invalid("sti.velocityRangeMs must be increasing")
        }
        if sti.coherencyLevels.isEmpty {
            throw SurveyError.invalid("sti.coherencyLevels must not be empty")
        }
        return self
    }
}

public enum SurveyError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String {
        switch self { case let .invalid(m): return m }
    }
}
