//  Analysis.swift
//  The STIV pipeline: recorded space-time images in, cross-section velocity
//  profile out.
//
//  A port of `stiv/analysis.py`.  Order of operations, following Fig. 5 of the
//  paper with the phone-specific additions:
//
//  1. the cross-section and its search lines are already laid out in world
//     metres (`Geometry.swift`) and recorded (`SpaceTimeImage.swift`);
//  2. condition each STI and run the structure-tensor orientation analysis over
//     a grid of local windows (`Orientation.swift`);
//  3. reduce each STI's windows to one angle by the coherency-weighted average
//     of Eq. (14), and convert to velocity by Eq. (16);
//  4. quality-control the profile, including the new scale term
//     (`Uncertainty.swift`), and report.
//
//  Everything here is pure: given the same recording and the same `Survey` it
//  returns the same answer, which is what lets the results screen re-run it
//  with a different `frameStep` while the user watches.

import Foundation

// MARK: - flags

/// Why a station should be treated with suspicion.
public enum StationFlag: String, Sendable, CaseIterable {
    // Failures: the station produced nothing trustworthy at all.  Only these
    // make a station eligible for replacement by interpolation.
    case noOrientation
    case stiTooSmall
    case lowValidFraction
    case lowCoherency
    case highUncertainty
    case outOfFrame

    // Advisories: the number is real but near the edge of the method's range.
    // A genuinely slow station by the bank is the common case, and
    // interpolating it away from its neighbours would be worse than reporting
    // it.
    case tanPhiLow
    case tanPhiHigh
    case scaleUncertain
    case interpolated

    public var isFailure: Bool {
        switch self {
        case .noOrientation, .stiTooSmall, .lowValidFraction,
             .lowCoherency, .highUncertainty, .outOfFrame:
            return true
        case .tanPhiLow, .tanPhiHigh, .scaleUncertain, .interpolated:
            return false
        }
    }

    /// The plain-English meaning, for the flag legend on the results screen.
    public var explanation: String {
        switch self {
        case .noOrientation:     return "no orientation could be measured at all"
        case .stiTooSmall:       return "too few frames or samples for the analysis window"
        case .lowValidFraction:  return "most windows fell outside the velocity range -- widen it"
        case .lowCoherency:      return "no streak-like structure -- check the surface has texture"
        case .highUncertainty:   return "the angle histogram is flat; usually a line straddling the bank"
        case .outOfFrame:        return "the search line left the image"
        case .tanPhiHigh:        return "features move far between frames -- raise the frame step"
        case .tanPhiLow:         return "features barely move between frames -- lower the frame step"
        case .scaleUncertain:    return "scale here is extrapolated far from the water-plane fit"
        case .interpolated:      return "this station failed and was filled from its neighbours"
        }
    }
}

// MARK: - results

/// Velocity and quality metrics at one vertical of the cross-section.
public struct StationResult: Sendable {
    public let index: Int
    public let stationM: Double
    public let world: PlanePoint
    /// Horizontal distance from the camera, metres.
    public let rangeM: Double

    public var phiRad: Double = .nan
    public var tanPhi: Double = .nan
    public var velocityMs: Double = .nan
    /// Velocity before quality control replaced or dropped it.
    public var velocityRawMs: Double = .nan
    public var coherency: Double = 0
    public var validFraction: Double = 0
    public var windowCount: Int = 0
    public var phiStdRad: Double = .nan
    /// 1-sigma velocity spread, m/s: the angle histogram propagated through
    /// Eq. (16), combined with the scale uncertainty of this station's range.
    public var velocityStdMs: Double = .nan
    /// The scale component on its own, as a fraction.
    public var relativeScaleSigma: Double = .nan
    public var sxMetresPerColumn: Double = .nan
    public var stSecondsPerRow: Double = .nan
    public var flags: [StationFlag] = []

    /// True when the station produced nothing usable.
    public var failed: Bool {
        flags.contains(where: \.isFailure) || !velocityMs.isFinite
    }

    /// True when the station produced a measurement with no caveats.
    public var ok: Bool { flags.isEmpty && velocityMs.isFinite }

    init(line: SearchLine) {
        self.index = line.index
        self.stationM = line.stationM
        self.world = line.centre
        self.rangeM = line.rangeM
        self.sxMetresPerColumn = line.sxMetresPerColumn
    }
}

/// Per-frame camera-motion diagnostics.
///
/// ARKit's 6-DoF pose does the actual compensation -- the search-line sample
/// points are fixed in world space and re-projected every frame -- so unlike
/// `stiv/stabilise.py` this report never feeds back into the STI.  It exists to
/// answer one question: *did the pose fully explain the image motion?*  If it
/// did, the residual shifts measured on static ground are zero.
///
/// The fields and warnings mirror `StabilisationReport` so that a phone run and
/// a desktop run read the same.
public struct MotionReport: Sendable {
    public var enabled: Bool = true
    public var frameCount: Int = 0
    public var failedCount: Int = 0
    /// Patches tracked on static ground.
    public var patchCount: Int = 0
    /// Per-frame RMS residual shift not explained by the AR pose, pixels.
    public var residualPx: [Double] = []
    /// Per-frame displacement of the section centre in the image, pixels.
    /// Large values are fine -- the pose removes them -- but they tell the user
    /// whether the search lines stayed over water.
    public var shiftPx: [Double] = []
    /// Fraction of patches agreeing with the fitted residual motion.
    public var inlierFraction: [Double] = []
    /// Whether world tracking was ever anything but `.normal`.
    public var trackingDegraded: Bool = false
    /// Worst pose residual actually corrected, pixels.
    public var maxCorrectionPx: Double = 0

    public init() {}

    public func summary() -> String {
        guard enabled else { return "motion: ARKit pose only, residual check disabled" }
        guard !residualPx.isEmpty else { return "motion: no frames processed" }
        let finite = residualPx.filter(\.isFinite).sorted()
        let median = finite.isEmpty ? Double.nan : finite[finite.count / 2]
        let maxShift = shiftPx.filter(\.isFinite).max() ?? 0
        return "motion: \(patchCount) static patches, "
             + "median pose residual \(decimals(median, 2)) px, "
             + "max frame shift \(decimals(maxShift, 1)) px, "
             + "\(failedCount)/\(frameCount) frames unchecked"
    }

    public func warnings() -> [String] {
        guard enabled else { return [] }
        var out: [String] = []
        if patchCount < 12 {
            out.append("""
                only \(patchCount) trackable patches were found on static ground, so the \
                ARKit pose could not be cross-checked; outline the water so the check has \
                bank to work with, or brace the phone
                """)
        }
        if frameCount > 0 && failedCount > Int(0.1 * Double(frameCount)) {
            out.append("""
                \(failedCount) of \(frameCount) frames could not be cross-checked; the \
                velocities rely on the ARKit pose alone
                """)
        }
        let medianInlier = inlierFraction.sorted().dropFirst(inlierFraction.count / 2).first
        if let m = medianInlier, m < 0.5 {
            out.append("""
                fewer than half the tracked patches agree with the fitted residual motion; \
                patches are probably landing on the water surface -- outline the water on \
                the cross-section screen
                """)
        }
        let finiteResiduals = residualPx.filter(\.isFinite)
        if let worst = finiteResiduals.max(), worst > 3.0 {
            out.append("the ARKit pose left up to \(decimals(worst, 1)) px of unexplained "
                     + "image motion; world tracking may have drifted, so treat the scale "
                     + "and the velocities as provisional")
        }
        if let worst = shiftPx.filter(\.isFinite).max(), worst > 100 {
            out.append("the camera moved up to \(decimals(worst, 0)) px during the "
                     + "recording; check that the search lines stayed over water throughout")
        }
        if trackingDegraded {
            out.append("""
                world tracking was not in its normal state for part of the recording, which \
                degrades the metric scale; the reported uncertainties account for this
                """)
        }
        return out
    }
}

/// Everything a run produced.
public struct StivResult: Sendable {
    public let survey: Survey
    public let stations: [StationResult]
    public let stis: [SpaceTimeImage]
    public let motion: MotionReport
    public let rowCount: Int
    public let durationSeconds: Double
    public let stSecondsPerRow: Double
    public let widthM: Double
    public let flowBearingDegrees: Double
    public let cameraHeightM: Double
    public let rectifierDescription: String
    public var warnings: [String]
    public var elapsedSeconds: Double = 0

    public var velocities: [Double] { stations.map(\.velocityMs) }
    public var stationsM: [Double] { stations.map(\.stationM) }

    /// A `frameStep` that would put the median `tan phi` near 1, where the
    /// method is most accurate -- what the results-screen slider snaps to.
    public var suggestedFrameStep: Int?

    public func summary() -> String {
        let finite = velocities.filter(\.isFinite).sorted()
        let clean = stations.filter(\.ok).count
        let failed = stations.filter(\.failed).count
        var lines = [
            "section        \(survey.section.name): \(decimals(widthM, 2)) m wide, "
                + "\(stations.count) stations",
            "scale          \(rectifierDescription)",
            "frames         \(rowCount) over \(decimals(durationSeconds, 2)) s "
                + "(St = \(decimals(stSecondsPerRow * 1000, 1)) ms/row, "
                + "step \(survey.capture.frameStep))",
            "flow direction \(decimals(flowBearingDegrees, 1)) deg in world frame",
            motion.summary(),
            "stations       \(clean)/\(stations.count) clean, \(failed) failed and interpolated",
        ]
        if !finite.isEmpty {
            let mean = finite.reduce(0, +) / Double(finite.count)
            lines.append("surface U      mean \(decimals(mean, 3))  "
                       + "median \(decimals(finite[finite.count / 2], 3))  "
                       + "max \(decimals(finite[finite.count - 1], 3)) m/s")
        }
        if !warnings.isEmpty {
            lines.append("")
            lines.append(contentsOf: warnings.map { "warning: \($0)" })
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - the pipeline

/// Execute a full STIV analysis over already-recorded space-time images.
public func analyse(
    survey: Survey,
    layout: SectionLayout,
    stis: [SpaceTimeImage],
    motion: MotionReport = MotionReport()
) -> StivResult {
    let clock = Date()
    var warnings: [String] = []

    if !layout.stationsOutsideFrame.isEmpty {
        let shown = layout.stationsOutsideFrame.prefix(5).map(String.init).joined(separator: ", ")
        let more = layout.stationsOutsideFrame.count > 5 ? "..." : ""
        warnings.append("""
            \(layout.stationsOutsideFrame.count) search line(s) extend outside the frame \
            (stations \(shown)\(more)); shorten the search line, step back, or move the section
            """)
    }

    var conditioned = stis
    var stations: [StationResult] = []
    var fields: [OrientationField?] = []

    for (i, line) in layout.searchLines.enumerated() {
        var station = StationResult(line: line)
        guard i < conditioned.count else {
            station.flags.append(.noOrientation)
            stations.append(station)
            fields.append(nil)
            continue
        }
        station.stSecondsPerRow = conditioned[i].stSecondsPerRow
        conditioned[i].data = preprocess(conditioned[i].data, settings: survey.sti)

        var field: OrientationField?
        var estimate: AngleEstimate?
        do {
            let f = try structureTensorField(
                sti: conditioned[i].data,
                windowPx: survey.sti.windowPx,
                windowType: survey.sti.windowType,
                overlap: survey.sti.windowOverlap,
                gradientOrder: survey.sti.gradientOrder)
            field = f
            estimate = try estimateAngle(survey: survey, sti: conditioned[i], field: f,
                                         vLow: survey.sti.velocityRangeMs.lowerBound,
                                         vHigh: survey.sti.velocityRangeMs.upperBound)
        } catch {
            station.flags.append(.stiTooSmall)
        }

        apply(estimate, to: &station, sti: conditioned[i], survey: survey,
              cameraHeightM: layout.cameraHeightM,
              trackingDegraded: motion.trackingDegraded)

        if conditioned[i].outOfFrameFraction > 0.02 {
            station.flags.append(.outOfFrame)
        }

        stations.append(station)
        fields.append(field)
    }

    directionCheck(survey: survey, stations: &stations, stis: conditioned, fields: fields,
                   cameraHeightM: layout.cameraHeightM, motion: motion, warnings: &warnings)
    qualityControl(survey: survey, stations: &stations, warnings: &warnings)
    scaleWarning(survey: survey, stations: stations, cameraHeightM: layout.cameraHeightM,
                 warnings: &warnings)
    warnings.append(contentsOf: motion.warnings())

    let rows = conditioned.first?.rows ?? 0
    let st = conditioned.first?.stSecondsPerRow ?? survey.capture.stSecondsPerRow

    var result = StivResult(
        survey: survey,
        stations: stations,
        stis: conditioned,
        motion: motion,
        rowCount: rows,
        durationSeconds: Double(rows) * st,
        stSecondsPerRow: st,
        widthM: layout.widthM,
        flowBearingDegrees: layout.flowBearingDegrees,
        cameraHeightM: layout.cameraHeightM,
        rectifierDescription: layout.rectifierDescription,
        warnings: warnings)

    let advice = rangeAdvice(survey: survey, stations: stations)
    result.suggestedFrameStep = advice.step
    if let message = advice.message { result.warnings.append(message) }
    result.elapsedSeconds = -clock.timeIntervalSinceNow
    return result
}

/// Run Eq. (14) over one orientation field for a given velocity range.
func estimateAngle(
    survey: Survey,
    sti: SpaceTimeImage,
    field: OrientationField,
    vLow: Double,
    vHigh: Double
) throws -> AngleEstimate {
    let range = angleRange(forVelocities: vLow, vHigh,
                           sxMetresPerColumn: sti.sxMetresPerColumn,
                           stSecondsPerRow: sti.stSecondsPerRow)
    return try meanOrientationAngle(
        field: field,
        phiMin: range.min,
        phiMax: range.max,
        coherencyLevels: survey.sti.coherencyLevels,
        binCount: survey.sti.angleBins,
        minCoherency: survey.sti.minCoherency,
        estimator: survey.sti.estimator)
}

/// Write an angle estimate onto a station, converting it to a velocity.
///
/// `negate` is used when the search range was mirrored to follow flow running
/// against the assumed downstream direction: reversing the search line negates
/// `phi`, and Eq. (16) is odd in `phi`, so the velocity negates with it.
func apply(
    _ estimate: AngleEstimate?,
    to station: inout StationResult,
    sti: SpaceTimeImage,
    survey: Survey,
    cameraHeightM: Double,
    trackingDegraded: Bool,
    negate: Bool = false
) {
    station.sxMetresPerColumn = sti.sxMetresPerColumn
    station.stSecondsPerRow = sti.stSecondsPerRow
    station.relativeScaleSigma = relativeScaleUncertainty(
        rangeM: station.rangeM,
        cameraHeightM: cameraHeightM,
        settings: survey.scale,
        trackingDegraded: trackingDegraded)

    guard let estimate, estimate.phi.isFinite else {
        station.phiRad = .nan
        station.tanPhi = .nan
        station.velocityMs = .nan
        station.velocityRawMs = .nan
        station.velocityStdMs = .nan
        station.phiStdRad = .nan
        station.coherency = 0
        station.validFraction = estimate?.validFraction ?? 0
        station.windowCount = 0
        if !station.flags.contains(.stiTooSmall) && !station.flags.contains(.noOrientation) {
            station.flags.append(.noOrientation)
        }
        return
    }

    let phi = (negate ? -1.0 : 1.0) * estimate.phi
    station.phiRad = phi
    station.tanPhi = tan(phi)
    station.velocityMs = velocity(fromAngle: phi,
                                  sxMetresPerColumn: sti.sxMetresPerColumn,
                                  stSecondsPerRow: sti.stSecondsPerRow)
    station.velocityRawMs = station.velocityMs
    station.phiStdRad = estimate.phiStd
    station.coherency = estimate.meanCoherency
    station.validFraction = estimate.validFraction
    station.windowCount = estimate.usedCount

    // Propagate the angle spread through Eq. (16): dU/dphi = k sec^2(phi).
    let k = sti.sxMetresPerColumn / sti.stSecondsPerRow
    let angleTerm = abs(k / max(cos(phi) * cos(phi), 1e-6)) * estimate.phiStd
    // Eq. (16) is linear in Sx, so a relative scale error is a relative
    // velocity error.  The two are independent.
    let scaleTerm = station.relativeScaleSigma.isFinite
        ? abs(station.velocityMs) * station.relativeScaleSigma
        : Double.infinity
    station.velocityStdMs = (angleTerm * angleTerm + scaleTerm * scaleTerm).squareRoot()
}

// MARK: - quality control

func qualityControl(survey: Survey, stations: inout [StationResult], warnings: inout [String]) {
    let q = survey.quality

    // The uncertainty test is relative to the section's own velocity scale, so
    // that a genuinely slow but well-resolved station near the bank is not
    // condemned merely for having a small denominator.
    let finite = stations.map { abs($0.velocityMs) }.filter(\.isFinite).sorted()
    let velocityScale = finite.isEmpty ? 0 : 0.25 * finite[finite.count / 2]

    for i in stations.indices {
        var s = stations[i]
        if s.validFraction.isFinite && s.validFraction < q.minValidFraction {
            s.flags.append(.lowValidFraction)
        }
        if s.coherency.isFinite && s.coherency < q.minMeanCoherency {
            s.flags.append(.lowCoherency)
        }
        if s.velocityStdMs.isFinite && s.velocityMs.isFinite {
            let reference = max(abs(s.velocityMs), velocityScale)
            if reference > 0 && s.velocityStdMs > q.maxRelativeUncertainty * reference {
                s.flags.append(.highUncertainty)
            }
        }
        if s.tanPhi.isFinite {
            let a = abs(s.tanPhi)
            if a > q.tanPhiRange.upperBound {
                // Steep streaks: too little spatial displacement per frame is
                // resolvable.  Section 2.3 tested only up to tan phi = 7.
                s.flags.append(.tanPhiHigh)
            } else if a < q.tanPhiRange.lowerBound {
                s.flags.append(.tanPhiLow)
            }
        }
        if !s.relativeScaleSigma.isFinite || s.relativeScaleSigma > q.maxRelativeScaleUncertainty {
            s.flags.append(.scaleUncertain)
        }
        stations[i] = s
    }

    if q.medianFilterStations > 1 {
        medianFilter(&stations, window: q.medianFilterStations)
    }
    if q.interpolateFlagged {
        interpolateFailed(&stations)
    }

    let bad = stations.filter(\.failed).count
    if Double(bad) > 0.3 * Double(max(1, stations.count)) {
        warnings.append("""
            \(bad) of \(stations.count) stations failed quality control; check that the water \
            surface has visible texture, that the velocity range brackets the true velocity, \
            and that the search lines are parallel to the flow
            """)
    }
}

func medianFilter(_ stations: inout [StationResult], window k: Int) {
    let v = stations.map(\.velocityMs)
    let half = k / 2
    for i in stations.indices {
        let lo = max(0, i - half)
        let hi = min(v.count, i + half + 1)
        let w = v[lo..<hi].filter(\.isFinite).sorted()
        if !w.isEmpty { stations[i].velocityMs = w[w.count / 2] }
    }
}

/// Replace *failed* stations with an interpolation of their neighbours.
///
/// Advisory flags are deliberately left alone: a station reading a genuinely
/// slow patch near the bank is measuring correctly, and pulling it towards its
/// faster neighbours would introduce error rather than remove it.
func interpolateFailed(_ stations: inout [StationResult]) {
    let usable = stations.indices.filter { !stations[$0].failed && stations[$0].velocityMs.isFinite }
    guard usable.count >= 2, stations.contains(where: \.failed) else { return }

    for i in stations.indices where stations[i].failed {
        let x = stations[i].stationM
        // Nearest usable station on each side; clamp at the ends.
        let before = usable.last { stations[$0].stationM <= x }
        let after = usable.first { stations[$0].stationM >= x }
        let value: Double
        switch (before, after) {
        case let (b?, a?) where b != a:
            let xb = stations[b].stationM, xa = stations[a].stationM
            let w = xa > xb ? (x - xb) / (xa - xb) : 0
            value = stations[b].velocityMs + w * (stations[a].velocityMs - stations[b].velocityMs)
        case let (b?, _):
            value = stations[b].velocityMs
        case let (_, a?):
            value = stations[a].velocityMs
        default:
            continue
        }
        stations[i].velocityMs = value
        if !stations[i].flags.contains(.interpolated) {
            stations[i].flags.append(.interpolated)
        }
    }
}

// MARK: - direction and range advice

/// Detect and, if allowed, correct search lines that point upstream.
///
/// Simply negating the velocities would not be enough.  The velocity range is
/// nearly always asymmetric -- something like [-0.5, 8] -- so if the lines face
/// upstream the true angle lies *outside* the search range of Eq. (14).  Every
/// station then reports a value pinned near the range boundary, and flipping the
/// sign would turn that garbage into plausible-looking positive numbers.
///
/// Instead the angle range is mirrored and the stations re-estimated.  That is
/// cheap: reversing the search line mirrors the STI in space, which negates
/// `phi` and leaves the structure tensor untouched, so the existing orientation
/// fields are reused and only the histogram of Eq. (14) is recomputed.
func directionCheck(
    survey: Survey,
    stations: inout [StationResult],
    stis: [SpaceTimeImage],
    fields: [OrientationField?],
    cameraHeightM: Double,
    motion: MotionReport,
    warnings: inout [String]
) {
    let raw = stations.map(\.velocityRawMs).filter(\.isFinite).sorted()
    guard !raw.isEmpty else { return }
    let median = raw[raw.count / 2]
    guard median < 0 else { return }

    let message = "the median velocity along the search lines is \(decimals(median, 3)) m/s, "
                + "i.e. the flow runs opposite to the assumed downstream direction"

    guard survey.flow.autoFlip else {
        warnings.append(message + "; set flow.reverse, or tap a downstream point.")
        return
    }

    let vLow = survey.sti.velocityRangeMs.lowerBound
    let vHigh = survey.sti.velocityRangeMs.upperBound
    for i in stations.indices {
        guard i < fields.count, let field = fields[i], i < stis.count else { continue }
        // Mirror the search range about zero and re-run Eq. (14).
        let estimate = try? estimateAngle(survey: survey, sti: stis[i], field: field,
                                          vLow: -vHigh, vHigh: -vLow)
        // Every flag except the structural ones came from the discarded first
        // pass; drop them before re-running quality control, or the advisories
        // would be appended twice.
        stations[i].flags = stations[i].flags.filter { $0 == .stiTooSmall || $0 == .outOfFrame }
        apply(estimate, to: &stations[i], sti: stis[i], survey: survey,
              cameraHeightM: cameraHeightM, trackingDegraded: motion.trackingDegraded,
              negate: true)
    }

    warnings.append(message + "; the search range was mirrored and the stations re-analysed. "
                            + "Tap a downstream point to point the search lines the right way "
                            + "in the first place.")
}

/// Suggest a `frameStep` that puts `tan phi` in the paper's sweet spot.
///
/// Accuracy is best when `tan phi` is around 1 (Fig. 7), i.e. when a surface
/// feature moves about one STI column per row.  Since `tan phi = U St / Sx` and
/// `St = frameStep / fps`, the step scales it directly -- and on the phone the
/// user can act on this without re-recording.
func rangeAdvice(survey: Survey, stations: [StationResult]) -> (step: Int?, message: String?) {
    let tans = stations.map { abs($0.tanPhi) }.filter(\.isFinite).sorted()
    guard !tans.isEmpty else { return (nil, nil) }
    let median = tans[tans.count / 2]
    guard median > 0 else { return (nil, nil) }

    let step = survey.capture.frameStep
    let suggested = max(1, Int((Double(step) / median).rounded()))
    if median >= 0.2 && median <= 4.0 { return (suggested == step ? nil : suggested, nil) }
    if suggested == step { return (nil, nil) }

    if median > 4.0 {
        return (suggested,
                "median tan(phi) is \(decimals(median, 1)), near the top of the range the "
              + "method was validated over (Fig. 7 of the paper stops at 7). Features move "
              + "far between frames -- try frame step \(suggested), or a longer search line.")
    }
    return (suggested,
            "median tan(phi) is \(decimals(median, 2)): features barely move between frames, "
          + "so velocity resolution is poor. Try frame step \(suggested) to space the samples "
          + "further apart in time.")
}

func scaleWarning(
    survey: Survey,
    stations: [StationResult],
    cameraHeightM: Double,
    warnings: inout [String]
) {
    guard let worst = stations.max(by: { a, b in
        (a.relativeScaleSigma.isFinite ? a.relativeScaleSigma : .greatestFiniteMagnitude)
            < (b.relativeScaleSigma.isFinite ? b.relativeScaleSigma : .greatestFiniteMagnitude)
    }) else { return }
    if let advice = scaleAdvice(
        worstRelative: worst.relativeScaleSigma,
        atRangeM: worst.rangeM,
        cameraHeightM: cameraHeightM,
        threshold: survey.quality.maxRelativeScaleUncertainty) {
        warnings.append(advice)
    }
}
