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
    case medianFiltered

    /// The spelling `stiv/analysis.py` uses -- `FAILURE_FLAGS` and
    /// `ADVISORY_FLAGS` -- for files a desktop tool may read back.
    /// `scale_uncertain` is the phone's own and has no desktop counterpart.
    public var pythonName: String {
        switch self {
        case .noOrientation:    return "no_orientation"
        case .stiTooSmall:      return "sti_too_small"
        case .lowValidFraction: return "low_valid_fraction"
        case .lowCoherency:     return "low_coherency"
        case .highUncertainty:  return "high_uncertainty"
        case .outOfFrame:       return "out_of_frame"
        case .tanPhiLow:        return "tan_phi_low"
        case .tanPhiHigh:       return "tan_phi_high"
        case .scaleUncertain:   return "scale_uncertain"
        case .interpolated:     return "interpolated"
        case .medianFiltered:   return "median_filtered"
        }
    }

    public var isFailure: Bool {
        switch self {
        case .noOrientation, .stiTooSmall, .lowValidFraction,
             .lowCoherency, .highUncertainty, .outOfFrame:
            return true
        case .tanPhiLow, .tanPhiHigh, .scaleUncertain, .interpolated, .medianFiltered:
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
        case .medianFiltered:    return "the profile median filter moved this station's velocity"
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
    /// The station centre in the reference frame, if it was in front of the
    /// camera: the desktop's `u_px`, `v_px`.
    public let pixel: PixelPoint?

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
    /// This is the figure to report and to draw error bars with.
    public var velocityStdMs: Double = .nan
    /// The angle-histogram part of `velocityStdMs` alone, m/s -- what
    /// `stiv/analysis.py` calls `velocity_std_ms`.
    ///
    /// Quality control tests this one, not the total.  The `.highUncertainty`
    /// failure means "the angle histogram is flat", which the scale term
    /// cannot speak to; letting it in would fail -- and interpolate away --
    /// clean far-bank stations purely for being far away, when the scale
    /// advice promises they are still reported.
    public var velocityStdAngleMs: Double = .nan
    /// The scale component on its own, as a fraction.
    public var relativeScaleSigma: Double = .nan
    public var sxMetresPerColumn: Double = .nan
    public var stSecondsPerRow: Double = .nan
    public var flags: [StationFlag] = []

    /// Flag the station, once: a flag already present is not added again.
    /// `StationResult.add_flag`; every flag the pipeline sets goes through it.
    public mutating func addFlag(_ flag: StationFlag) {
        if !flags.contains(flag) { flags.append(flag) }
    }

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
        self.pixel = line.centrePx
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
        // `np.nanmedian(rmse_px)`, as `StabilisationReport.summary`.
        let median = numpyMedian(residualPx.filter { !$0.isNaN })
        let maxShift = shiftPx.filter(\.isFinite).max() ?? 0
        return "motion: \(patchCount) static patches, "
             + "median pose residual \(decimals(median, 2)) px, "
             + "max frame shift \(decimals(maxShift, 1)) px, "
             + "\(failedCount)/\(frameCount) frames unchecked"
    }

    public func warnings() -> [String] {
        var out: [String] = []
        // ARKit's own tracking state, not a finding of the cross-check, so it
        // is reported whether or not the check was armed -- and an unarmed
        // check (no LiDAR, a bank out of reach) is when it matters most.
        if trackingDegraded {
            out.append("""
                world tracking was not in its normal state for part of the recording, which \
                degrades the metric scale; the reported uncertainties account for this
                """)
        }
        guard enabled else { return out }
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
        // `np.median(inlier_fraction) < 0.5`, as `StabilisationReport.warnings`.
        if !inlierFraction.isEmpty, numpyMedian(inlierFraction) < 0.5 {
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
        return out
    }
}

/// Everything a run produced.
public struct StivResult: Sendable {
    public let survey: Survey
    public let stations: [StationResult]
    /// The conditioned STIs, in single precision, for display.  Nothing
    /// numerical reads them back: the analysis has finished with them.
    public let stiPreviews: [StiPreview]
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

    /// A `frameStep` that would put the median `tan phi` near 1, where the
    /// method is most accurate -- what the results-screen slider snaps to.
    public var suggestedFrameStep: Int?

    public func summary() -> String {
        let finite = velocities.filter(\.isFinite).sorted()
        let clean = stations.filter(\.ok).count
        let failed = stations.filter(\.failed)
        let filled = failed.filter { $0.flags.contains(.interpolated) }.count
        var lines = [
            "section        \(survey.section.name): \(decimals(widthM, 2)) m wide, "
                + "\(stations.count) stations",
            "scale          \(rectifierDescription)",
            "frames         \(rowCount) over \(decimals(durationSeconds, 2)) s "
                + "(St = \(decimals(stSecondsPerRow * 1000, 1)) ms/row, "
                + "step \(survey.capture.frameStep))",
            "flow direction \(decimals(flowBearingDegrees, 1)) deg in world frame",
            motion.summary(),
            "stations       \(clean)/\(stations.count) clean, "
                + "\(failed.count) failed (\(filled) interpolated)",
        ]
        if !finite.isEmpty {
            let mean = finite.reduce(0, +) / Double(finite.count)
            lines.append("surface U      mean \(decimals(mean, 3))  "
                       + "median \(decimals(numpyMedian(finite), 3))  "
                       + "max \(decimals(finite[finite.count - 1], 3)) m/s")
        }
        if !warnings.isEmpty {
            lines.append("")
            lines.append(contentsOf: warnings.map { "warning: \($0)" })
        }
        return lines.joined(separator: "\n")
    }
}

/// `numpy.median`: the middle value, or the mean of the two middle values for
/// an even count.  NaN when empty or when any element is NaN, as numpy.
///
/// The upper median (`sorted[count / 2]`) is a different statistic for an even
/// count, and every threshold downstream of a median -- the uncertainty scale,
/// the profile filter, the frame-step advice -- moved with it.
func numpyMedian(_ xs: [Double]) -> Double {
    guard !xs.isEmpty, !xs.contains(where: \.isNaN) else { return .nan }
    let sorted = xs.sorted()
    let n = sorted.count
    return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2
}

// MARK: - the pipeline

/// Execute a full STIV analysis, refusing inputs it cannot honestly analyse.
///
/// Throws `SurveyError.invalid` when the survey fails `Survey.validated()`;
/// when the STIs do not belong to `layout` -- a different number of them, a
/// different sample count, or a non-positive scale; or when the recording has
/// too few rows, or every search line too few samples, for even one analysis
/// window (`minStiSize`), as `stiv.analysis.run` refuses such a clip before
/// reading it.  Prefer this to the non-throwing
/// `analyse(survey:layout:stis:motion:)`.
public func analyse(
    validating survey: Survey,
    layout: SectionLayout,
    stis: [SpaceTimeImage],
    motion: MotionReport = MotionReport()
) throws -> StivResult {
    try validateAnalysisInputs(survey: survey, layout: layout, stis: stis)
    return analyse(survey: survey, layout: layout, stis: stis, motion: motion)
}

func validateAnalysisInputs(survey: Survey, layout: SectionLayout, stis: [SpaceTimeImage]) throws {
    _ = try survey.validated()
    let lines = layout.searchLines
    guard stis.count == lines.count else {
        throw SurveyError.invalid("got \(stis.count) STIs for \(lines.count) search lines; "
                                + "the recording does not belong to this layout")
    }
    for (line, sti) in zip(lines, stis) {
        guard sti.cols == line.sampleCount else {
            throw SurveyError.invalid("station \(line.index): the STI has \(sti.cols) samples per row, "
                                    + "but its search line has \(line.sampleCount)")
        }
        guard sti.sxMetresPerColumn.isFinite, sti.sxMetresPerColumn > 0,
              sti.stSecondsPerRow.isFinite, sti.stSecondsPerRow > 0 else {
            throw SurveyError.invalid("station \(line.index): Sx and St must be positive and finite, "
                                    + "got \(sti.sxMetresPerColumn) m and \(sti.stSecondsPerRow) s")
        }
    }

    // Too short to fit one window: every station would fail, and used to be
    // reported afterwards as a texture or velocity-range problem.
    let s = survey.sti
    let need = minStiSize(windowPx: s.windowPx, windowType: s.windowType, gradientOrder: s.gradientOrder)
    let rows = stis.map(\.rows).min() ?? 0
    if rows < need {
        throw SurveyError.invalid("""
            the recording gives only \(rows) STI rows at frame step \(survey.capture.frameStep), \
            but a \(s.windowPx) px \(s.windowType.rawValue) window needs at least \(need); record for \
            longer, use a smaller frame step, or use a smaller window
            """)
    }
    if !lines.isEmpty && lines.allSatisfy({ $0.sampleCount < need }) {
        throw SurveyError.invalid("""
            the search lines have only \(lines.map(\.sampleCount).min() ?? 0) samples, but a \
            \(s.windowPx) px \(s.windowType.rawValue) window needs at least \(need); lengthen the \
            search lines, raise the sample count, or use a smaller window
            """)
    }
}

/// The warning for search lines too short for one window, when only some
/// are -- when all are, `analyse(validating:...)` refuses the run outright.
func narrowLineWarning(survey: Survey, layout: SectionLayout) -> String? {
    let s = survey.sti
    let need = minStiSize(windowPx: s.windowPx, windowType: s.windowType, gradientOrder: s.gradientOrder)
    let narrow = layout.searchLines.filter { $0.sampleCount < need }.map(\.index)
    guard !narrow.isEmpty, narrow.count < layout.searchLines.count else { return nil }
    let shown = narrow.prefix(5).map(String.init).joined(separator: ", ")
    let more = narrow.count > 5 ? "..." : ""
    return """
        \(narrow.count) search line(s) have fewer than the \(need) samples a \(s.windowPx) px \
        window needs (stations \(shown)\(more)) and will fail; lengthen the search lines or use \
        a smaller window
        """
}

/// Execute a full STIV analysis over already-recorded space-time images.
///
/// Never traps.  Inputs that `analyse(validating:layout:stis:motion:)` would
/// refuse are analysed anyway, as far as they can be, and the result carries a
/// warning saying why its numbers should not be trusted.
public func analyse(
    survey: Survey,
    layout: SectionLayout,
    stis: [SpaceTimeImage],
    motion: MotionReport = MotionReport()
) -> StivResult {
    let clock = Date()
    var warnings: [String] = []

    do {
        try validateAnalysisInputs(survey: survey, layout: layout, stis: stis)
    } catch {
        warnings.append("the analysis inputs are invalid (\(error)); "
                      + "treat every number in this result as meaningless")
    }
    if let narrow = narrowLineWarning(survey: survey, layout: layout) {
        warnings.append(narrow)
    }

    if !layout.stationsOutsideFrame.isEmpty {
        let shown = layout.stationsOutsideFrame.prefix(5).map(String.init).joined(separator: ", ")
        let more = layout.stationsOutsideFrame.count > 5 ? "..." : ""
        warnings.append("""
            \(layout.stationsOutsideFrame.count) search line(s) extend outside the frame \
            (stations \(shown)\(more)); shorten the search line, step back, or move the section
            """)
    }

    // Each STI is conditioned, analysed and then dropped to a single-precision
    // display copy before the next is touched, so at most one conditioned
    // `Double` STI is alive at a time.  Nothing after the orientation field
    // reads the pixels again: the direction check reuses the fields, and needs
    // only each STI's scales.
    var previews: [StiPreview] = []
    var stations: [StationResult] = []
    var fields: [OrientationField?] = []
    previews.reserveCapacity(stis.count)

    for (i, line) in layout.searchLines.enumerated() {
        var station = StationResult(line: line)
        guard i < stis.count else {
            station.addFlag(.noOrientation)
            stations.append(station)
            fields.append(nil)
            continue
        }
        let sti = stis[i]
        station.stSecondsPerRow = sti.stSecondsPerRow
        let conditioned = preprocess(sti.data, settings: survey.sti)

        var field: OrientationField?
        var estimate: AngleEstimate?
        do {
            let f = try structureTensorField(
                sti: conditioned,
                windowPx: survey.sti.windowPx,
                windowType: survey.sti.windowType,
                overlap: survey.sti.windowOverlap,
                gradientOrder: survey.sti.gradientOrder)
            field = f
            estimate = try estimateAngle(survey: survey, sti: sti, field: f,
                                         vLow: survey.sti.velocityRangeMs.lowerBound,
                                         vHigh: survey.sti.velocityRangeMs.upperBound)
        } catch {
            station.addFlag(.stiTooSmall)
        }
        previews.append(StiPreview(conditioned: conditioned, of: sti))

        apply(estimate, to: &station, sti: sti, survey: survey,
              cameraHeightM: layout.cameraHeightM,
              trackingDegraded: motion.trackingDegraded)

        if sti.outOfFrameFraction > 0.02 {
            station.addFlag(.outOfFrame)
        }

        stations.append(station)
        fields.append(field)
    }

    let flipped = directionCheck(survey: survey, stations: &stations, stis: stis, fields: fields,
                                 cameraHeightM: layout.cameraHeightM, motion: motion,
                                 warnings: &warnings)
    qualityControl(survey: survey, stations: &stations, warnings: &warnings)
    scaleWarning(survey: survey, stations: stations, cameraHeightM: layout.cameraHeightM,
                 warnings: &warnings)
    warnings.append(contentsOf: motion.warnings())

    let rows = stis.first?.rows ?? 0
    let st = stis.first?.stSecondsPerRow ?? survey.capture.stSecondsPerRow
    // Once flipped, the stations' velocities are along the reversed lines, so
    // the flow they describe -- and the bearing to report -- is reversed too.
    // The same arithmetic as `stiv.analysis.run`, into (-180, 180].
    var bearing = layout.flowBearingDegrees
    if flipped {
        bearing += 180
        if bearing > 180 { bearing -= 360 }
    }

    var result = StivResult(
        survey: survey,
        stations: stations,
        stiPreviews: previews,
        motion: motion,
        rowCount: rows,
        durationSeconds: Double(rows) * st,
        stSecondsPerRow: st,
        widthM: layout.widthM,
        flowBearingDegrees: bearing,
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
        station.velocityStdAngleMs = .nan
        station.phiStdRad = .nan
        station.coherency = 0
        station.validFraction = estimate?.validFraction ?? 0
        station.windowCount = 0
        if !station.flags.contains(.stiTooSmall) && !station.flags.contains(.noOrientation) {
            station.addFlag(.noOrientation)
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
    // Exactly `_apply_estimate`'s `velocity_std_ms`, which is all quality
    // control judges.
    let k = sti.sxMetresPerColumn / sti.stSecondsPerRow
    let angleTerm = abs(k / max(cos(phi) * cos(phi), 1e-6)) * estimate.phiStd
    station.velocityStdAngleMs = angleTerm
    // Eq. (16) is linear in Sx, so a relative scale error is a relative
    // velocity error.  The two are independent.  A scale the model cannot
    // bound (the camera at the water surface) makes the total infinite, which
    // is the honest figure to report -- and harmless, since quality control
    // reads the angle term.
    let scaleTerm = station.relativeScaleSigma.isFinite
        ? abs(station.velocityMs) * station.relativeScaleSigma
        : Double.infinity
    station.velocityStdMs = (angleTerm * angleTerm + scaleTerm * scaleTerm).squareRoot()
}

// MARK: - quality control

/// Flag doubtful stations, then fill or blank the failed ones.
///
/// The order is deliberate, and `_quality_control`'s:
///
/// 1. flag every station on its measured values;
/// 2. median-filter the *usable* stations only (if asked), so no failed value
///    can leak into a good neighbour;
/// 3. interpolate the failed stations from the *filtered* usable values, so a
///    spike the filter removed cannot come back through a filled gap;
/// 4. blank to NaN whatever failed and is still unfilled (its raw value stays
///    in `velocityRawMs`), so no failed number reaches the summary or the
///    export.
func qualityControl(survey: Survey, stations: inout [StationResult], warnings: inout [String]) {
    let q = survey.quality

    // The uncertainty test is relative to the section's own velocity scale, so
    // that a genuinely slow but well-resolved station near the bank is not
    // condemned merely for having a small denominator.
    let finite = stations.map { abs($0.velocityMs) }.filter(\.isFinite)
    let velocityScale = finite.isEmpty ? 0 : 0.25 * numpyMedian(finite)

    for i in stations.indices {
        var s = stations[i]
        if s.validFraction.isFinite && s.validFraction < q.minValidFraction {
            s.addFlag(.lowValidFraction)
        }
        if s.coherency.isFinite && s.coherency < q.minMeanCoherency {
            s.addFlag(.lowCoherency)
        }
        // The angle term only: see `StationResult.velocityStdAngleMs`.  The
        // scale term has its own, advisory, flag below.
        if s.velocityStdAngleMs.isFinite && s.velocityMs.isFinite {
            let reference = max(abs(s.velocityMs), velocityScale)
            if reference > 0 && s.velocityStdAngleMs > q.maxRelativeUncertainty * reference {
                s.addFlag(.highUncertainty)
            }
        }
        if s.tanPhi.isFinite {
            let a = abs(s.tanPhi)
            if a > q.tanPhiRange.upperBound {
                // Steep streaks: too little spatial displacement per frame is
                // resolvable.  Section 2.3 tested only up to tan phi = 7.
                s.addFlag(.tanPhiHigh)
            } else if a < q.tanPhiRange.lowerBound {
                s.addFlag(.tanPhiLow)
            }
        }
        if !s.relativeScaleSigma.isFinite || s.relativeScaleSigma > q.maxRelativeScaleUncertainty {
            s.addFlag(.scaleUncertain)
        }
        stations[i] = s
    }

    if q.medianFilterStations > 1 {
        medianFilter(&stations, window: q.medianFilterStations)
    }

    if q.interpolateFlagged {
        interpolateFailed(&stations)
    }

    let unfilled = stations.indices.filter {
        stations[$0].failed && !stations[$0].flags.contains(.interpolated)
    }
    for i in unfilled { reportVelocity(&stations[i], .nan) }
    if !unfilled.isEmpty && q.interpolateFlagged {
        let usable = stations.filter { !$0.failed }.count
        warnings.append("""
            \(unfilled.count) failed station(s) could not be interpolated -- only \(usable) \
            station(s) measured a usable velocity, and interpolation needs two -- so they \
            report no velocity
            """)
    }

    // A recording or a search line too small for the window fails every
    // station it touches, and is not a texture or velocity-range problem: say
    // so plainly, and keep those stations out of the count below.
    let tooSmall = stations.filter { $0.flags.contains(.stiTooSmall) }.count
    if tooSmall > 0 {
        let s = survey.sti
        let need = minStiSize(windowPx: s.windowPx, windowType: s.windowType,
                              gradientOrder: s.gradientOrder)
        warnings.append("""
            \(tooSmall) of \(stations.count) stations produced a space-time image too small for \
            the \(s.windowPx) px \(s.windowType.rawValue) window, which needs at least \(need) rows \
            (frames) and \(need) columns (samples); record for longer, use a smaller frame \
            step, lengthen the search lines, or use a smaller window
            """)
    }

    let bad = stations.filter { $0.failed && !$0.flags.contains(.stiTooSmall) }.count
    if Double(bad) > 0.3 * Double(max(1, stations.count)) {
        warnings.append("""
            \(bad) of \(stations.count) stations failed quality control; check that the water \
            surface has visible texture, that the velocity range brackets the true velocity, \
            and that the search lines are parallel to the flow
            """)
    }
}

/// Replace a station's reported velocity with one it did not measure;
/// `stiv.analysis._report_velocity`.
///
/// Everything that describes the reported number is kept consistent with it:
/// `tanPhi` and `phiRad` are recomputed through Eq. (16), and both spreads are
/// cleared, since they describe the measurement that was replaced rather than
/// the value now reported -- an interpolated value's uncertainty is the
/// interpolation error, which nothing here can estimate.  The raw measurement
/// stays in `velocityRawMs`; the diagnostics that explain why it was replaced
/// (coherency, valid fraction, the scale sigma) are left alone.
func reportVelocity(_ s: inout StationResult, _ velocity: Double) {
    s.velocityMs = velocity
    if velocity.isFinite && s.sxMetresPerColumn > 0 {
        s.tanPhi = velocity * s.stSecondsPerRow / s.sxMetresPerColumn
        s.phiRad = atan(s.tanPhi)
    } else {
        s.tanPhi = .nan
        s.phiRad = .nan
    }
    s.velocityStdMs = .nan
    s.velocityStdAngleMs = .nan
    s.phiStdRad = .nan
}

/// Fewest usable values a median-filter window must hold for the filter to
/// move its station: `stiv.analysis._MIN_MEDIAN_VALUES`.  The median of two
/// values is their mean, which a single spike drags as far as it likes; from
/// three up a lone outlier cannot win.
let minimumMedianValues = 3

/// Median-filter the velocity profile across the *usable* stations.
///
/// Only stations that did not fail take part, either as inputs or as outputs:
/// a failed station's raw value is exactly what the filter must not spread
/// into its neighbours.  It runs before interpolation, so the gaps are then
/// filled from the filtered values.  The window is `k` stations wide by index
/// and centred; failed stations and the ends of the section leave it holding
/// fewer values, and a station whose window holds fewer than
/// `minimumMedianValues` is left as measured rather than replaced by what
/// would really be the mean of two.  The median is numpy's.
///
/// Any station the filter actually moves is flagged.  The reported velocity is
/// then no longer the one that station measured, and a silently substituted
/// number that still reports `ok` is worse than no filter at all.  Matches
/// `stiv.analysis._median_filter`.
func medianFilter(_ stations: inout [StationResult], window k: Int) {
    let v = stations.map(\.velocityMs)
    let usable = stations.map { !$0.failed && $0.velocityMs.isFinite }
    let half = k / 2
    for i in stations.indices where usable[i] {
        let lo = max(0, i - half)
        let hi = min(v.count, i + half + 1)
        let window = (lo..<hi).filter { usable[$0] }.map { v[$0] }
        guard window.count >= minimumMedianValues else { continue }
        let filtered = numpyMedian(window)
        if filtered != v[i] {
            reportVelocity(&stations[i], filtered)
            stations[i].addFlag(.medianFiltered)
        }
    }
}

/// Replace *failed* stations with an interpolation of their neighbours.
///
/// Advisory flags are deliberately left alone: a station reading a genuinely
/// slow patch near the bank is measuring correctly, and pulling it towards its
/// faster neighbours would introduce error rather than remove it.
///
/// The usable values are taken as they stand -- after the median filter, when
/// there is one.  `numpy.interp`, operation for operation: failed stations
/// beyond the outermost usable one take that station's value, and between two
/// usable stations the fill is `slope * (x - x0) + v0`.  Nothing happens with
/// fewer than two usable stations.
func interpolateFailed(_ stations: inout [StationResult]) {
    let usable = stations.indices.filter { !stations[$0].failed && stations[$0].velocityMs.isFinite }
    guard usable.count >= 2, stations.contains(where: \.failed) else { return }

    for i in stations.indices where stations[i].failed {
        let x = stations[i].stationM
        // Nearest usable station on each side; clamp at the ends.
        let before = usable.last { stations[$0].stationM <= x }
        let after = usable.first { stations[$0].stationM > x }
        let value: Double
        switch (before, after) {
        case let (b?, a?):
            let x0 = stations[b].stationM, v0 = stations[b].velocityMs
            let slope = (stations[a].velocityMs - v0) / (stations[a].stationM - x0)
            value = x == x0 ? v0 : slope * (x - x0) + v0
        case let (b?, nil):
            value = stations[b].velocityMs
        case let (nil, a?):
            value = stations[a].velocityMs
        case (nil, nil):
            continue
        }
        reportVelocity(&stations[i], value)
        stations[i].addFlag(.interpolated)
    }
}

// MARK: - direction and range advice

/// How to point the search lines downstream, given how they were pointed;
/// `stiv.analysis._reverse_advice`, in the phone's own setting names.
func reverseAdvice(_ flow: FlowSettings) -> String {
    var advice = flow.reverse
        ? "set flow.reverse to false (it is currently true)"
        : "set flow.reverse to true"
    switch flow.direction {
    case .auto:
        advice += flow.downstream == nil
            ? ", or tap a point downstream of the section"
            : ", or check that the downstream point really is downstream of the section"
    case .points:
        advice += ", or swap flow.from and flow.to"
    case .angle:
        advice += ", or add 180 to flow.angleDegrees"
    }
    return advice
}

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
///
/// Returns whether the stations were re-analysed along the reversed lines.
@discardableResult
func directionCheck(
    survey: Survey,
    stations: inout [StationResult],
    stis: [SpaceTimeImage],
    fields: [OrientationField?],
    cameraHeightM: Double,
    motion: MotionReport,
    warnings: inout [String]
) -> Bool {
    // `velocityRawMs` came from the first pass, which searched only the
    // configured (and nearly always asymmetric) range.  When the lines face
    // upstream the true angle is not just near that range's boundary but can
    // sit entirely outside it, so *no* window's angle falls inside so narrow
    // a range and every station's first pass comes back NaN -- exactly the
    // case this check exists to catch, and exactly the case in which reading
    // the sign off `velocityRawMs` finds nothing.  Probe the full angle range
    // instead: it has no assumed direction to be wrong about.
    let limit = 0.5 * Double.pi - 1e-3
    let probe = stations.indices.compactMap { i -> Double? in
        guard i < fields.count, let field = fields[i], i < stis.count else { return nil }
        guard let estimate = try? meanOrientationAngle(
            field: field, phiMin: -limit, phiMax: limit,
            coherencyLevels: survey.sti.coherencyLevels,
            binCount: survey.sti.angleBins,
            minCoherency: survey.sti.minCoherency,
            estimator: survey.sti.estimator), estimate.phi.isFinite
        else { return nil }
        return velocity(fromAngle: estimate.phi,
                        sxMetresPerColumn: stis[i].sxMetresPerColumn,
                        stSecondsPerRow: stis[i].stSecondsPerRow)
    }.sorted()
    guard !probe.isEmpty else { return false }
    // The upper median for an even count, which `_direction_check` uses too,
    // so the two decide identically on the same input.
    let median = probe[probe.count / 2]
    guard median < 0 else { return false }

    let message = "the median velocity along the search lines is \(decimals(median, 3)) m/s, "
                + "i.e. the flow runs opposite to the assumed downstream direction"
    let advice = reverseAdvice(survey.flow)

    guard survey.flow.autoFlip else {
        warnings.append("\(message); \(advice).")
        return false
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

    warnings.append("\(message); the search range was mirrored and the stations re-analysed "
                  + "(flow.autoFlip). To point the search lines the right way in the first "
                  + "place, \(advice).")
    return true
}

/// Suggest a `frameStep` that puts `tan phi` in the paper's sweet spot.
///
/// Accuracy is best when `tan phi` is around 1 (Fig. 7), i.e. when a surface
/// feature moves about one STI column per row.  Since `tan phi = U St / Sx` and
/// `St = frameStep / fps`, the step scales it directly -- and on the phone the
/// user can act on this without re-recording.
///
/// Judged, as `_range_advice` judges it, on every station's *measured* angle,
/// failed or not -- it is when a run fails wholesale that this advice matters
/// most -- recovered from `velocityRawMs`, since the reported angle of a filled
/// station is not a measurement.
///
/// The step is returned whenever it differs from the current one, so the
/// results screen can offer it; the message only outside `0.2 ... 4`, where the
/// desktop would warn.  At step 1 with steep streaks no smaller step exists,
/// and the message says what does help instead.
func rangeAdvice(survey: Survey, stations: [StationResult]) -> (step: Int?, message: String?) {
    let tans = stations.compactMap { s -> Double? in
        guard s.velocityRawMs.isFinite, s.sxMetresPerColumn > 0 else { return nil }
        return abs(s.velocityRawMs * s.stSecondsPerRow / s.sxMetresPerColumn)
    }
    guard !tans.isEmpty else { return (nil, nil) }
    let median = numpyMedian(tans)
    guard median > 0 else { return (nil, nil) }

    let step = survey.capture.frameStep
    // `max(1, int(round(step / med)))`, with Python's halves-to-even.  The cap
    // only keeps a vanishingly small median from overflowing `Int(_:)`; no
    // recording has a million frames to stride over.
    let ratio = (Double(step) / median).rounded(.toNearestOrEven)
    let suggested = ratio.isFinite ? max(1, Int(min(ratio, 1e6))) : 1_000_000
    if median >= 0.2 && median <= 4.0 { return (suggested == step ? nil : suggested, nil) }

    if median > 4.0 {
        if suggested < step {
            return (suggested,
                    "median tan(phi) is \(decimals(median, 1)), near the top of the range the "
                  + "method was validated over (Fig. 7 of the paper stops at 7). Features move "
                  + "far between STI rows -- try frame step \(suggested).")
        }
        // Already reading every frame, so St cannot shrink any further and
        // only Sx is left.  A longer search line does not change Sx when the
        // sample count is automatic -- the samples stay about a pixel apart
        // and the line merely gains columns -- so that is not offered.
        let sx = numpyMedian(stations.map(\.sxMetresPerColumn))
        let columns = (survey.sti.searchLineLengthM / (sx * median)).rounded(.toNearestOrEven)
        let samples = columns.isFinite ? max(16, Int(min(columns, 1e6)) + 1) : 16
        let longer = survey.sti.sampleCount == nil
            ? " A longer search line does not help: with an automatic sample count it only "
              + "adds columns."
            : ""
        return (nil,
                "median tan(phi) is \(decimals(median, 1)) with the frame step already 1, near "
              + "the top of the range the method was validated over (Fig. 7 of the paper stops "
              + "at 7): features move about \(decimals(median, 0)) STI columns per frame. Since "
              + "tan(phi) = U St / Sx, what helps is a shorter St -- a higher capture frame "
              + "rate -- or a coarser Sx: film from further away or higher up, or set the "
              + "sample count to about \(samples) so the samples are spaced wider (only if the "
              + "surface texture is coarser than that spacing).\(longer)")
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
