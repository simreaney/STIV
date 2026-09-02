//  Export.swift
//  Getting the measurement off the phone.
//
//  Three artefacts, matching what `stiv/report.py` writes on the desktop:
//
//  * `velocity.csv`  -- the same columns as the desktop's, so a phone run drops
//                       straight into an existing spreadsheet or analysis.
//  * `summary.json`  -- machine-readable, including the geometry and the scale
//                       error budget, which the desktop has no equivalent for.
//  * `survey.yaml`   -- the run expressed as a `stiv.Config`, so the numbers can
//                       be re-derived and audited off the phone.
//
//  The YAML deserves a word.  The phone's image-to-world mapping is a ray/plane
//  intersection, and a plane-to-plane projective map is exactly what
//  `stiv.geometry.HomographyRectifier` fits -- Eq. (15) of the paper restricted
//  to the water surface.  So the mapping exports losslessly as a set of ground
//  control points: sample the reference frame at known plane points, write both
//  coordinates, and the desktop recovers the same rectification without ever
//  knowing about ARKit.

import Foundation

/// What the desktop needs that lives only in the app: reference-frame pixels.
public struct DesktopBridge: Sendable {
    public let imageWidth: Int
    public let imageHeight: Int
    public let sectionStartPx: PixelPoint
    public let sectionEndPx: PixelPoint
    public let downstreamPx: PixelPoint?
    /// Correspondences sampled from the reference frame; four or more, spread
    /// over the section, and not collinear.
    public let controlPoints: [(pixel: PixelPoint, world: PlanePoint)]

    public init(
        imageWidth: Int,
        imageHeight: Int,
        sectionStartPx: PixelPoint,
        sectionEndPx: PixelPoint,
        downstreamPx: PixelPoint?,
        controlPoints: [(pixel: PixelPoint, world: PlanePoint)]
    ) {
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.sectionStartPx = sectionStartPx
        self.sectionEndPx = sectionEndPx
        self.downstreamPx = downstreamPx
        self.controlPoints = controlPoints
    }
}

// MARK: - CSV

/// Per-station table, column-for-column as `StivResult.to_frame`.
public func velocityCSV(_ result: StivResult) -> String {
    var out = [
        "index,station_m,x_world_m,z_world_m,range_m,phi_deg,tan_phi,velocity_ms,"
        + "velocity_raw_ms,velocity_std_ms,scale_sigma_rel,coherency,valid_fraction,"
        + "n_windows,sx_m_per_px,st_s_per_px,flags"
    ]
    for s in result.stations {
        let phiDeg = s.phiRad.isFinite ? s.phiRad * 180 / .pi : Double.nan
        out.append([
            String(s.index),
            csvNumber(s.stationM), csvNumber(s.world.x), csvNumber(s.world.z),
            csvNumber(s.rangeM), csvNumber(phiDeg), csvNumber(s.tanPhi),
            csvNumber(s.velocityMs), csvNumber(s.velocityRawMs), csvNumber(s.velocityStdMs),
            csvNumber(s.relativeScaleSigma), csvNumber(s.coherency), csvNumber(s.validFraction),
            String(s.windowCount), csvNumber(s.sxMetresPerColumn), csvNumber(s.stSecondsPerRow),
            s.flags.map(\.rawValue).joined(separator: ";"),
        ].joined(separator: ","))
    }
    return out.joined(separator: "\n") + "\n"
}

/// Empty rather than `nan` for missing values, which is what pandas and Excel
/// both read back as missing.
func csvNumber(_ v: Double, precision: Int = 6) -> String {
    guard v.isFinite else { return "" }
    return String(format: "%.\(precision)g", v)
}

// MARK: - JSON

public func summaryJSON(_ result: StivResult, bridge: DesktopBridge? = nil) throws -> Data {
    let finite = result.velocities.filter(\.isFinite).sorted()
    var run: [String: Any] = [
        "generated": ISO8601DateFormatter().string(from: Date()),
        "section_name": result.survey.section.name,
        "width_m": result.widthM,
        "station_count": result.stations.count,
        "clean_stations": result.stations.filter(\.ok).count,
        "failed_stations": result.stations.filter(\.failed).count,
        "rows": result.rowCount,
        "duration_s": result.durationSeconds,
        "st_s_per_row": result.stSecondsPerRow,
        "frame_step": result.survey.capture.frameStep,
        "flow_bearing_deg": result.flowBearingDegrees,
        "camera_height_m": result.cameraHeightM,
        "scale_description": result.rectifierDescription,
        "elapsed_s": result.elapsedSeconds,
        "warnings": result.warnings,
        "motion": result.motion.summary(),
        "tracking_degraded": result.motion.trackingDegraded,
    ]
    if !finite.isEmpty {
        run["velocity_mean_ms"] = finite.reduce(0, +) / Double(finite.count)
        run["velocity_median_ms"] = finite[finite.count / 2]
        run["velocity_max_ms"] = finite[finite.count - 1]
    }
    if let step = result.suggestedFrameStep { run["suggested_frame_step"] = step }

    let stations: [[String: Any]] = result.stations.map { s in
        var d: [String: Any] = [
            "index": s.index,
            "station_m": s.stationM,
            "x_world_m": s.world.x,
            "z_world_m": s.world.z,
            "range_m": s.rangeM,
            "n_windows": s.windowCount,
            "coherency": jsonNumber(s.coherency),
            "valid_fraction": jsonNumber(s.validFraction),
            "sx_m_per_px": jsonNumber(s.sxMetresPerColumn),
            "st_s_per_px": jsonNumber(s.stSecondsPerRow),
            "flags": s.flags.map(\.rawValue),
        ]
        d["phi_deg"] = jsonNumber(s.phiRad.isFinite ? s.phiRad * 180 / .pi : .nan)
        d["tan_phi"] = jsonNumber(s.tanPhi)
        d["velocity_ms"] = jsonNumber(s.velocityMs)
        d["velocity_raw_ms"] = jsonNumber(s.velocityRawMs)
        d["velocity_std_ms"] = jsonNumber(s.velocityStdMs)
        d["scale_sigma_rel"] = jsonNumber(s.relativeScaleSigma)
        return d
    }

    var root: [String: Any] = ["run": run, "stations": stations]
    if let survey = try? JSONSerialization.jsonObject(
        with: JSONEncoder().encode(result.survey)) {
        root["survey"] = survey
    }
    if let bridge {
        let reference: [String: Any] = [
            "width": bridge.imageWidth,
            "height": bridge.imageHeight,
            "section_start_px": [bridge.sectionStartPx.u, bridge.sectionStartPx.v],
            "section_end_px": [bridge.sectionEndPx.u, bridge.sectionEndPx.v],
        ]
        root["reference_frame"] = reference
    }
    return try JSONSerialization.data(
        withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
}

/// JSON has no `NaN`, and a silent 0 would be a lie.
func jsonNumber(_ v: Double) -> Any { v.isFinite ? v : NSNull() }

// MARK: - YAML for the desktop package

/// Write the run as a `stiv.Config` the repository's `stiv run` can consume.
///
/// `video.path` is left for the user to fill in: v1 does not save the raw
/// footage, so the YAML is a faithful record of *how* the section was measured
/// rather than a self-contained re-run.  Everything else -- the rectification,
/// the section, the flow direction, every analysis parameter -- is exact.
public func surveyYAML(result: StivResult, bridge: DesktopBridge?) -> String {
    let s = result.survey
    var out: [String] = [
        "# STIV survey exported from the iPhone app.",
        "#",
        "# The phone maps image to world by intersecting each pixel's ray with the",
        "# gravity-aligned water-surface plane.  That is a plane-to-plane projective",
        "# map, so it is exported below as ground control points and refitted by",
        "# stiv.geometry.HomographyRectifier -- Eq. (15) restricted to the water",
        "# surface.  World coordinates are ARKit metres: X and Z on the water plane.",
        "#",
        "# Fill in video.path with footage of this section to re-run the analysis:",
        "#     stiv run survey.yaml",
        "",
        "video:",
        "  path: \"\"                     # <- your video of this section",
        "  frame_step: \(s.capture.frameStep)",
        "  fps_override: \(fmt(s.capture.framesPerSecond))",
        "",
    ]

    if let bridge, bridge.controlPoints.count >= 4 {
        out.append("rectify:")
        out.append("  method: gcp")
        out.append("  gcp_px:")
        for c in bridge.controlPoints {
            out.append("    - [\(fmt(c.pixel.u)), \(fmt(c.pixel.v))]")
        }
        out.append("  gcp_world:")
        for c in bridge.controlPoints {
            out.append("    - [\(fmt(c.world.x)), \(fmt(c.world.z))]")
        }
    } else {
        out.append("# No control points were exported, so the desktop cannot reproduce the")
        out.append("# rectification.  Supply your own m_per_px or GCPs here.")
        out.append("rectify:")
        out.append("  method: scale")
        out.append("  m_per_px: null")
    }
    out.append("")

    out.append("section:")
    out.append("  name: \(yamlString(s.section.name))")
    if let bridge {
        out.append("  start_px: [\(fmt(bridge.sectionStartPx.u)), \(fmt(bridge.sectionStartPx.v))]")
        out.append("  end_px: [\(fmt(bridge.sectionEndPx.u)), \(fmt(bridge.sectionEndPx.v))]")
    } else {
        out.append("  start_px: null")
        out.append("  end_px: null")
    }
    out.append("  n_stations: \(s.section.stationCount)")
    out.append("  edge_trim: \(fmt(s.section.edgeTrim))")
    out.append("")

    out.append("flow:")
    if let px = bridge?.downstreamPx {
        out.append("  direction: auto")
        out.append("  downstream_px: [\(fmt(px.u)), \(fmt(px.v))]")
    } else {
        out.append("  direction: angle")
        out.append("  angle_deg: \(fmt(result.flowBearingDegrees))")
    }
    out.append("  reverse: \(s.flow.reverse)")
    out.append("  auto_flip: \(s.flow.autoFlip)")
    out.append("")

    out.append("stabilise:")
    out.append("  # The phone compensated for camera motion with its own 6-DoF pose; a")
    out.append("  # desktop re-run of ordinary video needs the feature tracker instead.")
    out.append("  enabled: true")
    out.append("")

    out.append("stiv:")
    out.append("  search_line_length_m: \(fmt(s.sti.searchLineLengthM))")
    if let n = s.sti.sampleCount { out.append("  n_samples: \(n)") }
    out.append("  remove_static_background: \(s.sti.removeStaticBackground)")
    out.append("  normalise: \(s.sti.normalise)")
    out.append("  smooth_sigma: \(fmt(s.sti.smoothSigma))")
    out.append("  gradient_order: \(s.sti.gradientOrder.rawValue)")
    out.append("  window_px: \(s.sti.windowPx)")
    out.append("  window_type: \(s.sti.windowType.rawValue)")
    out.append("  window_overlap: \(fmt(s.sti.windowOverlap))")
    out.append("  velocity_range_ms: [\(fmt(s.sti.velocityRangeMs.lowerBound)), "
             + "\(fmt(s.sti.velocityRangeMs.upperBound))]")
    out.append("  coherency_levels: [\(s.sti.coherencyLevels.map(fmt).joined(separator: ", "))]")
    out.append("  n_angle_bins: \(s.sti.angleBins)")
    out.append("  min_coherency: \(fmt(s.sti.minCoherency))")
    out.append("  estimator: \(s.sti.estimator.rawValue)")
    out.append("")

    out.append("quality:")
    out.append("  min_valid_fraction: \(fmt(s.quality.minValidFraction))")
    out.append("  min_mean_coherency: \(fmt(s.quality.minMeanCoherency))")
    out.append("  max_relative_uncertainty: \(fmt(s.quality.maxRelativeUncertainty))")
    out.append("  tan_phi_range: [\(fmt(s.quality.tanPhiRange.lowerBound)), "
             + "\(fmt(s.quality.tanPhiRange.upperBound))]")
    out.append("  interpolate_flagged: \(s.quality.interpolateFlagged)")
    out.append("  median_filter_stations: \(s.quality.medianFilterStations)")
    out.append("")

    return out.joined(separator: "\n") + "\n"
}

func fmt(_ v: Double) -> String {
    guard v.isFinite else { return "null" }
    if v == v.rounded() && abs(v) < 1e15 { return String(format: "%.1f", v) }
    return String(format: "%.6g", v)
}

func yamlString(_ s: String) -> String {
    "\"" + s.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"") + "\""
}
