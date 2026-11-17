//  ExportTests.swift
//  What leaves the phone: `velocity.csv`, `summary.json` and `survey.yaml`.

import XCTest
@testable import StivKit

private func projection() -> WaterPlaneProjection {
    WaterPlaneProjection(
        intrinsics: PinholeIntrinsics(horizontalFovDegrees: 60, width: 1920, height: 1440),
        pose: .looking(fromHeight: 3.0, downDegrees: 25),
        plane: WaterPlane(y: 0, fitResidualM: 0.02, sampleCount: 60))
}

private func survey() -> Survey {
    var s = Survey()
    s.section.stationCount = 4
    s.section.start = PlanePoint(x: -4, z: -8)
    s.section.end = PlanePoint(x: 4, z: -8)
    s.sti.searchLineLengthM = 4.0
    s.sti.sampleCount = 96
    s.flow.downstream = PlanePoint(x: 0, z: -20)
    s.capture.frameStep = 1
    return s
}

/// A small, real result to export.
private func result(_ s: Survey = survey()) throws -> StivResult {
    let layout = try buildSection(survey: s, rectifier: projection())
    let st = s.capture.stSecondsPerRow
    let stis = layout.searchLines.map { line in
        makeSTI(syntheticSTI(tanPhi: 0.9 * st / line.sxMetresPerColumn, rows: 160,
                             cols: line.sampleCount, noise: 0.1, seed: UInt64(line.index) + 1),
                index: line.index, sx: line.sxMetresPerColumn, st: st)
    }
    return analyse(survey: s, layout: layout, stis: stis)
}

/// The same result with some fields replaced.
private func replacing(
    _ r: StivResult,
    stations: [StationResult]? = nil,
    widthM: Double? = nil,
    durationSeconds: Double? = nil,
    cameraHeightM: Double? = nil
) -> StivResult {
    StivResult(
        survey: r.survey, stations: stations ?? r.stations, stiPreviews: r.stiPreviews,
        motion: r.motion, rowCount: r.rowCount,
        durationSeconds: durationSeconds ?? r.durationSeconds,
        stSecondsPerRow: r.stSecondsPerRow, widthM: widthM ?? r.widthM,
        flowBearingDegrees: r.flowBearingDegrees, cameraHeightM: cameraHeightM ?? r.cameraHeightM,
        rectifierDescription: r.rectifierDescription, warnings: r.warnings)
}

/// `stiv.geometry.resolve_flow_direction` for the `angle` form, from the YAML.
private func pythonDirection(yaml: String) throws -> PlaneVector {
    let flow = try XCTUnwrap(yaml.components(separatedBy: "\nflow:\n").last)
        .components(separatedBy: "\n\n")[0]
    func value(_ key: String) throws -> String {
        let line = try XCTUnwrap(flow.split(separator: "\n").first { $0.contains("\(key):") })
        return String(line.split(separator: ":", maxSplits: 1)[1])
            .split(separator: "#")[0].trimmingCharacters(in: .whitespaces)
    }
    XCTAssertEqual(try value("direction"), "angle")
    let a = try XCTUnwrap(Double(try value("angle_deg"))) * .pi / 180
    let d = PlaneVector(x: cos(a), z: sin(a))
    return try value("reverse") == "true" ? -d : d
}

final class CSVExportTests: XCTestCase {
    /// `StivResult.to_frame`'s columns, in its order, then the phone's own.
    func testColumnsMatchTheDesktopThenAddThePhonesOwn() throws {
        let csv = velocityCSV(try result())
        let lines = csv.split(separator: "\n").map(String.init)
        XCTAssertEqual(lines[0].split(separator: ",").map(String.init), [
            "index", "station_m", "x_world_m", "y_world_m", "u_px", "v_px", "phi_deg", "tan_phi",
            "velocity_ms", "velocity_raw_ms", "velocity_std_ms", "coherency", "valid_fraction",
            "n_windows", "sx_m_per_px", "st_s_per_px", "flags",
            "z_world_m", "range_m", "scale_sigma_rel", "velocity_std_total_ms",
        ])
        XCTAssertEqual(lines.count, 5)
        for line in lines.dropFirst() {
            XCTAssertEqual(line.split(separator: ",", omittingEmptySubsequences: false).count, 21, line)
        }
    }

    func testRowsCarryTheDesktopsQuantities() throws {
        let r = try result()
        let rows = velocityCSV(r).split(separator: "\n").dropFirst()
            .map { $0.split(separator: ",", omittingEmptySubsequences: false).map(String.init) }
        for (s, row) in zip(r.stations, rows) {
            XCTAssertEqual(Double(row[3]) ?? .nan, s.world.z, accuracy: 1e-5, "y_world_m is ARKit z")
            let px = try XCTUnwrap(s.pixel)
            XCTAssertEqual(Double(row[4]) ?? .nan, px.u, accuracy: 1e-2)
            XCTAssertEqual(Double(row[5]) ?? .nan, px.v, accuracy: 1e-2)
            // The desktop's velocity_std_ms is the angle spread alone.
            XCTAssertEqual(Double(row[10]) ?? .nan, s.velocityStdAngleMs,
                           accuracy: 1e-5 * max(1, s.velocityStdAngleMs))
            XCTAssertEqual(Double(row[20]) ?? .nan, s.velocityStdMs,
                           accuracy: 1e-5 * max(1, s.velocityStdMs))
        }
    }

    func testFlagsUseThePythonSpelling() throws {
        let r = try result()
        var stations = r.stations
        stations[0].flags = [.noOrientation, .tanPhiLow, .scaleUncertain]
        stations[1].flags = [.lowValidFraction, .medianFiltered, .interpolated]
        let rows = velocityCSV(replacing(r, stations: stations)).split(separator: "\n")
        XCTAssertTrue(rows[1].contains(",no_orientation;tan_phi_low;scale_uncertain,"), String(rows[1]))
        XCTAssertTrue(rows[2].contains(",low_valid_fraction;median_filtered;interpolated,"), String(rows[2]))
        for flag in StationFlag.allCases {
            XCTAssertEqual(flag.pythonName, flag.pythonName.lowercased())
            XCTAssertFalse(flag.pythonName.contains(where: \.isUppercase))
        }
    }
}

final class JSONExportTests: XCTestCase {
    /// `JSONSerialization` raises an Objective-C exception on NaN or infinity,
    /// which `try` cannot catch.  If any number slipped past `jsonNumber`
    /// this test would not fail -- it would crash the test runner.
    func testNonFiniteValuesBecomeNullInsteadOfCrashing() throws {
        let r = try result()
        var stations = r.stations
        stations[0] = StationResult(line: makeLine(index: 0, stationM: .nan, rangeM: .infinity))
        let broken = replacing(r, stations: stations, widthM: .nan,
                               durationSeconds: .infinity, cameraHeightM: -.infinity)
        let bridge = DesktopBridge(
            imageWidth: 1920, imageHeight: 1440,
            sectionStartPx: PixelPoint(u: .nan, v: 10),
            sectionEndPx: PixelPoint(u: 5, v: .infinity),
            downstreamPx: PixelPoint(u: .nan, v: .nan),
            controlPoints: [])

        let data = try summaryJSON(broken, bridge: bridge)
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let run = try XCTUnwrap(root["run"] as? [String: Any])
        XCTAssertTrue(run["width_m"] is NSNull)
        XCTAssertTrue(run["duration_s"] is NSNull)
        XCTAssertTrue(run["camera_height_m"] is NSNull)
        let first = try XCTUnwrap((root["stations"] as? [[String: Any]])?.first)
        XCTAssertTrue(first["station_m"] is NSNull)
        XCTAssertTrue(first["range_m"] is NSNull)
        let reference = try XCTUnwrap(root["reference_frame"] as? [String: Any])
        XCTAssertTrue((reference["section_start_px"] as? [Any])?.first is NSNull)
    }

    func testStationsCarryBothSigmasAndPythonFlags() throws {
        let r = try result()
        var stations = r.stations
        stations[0].flags = [.highUncertainty]
        let data = try summaryJSON(replacing(r, stations: stations))
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let first = try XCTUnwrap((root["stations"] as? [[String: Any]])?.first)
        XCTAssertEqual(first["flags"] as? [String], ["high_uncertainty"])
        XCTAssertEqual(try XCTUnwrap(first["velocity_std_ms"] as? Double),
                       stations[0].velocityStdAngleMs, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(first["velocity_std_total_ms"] as? Double),
                       stations[0].velocityStdMs, accuracy: 1e-12)
    }
}

final class YAMLExportTests: XCTestCase {
    /// `fps_override` replaces the frame rate of whatever footage the desktop
    /// re-run is given: 30 fps video read at the phone's 60 doubled every
    /// velocity.  The phone's rate is a comment only, and it is the rate the
    /// phone measured -- here 30 -- not the format's nominal 60.
    func testTheYAMLDoesNotImposeThePhonesFrameRate() throws {
        var s = survey()
        s.capture.framesPerSecond = 60
        let layout = try buildSection(survey: s, rectifier: projection())
        let stis = layout.searchLines.map { line in
            makeSTI(syntheticSTI(tanPhi: 0.9 / 30 / line.sxMetresPerColumn, rows: 160,
                                 cols: line.sampleCount, noise: 0.1, seed: UInt64(line.index) + 1),
                    index: line.index, sx: line.sxMetresPerColumn, st: 1.0 / 30)
        }
        let yaml = surveyYAML(result: analyse(survey: s, layout: layout, stis: stis), bridge: nil)
        let keys = yaml.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }
        XCTAssertFalse(keys.contains { $0.hasPrefix("fps_override") }, yaml)
        XCTAssertTrue(yaml.contains("# The phone recorded at 30.0 fps (measured)"), yaml)
    }

    /// The bearing already includes the reversal.  Writing it next to
    /// `reverse: true` made the desktop reverse it again and run the flow
    /// backwards; resolved the way the Python resolves it, the YAML must give
    /// the direction the phone measured along.
    func testAReversedBearingIsNotReversedTwice() throws {
        var s = survey()
        s.flow.direction = .angle
        s.flow.angleDegrees = 30
        s.flow.reverse = true
        let r = try result(s)
        XCTAssertEqual(r.flowBearingDegrees, -150, accuracy: 1e-9)

        let yaml = surveyYAML(result: r, bridge: nil)
        XCTAssertTrue(yaml.contains("  angle_deg: -150"), yaml)
        XCTAssertTrue(yaml.contains("  reverse: false"), yaml)

        let layout = try buildSection(survey: s, rectifier: projection())
        let d = try pythonDirection(yaml: yaml)
        XCTAssertEqual(d.x, layout.flowDirection.x, accuracy: 1e-6)
        XCTAssertEqual(d.z, layout.flowDirection.z, accuracy: 1e-6)
    }

    /// The `auto` form is resolved afresh on the desktop -- perpendicular to
    /// the section, towards the downstream marker, *then* reversed -- so it is
    /// the one form that must carry the flag.
    func testTheAutoFormKeepsTheReverseFlag() throws {
        var s = survey()
        s.flow.reverse = true
        let r = try result(s)
        let bridge = DesktopBridge(
            imageWidth: 1920, imageHeight: 1440,
            sectionStartPx: PixelPoint(u: 100, v: 900), sectionEndPx: PixelPoint(u: 1800, v: 900),
            downstreamPx: PixelPoint(u: 960, v: 700), controlPoints: [])
        let yaml = surveyYAML(result: r, bridge: bridge)
        XCTAssertTrue(yaml.contains("  direction: auto"), yaml)
        XCTAssertTrue(yaml.contains("  reverse: true"), yaml)
        XCTAssertFalse(yaml.contains("angle_deg"), yaml)
    }

    /// A survey that used a fixed angle exports that angle even when a
    /// downstream marker happens to be on hand.
    func testAnAngleSurveyExportsItsAngle() throws {
        var s = survey()
        s.flow.direction = .angle
        s.flow.angleDegrees = 80
        let r = try result(s)
        let bridge = DesktopBridge(
            imageWidth: 1920, imageHeight: 1440,
            sectionStartPx: PixelPoint(u: 100, v: 900), sectionEndPx: PixelPoint(u: 1800, v: 900),
            downstreamPx: PixelPoint(u: 960, v: 700), controlPoints: [])
        let yaml = surveyYAML(result: r, bridge: bridge)
        let d = try pythonDirection(yaml: yaml)
        XCTAssertEqual(atan2(d.z, d.x) * 180 / .pi, 80, accuracy: 1e-4)
    }
}
