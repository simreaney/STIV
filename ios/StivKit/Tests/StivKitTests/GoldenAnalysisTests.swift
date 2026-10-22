//  GoldenAnalysisTests.swift
//  Pins the analysis stage -- what happens between the Eq. (14) estimates and
//  the reported profile -- to `stiv/analysis.py`.
//
//  Quality control, the profile median filter, interpolation, the direction
//  check, the frame-step advice and the velocity sigma are all small, and all
//  easy to get subtly different: an upper median for numpy's, Swift's
//  round-half-up for Python's half-to-even, flags appended in another order.
//  None of those moves a synthetic end-to-end velocity far enough to notice.
//
//  The section comes from `make_golden_fixture.py` and is newer than the rest
//  of the fixture; these tests skip until the fixture is regenerated with it.

import XCTest
@testable import StivKit

/// The `analysis` section of the golden fixture.  Nulls are NaN.
private struct GoldenAnalysis: Decodable {
    struct Quality: Decodable {
        let min_valid_fraction: Double
        let min_mean_coherency: Double
        let max_relative_uncertainty: Double
        let tan_phi_range: [Double]
        let interpolate_flagged: Bool
        let median_filter_stations: Int

        func apply(to q: inout QualitySettings) {
            q.minValidFraction = min_valid_fraction
            q.minMeanCoherency = min_mean_coherency
            q.maxRelativeUncertainty = max_relative_uncertainty
            q.tanPhiRange = tan_phi_range[0] ... tan_phi_range[1]
            q.interpolateFlagged = interpolate_flagged
            q.medianFilterStations = median_filter_stations
        }
    }
    struct Station: Decodable {
        let station_m: Double?
        let velocity_ms: Double?
        let velocity_raw_ms: Double?
        let velocity_std_ms: Double?
        let phi_rad: Double?
        let tan_phi: Double?
        let coherency: Double?
        let valid_fraction: Double?
        let n_windows: Int
        let flags: [String]
    }
    struct QualityControl: Decodable {
        let quality: Quality
        let stations_in: [Station]
        let stations_out: [Station]
        let n_warnings: Int
    }
    struct RangeAdvice: Decodable {
        let frame_step: Int
        let tan_phi: [Double]
        let median: Double
        let suggested: Int?
        let warned: Bool
    }
    struct VelocityStd: Decodable {
        let phi: Double
        let phi_std: Double
        let sx: Double
        let st: Double
        let negate: Bool
        let velocity_ms: Double?
        let velocity_std_ms: Double?
    }
    struct Direction: Decodable {
        struct Settings: Decodable {
            let remove_static_background: Bool
            let normalise: Bool
            let clahe: Bool
            let smooth_sigma: Double
        }
        struct Run: Decodable {
            let auto_flip: Bool
            let quality: Quality
            let stations_out: [Station]
            let opposite_warning: Bool
            let n_warnings: Int
        }
        let sti_settings: Settings
        let window_px: Int
        let window_type: String
        let overlap: Double
        let gradient_order: Int
        let sx: Double
        let st: Double
        let mirrored: [Bool]
        let runs: [Run]
    }
    let quality_control: QualityControl
    let range_advice: [RangeAdvice]
    let velocity_std: [VelocityStd]
    let direction: Direction
}

private struct AnalysisSection: Decodable { let analysis: GoldenAnalysis? }

private extension Optional where Wrapped == Double {
    var orNaN: Double { self ?? .nan }
}

final class GoldenAnalysisTests: XCTestCase {
    private func section() throws -> GoldenAnalysis {
        guard let url = Golden.jsonURL() else {
            throw XCTSkip("No golden fixture; run `python ios/tools/make_golden_fixture.py`.")
        }
        let decoded = try JSONDecoder().decode(AnalysisSection.self, from: Data(contentsOf: url))
        return try Golden.require(decoded.analysis, "analysis")
    }

    private func flags(_ station: StationResult) -> [String] { station.flags.map(\.pythonName) }

    private func assertStation(
        _ got: StationResult, _ want: GoldenAnalysis.Station, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let pairs: [(String, Double, Double?, Double)] = [
            ("velocity_ms", got.velocityMs, want.velocity_ms, 1e-9),
            ("velocity_raw_ms", got.velocityRawMs, want.velocity_raw_ms, 1e-9),
            ("velocity_std_ms", got.velocityStdAngleMs, want.velocity_std_ms, 1e-9),
            ("phi_rad", got.phiRad, want.phi_rad, 1e-9),
            ("tan_phi", got.tanPhi, want.tan_phi, 1e-9),
            ("coherency", got.coherency, want.coherency, 1e-9),
            ("valid_fraction", got.validFraction, want.valid_fraction, 1e-12),
        ]
        for (name, value, expected, accuracy) in pairs {
            XCTAssertTrue(sameOrBothNaN(value, expected.orNaN, accuracy: accuracy),
                          "\(label) \(name): got \(value), want \(expected.orNaN)", file: file, line: line)
        }
        XCTAssertEqual(got.windowCount, want.n_windows, "\(label) n_windows", file: file, line: line)
        XCTAssertEqual(flags(got), want.flags, "\(label) flags", file: file, line: line)
    }

    /// Flags, the median filter and interpolation, in the pipeline's order.
    func testQualityControlMatchesThePython() throws {
        let want = try section().quality_control
        var survey = Survey()
        want.quality.apply(to: &survey.quality)

        var stations = want.stations_in.enumerated().map { i, s -> StationResult in
            var station = makeStation(
                index: i, stationM: s.station_m ?? Double(i),
                velocity: s.velocity_ms.orNaN, angleStd: s.velocity_std_ms.orNaN,
                tanPhi: s.tan_phi.orNaN, coherency: s.coherency.orNaN,
                validFraction: s.valid_fraction.orNaN,
                flags: s.flags.compactMap(StationFlag.init(pythonName:)))
            station.velocityRawMs = s.velocity_raw_ms.orNaN
            return station
        }
        var warnings: [String] = []
        qualityControl(survey: survey, stations: &stations, warnings: &warnings)

        XCTAssertEqual(stations.count, want.stations_out.count)
        for (i, (got, expected)) in zip(stations, want.stations_out).enumerated() {
            // Every reported field: a filled, blanked or filtered station
            // must also recompute its angle and drop its spread.
            assertStation(got, expected, "station \(i)")
        }
        XCTAssertEqual(warnings.count, want.n_warnings, "\(warnings)")
    }

    /// The frame-step advice: numpy's median and Python's half-to-even
    /// `round` both decide which step is suggested, and whether one is.
    func testRangeAdviceMatchesThePython() throws {
        for want in try section().range_advice {
            var survey = Survey()
            survey.capture.frameStep = want.frame_step
            let stations = want.tan_phi.enumerated().map { i, t in
                makeStation(index: i, velocity: t, tanPhi: t, sx: 1, st: 1)
            }
            let advice = rangeAdvice(survey: survey, stations: stations)
            let label = "step \(want.frame_step), tan phi \(want.tan_phi)"
            XCTAssertEqual(advice.message != nil, want.warned, label)
            if want.warned {
                // No suggestion in the Python's message means there is no
                // smaller step to suggest.
                XCTAssertEqual(advice.step, want.suggested, label)
            }
        }
    }

    /// Eq. (16)'s error propagation, including the `cos^2` floor near 90 deg.
    func testVelocitySigmaMatchesThePython() throws {
        for want in try section().velocity_std {
            var station = StationResult(line: makeLine(index: 0, stationM: 0, sx: want.sx))
            let estimate = AngleEstimate(
                phi: want.phi, tanPhi: tan(want.phi), phiStd: want.phi_std,
                validFraction: 0.9, meanCoherency: 0.7, usedCount: 12,
                histogram: [0], binCentres: [0], phiMin: -1, phiMax: 1)
            apply(estimate, to: &station,
                  sti: makeSTI(Matrix(rows: 1, cols: 1), sx: want.sx, st: want.st),
                  survey: Survey(), cameraHeightM: 10, trackingDegraded: false,
                  negate: want.negate)
            let label = "phi \(want.phi), negate \(want.negate)"
            let v = want.velocity_ms.orNaN
            let sd = want.velocity_std_ms.orNaN
            XCTAssertEqual(station.velocityMs, v, accuracy: 1e-12 * max(1, abs(v)), label)
            XCTAssertEqual(station.velocityStdAngleMs, sd, accuracy: 1e-12 * max(1, abs(sd)), label)
        }
    }

    /// The direction check on real orientation fields: the fixture STI and its
    /// mirror image, so that most search lines face upstream.
    func testDirectionCheckMatchesThePython() throws {
        let want = try section().direction
        let (_, raw) = try Golden.skipUnlessPresent()
        XCTAssertFalse(want.sti_settings.clahe, "the Swift port has no CLAHE")

        var survey = Survey()
        survey.sti.removeStaticBackground = want.sti_settings.remove_static_background
        survey.sti.normalise = want.sti_settings.normalise
        survey.sti.smoothSigma = want.sti_settings.smooth_sigma
        survey.sti.windowPx = want.window_px
        survey.sti.windowType = try XCTUnwrap(WindowType(rawValue: want.window_type))
        survey.sti.windowOverlap = want.overlap
        survey.sti.gradientOrder = try XCTUnwrap(GradientOrder(rawValue: want.gradient_order))
        // The scale term is the phone's own; keep it out of the comparison.
        survey.quality.maxRelativeScaleUncertainty = .infinity

        for run in want.runs {
            survey.flow.autoFlip = run.auto_flip
            run.quality.apply(to: &survey.quality)

            var stations: [StationResult] = []
            var stis: [SpaceTimeImage] = []
            var fields: [OrientationField?] = []
            for (i, isMirrored) in want.mirrored.enumerated() {
                let sti = makeSTI(isMirrored ? mirrored(raw) : raw, index: i, sx: want.sx, st: want.st)
                let conditioned = preprocess(sti.data, settings: survey.sti)
                let field = try structureTensorField(
                    sti: conditioned, windowPx: survey.sti.windowPx,
                    windowType: survey.sti.windowType, overlap: survey.sti.windowOverlap,
                    gradientOrder: survey.sti.gradientOrder)
                let estimate = try estimateAngle(
                    survey: survey, sti: sti, field: field,
                    vLow: survey.sti.velocityRangeMs.lowerBound,
                    vHigh: survey.sti.velocityRangeMs.upperBound)
                var station = StationResult(line: makeLine(index: i, stationM: Double(i), sx: want.sx))
                apply(estimate, to: &station, sti: sti, survey: survey,
                      cameraHeightM: 10, trackingDegraded: false)
                stations.append(station)
                stis.append(sti)
                fields.append(field)
            }

            var warnings: [String] = []
            directionCheck(survey: survey, stations: &stations, stis: stis, fields: fields,
                           cameraHeightM: 10, motion: MotionReport(), warnings: &warnings)
            qualityControl(survey: survey, stations: &stations, warnings: &warnings)

            XCTAssertEqual(warnings.contains { $0.contains("opposite to the assumed") },
                           run.opposite_warning, "autoFlip \(run.auto_flip): \(warnings)")
            XCTAssertEqual(warnings.count, run.n_warnings, "autoFlip \(run.auto_flip): \(warnings)")
            for (i, (got, expected)) in zip(stations, run.stations_out).enumerated() {
                assertStation(got, expected, "autoFlip \(run.auto_flip) station \(i)")
            }
        }
    }
}
