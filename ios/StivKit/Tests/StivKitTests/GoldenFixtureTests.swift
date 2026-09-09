//  GoldenFixtureTests.swift
//  Pins the Swift port to the Python original, number for number.
//
//  The synthetic-slope tests in OrientationTests catch a port that is *wrong*.
//  This one catches a port that is merely *different*: a boundary rule off by
//  one pixel, scipy's `reflect` where numpy's was meant, an even-window origin
//  placed on the other side, a histogram bin edge counted the other way.  None
//  of those move a clean synthetic answer far enough to fail a 10% tolerance,
//  and all of them bias real measurements.
//
//  The fixture is produced by `ios/tools/make_golden_fixture.py`, which must be
//  run once on a machine with the Python package installed.  Absent it, these
//  tests skip rather than fail -- the suite has to stay runnable on a machine
//  that has never seen numpy.

import XCTest
@testable import StivKit

final class GoldenFixtureTests: XCTestCase {
    private struct Golden: Decodable {
        struct Sti: Decodable { let rows: Int; let cols: Int; let file: String }
        struct Parameters: Decodable {
            let window_px: Int
            let window_type: String
            let overlap: Double
            let gradient_order: Int
            let phi_min: Double
            let phi_max: Double
            let coherency_levels: [Double]
            let n_bins: Int
            let min_coherency: Double
            let tan_phi_truth: Double
        }
        struct Probe: Decodable {
            let i: Int; let j: Int
            let phi: Double; let coherency: Double; let energy: Double
        }
        struct Field: Decodable {
            let shape: [Int]
            let centres_t: [Int]
            let centres_x: [Int]
            let phi_mean: Double
            let phi_min: Double
            let phi_max: Double
            let coherency_mean: Double
            let coherency_min: Double
            let coherency_max: Double
            let energy_mean: Double
            let probes: [Probe]
        }
        struct Estimate: Decodable {
            let phi: Double
            let tan_phi: Double
            let phi_std: Double
            let valid_fraction: Double
            let mean_coherency: Double
            let n_used: Int
            let hist: [Double]
            let bin_centres: [Double]
        }
        struct Preprocess: Decodable {
            struct Settings: Decodable {
                let remove_static_background: Bool
                let normalise: Bool
                let clahe: Bool
                let smooth_sigma: Double
            }
            struct Cell: Decodable { let r: Int; let c: Int; let value: Double }
            let settings: Settings
            let mean: Double
            let std: Double
            let min: Double
            let max: Double
            let probes: [Cell]
        }
        let sti: Sti
        let parameters: Parameters
        let preprocess: Preprocess
        let field: Field
        let estimate: Estimate
    }

    /// Both fixture files, or `nil` when the generator has not been run.
    private func loadFixture() throws -> (Golden, Matrix)? {
        let bundle = Bundle.module
        guard let jsonURL = bundle.url(forResource: "golden", withExtension: "json",
                                       subdirectory: "Fixtures")
                ?? bundle.url(forResource: "golden", withExtension: "json") else {
            return nil
        }
        let golden = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: jsonURL))

        let stiURL = jsonURL.deletingLastPathComponent().appendingPathComponent(golden.sti.file)
        guard FileManager.default.fileExists(atPath: stiURL.path) else { return nil }
        let raw = try Data(contentsOf: stiURL)

        let expected = golden.sti.rows * golden.sti.cols
        guard raw.count == expected * 4 else {
            XCTFail("golden_sti.f32 is \(raw.count) bytes, expected \(expected * 4)")
            return nil
        }
        var floats = [Float](repeating: 0, count: expected)
        _ = floats.withUnsafeMutableBytes { raw.copyBytes(to: $0) }
        return (golden, Matrix(rows: golden.sti.rows, cols: golden.sti.cols, floats: floats))
    }

    private func skipUnlessPresent() throws -> (Golden, Matrix) {
        guard let fixture = try loadFixture() else {
            throw XCTSkip("""
                No golden fixture. Generate it on a machine with the Python package \
                installed: `python ios/tools/make_golden_fixture.py`.
                """)
        }
        return fixture
    }

    private func rebuildField(_ golden: Golden, _ sti: Matrix) throws -> OrientationField {
        let order = try XCTUnwrap(GradientOrder(rawValue: golden.parameters.gradient_order))
        let type = try XCTUnwrap(WindowType(rawValue: golden.parameters.window_type))
        return try structureTensorField(
            sti: sti,
            windowPx: golden.parameters.window_px,
            windowType: type,
            overlap: golden.parameters.overlap,
            gradientOrder: order)
    }

    /// `preprocess` runs *before* the structure tensor, so a divergence here
    /// moves every angle that follows.  Nothing pinned it until this test,
    /// which is why the port's missing CLAHE branch went unnoticed.
    ///
    /// At 1e-9 this catches normalising after smoothing instead of before, a
    /// boundary rule of "reflect" where scipy's "nearest" was meant, a
    /// different Gaussian truncation radius, and a wrong sigma.  It does not
    /// catch the order of the two mean subtractions, because double centring
    /// commutes: either order gives X - colmean - rowmean + grandmean.
    func testPreprocessMatchesThePython() throws {
        let (golden, sti) = try skipUnlessPresent()
        let want = golden.preprocess

        XCTAssertFalse(want.settings.clahe,
                       "the Swift port has no CLAHE; the fixture must not enable it")

        var settings = StiSettings()
        settings.removeStaticBackground = want.settings.remove_static_background
        settings.normalise = want.settings.normalise
        settings.smoothSigma = want.settings.smooth_sigma

        let got = StivKit.preprocess(sti, settings: settings)
        XCTAssertEqual(got.rows, sti.rows)
        XCTAssertEqual(got.cols, sti.cols)

        let (mean, std) = meanAndStd(got.data)
        XCTAssertEqual(mean, want.mean, accuracy: 1e-9)
        XCTAssertEqual(std, want.std, accuracy: 1e-9)
        XCTAssertEqual(got.data.min() ?? .nan, want.min, accuracy: 1e-9)
        XCTAssertEqual(got.data.max() ?? .nan, want.max, accuracy: 1e-9)

        // Corners and edges, where a boundary rule differs.
        for cell in want.probes {
            XCTAssertEqual(got[cell.r, cell.c], cell.value, accuracy: 1e-9,
                           "cell (\(cell.r), \(cell.c))")
        }
    }

    func testWindowGridMatchesThePython() throws {
        let (golden, sti) = try skipUnlessPresent()
        let field = try rebuildField(golden, sti)
        // The grid is where an off-by-one in the margin arithmetic surfaces.
        XCTAssertEqual([field.phi.rows, field.phi.cols], golden.field.shape)
        XCTAssertEqual(field.centresT, golden.field.centres_t)
        XCTAssertEqual(field.centresX, golden.field.centres_x)
    }

    func testStructureTensorFieldMatchesThePython() throws {
        let (golden, sti) = try skipUnlessPresent()
        let field = try rebuildField(golden, sti)

        let phiMean = field.phi.data.reduce(0, +) / Double(field.phi.count)
        let cohMean = field.coherency.data.reduce(0, +) / Double(field.coherency.count)
        let energyMean = field.energy.data.reduce(0, +) / Double(field.energy.count)

        XCTAssertEqual(phiMean, golden.field.phi_mean, accuracy: 1e-9)
        XCTAssertEqual(field.phi.data.min() ?? .nan, golden.field.phi_min, accuracy: 1e-9)
        XCTAssertEqual(field.phi.data.max() ?? .nan, golden.field.phi_max, accuracy: 1e-9)
        XCTAssertEqual(cohMean, golden.field.coherency_mean, accuracy: 1e-9)
        XCTAssertEqual(field.coherency.data.min() ?? .nan,
                       golden.field.coherency_min, accuracy: 1e-9)
        XCTAssertEqual(field.coherency.data.max() ?? .nan,
                       golden.field.coherency_max, accuracy: 1e-9)
        // Energy is the trace of J, so it carries the absolute scale of the
        // gradients: a stencil normalised differently shows up here and
        // nowhere else.
        XCTAssertEqual(energyMean / golden.field.energy_mean, 1.0, accuracy: 1e-9)

        for probe in golden.field.probes {
            XCTAssertEqual(field.phi[probe.i, probe.j], probe.phi, accuracy: 1e-9,
                           "phi at window (\(probe.i), \(probe.j))")
            XCTAssertEqual(field.coherency[probe.i, probe.j], probe.coherency, accuracy: 1e-9,
                           "coherency at window (\(probe.i), \(probe.j))")
        }
    }

    func testEquationFourteenMatchesThePython() throws {
        let (golden, sti) = try skipUnlessPresent()
        let field = try rebuildField(golden, sti)
        let estimate = try meanOrientationAngle(
            field: field,
            phiMin: golden.parameters.phi_min,
            phiMax: golden.parameters.phi_max,
            coherencyLevels: golden.parameters.coherency_levels,
            binCount: golden.parameters.n_bins,
            minCoherency: golden.parameters.min_coherency)

        XCTAssertEqual(estimate.phi, golden.estimate.phi, accuracy: 1e-9)
        XCTAssertEqual(estimate.tanPhi, golden.estimate.tan_phi, accuracy: 1e-9)
        XCTAssertEqual(estimate.phiStd, golden.estimate.phi_std, accuracy: 1e-9)
        XCTAssertEqual(estimate.validFraction, golden.estimate.valid_fraction, accuracy: 1e-12)
        XCTAssertEqual(estimate.meanCoherency, golden.estimate.mean_coherency, accuracy: 1e-9)
        XCTAssertEqual(estimate.usedCount, golden.estimate.n_used)

        // The histogram is where a bin-edge convention hides: numpy's bins are
        // half-open except the last, and counting the other way moves a handful
        // of windows between adjacent bins without changing the mean much.
        XCTAssertEqual(estimate.histogram.count, golden.estimate.hist.count)
        for (i, expected) in golden.estimate.hist.enumerated() {
            XCTAssertEqual(estimate.histogram[i], expected, accuracy: 1e-9, "hist bin \(i)")
        }
        for (i, expected) in golden.estimate.bin_centres.enumerated() {
            XCTAssertEqual(estimate.binCentres[i], expected, accuracy: 1e-12, "bin centre \(i)")
        }
    }

    func testTheFixtureItselfRecoversItsKnownSlope() throws {
        let (golden, sti) = try skipUnlessPresent()
        let field = try rebuildField(golden, sti)
        let estimate = try meanOrientationAngle(
            field: field,
            phiMin: golden.parameters.phi_min,
            phiMax: golden.parameters.phi_max,
            coherencyLevels: golden.parameters.coherency_levels,
            binCount: golden.parameters.n_bins,
            minCoherency: golden.parameters.min_coherency)
        // Agreeing with the Python is necessary but not sufficient: both could
        // be wrong together.  The fixture's slope is known by construction.
        XCTAssertEqual(estimate.tanPhi, golden.parameters.tan_phi_truth,
                       accuracy: 0.1 * golden.parameters.tan_phi_truth)
    }
}
