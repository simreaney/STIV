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
//  that has never seen numpy.  The later sections of the fixture (the Gaussian
//  window, the rounding-sensitive window grids and the analysis stage) are
//  optional in the same way: an older fixture lacks them, and their tests skip
//  until it is regenerated.
//
//  `STIV_GOLDEN_FIXTURE_DIR` points the tests at another directory, so a
//  fixture written with `make_golden_fixture.py --out DIR` can be tried before
//  it replaces the committed one.

import XCTest
@testable import StivKit

/// The fixture as `make_golden_fixture.py` writes it.
struct Golden: Decodable {
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
        // Present on the Gaussian field and the window grids, which carry
        // their own parameters; the original box field uses `parameters`.
        let window_px: Int?
        let window_type: String?
        let overlap: Double?
        let gradient_order: Int?
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
        let estimate: Estimate?
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
    // Newer sections; nil in a fixture generated before they existed.  The
    // analysis-stage section is decoded on its own, by `GoldenAnalysisTests`,
    // so that a change to its schema cannot break the tests above.
    let gaussian_field: Field?
    let window_grids: [Field]?
    let flat_windows: FlatWindows?
    let peak_estimate: Estimate?
    let preprocess_missing: PreprocessMissing?

    /// `preprocess` on the fixture STI with NaN at `cells` and down all of
    /// `column`, once per settings in `runs`.
    struct PreprocessMissing: Decodable {
        struct Run: Decodable {
            let settings: Preprocess.Settings
            let mean: Double
            let std: Double
            let min: Double
            let max: Double
            let probes: [Preprocess.Cell]
        }
        let cells: [[Int]]
        let column: Int
        let runs: [Run]
    }

    struct FlatWindows: Decodable {
        let flat_cols: Int
        let fill: Double
        let window_px: Int
        let window_type: String
        let overlap: Double
        let gradient_order: Int
        let min_coherency: Double
        let rel_energy: Double
        let n_flat: Int
        let flat: [Bool]
        let estimate: Estimate
    }


    /// `golden.json`, from `STIV_GOLDEN_FIXTURE_DIR` if set and otherwise from
    /// the test bundle, or `nil` when there is none.
    static func jsonURL() -> URL? {
        if let dir = ProcessInfo.processInfo.environment["STIV_GOLDEN_FIXTURE_DIR"], !dir.isEmpty {
            let url = URL(fileURLWithPath: dir).appendingPathComponent("golden.json")
            return FileManager.default.fileExists(atPath: url.path) ? url : nil
        }
        let bundle = Bundle.module
        return bundle.url(forResource: "golden", withExtension: "json", subdirectory: "Fixtures")
            ?? bundle.url(forResource: "golden", withExtension: "json")
    }

    /// Both fixture files, or `nil` when the generator has not been run.
    static func load() throws -> (Golden, Matrix)? {
        guard let jsonURL = jsonURL() else { return nil }
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

    static func skipUnlessPresent() throws -> (Golden, Matrix) {
        guard let fixture = try load() else {
            throw XCTSkip("""
                No golden fixture. Generate it on a machine with the Python package \
                installed: `python ios/tools/make_golden_fixture.py`.
                """)
        }
        return fixture
    }

    /// A section added to the fixture after it was first committed.
    static func require<T>(_ section: T?, _ name: String) throws -> T {
        guard let section else {
            throw XCTSkip("""
                The golden fixture predates its `\(name)` section; regenerate it with \
                `python ios/tools/make_golden_fixture.py`.
                """)
        }
        return section
    }
}

final class GoldenFixtureTests: XCTestCase {
    private func skipUnlessPresent() throws -> (Golden, Matrix) {
        try Golden.skipUnlessPresent()
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

    // MARK: - newer sections

    /// Compare a field against its fixture summary, to the same 1e-9.
    private func assertField(_ field: OrientationField, matches want: Golden.Field,
                             _ label: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual([field.phi.rows, field.phi.cols], want.shape, "\(label) shape", file: file, line: line)
        XCTAssertEqual(field.centresT, want.centres_t, "\(label) centresT", file: file, line: line)
        XCTAssertEqual(field.centresX, want.centres_x, "\(label) centresX", file: file, line: line)
        guard [field.phi.rows, field.phi.cols] == want.shape else { return }

        let n = Double(field.phi.count)
        XCTAssertEqual(field.phi.data.reduce(0, +) / n, want.phi_mean, accuracy: 1e-9, label, file: file, line: line)
        XCTAssertEqual(field.phi.data.min() ?? .nan, want.phi_min, accuracy: 1e-9, label, file: file, line: line)
        XCTAssertEqual(field.phi.data.max() ?? .nan, want.phi_max, accuracy: 1e-9, label, file: file, line: line)
        XCTAssertEqual(field.coherency.data.reduce(0, +) / n, want.coherency_mean, accuracy: 1e-9,
                       label, file: file, line: line)
        XCTAssertEqual(field.coherency.data.min() ?? .nan, want.coherency_min, accuracy: 1e-9,
                       label, file: file, line: line)
        XCTAssertEqual(field.coherency.data.max() ?? .nan, want.coherency_max, accuracy: 1e-9,
                       label, file: file, line: line)
        XCTAssertEqual(field.energy.data.reduce(0, +) / n / want.energy_mean, 1.0, accuracy: 1e-9,
                       label, file: file, line: line)
        for probe in want.probes {
            XCTAssertEqual(field.phi[probe.i, probe.j], probe.phi, accuracy: 1e-9,
                           "\(label) phi at (\(probe.i), \(probe.j))", file: file, line: line)
            XCTAssertEqual(field.coherency[probe.i, probe.j], probe.coherency, accuracy: 1e-9,
                           "\(label) coherency at (\(probe.i), \(probe.j))", file: file, line: line)
        }
    }

    private func rebuild(_ want: Golden.Field, _ sti: Matrix) throws -> OrientationField {
        try structureTensorField(
            sti: sti,
            windowPx: try XCTUnwrap(want.window_px),
            windowType: try XCTUnwrap(WindowType(rawValue: try XCTUnwrap(want.window_type))),
            overlap: try XCTUnwrap(want.overlap),
            gradientOrder: try XCTUnwrap(GradientOrder(rawValue: try XCTUnwrap(want.gradient_order))))
    }

    /// The Gaussian window, which the port evaluates only at the window
    /// centres: twice the box's support, and a different code path entirely.
    func testGaussianWindowMatchesThePython() throws {
        let (golden, sti) = try skipUnlessPresent()
        let want = try Golden.require(golden.gaussian_field, "gaussian_field")
        let field = try rebuild(want, sti)
        assertField(field, matches: want, "gaussian")

        let expected = try XCTUnwrap(want.estimate)
        let estimate = try meanOrientationAngle(
            field: field,
            phiMin: golden.parameters.phi_min,
            phiMax: golden.parameters.phi_max,
            coherencyLevels: golden.parameters.coherency_levels,
            binCount: golden.parameters.n_bins,
            minCoherency: golden.parameters.min_coherency)
        XCTAssertEqual(estimate.phi, expected.phi, accuracy: 1e-9)
        XCTAssertEqual(estimate.phiStd, expected.phi_std, accuracy: 1e-9)
        XCTAssertEqual(estimate.usedCount, expected.n_used)
        XCTAssertEqual(estimate.histogram, expected.hist)
    }

    /// Window sizes whose 50% step lands on a half, where Python's
    /// round-half-to-even and Swift's default round-half-up disagree.
    func testRoundingSensitiveWindowGridsMatchThePython() throws {
        let (golden, sti) = try skipUnlessPresent()
        let grids = try Golden.require(golden.window_grids, "window_grids")
        XCTAssertFalse(grids.isEmpty)
        for want in grids {
            let label = "\(want.window_type ?? "?") \(want.window_px ?? 0) px"
            assertField(try rebuild(want, sti), matches: want, label)
        }
    }

    /// Featureless windows: Eq. (6) gives them phi = 0, and without
    /// `flatWindows` they vote for zero velocity.  The fixture holds a block of
    /// the STI constant, so a third of the windows have exactly no texture,
    /// and asks for `minCoherency` 0 so that nothing else would exclude them.
    func testFeaturelessWindowsAreLeftOutAsInThePython() throws {
        let (golden, sti) = try skipUnlessPresent()
        let want = try Golden.require(golden.flat_windows, "flat_windows")
        XCTAssertEqual(flatWindowRelativeEnergy, want.rel_energy)

        var flatSTI = sti
        for r in 0..<sti.rows {
            for c in 0..<want.flat_cols { flatSTI[r, c] = want.fill }
        }
        let field = try structureTensorField(
            sti: flatSTI,
            windowPx: want.window_px,
            windowType: try XCTUnwrap(WindowType(rawValue: want.window_type)),
            overlap: want.overlap,
            gradientOrder: try XCTUnwrap(GradientOrder(rawValue: want.gradient_order)))
        let flat = flatWindows(field)
        XCTAssertEqual(flat, want.flat)
        XCTAssertEqual(flat.filter { $0 }.count, want.n_flat)
        XCTAssertGreaterThan(want.n_flat, 0, "the fixture must actually contain flat windows")

        let estimate = try meanOrientationAngle(
            field: field,
            phiMin: golden.parameters.phi_min,
            phiMax: golden.parameters.phi_max,
            coherencyLevels: golden.parameters.coherency_levels,
            binCount: golden.parameters.n_bins,
            minCoherency: want.min_coherency)
        let expected = want.estimate
        XCTAssertEqual(estimate.phi, expected.phi, accuracy: 1e-9)
        XCTAssertEqual(estimate.tanPhi, expected.tan_phi, accuracy: 1e-9)
        XCTAssertEqual(estimate.phiStd, expected.phi_std, accuracy: 1e-9)
        XCTAssertEqual(estimate.validFraction, expected.valid_fraction, accuracy: 1e-12)
        XCTAssertEqual(estimate.meanCoherency, expected.mean_coherency, accuracy: 1e-9)
        XCTAssertEqual(estimate.usedCount, expected.n_used)
        XCTAssertEqual(estimate.histogram, expected.hist)
    }

    /// The histogram-peak estimator, the app's other "Angle estimator".  Its
    /// parabolic sub-bin step is a sign away from moving phi by up to two bins
    /// -- several per cent of the velocity -- which the synthetic-slope
    /// tolerances in OrientationTests cannot see.
    func testPeakEstimatorMatchesThePython() throws {
        let (golden, sti) = try skipUnlessPresent()
        let want = try Golden.require(golden.peak_estimate, "peak_estimate")
        let estimate = try meanOrientationAngle(
            field: try rebuildField(golden, sti),
            phiMin: golden.parameters.phi_min,
            phiMax: golden.parameters.phi_max,
            coherencyLevels: golden.parameters.coherency_levels,
            binCount: golden.parameters.n_bins,
            minCoherency: golden.parameters.min_coherency,
            estimator: .peak)
        XCTAssertEqual(estimate.phi, want.phi, accuracy: 1e-9)
        XCTAssertEqual(estimate.tanPhi, want.tan_phi, accuracy: 1e-9)
        XCTAssertEqual(estimate.phiStd, want.phi_std, accuracy: 1e-9)
        XCTAssertEqual(estimate.usedCount, want.n_used)
        XCTAssertEqual(estimate.histogram, want.hist)
    }

    /// Samples that left the frame: every search line touching the frame edge
    /// has them.  The first run conditions with every other stage off, so its
    /// output is the fill itself -- a scattered gap takes its column's mean, an
    /// all-NaN column the grand mean; the second is the pipeline as run.
    func testMissingSamplesAreFilledAsInThePython() throws {
        let (golden, sti) = try skipUnlessPresent()
        let want = try Golden.require(golden.preprocess_missing, "preprocess_missing")
        var holed = sti
        for cell in want.cells { holed[cell[0], cell[1]] = .nan }
        for r in 0..<sti.rows { holed[r, want.column] = .nan }
        XCTAssertFalse(want.runs.isEmpty)

        for run in want.runs {
            XCTAssertFalse(run.settings.clahe, "the Swift port has no CLAHE")
            var settings = StiSettings()
            settings.removeStaticBackground = run.settings.remove_static_background
            settings.normalise = run.settings.normalise
            settings.smoothSigma = run.settings.smooth_sigma
            let got = StivKit.preprocess(holed, settings: settings)
            let label = "background \(run.settings.remove_static_background), "
                + "sigma \(run.settings.smooth_sigma)"

            XCTAssertFalse(got.data.contains { !$0.isFinite }, label)
            let (mean, std) = meanAndStd(got.data)
            XCTAssertEqual(mean, run.mean, accuracy: 1e-9, label)
            XCTAssertEqual(std, run.std, accuracy: 1e-9, label)
            XCTAssertEqual(got.data.min() ?? .nan, run.min, accuracy: 1e-9, label)
            XCTAssertEqual(got.data.max() ?? .nan, run.max, accuracy: 1e-9, label)
            for cell in run.probes {
                XCTAssertEqual(got[cell.r, cell.c], cell.value, accuracy: 1e-9,
                               "\(label): cell (\(cell.r), \(cell.c))")
            }
        }
    }
}
