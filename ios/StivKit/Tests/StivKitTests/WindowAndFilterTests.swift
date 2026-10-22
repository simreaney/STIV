//  WindowAndFilterTests.swift
//  The window grid's rounding, the fast filters' exactness, and parameters that
//  used to trap.

import XCTest
@testable import StivKit

/// `int(round(x))` as Python computes it: halves go to the even neighbour.
/// Written out longhand rather than with `.toNearestOrEven`, so the test does
/// not share the implementation's choice of rounding rule.
private func pythonRound(_ x: Double) -> Int {
    let lower = x.rounded(.down)
    let fraction = x - lower
    if fraction < 0.5 { return Int(lower) }
    if fraction > 0.5 { return Int(lower) + 1 }
    return Int(lower) % 2 == 0 ? Int(lower) : Int(lower) + 1
}

/// `np.arange(margin, n - margin, step)`.
private func arange(_ start: Int, _ stop: Int, _ step: Int) -> [Int] {
    Array(stride(from: start, to: stop, by: step))
}

/// The original per-pixel separable Gaussian, kept as a reference: plain
/// loops, a clamp on every tap, and nothing clever.
private func naiveGaussian(_ m: Matrix, sigma: Double, truncate: Double = 4.0) -> Matrix {
    let radius = Int(truncate * sigma + 0.5)
    var kernel = (-radius...radius).map { exp(-Double($0 * $0) / (2.0 * sigma * sigma)) }
    var total = 0.0
    for w in kernel { total += w }
    for i in kernel.indices { kernel[i] /= total }

    var pass1 = Matrix(rows: m.rows, cols: m.cols)
    for r in 0..<m.rows {
        for c in 0..<m.cols {
            var acc = 0.0
            for (k, w) in kernel.enumerated() {
                acc += w * m[r, Matrix.clampIndex(c + k - radius, m.cols)]
            }
            pass1[r, c] = acc
        }
    }
    var out = Matrix(rows: m.rows, cols: m.cols)
    for c in 0..<m.cols {
        for r in 0..<m.rows {
            var acc = 0.0
            for (k, w) in kernel.enumerated() {
                acc += w * pass1[Matrix.clampIndex(r + k - radius, m.rows), c]
            }
            out[r, c] = acc
        }
    }
    return out
}

/// The original central difference: numpy's reflect rule on every tap.
private func naiveCentralDifference(_ m: Matrix, axis: Axis, order: GradientOrder) -> Matrix {
    let k = order.stencil
    let pad = order.pad
    var out = Matrix(rows: m.rows, cols: m.cols)
    for r in 0..<m.rows {
        for c in 0..<m.cols {
            var acc = 0.0
            for (j, w) in k.enumerated() where w != 0.0 {
                if axis == .space {
                    acc += w * m[r, Matrix.reflectIndex(c + j - pad, m.cols)]
                } else {
                    acc += w * m[Matrix.reflectIndex(r + j - pad, m.rows), c]
                }
            }
            out[r, c] = acc
        }
    }
    return out
}

private func randomMatrix(rows: Int, cols: Int, seed: UInt64) -> Matrix {
    var rng = DeterministicNormals(seed: seed)
    var m = Matrix(rows: rows, cols: cols)
    for r in 0..<rows { for c in 0..<cols { m[r, c] = rng.next() } }
    return m
}

final class WindowGridTests: XCTestCase {
    /// At 50% overlap a window of 4k + 1 pixels steps by exactly 2k + 0.5.
    /// Python rounds that half to even and the port used to round it up, so
    /// for the golden STI's 187 x 143 a 29 px box gave 11 x 8 windows where
    /// the desktop has 12 x 8.
    func testHalfwayStepsRoundToEvenAsPythonDoes() throws {
        let sti = syntheticSTI(tanPhi: 1.35, rows: 187, cols: 143, seed: 4)
        for windowPx in [21, 29] {
            let field = try structureTensorField(sti: sti, windowPx: windowPx, windowType: .box,
                                                 overlap: 0.5, gradientOrder: .four)
            let margin = windowPx / 2 + 2
            let step = pythonRound(Double(windowPx) * 0.5)
            XCTAssertEqual(field.centresT, arange(margin, 187 - margin, step), "\(windowPx) px")
            XCTAssertEqual(field.centresX, arange(margin, 143 - margin, step), "\(windowPx) px")
        }

        // And the literal values, from `stiv.orientation.structure_tensor_field`.
        let f29 = try structureTensorField(sti: sti, windowPx: 29, overlap: 0.5)
        XCTAssertEqual([f29.phi.rows, f29.phi.cols], [12, 8])
        XCTAssertEqual(Array(f29.centresT.prefix(3)), [16, 30, 44])
        let f21 = try structureTensorField(sti: sti, windowPx: 21, overlap: 0.5)
        XCTAssertEqual([f21.phi.rows, f21.phi.cols], [17, 12])
        XCTAssertEqual(Array(f21.centresT.prefix(3)), [12, 22, 32])
    }

    func testWindowStepMatchesPythonsRoundAcrossTheSettingsRange() {
        for windowPx in stride(from: 5, through: 81, by: 2) {
            for overlap in [0.0, 0.25, 0.3, 0.5, 0.75] {
                XCTAssertEqual(windowStep(windowPx: windowPx, overlap: overlap),
                               max(1, pythonRound(Double(windowPx) * (1 - overlap))),
                               "\(windowPx) px at overlap \(overlap)")
            }
        }
        // Absurd overlaps clamp instead of overflowing `Int(_:)`.
        XCTAssertEqual(windowStep(windowPx: 31, overlap: 5), 1)
        XCTAssertGreaterThan(windowStep(windowPx: 31, overlap: -1e300), 1)
    }
}

final class FastFilterTests: XCTestCase {
    /// The boundary-hoisted filter must be the per-pixel loop, bit for bit:
    /// the golden fixture compares to 1e-9, but nothing short of identity
    /// guarantees that every future fixture will agree too.
    func testGaussianMeanIsBitIdenticalToThePerPixelLoop() {
        for (rows, cols, sigma) in [(37, 29, 1.5), (60, 23, 7.75), (12, 90, 4.0), (9, 9, 6.0)] {
            let m = randomMatrix(rows: rows, cols: cols, seed: UInt64(rows * cols))
            let fast = m.gaussianMean(sigma: sigma)
            let slow = naiveGaussian(m, sigma: sigma)
            XCTAssertEqual(fast.data.map(\.bitPattern), slow.data.map(\.bitPattern),
                           "\(rows)x\(cols), sigma \(sigma)")
        }
    }

    /// Evaluating only at the window centres changes no bit either, including
    /// centres within a kernel radius of the edge.
    func testSampledGaussianIsBitIdenticalToTheFullFilter() {
        let m = randomMatrix(rows: 140, cols: 97, seed: 5)
        for sigma in [7.25, 5.25, 2.0] {
            let rows = [0, 3, 17, 40, 70, 121, 139]
            let cols = [0, 1, 30, 48, 90, 96]
            let full = naiveGaussian(m, sigma: sigma)
            let sampled = m.gaussianMean(sigma: sigma, rowIndices: rows, colIndices: cols)
            for (i, r) in rows.enumerated() {
                for (j, c) in cols.enumerated() {
                    XCTAssertEqual(sampled[i, j].bitPattern, full[r, c].bitPattern,
                                   "sigma \(sigma) at (\(r), \(c))")
                }
            }
        }
    }

    /// The Gaussian structure tensor through the fast path equals the field
    /// computed the old way: full-resolution filtering, then sampling.
    func testGaussianFieldMatchesFullResolutionFiltering() throws {
        let sti = syntheticSTI(tanPhi: 0.8, rows: 150, cols: 120, noise: 0.2, seed: 9)
        for windowPx in [9, 29] {
            let field = try structureTensorField(sti: sti, windowPx: windowPx, windowType: .gaussian)
            let gx = naiveCentralDifference(sti, axis: .space, order: .four)
            let gt = naiveCentralDifference(sti, axis: .time, order: .four)
            let sigma = Double(windowPx) / 4
            let jxx = naiveGaussian(gx.multiplied(by: gx), sigma: sigma)
            let jxt = naiveGaussian(gx.multiplied(by: gt), sigma: sigma)
            let jtt = naiveGaussian(gt.multiplied(by: gt), sigma: sigma)
            for (i, r) in field.centresT.enumerated() {
                for (j, c) in field.centresX.enumerated() {
                    let phi = 0.5 * atan2(-2.0 * jxt[r, c], jxx[r, c] - jtt[r, c])
                    XCTAssertEqual(field.phi[i, j].bitPattern, phi.bitPattern, "\(windowPx) px")
                    XCTAssertEqual(field.energy[i, j].bitPattern, (jxx[r, c] + jtt[r, c]).bitPattern)
                }
            }
        }
    }

    func testCentralDifferenceIsBitIdenticalToTheReflectLoop() {
        let m = randomMatrix(rows: 31, cols: 26, seed: 21)
        for order in GradientOrder.allCases {
            for axis in [Axis.space, .time] {
                XCTAssertEqual(centralDifference(m, axis: axis, order: order).data.map(\.bitPattern),
                               naiveCentralDifference(m, axis: axis, order: order).data.map(\.bitPattern),
                               "order \(order.rawValue), \(axis)")
            }
        }
    }
}

final class ParameterGuardTests: XCTestCase {
    private let field = try! structureTensorField(sti: syntheticSTI(tanPhi: 1.0, rows: 120, cols: 120))

    /// `binIndex` returned -1 for zero bins and `hist[-1]` trapped.
    func testZeroOrNegativeBinsThrowRatherThanTrap() {
        for bins in [0, -3] {
            XCTAssertThrowsError(try meanOrientationAngle(field: field, phiMin: -1, phiMax: 1,
                                                          binCount: bins)) { error in
                XCTAssertTrue("\(error)".contains("bin"), "\(error)")
            }
        }
    }

    /// Infinite bounds give NaN bin edges, and `Int(NaN)` traps.
    func testNonFiniteAngleRangeThrows() {
        XCTAssertThrowsError(try meanOrientationAngle(field: field, phiMin: -.infinity, phiMax: 1))
        XCTAssertThrowsError(try meanOrientationAngle(field: field, phiMin: .nan, phiMax: 1))
    }

    func testDegenerateWindowsThrowRatherThanTrap() {
        let sti = syntheticSTI(tanPhi: 1.0, rows: 80, cols: 80)
        XCTAssertThrowsError(try structureTensorField(sti: sti, windowPx: 0))
        XCTAssertThrowsError(try structureTensorField(sti: sti, windowPx: -5, windowType: .gaussian))
        XCTAssertThrowsError(try structureTensorField(sti: sti, windowPx: 15, overlap: .nan))
        XCTAssertThrowsError(try structureTensorField(sti: sti, windowPx: 15, overlap: -.infinity))
    }

    func testANonFiniteSmoothingSigmaIsNotApplied() {
        var settings = StiSettings()
        settings.smoothSigma = .infinity
        let out = preprocess(syntheticSTI(tanPhi: 1.0, rows: 40, cols: 40), settings: settings)
        XCTAssertTrue(out.data.allSatisfy(\.isFinite))
    }
}
