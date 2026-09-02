//  OrientationTests.swift
//  The load-bearing tests.  Mirrors `tests/test_orientation.py`.
//
//  A port of a structure-tensor implementation can fail in three ways that all
//  still produce plausible-looking numbers: the eigenvector branch can be
//  inverted (giving -1/tan phi), a gradient sign can be flipped (giving -tan
//  phi), or the two axes can be transposed (giving 1/tan phi).  Every test
//  below exists to catch one of those.

import XCTest
@testable import StivKit

final class GradientTests: XCTestCase {
    func testStencilsAreExactOnALinearRamp() throws {
        // A central difference of any order must return the exact slope.
        for order in GradientOrder.allCases {
            var ramp = Matrix(rows: 40, cols: 60)
            for r in 0..<40 { for c in 0..<60 { ramp[r, c] = Double(c) * 3.0 } }
            let gx = centralDifference(ramp, axis: .space, order: order)
            for r in 0..<40 {
                for c in order.rawValue..<(60 - order.rawValue) {
                    XCTAssertEqual(gx[r, c], 3.0, accuracy: 1e-9,
                                   "order \(order.rawValue) at (\(r), \(c))")
                }
            }
        }
    }

    func testConstantFieldHasZeroGradient() {
        for order in GradientOrder.allCases {
            let flat = Matrix(rows: 30, cols: 30, repeating: 7.0)
            for v in centralDifference(flat, axis: .time, order: order).data {
                XCTAssertEqual(v, 0, accuracy: 1e-12)
            }
            for v in centralDifference(flat, axis: .space, order: order).data {
                XCTAssertEqual(v, 0, accuracy: 1e-12)
            }
        }
    }

    func testTimeAndSpaceAxesAreNotInterchangeable() {
        // Guards against a transposed port: a ramp in time must have zero
        // spatial gradient and vice versa.
        var ramp = Matrix(rows: 40, cols: 40)
        for r in 0..<40 { for c in 0..<40 { ramp[r, c] = Double(r) * 2.0 } }
        let gt = centralDifference(ramp, axis: .time)
        let gx = centralDifference(ramp, axis: .space)
        XCTAssertEqual(gt[20, 20], 2.0, accuracy: 1e-9)
        XCTAssertEqual(gx[20, 20], 0.0, accuracy: 1e-9)
    }
}

final class OrientationTests: XCTestCase {
    /// Reproduces Fig. 7: accurate across the validated tan-phi range.
    func testRecoversKnownInclination() throws {
        for tanPhi in [0.1, 0.3, 0.5, 1.0, 2.0, 3.0] {
            let field = try structureTensorField(sti: syntheticSTI(tanPhi: tanPhi))
            let est = try meanOrientationAngle(field: field,
                                               phiMin: fullAngleRange.min,
                                               phiMax: fullAngleRange.max)
            XCTAssertEqual(est.tanPhi, tanPhi, accuracy: 0.10 * tanPhi + 0.02,
                           "tan phi = \(tanPhi)")
        }
    }

    /// A pattern moving upstream must give a negative angle, not |angle|.
    ///
    /// This is the branch Eq. (6) leaves ambiguous: reading the major
    /// eigenvector instead of the minor one returns -1/tan(phi), which has the
    /// wrong sign for half the cases and the wrong magnitude for all of them.
    func testSignFollowsDirectionOfTravel() throws {
        for tanPhi in [-2.0, -0.5, 0.5, 2.0] {
            let field = try structureTensorField(sti: syntheticSTI(tanPhi: tanPhi))
            let est = try meanOrientationAngle(field: field,
                                               phiMin: fullAngleRange.min,
                                               phiMax: fullAngleRange.max)
            XCTAssertEqual(est.tanPhi.sign, tanPhi.sign, "tan phi = \(tanPhi)")
            XCTAssertEqual(est.tanPhi, tanPhi, accuracy: 0.10 * abs(tanPhi) + 0.02)
        }
    }

    /// The specific wrong answer a mis-read eigenvector produces.
    ///
    /// Pinning it explicitly is worth more than pinning the right answer alone:
    /// if someone "simplifies" the branch back to Eq. (6) as printed, this test
    /// says exactly what went wrong instead of just failing.
    func testDoesNotReturnTheGradientDirection() throws {
        let tanPhi = 2.0
        let field = try structureTensorField(sti: syntheticSTI(tanPhi: tanPhi))
        let est = try meanOrientationAngle(field: field,
                                           phiMin: fullAngleRange.min,
                                           phiMax: fullAngleRange.max)
        XCTAssertFalse(abs(est.tanPhi - (-1.0 / tanPhi)) < 0.1,
                       "returned the major eigenvector (gradient) branch, not the streak")
        XCTAssertFalse(abs(est.tanPhi - (1.0 / tanPhi)) < 0.1,
                       "the space and time axes look transposed")
    }

    func testStationaryPatternReadsZero() throws {
        let field = try structureTensorField(sti: syntheticSTI(tanPhi: 0.0))
        let est = try meanOrientationAngle(field: field,
                                           phiMin: fullAngleRange.min,
                                           phiMax: fullAngleRange.max)
        XCTAssertEqual(est.tanPhi, 0.0, accuracy: 0.03)
    }

    func testCoherencyIsHighForAStreakAndLowForNoise() throws {
        let streak = try structureTensorField(sti: syntheticSTI(tanPhi: 1.0))
        var rng = DeterministicNormals(seed: 7)
        var noise = Matrix(rows: 200, cols: 200)
        for i in noise.data.indices { noise.data[i] = rng.next() }
        let isotropic = try structureTensorField(sti: noise)

        let streakMean = streak.coherency.data.reduce(0, +) / Double(streak.windowCount)
        let noiseMean = isotropic.coherency.data.reduce(0, +) / Double(isotropic.windowCount)
        XCTAssertGreaterThan(streakMean, 0.7)
        XCTAssertLessThan(noiseMean, 0.4)
        XCTAssertGreaterThan(streakMean, noiseMean + 0.4)
    }

    func testSurvivesNoise() throws {
        let field = try structureTensorField(sti: syntheticSTI(tanPhi: 1.0, noise: 0.5, seed: 3))
        let est = try meanOrientationAngle(field: field,
                                           phiMin: fullAngleRange.min,
                                           phiMax: fullAngleRange.max)
        XCTAssertEqual(est.tanPhi, 1.0, accuracy: 0.15)
    }

    func testTooSmallStiIsRejectedWithAnActionableMessage() {
        XCTAssertThrowsError(try structureTensorField(sti: Matrix(rows: 20, cols: 20))) { error in
            XCTAssertTrue("\(error)".contains("too small"), "got: \(error)")
        }
    }

    func testEmptyHistogramReturnsNotANumberRatherThanZero() throws {
        // A range that excludes every window must report "no answer", never a
        // confident zero -- the quality flags depend on telling those apart.
        let field = try structureTensorField(sti: syntheticSTI(tanPhi: 1.0))
        let est = try meanOrientationAngle(field: field, phiMin: -1.5, phiMax: -1.4)
        XCTAssertTrue(est.phi.isNaN)
        XCTAssertEqual(est.usedCount, 0)
    }

    func testAngleRangeInvertsEquationSixteen() {
        let sx = 0.02, st = 1.0 / 30.0
        let range = angleRange(forVelocities: -0.5, 8.0, sxMetresPerColumn: sx, stSecondsPerRow: st)
        XCTAssertEqual(velocity(fromAngle: range.min, sxMetresPerColumn: sx, stSecondsPerRow: st),
                       -0.5, accuracy: 1e-9)
        XCTAssertEqual(velocity(fromAngle: range.max, sxMetresPerColumn: sx, stSecondsPerRow: st),
                       8.0, accuracy: 1e-9)
    }

    func testVelocityIsOddInPhi() {
        let sx = 0.02, st = 1.0 / 30.0
        let up = velocity(fromAngle: 0.4, sxMetresPerColumn: sx, stSecondsPerRow: st)
        let down = velocity(fromAngle: -0.4, sxMetresPerColumn: sx, stSecondsPerRow: st)
        XCTAssertEqual(up, -down, accuracy: 1e-12)
    }

    func testPeakEstimatorAgreesWithTheMeanOnACleanStreak() throws {
        let field = try structureTensorField(sti: syntheticSTI(tanPhi: 1.0))
        let mean = try meanOrientationAngle(field: field, phiMin: fullAngleRange.min,
                                           phiMax: fullAngleRange.max, estimator: .eq14)
        let peak = try meanOrientationAngle(field: field, phiMin: fullAngleRange.min,
                                            phiMax: fullAngleRange.max, estimator: .peak)
        XCTAssertEqual(mean.phi, peak.phi, accuracy: 0.05)
    }
}

final class MatrixTests: XCTestCase {
    func testBoxMeanMatchesTheNaiveDefinition() {
        var rng = DeterministicNormals(seed: 11)
        var m = Matrix(rows: 23, cols: 19)
        for i in m.data.indices { m.data[i] = rng.next() }

        let size = 5
        let fast = m.boxMean(size: size)
        let before = size / 2, after = size - size / 2 - 1

        for r in 0..<m.rows {
            for c in 0..<m.cols {
                var total = 0.0
                for dr in -before...after {
                    for dc in -before...after {
                        total += m[Matrix.clampIndex(r + dr, m.rows),
                                   Matrix.clampIndex(c + dc, m.cols)]
                    }
                }
                XCTAssertEqual(fast[r, c], total / Double(size * size), accuracy: 1e-10)
            }
        }
    }

    func testBoxMeanOfAConstantIsTheConstant() {
        let flat = Matrix(rows: 40, cols: 40, repeating: 3.5)
        for v in flat.boxMean(size: 31).data {
            XCTAssertEqual(v, 3.5, accuracy: 1e-12)
        }
    }

    func testGaussianMeanPreservesADcLevel() {
        let flat = Matrix(rows: 40, cols: 40, repeating: -2.0)
        for v in flat.gaussianMean(sigma: 4.0).data {
            XCTAssertEqual(v, -2.0, accuracy: 1e-12)
        }
    }

    /// numpy's `reflect`: the edge sample is not repeated.
    func testReflectIndexDoesNotRepeatTheEdge() {
        let n = 4  // [a b c d] padded by 2 -> [c b | a b c d | c b]
        XCTAssertEqual(Matrix.reflectIndex(-1, n), 1)
        XCTAssertEqual(Matrix.reflectIndex(-2, n), 2)
        XCTAssertEqual(Matrix.reflectIndex(0, n), 0)
        XCTAssertEqual(Matrix.reflectIndex(3, n), 3)
        XCTAssertEqual(Matrix.reflectIndex(4, n), 2)
        XCTAssertEqual(Matrix.reflectIndex(5, n), 1)
        XCTAssertEqual(Matrix.reflectIndex(1, 1), 0)
    }

    func testClampIndex() {
        XCTAssertEqual(Matrix.clampIndex(-5, 4), 0)
        XCTAssertEqual(Matrix.clampIndex(9, 4), 3)
        XCTAssertEqual(Matrix.clampIndex(2, 4), 2)
    }
}
