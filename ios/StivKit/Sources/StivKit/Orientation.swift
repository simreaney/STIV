//  Orientation.swift
//  Structure-tensor orientation analysis of a space-time image.
//
//  A direct port of `stiv/orientation.py` -- section 2.2 of Fujita, Watanabe &
//  Tsubaki (2007).  A pattern advected along the search line at speed `U` traces
//  a straight streak in the space-time image (STI); the streak's inclination
//  `phi` from the time axis carries the velocity, via Eq. (16),
//  `U = (Sx / St) tan phi`.
//
//  The orientation is recovered by maximising
//
//      integral w(x - x') (grad g(x')^T n)^2 dx' = n^T J n            (Eq. 2)
//
//  over unit vectors `n`, where `J` is the structure tensor of Eqs. (3), (4)
//  and (7)-(9).  Diagonalising `J` gives the orientation angle of Eq. (6) and
//  the coherency of Eq. (11),
//
//      C = sqrt((Jtt - Jxx)^2 + 4 Jxt^2) / (Jxx + Jtt)
//        = (lambda1 - lambda2) / (lambda1 + lambda2)
//
//  which is 1 for a perfectly oriented local pattern and 0 for an isotropic
//  one.  Coherency is what makes the method robust: section 2.4 weights the
//  angle average by it, so clean streaks dominate and noisy patches are ignored.
//
//  Nothing in this file should be "improved" without a reason that survives the
//  tests.  Two places in particular are load-bearing and counter-intuitive:
//  the eigenvector branch in `structureTensorField` and the multi-threshold
//  histogram of `meanOrientationAngle`.

import Foundation

// MARK: - gradient stencils

/// Antisymmetric central-difference stencils.
///
/// The paper's Eqs. (12) and (13) are printed with reversed signs; what matters
/// for the structure tensor is that the *same* convention is used for d/dx and
/// d/dt, since flipping only one of them would flip the sign of `Jxt` and hence
/// of the velocity.  These are the standard forms, applied consistently to both
/// axes.
public enum GradientOrder: Int, CaseIterable, Codable, Sendable {
    /// Plain central difference.
    case two = 2
    /// The four-point stencil of Eq. (12); the default.
    case four = 4
    /// The six-point stencil of Eq. (13).
    case six = 6

    var stencil: [Double] {
        switch self {
        case .two:  return [-1.0, 0.0, 1.0].map { $0 / 2.0 }
        case .four: return [1.0, -8.0, 0.0, 8.0, -1.0].map { $0 / 12.0 }
        case .six:  return [-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0].map { $0 / 60.0 }
        }
    }

    /// Half-width of the stencil, i.e. the margin it makes unusable.
    var pad: Int { stencil.count / 2 }
}

/// The window function `w` of Eq. (3).
public enum WindowType: String, Codable, Sendable {
    /// Reproduces the paper's rectangular integration area `A`.
    case box
    /// Smoothly tapered over the same nominal support; better-conditioned tensor.
    case gaussian
}

// MARK: - results

/// Per-window orientation angles and coherencies over one STI.
public struct OrientationField {
    /// Radians at window centres, measured from the time axis toward the space
    /// axis, positive for downstream motion along the search line.
    public let phi: Matrix
    /// In `[0, 1]`; Eq. (11).
    public let coherency: Matrix
    /// Trace of `J`: gradient power. Used to spot dead (textureless) windows.
    public let energy: Matrix
    /// STI row index of each window centre.
    public let centresT: [Int]
    /// STI column index of each window centre.
    public let centresX: [Int]

    public var windowCount: Int { phi.count }
}

/// Result of the Eq. (14) weighted average over one STI.
public struct AngleEstimate {
    /// Radians; `nan` if nothing usable.
    public let phi: Double
    public let tanPhi: Double
    /// Coherency-weighted spread of in-range angles, radians.
    public let phiStd: Double
    /// Fraction of windows falling inside `[phiMin, phiMax]`.
    public let validFraction: Double
    /// Mean coherency of the in-range windows.
    public let meanCoherency: Double
    public let usedCount: Int
    /// Summed histogram of Fig. 11 and its bin centres, for diagnostics.
    public let histogram: [Double]
    public let binCentres: [Double]
    public let phiMin: Double
    public let phiMax: Double

    /// The "nothing usable" result, so callers never have to build one by hand.
    static func empty(
        validFraction: Double,
        histogram: [Double],
        binCentres: [Double],
        phiMin: Double,
        phiMax: Double
    ) -> AngleEstimate {
        AngleEstimate(
            phi: .nan, tanPhi: .nan, phiStd: .nan,
            validFraction: validFraction, meanCoherency: 0, usedCount: 0,
            histogram: histogram, binCentres: binCentres,
            phiMin: phiMin, phiMax: phiMax
        )
    }
}

/// Which reduction turns the per-window angles into one angle for the STI.
public enum AngleEstimator: String, Codable, Sendable {
    /// The paper's coherency-weighted mean, Eq. (14).
    case eq14
    /// Parabolically interpolated mode of the summed histogram; more robust
    /// when a strong secondary peak is present.
    case peak
}

/// Raised when an STI cannot support the requested analysis geometry.
public enum OrientationError: Error, CustomStringConvertible {
    case stiTooSmall(rows: Int, cols: Int, windowPx: Int, order: Int)
    case noWindowsFit(windowPx: Int)
    case badAngleRange(min: Double, max: Double)

    public var description: String {
        switch self {
        case let .stiTooSmall(rows, cols, windowPx, order):
            return "STI of shape \(rows)x\(cols) is too small for a \(windowPx)px window "
                 + "with an order-\(order) gradient; use more frames, a longer search "
                 + "line, or a smaller windowPx"
        case let .noWindowsFit(windowPx):
            return "no valid local windows fit inside the STI; reduce windowPx (\(windowPx))"
        case let .badAngleRange(min, max):
            return "need phiMin < phiMax, got \(min) and \(max)"
        }
    }
}

// MARK: - gradients and the structure tensor

/// Central-difference derivative along `axis` with the given stencil order.
///
/// Edges are handled by reflection; the affected margin is masked out later by
/// `structureTensorField`, which trims `order/2 + window/2` pixels from every
/// side.
public func centralDifference(_ img: Matrix, axis: Axis, order: GradientOrder = .four) -> Matrix {
    let k = order.stencil
    let pad = order.pad
    let n = axis == .space ? img.cols : img.rows
    let outer = axis == .space ? img.rows : img.cols

    var out = Matrix(rows: img.rows, cols: img.cols)
    for o in 0..<outer {
        for i in 0..<n {
            var acc = 0.0
            for (j, c) in k.enumerated() where c != 0.0 {
                let src = Matrix.reflectIndex(i + j - pad, n)
                acc += c * img.at(outer: o, inner: src, axis: axis)
            }
            out.set(outer: o, inner: i, axis: axis, value: acc)
        }
    }
    return out
}

/// Compute the orientation angle and coherency over a grid of local windows.
///
/// - Parameters:
///   - sti: Space-time image, `(time, space)`: rows are frames, columns are
///     positions along the search line.
///   - windowPx: Side length of the local integration area `A` of Eqs. (7)-(9).
///     The paper uses 31 x 31 pixels.
///   - overlap: Fractional overlap between neighbouring windows; 0.5 reproduces
///     the 50% overlap used in the paper.
///
/// The tensor components are evaluated everywhere by filtering `gx*gx`,
/// `gx*gt` and `gt*gt`, then sampled on the window grid.  That is algebraically
/// identical to stepping a window across the image, but far faster.
public func structureTensorField(
    sti: Matrix,
    windowPx: Int = 31,
    windowType: WindowType = .box,
    overlap: Double = 0.5,
    gradientOrder: GradientOrder = .four
) throws -> OrientationField {
    let nt = sti.rows
    let nx = sti.cols
    let half = windowPx / 2
    let margin = half + gradientOrder.pad
    guard nt > 2 * margin + 1, nx > 2 * margin + 1 else {
        throw OrientationError.stiTooSmall(
            rows: nt, cols: nx, windowPx: windowPx, order: gradientOrder.rawValue)
    }

    // x is the space axis (columns); t is the time axis (rows).
    let gx = centralDifference(sti, axis: .space, order: gradientOrder)
    let gt = centralDifference(sti, axis: .time, order: gradientOrder)

    let window: (Matrix) -> Matrix = { m in
        switch windowType {
        case .box:      return m.boxMean(size: windowPx)
        case .gaussian: return m.gaussianMean(sigma: Double(windowPx) / 4.0)
        }
    }

    let Jxx = window(gx.multiplied(by: gx))   // Eq. (7)
    let Jxt = window(gx.multiplied(by: gt))   // Eq. (8)
    let Jtt = window(gt.multiplied(by: gt))   // Eq. (9)

    let step = max(1, Int((Double(windowPx) * (1.0 - overlap)).rounded()))
    let rowsIdx = Array(stride(from: margin, to: nt - margin, by: step))
    let colsIdx = Array(stride(from: margin, to: nx - margin, by: step))
    guard !rowsIdx.isEmpty, !colsIdx.isEmpty else {
        throw OrientationError.noWindowsFit(windowPx: windowPx)
    }

    var phi = Matrix(rows: rowsIdx.count, cols: colsIdx.count)
    var coh = Matrix(rows: rowsIdx.count, cols: colsIdx.count)
    var energy = Matrix(rows: rowsIdx.count, cols: colsIdx.count)

    for (i, r) in rowsIdx.enumerated() {
        for (j, c) in colsIdx.enumerated() {
            let jxx = Jxx[r, c]
            let jxt = Jxt[r, c]
            let jtt = Jtt[r, c]

            // Streak inclination, measured from the time axis toward the space
            // axis.
            //
            // Mind the eigenvector branch here.  The vector n of Eqs. (1)-(2)
            // is normal to the lines of constant grey value, so maximising
            // n^T J n picks the *major* eigenvector -- the image gradient,
            // which lies across the streaks.  Eq. (16), however, needs the
            // inclination of the streaks themselves, i.e. the *minor*
            // eigenvector, 90 degrees away.  Eq. (6) is a statement about
            // tan 2*phi and so cannot distinguish the two branches; taking
            // 0.5*atan2(2 Jxt, Jtt - Jxx) literally returns the gradient
            // direction and yields a velocity of -(Sx/St)/tan(phi_true).
            // Rotating by 90 degrees -- equivalently swapping the arguments'
            // roles as below -- gives the streak inclination Eq. (16) expects.
            //
            // atan2 leaves phi in (-pi/2, pi/2], which spans every
            // distinguishable inclination: an orientation is only defined
            // modulo 180 degrees, and phi and phi + 180 share a tangent, so the
            // sign of the velocity along the search line is unambiguous.
            phi[i, j] = 0.5 * atan2(-2.0 * jxt, jxx - jtt)

            let trace = jxx + jtt
            energy[i, j] = trace
            if trace > 1e-20 {
                let c11 = ((jtt - jxx) * (jtt - jxx) + 4.0 * jxt * jxt).squareRoot() / trace
                coh[i, j] = min(max(c11, 0.0), 1.0)      // Eq. (11)
            } else {
                coh[i, j] = 0.0
            }
        }
    }

    return OrientationField(
        phi: phi, coherency: coh, energy: energy,
        centresT: rowsIdx, centresX: colsIdx
    )
}

// MARK: - Eq. (14): coherency-weighted mean orientation angle

/// Average the per-window angles into one angle for the STI.
///
/// Implements Eq. (14).  For each coherency threshold `C` the histogram
/// `W(phi)` of angles from windows with `C_i >= C` is accumulated over the
/// search range `[phiMin, phiMax]` (Fig. 10), the histograms are summed
/// (Fig. 11), and the weighted mean is taken.  Because a window with high
/// coherency enters every threshold's histogram while a low-coherency one
/// enters only the lowest, clean streaks are counted more often -- which is
/// exactly the intent stated in the paper.
///
/// `phiMin`/`phiMax` come from the user's velocity search range, and are the
/// mechanism the paper uses to discard vectors corrupted by obstacles in the
/// field of view (Fig. 12).
public func meanOrientationAngle(
    field: OrientationField,
    phiMin: Double,
    phiMax: Double,
    coherencyLevels: [Double] = [0.0, 0.2, 0.4, 0.6, 0.8],
    binCount: Int = 180,
    minCoherency: Double = 0.0,
    estimator: AngleEstimator = .eq14
) throws -> AngleEstimate {
    guard phiMin < phiMax else {
        throw OrientationError.badAngleRange(min: phiMin, max: phiMax)
    }

    let phi = field.phi.data
    let coh = field.coherency.data

    var inRange = [Bool](repeating: false, count: phi.count)
    var nInRange = 0
    for i in phi.indices {
        let p = phi[i], c = coh[i]
        if p.isFinite && c.isFinite && p >= phiMin && p <= phiMax && c >= minCoherency {
            inRange[i] = true
            nInRange += 1
        }
    }
    let validFraction = Double(nInRange) / Double(max(1, phi.count))

    // Bin edges and centres are built the way numpy does -- `linspace` edges,
    // then midpoints -- rather than by the algebraically equivalent
    // `phiMin + width * (b + 0.5)`.  The two differ in the last bit, and the
    // golden fixture compares histograms bin for bin, so "algebraically
    // equivalent" is not good enough.
    let span = phiMax - phiMin
    var edges = [Double](repeating: 0, count: binCount + 1)
    for b in 0...binCount { edges[b] = phiMin + span * Double(b) / Double(binCount) }
    edges[binCount] = phiMax          // linspace pins the endpoint exactly
    var centres = [Double](repeating: 0, count: binCount)
    for b in 0..<binCount { centres[b] = 0.5 * (edges[b] + edges[b + 1]) }

    var hist = [Double](repeating: 0, count: binCount)
    for level in coherencyLevels {
        for i in phi.indices where inRange[i] && coh[i] >= level {
            hist[binIndex(phi[i], phiMin: phiMin, span: span,
                          binCount: binCount, edges: edges)] += 1
        }
    }

    let total = hist.reduce(0, +)
    guard total > 0 else {
        return .empty(validFraction: validFraction, histogram: hist,
                      binCentres: centres, phiMin: phiMin, phiMax: phiMax)
    }

    var meanPhi = 0.0
    for b in 0..<binCount { meanPhi += centres[b] * hist[b] }
    meanPhi /= total

    var variance = 0.0
    for b in 0..<binCount {
        let d = centres[b] - meanPhi
        variance += hist[b] * d * d
    }
    let stdPhi = (max(variance / total, 0.0)).squareRoot()

    var meanCoherency = 0.0
    if nInRange > 0 {
        for i in coh.indices where inRange[i] { meanCoherency += coh[i] }
        meanCoherency /= Double(nInRange)
    }

    let reported = estimator == .peak ? parabolicPeak(centres: centres, hist: hist) : meanPhi

    return AngleEstimate(
        phi: reported,
        tanPhi: tan(reported),
        phiStd: stdPhi,
        validFraction: validFraction,
        meanCoherency: meanCoherency,
        usedCount: nInRange,
        histogram: hist,
        binCentres: centres,
        phiMin: phiMin,
        phiMax: phiMax
    )
}

/// numpy's uniform-bin index, including its correction step.
///
/// `numpy.histogram` scales the value across the whole range in one operation
/// and then checks the result against the neighbouring edges, because the scaled
/// value can land one bin out through rounding.  Bins are half-open except the
/// last, which is closed.  Reproduced here because the golden fixture compares
/// bin counts exactly, and a value sitting on an edge is common rather than
/// exotic: a saturated `phi` pinned at the search-range boundary lands there
/// every time.
///
/// The caller must have established that `value` is inside `[phiMin, phiMax]`.
func binIndex(
    _ value: Double,
    phiMin: Double,
    span: Double,
    binCount: Int,
    edges: [Double]
) -> Int {
    var index = Int((value - phiMin) / span * Double(binCount))
    if index == binCount { index -= 1 }          // the closed right edge
    index = min(max(index, 0), binCount - 1)
    if index > 0, value < edges[index] { index -= 1 }
    if index < binCount - 1, value >= edges[index + 1] { index += 1 }
    return index
}

/// Sub-bin location of the histogram maximum by parabolic interpolation.
func parabolicPeak(centres: [Double], hist: [Double]) -> Double {
    guard !hist.isEmpty else { return .nan }
    var peak = 0
    for i in hist.indices where hist[i] > hist[peak] { peak = i }
    if peak == 0 || peak == hist.count - 1 { return centres[peak] }
    let y0 = hist[peak - 1], y1 = hist[peak], y2 = hist[peak + 1]
    let denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12 { return centres[peak] }
    let delta = min(max(0.5 * (y0 - y2) / denom, -1.0), 1.0)
    return centres[peak] + delta * (centres[1] - centres[0])
}

// MARK: - velocity conversion

/// Eq. (16): `U = (Sx / St) tan phi`.
///
/// - Parameters:
///   - sxMetresPerColumn: length of one STI column on the water surface.
///   - stSecondsPerRow: duration of one STI row.
public func velocity(
    fromAngle phi: Double,
    sxMetresPerColumn: Double,
    stSecondsPerRow: Double
) -> Double {
    guard phi.isFinite else { return .nan }
    return (sxMetresPerColumn / stSecondsPerRow) * tan(phi)
}

/// Invert Eq. (16) to get the `phiMin`, `phiMax` of Eq. (14).
///
/// The result is clipped just inside +/- 90 degrees, where `tan` diverges.
public func angleRange(
    forVelocities vMin: Double,
    _ vMax: Double,
    sxMetresPerColumn: Double,
    stSecondsPerRow: Double
) -> (min: Double, max: Double) {
    let k = stSecondsPerRow / sxMetresPerColumn
    let limit = 0.5 * Double.pi - 1e-4
    let lo = min(max(atan(vMin * k), -limit), limit)
    let hi = min(max(atan(vMax * k), -limit), limit)
    return (lo, hi)
}
