//  Matrix.swift
//  A minimal row-major 2-D array of Double, plus the two neighbourhood filters
//  the orientation analysis needs.
//
//  The Python implementation leans on numpy and scipy.ndimage; this is the
//  small subset of that behaviour the port depends on, reproduced exactly --
//  including scipy's edge-handling conventions, because the golden fixture test
//  compares the two to ~1e-6 and an off-by-one in a boundary rule shows up
//  there as a systematic angle bias.

import Foundation

/// Which axis of a space-time image an operation runs along.
///
/// An STI is stored `(time, space)`, matching `stiv.sti.SpaceTimeImage.data`:
/// rows are frames, columns are positions along the search line.
public enum Axis {
    /// Rows: the time axis, `t` in the paper's `g(x, t)`.
    case time
    /// Columns: the space axis, `x`.
    case space
}

/// Row-major matrix of `Double`.
public struct Matrix: Sendable {
    public let rows: Int
    public let cols: Int
    /// The samples, row-major.  Read-only outside the package: `rows` and
    /// `cols` are fixed at construction, and a caller that could reassign the
    /// array could also change its length behind their back.  Element-wise
    /// writes go through the subscript, which cannot.
    public internal(set) var data: [Double]

    public init(rows: Int, cols: Int, repeating value: Double = 0) {
        precondition(rows > 0 && cols > 0, "Matrix must be non-empty, got \(rows)x\(cols)")
        self.rows = rows
        self.cols = cols
        self.data = [Double](repeating: value, count: rows * cols)
    }

    public init(rows: Int, cols: Int, data: [Double]) {
        precondition(rows > 0 && cols > 0, "Matrix must be non-empty, got \(rows)x\(cols)")
        precondition(data.count == rows * cols,
                     "expected \(rows * cols) elements for a \(rows)x\(cols) matrix, got \(data.count)")
        self.rows = rows
        self.cols = cols
        self.data = data
    }

    /// Build from `Float` samples, as the STI store holds them.
    public init(rows: Int, cols: Int, floats: [Float]) {
        self.init(rows: rows, cols: cols, data: floats.map(Double.init))
    }

    public subscript(r: Int, c: Int) -> Double {
        get { data[r * cols + c] }
        set { data[r * cols + c] = newValue }
    }

    public var count: Int { rows * cols }

    /// Element-wise product, used for the `gx*gx`, `gx*gt`, `gt*gt` of Eqs. (7)-(9).
    public func multiplied(by other: Matrix) -> Matrix {
        precondition(rows == other.rows && cols == other.cols, "shape mismatch")
        var out = self
        for i in 0..<data.count { out.data[i] = data[i] * other.data[i] }
        return out
    }
}

// MARK: - boundary rules

extension Matrix {
    /// numpy's `mode="reflect"`: the edge sample is *not* repeated, so
    /// `[a b c d]` padded by 2 becomes `[c b | a b c d | c b]`.
    ///
    /// This is the rule `stiv.orientation.central_difference` uses, and it
    /// differs from scipy's `mode="reflect"` (which is numpy's `"symmetric"`) --
    /// a trap worth naming, since picking the wrong one biases the gradient in
    /// the outermost few pixels only, which is invisible until the golden test.
    static func reflectIndex(_ i: Int, _ n: Int) -> Int {
        if n == 1 { return 0 }
        var j = i
        let period = 2 * (n - 1)
        j = ((j % period) + period) % period
        return j <= n - 1 ? j : period - j
    }

    /// scipy's `mode="nearest"`: clamp to the edge sample.
    static func clampIndex(_ i: Int, _ n: Int) -> Int {
        i < 0 ? 0 : (i > n - 1 ? n - 1 : i)
    }
}

// MARK: - windowed means (the window function w of Eq. 3)

extension Matrix {
    /// Separable running-sum box mean -- `scipy.ndimage.uniform_filter`
    /// with `mode="nearest"`.
    ///
    /// This is the paper's rectangular integration area `A` (31 x 31 px by
    /// default).  scipy places the window asymmetrically for even sizes:
    /// `size // 2` samples before the centre and `size - size//2 - 1` after,
    /// which is reproduced here so the two implementations agree for any size.
    ///
    /// Cost is O(1) per pixel regardless of window size, which is what lets a
    /// phone run 31 x 31 windows over thirty 1800-row STIs without a GPU.
    public func boxMean(size: Int) -> Matrix {
        precondition(size >= 1, "box size must be >= 1, got \(size)")
        if size == 1 { return self }
        let before = size / 2
        let after = size - size / 2 - 1
        return runningMean(before: before, after: after, along: .space)
            .runningMean(before: before, after: after, along: .time)
    }

    /// One separable pass of the box mean.
    func runningMean(before: Int, after: Int, along axis: Axis) -> Matrix {
        let n = axis == .space ? cols : rows
        let outer = axis == .space ? rows : cols
        let width = Double(before + after + 1)
        var out = Matrix(rows: rows, cols: cols)

        for o in 0..<outer {
            // Seed the running sum for the first output sample.
            var sum = 0.0
            for k in -before...after {
                sum += at(outer: o, inner: Matrix.clampIndex(k, n), axis: axis)
            }
            out.set(outer: o, inner: 0, axis: axis, value: sum / width)

            for i in 1..<n {
                // Slide: drop the sample leaving the window, add the one entering.
                sum -= at(outer: o, inner: Matrix.clampIndex(i - before - 1, n), axis: axis)
                sum += at(outer: o, inner: Matrix.clampIndex(i + after, n), axis: axis)
                out.set(outer: o, inner: i, axis: axis, value: sum / width)
            }
        }
        return out
    }

    /// Separable Gaussian mean -- `scipy.ndimage.gaussian_filter` with
    /// `mode="nearest"` and the default `truncate=4.0`.
    ///
    /// Same nominal support as the box window but smoothly tapered, which gives
    /// a better-conditioned tensor; `StiSettings.windowType` selects it.
    public func gaussianMean(sigma: Double, truncate: Double = 4.0) -> Matrix {
        precondition(sigma > 0, "sigma must be positive, got \(sigma)")
        guard let (kernel, radius) = Matrix.gaussianKernel(sigma: sigma, truncate: truncate) else {
            return self
        }
        return convolved(kernel: kernel, radius: radius, along: .space)
            .convolved(kernel: kernel, radius: radius, along: .time)
    }

    /// The Gaussian mean evaluated only at the grid `rowIndices x colIndices`.
    ///
    /// Bit-for-bit what `gaussianMean(sigma:truncate:)` returns at those
    /// points, at a fraction of the cost.  The orientation analysis filters
    /// three full-resolution tensor components only to read back some seventy
    /// window centres, and a 63-tap kernel (the default 31 px window) applied
    /// everywhere made the Gaussian window ten times slower than the box.
    ///
    /// The separable structure is what makes this exact rather than an
    /// approximation.  The full filter's space pass is needed only in the
    /// sampled *columns* -- but in every row, because the time pass then reads
    /// a full kernel's height of it -- and the time pass only in the sampled
    /// rows.  Each output is the same sum, of the same products, accumulated in
    /// the same order as the full filter computes it, so no bit changes.
    public func gaussianMean(
        sigma: Double,
        truncate: Double = 4.0,
        rowIndices: [Int],
        colIndices: [Int]
    ) -> Matrix {
        precondition(sigma > 0, "sigma must be positive, got \(sigma)")
        precondition(!rowIndices.isEmpty && !colIndices.isEmpty, "need at least one sample point")
        precondition(rowIndices.allSatisfy { (0..<rows).contains($0) }
                     && colIndices.allSatisfy { (0..<cols).contains($0) },
                     "sample indices must lie inside the \(rows)x\(cols) matrix")

        let nr = rowIndices.count
        let nc = colIndices.count
        guard let (kernel, radius) = Matrix.gaussianKernel(sigma: sigma, truncate: truncate) else {
            var out = Matrix(rows: nr, cols: nc)
            for (i, r) in rowIndices.enumerated() {
                for (j, c) in colIndices.enumerated() { out[i, j] = self[r, c] }
            }
            return out
        }
        let taps = kernel.count

        // Space pass, every row, sampled columns only.
        var partial = [Double](repeating: 0, count: rows * nc)
        data.withUnsafeBufferPointer { src in
            kernel.withUnsafeBufferPointer { w in
                partial.withUnsafeMutableBufferPointer { dst in
                    for r in 0..<rows {
                        let rowBase = r * cols
                        for (j, c) in colIndices.enumerated() {
                            var acc = 0.0
                            if c >= radius && c + radius < cols {
                                let start = rowBase + c - radius
                                for k in 0..<taps { acc += w[k] * src[start + k] }
                            } else {
                                for k in 0..<taps {
                                    acc += w[k] * src[rowBase + Matrix.clampIndex(c + k - radius, cols)]
                                }
                            }
                            dst[r * nc + j] = acc
                        }
                    }
                }
            }
        }

        // Time pass, sampled rows only.  Accumulated a whole row of outputs at
        // a time, tap by tap, which keeps each output's summation order --
        // tap 0 first -- while reading memory contiguously.
        var out = [Double](repeating: 0, count: nr * nc)
        partial.withUnsafeBufferPointer { src in
            kernel.withUnsafeBufferPointer { w in
                out.withUnsafeMutableBufferPointer { dst in
                    for (i, r) in rowIndices.enumerated() {
                        let o = i * nc
                        for k in 0..<taps {
                            let s = Matrix.clampIndex(r + k - radius, rows) * nc
                            let wk = w[k]
                            for j in 0..<nc { dst[o + j] += wk * src[s + j] }
                        }
                    }
                }
            }
        }
        return Matrix(rows: nr, cols: nc, data: out)
    }

    /// scipy's normalised Gaussian kernel and its radius, or `nil` when the
    /// radius rounds to zero and the filter is the identity.
    static func gaussianKernel(sigma: Double, truncate: Double) -> (kernel: [Double], radius: Int)? {
        let reach = truncate * sigma + 0.5
        // A non-finite or absurd reach would trap in `Int(_:)` or try to
        // allocate a kernel larger than memory; no image is that big anyway.
        let radius = reach.isFinite ? Int(min(max(reach, 0), 1e7)) : 0
        if radius < 1 { return nil }

        var kernel = [Double](repeating: 0, count: 2 * radius + 1)
        var total = 0.0
        for (i, x) in (-radius...radius).enumerated() {
            let w = exp(-Double(x * x) / (2.0 * sigma * sigma))
            kernel[i] = w
            total += w
        }
        for i in kernel.indices { kernel[i] /= total }
        return (kernel, radius)
    }

    /// One separable pass of the Gaussian mean, with scipy's `"nearest"` edges.
    ///
    /// The clamp is only needed within `radius` of an edge, so the interior
    /// runs without it; and the time pass accumulates a whole row of outputs
    /// tap by tap rather than striding down each column.  Neither changes
    /// what is summed or in what order, so the result is bit-for-bit that of
    /// the plain per-pixel loop.
    func convolved(kernel: [Double], radius: Int, along axis: Axis) -> Matrix {
        let taps = kernel.count
        var out = [Double](repeating: 0, count: count)
        data.withUnsafeBufferPointer { src in
            kernel.withUnsafeBufferPointer { w in
                out.withUnsafeMutableBufferPointer { dst in
                    switch axis {
                    case .space:
                        for r in 0..<rows {
                            let rowBase = r * cols
                            for c in 0..<cols {
                                var acc = 0.0
                                if c >= radius && c + radius < cols {
                                    let start = rowBase + c - radius
                                    for k in 0..<taps { acc += w[k] * src[start + k] }
                                } else {
                                    for k in 0..<taps {
                                        acc += w[k] * src[rowBase + Matrix.clampIndex(c + k - radius, cols)]
                                    }
                                }
                                dst[rowBase + c] = acc
                            }
                        }
                    case .time:
                        for r in 0..<rows {
                            let o = r * cols
                            for k in 0..<taps {
                                let s = Matrix.clampIndex(r + k - radius, rows) * cols
                                let wk = w[k]
                                for c in 0..<cols { dst[o + c] += wk * src[s + c] }
                            }
                        }
                    }
                }
            }
        }
        return Matrix(rows: rows, cols: cols, data: out)
    }

    func at(outer: Int, inner: Int, axis: Axis) -> Double {
        axis == .space ? self[outer, inner] : self[inner, outer]
    }

    mutating func set(outer: Int, inner: Int, axis: Axis, value: Double) {
        if axis == .space { self[outer, inner] = value } else { self[inner, outer] = value }
    }
}
