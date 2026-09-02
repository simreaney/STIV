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
    public var data: [Double]

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

    @inlinable
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
        let radius = Int(truncate * sigma + 0.5)
        if radius < 1 { return self }

        var kernel = [Double](repeating: 0, count: 2 * radius + 1)
        var total = 0.0
        for (i, x) in (-radius...radius).enumerated() {
            let w = exp(-Double(x * x) / (2.0 * sigma * sigma))
            kernel[i] = w
            total += w
        }
        for i in kernel.indices { kernel[i] /= total }

        return convolved(kernel: kernel, radius: radius, along: .space)
            .convolved(kernel: kernel, radius: radius, along: .time)
    }

    func convolved(kernel: [Double], radius: Int, along axis: Axis) -> Matrix {
        let n = axis == .space ? cols : rows
        let outer = axis == .space ? rows : cols
        var out = Matrix(rows: rows, cols: cols)
        for o in 0..<outer {
            for i in 0..<n {
                var acc = 0.0
                for (k, w) in kernel.enumerated() {
                    acc += w * at(outer: o, inner: Matrix.clampIndex(i + k - radius, n), axis: axis)
                }
                out.set(outer: o, inner: i, axis: axis, value: acc)
            }
        }
        return out
    }

    func at(outer: Int, inner: Int, axis: Axis) -> Double {
        axis == .space ? self[outer, inner] : self[inner, outer]
    }

    mutating func set(outer: Int, inner: Int, axis: Axis, value: Double) {
        if axis == .space { self[outer, inner] = value } else { self[inner, outer] = value }
    }
}
