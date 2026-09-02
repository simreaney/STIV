//  Synthetic.swift
//  Test fixtures: space-time images whose answer is known in closed form.
//
//  Mirrors `synthetic_sti` in `tests/test_orientation.py`: a broadband 1-D
//  pattern advected along the search line at exactly `tanPhi` columns per row.
//  Broadband rather than a single sinusoid on purpose -- a sinusoid can be
//  recovered by accident from an aliased grid, whereas smoothed noise cannot.

import Foundation
import StivKit

/// splitmix64 + Box-Muller.  Deterministic across platforms, which matters
/// because a flaky orientation test is worse than no orientation test.
struct DeterministicNormals {
    private var state: UInt64

    init(seed: UInt64 = 0) { state = seed &+ 0x9E3779B97F4A7C15 }

    private mutating func nextUInt64() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    private mutating func nextUnit() -> Double {
        Double(nextUInt64() >> 11) * (1.0 / 9007199254740992.0)
    }

    mutating func next() -> Double {
        let u1 = max(nextUnit(), 1e-12)
        let u2 = nextUnit()
        return (-2.0 * log(u1)).squareRoot() * cos(2.0 * .pi * u2)
    }

    mutating func array(_ n: Int) -> [Double] {
        (0..<n).map { _ in next() }
    }
}

/// 1-D Gaussian smoothing with edge clamping, matching `scipy.ndimage`'s
/// defaults closely enough for a test pattern.
func smooth1D(_ xs: [Double], sigma: Double) -> [Double] {
    guard sigma > 0 else { return xs }
    let radius = Int(4.0 * sigma + 0.5)
    var kernel = (-radius...radius).map { exp(-Double($0 * $0) / (2 * sigma * sigma)) }
    let total = kernel.reduce(0, +)
    for i in kernel.indices { kernel[i] /= total }
    return xs.indices.map { i in
        var acc = 0.0
        for (k, w) in kernel.enumerated() {
            let j = min(max(i + k - radius, 0), xs.count - 1)
            acc += w * xs[j]
        }
        return acc
    }
}

/// An STI whose pattern advects at exactly `tanPhi` columns per row, so the
/// correct answer is `phi = atan(tanPhi)` by construction.
func syntheticSTI(
    tanPhi: Double,
    rows: Int = 200,
    cols: Int = 200,
    sigma: Double = 3.0,
    noise: Double = 0.0,
    seed: UInt64 = 0
) -> Matrix {
    var rng = DeterministicNormals(seed: seed)
    let length = cols + Int(abs(tanPhi)) * rows + 8 * Int(sigma) + 10
    let base = smooth1D(rng.array(length), sigma: sigma)

    var m = Matrix(rows: rows, cols: cols)
    for r in 0..<rows {
        for c in 0..<cols {
            // Wrap so the pattern is defined for every (r, c) without running
            // off the end of `base`.
            var idx = (Double(c) - tanPhi * Double(r)).truncatingRemainder(dividingBy: Double(length))
            if idx < 0 { idx += Double(length) }
            let i0 = Int(idx)
            let f = idx - Double(i0)
            m[r, c] = base[i0] * (1 - f) + base[(i0 + 1) % length] * f
        }
    }

    if noise > 0 {
        let (_, std) = standardDeviation(m.data)
        for i in m.data.indices { m.data[i] += noise * std * rng.next() }
    }
    return m
}

func standardDeviation(_ xs: [Double]) -> (mean: Double, std: Double) {
    let mean = xs.reduce(0, +) / Double(xs.count)
    let variance = xs.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(xs.count)
    return (mean, variance.squareRoot())
}

/// The widest angle range Eq. (14) can search, used when the test is about the
/// orientation rather than about the range gate.
let fullAngleRange = (min: -Double.pi / 2 + 1e-4, max: Double.pi / 2 - 1e-4)
