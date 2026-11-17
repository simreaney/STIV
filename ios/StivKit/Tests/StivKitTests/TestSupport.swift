//  TestSupport.swift
//  Building blocks for tests that drive the analysis stages directly, without a
//  camera, a layout or a recording.

import Foundation
@testable import StivKit

/// A search line with the given scale and nothing else of interest.
func makeLine(
    index: Int,
    stationM: Double,
    sx: Double = 0.01,
    rangeM: Double = 0,
    samples: Int = 16
) -> SearchLine {
    SearchLine(
        index: index,
        stationM: stationM,
        centre: PlanePoint(x: stationM, z: 0),
        direction: PlaneVector(x: 0, z: 1),
        samples: (0..<samples).map { PlanePoint(x: stationM, z: Double($0) * sx) },
        sxMetresPerColumn: sx,
        lengthM: sx * Double(samples - 1),
        rangeM: rangeM)
}

/// A station as the first pass of the pipeline would leave it, with no scale
/// uncertainty, so quality control judges only what the test sets.
func makeStation(
    index: Int,
    stationM: Double? = nil,
    velocity: Double,
    angleStd: Double = 0.01,
    tanPhi: Double? = nil,
    coherency: Double = 0.6,
    validFraction: Double = 0.8,
    flags: [StationFlag] = [],
    sx: Double = 0.01,
    st: Double = 1.0 / 30.0
) -> StationResult {
    var s = StationResult(line: makeLine(index: index, stationM: stationM ?? Double(index), sx: sx))
    s.velocityMs = velocity
    s.velocityRawMs = velocity
    s.velocityStdAngleMs = angleStd
    s.velocityStdMs = angleStd
    s.tanPhi = tanPhi ?? (velocity.isFinite ? velocity * st / sx : .nan)
    s.phiRad = atan(s.tanPhi)
    s.coherency = coherency
    s.validFraction = validFraction
    s.windowCount = 10
    s.relativeScaleSigma = 0
    s.stSecondsPerRow = st
    s.flags = flags
    return s
}

/// An STI of the given data with the given scales.
func makeSTI(_ data: Matrix, index: Int = 0, sx: Double = 0.01, st: Double = 1.0 / 30.0) -> SpaceTimeImage {
    SpaceTimeImage(data: data, stationIndex: index, sxMetresPerColumn: sx,
                   stSecondsPerRow: st, outOfFrameFraction: 0)
}

extension StationFlag {
    init?(pythonName: String) {
        guard let flag = StationFlag.allCases.first(where: { $0.pythonName == pythonName }) else {
            return nil
        }
        self = flag
    }
}

/// Mirror an STI in space: the same flow seen by a search line pointing the
/// other way.
func mirrored(_ m: Matrix) -> Matrix {
    var out = m
    for r in 0..<m.rows {
        for c in 0..<m.cols { out[r, c] = m[r, m.cols - 1 - c] }
    }
    return out
}

/// Equal, or both NaN.
func sameOrBothNaN(_ a: Double, _ b: Double, accuracy: Double) -> Bool {
    (a.isNaN && b.isNaN) || abs(a - b) <= accuracy
}

extension WaterPlaneProjection {
    /// Local ground sample distance (metres per pixel) at an image point, by
    /// finite difference -- the same trick as
    /// `HomographyRectifier._local_scale`.  In an oblique view it varies by an
    /// order of magnitude between the near and far bank, which is why the
    /// samples along a search line are spaced in world metres, not pixels.
    func localScale(at px: PixelPoint) -> Double? {
        guard let a = imageToWorld(px),
              let b = imageToWorld(PixelPoint(u: px.u + 1, v: px.v)),
              let c = imageToWorld(PixelPoint(u: px.u, v: px.v + 1)) else { return nil }
        return 0.5 * (distance(a, b) + distance(a, c))
    }
}
