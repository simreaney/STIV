//  Geometry.swift
//  The cross-section and its streamwise search lines, laid out in world metres.
//
//  A port of the section-building half of `stiv/geometry.py`.  The rectifiers
//  themselves do not come across: `ScaleRectifier` assumes a nadir view a
//  handheld phone never has, and `HomographyRectifier` needs surveyed ground
//  control points the whole design exists to avoid.  Both are replaced by a
//  single `PlaneRectifier` protocol -- ray/plane intersection against the
//  gravity-aligned water surface -- whose ARKit implementation lives in the app
//  target.  Keeping the protocol here means the section layout can be tested
//  against a synthetic pinhole camera with a closed-form answer.
//
//  Two coordinate systems, as in the Python:
//
//  image   `(u, v)` = (column, row) in pixels of the captured frame, origin at
//          the top-left, floating point.
//  world   ARKit world metres on the horizontal water-surface plane, carried as
//          `PlanePoint(x, z)`.  ARKit's world frame is gravity-aligned with +Y
//          up, so the water surface is the plane `y = y0` and the two remaining
//          axes are exactly the plane's own coordinates.

import Foundation

/// A point in captured-image pixels.
public struct PixelPoint: Sendable, Equatable {
    public var u: Double
    public var v: Double
    public init(u: Double, v: Double) { self.u = u; self.v = v }
}

/// Bidirectional mapping between captured-image pixels and world metres on the
/// water plane.
public protocol PlaneRectifier {
    /// Where a pixel's ray meets the water plane, or `nil` if it misses (the
    /// ray points at the sky, or above the horizon).
    func imageToWorld(_ px: PixelPoint) -> PlanePoint?
    /// Where a point on the water plane lands in the image, or `nil` if it is
    /// behind the camera.
    func worldToImage(_ p: PlanePoint) -> PixelPoint?
    /// Captured-image dimensions, for the in-frame test.
    var imageWidth: Int { get }
    var imageHeight: Int { get }
    /// Horizontal distance from the camera to a point on the plane, metres.
    /// Drives the range-dependent term of the scale uncertainty.
    func horizontalRange(to p: PlanePoint) -> Double
    /// Height of the camera above the water plane, metres.
    var cameraHeightM: Double { get }
    /// Human-readable summary, as `Rectifier.describe`.
    func describe() -> String
}

public extension PlaneRectifier {
    /// Local ground sample distance (metres per pixel) at an image point, by
    /// finite difference -- the same trick as
    /// `HomographyRectifier._local_scale`.
    ///
    /// Useful for reporting: in an oblique view this varies by an order of
    /// magnitude between the near and far bank, which is precisely why the
    /// samples along a search line are spaced in world metres and not pixels.
    func localScale(at px: PixelPoint) -> Double? {
        guard let a = imageToWorld(px),
              let b = imageToWorld(PixelPoint(u: px.u + 1, v: px.v)),
              let c = imageToWorld(PixelPoint(u: px.u, v: px.v + 1)) else { return nil }
        return 0.5 * (distance(a, b) + distance(a, c))
    }
}

// MARK: - plane vector helpers

@inlinable
public func distance(_ a: PlanePoint, _ b: PlanePoint) -> Double {
    ((a.x - b.x) * (a.x - b.x) + (a.z - b.z) * (a.z - b.z)).squareRoot()
}

/// A vector on the water plane.
public struct PlaneVector: Sendable, Equatable {
    public var x: Double
    public var z: Double
    public init(x: Double, z: Double) { self.x = x; self.z = z }

    public var length: Double { (x * x + z * z).squareRoot() }

    public func normalised() throws -> PlaneVector {
        let n = length
        guard n > 1e-12 else {
            throw SurveyError.invalid("degenerate direction vector (zero length)")
        }
        return PlaneVector(x: x / n, z: z / n)
    }

    /// A perpendicular, by a quarter turn in the plane.  *Which* of the two
    /// perpendiculars this is depends on the handedness of `(x, z)`, and it does
    /// not matter: the caller resolves the sign against the user's downstream
    /// marker, exactly as `resolve_flow_direction` does.
    public var perpendicular: PlaneVector { PlaneVector(x: -z, z: x) }

    public static func * (v: PlaneVector, s: Double) -> PlaneVector {
        PlaneVector(x: v.x * s, z: v.z * s)
    }

    public static prefix func - (v: PlaneVector) -> PlaneVector {
        PlaneVector(x: -v.x, z: -v.z)
    }

    public func dot(_ o: PlaneVector) -> Double { x * o.x + z * o.z }
}

public func + (p: PlanePoint, v: PlaneVector) -> PlanePoint {
    PlanePoint(x: p.x + v.x, z: p.z + v.z)
}

public func - (a: PlanePoint, b: PlanePoint) -> PlaneVector {
    PlaneVector(x: a.x - b.x, z: a.z - b.z)
}

// MARK: - search lines

/// One streamwise search line: the seed of a single space-time image.
public struct SearchLine {
    /// Station number along the cross-section, 0 at `section.start`.
    public let index: Int
    /// Distance along the cross-section from its start, metres.
    public let stationM: Double
    /// Where the line crosses the section.
    public let centre: PlanePoint
    public let direction: PlaneVector
    /// World points sampled uniformly along the line, so `Sx` is constant along
    /// the STI even in a strongly oblique view.
    public let samples: [PlanePoint]
    /// Metres per STI column: the `Sx` of Eq. (16).
    public let sxMetresPerColumn: Double
    public let lengthM: Double
    /// Horizontal distance from the camera to `centre` at setup, metres.
    public let rangeM: Double

    public var sampleCount: Int { samples.count }
}

/// Everything about where the measurement happens.
public struct SectionLayout {
    public let start: PlanePoint
    public let end: PlanePoint
    public let widthM: Double
    public let flowDirection: PlaneVector
    public let searchLines: [SearchLine]
    /// Bearing of the flow, degrees counter-clockwise from world +X.
    public let flowBearingDegrees: Double
    /// Camera height above the water plane at setup, metres.
    public let cameraHeightM: Double
    /// Stations whose search line left the frame; cf. `check_lines_in_frame`.
    public let stationsOutsideFrame: [Int]
    public let rectifierDescription: String

    public var stationsM: [Double] { searchLines.map(\.stationM) }
}

/// Return the unit downstream direction on the water plane.
public func resolveFlowDirection(
    _ flow: FlowSettings,
    sectionDirection: PlaneVector,
    sectionCentre: PlanePoint
) throws -> PlaneVector {
    var d: PlaneVector
    switch flow.direction {
    case .bearing:
        let a = (flow.bearingDegrees ?? 0) * .pi / 180
        d = PlaneVector(x: cos(a), z: sin(a))
    case .points:
        guard let from = flow.from, let to = flow.to else {
            throw SurveyError.invalid("flow.direction == .points needs both from and to")
        }
        d = to - from
    case .auto:
        // Perpendicular to the section, in world space.
        d = sectionDirection.perpendicular
        if let hint = flow.downstream, (hint - sectionCentre).dot(d) < 0 {
            // Point the normal at the user's downstream marker.
            d = -d
        }
    }
    d = try d.normalised()
    return flow.reverse ? -d : d
}

/// Lay out the cross-section and its search lines.
///
/// Stations are spaced evenly in *world* space and samples along each line are
/// spaced evenly in world metres, so an oblique view still yields evenly spaced
/// verticals and a constant `Sx` -- the paper notes (section 2.1, on Fig. 3)
/// that "the horizontal length of the respective STI is not constant because of
/// the image obliqueness"; sampling through the rectifier removes that caveat.
public func buildSection(
    survey: Survey,
    rectifier: PlaneRectifier,
    inFrameMargin: Int = 2
) throws -> SectionLayout {
    let section = survey.section
    let start = section.start
    let end = section.end
    let widthM = distance(start, end)
    guard widthM > 0 else {
        throw SurveyError.invalid("section.start and section.end are the same world point")
    }
    let span = end - start
    let sectionDirection = try span.normalised()
    let centre = PlanePoint(x: 0.5 * (start.x + end.x), z: 0.5 * (start.z + end.z))
    let flowDirection = try resolveFlowDirection(
        survey.flow, sectionDirection: sectionDirection, sectionCentre: centre)

    let lengthM = survey.sti.searchLineLengthM
    let half = 0.5 * lengthM
    let t0 = section.edgeTrim
    let t1 = 1.0 - section.edgeTrim
    let n = section.stationCount

    var lines: [SearchLine] = []
    var outside: [Int] = []
    lines.reserveCapacity(n)

    for i in 0..<n {
        let f = n == 1 ? 0.5 * (t0 + t1) : t0 + (t1 - t0) * Double(i) / Double(n - 1)
        let stationCentre = start + span * f
        let a = stationCentre + flowDirection * (-half)
        let b = stationCentre + flowDirection * half

        // One STI column per image pixel is the natural resolution limit: finer
        // sampling would only interpolate, coarser would discard resolution the
        // sensor already gave us.
        var count: Int
        if let fixed = survey.sti.sampleCount {
            count = fixed
        } else if let pa = rectifier.worldToImage(a), let pb = rectifier.worldToImage(b) {
            let du = pa.u - pb.u
            let dv = pa.v - pb.v
            count = min(max(Int((du * du + dv * dv).squareRoot().rounded()) + 1, 16), 4096)
        } else {
            count = 128
        }
        if count < 16 {
            throw SurveyError.invalid("sti.sampleCount must be >= 16, got \(count)")
        }

        var samples: [PlanePoint] = []
        samples.reserveCapacity(count)
        for k in 0..<count {
            let s = Double(k) / Double(count - 1)
            samples.append(PlanePoint(x: a.x * (1 - s) + b.x * s,
                                      z: a.z * (1 - s) + b.z * s))
        }

        // A line that leaves the frame samples NaN for part of its length --
        // worth telling the user before they record thirty seconds of it.
        let leaves = samples.contains { p in
            guard let px = rectifier.worldToImage(p) else { return true }
            return px.u < Double(inFrameMargin)
                || px.u > Double(rectifier.imageWidth - 1 - inFrameMargin)
                || px.v < Double(inFrameMargin)
                || px.v > Double(rectifier.imageHeight - 1 - inFrameMargin)
        }
        if leaves { outside.append(i) }

        lines.append(SearchLine(
            index: i,
            stationM: f * widthM,
            centre: stationCentre,
            direction: flowDirection,
            samples: samples,
            sxMetresPerColumn: lengthM / Double(count - 1),
            lengthM: lengthM,
            rangeM: rectifier.horizontalRange(to: stationCentre)
        ))
    }

    return SectionLayout(
        start: start,
        end: end,
        widthM: widthM,
        flowDirection: flowDirection,
        searchLines: lines,
        flowBearingDegrees: atan2(flowDirection.z, flowDirection.x) * 180 / .pi,
        cameraHeightM: rectifier.cameraHeightM,
        stationsOutsideFrame: outside,
        rectifierDescription: rectifier.describe()
    )
}
