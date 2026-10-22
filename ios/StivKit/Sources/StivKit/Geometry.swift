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
///
/// `Sendable` because a rectifier describes one fixed camera pose and plane:
/// it is a value to be handed to the analysis queue, not shared mutable state.
public protocol PlaneRectifier: Sendable {
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
public struct SearchLine: Sendable {
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
    /// Where `centre` lands in the reference frame, or `nil` if it is behind
    /// the camera.  The desktop's `centre_px`, exported as `u_px`, `v_px`.
    public internal(set) var centrePx: PixelPoint? = nil

    public var sampleCount: Int { samples.count }
}

/// Everything about where the measurement happens.
public struct SectionLayout: Sendable {
    public let start: PlanePoint
    public let end: PlanePoint
    public let widthM: Double
    public let flowDirection: PlaneVector
    public let searchLines: [SearchLine]
    /// Bearing of the flow, `atan2(z, x)` in degrees: measured from world +X
    /// towards world +Z.
    ///
    /// Viewed from above that is *clockwise*: ARKit's world frame is
    /// right-handed with +Y up, so with +X to the right +Z points towards the
    /// bottom of a map.  It is nonetheless the very number `stiv/config.py`
    /// calls counter-clockwise, because the export hands the desktop the
    /// plane's `(x, z)` as its `(X, Y)`, and in that frame the rotation from
    /// +X to +Y is the conventional positive one.
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
    case .angle:
        // No silent default: an `angle` survey without its angle used to run
        // along world +X, which is a direction nobody chose.
        guard let degrees = flow.angleDegrees, degrees.isFinite else {
            throw SurveyError.invalid("flow.direction == .angle needs a finite angleDegrees")
        }
        let a = degrees * .pi / 180
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

/// Points along a search line at which the local image scale is evaluated
/// when choosing its sample count: `stiv.geometry._SCALE_PROBES`.
let scaleProbes = 33
/// Relative difference between the largest and the mean image scale along a
/// line below which the scale is taken to be uniform (a nadir view):
/// `stiv.geometry._UNIFORM_SCALE_RTOL`.
let uniformScaleTolerance = 1e-6

/// Largest image distance, pixels per metre, anywhere along a world segment:
/// `stiv.geometry._max_px_per_m`, by central differences of `worldToImage`
/// at `scaleProbes` points with a step of a thousandth of the length.
/// Probes the rectifier cannot project (behind the camera) are skipped; `nil`
/// if none can be.
func maxPixelsPerMetre(rectifier: PlaneRectifier, from a: PlanePoint, to b: PlanePoint) -> Double? {
    let length = distance(a, b)
    guard length > 0, length.isFinite else { return nil }
    let u = PlaneVector(x: (b.x - a.x) / length, z: (b.z - a.z) / length)
    let h = 1e-3 * length
    var best: Double?
    for k in 0..<scaleProbes {
        // `np.linspace(0, length, 33)`: `k * step`, with the last pinned.
        let s = k == scaleProbes - 1 ? length : Double(k) * (length / Double(scaleProbes - 1))
        let centre = PlanePoint(x: a.x + s * u.x, z: a.z + s * u.z)
        guard let forward = rectifier.worldToImage(centre + u * h),
              let back = rectifier.worldToImage(centre + u * -h) else { continue }
        let du = forward.u - back.u
        let dv = forward.v - back.v
        let span = (du * du + dv * dv).squareRoot()
        guard span.isFinite else { continue }
        best = max(best ?? 0, span)
    }
    return best.map { $0 / (2.0 * h) }
}

/// Samples for a search line from `a` to `b` when none are configured:
/// `stiv.geometry.default_n_samples`.
///
/// One STI column per image pixel is the natural resolution limit -- *where
/// the image is finest along the line*.  In an oblique view, which on a phone
/// held on a bank is every view, the near end of a line covers more pixels
/// per metre than the far end, so counting pixels end to end (the mean
/// spacing) left the near field undersampled and aliased its finest texture.
/// The count follows the largest local pixels-per-metre instead, rounded half
/// to even as Python's `round`, and clamped to 16...4096.  When the scale is
/// uniform to within `uniformScaleTolerance` -- a camera looking straight down
/// -- the original end-to-end count is kept, so rounding error in the
/// derivative cannot flip a count that sits on a half.
///
/// `nil` when the endpoints do not project, for the caller to fall back on.
func defaultSampleCount(rectifier: PlaneRectifier, from a: PlanePoint, to b: PlanePoint) -> Int? {
    guard let pa = rectifier.worldToImage(a), let pb = rectifier.worldToImage(b) else { return nil }
    let spanMean = ((pa.u - pb.u) * (pa.u - pb.u) + (pa.v - pb.v) * (pa.v - pb.v)).squareRoot()
    guard spanMean.isFinite else { return nil }
    var span = spanMean
    if let perMetre = maxPixelsPerMetre(rectifier: rectifier, from: a, to: b) {
        let spanMax = distance(a, b) * perMetre
        if spanMax.isFinite && spanMax > spanMean * (1.0 + uniformScaleTolerance) { span = spanMax }
    }
    return Int(min(max(span.rounded(.toNearestOrEven) + 1, 16), 4096))
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
    guard section.stationCount >= 1 else {
        throw SurveyError.invalid("section.stationCount must be >= 1, got \(section.stationCount)")
    }
    let start = section.start
    let end = section.end
    let widthM = distance(start, end)
    guard widthM.isFinite else {
        throw SurveyError.invalid("section.start and section.end must be finite world points")
    }
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

        var count: Int
        if let fixed = survey.sti.sampleCount {
            count = fixed
        } else {
            count = defaultSampleCount(rectifier: rectifier, from: a, to: b) ?? 128
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
            rangeM: rectifier.horizontalRange(to: stationCentre),
            centrePx: rectifier.worldToImage(stationCentre)
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
