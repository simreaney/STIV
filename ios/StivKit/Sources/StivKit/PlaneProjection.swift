//  PlaneProjection.swift
//  Image <-> world by intersecting each pixel's ray with the water surface.
//
//  This is what replaces `stiv.geometry.ScaleRectifier` and
//  `HomographyRectifier`.  Both of those infer the mapping from something the
//  user had to measure -- a flight altitude, or four surveyed control points.
//  A phone does not need to infer it: ARKit hands over a metric camera pose and
//  the camera intrinsics every frame, and LiDAR fixes the one remaining unknown.
//
//  That unknown is a single number.  ARKit's world frame is gravity-aligned
//  (+Y up), still water is a horizontal plane, so the water surface is
//  `y = planeY` and nothing else about it needs measuring.  A LiDAR range to
//  the waterline -- the contour where water meets bank, which lies by
//  definition at water level, and which is within the sensor's ~5 m reach from
//  the bank -- pins `planeY`.  Every distance on the water surface then follows
//  from ray/plane intersection, including at the far bank thirty metres away
//  where no depth measurement is possible at all.
//
//  Deliberately free of `simd` and ARKit so it can be tested against a camera
//  whose answer is known in closed form; the app's `ARGeometry.swift` is a thin
//  adapter that fills these types in from an `ARCamera`.

import Foundation

/// Pinhole intrinsics in captured-image pixels, as `ARCamera.intrinsics` gives
/// them: focal lengths and a principal point measured from the top-left of the
/// image, in the camera's native sensor orientation.
public struct PinholeIntrinsics: Sendable, Equatable {
    public var fx: Double
    public var fy: Double
    public var cx: Double
    public var cy: Double
    public var width: Int
    public var height: Int

    public init(fx: Double, fy: Double, cx: Double, cy: Double, width: Int, height: Int) {
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
    }

    /// Intrinsics for a camera with a given horizontal field of view, for tests
    /// and for the manual-scale fallback on devices without LiDAR.
    public init(horizontalFovDegrees fov: Double, width: Int, height: Int) {
        let f = Double(width) / (2.0 * tan(fov * .pi / 360.0))
        self.init(fx: f, fy: f, cx: Double(width) / 2, cy: Double(height) / 2,
                  width: width, height: height)
    }
}

/// A camera pose: the world-from-camera rigid transform, stored column-major in
/// 16 elements exactly as `simd_float4x4` lays it out.
///
/// ARKit's camera-local axes are +X right, +Y up and +Z *out of the screen*, so
/// the camera looks along **-Z**.  Every sign below follows from that.
public struct CameraPose: Sendable, Equatable {
    /// Column-major 4x4: element (row, col) is `m[4 * col + row]`.
    public var m: [Double]

    public init(columnMajor m: [Double]) {
        precondition(m.count == 16, "a 4x4 transform needs 16 elements, got \(m.count)")
        self.m = m
    }

    @inlinable
    public func at(_ row: Int, _ col: Int) -> Double { m[4 * col + row] }

    /// Camera position in world metres.
    public var position: (x: Double, y: Double, z: Double) {
        (at(0, 3), at(1, 3), at(2, 3))
    }

    /// Rotate a camera-space direction into world space.
    public func rotateToWorld(_ v: (x: Double, y: Double, z: Double))
        -> (x: Double, y: Double, z: Double) {
        (
            at(0, 0) * v.x + at(0, 1) * v.y + at(0, 2) * v.z,
            at(1, 0) * v.x + at(1, 1) * v.y + at(1, 2) * v.z,
            at(2, 0) * v.x + at(2, 1) * v.y + at(2, 2) * v.z
        )
    }

    /// Bring a world point into camera space.
    ///
    /// The rotation block is orthonormal, so the inverse is its transpose --
    /// no general 4x4 inversion, and no chance of a numerically noisy one.
    public func toCamera(_ p: (x: Double, y: Double, z: Double))
        -> (x: Double, y: Double, z: Double) {
        let t = position
        let d = (x: p.x - t.x, y: p.y - t.y, z: p.z - t.z)
        return (
            at(0, 0) * d.x + at(1, 0) * d.y + at(2, 0) * d.z,
            at(0, 1) * d.x + at(1, 1) * d.y + at(2, 1) * d.z,
            at(0, 2) * d.x + at(1, 2) * d.y + at(2, 2) * d.z
        )
    }

    /// A camera at `height` above the origin, tilted `downDegrees` below the
    /// horizon and rotated `yawDegrees` about the vertical.  For tests and for
    /// the non-LiDAR fallback, where the user types the height instead of
    /// measuring it.
    public static func looking(
        fromHeight height: Double,
        downDegrees pitch: Double,
        yawDegrees yaw: Double = 0,
        atX x: Double = 0,
        atZ z: Double = 0
    ) -> CameraPose {
        let p = -pitch * .pi / 180   // rotation about +X: negative tips the lens down
        let y = yaw * .pi / 180      // rotation about +Y
        let (cp, sp) = (cos(p), sin(p))
        let (cy, sy) = (cos(y), sin(y))

        // R = Ry(yaw) * Rx(pitch), column-major.
        let r00 = cy,            r01 = sy * sp,       r02 = sy * cp
        let r10 = 0.0,           r11 = cp,            r12 = -sp
        let r20 = -sy,           r21 = cy * sp,       r22 = cy * cp

        return CameraPose(columnMajor: [
            r00, r10, r20, 0,
            r01, r11, r21, 0,
            r02, r12, r22, 0,
            x,   height, z, 1,
        ])
    }
}

/// The water surface: a horizontal plane at a known height in ARKit world
/// coordinates, with an honest account of how well that height is known.
public struct WaterPlane: Sendable, Equatable {
    /// World Y of the water surface, metres.
    public var y: Double
    /// 1-sigma scatter of the depth samples the plane was fitted to, metres.
    public var fitResidualM: Double
    /// How many high-confidence LiDAR samples went into the fit.  Zero means
    /// the height was typed in rather than measured.
    public var sampleCount: Int
    /// Whether the samples came from the water surface itself (close range) or
    /// from the waterline on the bank (the usual case beyond ~5 m).
    public var measuredOnWaterSurface: Bool

    public init(
        y: Double,
        fitResidualM: Double = 0.03,
        sampleCount: Int = 0,
        measuredOnWaterSurface: Bool = false
    ) {
        self.y = y
        self.fitResidualM = fitResidualM
        self.sampleCount = sampleCount
        self.measuredOnWaterSurface = measuredOnWaterSurface
    }

    public func describe(cameraHeightM: Double) -> String {
        let how = sampleCount == 0
            ? "camera height entered manually"
            : "\(sampleCount) LiDAR samples on the \(measuredOnWaterSurface ? "water surface" : "waterline")"
        return "gravity-aligned water plane, \(how), "
             + "fit residual \(decimals(fitResidualM * 1000, 0)) mm, "
             + "camera \(decimals(cameraHeightM, 2)) m above the surface"
    }
}

/// Ray/plane rectifier for one camera pose.
///
/// One instance describes the *reference* pose -- the pose at which the user
/// picked the cross-section.  Recording does not use it: the search-line sample
/// points are world points, and each frame projects them through its own pose.
public struct WaterPlaneProjection: Sendable {
    public let intrinsics: PinholeIntrinsics
    public let pose: CameraPose
    public let plane: WaterPlane

    public init(intrinsics: PinholeIntrinsics, pose: CameraPose, plane: WaterPlane) {
        self.intrinsics = intrinsics
        self.pose = pose
        self.plane = plane
    }

    /// Captured-image dimensions, for the in-frame test.
    public var imageWidth: Int { intrinsics.width }
    public var imageHeight: Int { intrinsics.height }

    /// Height of the camera above the water plane, metres.
    public var cameraHeightM: Double { pose.position.y - plane.y }

    /// Where a pixel's ray meets the water plane, or `nil` if it misses.
    public func imageToWorld(_ px: PixelPoint) -> PlanePoint? {
        // Ray direction in the computer-vision convention (x right, y down,
        // z forward), then into ARKit's camera axes (y up, -z forward).
        let cvx = (px.u - intrinsics.cx) / intrinsics.fx
        let cvy = (px.v - intrinsics.cy) / intrinsics.fy
        let direction = pose.rotateToWorld((x: cvx, y: -cvy, z: -1))

        let origin = pose.position
        let height = origin.y - plane.y

        // Two conditions, stated rather than implied.
        //
        // The camera must be above the water.  With a correctly fitted plane it
        // always is, but a failed fit -- a sweep that caught the bank above the
        // phone, or a typed height with the wrong sign -- puts the plane over the
        // camera, and then the algebra below happily intersects *upward* rays and
        // returns a mirrored world that looks entirely plausible.  That is the
        // failure mode this whole file is written to avoid, so it is refused.
        guard height > 0 else { return nil }

        // And the ray must descend toward the plane.  A pixel above the horizon
        // intersects nothing; returning nil is what lets the UI say "that tap is
        // above the waterline" instead of inventing a point at infinity.
        guard direction.y < -1e-9 else { return nil }

        let t = -height / direction.y
        guard t > 0, t.isFinite else { return nil }

        return PlanePoint(x: origin.x + t * direction.x, z: origin.z + t * direction.z)
    }

    /// Where a point on the water plane lands in the image, or `nil` if it is
    /// behind the camera.
    public func worldToImage(_ p: PlanePoint) -> PixelPoint? {
        let camera = pose.toCamera((x: p.x, y: plane.y, z: p.z))
        // Forward is -Z in ARKit's camera space.
        let depth = -camera.z
        guard depth > 1e-6 else { return nil }
        return PixelPoint(
            u: intrinsics.fx * (camera.x / depth) + intrinsics.cx,
            v: intrinsics.fy * (-camera.y / depth) + intrinsics.cy
        )
    }

    /// Horizontal distance from the camera to a point on the plane, metres.
    /// Drives the range-dependent term of the scale uncertainty.
    public func horizontalRange(to p: PlanePoint) -> Double {
        let c = pose.position
        return ((p.x - c.x) * (p.x - c.x) + (p.z - c.z) * (p.z - c.z)).squareRoot()
    }

    /// Human-readable summary, as `Rectifier.describe`.
    public func describe() -> String {
        plane.describe(cameraHeightM: cameraHeightM)
    }

    /// Control points for the desktop export: a grid over the image, each pixel
    /// paired with the world point its ray hits.  Refitting a homography to
    /// these reproduces this mapping exactly, because a plane-to-plane
    /// projective map is what this *is*.
    public func controlPoints(gridSteps: Int = 3) -> [(pixel: PixelPoint, world: PlanePoint)] {
        var out: [(pixel: PixelPoint, world: PlanePoint)] = []
        let w = Double(intrinsics.width - 1)
        let h = Double(intrinsics.height - 1)
        for i in 0...gridSteps {
            for j in 0...gridSteps {
                // Skip the top edge, which in a downward-looking view is where
                // the horizon lives and the intersection is ill-conditioned.
                let px = PixelPoint(u: w * Double(i) / Double(gridSteps),
                                    v: h * (0.35 + 0.65 * Double(j) / Double(gridSteps)))
                if let world = imageToWorld(px) { out.append((px, world)) }
            }
        }
        return out
    }
}
