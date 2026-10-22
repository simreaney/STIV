//  ARGeometry.swift
//  Adapting ARKit's camera into StivKit's platform-free geometry.
//
//  Everything here is conversion, deliberately: the geometry itself lives in
//  `StivKit.PlaneProjection`, where it can be tested against a camera whose
//  answer is known in closed form.  This file only fills those types in from an
//  `ARFrame`, and provides the one thing StivKit cannot -- the mapping between
//  the on-screen viewport the user taps and the captured image the STI samples.
//
//  Those two pixel spaces are not the same, and conflating them is the easiest
//  way to get a plausible-looking wrong answer:
//
//  captured image   `ARCamera.intrinsics` and `imageResolution`, in the
//                   sensor's native (landscape) orientation.  This is where the
//                   luma plane lives, so this is where STI sampling happens.
//  viewport         SwiftUI points in the device's current orientation, after
//                   ARKit's display transform.  This is where taps and overlays
//                   live, and ARKit's own `projectPoint`/`unprojectPoint` are
//                   the only sane way to reach it.
//
//  So: taps and overlays go through ARKit; sampling goes through StivKit, by
//  way of `projectToImage` below and nothing else.  That projection is pinned
//  against `ARCamera.projectPoint` by
//  `OnDeviceTests.testTheManualProjectionAgreesWithARKit`, which has to run on
//  a device.
//
//  The viewport has one more trap: it is the *whole* view the feed is drawn
//  into, not the safe area.  The camera feed ignores the safe area, so the
//  overlay, the tap target and the size handed to `projectPoint` must ignore
//  it too, or every marker lands a status bar's height from where it belongs.
//  `CameraFeed` in ARPreview.swift is the one place that is arranged.

import ARKit
import Foundation
import StivKit
import simd
import UIKit

extension CameraPose {
    /// `ARCamera.transform` is the world-from-camera transform, column-major,
    /// with the camera looking down its own -Z axis.  `CameraPose` uses the
    /// same layout and the same convention, so this is a widening copy.
    init(_ transform: simd_float4x4) {
        var m = [Double](repeating: 0, count: 16)
        for column in 0..<4 {
            let c = transform[column]
            m[4 * column + 0] = Double(c.x)
            m[4 * column + 1] = Double(c.y)
            m[4 * column + 2] = Double(c.z)
            m[4 * column + 3] = Double(c.w)
        }
        self.init(columnMajor: m)
    }
}

extension PinholeIntrinsics {
    /// `ARCamera.intrinsics` is a pinhole `K` in captured-image pixels, with the
    /// principal point measured from the top-left of the image and the origin at
    /// the centre of the upper-left pixel.  `simd_float3x3` is column-major, so
    /// the offsets live in column 2.
    init(_ intrinsics: simd_float3x3, resolution: CGSize) {
        self.init(
            fx: Double(intrinsics[0][0]),
            fy: Double(intrinsics[1][1]),
            cx: Double(intrinsics[2][0]),
            cy: Double(intrinsics[2][1]),
            width: Int(resolution.width.rounded()),
            height: Int(resolution.height.rounded()))
    }
}

extension ARCamera {
    var stivIntrinsics: PinholeIntrinsics {
        PinholeIntrinsics(intrinsics, resolution: imageResolution)
    }

    var stivPose: CameraPose { CameraPose(transform) }

    /// The rectifier for this camera pose against a known water surface.
    func projection(onto plane: WaterPlane) -> WaterPlaneProjection {
        WaterPlaneProjection(intrinsics: stivIntrinsics, pose: stivPose, plane: plane)
    }
}

/// One camera view, frozen: the intrinsics and pose a layout is built from.
///
/// A search line's sample count -- one per image pixel along it -- depends on
/// where the camera was when the layout was built, so *which* view that was is
/// a decision, not an accident.  Holding it as a value rather than reaching for
/// `ARCaptureSession.camera` makes the decision visible, and lets the tests
/// build layouts from views of their own; an `ARCamera` cannot be constructed
/// outside ARKit.
struct CameraSnapshot: Equatable {
    let intrinsics: PinholeIntrinsics
    let pose: CameraPose

    init(intrinsics: PinholeIntrinsics, pose: CameraPose) {
        self.intrinsics = intrinsics
        self.pose = pose
    }

    init(_ camera: ARCamera) {
        self.init(intrinsics: camera.stivIntrinsics, pose: camera.stivPose)
    }

    func projection(onto plane: WaterPlane) -> WaterPlaneProjection {
        WaterPlaneProjection(intrinsics: intrinsics, pose: pose, plane: plane)
    }
}

/// Where a 3-D world point lands in the captured image.
///
/// Every projection the app makes into captured-image pixels -- the STI
/// samples, the pose cross-check's patches -- goes through this one function,
/// and it adds no arithmetic of its own: it is StivKit's
/// `WaterPlaneProjection.worldToImage`, on a plane through the point itself.
/// That is the projection `OnDeviceTests` and the debug parity check pin
/// against `ARCamera.projectPoint`, so a convention error cannot hide in a
/// second, unchecked copy.
///
/// Returns `nil` for a point behind the camera.
func projectToImage(
    _ world: (x: Double, y: Double, z: Double),
    intrinsics: PinholeIntrinsics,
    pose: CameraPose
) -> PixelPoint? {
    WaterPlaneProjection(intrinsics: intrinsics, pose: pose, plane: WaterPlane(y: world.y))
        .worldToImage(PlanePoint(x: world.x, z: world.z))
}

/// How the captured image is shown in a viewport: rotated into the interface
/// orientation, then scaled to *fill* the viewport and centred, with whatever
/// overhangs cropped away.
///
/// This is what ARKit's display transform does for `ARSCNView`, and it is why a
/// box drawn at a quarter of the screen is not a quarter of the image: on a
/// phone the screen is far narrower than the 4:3 sensor, so filling it crops
/// the sensor's long axis and magnifies the rest.  Used for the depth reticle,
/// which has to draw exactly the image region `WaterPlaneFitter` samples.
///
/// Overlays and taps still go through `ARCamera.projectPoint` and
/// `unprojectPoint`, which are ARKit's own and need no checking.  This mapping
/// is the app's, so `OnDeviceTests` compares it against `projectPoint` on a
/// phone.  The reticle itself is a centred square, which no rotation can move
/// or reshape -- only the scale matters there, and the scale does not depend
/// on which way round the rotation goes.
struct ViewportGeometry {
    /// Captured-image size in pixels, in the sensor's native landscape
    /// orientation (`ARCamera.imageResolution`).
    let imageSize: CGSize
    /// The viewport in points: the *whole* view the feed is drawn into.
    let viewportSize: CGSize
    let orientation: UIInterfaceOrientation

    private var isPortrait: Bool { orientation.isPortrait }

    /// The image's size once rotated into the interface orientation.
    private var rotatedImageSize: CGSize {
        isPortrait ? CGSize(width: imageSize.height, height: imageSize.width) : imageSize
    }

    /// Points per captured-image pixel: the larger of the two axis ratios,
    /// because the image *fills* the viewport.
    var scale: Double {
        let rotated = rotatedImageSize
        guard rotated.width > 0, rotated.height > 0 else { return 0 }
        return max(viewportSize.width / rotated.width, viewportSize.height / rotated.height)
    }

    private var offset: CGPoint {
        let rotated = rotatedImageSize
        return CGPoint(x: (viewportSize.width - rotated.width * scale) / 2,
                       y: (viewportSize.height - rotated.height * scale) / 2)
    }

    /// Normalised image coordinates to normalised rotated coordinates.
    ///
    /// The sensor's native orientation is landscape-right.  Turning the phone
    /// to portrait turns it a quarter clockwise, so the image's `u` axis ends
    /// up pointing down the screen and its `v` axis pointing left.
    private func rotate(_ u: Double, _ v: Double) -> (x: Double, y: Double) {
        switch orientation {
        case .portrait:           return (1 - v, u)
        case .portraitUpsideDown: return (v, 1 - u)
        case .landscapeLeft:      return (1 - u, 1 - v)
        default:                  return (u, v)          // .landscapeRight
        }
    }

    private func unrotate(_ x: Double, _ y: Double) -> (u: Double, v: Double) {
        switch orientation {
        case .portrait:           return (y, 1 - x)
        case .portraitUpsideDown: return (1 - y, x)
        case .landscapeLeft:      return (1 - x, 1 - y)
        default:                  return (x, y)
        }
    }

    /// Where a captured-image pixel is drawn in the viewport.
    func viewportPoint(image p: PixelPoint) -> CGPoint {
        let r = rotate(p.u / imageSize.width, p.v / imageSize.height)
        let rotated = rotatedImageSize
        return CGPoint(x: offset.x + r.x * rotated.width * scale,
                       y: offset.y + r.y * rotated.height * scale)
    }

    /// Which captured-image pixel is drawn at a viewport point.
    func imagePoint(viewport q: CGPoint) -> PixelPoint {
        let rotated = rotatedImageSize
        let x = (q.x - offset.x) / (rotated.width * scale)
        let y = (q.y - offset.y) / (rotated.height * scale)
        let n = unrotate(x, y)
        return PixelPoint(u: n.u * imageSize.width, v: n.v * imageSize.height)
    }

    /// The viewport rectangle covering an axis-aligned image rectangle.
    func viewportRect(image rect: CGRect) -> CGRect {
        let a = viewportPoint(image: PixelPoint(u: rect.minX, v: rect.minY))
        let b = viewportPoint(image: PixelPoint(u: rect.maxX, v: rect.maxY))
        return CGRect(x: min(a.x, b.x), y: min(a.y, b.y),
                      width: abs(b.x - a.x), height: abs(b.y - a.y))
    }
}

/// The transform `ARCamera.unprojectPoint(_:ontoPlane:…)` wants for a horizontal
/// water surface at height `y`.
///
/// That API intersects the ray with the **xz-plane** of the space this transform
/// defines, taking its +Y axis as the plane normal.  ARKit's world frame is
/// already gravity-aligned with +Y up, so no rotation is needed and the whole
/// water surface is one translation -- which is the entire reason a single LiDAR
/// range is enough to scale a river.
func horizontalPlaneTransform(y: Double) -> simd_float4x4 {
    var m = matrix_identity_float4x4
    m.columns.3 = simd_float4(0, Float(y), 0, 1)
    return m
}

/// Where a viewport tap lands on the water surface.
///
/// Returns `nil` when the ray misses -- a tap above the horizon, or above the
/// far bank's waterline.  Surfacing that as "no point" rather than clamping is
/// what lets the UI say *why* a tap did not take.
func worldPoint(
    viewportPoint: CGPoint,
    camera: ARCamera,
    plane: WaterPlane,
    orientation: UIInterfaceOrientation,
    viewportSize: CGSize
) -> PlanePoint? {
    guard let hit = camera.unprojectPoint(
        viewportPoint,
        ontoPlane: horizontalPlaneTransform(y: plane.y),
        orientation: orientation,
        viewportSize: viewportSize) else { return nil }
    guard hit.x.isFinite, hit.y.isFinite, hit.z.isFinite else { return nil }
    return PlanePoint(x: Double(hit.x), z: Double(hit.z))
}

/// Where a point on the water surface appears in the viewport, for drawing the
/// section and the search lines over the camera feed.
func viewportPoint(
    world: PlanePoint,
    planeY: Double,
    camera: ARCamera,
    orientation: UIInterfaceOrientation,
    viewportSize: CGSize
) -> CGPoint? {
    let point = simd_float3(Float(world.x), Float(planeY), Float(world.z))
    // Behind the camera, projectPoint still returns a finite point; reject those
    // ourselves or the overlay draws a mirrored ghost of the section.
    let local = camera.transform.inverse * simd_float4(point, 1)
    guard -local.z > 0.01 else { return nil }
    let projected = camera.projectPoint(point, orientation: orientation,
                                        viewportSize: viewportSize)
    guard projected.x.isFinite, projected.y.isFinite else { return nil }
    return projected
}

/// Back-project a depth sample to a world point.
///
/// `ARDepthData.depthMap` holds metres along the camera's forward axis -- a
/// Z-depth, not a slant range -- which is the convention Apple's own point-cloud
/// sample uses: `K^-1 (u, v, 1) * depth` has `z = depth` by construction.
///
/// `depthPixel` is in the depth map's own resolution, which is much coarser than
/// the captured image (256x192 against 1920x1440 on shipped devices) but shares
/// its aspect ratio and orientation, so the two are related by a simple scale.
func worldPoint(
    depthPixel: (x: Int, y: Int),
    depthSize: (width: Int, height: Int),
    depth: Float,
    intrinsics: PinholeIntrinsics,
    pose: CameraPose
) -> (x: Double, y: Double, z: Double) {
    let scaleU = Double(intrinsics.width) / Double(depthSize.width)
    let scaleV = Double(intrinsics.height) / Double(depthSize.height)
    // Sample centres, not corners: the half-pixel matters at 256x192, where one
    // depth pixel covers seven and a half image pixels.
    let u = (Double(depthPixel.x) + 0.5) * scaleU
    let v = (Double(depthPixel.y) + 0.5) * scaleV

    let d = Double(depth)
    let cameraSpace = (
        x: (u - intrinsics.cx) / intrinsics.fx * d,
        y: -(v - intrinsics.cy) / intrinsics.fy * d,   // image v grows downward
        z: -d                                          // ARKit looks along -Z
    )
    let world = pose.rotateToWorld(cameraSpace)
    let origin = pose.position
    return (origin.x + world.x, origin.y + world.y, origin.z + world.z)
}
