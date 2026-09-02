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
//  So: taps and overlays go through ARKit; sampling goes through StivKit.  The
//  manual projection is pinned against `ARCamera.projectPoint` in
//  `ARProjectionParityTests`, which has to run on a device.

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
