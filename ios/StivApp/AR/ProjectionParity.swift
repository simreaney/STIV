//  ProjectionParity.swift
//  A self-check for the one piece of arithmetic that cannot be unit-tested off
//  the device.
//
//  `StivKit.WaterPlaneProjection` projects world points to captured-image pixels
//  by hand, because that is where the luma plane lives and ARKit offers no API
//  that lands there.  Its unit tests pin it against a camera whose answer is
//  known in closed form, which proves the geometry is self-consistent -- but not
//  that its *conventions* match ARKit's.  Get the sign of the camera's -Z
//  forward axis, the downward image `v` axis, or the column-major reading of
//  `intrinsics` wrong, and the tests still pass while every measurement is
//  mirrored.
//
//  ARKit does provide the answer, in the viewport rather than the image:
//  `projectPoint(_:orientation:viewportSize:)` with the sensor's own landscape
//  orientation and a viewport the size of the captured image is that same
//  projection.  Comparing the two is therefore a complete check, and it costs
//  one frame at launch.
//
//  Debug builds only, and logged rather than asserted: a failure here means the
//  numbers are wrong, but crashing on a riverbank helps nobody.

import ARKit
import Foundation
import OSLog
import StivKit
import simd

private let log = Logger(subsystem: "org.stiv.app", category: "geometry")

#if DEBUG
/// Compare the manual projection against ARKit's for a spread of world points.
///
/// Returns the largest disagreement in pixels.  Sub-pixel is a pass; anything
/// larger is a convention mismatch and the message says which one to look at.
@discardableResult
func verifyProjectionParity(frame: ARFrame) -> Double {
    let camera = frame.camera
    let intrinsics = camera.stivIntrinsics
    let resolution = camera.imageResolution

    // A plane two metres below the camera, so the test points are where a water
    // surface would actually be rather than somewhere degenerate.
    let planeY = Double(camera.transform.columns.3.y) - 2.0
    let projection = camera.projection(onto: WaterPlane(y: planeY))
    let origin = projection.pose.position

    var worst = 0.0
    var worstPoint = PlanePoint(x: 0, z: 0)
    var compared = 0

    // Points spread over a patch of the plane in front of the camera.  The
    // camera looks along -Z at start-up, so "in front" is negative Z.
    for dx in stride(from: -6.0, through: 6.0, by: 3.0) {
        for dz in stride(from: -18.0, through: -3.0, by: 3.0) {
            let world = PlanePoint(x: origin.x + dx, z: origin.z + dz)
            guard let mine = projection.worldToImage(world) else { continue }
            guard mine.u >= 0, mine.v >= 0,
                  mine.u <= Double(intrinsics.width), mine.v <= Double(intrinsics.height)
            else { continue }

            // ARKit's projection into a viewport that *is* the captured image,
            // in the sensor's native orientation: the same mapping by
            // definition, so any difference is a convention error in mine.
            let theirs = camera.projectPoint(
                simd_float3(Float(world.x), Float(planeY), Float(world.z)),
                orientation: .landscapeRight,
                viewportSize: resolution)

            let error = hypot(mine.u - Double(theirs.x), mine.v - Double(theirs.y))
            compared += 1
            if error > worst {
                worst = error
                worstPoint = world
            }
        }
    }

    guard compared > 0 else {
        log.warning("projection parity: no test point projected into the frame")
        return .nan
    }

    if worst < 1.0 {
        log.info("""
            projection parity: \(compared) points agree with ARKit to \
            \(worst, format: .fixed(precision: 3)) px
            """)
    } else {
        log.error("""
            projection parity FAILED: worst disagreement \
            \(worst, format: .fixed(precision: 1)) px at world \
            (\(worstPoint.x, format: .fixed(precision: 2)), \
            \(worstPoint.z, format: .fixed(precision: 2))) over \(compared) points. \
            Check, in this order: the camera looks along -Z; image v grows \
            downward so camera y is negated; ARCamera.intrinsics is \
            column-major with the principal point in column 2.
            """)
    }
    return worst
}
#endif
