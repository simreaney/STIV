//  WaterProbe.swift
//  The short burst of frames the automatic water outline is found from.
//
//  `StivKit.detectWaterOutline` compares a handful of frames a fraction of a
//  second apart, each with its own pose.  Pixel buffers cannot leave the
//  delegate call -- ARKit recycles them -- so each frame is copied out as it
//  arrives, reduced six times or so in each direction: water texture moving a
//  few centimetres is still several pixels at that size, a burst costs well
//  under a megabyte, and the comparison runs in a fraction of a second.

import Foundation
import StivKit

/// Collects the burst.  Owned by the capture session's frame queue.
final class WaterProbe {
    /// Five frames, 0.3 s apart: 1.2 s of holding still, after the tap that
    /// started it.
    static let frameCount = 5
    static let interval = 0.3
    /// Frames are reduced to about this width.
    static let targetWidth = 320
    /// Give up this long after the first frame offered -- tracking may never
    /// recover -- and hand over whatever was collected.
    static let deadline = 5.0

    let plane: WaterPlane
    private(set) var frames: [LumaFrame] = []
    private var firstOffered: Double?
    private var nextDue: Double?

    init(plane: WaterPlane) {
        self.plane = plane
    }

    /// Offer one camera frame.  Returns true when the burst is finished --
    /// complete, or out of time.
    func offer(luma: LumaPlane, intrinsics: PinholeIntrinsics, pose: CameraPose,
               timestamp: Double, trackingNormal: Bool) -> Bool {
        if firstOffered == nil { firstOffered = timestamp }
        if timestamp - (firstOffered ?? timestamp) > Self.deadline { return true }
        // The intrinsics describe the captured image; a luma plane of any other
        // size would be reduced into a frame they do not describe.
        guard luma.width == intrinsics.width, luma.height == intrinsics.height else { return false }
        // A pose ARKit does not vouch for would put the plane points in the
        // wrong place, and every patch on the bank would look as if it moved.
        guard trackingNormal, timestamp >= (nextDue ?? timestamp) else { return false }

        let factor = max(1, luma.width / Self.targetWidth)
        let reduced = intrinsics.reduced(by: factor)
        frames.append(LumaFrame(
            width: reduced.width, height: reduced.height,
            pixels: reduce(luma, by: factor, width: reduced.width, height: reduced.height),
            projection: WaterPlaneProjection(intrinsics: reduced, pose: pose, plane: plane),
            timestamp: timestamp))
        nextDue = timestamp + Self.interval
        return frames.count >= Self.frameCount
    }
}

/// Average `factor`-square blocks of `luma` into a `width` x `height` image --
/// the reduction `PinholeIntrinsics.reduced(by:)` describes.
func reduce(_ luma: LumaPlane, by factor: Int, width: Int, height: Int) -> [Float] {
    var out = [Float](repeating: 0, count: width * height)
    let scale = 1 / Float(factor * factor)
    for j in 0..<height {
        for i in 0..<width {
            var sum = 0
            for dv in 0..<factor {
                let row = luma.base + (j * factor + dv) * luma.bytesPerRow + i * factor
                for du in 0..<factor { sum += Int(row[du]) }
            }
            out[j * width + i] = Float(sum) * scale
        }
    }
    return out
}
