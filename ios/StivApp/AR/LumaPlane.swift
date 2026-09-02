//  LumaPlane.swift
//  Reading the brightness field g(x, t) straight off the AR frame.
//
//  `ARFrame.capturedImage` is bi-planar full-range YCbCr ('420f' in practice --
//  query it, Apple documents the family and not the constant).  Plane 0 is the
//  full-resolution luma plane, and full-range Rec.601 luma is exactly the scalar
//  brightness that `stiv.video.FrameSource.to_grey` computes from BGR.  So the
//  desktop's colour conversion has no analogue here: the sensor already hands
//  over the quantity STIV wants, and sampling plane 0 directly saves a
//  conversion of every pixel of every frame.
//
//  (The desktop's `channel_weights` -- isolating the red channel for
//  turbidity-driven patterns -- has no equivalent and is dropped.  Recovering it
//  would mean converting the chroma plane too, at which point the saving is
//  gone.)

import CoreVideo
import Foundation

/// A borrowed view of a locked pixel buffer's luma plane.
///
/// Valid only while the buffer is locked, which is why it is a non-escaping
/// value handed to a closure rather than something the recorder holds.
struct LumaPlane {
    let base: UnsafePointer<UInt8>
    let width: Int
    let height: Int
    let bytesPerRow: Int

    /// Bilinear sample at floating-point image coordinates.
    ///
    /// Points outside the frame come back as `nil` so the caller can tell a
    /// genuinely dark pixel from a missing one -- the same distinction
    /// `stiv.sti.sample_bilinear` makes with NaN, and the reason
    /// `SpaceTimeImage.outOfFrameFraction` can be trusted.
    func sample(u: Double, v: Double) -> Float? {
        guard u >= 0, v >= 0, u <= Double(width - 1), v <= Double(height - 1) else { return nil }

        let u0 = Int(u), v0 = Int(v)
        let u1 = min(u0 + 1, width - 1)
        let v1 = min(v0 + 1, height - 1)
        let fu = Float(u - Double(u0))
        let fv = Float(v - Double(v0))

        let row0 = base + v0 * bytesPerRow
        let row1 = base + v1 * bytesPerRow
        let top = Float(row0[u0]) * (1 - fu) + Float(row0[u1]) * fu
        let bottom = Float(row1[u0]) * (1 - fu) + Float(row1[u1]) * fu
        return top * (1 - fv) + bottom * fv
    }

    /// Variance of a square patch, used to reject textureless candidates for
    /// the pose cross-check.  A patch with no texture cannot be aligned, and a
    /// patch that cannot be aligned reports a spurious zero residual -- which
    /// would be worse than reporting nothing.
    func patchVariance(centreU: Int, centreV: Int, half: Int) -> Double {
        var sum = 0.0
        var sumSquares = 0.0
        var count = 0.0
        for v in (centreV - half)...(centreV + half) {
            guard v >= 0, v < height else { continue }
            let row = base + v * bytesPerRow
            for u in (centreU - half)...(centreU + half) {
                guard u >= 0, u < width else { continue }
                let x = Double(row[u])
                sum += x
                sumSquares += x * x
                count += 1
            }
        }
        guard count > 1 else { return 0 }
        let mean = sum / count
        return max(sumSquares / count - mean * mean, 0)
    }
}

/// Lock a pixel buffer's luma plane and hand it to `body`.
///
/// Read-only lock: ARKit owns the buffer and may recycle it after the delegate
/// callback returns, so nothing derived from the pointer may escape.
func withLumaPlane<T>(of pixelBuffer: CVPixelBuffer, _ body: (LumaPlane) -> T) -> T? {
    guard CVPixelBufferGetPlaneCount(pixelBuffer) >= 1 else { return nil }
    guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess else {
        return nil
    }
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

    guard let address = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return nil }
    let plane = LumaPlane(
        base: address.assumingMemoryBound(to: UInt8.self),
        width: CVPixelBufferGetWidthOfPlane(pixelBuffer, 0),
        height: CVPixelBufferGetHeightOfPlane(pixelBuffer, 0),
        // Never assume stride == width: it is padded for alignment.
        bytesPerRow: CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0))
    return body(plane)
}

/// A depth map's Float32 samples with their per-pixel confidence.
struct DepthPlane {
    let depth: UnsafePointer<Float32>
    let confidence: UnsafePointer<UInt8>?
    let width: Int
    let height: Int
    let depthBytesPerRow: Int
    let confidenceBytesPerRow: Int

    /// Metres along the camera's forward axis, or `nil` when the sample is
    /// missing or not high-confidence.
    ///
    /// Only `.high` is accepted.  Medium-confidence LiDAR returns are common on
    /// water -- infrared is absorbed, so the sensor guesses -- and a guessed
    /// depth on the water surface would corrupt the one number the whole scale
    /// depends on.
    func highConfidenceDepth(x: Int, y: Int) -> Float? {
        guard x >= 0, x < width, y >= 0, y < height else { return nil }
        if let confidence {
            let row = confidence + y * confidenceBytesPerRow
            guard row[x] >= 2 else { return nil }   // ARConfidenceLevel.high
        }
        let row = (UnsafeRawPointer(depth) + y * depthBytesPerRow)
            .assumingMemoryBound(to: Float32.self)
        let value = row[x]
        guard value.isFinite, value > 0 else { return nil }
        return value
    }
}

/// Lock a depth map (and its confidence map, if present) and hand both to
/// `body`.  Resolution is read at runtime: Apple documents neither the format
/// nor the size, and 256x192 is an observation about shipped devices rather
/// than a contract.
func withDepthPlane<T>(
    depthMap: CVPixelBuffer,
    confidenceMap: CVPixelBuffer?,
    _ body: (DepthPlane) -> T
) -> T? {
    guard CVPixelBufferGetPixelFormatType(depthMap) == kCVPixelFormatType_DepthFloat32 else {
        return nil
    }
    guard CVPixelBufferLockBaseAddress(depthMap, .readOnly) == kCVReturnSuccess else { return nil }
    defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
    guard let depthAddress = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

    var confidenceAddress: UnsafeMutableRawPointer?
    var confidenceStride = 0
    if let confidenceMap,
       CVPixelBufferLockBaseAddress(confidenceMap, .readOnly) == kCVReturnSuccess {
        confidenceAddress = CVPixelBufferGetBaseAddress(confidenceMap)
        confidenceStride = CVPixelBufferGetBytesPerRow(confidenceMap)
    }
    defer {
        if confidenceAddress != nil, let confidenceMap {
            CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly)
        }
    }

    let plane = DepthPlane(
        depth: depthAddress.assumingMemoryBound(to: Float32.self),
        confidence: confidenceAddress?.assumingMemoryBound(to: UInt8.self),
        width: CVPixelBufferGetWidth(depthMap),
        height: CVPixelBufferGetHeight(depthMap),
        depthBytesPerRow: CVPixelBufferGetBytesPerRow(depthMap),
        confidenceBytesPerRow: confidenceStride)
    return body(plane)
}
