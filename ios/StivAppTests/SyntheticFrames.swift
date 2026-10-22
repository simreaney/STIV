//  SyntheticFrames.swift
//  Camera frames of a scene whose answer is known: the app-side counterpart of
//  StivKit's `Synthetic.swift`.
//
//  StivKit's pipeline tests synthesise space-time images directly, which proves
//  the analysis but skips the step this app adds: turning a stream of posed
//  camera frames into those images.  Here the frames themselves are rendered --
//  a textured plane through a pinhole camera, into the same bi-planar YCbCr
//  buffers ARKit delivers -- so the per-frame code runs on pixels, through the
//  same `withLumaPlane` it uses on the phone.
//
//  An `ARFrame` cannot be constructed outside ARKit, which is why the code under
//  test takes a luma plane, intrinsics and pose rather than a frame.

import CoreVideo
import Foundation
import StivKit
@testable import STIV

// MARK: - pixel buffers

/// A full-range bi-planar YCbCr buffer, the format `ARFrame.capturedImage`
/// arrives in.
///
/// Rows are aligned to 64 bytes, so any width that is not a multiple of 64 gets
/// a padded stride -- which is what makes a test of the stride handling mean
/// something.
func makeYCbCrBuffer(width: Int, height: Int) -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let attributes: [CFString: Any] = [kCVPixelBufferBytesPerRowAlignmentKey: 64]
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                     kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                                     attributes as CFDictionary, &buffer)
    precondition(status == kCVReturnSuccess && buffer != nil,
                 "CVPixelBufferCreate failed: \(status)")
    // Neutral chroma, so the buffer is a valid grey image if anyone looks.
    CVPixelBufferLockBaseAddress(buffer!, [])
    if let chroma = CVPixelBufferGetBaseAddressOfPlane(buffer!, 1) {
        memset(chroma, 128,
               CVPixelBufferGetBytesPerRowOfPlane(buffer!, 1)
               * CVPixelBufferGetHeightOfPlane(buffer!, 1))
    }
    CVPixelBufferUnlockBaseAddress(buffer!, [])
    return buffer!
}

/// Write the luma plane.  Every byte of every row is first set to `padding`,
/// including the bytes past the image width, then `value(u, v)` is written over
/// the image itself -- so a reader that ignored the stride would land on
/// `padding` and say so.
func writeLuma(_ buffer: CVPixelBuffer, padding: UInt8 = 255,
               _ value: (_ u: Int, _ v: Int) -> UInt8) {
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!
        .assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
    let width = CVPixelBufferGetWidthOfPlane(buffer, 0)
    let height = CVPixelBufferGetHeightOfPlane(buffer, 0)
    memset(base, Int32(padding), stride * height)
    for v in 0..<height {
        let row = base + v * stride
        for u in 0..<width { row[u] = value(u, v) }
    }
}

/// A LiDAR-shaped pair: Float32 metres along the camera's forward axis, and an
/// 8-bit `ARConfidenceLevel` per pixel.
func makeDepthBuffers(
    width: Int,
    height: Int,
    _ sample: (_ x: Int, _ y: Int) -> (depth: Float, confidence: UInt8)
) -> (depth: CVPixelBuffer, confidence: CVPixelBuffer) {
    var depth: CVPixelBuffer?
    var confidence: CVPixelBuffer?
    let attributes: [CFString: Any] = [kCVPixelBufferBytesPerRowAlignmentKey: 64]
    CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_DepthFloat32,
                        attributes as CFDictionary, &depth)
    CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent8,
                        attributes as CFDictionary, &confidence)
    precondition(depth != nil && confidence != nil, "could not allocate depth buffers")

    CVPixelBufferLockBaseAddress(depth!, [])
    CVPixelBufferLockBaseAddress(confidence!, [])
    let depthBase = CVPixelBufferGetBaseAddress(depth!)!
    let confidenceBase = CVPixelBufferGetBaseAddress(confidence!)!
        .assumingMemoryBound(to: UInt8.self)
    let depthStride = CVPixelBufferGetBytesPerRow(depth!)
    let confidenceStride = CVPixelBufferGetBytesPerRow(confidence!)
    for y in 0..<height {
        let depthRow = (depthBase + y * depthStride).assumingMemoryBound(to: Float32.self)
        let confidenceRow = confidenceBase + y * confidenceStride
        for x in 0..<width {
            let s = sample(x, y)
            depthRow[x] = s.depth
            confidenceRow[x] = s.confidence
        }
    }
    CVPixelBufferUnlockBaseAddress(confidence!, [])
    CVPixelBufferUnlockBaseAddress(depth!, [])
    return (depth!, confidence!)
}

/// The depth map of a horizontal plane as a camera sees it, at the depth map's
/// own resolution.
///
/// Uses the same pixel-centre convention `worldPoint(depthPixel:...)` reads it
/// back with, so a back-projected sample lands exactly on the plane.  Returns
/// zero depth and zero confidence where the ray misses -- above the horizon.
func planeDepth(x: Int, y: Int, depthSize: (width: Int, height: Int),
                projection: WaterPlaneProjection) -> (depth: Float, confidence: UInt8) {
    let u = (Double(x) + 0.5) * Double(projection.intrinsics.width) / Double(depthSize.width)
    let v = (Double(y) + 0.5) * Double(projection.intrinsics.height) / Double(depthSize.height)
    guard let hit = projection.imageToWorld(PixelPoint(u: u, v: v)) else { return (0, 0) }
    let camera = projection.pose.toCamera((hit.x, projection.plane.y, hit.z))
    return (Float(-camera.z), 2)   // ARConfidenceLevel.high
}

// MARK: - the scene

/// Smooth value noise on a world-space lattice.
///
/// Broadband and deterministic, like the smoothed noise of `syntheticSTI`, and
/// precomputed into a table so it costs a handful of multiplies per pixel:
/// these tests render millions of pixels in a debug build.
struct SurfaceTexture {
    private static let size = 256   // lattice cells per side; the pattern tiles beyond
    private let octaves: [(spacing: Double, amplitude: Double, table: [Float])]

    /// - Parameter spacings: lattice spacing of each octave, metres.  Each
    ///   octave has half the amplitude of the one before.
    init(spacings: [Double], seed: UInt64) {
        var state = seed &+ 0x9E3779B97F4A7C15
        func next() -> Float {
            // splitmix64, as in StivKit's `DeterministicNormals`.
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            z ^= z >> 31
            return Float(Double(z >> 11) / 9007199254740992.0) * 2 - 1
        }
        var octaves: [(Double, Double, [Float])] = []
        for (k, spacing) in spacings.enumerated() {
            let table = (0..<(Self.size * Self.size)).map { _ in next() }
            octaves.append((spacing, pow(0.5, Double(k)), table))
        }
        self.octaves = octaves
    }

    /// Roughly zero-mean, within about +-1.5.
    func value(x: Double, z: Double) -> Double {
        let mask = Self.size - 1
        var total = 0.0
        for octave in octaves {
            let fx = x / octave.spacing
            let fz = z / octave.spacing
            let ix = fx.rounded(.down)
            let iz = fz.rounded(.down)
            let tx = fx - ix, tz = fz - iz
            let sx = tx * tx * (3 - 2 * tx)   // smoothstep: no lattice-line creases
            let sz = tz * tz * (3 - 2 * tz)
            // Two's-complement masking wraps negative cells correctly.
            let i0 = Int(ix) & mask, i1 = (i0 + 1) & mask
            let j0 = Int(iz) & mask, j1 = (j0 + 1) & mask
            let t = octave.table
            let top = Double(t[j0 * Self.size + i0]) * (1 - sx) + Double(t[j0 * Self.size + i1]) * sx
            let bottom = Double(t[j1 * Self.size + i0]) * (1 - sx) + Double(t[j1 * Self.size + i1]) * sx
            total += octave.amplitude * (top * (1 - sz) + bottom * sz)
        }
        return total
    }
}

/// An inclusive pixel rectangle.
struct PixelBox {
    var u: ClosedRange<Int>
    var v: ClosedRange<Int>
}

extension PixelBox {
    /// The box around `points`, grown by `margin` and clipped to the image.
    init(around points: [PixelPoint], margin: Int, width: Int, height: Int) {
        let us = points.map(\.u), vs = points.map(\.v)
        let u0 = max(Int((us.min() ?? 0).rounded(.down)) - margin, 0)
        let u1 = min(Int((us.max() ?? 0).rounded(.up)) + margin, width - 1)
        let v0 = max(Int((vs.min() ?? 0).rounded(.down)) - margin, 0)
        let v1 = min(Int((vs.max() ?? 0).rounded(.up)) + margin, height - 1)
        u = u0...max(u0, u1)
        v = v0...max(v0, v1)
    }
}

/// Render a horizontal textured plane as `pose` sees it.
///
/// - Parameter drift: how far the surface pattern has been carried, world
///   metres -- `velocity * t` for a river, zero for a bank.
/// - Parameter region: render only these pixels and leave the rest flat grey.
///   Flat grey has no pattern, so a sampler that strayed outside the region
///   would see a featureless surface and fail rather than pass.
///
/// Pixel `(u, v)` is rendered along the ray through image point `(u, v)`: the
/// convention `LumaPlane.sample` reads it back with, where integer coordinates
/// are pixel centres.
func renderPlane(
    into buffer: CVPixelBuffer,
    intrinsics: PinholeIntrinsics,
    pose: CameraPose,
    planeY: Double,
    texture: SurfaceTexture,
    drift: (x: Double, z: Double) = (0, 0),
    region: PixelBox? = nil,
    contrast: Double = 80
) {
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!
        .assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
    memset(base, 128, stride * intrinsics.height)

    let box = region ?? PixelBox(u: 0...(intrinsics.width - 1), v: 0...(intrinsics.height - 1))
    // `CameraPose.rotateToWorld` and `WaterPlaneProjection.imageToWorld`,
    // unrolled: the same arithmetic, without a call per pixel.
    let r00 = pose.at(0, 0), r01 = pose.at(0, 1), r02 = pose.at(0, 2)
    let r10 = pose.at(1, 0), r11 = pose.at(1, 1), r12 = pose.at(1, 2)
    let r20 = pose.at(2, 0), r21 = pose.at(2, 1), r22 = pose.at(2, 2)
    let origin = pose.position
    let height = origin.y - planeY

    for v in box.v {
        let row = base + v * stride
        // Camera-space ray (rayX, rayY, -1): image v grows downward, camera y up.
        let rayY = -(Double(v) - intrinsics.cy) / intrinsics.fy
        for u in box.u {
            let rayX = (Double(u) - intrinsics.cx) / intrinsics.fx
            let dy = r10 * rayX + r11 * rayY - r12
            guard height > 0, dy < -1e-9 else { continue }   // sky: stays grey
            let dx = r00 * rayX + r01 * rayY - r02
            let dz = r20 * rayX + r21 * rayY - r22
            let t = -height / dy
            let x = origin.x + t * dx - drift.x
            let z = origin.z + t * dz - drift.z
            let luma = 128 + contrast * texture.value(x: x, z: z)
            row[u] = UInt8(min(max(luma.rounded(), 0), 255))
        }
    }
}

// MARK: - poses

/// A camera rotated about its own vertical axis by `radians`: the pose error a
/// small yaw drift in VIO would produce.  Shifts the image sideways by about
/// `fx * radians` pixels at the centre.
func yawedInCameraFrame(_ pose: CameraPose, radians: Double) -> CameraPose {
    let c = cos(radians), s = sin(radians)
    var m = pose.m
    // R' = R * Ry: new X = c X - s Z, new Z = s X + c Z, Y unchanged.
    for row in 0..<3 {
        let x = pose.at(row, 0), z = pose.at(row, 2)
        m[0 * 4 + row] = c * x - s * z
        m[2 * 4 + row] = s * x + c * z
    }
    return CameraPose(columnMajor: m)
}

/// The sway of a phone held at arm's length: a degree or so of pitch and yaw,
/// a couple of centimetres of translation, at the 0.5-1.5 Hz of a person
/// trying to hold still.  Incommensurate frequencies, so it never repeats
/// within a recording.
func handheld(fromHeight height: Double, downDegrees pitch: Double,
              scale: Double = 1) -> (Double) -> CameraPose {
    { t in
        let w = 2 * Double.pi * t
        return .looking(
            fromHeight: height + scale * 0.015 * sin(0.7 * w),
            downDegrees: pitch + scale * 0.6 * sin(1.3 * w + 0.4),
            yawDegrees: scale * 0.5 * sin(0.9 * w + 1.1),
            atX: scale * 0.02 * sin(0.5 * w + 2.0),
            atZ: scale * 0.02 * sin(0.6 * w + 0.3))
    }
}
