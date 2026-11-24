//  WaterOutlineTests.swift
//  The automatic water outline, on a river whose banks are known.
//
//  The scene: a hand-held phone 3 m above the water on the near bank, looking
//  25 degrees down across a channel 9 m wide.  The water's pattern drifts
//  downstream; both banks stand 0.4 m above it, so they are *not* on the water
//  plane and show the parallax a real bank does while the hand sways.

import XCTest
@testable import StivKit

/// Smooth value noise on a world-space lattice, as the app tests' texture.
private struct ValueNoise {
    private static let size = 128
    private let octaves: [(spacing: Double, amplitude: Double, table: [Double])]

    init(spacings: [Double], seed: UInt64) {
        var normals = DeterministicNormals(seed: seed)
        octaves = spacings.enumerated().map { k, spacing in
            (spacing, pow(0.6, Double(k)), normals.array(Self.size * Self.size))
        }
    }

    func value(x: Double, z: Double) -> Double {
        let mask = Self.size - 1
        var total = 0.0
        for octave in octaves {
            let fx = x / octave.spacing, fz = z / octave.spacing
            let ix = fx.rounded(.down), iz = fz.rounded(.down)
            let tx = fx - ix, tz = fz - iz
            let sx = tx * tx * (3 - 2 * tx), sz = tz * tz * (3 - 2 * tz)
            let i0 = Int(ix) & mask, i1 = (i0 + 1) & mask
            let j0 = Int(iz) & mask, j1 = (j0 + 1) & mask
            let t = octave.table
            let top = t[j0 * Self.size + i0] * (1 - sx) + t[j0 * Self.size + i1] * sx
            let bottom = t[j1 * Self.size + i0] * (1 - sx) + t[j1 * Self.size + i1] * sx
            total += octave.amplitude * (top * (1 - sz) + bottom * sz)
        }
        return total
    }
}

private struct RiverScene {
    var nearBankZ = -5.0          // the water runs from here...
    var farBankZ = -14.0          // ...to here
    var bankHeight = 0.4
    var velocity = 0.8            // m/s along +x
    let water = ValueNoise(spacings: [0.5, 0.2, 0.08], seed: 11)
    let bank = ValueNoise(spacings: [1.5, 0.6, 0.25], seed: 29)

    func isWater(_ p: PlanePoint) -> Bool { p.z < nearBankZ && p.z > farBankZ }

    /// The brightness seen along the ray through `(u, v)` at time `t`.
    func luma(_ u: Double, _ v: Double, intrinsics k: PinholeIntrinsics,
              pose: CameraPose, t: Double) -> Float {
        let ray = pose.rotateToWorld((x: (u - k.cx) / k.fx, y: -(v - k.cy) / k.fy, z: -1))
        let o = pose.position
        guard ray.y < -1e-9 else { return 128 }                    // sky: flat
        func hit(_ y: Double) -> (x: Double, z: Double, range: Double) {
            let s = (y - o.y) / ray.y
            return (o.x + s * ray.x, o.z + s * ray.z, s)
        }
        let top = hit(bankHeight)
        if top.range > 40 { return 128 }                           // far haze: flat
        if top.z >= nearBankZ || top.z <= farBankZ {
            return Float(110 + 45 * bank.value(x: top.x, z: top.z))
        }
        let surface = hit(0)
        if surface.z <= farBankZ {                                 // the far bank's face
            let s = (farBankZ - o.z) / ray.z
            return Float(100 + 45 * bank.value(x: o.x + s * ray.x, z: 3 * (o.y + s * ray.y)))
        }
        return Float(128 + 50 * water.value(x: surface.x - velocity * t, z: surface.z))
    }

    /// A frame, supersampled 2x2 as a lens and sensor would average, with
    /// sensor noise; it is handed over with `reported`, the pose the phone
    /// *thinks* it had, which tracking error puts slightly off the true one.
    func frame(at t: Double, pose: CameraPose, reported: CameraPose, plane: WaterPlane,
               intrinsics k: PinholeIntrinsics, noise: inout DeterministicNormals) -> LumaFrame {
        var pixels = [Float](repeating: 0, count: k.width * k.height)
        for v in 0..<k.height {
            for u in 0..<k.width {
                var sum: Float = 0
                for (du, dv) in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] {
                    sum += luma(Double(u) + du, Double(v) + dv, intrinsics: k, pose: pose, t: t)
                }
                pixels[v * k.width + u] = sum / 4 + Float(2 * noise.next())
            }
        }
        return LumaFrame(width: k.width, height: k.height, pixels: pixels,
                         projection: WaterPlaneProjection(intrinsics: k, pose: reported,
                                                          plane: plane),
                         timestamp: t)
    }

    /// The burst the app collects: five frames 0.3 s apart, hand-held, at the
    /// reduced size it analyses (a 1920-wide frame reduced six times is 320).
    ///
    /// The reported poses carry a tracking error of up to 0.1 degree -- a
    /// third of a reduced pixel, two full-resolution pixels -- which is what
    /// the pose cross-check exists to catch, and more than it usually finds.
    func burst(plane: WaterPlane = WaterPlane(y: 0)) -> [LumaFrame] {
        let k = PinholeIntrinsics(horizontalFovDegrees: 60, width: 240, height: 180)
        var noise = DeterministicNormals(seed: 5)
        return (0..<5).map { i in
            let t = 0.3 * Double(i)
            let w = 2 * Double.pi * t
            func pose(yawError: Double) -> CameraPose {
                .looking(fromHeight: 3 + 0.015 * sin(0.7 * w),
                         downDegrees: 25 + 0.6 * sin(1.3 * w + 0.4) + 0.5 * yawError,
                         yawDegrees: 0.5 * sin(0.9 * w + 1.1) + yawError,
                         atX: 0.02 * sin(0.5 * w + 2.0),
                         atZ: 0.02 * sin(0.6 * w + 0.3))
            }
            let error = 0.1 * sin(3.7 * Double(i) + 0.5)
            return frame(at: t, pose: pose(yawError: 0), reported: pose(yawError: error),
                         plane: plane, intrinsics: k, noise: &noise)
        }
    }
}

final class WaterOutlineTests: XCTestCase {
    private let start = PlanePoint(x: -1, z: -5)       // near bank, at the water's edge
    private let end = PlanePoint(x: 1, z: -14)         // far bank, at the water's edge
    private let downstream = PlanePoint(x: 6, z: -9)   // the river runs toward +x

    /// Points on a grid that the first frame sees, split by where they are.
    private func visiblePoints(_ frames: [LumaFrame], scene: RiverScene)
        -> (water: [PlanePoint], bank: [PlanePoint]) {
        let reference = frames[0].projection
        var water: [PlanePoint] = [], bank: [PlanePoint] = []
        for x in stride(from: -12.0, through: 12.0, by: 0.5) {
            for z in stride(from: -30.0, through: -2.5, by: 0.25) {
                let p = PlanePoint(x: x, z: z)
                guard let px = reference.worldToImage(p), px.u >= 8, px.v >= 8,
                      px.u < Double(reference.imageWidth - 8),
                      px.v < Double(reference.imageHeight - 8) else { continue }
                // A metre clear of the waterline either way: the outline is
                // grown by a cell on purpose, and a cell is ~0.4 m out there.
                if p.z < scene.nearBankZ - 1, p.z > scene.farBankZ + 1 { water.append(p) }
                if p.z > scene.nearBankZ + 1.5 || p.z < scene.farBankZ - 2 { bank.append(p) }
            }
        }
        return (water, bank)
    }

    func testOutlinesTheWaterAndNotTheBanks() throws {
        let scene = RiverScene()
        let frames = scene.burst()
        let outline = detectWaterOutline(frames: frames, sectionStart: start, sectionEnd: end,
                                         downstream: downstream)
        XCTAssertEqual(outline.confidence, .good, outline.message)
        XCTAssertGreaterThanOrEqual(outline.polygon.count, 3)
        XCTAssertLessThanOrEqual(outline.polygon.count, 48)
        XCTAssertGreaterThan(outline.waterChange, 4 * outline.bankChange)

        let (water, bank) = visiblePoints(frames, scene: scene)
        XCTAssertGreaterThan(water.count, 100)
        XCTAssertGreaterThan(bank.count, 100)
        let waterIn = Double(water.filter { contains(polygon: outline.polygon, point: $0) }.count)
        let bankIn = Double(bank.filter { contains(polygon: outline.polygon, point: $0) }.count)
        XCTAssertGreaterThan(waterIn / Double(water.count), 0.9, "water left outside the outline")
        XCTAssertLessThan(bankIn / Double(bank.count), 0.1, "bank inside the outline")
        XCTAssertEqual(outline.flowAgrees, true)
    }

    /// A slow river -- 0.3 m/s -- moves the pattern only a pixel or two
    /// between frames.  It is still found where it matters: within LiDAR range
    /// of the phone, which is the only place the pose cross-check puts patches.
    func testASlowRiverIsStillOutlinedNearThePhone() {
        var scene = RiverScene()
        scene.velocity = 0.3
        let frames = scene.burst()
        let outline = detectWaterOutline(frames: frames, sectionStart: start, sectionEnd: end,
                                         downstream: downstream)
        XCTAssertEqual(outline.confidence, .good, outline.message)
        let camera = frames[0].projection
        let near = visiblePoints(frames, scene: scene).water
            .filter { camera.horizontalRange(to: $0) < 9 }
        XCTAssertGreaterThan(near.count, 30)
        let inside = Double(near.filter { contains(polygon: outline.polygon, point: $0) }.count)
        XCTAssertGreaterThan(inside / Double(near.count), 0.9)
        XCTAssertEqual(outline.flowAgrees, true)
    }

    func testADownstreamPointOnTheWrongSideIsCaught() {
        let outline = detectWaterOutline(frames: RiverScene().burst(), sectionStart: start,
                                         sectionEnd: end, downstream: PlanePoint(x: -6, z: -9))
        XCTAssertTrue(outline.ok, outline.message)
        XCTAssertEqual(outline.flowAgrees, false)
    }

    func testStillWaterIsRefusedRatherThanGuessed() {
        var scene = RiverScene()
        scene.velocity = 0
        let outline = detectWaterOutline(frames: scene.burst(), sectionStart: start,
                                         sectionEnd: end, downstream: downstream)
        XCTAssertFalse(outline.ok)
        XCTAssertEqual(outline.confidence, .failed)
        XCTAssertTrue(outline.polygon.isEmpty)
        XCTAssertTrue(outline.message.contains("does not change"), outline.message)
    }

    func testOneFrameIsNotEnough() {
        let outline = detectWaterOutline(frames: Array(RiverScene().burst().prefix(1)),
                                         sectionStart: start, sectionEnd: end, downstream: nil)
        XCTAssertEqual(outline.confidence, .failed)
    }

    func testReducedIntrinsicsKeepPixelCentres() throws {
        let full = PinholeIntrinsics(horizontalFovDegrees: 63, width: 1920, height: 1440)
        let reduced = full.reduced(by: 6)
        XCTAssertEqual(reduced.width, 320)
        XCTAssertEqual(reduced.height, 240)
        let plane = WaterPlane(y: 0)
        let pose = CameraPose.looking(fromHeight: 2, downDegrees: 30)
        let a = WaterPlaneProjection(intrinsics: full, pose: pose, plane: plane)
        let b = WaterPlaneProjection(intrinsics: reduced, pose: pose, plane: plane)
        let p = PlanePoint(x: 1.3, z: -4.2)
        let u = try XCTUnwrap(a.worldToImage(p)), w = try XCTUnwrap(b.worldToImage(p))
        XCTAssertEqual(w.u, (u.u + 0.5) / 6 - 0.5, accuracy: 1e-9)
        XCTAssertEqual(w.v, (u.v + 0.5) / 6 - 0.5, accuracy: 1e-9)
    }

    func testPointInPolygonOnThePlane() {
        let square = [PlanePoint(x: 0, z: 0), PlanePoint(x: 2, z: 0),
                      PlanePoint(x: 2, z: 2), PlanePoint(x: 0, z: 2)]
        XCTAssertTrue(contains(polygon: square, point: PlanePoint(x: 1, z: 1)))
        XCTAssertFalse(contains(polygon: square, point: PlanePoint(x: 3, z: 1)))
    }
}
