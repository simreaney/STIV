//  ARAdapterTests.swift
//  The conversions from ARKit's types into StivKit's, and the small pieces of
//  geometry that live in the app rather than the package.
//
//  StivKit's tests prove its geometry is self-consistent.  These prove that the
//  numbers going into it are read the right way round: `simd` matrices are
//  column-major, and a transposed read of either the pose or the intrinsics
//  produces a camera that is wrong without being obviously wrong.  What these
//  cannot prove is that StivKit's axis conventions match ARKit's own -- that
//  needs a real `ARCamera`, and lives in `OnDeviceTests`.

import ARKit
import simd
import StivKit
import XCTest
@testable import STIV

final class ARAdapterTests: XCTestCase {
    // MARK: - simd into StivKit

    /// An arbitrary rigid transform: no axis-aligned rotation, so a transposed
    /// read cannot pass by symmetry.
    private let transform: simd_float4x4 = {
        let rotation = simd_float4x4(simd_quatf(angle: 0.7, axis: simd_normalize(simd_float3(1, 2, 3))))
        var t = rotation
        t.columns.3 = simd_float4(0.5, 1.5, -2.0, 1)
        return t
    }()

    func testPoseIsReadColumnMajorLikeSimd() {
        let pose = CameraPose(transform)
        XCTAssertEqual(pose.position.x, 0.5, accuracy: 1e-6)
        XCTAssertEqual(pose.position.y, 1.5, accuracy: 1e-6)
        XCTAssertEqual(pose.position.z, -2.0, accuracy: 1e-6)

        // The definitive check: StivKit's world-to-camera agrees with simd's
        // own inverse, for points in every direction.
        let inverse = transform.inverse
        for p in [simd_float3(1, 0, 0), simd_float3(0, 1, 0), simd_float3(0, 0, 1),
                  simd_float3(-3, 2, -7), simd_float3(4, -1, 0.5)] {
            let expected = inverse * simd_float4(p, 1)
            let mine = pose.toCamera((Double(p.x), Double(p.y), Double(p.z)))
            XCTAssertEqual(mine.x, Double(expected.x), accuracy: 1e-5, "\(p)")
            XCTAssertEqual(mine.y, Double(expected.y), accuracy: 1e-5, "\(p)")
            XCTAssertEqual(mine.z, Double(expected.z), accuracy: 1e-5, "\(p)")
        }
    }

    /// `ARCamera.intrinsics` keeps the principal point in column 2.  Read
    /// row-major, `cx` and `cy` come out as zero and `K[2][0]` lands nowhere --
    /// distinct values here make every possible mix-up visible.
    func testIntrinsicsReadThePrincipalPointFromColumnTwo() {
        let k = simd_float3x3(columns: (simd_float3(1598, 0, 0),
                                        simd_float3(0, 1601, 0),
                                        simd_float3(955.5, 718.25, 1)))
        let intrinsics = PinholeIntrinsics(k, resolution: CGSize(width: 1920, height: 1440))
        XCTAssertEqual(intrinsics.fx, 1598)
        XCTAssertEqual(intrinsics.fy, 1601)
        XCTAssertEqual(intrinsics.cx, 955.5)
        XCTAssertEqual(intrinsics.cy, 718.25)
        XCTAssertEqual(intrinsics.width, 1920)
        XCTAssertEqual(intrinsics.height, 1440)
    }

    func testTheWaterPlaneTransformIsAPureLift() {
        let m = horizontalPlaneTransform(y: -1.75)
        XCTAssertEqual(m.columns.3, simd_float4(0, -1.75, 0, 1))
        XCTAssertEqual(m.columns.0, simd_float4(1, 0, 0, 0))
        XCTAssertEqual(m.columns.1, simd_float4(0, 1, 0, 0), "+Y must stay the plane normal")
        XCTAssertEqual(m.columns.2, simd_float4(0, 0, 1, 0))
    }

    // MARK: - depth back-projection

    /// LiDAR depth is a Z-depth along the camera's forward axis, not a slant
    /// range, and a depth pixel is a coarse block of image pixels whose centre
    /// is what was measured.
    func testDepthBackProjectionIsAZDepthThroughThePixelCentre() {
        let intrinsics = PinholeIntrinsics(fx: 1590, fy: 1590, cx: 951, cy: 722,
                                           width: 1920, height: 1440)
        let pose = CameraPose.looking(fromHeight: 1.4, downDegrees: 35, yawDegrees: 20,
                                      atX: 0.3, atZ: -0.2)
        let depthSize = (width: 256, height: 192)

        for (x, y, depth) in [(37, 22, Float(2.75)), (128, 96, 1.0), (250, 180, 4.5)] {
            let world = worldPoint(depthPixel: (x, y), depthSize: depthSize, depth: depth,
                                   intrinsics: intrinsics, pose: pose)

            let camera = pose.toCamera(world)
            XCTAssertEqual(-camera.z, Double(depth), accuracy: 1e-9,
                           "depth must be measured along -Z, not along the ray")

            let projection = WaterPlaneProjection(intrinsics: intrinsics, pose: pose,
                                                  plane: WaterPlane(y: world.y))
            let pixel = projection.worldToImage(PlanePoint(x: world.x, z: world.z))!
            XCTAssertEqual(pixel.u, (Double(x) + 0.5) * 7.5, accuracy: 1e-6)
            XCTAssertEqual(pixel.v, (Double(y) + 0.5) * 7.5, accuracy: 1e-6)
        }
    }

    // MARK: - water outline and residual

    func testWaterOutlineContainment() {
        let square = [PixelPoint(u: 0, v: 0), PixelPoint(u: 10, v: 0),
                      PixelPoint(u: 10, v: 10), PixelPoint(u: 0, v: 10)]
        XCTAssertTrue(contains(polygon: square, point: PixelPoint(u: 5, v: 5)))
        XCTAssertFalse(contains(polygon: square, point: PixelPoint(u: 15, v: 5)))
        XCTAssertFalse(contains(polygon: square, point: PixelPoint(u: 5, v: -1)))

        // A U shape, tapped round a bend: the notch is outside.
        let bend = [PixelPoint(u: 0, v: 0), PixelPoint(u: 30, v: 0), PixelPoint(u: 30, v: 30),
                    PixelPoint(u: 20, v: 30), PixelPoint(u: 20, v: 10), PixelPoint(u: 10, v: 10),
                    PixelPoint(u: 10, v: 30), PixelPoint(u: 0, v: 30)]
        XCTAssertTrue(contains(polygon: bend, point: PixelPoint(u: 5, v: 20)))
        XCTAssertTrue(contains(polygon: bend, point: PixelPoint(u: 25, v: 20)))
        XCTAssertFalse(contains(polygon: bend, point: PixelPoint(u: 15, v: 20)))

        XCTAssertFalse(contains(polygon: Array(square.prefix(2)), point: PixelPoint(u: 5, v: 0)),
                       "fewer than three taps is not an outline")
    }

    /// The pose cross-check corrects the *pose*, so its correction has to
    /// compose with the pose the way a camera motion does: the corrected pose
    /// puts every world point at `R(rotation) P + translation` in camera space.
    func testAPoseCorrectionComposesLikeACameraMotion() {
        let pose = CameraPose(transform)
        XCTAssertEqual(PoseCorrection.identity.corrected(pose), pose)
        XCTAssertTrue(PoseCorrection.identity.isIdentity)

        let correction = PoseCorrection(rotation: SIMD3(0.003, -0.002, 0.004),
                                        translation: SIMD3(0.01, -0.02, 0.015))
        let corrected = correction.corrected(pose)
        let w = correction.rotation
        let theta = (w * w).sum().squareRoot()
        let r = simd_double3x3(simd_quatd(angle: theta, axis: w / theta))
        for p in [(1.0, 0.0, -3.0), (-2.0, 1.5, -7.0), (0.3, -0.4, -1.2)] {
            let before = pose.toCamera(p)
            let expected = r * SIMD3(before.x, before.y, before.z) + correction.translation
            let after = corrected.toCamera(p)
            XCTAssertEqual(after.x, expected.x, accuracy: 1e-9)
            XCTAssertEqual(after.y, expected.y, accuracy: 1e-9)
            XCTAssertEqual(after.z, expected.z, accuracy: 1e-9)
        }
    }

    /// A rotation's image shift is the same at every depth; a translation's
    /// falls off as `f t / Z`.  That difference is the whole reason the
    /// correction is a pose and not an image warp.
    func testARotationShiftsEveryDepthAlikeAndATranslationDoesNot() throws {
        let intrinsics = PinholeIntrinsics(horizontalFovDegrees: 60, width: 1920, height: 1440)
        let pose = CameraPose.looking(fromHeight: 0, downDegrees: 0)
        let near = (x: 0.0, y: 0.0, z: -2.0), far = (x: 0.0, y: 0.0, z: -20.0)
        func shift(_ c: PoseCorrection, _ p: (x: Double, y: Double, z: Double)) throws -> Double {
            let a = try XCTUnwrap(projectToImage(p, intrinsics: intrinsics, pose: pose))
            let b = try XCTUnwrap(projectToImage(p, intrinsics: intrinsics, pose: c.corrected(pose)))
            return hypot(b.u - a.u, b.v - a.v)
        }
        let yaw = PoseCorrection(rotation: SIMD3(0, 2 / intrinsics.fx, 0))
        XCTAssertEqual(try shift(yaw, near), 2, accuracy: 0.01)
        XCTAssertEqual(try shift(yaw, far), 2, accuracy: 0.01)
        XCTAssertEqual(yaw.farFieldShiftPx(intrinsics: intrinsics), 2, accuracy: 1.1,
                       "the cap's far-field figure bounds the rotation's shift")

        let slide = PoseCorrection(translation: SIMD3(0.01, 0, 0))
        XCTAssertEqual(try shift(slide, near), intrinsics.fx * 0.01 / 2, accuracy: 1e-6)
        XCTAssertEqual(try shift(slide, far), intrinsics.fx * 0.01 / 20, accuracy: 1e-6)
        XCTAssertEqual(slide.farFieldShiftPx(intrinsics: intrinsics), 0)
    }

    /// Every projection into captured-image pixels goes through
    /// `projectToImage`, which must be StivKit's own and nothing else.
    func testTheSharedProjectionIsStivKits() throws {
        let intrinsics = PinholeIntrinsics(fx: 1590, fy: 1585, cx: 951, cy: 722,
                                           width: 1920, height: 1440)
        let pose = CameraPose(transform)
        for p in [(0.5, -1.0, -6.0), (2.0, 0.4, -3.5), (-1.5, -0.2, -9.0)] {
            let mine = try XCTUnwrap(projectToImage(p, intrinsics: intrinsics, pose: pose))
            let camera = pose.toCamera(p)
            XCTAssertEqual(mine.u, intrinsics.fx * camera.x / -camera.z + intrinsics.cx,
                           accuracy: 1e-9)
            XCTAssertEqual(mine.v, intrinsics.fy * -camera.y / -camera.z + intrinsics.cy,
                           accuracy: 1e-9)
        }
        XCTAssertNil(projectToImage((0.5, 1.5, -2.0), intrinsics: intrinsics, pose: pose),
                     "the camera's own position projects nowhere")
    }

    // MARK: - recording plan

    func testTheRecordingPlanPutsEverySampleOnTheWaterPlane() throws {
        let plane = WaterPlane(y: -2.2, fitResidualM: 0.02, sampleCount: 80,
                               measuredOnWaterSurface: false)
        let intrinsics = PinholeIntrinsics(horizontalFovDegrees: 60, width: 1920, height: 1440)
        let projection = WaterPlaneProjection(
            intrinsics: intrinsics,
            pose: .looking(fromHeight: 0.8, downDegrees: 25),   // 3 m above y = -2.2
            plane: plane)
        var survey = Survey()
        survey.section.start = PlanePoint(x: -4, z: -9)
        survey.section.end = PlanePoint(x: 4, z: -9)
        survey.section.stationCount = 6
        survey.flow.downstream = PlanePoint(x: 0, z: -20)
        let layout = try buildSection(survey: survey.validated(), rectifier: projection)

        let plan = RecordingPlan(survey: survey, plane: plane, layout: layout)
        XCTAssertEqual(plan.sampleCounts, layout.searchLines.map(\.sampleCount))
        XCTAssertEqual(plan.worldPoints.count, plan.sampleCounts.reduce(0, +))
        XCTAssertTrue(plan.worldPoints.allSatisfy { $0.y == plane.y })

        // Flattened in station order, samples in line order.
        var k = 0
        for line in layout.searchLines {
            for sample in line.samples {
                XCTAssertEqual(plan.worldPoints[k].x, sample.x)
                XCTAssertEqual(plan.worldPoints[k].z, sample.z)
                k += 1
            }
        }
    }

    // MARK: - tracking

    func testTrackingAdviceSaysWhatToDo() {
        XCTAssertNil(ARCamera.TrackingState.normal.advice)
        XCTAssertTrue(ARCamera.TrackingState.normal.isNormal)
        XCTAssertFalse(ARCamera.TrackingState.notAvailable.isNormal)
        for state: ARCamera.TrackingState in [.notAvailable, .limited(.initializing),
                                              .limited(.excessiveMotion),
                                              .limited(.insufficientFeatures),
                                              .limited(.relocalizing)] {
            XCTAssertFalse(state.advice?.isEmpty ?? true, "\(state) has no advice")
        }
        XCTAssertTrue(ARCamera.TrackingState.limited(.excessiveMotion).advice!.contains("motion"))
    }
}
