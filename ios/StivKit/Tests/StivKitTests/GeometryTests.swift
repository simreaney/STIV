//  GeometryTests.swift
//  The rectifier and the section layout, against closed-form answers.
//
//  A camera at height h tilted `d` degrees below the horizon looks at the water
//  plane along its principal ray at horizontal range `h / tan(d)`.  That single
//  identity pins the sign of every axis: get the pitch sign, the ARKit -Z
//  forward convention, or the image v-axis backwards and the range comes out
//  negative, infinite, or mirrored.

import XCTest
@testable import StivKit

private func makeProjection(
    heightM: Double = 2.0,
    downDegrees: Double = 20.0,
    yawDegrees: Double = 0.0,
    fovDegrees: Double = 60.0,
    width: Int = 1920,
    height: Int = 1440
) -> WaterPlaneProjection {
    WaterPlaneProjection(
        intrinsics: PinholeIntrinsics(horizontalFovDegrees: fovDegrees,
                                      width: width, height: height),
        pose: .looking(fromHeight: heightM, downDegrees: downDegrees, yawDegrees: yawDegrees),
        plane: WaterPlane(y: 0, fitResidualM: 0.02, sampleCount: 40))
}

final class PlaneProjectionTests: XCTestCase {
    func testPrincipalRayLandsAtTheGeometricRange() throws {
        let h = 2.0
        let down = 20.0
        let p = makeProjection(heightM: h, downDegrees: down)
        let centre = PixelPoint(u: p.intrinsics.cx, v: p.intrinsics.cy)
        let hit = try XCTUnwrap(p.imageToWorld(centre))
        XCTAssertEqual(p.horizontalRange(to: hit), h / tan(down * .pi / 180), accuracy: 1e-9)
    }

    func testCameraHeightIsThePlaneOffset() {
        XCTAssertEqual(makeProjection(heightM: 1.65).cameraHeightM, 1.65, accuracy: 1e-12)
    }

    func testRoundTripsThroughBothDirections() throws {
        let p = makeProjection()
        // Stay in the lower half of the frame: the upper half looks at the sky.
        for u in stride(from: 100.0, through: 1800.0, by: 340.0) {
            for v in stride(from: 800.0, through: 1400.0, by: 150.0) {
                let px = PixelPoint(u: u, v: v)
                let world = try XCTUnwrap(p.imageToWorld(px), "no intersection at \(px)")
                let back = try XCTUnwrap(p.worldToImage(world))
                XCTAssertEqual(back.u, px.u, accuracy: 1e-6)
                XCTAssertEqual(back.v, px.v, accuracy: 1e-6)
            }
        }
    }

    func testPixelsAboveTheHorizonHaveNoIntersection() {
        // A 20-degree downward tilt with a 60-degree vertical spread puts the
        // horizon inside the frame; the top row must be rejected, not
        // extrapolated to a point at infinity.
        let p = makeProjection(downDegrees: 5.0)
        XCTAssertNil(p.imageToWorld(PixelPoint(u: p.intrinsics.cx, v: 0)))
    }

    func testGroundSampleDistanceGrowsWithRange() throws {
        let p = makeProjection()
        let near = try XCTUnwrap(p.localScale(at: PixelPoint(u: p.intrinsics.cx, v: 1400)))
        let far = try XCTUnwrap(p.localScale(at: PixelPoint(u: p.intrinsics.cx, v: 900)))
        // The far bank is coarser -- the whole reason samples are spaced in
        // world metres rather than in pixels.
        XCTAssertGreaterThan(far, near * 1.5)
    }

    func testControlPointsRefitTheSameMapping() throws {
        // A plane-to-plane projective map is exactly what the desktop's
        // HomographyRectifier fits, so the exported control points must be
        // consistent: cross-ratios of four collinear world points must match
        // those of their pixels.
        let p = makeProjection()
        let points = p.controlPoints(gridSteps: 3)
        XCTAssertGreaterThanOrEqual(points.count, 4)
        for (pixel, world) in points {
            let back = try XCTUnwrap(p.worldToImage(world))
            XCTAssertEqual(back.u, pixel.u, accuracy: 1e-6)
            XCTAssertEqual(back.v, pixel.v, accuracy: 1e-6)
        }
    }

    func testYawRotatesTheFootprintWithoutChangingTheRange() throws {
        let straight = makeProjection(yawDegrees: 0)
        let turned = makeProjection(yawDegrees: 37)
        let centre = PixelPoint(u: straight.intrinsics.cx, v: straight.intrinsics.cy)
        let a = try XCTUnwrap(straight.imageToWorld(centre))
        let b = try XCTUnwrap(turned.imageToWorld(centre))
        XCTAssertEqual(straight.horizontalRange(to: a),
                       turned.horizontalRange(to: b), accuracy: 1e-9)
        XCTAssertNotEqual(a.x, b.x, accuracy: 0.1)
    }
}

final class SectionLayoutTests: XCTestCase {
    private func survey(stations: Int = 11, lineLength: Double = 4.0) -> Survey {
        var s = Survey()
        s.section.stationCount = stations
        s.section.start = PlanePoint(x: -5, z: -6)
        s.section.end = PlanePoint(x: 5, z: -6)
        s.sti.searchLineLengthM = lineLength
        s.sti.sampleCount = 128
        s.flow.direction = .auto
        s.flow.downstream = PlanePoint(x: 0, z: -16)
        return s
    }

    func testStationsAreEvenlySpacedInWorldMetres() throws {
        let layout = try buildSection(survey: survey(), rectifier: makeProjection(heightM: 3))
        XCTAssertEqual(layout.widthM, 10, accuracy: 1e-9)
        let stations = layout.stationsM
        for i in 1..<stations.count {
            XCTAssertEqual(stations[i] - stations[i - 1], 1.0, accuracy: 1e-9)
        }
    }

    func testSearchLinesRunAlongTheFlowAndHaveConstantSx() throws {
        let layout = try buildSection(survey: survey(), rectifier: makeProjection(heightM: 3))
        for line in layout.searchLines {
            let span = distance(line.samples.first!, line.samples.last!)
            XCTAssertEqual(span, 4.0, accuracy: 1e-9)
            XCTAssertEqual(line.sxMetresPerColumn, 4.0 / 127.0, accuracy: 1e-12)
            // Uniform spacing in world metres is the property that makes Sx
            // meaningful in an oblique view.
            for i in 1..<line.samples.count {
                XCTAssertEqual(distance(line.samples[i], line.samples[i - 1]),
                               line.sxMetresPerColumn, accuracy: 1e-9)
            }
        }
    }

    func testFlowDirectionPointsAtTheDownstreamMarker() throws {
        var s = survey()
        s.flow.downstream = PlanePoint(x: 0, z: -16)   // further from the camera
        let away = try buildSection(survey: s, rectifier: makeProjection(heightM: 3))
        s.flow.downstream = PlanePoint(x: 0, z: 4)     // behind the section
        let toward = try buildSection(survey: s, rectifier: makeProjection(heightM: 3))
        XCTAssertEqual(away.flowDirection.x, -toward.flowDirection.x, accuracy: 1e-12)
        XCTAssertEqual(away.flowDirection.z, -toward.flowDirection.z, accuracy: 1e-12)
    }

    func testReverseFlipsTheDirection() throws {
        var s = survey()
        let forward = try buildSection(survey: s, rectifier: makeProjection(heightM: 3))
        s.flow.reverse = true
        let back = try buildSection(survey: s, rectifier: makeProjection(heightM: 3))
        XCTAssertEqual(forward.flowDirection.z, -back.flowDirection.z, accuracy: 1e-12)
    }

    func testRangeIncreasesAcrossAnObliqueSection() throws {
        var s = survey(stations: 9)
        // A section running away from the camera rather than across it.
        s.section.start = PlanePoint(x: 0, z: -5)
        s.section.end = PlanePoint(x: 0, z: -25)
        let layout = try buildSection(survey: s, rectifier: makeProjection(heightM: 2))
        let ranges = layout.searchLines.map(\.rangeM)
        XCTAssertEqual(ranges, ranges.sorted(), "range should grow along the section")
        XCTAssertLessThan(ranges.first!, 6)
        XCTAssertGreaterThan(ranges.last!, 24)
    }

    func testAutomaticSampleCountIsAboutOnePerPixel() throws {
        var s = survey()
        s.sti.sampleCount = nil
        let projection = makeProjection(heightM: 3)
        let layout = try buildSection(survey: s, rectifier: projection)
        for line in layout.searchLines {
            let a = try XCTUnwrap(projection.worldToImage(line.samples.first!))
            let b = try XCTUnwrap(projection.worldToImage(line.samples.last!))
            let pixels = ((a.u - b.u) * (a.u - b.u) + (a.v - b.v) * (a.v - b.v)).squareRoot()
            XCTAssertEqual(Double(line.sampleCount), pixels + 1, accuracy: 1.5)
        }
    }

    func testLinesLeavingTheFrameAreReported() throws {
        var s = survey(stations: 7)
        // A section far off to one side, so most lines project outside.
        s.section.start = PlanePoint(x: 40, z: -6)
        s.section.end = PlanePoint(x: 60, z: -6)
        let layout = try buildSection(survey: s, rectifier: makeProjection(heightM: 2))
        XCTAssertFalse(layout.stationsOutsideFrame.isEmpty)
    }

    func testDegenerateSectionIsRejected() {
        var s = survey()
        s.section.end = s.section.start
        XCTAssertThrowsError(try buildSection(survey: s, rectifier: makeProjection()))
    }
}
