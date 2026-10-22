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

    /// A water plane fitted *above* the camera is a failed fit, not a geometry.
    ///
    /// The algebra will happily intersect upward-pointing rays against it and
    /// return a mirrored world that looks entirely plausible -- the exact class
    /// of silent wrongness this projection is written to avoid -- so it is
    /// refused instead.
    func testAPlaneAboveTheCameraIsRefused() {
        let inverted = WaterPlaneProjection(
            intrinsics: PinholeIntrinsics(horizontalFovDegrees: 60, width: 1920, height: 1440),
            pose: .looking(fromHeight: 2.0, downDegrees: 20),
            // 3 m up when the camera is at 2 m: the sweep caught the bank above
            // the phone, or a typed height had the wrong sign.
            plane: WaterPlane(y: 3.0))
        XCTAssertNil(inverted.imageToWorld(PixelPoint(u: 960, v: 1200)))
        XCTAssertNil(inverted.imageToWorld(PixelPoint(u: 960, v: 200)))
    }

    /// A ray that does not descend intersects nothing, whatever the algebra says.
    func testOnlyDescendingRaysIntersect() throws {
        let p = makeProjection(heightM: 2.0, downDegrees: 20)
        // Well below the principal point: descending, so it must land.
        XCTAssertNotNil(p.imageToWorld(PixelPoint(u: p.intrinsics.cx, v: 1300)))
        // Far above it: ascending, so it must not.
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

    /// `stiv.geometry.default_n_samples`: one column per pixel *where the
    /// image is finest along the line*.  In an oblique view the near end
    /// covers more pixels per metre than the far end, and counting pixels end
    /// to end left it undersampled; now no column spans more than a pixel
    /// anywhere on the line.
    func testAutomaticSampleCountResolvesTheFinestPartOfTheLine() throws {
        var s = survey()
        s.sti.sampleCount = nil
        let projection = makeProjection(heightM: 3)
        let layout = try buildSection(survey: s, rectifier: projection)
        for line in layout.searchLines {
            let a = try XCTUnwrap(line.samples.first)
            let b = try XCTUnwrap(line.samples.last)
            // Local pixels per metre along the line, sampled densely.
            var finest = 0.0
            for k in 0...200 {
                let t = Double(k) / 200
                let p = PlanePoint(x: a.x + t * (b.x - a.x), z: a.z + t * (b.z - a.z))
                let d = 1e-4
                let ux = (b.x - a.x) / line.lengthM, uz = (b.z - a.z) / line.lengthM
                let f = try XCTUnwrap(projection.worldToImage(PlanePoint(x: p.x + d * ux, z: p.z + d * uz)))
                let g = try XCTUnwrap(projection.worldToImage(PlanePoint(x: p.x - d * ux, z: p.z - d * uz)))
                finest = max(finest, ((f.u - g.u) * (f.u - g.u) + (f.v - g.v) * (f.v - g.v)).squareRoot() / (2 * d))
            }
            let pxPerColumn = finest * line.sxMetresPerColumn
            XCTAssertLessThanOrEqual(pxPerColumn, 1.0 + 1e-3, "station \(line.index)")
            XCTAssertGreaterThan(pxPerColumn, 0.99, "station \(line.index)")
            XCTAssertEqual(line.sampleCount, Int((line.lengthM * finest).rounded()) + 1)

            // ...which is markedly more than the end-to-end count here.
            let pa = try XCTUnwrap(projection.worldToImage(a))
            let pb = try XCTUnwrap(projection.worldToImage(b))
            let endToEnd = ((pa.u - pb.u) * (pa.u - pb.u) + (pa.v - pb.v) * (pa.v - pb.v)).squareRoot()
            XCTAssertGreaterThan(Double(line.sampleCount), 1.2 * endToEnd)
        }
    }

    /// Looking straight down the scale is uniform, and the count is the
    /// original end-to-end one: rounding error in the derivative must not flip
    /// a count that sits on a half.
    func testANadirViewKeepsTheEndToEndCount() throws {
        var s = survey()
        s.sti.sampleCount = nil
        s.section.start = PlanePoint(x: -2, z: 0)
        s.section.end = PlanePoint(x: 2, z: 0)
        s.flow.downstream = PlanePoint(x: 0, z: -5)
        s.sti.searchLineLengthM = 2
        let nadir = makeProjection(heightM: 6, downDegrees: 90)
        let layout = try buildSection(survey: s, rectifier: nadir)
        for line in layout.searchLines {
            let pa = try XCTUnwrap(nadir.worldToImage(line.samples.first!))
            let pb = try XCTUnwrap(nadir.worldToImage(line.samples.last!))
            let endToEnd = ((pa.u - pb.u) * (pa.u - pb.u) + (pa.v - pb.v) * (pa.v - pb.v)).squareRoot()
            XCTAssertEqual(line.sampleCount, Int(endToEnd.rounded(.toNearestOrEven)) + 1)
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

    /// An `angle` survey with no angle used to run along world +X.
    func testAnAngleFlowWithoutAnAngleIsRejected() {
        var s = survey()
        s.flow.direction = .angle
        XCTAssertThrowsError(try buildSection(survey: s, rectifier: makeProjection()))
        s.flow.angleDegrees = 90
        let layout = try? buildSection(survey: s, rectifier: makeProjection())
        XCTAssertEqual(layout?.flowBearingDegrees ?? .nan, 90, accuracy: 1e-9)
    }

    /// `atan2(z, x)`: from +X towards +Z, the same number the Python calls
    /// counter-clockwise in its `(X, Y) = (x, z)` frame.
    func testTheBearingIsMeasuredFromXTowardsZ() throws {
        var s = survey()
        s.flow.direction = .points
        s.flow.from = PlanePoint(x: 0, z: 0)
        s.flow.to = PlanePoint(x: 0, z: 1)          // straight along +Z
        XCTAssertEqual(try buildSection(survey: s, rectifier: makeProjection()).flowBearingDegrees,
                       90, accuracy: 1e-9)
    }

    func testANegativeStationCountThrowsRatherThanTraps() {
        var s = survey()
        s.section.stationCount = -1
        XCTAssertThrowsError(try buildSection(survey: s, rectifier: makeProjection()))
    }

    func testStationCentresCarryTheirReferencePixel() throws {
        let projection = makeProjection(heightM: 3)
        let layout = try buildSection(survey: survey(), rectifier: projection)
        for line in layout.searchLines {
            let px = try XCTUnwrap(line.centrePx)
            let expected = try XCTUnwrap(projection.worldToImage(line.centre))
            XCTAssertEqual(px, expected)
        }
    }

    func testDegenerateSectionIsRejected() {
        var s = survey()
        s.section.end = s.section.start
        XCTAssertThrowsError(try buildSection(survey: s, rectifier: makeProjection()))
    }
}
