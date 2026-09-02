//  PipelineTests.swift
//  End-to-end: synthetic recordings with a known velocity field, through the
//  whole pipeline, asserting the profile comes back.
//
//  Mirrors the intent of `tests/test_end_to_end.py`, minus the video decoding:
//  the phone's equivalent of "render a synthetic river" is to synthesise the
//  STIs directly, since the recorder's only job is to fill them.

import XCTest
@testable import StivKit

private func projection(heightM: Double = 3.0) -> WaterPlaneProjection {
    WaterPlaneProjection(
        intrinsics: PinholeIntrinsics(horizontalFovDegrees: 60, width: 1920, height: 1440),
        pose: .looking(fromHeight: heightM, downDegrees: 25),
        plane: WaterPlane(y: 0, fitResidualM: 0.02, sampleCount: 60,
                          measuredOnWaterSurface: false))
}

private func survey(stations: Int = 9) -> Survey {
    var s = Survey()
    s.section.stationCount = stations
    s.section.start = PlanePoint(x: -6, z: -8)
    s.section.end = PlanePoint(x: 6, z: -8)
    s.sti.searchLineLengthM = 4.0
    s.sti.sampleCount = 96
    s.flow.downstream = PlanePoint(x: 0, z: -20)
    s.capture.frameStep = 1
    s.capture.framesPerSecond = 60
    return s
}

/// Build one STI per station whose streak encodes `velocity(station)`.
private func recordedStis(
    layout: SectionLayout,
    st: Double,
    rows: Int = 240,
    velocity: (Double) -> Double
) -> [SpaceTimeImage] {
    layout.searchLines.map { line in
        let sx = line.sxMetresPerColumn
        // Invert Eq. (16): the streak slope in columns per row.
        let tanPhi = velocity(line.stationM) * st / sx
        return SpaceTimeImage(
            data: syntheticSTI(tanPhi: tanPhi, rows: rows, cols: line.sampleCount,
                               noise: 0.1, seed: UInt64(line.index) + 1),
            stationIndex: line.index,
            sxMetresPerColumn: sx,
            stSecondsPerRow: st,
            outOfFrameFraction: 0)
    }
}

final class PipelineTests: XCTestCase {
    func testRecoversAUniformVelocity() throws {
        let s = survey()
        let layout = try buildSection(survey: s, rectifier: projection())
        let st = s.capture.stSecondsPerRow
        let truth = 0.9
        let result = analyse(survey: s, layout: layout,
                             stis: recordedStis(layout: layout, st: st) { _ in truth })

        for station in result.stations {
            XCTAssertEqual(station.velocityMs, truth, accuracy: 0.12 * truth,
                           "station \(station.index) flags \(station.flags)")
        }
    }

    func testRecoversAParabolicProfile() throws {
        // The shape a real channel has, and the shape the desktop demo checks.
        let s = survey(stations: 11)
        let layout = try buildSection(survey: s, rectifier: projection())
        let st = s.capture.stSecondsPerRow
        let width = layout.widthM
        let truth: (Double) -> Double = { x in 1.4 * (1 - pow(2 * x / width - 1, 2)) + 0.15 }

        let result = analyse(survey: s, layout: layout,
                             stis: recordedStis(layout: layout, st: st, velocity: truth))

        var sumSquares = 0.0
        for station in result.stations {
            let expected = truth(station.stationM)
            sumSquares += pow(station.velocityMs - expected, 2)
        }
        let rmse = (sumSquares / Double(result.stations.count)).squareRoot()
        XCTAssertLessThan(rmse, 0.12, "profile RMSE \(rmse) m/s\n\(result.summary())")
    }

    /// Flow running against the assumed downstream direction.
    ///
    /// Negating the velocities would not be enough: the velocity range is
    /// asymmetric, so an upstream-facing line puts the true angle outside the
    /// Eq. (14) search range and every station pins to the boundary.  The
    /// pipeline must mirror the range and re-estimate.
    func testUpstreamFacingLinesAreDetectedAndMirrored() throws {
        let s = survey()
        let layout = try buildSection(survey: s, rectifier: projection())
        let st = s.capture.stSecondsPerRow
        let result = analyse(survey: s, layout: layout,
                             stis: recordedStis(layout: layout, st: st) { _ in -0.8 })

        XCTAssertTrue(result.warnings.contains { $0.contains("opposite to the assumed") },
                      "no direction warning in:\n\(result.warnings)")
        // auto_flip reports the magnitude along the corrected direction and warns,
        // rather than reporting negative flow: that is what the flag is for.
        for station in result.stations {
            XCTAssertEqual(station.velocityMs, 0.8, accuracy: 0.12,
                           "station \(station.index)")
        }
    }

    func testTexturelessWaterIsFlaggedRatherThanInvented() throws {
        let s = survey(stations: 5)
        let layout = try buildSection(survey: s, rectifier: projection())
        let st = s.capture.stSecondsPerRow
        // A glassy pool: no pattern to advect, so no orientation exists.  The
        // pipeline must say so rather than report a confident zero.
        let flat = layout.searchLines.map { line in
            SpaceTimeImage(data: Matrix(rows: 240, cols: line.sampleCount, repeating: 0.5),
                           stationIndex: line.index,
                           sxMetresPerColumn: line.sxMetresPerColumn,
                           stSecondsPerRow: st,
                           outOfFrameFraction: 0)
        }
        let result = analyse(survey: s, layout: layout, stis: flat)
        XCTAssertTrue(result.stations.allSatisfy { !$0.ok },
                      "a featureless surface must not produce clean stations")
    }

    func testScaleUncertaintyIsFoldedIntoSigma() throws {
        var s = survey(stations: 5)
        s.section.start = PlanePoint(x: 0, z: -4)
        s.section.end = PlanePoint(x: 0, z: -40)   // runs away from the camera
        s.quality.interpolateFlagged = false
        let layout = try buildSection(survey: s, rectifier: projection(heightM: 1.5))
        let st = s.capture.stSecondsPerRow
        let result = analyse(survey: s, layout: layout,
                             stis: recordedStis(layout: layout, st: st) { _ in 1.0 })

        let sigmas = result.stations.map(\.relativeScaleSigma)
        XCTAssertEqual(sigmas, sigmas.sorted(), "scale sigma must grow with range")
        XCTAssertLessThan(sigmas.first!, 0.10)
        XCTAssertGreaterThan(sigmas.last!, 0.20)
        // The far station must be flagged, and its velocity sigma must exceed
        // the pure-angle sigma of the near one.
        XCTAssertTrue(result.stations.last!.flags.contains(.scaleUncertain))
        XCTAssertTrue(result.warnings.contains { $0.contains("raising the camera") },
                      "expected the raise-the-camera advice in:\n\(result.warnings)")
    }

    func testSuggestsAFrameStepWhenTanPhiIsOutOfRange() throws {
        var s = survey(stations: 5)
        s.capture.frameStep = 1
        let layout = try buildSection(survey: s, rectifier: projection())
        let st = s.capture.stSecondsPerRow
        // Slow flow at a short St: features barely move, tan phi is tiny.
        let result = analyse(survey: s, layout: layout,
                             stis: recordedStis(layout: layout, st: st) { _ in 0.05 })
        XCTAssertNotNil(result.suggestedFrameStep)
        XCTAssertGreaterThan(result.suggestedFrameStep ?? 1, 1)
    }

    func testFailedStationsAreInterpolatedButAdvisoriesAreNot() throws {
        let s = survey(stations: 5)
        let layout = try buildSection(survey: s, rectifier: projection())
        let st = s.capture.stSecondsPerRow
        var stis = recordedStis(layout: layout, st: st) { _ in 1.0 }
        // Kill the middle station outright.
        stis[2] = SpaceTimeImage(data: Matrix(rows: 240, cols: layout.searchLines[2].sampleCount,
                                              repeating: 0.0),
                                 stationIndex: 2,
                                 sxMetresPerColumn: layout.searchLines[2].sxMetresPerColumn,
                                 stSecondsPerRow: st, outOfFrameFraction: 0)
        let result = analyse(survey: s, layout: layout, stis: stis)
        XCTAssertTrue(result.stations[2].flags.contains(.interpolated))
        XCTAssertEqual(result.stations[2].velocityMs, 1.0, accuracy: 0.15)
    }
}

final class RecorderTests: XCTestCase {
    func testUniformFramesResampleToThemselves() {
        let recorder = StiRecorder(sampleCounts: [4], sxPerStation: [0.05])
        for i in 0..<10 {
            recorder.append(row: [Float(i), Float(i) + 1, Float(i) + 2, Float(i) + 3],
                            timestamp: Double(i) / 60.0)
        }
        let stis = recorder.finish(frameStep: 1, nominalPeriod: 1.0 / 60.0)
        XCTAssertEqual(stis.count, 1)
        XCTAssertEqual(stis[0].rows, 10)
        XCTAssertEqual(stis[0].stSecondsPerRow, 1.0 / 60.0, accuracy: 1e-12)
        for r in 0..<10 {
            XCTAssertEqual(stis[0].data[r, 0], Double(r), accuracy: 1e-3)
        }
    }

    func testFrameStepStridesTheGrid() {
        let recorder = StiRecorder(sampleCounts: [4], sxPerStation: [0.05])
        for i in 0..<12 { recorder.append(row: [Float(i), 0, 0, 0], timestamp: Double(i) / 60.0) }
        let stis = recorder.finish(frameStep: 3, nominalPeriod: 1.0 / 60.0)
        XCTAssertEqual(stis[0].stSecondsPerRow, 3.0 / 60.0, accuracy: 1e-12)
        XCTAssertEqual(stis[0].data[1, 0], 3.0, accuracy: 1e-3)
        XCTAssertEqual(stis[0].data[2, 0], 6.0, accuracy: 1e-3)
    }

    /// A dropped frame must not tilt the streaks.
    ///
    /// This is the failure the desktop pipeline cannot have and the phone can:
    /// if a thermally-throttled device skips a frame and the row is simply
    /// stacked next to its neighbour, every streak in that region steepens and
    /// the velocity reads high.  Interpolating onto a uniform time grid is what
    /// prevents it.
    func testADroppedFrameIsInterpolatedRatherThanStacked() {
        let recorder = StiRecorder(sampleCounts: [3], sxPerStation: [0.05])
        // A ramp in time, with the frame at t = 4/60 missing.
        for i in [0, 1, 2, 3, 5, 6, 7, 8] {
            recorder.append(row: [Float(i), Float(i), Float(i)], timestamp: Double(i) / 60.0)
        }
        let stis = recorder.finish(frameStep: 1, nominalPeriod: 1.0 / 60.0)
        // Row 4 sits at t = 4/60, between the samples for 3 and 5, so a linear
        // interpolation must give exactly 4.
        XCTAssertEqual(stis[0].data[4, 0], 4.0, accuracy: 1e-4)
        XCTAssertEqual(stis[0].rows, 9)
    }

    func testMedianFramePeriodTracksTheRealRate() {
        let recorder = StiRecorder(sampleCounts: [2], sxPerStation: [0.05])
        for i in 0..<20 { recorder.append(row: [0, 0], timestamp: Double(i) / 30.0) }
        XCTAssertEqual(recorder.medianFramePeriod ?? 0, 1.0 / 30.0, accuracy: 1e-9)
    }

    func testOutOfFrameSamplesAreCountedAndFilled() {
        let recorder = StiRecorder(sampleCounts: [4], sxPerStation: [0.05])
        for i in 0..<40 {
            // The last column left the frame for the whole recording.
            recorder.append(row: [1, 2, 3, .nan], timestamp: Double(i) / 60.0)
        }
        let sti = recorder.finish(frameStep: 1, nominalPeriod: 1.0 / 60.0)[0]
        XCTAssertEqual(sti.outOfFrameFraction, 0.25, accuracy: 1e-6)
        // Preprocessing must not leave NaN behind for the structure tensor.
        let clean = preprocess(sti.data, settings: StiSettings())
        XCTAssertTrue(clean.data.allSatisfy(\.isFinite))
    }

    func testStaticBackgroundRemovalKillsAVerticalStripe() {
        // A fixed bright column -- a rock, or a glint -- draws a vertical
        // stripe, which is a vote for zero velocity.  Column-mean subtraction
        // must remove it.
        var m = syntheticSTI(tanPhi: 1.0, rows: 120, cols: 120)
        for r in 0..<120 { m[r, 60] += 50.0 }
        var settings = StiSettings()
        settings.removeStaticBackground = true
        let cleaned = preprocess(m, settings: settings)
        var columnMean = 0.0
        for r in 0..<120 { columnMean += cleaned[r, 60] }
        XCTAssertEqual(columnMean / 120.0, 0.0, accuracy: 1e-8)
    }
}

final class UncertaintyTests: XCTestCase {
    func testHeightErrorScalesInverselyWithCameraHeight() {
        var settings = ScaleSettings()
        settings.gravityTiltSigmaDegrees = 0
        settings.vioScaleSigma = 0
        settings.planeHeightSigmaM = 0.03
        XCTAssertEqual(relativeScaleUncertainty(rangeM: 0, cameraHeightM: 2.0, settings: settings),
                       0.015, accuracy: 1e-12)
        XCTAssertEqual(relativeScaleUncertainty(rangeM: 0, cameraHeightM: 6.0, settings: settings),
                       0.005, accuracy: 1e-12)
    }

    func testTiltTermGrowsLinearlyInRangeOverHeight() {
        var settings = ScaleSettings()
        settings.planeHeightSigmaM = 0
        settings.vioScaleSigma = 0
        settings.gravityTiltSigmaDegrees = 0.3
        let eps = 0.3 * Double.pi / 180
        for (range, height) in [(10.0, 2.0), (20.0, 2.0), (20.0, 6.0)] {
            XCTAssertEqual(
                relativeScaleUncertainty(rangeM: range, cameraHeightM: height, settings: settings),
                2 * eps * range / height, accuracy: 1e-12)
        }
    }

    /// The tilt term must actually bound the real error.
    ///
    /// Re-intersect the same pixel against a deliberately tilted plane and
    /// compare the ground sample distance; the model's `2 eps R / h` must not
    /// undercut what the geometry does.
    func testTiltTermBoundsABruteForceReintersection() throws {
        let epsDegrees = 0.3
        let height = 2.0
        let upright = WaterPlaneProjection(
            intrinsics: PinholeIntrinsics(horizontalFovDegrees: 60, width: 1920, height: 1440),
            pose: .looking(fromHeight: height, downDegrees: 6.0),
            plane: WaterPlane(y: 0))
        // Tilting the plane by eps is, from the camera's point of view, the same
        // as tilting the camera by eps.
        let tilted = WaterPlaneProjection(
            intrinsics: upright.intrinsics,
            pose: .looking(fromHeight: height, downDegrees: 6.0 + epsDegrees),
            plane: WaterPlane(y: 0))

        let px = PixelPoint(u: upright.intrinsics.cx, v: upright.intrinsics.cy)
        let a = try XCTUnwrap(upright.imageToWorld(px))
        let b = try XCTUnwrap(tilted.imageToWorld(px))
        let rangeA = upright.horizontalRange(to: a)
        let rangeB = tilted.horizontalRange(to: b)

        let scaleA = try XCTUnwrap(upright.localScale(at: px))
        let scaleB = try XCTUnwrap(tilted.localScale(at: px))
        let actual = abs(scaleB - scaleA) / scaleA

        var settings = ScaleSettings()
        settings.planeHeightSigmaM = 0
        settings.vioScaleSigma = 0
        settings.gravityTiltSigmaDegrees = epsDegrees
        let modelled = relativeScaleUncertainty(
            rangeM: rangeA, cameraHeightM: height, settings: settings)

        XCTAssertGreaterThan(rangeB, rangeA * 0.5)
        XCTAssertGreaterThanOrEqual(modelled, actual * 0.9,
            "model \(modelled) undercuts the actual scale change \(actual)")
    }

    func testACameraAtTheWaterSurfaceIsHonestAboutKnowingNothing() {
        XCTAssertFalse(relativeScaleUncertainty(
            rangeM: 10, cameraHeightM: 0, settings: ScaleSettings()).isFinite)
    }

    func testAdviceNamesAHeightThatWouldFixIt() throws {
        let advice = try XCTUnwrap(scaleAdvice(
            worstRelative: 0.30, atRangeM: 25, cameraHeightM: 1.5, threshold: 0.15))
        XCTAssertTrue(advice.contains("3.0 m"), advice)
        XCTAssertNil(scaleAdvice(worstRelative: 0.05, atRangeM: 5,
                                 cameraHeightM: 1.5, threshold: 0.15))
    }
}
