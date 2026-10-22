//  AnalysisStageTests.swift
//  Quality control, the profile median filter, interpolation, the direction
//  check and the frame-step advice, each against what `stiv/analysis.py` does
//  with the same numbers.

import XCTest
@testable import StivKit

final class MedianTests: XCTestCase {
    func testNumpyMedianAveragesTheMiddlePairForAnEvenCount() {
        XCTAssertEqual(numpyMedian([0.1, 0.18, 0.21, 0.3]), 0.195, accuracy: 1e-15)
        XCTAssertEqual(numpyMedian([3, 1, 2]), 2)
        XCTAssertEqual(numpyMedian([4, 1, 3, 2]), 2.5)
        XCTAssertTrue(numpyMedian([]).isNaN)
        XCTAssertTrue(numpyMedian([1, .nan, 3]).isNaN, "np.median propagates NaN")
    }
}

final class MedianFilterTests: XCTestCase {
    private func filtered(_ velocities: [Double], window: Int) -> [StationResult] {
        var stations = velocities.enumerated().map { makeStation(index: $0, velocity: $1) }
        medianFilter(&stations, window: window)
        return stations
    }

    private func same(_ a: [Double], _ b: [Double]) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { $0 == $1 || ($0.isNaN && $1.isNaN) }
    }

    /// A station moves only when its window holds at least three usable
    /// values: the median of two is their mean, which one spike drags as far
    /// as it likes.  So at k = 3 the two end stations stay as measured, and at
    /// k = 5 they move to the median of three.  `_median_filter` on the same
    /// numbers.
    func testAWindowNeedsThreeValuesToMoveItsStation() {
        let three = filtered([1, 2, 3, 4, 5, 10], window: 3)
        XCTAssertEqual(three.map(\.velocityMs), [1, 2, 3, 4, 5, 10])
        XCTAssertFalse(three.contains { $0.flags.contains(.medianFiltered) })

        let five = filtered([1, 2, 3, 4, 5, 10], window: 5)
        XCTAssertEqual(five.map(\.velocityMs), [2, 2.5, 3, 4, 4.5, 5])
        XCTAssertEqual(five.map { $0.flags.contains(.medianFiltered) },
                       [true, true, false, false, true, true])
    }

    /// Only usable stations take part, as inputs or as outputs: a NaN station
    /// is neither filled by the filter nor counted in its neighbours' windows,
    /// and a window that loses values to it may fall below three.
    func testFailedStationsNeitherFeedNorReceiveTheFilter() {
        let three = filtered([1, .nan, 3, 4, .nan, 10, 2], window: 3)
        XCTAssertTrue(same(three.map(\.velocityMs), [1, .nan, 3, 4, .nan, 10, 2]))
        XCTAssertFalse(three.contains { $0.flags.contains(.medianFiltered) })

        let five = filtered([1, .nan, 3, 4, .nan, 10, 2], window: 5)
        XCTAssertTrue(same(five.map(\.velocityMs), [1, .nan, 3, 4, .nan, 4, 2]),
                      "\(five.map(\.velocityMs))")
        XCTAssertEqual(five.map { $0.flags.contains(.medianFiltered) },
                       [false, false, false, false, false, true, false])

        // A flagged station with a perfectly finite value is excluded too:
        // its raw number is exactly what must not spread.
        var stations = [
            makeStation(index: 0, velocity: 1),
            makeStation(index: 1, velocity: 100, flags: [.lowCoherency]),
            makeStation(index: 2, velocity: 3),
            makeStation(index: 3, velocity: 4),
            makeStation(index: 4, velocity: 5),
        ]
        medianFilter(&stations, window: 3)
        XCTAssertEqual(stations.map(\.velocityMs), [1, 100, 3, 4, 5])
    }

    /// A moved station reports a number it did not measure: its angle follows
    /// the new velocity through Eq. (16), and its spread is cleared.
    func testAMovedStationRecomputesItsAngleAndDropsItsSpread() {
        let out = filtered([1, 2, 3, 4, 5, 10], window: 5)
        let moved = out[5]
        XCTAssertEqual(moved.velocityMs, 5)
        XCTAssertEqual(moved.velocityRawMs, 10, "the measurement is kept")
        XCTAssertEqual(moved.tanPhi, 5 * (1.0 / 30) / 0.01, accuracy: 1e-12)
        XCTAssertEqual(moved.phiRad, atan(moved.tanPhi), accuracy: 1e-15)
        XCTAssertTrue(moved.velocityStdMs.isNaN && moved.velocityStdAngleMs.isNaN && moved.phiStdRad.isNaN)
        XCTAssertFalse(out[2].velocityStdAngleMs.isNaN, "an untouched station keeps its spread")
    }

    /// An even k centres a k + 1 window, as the Python does (and warns about).
    func testAnEvenWindowMatchesThePython() {
        XCTAssertEqual(filtered([1, 2, 3, 4, 5, 10], window: 4).map(\.velocityMs),
                       [2, 2.5, 3, 4, 4.5, 5])
    }
}

final class QualityControlTests: XCTestCase {
    /// The uncertainty test's velocity scale is a quarter of np.median, not of
    /// the upper median: [0.2, 2, 3, 4] gives 0.625, not 0.75.  A slow station
    /// whose spread falls between the two thresholds tells them apart.
    func testUncertaintyScaleUsesTheNumpyMedian() {
        var stations = [
            makeStation(index: 0, velocity: 0.2, angleStd: 0.35),
            makeStation(index: 1, velocity: 2, angleStd: 0.01),
            makeStation(index: 2, velocity: 3, angleStd: 0.01),
            makeStation(index: 3, velocity: 4, angleStd: 0.01),
        ]
        var survey = Survey()
        survey.quality.interpolateFlagged = false
        var warnings: [String] = []
        qualityControl(survey: survey, stations: &stations, warnings: &warnings)
        // 0.35 > 0.5 * max(0.2, 0.25 * 2.5) = 0.3125, but not > 0.375.
        XCTAssertTrue(stations[0].flags.contains(.highUncertainty), "\(stations[0].flags)")
    }

    /// The failure test reads the angle spread alone.  A large total -- the
    /// scale term of a far, low-camera station -- must not fail a station
    /// whose angle histogram is sharp.
    func testOnlyTheAngleSpreadCanFailAStation() {
        var station = makeStation(index: 0, velocity: 1.0, angleStd: 0.05)
        station.velocityStdMs = 0.9            // total, dominated by the scale term
        station.relativeScaleSigma = 0.89
        var stations = [station, makeStation(index: 1, velocity: 1.0)]
        var warnings: [String] = []
        qualityControl(survey: Survey(), stations: &stations, warnings: &warnings)
        XCTAssertFalse(stations[0].flags.contains(.highUncertainty))
        XCTAssertTrue(stations[0].flags.contains(.scaleUncertain))
        XCTAssertFalse(stations[0].failed)
        XCTAssertEqual(stations[0].velocityMs, 1.0)
    }

    /// With the camera 0.05 m or less above the plane the scale sigma is
    /// infinite, which used to make the total infinite and skip the
    /// uncertainty test altogether.  A flat angle histogram must still fail.
    func testAFlatHistogramFailsEvenWhenTheScaleIsUnbounded() {
        var station = StationResult(line: makeLine(index: 0, stationM: 0, sx: 0.02, rangeM: 10))
        let estimate = AngleEstimate(
            phi: 0.5, tanPhi: tan(0.5), phiStd: 0.6, validFraction: 0.9, meanCoherency: 0.7,
            usedCount: 20, histogram: [0], binCentres: [0], phiMin: -1, phiMax: 1)
        apply(estimate, to: &station, sti: makeSTI(Matrix(rows: 1, cols: 1), sx: 0.02, st: 1.0 / 30),
              survey: Survey(), cameraHeightM: 0.04, trackingDegraded: false)
        XCTAssertFalse(station.relativeScaleSigma.isFinite)
        XCTAssertFalse(station.velocityStdMs.isFinite, "the total is honestly unbounded")
        XCTAssertTrue(station.velocityStdAngleMs.isFinite)

        var stations = [station]
        var warnings: [String] = []
        var survey = Survey()
        survey.quality.interpolateFlagged = false
        qualityControl(survey: survey, stations: &stations, warnings: &warnings)
        XCTAssertTrue(stations[0].flags.contains(.highUncertainty), "\(stations[0].flags)")
        XCTAssertTrue(stations[0].flags.contains(.scaleUncertain))
    }

    func testTheUnboundedScaleIsReportedNotSilenced() throws {
        let advice = try XCTUnwrap(scaleAdvice(worstRelative: .infinity, atRangeM: 12,
                                               cameraHeightM: 0.03, threshold: 0.15))
        XCTAssertTrue(advice.contains("water plane"), advice)
        var warnings: [String] = []
        var station = makeStation(index: 0, velocity: 1)
        station.relativeScaleSigma = .infinity
        scaleWarning(survey: Survey(), stations: [station], cameraHeightM: 0.03, warnings: &warnings)
        XCTAssertEqual(warnings.count, 1)
    }
}

final class InterpolationTests: XCTestCase {
    /// `np.interp` clamps beyond the outermost usable stations, and
    /// interpolates linearly in station metres between them.
    func testEndsClampAndTheMiddleInterpolates() {
        var stations = [
            makeStation(index: 0, stationM: 0, velocity: .nan, flags: [.noOrientation]),
            makeStation(index: 1, stationM: 1, velocity: 1.0),
            makeStation(index: 2, stationM: 2, velocity: 5.0, flags: [.lowCoherency]),
            makeStation(index: 3, stationM: 4, velocity: 3.0),
            makeStation(index: 4, stationM: 5, velocity: 9.0, flags: [.stiTooSmall]),
        ]
        interpolateFailed(&stations)
        // np.interp's own arithmetic: slope * (x - x0) + v0.
        XCTAssertEqual(stations.map(\.velocityMs), [1.0, 1.0, (3.0 - 1.0) / (4 - 1) * (2 - 1) + 1.0, 3.0, 3.0])
        XCTAssertEqual(stations.map { $0.flags.contains(.interpolated) },
                       [true, false, true, false, true])

        // A filled station's angle follows its new velocity, and its spread
        // -- which described the replaced measurement -- is cleared.
        for i in [0, 2, 4] {
            let s = stations[i]
            XCTAssertEqual(s.tanPhi, s.velocityMs * (1.0 / 30) / 0.01, accuracy: 1e-12, "station \(i)")
            XCTAssertEqual(s.phiRad, atan(s.tanPhi), accuracy: 1e-15)
            XCTAssertTrue(s.velocityStdAngleMs.isNaN && s.velocityStdMs.isNaN && s.phiStdRad.isNaN)
        }
        XCTAssertEqual(stations[2].velocityRawMs, 5.0, "the measurement is kept")
    }

    func testFewerThanTwoUsableStationsLeavesEverythingAlone() {
        var stations = [
            makeStation(index: 0, velocity: .nan, flags: [.noOrientation]),
            makeStation(index: 1, velocity: 1.0),
        ]
        interpolateFailed(&stations)
        XCTAssertTrue(stations[0].velocityMs.isNaN)
        XCTAssertFalse(stations[0].flags.contains(.interpolated))
    }
}

final class RangeAdviceTests: XCTestCase {
    /// Sx = St = 1, so a station's measured velocity is its tan phi.
    private func advice(step: Int, tans: [Double]) -> (step: Int?, message: String?) {
        var survey = Survey()
        survey.capture.frameStep = step
        let stations = tans.enumerated().map {
            makeStation(index: $0, velocity: $1, tanPhi: $1, sx: 1, st: 1)
        }
        return rangeAdvice(survey: survey, stations: stations)
    }

    /// Already at step 1 with steep streaks there is no smaller step: say what
    /// does help instead of saying nothing.
    func testAtStepOneSteepStreaksGetTheFrameRateAdvice() throws {
        let a = advice(step: 1, tans: [5, 6, 7])
        XCTAssertNil(a.step)
        let message = try XCTUnwrap(a.message)
        XCTAssertTrue(message.contains("frame step already 1"), message)
        XCTAssertTrue(message.contains("frame rate"), message)
        XCTAssertTrue(message.contains("sample count to about"), message)
    }

    /// The advice reads the *measured* angle -- `velocityRawMs St / Sx` -- of
    /// every station, failed or filled or not, as `_range_advice` does.
    func testItJudgesTheMeasuredAngleNotTheReportedOne() {
        var survey = Survey()
        survey.capture.frameStep = 1
        var stations = (0..<3).map { makeStation(index: $0, velocity: 0.1, sx: 1, st: 1) }
        for i in stations.indices {
            stations[i].flags = [.lowCoherency, .interpolated]
            stations[i].velocityMs = 1.0
            stations[i].tanPhi = 1.0                 // the filled value, in band
        }
        let a = rangeAdvice(survey: survey, stations: stations)
        XCTAssertEqual(a.step, 10)                   // round(1 / 0.1)
        XCTAssertNotNil(a.message)
    }

    /// The `> 4` branch: features move too far between frames.
    func testSteepStreaksSuggestASmallerStep() throws {
        let a = advice(step: 4, tans: [5, 6, 7, 6.5])   // median 6.25
        XCTAssertEqual(a.step, 1)
        XCTAssertTrue(try XCTUnwrap(a.message).contains("near the top of the range"))
    }

    /// np.median([0.1, 0.18, 0.21, 0.3]) = 0.195 is below the 0.2 threshold;
    /// the upper median, 0.21, is not, and gave no advice at all.
    func testAnEvenCountUsesTheNumpyMedian() throws {
        let a = advice(step: 1, tans: [0.1, 0.18, 0.21, 0.3])
        XCTAssertEqual(a.step, 5)                        // round(1 / 0.195)
        XCTAssertTrue(try XCTUnwrap(a.message).contains("barely move"))
    }

    /// 25 / 10 = 2.5, which Python rounds to 2.
    func testAHalfwaySuggestionRoundsToEven() {
        XCTAssertEqual(advice(step: 25, tans: [10, 10, 10]).step, 2)
    }

    func testAVanishingMedianDoesNotOverflow() {
        let a = advice(step: 30, tans: [1e-300, 1e-300, 1e-300])
        XCTAssertNotNil(a.step)
        XCTAssertNotNil(a.message)
    }
}

final class QualityControlOrderTests: XCTestCase {
    /// The filter runs on the usable stations first, and the failed ones are
    /// then filled from the *filtered* values.  `_quality_control` on the same
    /// numbers: the failed 50 neither feeds the filter nor survives it.
    func testFailedValuesDoNotLeakThroughTheMedianFilter() {
        var survey = Survey()
        survey.quality.medianFilterStations = 3
        var stations = [
            makeStation(index: 0, velocity: 1.0),
            makeStation(index: 1, velocity: 50.0, coherency: 0.01),   // fails: low coherency
            makeStation(index: 2, velocity: 1.2),
            makeStation(index: 3, velocity: 1.1),
            makeStation(index: 4, velocity: 1.3),
        ]
        var warnings: [String] = []
        qualityControl(survey: survey, stations: &stations, warnings: &warnings)
        XCTAssertEqual(stations.map(\.velocityMs), [1.0, (1.2 - 1.0) / (2 - 0) * (1 - 0) + 1.0, 1.2, 1.2, 1.3])
        XCTAssertEqual(stations.map { $0.flags.contains(.medianFiltered) },
                       [false, false, false, true, false])
        XCTAssertTrue(stations[1].flags.contains(.interpolated))
        XCTAssertEqual(stations[1].velocityRawMs, 50.0)
        XCTAssertEqual(warnings, [])
    }

    /// The reviewer's spike: a lone 3.0 among ones, beside a failed station.
    /// The filter removes the spike first, so the gap is filled from the
    /// cleaned profile and the spike cannot come back through it.
    func testASpikeBesideAFailedStationIsRemovedBeforeTheGapIsFilled() {
        var survey = Survey()
        survey.quality.medianFilterStations = 5
        var stations = [1, 1, 1, 3.0, 9.0, 1, 1, 1].enumerated().map {
            makeStation(index: $0, velocity: $1)
        }
        stations[4].coherency = 0.01                                     // station 4 fails
        var warnings: [String] = []
        qualityControl(survey: survey, stations: &stations, warnings: &warnings)
        XCTAssertEqual(stations[3].velocityMs, 1.0)
        XCTAssertTrue(stations[3].flags.contains(.medianFiltered))
        XCTAssertEqual(stations[4].velocityMs, 1.0)
        XCTAssertTrue(stations[4].flags.contains(.interpolated))
        XCTAssertEqual(stations.map(\.velocityMs), Array(repeating: 1.0, count: 8))
    }

    /// `StationResult.add_flag`: a flag is set once, however often the
    /// condition is met.
    func testFlagsAreAddedOnce() {
        var station = makeStation(index: 0, velocity: 1)
        station.addFlag(.tanPhiLow)
        station.addFlag(.tanPhiLow)
        XCTAssertEqual(station.flags, [.tanPhiLow])

        var stations = [makeStation(index: 0, velocity: 0.001, coherency: 0.01, flags: [.lowCoherency])]
        var warnings: [String] = []
        qualityControl(survey: Survey(), stations: &stations, warnings: &warnings)
        XCTAssertEqual(stations[0].flags.filter { $0 == .lowCoherency }.count, 1, "\(stations[0].flags)")
    }

    /// A failed station that cannot be filled reports no velocity at all --
    /// its raw value stays in `velocityRawMs` -- and the user is told why.
    func testUnfilledFailuresAreBlankedAndExplained() throws {
        var stations = [
            makeStation(index: 0, velocity: 1.0),
            makeStation(index: 1, velocity: 9.0, coherency: 0.01),
        ]
        var warnings: [String] = []
        qualityControl(survey: Survey(), stations: &stations, warnings: &warnings)
        let blank = stations[1]
        XCTAssertTrue(blank.velocityMs.isNaN && blank.tanPhi.isNaN && blank.phiRad.isNaN)
        XCTAssertTrue(blank.velocityStdMs.isNaN && blank.velocityStdAngleMs.isNaN)
        XCTAssertEqual(blank.velocityRawMs, 9.0)
        XCTAssertFalse(blank.flags.contains(.interpolated))
        XCTAssertTrue(warnings.contains { $0.contains("could not be interpolated") }, "\(warnings)")

        // With interpolation off the blanking still happens, but there is
        // nothing to apologise for.
        var survey = Survey()
        survey.quality.interpolateFlagged = false
        stations = [makeStation(index: 0, velocity: 1.0), makeStation(index: 1, velocity: 9.0, coherency: 0.01)]
        warnings = []
        qualityControl(survey: survey, stations: &stations, warnings: &warnings)
        XCTAssertTrue(stations[1].velocityMs.isNaN)
        XCTAssertFalse(warnings.contains { $0.contains("could not be interpolated") }, "\(warnings)")
    }

    /// An STI too small for the window gets its own warning, naming the size
    /// needed, and does not count towards "failed quality control".
    func testTooSmallStisAreExplainedSeparately() throws {
        var stations = (0..<4).map { makeStation(index: $0, velocity: .nan, flags: [.stiTooSmall]) }
        stations.append(makeStation(index: 4, velocity: 1.0))
        var warnings: [String] = []
        qualityControl(survey: Survey(), stations: &stations, warnings: &warnings)
        let tooSmall = try XCTUnwrap(warnings.first { $0.contains("too small") })
        XCTAssertTrue(tooSmall.contains("4 of 5"), tooSmall)
        XCTAssertTrue(tooSmall.contains("at least 36 rows"), tooSmall)
        XCTAssertFalse(warnings.contains { $0.contains("failed quality control") }, "\(warnings)")
    }
}

final class ReverseAdviceTests: XCTestCase {
    /// `_reverse_advice`: toggle `reverse`, and name the setting that chose
    /// the direction.
    func testAdviceNamesTheSettingThatChoseTheDirection() {
        var flow = FlowSettings()
        XCTAssertEqual(reverseAdvice(flow), "set flow.reverse to true, or tap a point downstream of the section")
        flow.reverse = true
        XCTAssertTrue(reverseAdvice(flow).hasPrefix("set flow.reverse to false (it is currently true)"))
        flow.reverse = false
        flow.downstream = PlanePoint(x: 0, z: 1)
        XCTAssertTrue(reverseAdvice(flow).hasSuffix("really is downstream of the section"))
        flow.direction = .points
        XCTAssertTrue(reverseAdvice(flow).hasSuffix("swap flow.from and flow.to"))
        flow.direction = .angle
        XCTAssertTrue(reverseAdvice(flow).hasSuffix("add 180 to flow.angleDegrees"))
    }
}

final class FlatWindowTests: XCTestCase {
    /// A window with no texture has phi = 0 from Eq. (6) and, at coherency
    /// level 0, would vote for zero velocity.  With part of an STI held
    /// constant, the estimate must be exactly what it would be if those
    /// windows had no angle at all -- and they must count against
    /// `validFraction`, whose denominator stays every window.
    func testFeaturelessWindowsDoNotVote() throws {
        // Held at exactly zero, so the flat windows have exactly no gradient.
        // (Any other constant leaves rounding residues of order 1e-18 in the
        // stencil sums: still caught by the relative threshold, but with an
        // angle of noise rather than zero.)
        var sti = syntheticSTI(tanPhi: 1.0, rows: 160, cols: 200, seed: 12)
        for r in 0..<sti.rows { for c in 0..<100 { sti[r, c] = 0 } }
        let field = try structureTensorField(sti: sti)
        let flat = flatWindows(field)
        let nFlat = flat.filter { $0 }.count
        XCTAssertGreaterThan(nFlat, 0)
        XCTAssertLessThan(nFlat, flat.count)
        for i in flat.indices where flat[i] {
            XCTAssertEqual(field.phi.data[i], 0, "Eq. (6) gives a flat window phi = 0")
        }

        var nearlyFlat = sti
        for r in 0..<sti.rows { for c in 0..<100 { nearlyFlat[r, c] = 0.25 } }
        XCTAssertEqual(flatWindows(try structureTensorField(sti: nearlyFlat)), flat,
                       "rounding residues are flat too")

        let estimate = try meanOrientationAngle(field: field, phiMin: -1.4, phiMax: 1.4, minCoherency: 0)
        XCTAssertEqual(estimate.validFraction, Double(estimate.usedCount) / Double(flat.count))

        // The same field with the flat windows' angles removed outright.
        var noAngle = field.phi
        for i in flat.indices where flat[i] { noAngle.data[i] = .nan }
        let removed = try meanOrientationAngle(
            field: OrientationField(phi: noAngle, coherency: field.coherency, energy: field.energy,
                                    centresT: field.centresT, centresX: field.centresX),
            phiMin: -1.4, phiMax: 1.4, minCoherency: 0)
        XCTAssertEqual(estimate.histogram, removed.histogram)
        XCTAssertEqual(estimate.phi, removed.phi)
        XCTAssertEqual(estimate.usedCount, removed.usedCount)

        // Without the energy test they would have voted, for exactly zero.
        let textured = Matrix(rows: field.energy.rows, cols: field.energy.cols, repeating: 1)
        let unfiltered = try meanOrientationAngle(
            field: OrientationField(phi: field.phi, coherency: field.coherency, energy: textured,
                                    centresT: field.centresT, centresX: field.centresX),
            phiMin: -1.4, phiMax: 1.4, minCoherency: 0)
        XCTAssertEqual(unfiltered.usedCount, estimate.usedCount + nFlat)
        XCTAssertLessThan(abs(unfiltered.phi), abs(estimate.phi))
    }

    func testAnStiWithNoGradientAnywhereIsAllFlat() throws {
        let field = try structureTensorField(sti: Matrix(rows: 80, cols: 80, repeating: 3))
        XCTAssertTrue(flatWindows(field).allSatisfy { $0 })
        let estimate = try meanOrientationAngle(field: field, phiMin: -1, phiMax: 1)
        XCTAssertTrue(estimate.phi.isNaN)
        XCTAssertEqual(estimate.validFraction, 0)
    }
}

final class MinimumSizeTests: XCTestCase {
    /// `stiv.orientation.min_sti_size` for the same arguments.
    func testMatchesThePython() {
        XCTAssertEqual(minStiSize(windowPx: 31, windowType: .box, gradientOrder: .four), 36)
        XCTAssertEqual(minStiSize(windowPx: 31, windowType: .gaussian, gradientOrder: .four), 68)
        XCTAssertEqual(minStiSize(windowPx: 5, windowType: .box, gradientOrder: .two), 8)
        XCTAssertEqual(minStiSize(windowPx: 29, windowType: .gaussian, gradientOrder: .six), 66)
        XCTAssertEqual(minStiSize(windowPx: 81, windowType: .box, gradientOrder: .four), 86)
    }

    /// It is exactly the smallest size `structureTensorField` accepts.
    func testIsTheSmallestSizeTheFieldAccepts() {
        for (w, type) in [(31, WindowType.box), (15, .gaussian), (9, .box)] {
            let need = minStiSize(windowPx: w, windowType: type)
            let sti = syntheticSTI(tanPhi: 1, rows: need + 5, cols: need + 5)
            var fits = Matrix(rows: need, cols: need)
            var short = Matrix(rows: need - 1, cols: need)
            for r in 0..<need { for c in 0..<need { fits[r, c] = sti[r, c] } }
            for r in 0..<(need - 1) { for c in 0..<need { short[r, c] = sti[r, c] } }
            XCTAssertNoThrow(try structureTensorField(sti: fits, windowPx: w, windowType: type))
            XCTAssertThrowsError(try structureTensorField(sti: short, windowPx: w, windowType: type))
        }
    }
}
