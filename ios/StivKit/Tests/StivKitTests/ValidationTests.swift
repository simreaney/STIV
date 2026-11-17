//  ValidationTests.swift
//  Bad input is refused with a message, never a crash: `Survey.validated()`,
//  `analyse(validating:...)`, the recorder, and decoding a survey with keys
//  missing.

import XCTest
@testable import StivKit

private func layout(stations: Int = 3) throws -> SectionLayout {
    var s = Survey()
    s.section.stationCount = stations
    s.section.start = PlanePoint(x: -3, z: -8)
    s.section.end = PlanePoint(x: 3, z: -8)
    s.sti.sampleCount = 64
    s.flow.downstream = PlanePoint(x: 0, z: -20)
    return try buildSection(survey: s, rectifier: WaterPlaneProjection(
        intrinsics: PinholeIntrinsics(horizontalFovDegrees: 60, width: 1920, height: 1440),
        pose: .looking(fromHeight: 3, downDegrees: 25),
        plane: WaterPlane(y: 0)))
}

private func stis(for layout: SectionLayout, rows: Int = 120, st: Double = 1.0 / 60) -> [SpaceTimeImage] {
    layout.searchLines.map { line in
        makeSTI(syntheticSTI(tanPhi: 0.7, rows: rows, cols: line.sampleCount, seed: UInt64(line.index)),
                index: line.index, sx: line.sxMetresPerColumn, st: st)
    }
}

final class SurveyValidationTests: XCTestCase {
    private func assertRejected(
        _ change: (inout Survey) -> Void, mentioning key: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        var s = Survey()
        change(&s)
        XCTAssertThrowsError(try s.validated(), file: file, line: line) { error in
            XCTAssertTrue("\(error)".contains(key), "\(error)", file: file, line: line)
        }
    }

    func testTheDefaultsAreValid() {
        XCTAssertNoThrow(try Survey().validated())
    }

    /// `Config.validate`, rule for rule.
    func testMirrorsConfigValidate() {
        assertRejected({ $0.capture.frameStep = 0 }, mentioning: "frameStep")
        assertRejected({ $0.capture.framesPerSecond = 0 }, mentioning: "framesPerSecond")
        assertRejected({ $0.capture.framesPerSecond = .nan }, mentioning: "framesPerSecond")
        assertRejected({ $0.section.stationCount = 1 }, mentioning: "stationCount")
        assertRejected({ $0.section.edgeTrim = 0.5 }, mentioning: "edgeTrim")
        assertRejected({ $0.section.edgeTrim = .nan }, mentioning: "edgeTrim")
        assertRejected({ $0.sti.searchLineLengthM = 0 }, mentioning: "searchLineLengthM")
        assertRejected({ $0.sti.searchLineLengthM = .infinity }, mentioning: "searchLineLengthM")
        assertRejected({ $0.sti.sampleCount = 15 }, mentioning: "sampleCount")
        assertRejected({ $0.sti.windowPx = 4 }, mentioning: "windowPx")
        assertRejected({ $0.sti.windowPx = 30 }, mentioning: "windowPx")
        assertRejected({ $0.sti.windowOverlap = 1 }, mentioning: "windowOverlap")
        assertRejected({ $0.sti.windowOverlap = .nan }, mentioning: "windowOverlap")
        assertRejected({ $0.sti.velocityRangeMs = 1 ... 1 }, mentioning: "velocityRangeMs")
        assertRejected({ $0.sti.velocityRangeMs = -1 ... .infinity }, mentioning: "velocityRangeMs")
        assertRejected({ $0.sti.coherencyLevels = [] }, mentioning: "coherencyLevels")
        assertRejected({ $0.sti.coherencyLevels = [0, 0.5, 1.2] }, mentioning: "coherencyLevels")
        assertRejected({ $0.sti.coherencyLevels = [-0.1] }, mentioning: "coherencyLevels")
        assertRejected({ $0.sti.smoothSigma = -0.5 }, mentioning: "smoothSigma")
        assertRejected({ $0.sti.smoothSigma = .infinity }, mentioning: "smoothSigma")
        assertRejected({ $0.sti.minCoherency = 1.5 }, mentioning: "minCoherency")
        assertRejected({ $0.sti.minCoherency = -0.1 }, mentioning: "minCoherency")
        assertRejected({ $0.sti.angleBins = 0 }, mentioning: "angleBins")
    }

    /// `resolveFlowDirection` silently used 0 degrees for an `angle` survey
    /// with no angle.
    func testFlowDirectionsNeedTheirParameters() {
        assertRejected({ $0.flow.direction = .angle }, mentioning: "angleDegrees")
        assertRejected({ $0.flow.direction = .angle; $0.flow.angleDegrees = .nan },
                       mentioning: "angleDegrees")
        assertRejected({ $0.flow.direction = .points; $0.flow.from = PlanePoint(x: 0, z: 0) },
                       mentioning: "flow.from and flow.to")
        var s = Survey()
        s.flow.direction = .angle
        s.flow.angleDegrees = 0
        XCTAssertNoThrow(try s.validated())
    }

    func testTheSettingsScreenRangesAreAllValid() {
        // What the app's steppers and sliders can produce.
        for windowPx in stride(from: 5, through: 81, by: 2) {
            for stations in [3, 80] {
                var s = Survey()
                s.sti.windowPx = windowPx
                s.section.stationCount = stations
                s.sti.smoothSigma = 4
                XCTAssertNoThrow(try s.validated(), "\(windowPx) px, \(stations) stations")
            }
        }
    }
}

final class AnalyseValidationTests: XCTestCase {
    func testTheThrowingEntryPointRefusesAnInvalidSurvey() throws {
        let l = try layout()
        var s = Survey()
        s.sti.angleBins = 0
        XCTAssertThrowsError(try analyse(validating: s, layout: l, stis: stis(for: l))) { error in
            XCTAssertTrue("\(error)".contains("angleBins"), "\(error)")
        }
    }

    func testTheThrowingEntryPointRefusesStisFromAnotherLayout() throws {
        let l = try layout()
        let s = Survey()
        XCTAssertThrowsError(try analyse(validating: s, layout: l, stis: Array(stis(for: l).dropLast())))
        var wrongScale = stis(for: l)
        wrongScale[1] = makeSTI(wrongScale[1].data, index: 1, sx: 0.01, st: 0)
        XCTAssertThrowsError(try analyse(validating: s, layout: l, stis: wrongScale))
        let other = try layout(stations: 3)
        var wrongWidth = stis(for: other)
        wrongWidth[0] = makeSTI(syntheticSTI(tanPhi: 0.5, rows: 120, cols: 50), sx: 0.05, st: 1.0 / 60)
        XCTAssertThrowsError(try analyse(validating: s, layout: l, stis: wrongWidth))
        XCTAssertNoThrow(try analyse(validating: s, layout: l, stis: stis(for: l)))
    }

    /// Every one of these used to trap somewhere in the pipeline.  The
    /// non-throwing entry point must survive them all and say why its numbers
    /// are worthless.
    func testTheNonThrowingEntryPointNeverTraps() throws {
        let l = try layout()
        let changes: [(String, (inout Survey) -> Void)] = [
            ("angleBins 0", { $0.sti.angleBins = 0 }),
            ("angleBins -1", { $0.sti.angleBins = -1 }),
            ("windowPx 0", { $0.sti.windowPx = 0 }),
            ("windowPx -9 gaussian", { $0.sti.windowPx = -9; $0.sti.windowType = .gaussian }),
            ("overlap NaN", { $0.sti.windowOverlap = .nan }),
            ("overlap -inf", { $0.sti.windowOverlap = -.infinity }),
            ("smoothSigma inf", { $0.sti.smoothSigma = .infinity }),
            ("velocity range inf", { $0.sti.velocityRangeMs = -.infinity ... .infinity }),
            ("frameStep 0", { $0.capture.frameStep = 0 }),
        ]
        for (label, change) in changes {
            var s = Survey()
            change(&s)
            let r = analyse(survey: s, layout: l, stis: stis(for: l))
            XCTAssertEqual(r.stations.count, 3, label)
            XCTAssertTrue(r.warnings.contains { $0.contains("inputs are invalid") }, "\(label): \(r.warnings)")
        }
    }

    /// `stiv.analysis.run` refuses a clip shorter than one window before
    /// reading it; every station would otherwise fail and be reported as a
    /// texture or velocity-range problem.
    func testARecordingTooShortForOneWindowIsRefused() throws {
        let l = try layout()
        let short = stis(for: l, rows: 35)                 // a 31 px box needs 36
        XCTAssertThrowsError(try analyse(validating: Survey(), layout: l, stis: short)) { error in
            XCTAssertTrue("\(error)".contains("needs at least 36"), "\(error)")
        }
        XCTAssertNoThrow(try analyse(validating: Survey(), layout: l, stis: stis(for: l, rows: 36)))

        // The non-throwing entry point carries on, and says why every
        // station failed -- not as a texture problem.
        let r = analyse(survey: Survey(), layout: l, stis: short)
        XCTAssertTrue(r.stations.allSatisfy { $0.flags.contains(.stiTooSmall) })
        XCTAssertTrue(r.warnings.contains { $0.contains("too small for the 31 px box window") }, "\(r.warnings)")
        XCTAssertFalse(r.warnings.contains { $0.contains("failed quality control") }, "\(r.warnings)")
    }

    func testSearchLinesAllTooNarrowForOneWindowAreRefused() throws {
        var s = Survey()
        s.sti.windowPx = 41                                // needs 46 columns; the lines have 64
        let l = try layout()
        XCTAssertNoThrow(try analyse(validating: s, layout: l, stis: stis(for: l)))
        s.sti.windowPx = 61                                // needs 66
        XCTAssertThrowsError(try analyse(validating: s, layout: l, stis: stis(for: l, rows: 200))) { error in
            XCTAssertTrue("\(error)".contains("samples"), "\(error)")
        }
    }

    /// When only some lines are too narrow the run goes ahead, with a warning
    /// naming them; those stations fail and are filled from their neighbours.
    func testSomeNarrowSearchLinesAreWarnedAbout() throws {
        let lines = [64, 20, 64].enumerated().map { i, n in
            makeLine(index: i, stationM: Double(i), sx: 0.04, samples: n)
        }
        let l = SectionLayout(
            start: PlanePoint(x: 0, z: 0), end: PlanePoint(x: 2, z: 0), widthM: 2,
            flowDirection: PlaneVector(x: 0, z: 1), searchLines: lines, flowBearingDegrees: 90,
            cameraHeightM: 3, stationsOutsideFrame: [], rectifierDescription: "test")
        let stis = lines.map { line in
            makeSTI(syntheticSTI(tanPhi: 0.6, rows: 150, cols: line.sampleCount, seed: UInt64(line.index)),
                    index: line.index, sx: line.sxMetresPerColumn, st: 1.0 / 60)
        }
        let r = try analyse(validating: Survey(), layout: l, stis: stis)
        XCTAssertTrue(r.warnings.contains { $0.contains("1 search line(s) have fewer than the 36 samples") },
                      "\(r.warnings)")
        XCTAssertTrue(r.stations[1].flags.contains(.stiTooSmall))
        XCTAssertTrue(r.stations[1].flags.contains(.interpolated))
        XCTAssertTrue(r.summary().contains("1 failed (1 interpolated)"), r.summary())
    }

    func testValidInputsProduceNoValidationWarning() throws {
        let l = try layout()
        let r = analyse(survey: Survey(), layout: l, stis: stis(for: l))
        XCTAssertFalse(r.warnings.contains { $0.contains("inputs are invalid") }, "\(r.warnings)")
    }
}

final class SurveyDecodingTests: XCTestCase {
    /// Synthesized decoding ignores property defaults, so any missing key used
    /// to fail the whole survey.
    func testAnEmptyObjectDecodesToTheDefaults() throws {
        let decoded = try JSONDecoder().decode(Survey.self, from: Data("{}".utf8))
        XCTAssertEqual(decoded, Survey())
    }

    /// Missing keys take their defaults, and a key this version no longer
    /// has -- `rollingShutterRowSeconds`, from older exports -- is ignored.
    func testMissingKeysTakeTheirDefaults() throws {
        let json = """
            {"capture": {"frameStep": 3, "rollingShutterRowSeconds": 1e-5},
             "section": {"stationCount": 12, "start": {"x": 1, "z": 2}},
             "flow": {"direction": "angle", "angleDegrees": 30},
             "scale": {"vioScaleSigma": 0.02},
             "sti": {}, "quality": {}}
            """
        let decoded = try JSONDecoder().decode(Survey.self, from: Data(json.utf8))
        var expected = Survey()
        expected.capture.frameStep = 3
        expected.section.stationCount = 12
        expected.section.start = PlanePoint(x: 1, z: 2)
        expected.flow.direction = .angle
        expected.flow.angleDegrees = 30
        expected.scale.vioScaleSigma = 0.02
        XCTAssertEqual(decoded, expected)
    }

    func testAFullSurveyStillRoundTrips() throws {
        var s = Survey()
        s.capture.durationSeconds = 12
        s.section.name = "weir"
        s.section.edgeTrim = 0.1
        s.flow.direction = .points
        s.flow.from = PlanePoint(x: 0, z: 0)
        s.flow.to = PlanePoint(x: 1, z: 1)
        s.flow.downstream = PlanePoint(x: 3, z: 4)
        s.flow.reverse = true
        s.flow.autoFlip = false
        s.scale.degradedVioScaleSigma = 0.07
        let round = try JSONDecoder().decode(Survey.self, from: JSONEncoder().encode(s))
        XCTAssertEqual(round, s)
    }

    func testAMistypedValueIsStillAnError() {
        XCTAssertThrowsError(try JSONDecoder().decode(
            Survey.self, from: Data(#"{"capture": {"frameStep": "two"}}"#.utf8)))
    }
}

final class RecorderRobustnessTests: XCTestCase {
    private func recorder(timestamps: [Double], samples: Int = 4) -> StiRecorder {
        let r = StiRecorder(sampleCounts: [samples, samples], sxPerStation: [0.01, 0.02])
        for (i, t) in timestamps.enumerated() {
            r.append(row: [Float](repeating: Float(i), count: 2 * samples), timestamp: t)
        }
        return r
    }

    /// Five identical timestamps made the median period zero, and
    /// `Int(inf)` took the app down.
    func testIdenticalTimestampsDoNotCrash() throws {
        let r = recorder(timestamps: Array(repeating: 12.5, count: 5))
        XCTAssertNil(r.medianFramePeriod)
        let images = r.finish(frameStep: 2)
        XCTAssertEqual(images.count, 2)
        XCTAssertTrue(images.allSatisfy { $0.rows == 1 && $0.stSecondsPerRow > 0 })

        // And the analysis reports it rather than tripping over it.
        let l = try layout(stations: 2)
        let padded = images.enumerated().map { i, image in
            makeSTI(Matrix(rows: 1, cols: l.searchLines[i].sampleCount, repeating: .nan),
                    index: i, sx: image.sxMetresPerColumn, st: image.stSecondsPerRow)
        }
        var s = Survey()
        s.section.stationCount = 2
        let result = analyse(survey: s, layout: l, stis: padded)
        XCTAssertTrue(result.stations.allSatisfy { $0.flags.contains(.stiTooSmall) })
    }

    /// Half the deltas repeated: the median of *positive* deltas is the real
    /// capture period.
    func testRepeatedTimestampsDoNotZeroThePeriod() throws {
        var t: [Double] = []
        for i in 0..<20 { t.append(Double(i) / 60); t.append(Double(i) / 60) }
        let r = recorder(timestamps: t)
        XCTAssertEqual(try XCTUnwrap(r.medianFramePeriod), 1.0 / 60, accuracy: 1e-12)
        let images = r.finish(frameStep: 1)
        XCTAssertEqual(images[0].rows, 20)
        XCTAssertEqual(images[0].stSecondsPerRow, 1.0 / 60, accuracy: 1e-12)
    }

    func testAnUnusableNominalPeriodFallsBackToTheMeasuredOne() {
        let r = recorder(timestamps: (0..<30).map { Double($0) / 30 })
        for bad in [0.0, -1.0, .nan, .infinity, 1e-300] {
            let images = r.finish(frameStep: 1, nominalPeriod: bad)
            XCTAssertEqual(images[0].stSecondsPerRow, 1.0 / 30, accuracy: 1e-9, "nominal \(bad)")
            XCTAssertEqual(images[0].rows, 30, "nominal \(bad)")
        }
    }

    func testNonFiniteTimestampsAndAZeroStepDoNotCrash() {
        XCTAssertEqual(recorder(timestamps: [0, 1.0 / 60, .nan]).finish(frameStep: 1)[0].rows, 1)
        XCTAssertEqual(recorder(timestamps: [.infinity, 0, 1]).finish(frameStep: 1)[0].rows, 1)
        XCTAssertEqual(recorder(timestamps: [3, 2, 1]).finish(frameStep: 1)[0].rows, 1)
        let r = recorder(timestamps: (0..<10).map { Double($0) / 60 })
        XCTAssertEqual(r.finish(frameStep: 0)[0].rows, 10, "a zero step is clamped to 1")
    }

    /// Resampling straight into per-station buffers changed no value.
    func testPerStationResamplingMatchesTheInterleavedRow() {
        let r = StiRecorder(sampleCounts: [3, 5], sxPerStation: [0.01, 0.02])
        let times: [Double] = [0, 0.016, 0.035, 0.05, 0.068, 0.083, 0.101]
        for (i, t) in times.enumerated() {
            r.append(row: (0..<8).map { Float(i * 10 + $0) + 0.25 }, timestamp: t)
        }
        let images = r.finish(frameStep: 1, nominalPeriod: 1.0 / 60)
        let period = 1.0 / 60
        for (s, (offset, n)) in [(0, 3), (3, 5)].enumerated() {
            for row in 0..<images[s].rows {
                let t = Double(row) * period
                var lower = 0
                while lower + 2 < times.count && times[lower + 1] < t { lower += 1 }
                let w = Float((t - times[lower]) / (times[lower + 1] - times[lower]))
                for c in 0..<n {
                    let a = Float(lower * 10 + offset + c) + 0.25
                    let b = Float((lower + 1) * 10 + offset + c) + 0.25
                    let want = w == 0 ? a : a + w * (b - a)
                    XCTAssertEqual(images[s].data[row, c], Double(want), "station \(s) (\(row), \(c))")
                }
            }
        }
    }

    func testASnapshotIsAnIndependentSendableValue() {
        let r = recorder(timestamps: (0..<10).map { Double($0) / 60 })
        let snapshot = r.snapshot
        r.append(row: [Float](repeating: 99, count: 8), timestamp: 10.0 / 60)
        XCTAssertEqual(snapshot.rowCount, 10)
        XCTAssertEqual(r.rowCount, 11)
        let before = snapshot.finish(frameStep: 1)[1].data
        let after = r.finish(frameStep: 1)[1].data
        XCTAssertEqual([before.rows, after.rows], [10, 11])
        XCTAssertEqual(before.data, Array(after.data.prefix(before.count)))
        let handOff: @Sendable () -> Int = { snapshot.finish(frameStep: 2)[0].rows }
        XCTAssertEqual(handOff(), 5)
    }
}

final class DisplayTests: XCTestCase {
    /// `np.percentile` interpolates; the port took the floor index.
    func testPercentilesInterpolateAsNumpyDoes() {
        let sorted = [0.0, 1, 2, 7, 10, 11, 100]
        XCTAssertEqual(numpyPercentile(sorted: sorted, 1.0), 0.06, accuracy: 1e-15)
        XCTAssertEqual(numpyPercentile(sorted: sorted, 99.0), 94.65999999999995)
        XCTAssertEqual(numpyPercentile(sorted: sorted, 100.0), 100)
        XCTAssertEqual(numpyPercentile(sorted: [5], 1.0), 5)
    }

    /// `stiv.sti.to_display` on the same values.
    func testDisplayBytesMatchToDisplay() {
        let m = Matrix(rows: 1, cols: 7, data: [0, 1, 2, 10, 11, 100, 7])
        XCTAssertEqual(displayBytes(m), [0, 2, 5, 26, 29, 255, 18])
    }

    func testAFlatImageFallsBackToTheFullRange() {
        var m = Matrix(rows: 1, cols: 200, repeating: 3)
        m[0, 199] = 4
        let bytes = displayBytes(m)
        XCTAssertEqual(bytes[0], 0)
        XCTAssertEqual(bytes[199], 255)
        XCTAssertEqual(displayBytes(Matrix(rows: 2, cols: 2, repeating: 1)), [0, 0, 0, 0])
    }
}

/// Compile-time: these must be `Sendable` to cross to the analysis queue.
final class SendableTests: XCTestCase {
    private func requireSendable<T: Sendable>(_: T.Type) {}

    func testTheAnalysisInputsAndOutputsAreSendable() {
        requireSendable(SectionLayout.self)
        requireSendable(SearchLine.self)
        requireSendable(OrientationField.self)
        requireSendable(AngleEstimate.self)
        requireSendable(StiRecording.self)
        requireSendable(StiPreview.self)
        requireSendable(StivResult.self)
        requireSendable(WaterPlaneProjection.self)
        let rectifier = WaterPlaneProjection(
            intrinsics: PinholeIntrinsics(horizontalFovDegrees: 60, width: 10, height: 10),
            pose: .looking(fromHeight: 1, downDegrees: 30), plane: WaterPlane(y: 0))
        let send: @Sendable () -> Int = { rectifier.imageWidth }
        XCTAssertEqual(send(), 10)
    }
}
