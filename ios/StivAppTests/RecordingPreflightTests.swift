//  RecordingPreflightTests.swift
//  Before Start: will the recording fit in memory, and is the phone pointing
//  at the section?

import StivKit
import XCTest
@testable import STIV

final class RecordingBudgetTests: XCTestCase {
    /// The defaults: thirty stations of about seven hundred samples, 84 KB a
    /// row, 5 MB a second at 60 fps.
    private let defaults = [Int](repeating: 700, count: 30)

    func testTheDefaultsFitAsTheyAre() {
        let d = RecordingBudget.decide(sampleCounts: defaults, framesPerSecond: 60,
                                       requestedSeconds: 30)
        XCTAssertEqual(d.seconds, 30)
        XCTAssertNil(d.samplesPerLine)
        XCTAssertNil(d.notice)
        XCTAssertEqual(d.bytesPerSecond, 30 * 700 * 4 * 60)
        XCTAssertLessThanOrEqual(d.totalBytes, Double(RecordingBudget.bytes))
    }

    /// The longest recording the slider allows, at the defaults, would hold
    /// 600 MB: it is shortened, and the user is told why.
    func testTooLongARecordingIsShortenedFirst() throws {
        let d = RecordingBudget.decide(sampleCounts: defaults, framesPerSecond: 60,
                                       requestedSeconds: 120)
        XCTAssertLessThan(d.seconds, 120)
        XCTAssertGreaterThanOrEqual(d.seconds, RecordingBudget.minimumSeconds)
        XCTAssertNil(d.samplesPerLine, "duration gives way before resolution")
        XCTAssertLessThanOrEqual(d.totalBytes, Double(RecordingBudget.bytes))
        let notice = try XCTUnwrap(d.notice)
        XCTAssertTrue(notice.contains("\(Int(d.seconds)) s"), notice)
        XCTAssertTrue(notice.contains("memory"), notice)
    }

    /// The largest the settings allow -- eighty stations at StivKit's
    /// 4096-sample cap -- is nearly 80 MB a second: not even the shortest
    /// useful recording fits, so the lines are thinned.
    func testWhenEvenTheShortestRecordingWontFitTheLinesAreThinned() throws {
        let worst = [Int](repeating: 4096, count: 80)
        let d = RecordingBudget.decide(sampleCounts: worst, framesPerSecond: 60,
                                       requestedSeconds: 120)
        let perLine = try XCTUnwrap(d.samplesPerLine)
        XCTAssertGreaterThanOrEqual(perLine, RecordingBudget.minimumSamplesPerLine)
        XCTAssertLessThan(perLine, 4096)
        XCTAssertEqual(d.seconds, RecordingBudget.minimumSeconds)
        XCTAssertLessThanOrEqual(d.totalBytes, Double(RecordingBudget.bytes))
        XCTAssertNotNil(d.notice)
    }

    /// The default settings -- thirty stations, 4 m search lines, thirty
    /// seconds at 60 fps -- on a typical bank-side survey, with the sample
    /// counts StivKit actually chooses at the phone's sensor resolution: they
    /// must fit as they are, with nothing shortened.
    func testTheDefaultsFitATypicalBankSideSurvey() throws {
        let sensor = PinholeIntrinsics(horizontalFovDegrees: 60, width: 1920, height: 1440)
        let views: [(String, CameraPose, PlanePoint, PlanePoint, PlanePoint)] = [
            // Across the river from a bank 2 m up, near edge 5 m out, far 15 m.
            ("across", .looking(fromHeight: 2, downDegrees: 15),
             PlanePoint(x: 0, z: -5), PlanePoint(x: 0, z: -15), PlanePoint(x: 10, z: -10)),
            // Downstream from a footbridge 4 m up.
            ("bridge", .looking(fromHeight: 4, downDegrees: 25),
             PlanePoint(x: -4, z: -8), PlanePoint(x: 4, z: -8), PlanePoint(x: 0, z: -28)),
        ]
        for (name, pose, start, end, downstream) in views {
            var s = Survey()
            s.section.start = start
            s.section.end = end
            s.flow.downstream = downstream
            let layout = try buildSection(
                survey: s.validated(),
                rectifier: WaterPlaneProjection(intrinsics: sensor, pose: pose,
                                                plane: WaterPlane(y: 0)))
            XCTAssertTrue(layout.stationsOutsideFrame.isEmpty, "\(name): not a sensible view")
            let d = RecordingBudget.decide(sampleCounts: layout.searchLines.map(\.sampleCount),
                                           framesPerSecond: 60,
                                           requestedSeconds: s.capture.durationSeconds)
            XCTAssertNil(d.notice, "\(name): \(d.notice ?? "")")
            XCTAssertEqual(d.seconds, s.capture.durationSeconds, name)
        }
    }

    /// Thinning never goes below the floor it is given -- the model passes the
    /// analysis window's own minimum when that is the larger.
    func testThinningRespectsItsFloor() throws {
        let worst = [Int](repeating: 4096, count: 80)
        let d = RecordingBudget.decide(sampleCounts: worst, framesPerSecond: 60,
                                       requestedSeconds: 120, minimumSamplesPerLine: 3000)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(d.samplesPerLine), 3000)
        XCTAssertLessThanOrEqual(d.totalBytes, Double(RecordingBudget.bytes))
        XCTAssertLessThan(d.seconds, RecordingBudget.minimumSeconds,
                          "at a high floor, duration gives way after all")
    }

    /// The decision uses the sample counts the layout actually has, whatever
    /// rule StivKit used to choose them: double the samples, half the time.
    func testTheBudgetFollowsTheLayoutsOwnSampleCounts() {
        let few = RecordingBudget.decide(sampleCounts: [Int](repeating: 2000, count: 30),
                                         framesPerSecond: 60, requestedSeconds: 120)
        let many = RecordingBudget.decide(sampleCounts: [Int](repeating: 4000, count: 30),
                                          framesPerSecond: 60, requestedSeconds: 120)
        XCTAssertEqual(many.bytesPerSecond, 2 * few.bytesPerSecond)
        XCTAssertLessThan(many.seconds, few.seconds)
    }
}

final class RecordingCheckTests: XCTestCase {
    private let intrinsics = PinholeIntrinsics(horizontalFovDegrees: 60, width: 640, height: 480)
    private let plane = WaterPlane(y: 0)

    /// The geometry of `SyntheticRiverTests`: a 6 m section 8 m out from a bank
    /// 3 m above the water.
    private func layout(from pose: CameraPose) throws -> (SectionLayout, WaterPlaneProjection) {
        var s = Survey()
        s.section.stationCount = 7
        s.section.start = PlanePoint(x: -3, z: -8)
        s.section.end = PlanePoint(x: 3, z: -8)
        s.flow.downstream = PlanePoint(x: 0, z: -20)
        s.sti.searchLineLengthM = 4
        let projection = WaterPlaneProjection(intrinsics: intrinsics, pose: pose, plane: plane)
        return (try buildSection(survey: s.validated(), rectifier: projection), projection)
    }

    func testAimedAtTheSectionItMayRecord() throws {
        let (layout, projection) = try layout(from: .looking(fromHeight: 3, downDegrees: 25))
        XCTAssertNil(RecordingCheck.problem(layout: layout, projection: projection))
    }

    /// Facing away from the river: StivKit falls back to 128 samples a line,
    /// and every one of them would be NaN.
    func testFacingAwayItMayNot() throws {
        let (layout, projection) = try layout(
            from: .looking(fromHeight: 3, downDegrees: 25, yawDegrees: 180))
        let problem = try XCTUnwrap(RecordingCheck.problem(layout: layout, projection: projection))
        XCTAssertTrue(problem.contains("behind"), problem)
    }

    /// Held flat over the section, close: every line runs to StivKit's cap.
    func testHeldFlatOverTheSectionItMayNot() throws {
        let (layout, projection) = try layout(
            from: .looking(fromHeight: 0.4, downDegrees: 89.9, atZ: -8))
        XCTAssertTrue(layout.searchLines.contains { $0.sampleCount >= RecordingCheck.sampleCap },
                      "the pose does not reproduce the bug")
        let problem = try XCTUnwrap(RecordingCheck.problem(layout: layout, projection: projection))
        XCTAssertTrue(problem.contains("straight down"), problem)
    }

    /// Pointing down at the bank from the bank: the section is mostly above
    /// the top of the frame.
    func testWithTheSectionMostlyOffScreenItMayNot() throws {
        let (layout, projection) = try layout(from: .looking(fromHeight: 3, downDegrees: 70))
        let problem = try XCTUnwrap(RecordingCheck.problem(layout: layout, projection: projection))
        XCTAssertTrue(problem.contains("off screen"), problem)
    }
}