//  FlowMapTests.swift
//  The flow arrows: one per station, as long as the station is fast.

import XCTest
@testable import StivKit

final class FlowMapTests: XCTestCase {
    /// Stations along world X, every metre, flowing along `bearing`.
    private func result(_ velocities: [Double], flags: [Int: [StationFlag]] = [:],
                        bearing: Double = 90, widthM: Double = 8) -> StivResult {
        let stations = velocities.indices.map { i in
            makeStation(index: i, stationM: Double(i), velocity: velocities[i], flags: flags[i] ?? [])
        }
        return StivResult(survey: Survey(), stations: stations, stiPreviews: [],
                          motion: MotionReport(), rowCount: 300, durationSeconds: 10,
                          stSecondsPerRow: 1 / 30, widthM: widthM, flowBearingDegrees: bearing,
                          cameraHeightM: 2, rectifierDescription: "", warnings: [])
    }

    func testArrowLengthIsProportionalToSpeed() throws {
        let map = FlowMap(result: result([0.2, 0.5, 1.0, 0.8]))
        XCTAssertEqual(map.arrows.count, 4)
        for arrow in map.arrows {
            XCTAssertEqual(distance(arrow.base, arrow.tip), abs(arrow.velocityMs) * map.metresPerMs,
                           accuracy: 1e-12)
        }
        // The fastest is a quarter of the section width, as on the desktop.
        let fastest = try XCTUnwrap(map.arrows.max { $0.velocityMs < $1.velocityMs })
        XCTAssertEqual(distance(fastest.base, fastest.tip), 0.25 * 8, accuracy: 1e-12)
        XCTAssertEqual(map.referenceSpeedMs, 0.5)               // 1-2-5 below 0.6 m/s
    }

    func testArrowsRunAlongTheFlowBearing() {
        // 90 degrees from +X toward +Z: the flow runs along +Z.
        let map = FlowMap(result: result([1, -0.5], bearing: 90))
        XCTAssertEqual(map.flowDirection.x, 0, accuracy: 1e-12)
        XCTAssertEqual(map.flowDirection.z, 1, accuracy: 1e-12)
        let (down, up) = (map.arrows[0], map.arrows[1])
        XCTAssertGreaterThan(down.tip.z, down.base.z)
        XCTAssertLessThan(up.tip.z, up.base.z, "a station flowing upstream points upstream")
        XCTAssertEqual(down.tip.x, down.base.x, accuracy: 1e-12)
    }

    func testFailedStationsAreMarkedNotDrawnAsMeasurements() {
        let map = FlowMap(result: result([0.4, .nan, 0.6, 0.5],
                                         flags: [1: [.lowCoherency], 2: [.lowCoherency, .interpolated]]))
        XCTAssertEqual(map.arrows.map(\.index), [0, 2, 3])
        XCTAssertEqual(map.missing.map(\.index), [1])
        XCTAssertEqual(map.arrows.map(\.interpolated), [false, true, false])
    }

    func testNoVelocityAtAllDrawsNoArrows() {
        let map = FlowMap(result: result([.nan, .nan]))
        XCTAssertTrue(map.arrows.isEmpty)
        XCTAssertEqual(map.missing.count, 2)
        XCTAssertEqual(map.metresPerMs, 0)
    }

    /// `plotting._nice_speed` gives the same numbers.
    func testReferenceSpeedMatchesTheDesktop() {
        let cases: [(Double, Double)] = [(0, 1), (0.03, 0.02), (0.72, 0.5), (1, 1), (1.36, 1),
                                         (2, 2), (4.99, 2), (12, 10)]
        for (x, expected) in cases {
            XCTAssertEqual(niceSpeed(x), expected, accuracy: 1e-12, "niceSpeed(\(x))")
        }
    }

    func testLabelsAreSpreadAcrossTheSection() {
        let map = FlowMap(result: result((0..<30).map { 0.5 + 0.01 * Double($0) }, widthM: 29))
        let labelled = map.labelledIndices(limit: 8)
        XCTAssertEqual(labelled.count, 8)
        XCTAssertEqual(labelled.first, 0)
        XCTAssertEqual(labelled.last, 29)
        XCTAssertEqual(FlowMap(result: result([1, 2, 3])).labelledIndices(), [0, 1, 2])
    }
}
