//  FlowMap.swift
//  One flow arrow per station, its length proportional to the surface velocity.
//
//  The geometry of `stiv/plotting.py`'s `plot_velocity_map`, without the
//  drawing: each arrow starts at its station on the water plane and runs along
//  the flow direction for `velocity * metresPerMs` metres, so the fastest is a
//  quarter of the section width, and a round reference speed gives the scale.
//  A station that failed quality control and was interpolated keeps an arrow
//  but is marked as such; one with no velocity at all has none.
//
//  The desktop draws these arrows on the video frame.  The app draws them in
//  plan, true to scale on the water plane, because a phone looks across a river
//  obliquely: projected into its frame, an arrow at the far bank would be drawn
//  several times shorter than the same speed at the near bank.

import Foundation

/// A station's arrow on the water plane.
public struct FlowArrow: Sendable, Equatable {
    public let index: Int
    public let stationM: Double
    public let base: PlanePoint
    public let tip: PlanePoint
    public let velocityMs: Double
    /// True when the velocity is an interpolation over a failed station.
    public let interpolated: Bool
}

/// The arrows for a result, and the scale they are drawn at.
public struct FlowMap: Sendable {
    /// Every station with a velocity, in station order.
    public let arrows: [FlowArrow]
    /// Stations with no velocity at all, where they are.
    public let missing: [(index: Int, world: PlanePoint)]
    /// Unit vector of the flow on the plane.
    public let flowDirection: PlaneVector
    /// Arrow length on the plane, metres, per metre per second.
    public let metresPerMs: Double
    /// A round speed -- 1, 2 or 5 times a power of ten -- for the key.
    public let referenceSpeedMs: Double
    public let stations: [PlanePoint]

    /// The longest arrow, as a fraction of the section width; the desktop's
    /// `_ARROW_FRACTION`.
    public static let longestArrowFraction = 0.25

    public init(result: StivResult) {
        let bearing = result.flowBearingDegrees * .pi / 180
        let direction = PlaneVector(x: cos(bearing), z: sin(bearing))
        let speeds = result.stations.map(\.velocityMs).filter(\.isFinite).map(abs)
        let fastest = speeds.max() ?? 0
        let k = fastest > 0 ? Self.longestArrowFraction * result.widthM / fastest : 0

        var arrows: [FlowArrow] = []
        var missing: [(index: Int, world: PlanePoint)] = []
        for station in result.stations {
            guard station.velocityMs.isFinite else {
                missing.append((station.index, station.world))
                continue
            }
            arrows.append(FlowArrow(
                index: station.index,
                stationM: station.stationM,
                base: station.world,
                tip: station.world + direction * (station.velocityMs * k),
                velocityMs: station.velocityMs,
                interpolated: station.failed && station.flags.contains(.interpolated)))
        }
        self.arrows = arrows
        self.missing = missing
        self.flowDirection = direction
        self.metresPerMs = k
        self.referenceSpeedMs = niceSpeed(0.6 * fastest)
        self.stations = result.stations.map(\.world)
    }

    /// The stations whose value is printed beside the arrow: all of them up
    /// to `limit`, otherwise evenly spaced across the section.
    public func labelledIndices(limit: Int = 8) -> [Int] {
        guard arrows.count > limit, limit > 1 else { return arrows.map(\.index) }
        var picks: [Int] = []
        for k in 0..<limit {
            let position = Int((Double(k) * Double(arrows.count - 1) / Double(limit - 1)).rounded())
            let index = arrows[position].index
            if picks.last != index { picks.append(index) }
        }
        return picks
    }
}

/// The largest of 1, 2 or 5 times a power of ten not above `x`: the speed the
/// key arrow is drawn for.  `plotting._nice_speed`.
public func niceSpeed(_ x: Double) -> Double {
    guard x.isFinite, x > 0 else { return 1 }
    let exponent = (log10(x)).rounded(.down)
    let unit = pow(10, exponent)
    for multiple in [5.0, 2.0, 1.0] where multiple * unit <= x * (1 + 1e-12) {
        return multiple * unit
    }
    return unit
}
