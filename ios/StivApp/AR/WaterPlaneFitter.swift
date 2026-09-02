//  WaterPlaneFitter.swift
//  Turning LiDAR returns into the one number the scale depends on.
//
//  The problem this solves, stated plainly: Apple's LiDAR reaches about five
//  metres, and a river cross-section is ten to fifty.  Depth alone cannot scale
//  a river.
//
//  What makes it work anyway is that a horizontal plane in a gravity-aligned
//  world frame has exactly one degree of freedom.  ARKit's +Y is gravity from
//  the IMU -- accurate to a fraction of a degree and free -- and still water is
//  horizontal, so the water surface is `y = y0` and `y0` is a scalar.  A scalar
//  is well within what LiDAR can measure at close range.
//
//  Which surface to measure, then?  Not the water: infrared is absorbed, and the
//  sensor's returns from open water are sparse and low-confidence.  The answer
//  is the **waterline** -- the contour where water meets bank.  It is at water
//  level by definition, it is texture the sensor handles well, and from a bank
//  it is the *nearest* thing in the scene, comfortably inside five metres.  Aim
//  at where the water meets the shore, sweep along it, and every accepted sample
//  is a direct measurement of `y0`.
//
//  At close range -- a flume, a culvert, a footbridge -- the depth map reaches
//  the water surface itself and the same fit runs on it unchanged, with better
//  accuracy since nothing is being inferred from a contour.  The fitter does not
//  need to know which case it is in; the caller records it for the report.

import ARKit
import Foundation
import StivKit

/// Accumulates LiDAR samples of the water surface and reduces them to a plane.
///
/// Robust statistics throughout.  The samples are gathered by a person waving a
/// phone at a riverbank, so a fraction of them will land on a passing duck, a
/// branch, or the bank a metre above the water.  A mean and a standard deviation
/// would follow those outliers; a median and a scaled MAD do not.
final class WaterPlaneFitter {
    /// LiDAR is specified to about 5 m and its accuracy degrades toward that
    /// limit.  Beyond it the returns are not measurements.
    static let maxRangeM: Double = 5.0
    /// Apple publishes no minimum; the commonly measured one is ~0.2 m, and
    /// nothing useful is being surveyed from 25 cm away in any case.
    static let minRangeM: Double = 0.3

    /// Enough samples that the median means something, and few enough that a
    /// two-second sweep supplies them.
    static let minimumSamples = 60

    private var heights: [Double] = []
    private var horizontalSpread: (minX: Double, maxX: Double, minZ: Double, maxZ: Double)?

    /// How many samples have been accepted.
    var sampleCount: Int { heights.count }

    /// Whether there are enough samples to fit.
    var isReady: Bool { heights.count >= Self.minimumSamples }

    func reset() {
        heights.removeAll(keepingCapacity: true)
        horizontalSpread = nil
    }

    /// Add every high-confidence depth sample inside a reticle at the centre of
    /// the frame.
    ///
    /// - Parameter reticleFraction: the side of the sampled square as a fraction
    ///   of the depth map, centred.  Small enough that the user knows what they
    ///   are aiming at, large enough to collect samples quickly.
    /// - Parameter stride: sample every nth depth pixel.  Neighbouring LiDAR
    ///   pixels are not independent measurements, so taking all of them would
    ///   inflate the count without improving the median.
    func add(frame: ARFrame, reticleFraction: Double = 0.25, stride: Int = 2) {
        guard let sceneDepth = frame.sceneDepth ?? frame.smoothedSceneDepth else { return }
        let intrinsics = frame.camera.stivIntrinsics
        let pose = frame.camera.stivPose

        _ = withDepthPlane(depthMap: sceneDepth.depthMap,
                           confidenceMap: sceneDepth.confidenceMap) { plane in
            let halfWidth = Int(Double(plane.width) * reticleFraction / 2)
            let halfHeight = Int(Double(plane.height) * reticleFraction / 2)
            let centreX = plane.width / 2
            let centreY = plane.height / 2

            for y in Swift.stride(from: centreY - halfHeight,
                                  through: centreY + halfHeight, by: stride) {
                for x in Swift.stride(from: centreX - halfWidth,
                                      through: centreX + halfWidth, by: stride) {
                    guard let depth = plane.highConfidenceDepth(x: x, y: y) else { continue }
                    let range = Double(depth)
                    guard range >= Self.minRangeM, range <= Self.maxRangeM else { continue }

                    let world = worldPoint(
                        depthPixel: (x, y),
                        depthSize: (plane.width, plane.height),
                        depth: depth,
                        intrinsics: intrinsics,
                        pose: pose)
                    heights.append(world.y)
                    record(x: world.x, z: world.z)
                }
            }
        }
    }

    private func record(x: Double, z: Double) {
        if var spread = horizontalSpread {
            spread.minX = min(spread.minX, x)
            spread.maxX = max(spread.maxX, x)
            spread.minZ = min(spread.minZ, z)
            spread.maxZ = max(spread.maxZ, z)
            horizontalSpread = spread
        } else {
            horizontalSpread = (x, x, z, z)
        }
    }

    /// The fitted water surface, or `nil` until there are enough samples.
    ///
    /// The residual is a scaled median absolute deviation, which for Gaussian
    /// scatter estimates the same sigma a standard deviation would but ignores
    /// the samples that landed on the bank instead of the water.  It feeds
    /// `ScaleSettings.planeHeightSigmaM`, so an untidy sweep widens the reported
    /// uncertainty rather than silently biasing the velocities.
    func fit(measuredOnWaterSurface: Bool) -> WaterPlane? {
        guard isReady else { return nil }
        let sorted = heights.sorted()
        let median = sorted[sorted.count / 2]
        let deviations = sorted.map { abs($0 - median) }.sorted()
        let mad = deviations[deviations.count / 2]
        // 1.4826 * MAD is the consistent estimator of sigma for a normal
        // distribution.  Floor it: a suspiciously tidy sweep does not mean the
        // sensor is better than its own specification.
        let sigma = max(1.4826 * mad, 0.01)

        return WaterPlane(
            y: median,
            fitResidualM: sigma,
            sampleCount: heights.count,
            measuredOnWaterSurface: measuredOnWaterSurface)
    }

    /// A plane whose height was typed in rather than measured, for devices with
    /// no LiDAR.
    ///
    /// The geometry is identical -- gravity still supplies the normal, and the
    /// camera's own world Y is known -- so only the provenance of `y0` changes.
    /// A person's estimate of their own eye height is good to perhaps 10 cm,
    /// which is what the residual says.
    static func manual(cameraHeightAboveWater: Double, cameraWorldY: Double) -> WaterPlane {
        WaterPlane(
            y: cameraWorldY - cameraHeightAboveWater,
            fitResidualM: 0.10,
            sampleCount: 0,
            measuredOnWaterSurface: false)
    }

    /// Advice for the user while they are aiming, phrased as what to do next.
    func guidance() -> String {
        if heights.isEmpty {
            return "Aim at the waterline -- where the water meets the bank -- within about "
                 + "5 m, and hold steady. No returns yet."
        }
        if !isReady {
            return "Keep sweeping along the waterline: \(heights.count) of "
                 + "\(Self.minimumSamples) samples."
        }
        guard let spread = horizontalSpread else { return "Ready." }
        let extent = max(spread.maxX - spread.minX, spread.maxZ - spread.minZ)
        if extent < 0.3 {
            return "\(heights.count) samples, but all from one spot. Sweep a metre or two "
                 + "along the waterline so the fit averages over it."
        }
        return "\(heights.count) samples over \(decimals(extent, 1)) m of waterline. Ready."
    }
}

/// Whether this device can measure the water plane at all.
enum LidarAvailability {
    case available
    /// ARKit works, but there is no depth sensor: the user types the camera
    /// height instead.  Every iPhone that is not a Pro is in this case, and
    /// there is no `UIRequiredDeviceCapabilities` value that could have kept
    /// them out of the App Store -- LiDAR can only be checked at runtime.
    case noDepthSensor
    /// No world tracking at all; nothing here can work.
    case noWorldTracking

    static var current: LidarAvailability {
        guard ARWorldTrackingConfiguration.isSupported else { return .noWorldTracking }
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            return .noDepthSensor
        }
        return .available
    }

    var explanation: String {
        switch self {
        case .available:
            return "LiDAR available: the water level can be measured directly."
        case .noDepthSensor:
            return """
                This device has no LiDAR scanner, so the water level cannot be measured. \
                You can still take a measurement by entering the camera's height above the \
                water surface -- the rest of the geometry comes from gravity and works the \
                same way -- but the scale will be only as good as that estimate.
                """
        case .noWorldTracking:
            return "This device does not support ARKit world tracking, which this app needs "
                 + "for both scale and camera-motion compensation."
        }
    }
}
