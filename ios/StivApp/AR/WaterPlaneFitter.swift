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
//  at where the water meets the shore, sweep along it, and the lowest returns
//  in the box are a direct measurement of `y0`.
//
//  "The lowest" is the important word.  A box straddling the waterline sees
//  water below the line and bank above it, and the water half returns almost
//  nothing the confidence gate lets through.  So most accepted samples are
//  bank, all of them *above* the water, and the median of everything comes out
//  ten or twenty centimetres high -- five to ten per cent of every velocity at
//  two metres' camera height.  The estimate is therefore the bottom of the
//  returns: the water's own cluster when there is one, the foot of the bank's
//  spread of heights when there is not (see `estimateLevel`), which is the
//  waterline itself however much bank shares the box with it.
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
/// phone at a riverbank, so many will land on the bank above the water, a few
/// on a passing duck or a branch.  The fit looks for the water level as the
/// lowest dense cluster of heights, and reports its scatter as a scaled MAD.
///
/// Storage is a histogram of heights, not the heights themselves: a sweep adds
/// a couple of hundred samples a frame at sixty frames a second, and neither
/// the memory nor a sort of all of them on the frame queue is warranted when
/// the answer is wanted to a few millimetres.
final class WaterPlaneFitter {
    /// LiDAR is specified to about 5 m and its accuracy degrades toward that
    /// limit.  Beyond it the returns are not measurements.
    static let maxRangeM: Double = 5.0
    /// Apple publishes no minimum; the commonly measured one is ~0.2 m, and
    /// nothing useful is being surveyed from 25 cm away in any case.
    static let minRangeM: Double = 0.3

    /// The sampled square's side as a fraction of the depth map's *short*
    /// side, centred.  A square in the image is a square on screen, and at
    /// this size it is drawn about a hundred points across: small enough that
    /// the user knows what they are aiming at.  `DepthReticle` draws exactly
    /// this region, from `reticleRect`.
    static let reticleFraction = 0.15

    // Readiness.  One frame supplies a couple of hundred samples from one spot,
    // so a sample count alone says nothing about the sweep; what makes the fit
    // trustworthy is time and ground covered.

    /// Frames that contributed returns: half a second at 60 fps.
    static let minimumFrames = 30
    /// How far the aim point must travel along the waterline, metres, so the
    /// fit averages over the bank's local irregularities rather than one spot.
    static let minimumSweepM = 0.3
    /// Enough samples that the histogram resolves a bank's foot.  Below about
    /// two thousand, a bank spread over a metre has only a handful of returns
    /// per bin and its foot can be misplaced by several centimetres; above it,
    /// by under two.  A thirty-frame sweep of the app's box at the shipped
    /// depth resolution brings up to six thousand.
    static let minimumSamples = 2000

    // The estimator.  See `estimateLevel` for how these are used.

    /// Histogram resolution: well inside the sensor's own centimetre noise.
    static let binWidthM = 0.005
    /// Half-width of the band each height's density is counted over: about
    /// twice the sensor's noise at a few metres.
    static let clusterHalfWidthM = 0.02
    /// A run of heights separated from the rest by an empty gap must hold at
    /// least this share of all returns to be the water.  Less is a reflection
    /// or multipath -- a few per cent of returns, clustered well below.
    static let minimumSurfaceShare = 0.05
    /// Band density below this share of all returns is a gap between runs.
    static let gapShare = 0.002
    /// A cluster wider than this at half height is not one surface's noise.
    static let maximumClusterWidthM = 0.20
    /// The scatter is measured over this band around the fitted level: the
    /// cluster and its shoulders, not the bank a metre above.
    static let residualBandM = 0.10

    /// Bin index to sample count.  Bounded by the height range in view, a few
    /// thousand bins at most, however long the sweep.
    private var histogram: [Int: Int] = [:]
    /// The bounding box of each frame's aim point -- the centroid of its
    /// accepted samples -- which is what moves when the user sweeps.
    private var sweep: (minX: Double, maxX: Double, minZ: Double, maxZ: Double)?

    /// How many samples have been accepted.
    private(set) var sampleCount = 0
    /// How many frames contributed at least one sample.
    private(set) var frameCount = 0
    /// Histogram bins in use: what the fitter actually stores, however many
    /// samples it has seen.
    var storedBins: Int { histogram.count }

    /// How far the aim point has travelled, metres: the longer side of its
    /// bounding box.
    var sweepExtentM: Double {
        guard let sweep else { return 0 }
        return max(sweep.maxX - sweep.minX, sweep.maxZ - sweep.minZ)
    }

    /// From 0 to 1, for the progress bar: the least advanced of the three
    /// conditions, so the bar fills only as fast as the slowest of them.
    var progress: Double {
        min(1, Double(frameCount) / Double(Self.minimumFrames),
            sweepExtentM / Self.minimumSweepM,
            Double(sampleCount) / Double(Self.minimumSamples))
    }

    /// Whether there is enough to fit.
    var isReady: Bool {
        frameCount >= Self.minimumFrames && sampleCount >= Self.minimumSamples
            && sweepExtentM >= Self.minimumSweepM
    }

    func reset() {
        histogram.removeAll(keepingCapacity: true)
        sweep = nil
        sampleCount = 0
        frameCount = 0
    }

    // MARK: - the sampled region

    /// The depth-map pixels inside the reticle: those whose centres fall in a
    /// centred square of side `fraction` times the short side.
    static func samplingRegion(depthWidth: Int, depthHeight: Int,
                               fraction: Double = reticleFraction)
        -> (x: ClosedRange<Int>, y: ClosedRange<Int>) {
        let half = 0.5 * fraction * Double(min(depthWidth, depthHeight))
        func range(_ size: Int) -> ClosedRange<Int> {
            let centre = 0.5 * Double(size)
            let first = max(Int((centre - half - 0.5).rounded(.up)), 0)
            let last = min(Int((centre + half - 0.5).rounded(.down)), size - 1)
            return first...max(first, last)
        }
        return (range(depthWidth), range(depthHeight))
    }

    /// The same square in captured-image pixels, for drawing.  The depth map
    /// shares the image's aspect ratio, so a fraction of the short side is the
    /// same square in both.
    static func reticleRect(imageWidth: Double, imageHeight: Double,
                            fraction: Double = reticleFraction) -> CGRect {
        let side = fraction * min(imageWidth, imageHeight)
        return CGRect(x: 0.5 * (imageWidth - side), y: 0.5 * (imageHeight - side),
                      width: side, height: side)
    }

    // MARK: - accumulating

    /// Add every high-confidence depth sample inside the reticle.
    ///
    /// - Parameter stride: sample every nth depth pixel.  Neighbouring LiDAR
    ///   pixels are not independent measurements, so taking all of them would
    ///   inflate the count without improving the fit.
    func add(frame: ARFrame, reticleFraction: Double = reticleFraction, stride: Int = 2) {
        guard let sceneDepth = frame.sceneDepth ?? frame.smoothedSceneDepth else { return }
        let intrinsics = frame.camera.stivIntrinsics
        let pose = frame.camera.stivPose

        _ = withDepthPlane(depthMap: sceneDepth.depthMap,
                           confidenceMap: sceneDepth.confidenceMap) { plane in
            add(depth: plane, intrinsics: intrinsics, pose: pose,
                reticleFraction: reticleFraction, stride: stride)
        }
    }

    /// The same, from a depth plane already in hand -- which, unlike an
    /// `ARFrame`, a test can construct.
    func add(depth plane: DepthPlane, intrinsics: PinholeIntrinsics, pose: CameraPose,
             reticleFraction: Double = reticleFraction, stride: Int = 2) {
        let region = Self.samplingRegion(depthWidth: plane.width, depthHeight: plane.height,
                                         fraction: reticleFraction)
        var accepted = 0
        var sumX = 0.0, sumZ = 0.0

        for y in Swift.stride(from: region.y.lowerBound, through: region.y.upperBound, by: stride) {
            for x in Swift.stride(from: region.x.lowerBound, through: region.x.upperBound,
                                  by: stride) {
                guard let depth = plane.highConfidenceDepth(x: x, y: y) else { continue }
                let range = Double(depth)
                guard range >= Self.minRangeM, range <= Self.maxRangeM else { continue }

                let world = worldPoint(
                    depthPixel: (x, y),
                    depthSize: (plane.width, plane.height),
                    depth: depth,
                    intrinsics: intrinsics,
                    pose: pose)
                histogram[Self.bin(world.y), default: 0] += 1
                accepted += 1
                sumX += world.x
                sumZ += world.z
            }
        }

        guard accepted > 0 else { return }
        sampleCount += accepted
        frameCount += 1
        recordAimPoint(x: sumX / Double(accepted), z: sumZ / Double(accepted))
    }

    private static func bin(_ y: Double) -> Int { Int((y / binWidthM).rounded(.down)) }

    private func recordAimPoint(x: Double, z: Double) {
        if var box = sweep {
            box.minX = min(box.minX, x)
            box.maxX = max(box.maxX, x)
            box.minZ = min(box.minZ, z)
            box.maxZ = max(box.maxZ, z)
            sweep = box
        } else {
            sweep = (x, x, z, z)
        }
    }

    // MARK: - fitting

    /// The fitted water surface, or `nil` until the sweep is good enough.
    ///
    /// The residual is a scaled median absolute deviation over the band
    /// `residualBandM` either side of the level, which for Gaussian scatter
    /// estimates the same sigma a standard deviation would.  It feeds
    /// `ScaleSettings.planeHeightSigmaM`, so an untidy sweep -- bank returns
    /// close above the water among them -- widens the reported uncertainty
    /// rather than silently biasing the velocities.
    func fit(measuredOnWaterSurface: Bool) -> WaterPlane? {
        guard isReady, let estimate = Self.estimateLevel(histogram: histogram) else { return nil }
        return WaterPlane(
            y: estimate.level,
            fitResidualM: estimate.sigma,
            sampleCount: sampleCount,
            measuredOnWaterSurface: measuredOnWaterSurface)
    }

    /// The same estimate from raw heights, for the tests.
    static func estimateLevel(heights: [Double]) -> (level: Double, sigma: Double)? {
        var histogram: [Int: Int] = [:]
        for y in heights { histogram[bin(y), default: 0] += 1 }
        return estimateLevel(histogram: histogram)
    }

    /// The water level, from a histogram of return heights.
    ///
    /// **What it looks for.**  Count, at every height, the returns within
    /// `clusterHalfWidthM` of it: the band density.  Split it into runs
    /// separated by empty gaps, and take the lowest run holding at least
    /// `minimumSurfaceShare` of the returns -- so a few per cent of returns
    /// clustered well below the rest, a reflection or multipath, cannot win.
    /// Climb that run from its foot while the density further up is higher by
    /// more than its own counting noise: the top of the first rise.  Then:
    ///
    /// * if the density falls away again within a cluster's width -- a bell,
    ///   the water surface's own returns, alone or with bank above -- the
    ///   level is the median of the returns across its half-height width;
    /// * if it levels off and stays up -- a plateau, the bank's spread of
    ///   heights with no water returns at all, which is the usual case at a
    ///   waterline -- the level is where the density first reaches half the
    ///   plateau: the foot of the bank, which is the waterline.
    ///
    /// The two are told apart by shape: a bell spends most of its half-height
    /// width rising and falling, a plateau rises in little more than the
    /// band's own blur.  Anything no wider than the band is one surface either
    /// way.
    ///
    /// **The assumption** is that nothing the user aims at lies below the water
    /// surface in useful numbers -- true of a waterline, since LiDAR does not
    /// see through water -- while anything above it is fair game.
    ///
    /// **What it achieves**, on two thousand returns or more (see the tests):
    /// water returns alone, within 5 mm for sensor noise up to 5 cm; a bank
    /// alone, sloping or diffuse over 25 cm to a metre, within 2 cm of its
    /// foot; water under a bank that fills most of the box, within 1.5 cm.  The
    /// one case it cannot separate is a bank spanning only about fifteen
    /// centimetres under heavy (2 cm) noise, which is bell-shaped and can read
    /// several centimetres high.
    static func estimateLevel(histogram: [Int: Int]) -> (level: Double, sigma: Double)? {
        let h = Int((clusterHalfWidthM / binWidthM).rounded())
        // Padded by a band either side, so the density falls away to nothing
        // beyond the returns instead of being clipped at the array's ends.
        guard let minKey = histogram.keys.min(), let maxKey = histogram.keys.max() else {
            return nil
        }
        let lowest = minKey - h
        let n = maxKey + h - lowest + 1
        var counts = [Int](repeating: 0, count: n)
        for (key, count) in histogram { counts[key - lowest] = count }
        var prefix = [Int](repeating: 0, count: n + 1)
        for i in 0..<n { prefix[i + 1] = prefix[i] + counts[i] }
        let total = prefix[n]
        guard total > 0 else { return nil }
        let density = (0..<n).map { i in
            Double(prefix[min(i + h, n - 1) + 1] - prefix[max(i - h, 0)])
        }
        func height(_ i: Double) -> Double { (i + 0.5 + Double(lowest)) * binWidthM }

        // Runs of heights, and the lowest that holds enough of the returns.
        let floor = max(1, gapShare * Double(total))
        var runs: [(a: Int, b: Int, mass: Int)] = []
        var i = 0
        while i < n {
            guard density[i] >= floor else { i += 1; continue }
            var j = i
            while j + 1 < n, density[j + 1] >= floor { j += 1 }
            runs.append((i, j, prefix[j + 1] - prefix[i]))
            i = j + 1
        }
        guard let run = runs.first(where: { Double($0.mass) >= minimumSurfaceShare * Double(total) })
                ?? runs.max(by: { $0.mass < $1.mass }) else { return nil }
        let a = run.a, b = run.b

        // The top of the first rise: climb while the density three bands on
        // is higher by more than three times its counting noise, then take
        // the highest point in the stretch beyond.  Neither a noise bump on
        // the way up nor a random walk along a plateau stops it.
        let reach = 3 * h
        var p = a
        while p + reach <= b, density[p + reach] > density[p] + 3 * max(density[p], 1).squareRoot() {
            p += 1
        }
        p = (p...min(p + reach, b)).max { density[$0] < density[$1] } ?? p
        let peak = density[p]

        var lo = p
        while lo > a, density[lo - 1] >= 0.5 * peak { lo -= 1 }
        var hi = p
        while hi < b, density[hi + 1] >= 0.5 * peak { hi += 1 }
        let width = hi - lo + 1
        let k10 = (a...p).first { density[$0] >= 0.1 * peak } ?? a
        let k90 = (a...p).first { density[$0] >= 0.9 * peak } ?? p
        let rise = k90 - k10 + 1
        let narrow = width <= 2 * h + 3
        let bell = Double(rise) >= 0.45 * Double(width)
            && Double(width) * binWidthM <= maximumClusterWidthM

        let level: Double
        if narrow || bell {
            // A cluster: the median of its returns across its half-height width.
            let span = prefix[hi + 1] - prefix[lo]
            var cumulative = 0
            var median = height(Double(hi))
            for k in lo...hi where counts[k] > 0 {
                if 2 * (cumulative + counts[k]) >= span {
                    let fraction = (0.5 * Double(span) - Double(cumulative)) / Double(counts[k])
                    median = (Double(k) + fraction + Double(lowest)) * binWidthM
                    break
                }
                cumulative += counts[k]
            }
            level = median
        } else {
            // A plateau: where the density first reaches half its typical
            // level -- the median over the stretch above the rise, not its
            // noisiest bin -- interpolated between bins.
            let plateau = density[p...min(p + reach, b)].sorted()
            let half = 0.5 * plateau[plateau.count / 2]
            var k = p
            while k > a, density[k - 1] >= half { k -= 1 }
            if k > a, density[k] != density[k - 1] {
                let t = (half - density[k - 1]) / (density[k] - density[k - 1])
                level = height(Double(k - 1) + t)
            } else {
                level = height(Double(k))
            }
        }

        // The scatter about that level, over the residual band.
        let band = Int((residualBandM / binWidthM).rounded())
        let centre = bin(level) - lowest
        var deviations: [(Double, Int)] = []
        for k in max(centre - band, 0)...min(max(centre + band, 0), n - 1) where counts[k] > 0 {
            deviations.append((abs(height(Double(k)) - level), counts[k]))
        }
        deviations.sort { $0.0 < $1.0 }
        let inBand = deviations.reduce(0) { $0 + $1.1 }
        var cumulative = 0
        var mad = 0.0
        for (deviation, count) in deviations {
            cumulative += count
            if 2 * cumulative >= inBand { mad = deviation; break }
        }
        // 1.4826 * MAD is the consistent estimator of sigma for a normal
        // distribution.  Floor it: a suspiciously tidy sweep does not mean the
        // sensor is better than its own specification.
        return (level, max(1.4826 * mad, 0.01))
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
        if sampleCount == 0 {
            return "Aim at the waterline -- where the water meets the bank -- within about "
                 + "5 m, and hold steady. No returns yet."
        }
        if isReady {
            return "\(sampleCount) samples over \(decimals(sweepExtentM, 1)) m of waterline. "
                 + "Ready."
        }
        if frameCount >= Self.minimumFrames, sweepExtentM < Self.minimumSweepM {
            return "\(sampleCount) samples, but all from one spot. Sweep a metre or two "
                 + "along the waterline so the fit averages over it."
        }
        return "Keep sweeping along the waterline: \(percent(progress)) of the way."
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