//  WaterPlaneFitterTests.swift
//  The one number every metre depends on: the height of the water surface.
//
//  Depth maps are rendered for a camera looking at horizontal planes, through
//  the same back-projection the fitter uses, so the right answer is the water
//  plane's height exactly.  What is tested is everything around that: the
//  estimator's choice of *which* returns are the water, the range and
//  confidence gates, the sweep the user is asked for, and the box they aim.

import CoreVideo
import StivKit
import XCTest
@testable import STIV

final class WaterPlaneFitterTests: XCTestCase {
    private let intrinsics = PinholeIntrinsics(horizontalFovDegrees: 60, width: 1920, height: 1440)
    /// The shipped LiDAR resolution is 256x192; a quarter of it is plenty here.
    private let depthSize = (width: 64, height: 48)
    /// A larger box than the app's, so each frame of this coarse depth map
    /// still brings a hundred samples, and a sweep the two thousand the fit
    /// needs.
    private let reticle = 0.4

    /// A phone held 1.2 m above water at y = -0.4, aimed down at the waterline.
    private let waterY = -0.4
    private let pose = CameraPose.looking(fromHeight: 0.8, downDegrees: 45)

    private func projection(planeY: Double, pose: CameraPose) -> WaterPlaneProjection {
        WaterPlaneProjection(intrinsics: intrinsics, pose: pose, plane: WaterPlane(y: planeY))
    }

    /// Feed `fitter` one depth map, built by `sample` for a camera at `pose`.
    private func add(_ fitter: WaterPlaneFitter, pose: CameraPose? = nil,
                     reticleFraction: Double? = nil,
                     size: (width: Int, height: Int)? = nil,
                     _ sample: (_ pose: CameraPose, _ x: Int, _ y: Int)
                        -> (depth: Float, confidence: UInt8)) {
        let pose = pose ?? self.pose
        let size = size ?? depthSize
        let (depth, confidence) = makeDepthBuffers(width: size.width,
                                                   height: size.height) { sample(pose, $0, $1) }
        _ = withDepthPlane(depthMap: depth, confidenceMap: confidence) { plane in
            fitter.add(depth: plane, intrinsics: intrinsics, pose: pose,
                       reticleFraction: reticleFraction ?? reticle)
        }
    }

    /// Sweep the aim point `metres` along the waterline over `frames` frames:
    /// the camera slides sideways, as a person panning along a bank does.
    private func sweep(_ fitter: WaterPlaneFitter, frames: Int = 40, metres: Double = 0.6,
                       _ sample: (_ pose: CameraPose, _ x: Int, _ y: Int)
                        -> (depth: Float, confidence: UInt8)) {
        for i in 0..<frames {
            let x = metres * (Double(i) / Double(max(frames - 1, 1)) - 0.5)
            add(fitter, pose: .looking(fromHeight: 0.8, downDegrees: 45, atX: x), sample)
        }
    }

    /// The returns of a surface at `planeY` as seen from `pose`.
    private func surface(_ planeY: Double, _ pose: CameraPose, _ x: Int, _ y: Int)
        -> (depth: Float, confidence: UInt8) {
        planeDepth(x: x, y: y, depthSize: depthSize, projection: projection(planeY: planeY, pose: pose))
    }

    // MARK: - which returns are the water

    func testReturnsFromTheBankDoNotMoveTheFit() throws {
        let fitter = WaterPlaneFitter()
        var rng = DeterministicNoise(seed: 3)
        // A quarter of the returns land on the bank half a metre above the
        // water -- far more than a careful sweep would collect.
        sweep(fitter) { pose, x, y in
            let onBank = (x + 3 * y) % 4 == 0
            var s = self.surface(onBank ? self.waterY + 0.5 : self.waterY, pose, x, y)
            s.depth += Float(0.004 * rng.next())   // sensor noise, a few mm
            return s
        }

        XCTAssertTrue(fitter.isReady, fitter.guidance())
        let plane = try XCTUnwrap(fitter.fit(measuredOnWaterSurface: false))
        XCTAssertEqual(plane.y, waterY, accuracy: 0.01, "a 25% contamination moved the fit")
        XCTAssertEqual(plane.sampleCount, fitter.sampleCount)
        XCTAssertGreaterThanOrEqual(plane.fitResidualM, 0.01,
                                    "a tidy sweep must not claim better than the sensor")
        XCTAssertLessThan(plane.fitResidualM, 0.1)
        XCTAssertFalse(plane.measuredOnWaterSurface)
    }

    /// The usual case at a real waterline: the water half of the box returns
    /// little that passes the confidence gate, so most accepted samples are
    /// bank -- all of them above the water.  A median of everything sits a
    /// tenth of a metre high; the water surface is the lowest dense cluster.
    func testABankFillingMostOfTheBoxDoesNotLiftTheLevel() throws {
        // The bank's height above the water, from uniform noise in [-1, 1].
        let banks: [(String, (Double) -> Double)] = [
            ("sloping", { u in 0.4 * (u + 1) / 2 }),   // anywhere from 0 to 40 cm up
            ("terrace", { _ in 0.3 }),                 // a flat top 30 cm up
        ]
        for (name, bank) in banks {
            let fitter = WaterPlaneFitter()
            var rng = DeterministicNoise(seed: 17)
            var heights: [Double] = []
            sweep(fitter) { pose, x, y in
                // Seven returns in ten from the bank.
                let onBank = (x * 7 + y * 3) % 10 < 7
                let h = onBank ? self.waterY + bank(rng.next()) : self.waterY
                var s = self.surface(h, pose, x, y)
                if s.confidence > 0 { heights.append(h) }
                s.depth += Float(0.004 * rng.next())
                return s
            }
            let plainMedian = heights.sorted()[heights.count / 2]
            XCTAssertGreaterThan(plainMedian - waterY, 0.05,
                                 "\(name): the bank does not dominate; the test proves nothing")

            let plane = try XCTUnwrap(fitter.fit(measuredOnWaterSurface: false), name)
            XCTAssertEqual(plane.y, waterY, accuracy: 0.01,
                           "\(name): the bank lifted the water level")
            XCTAssertLessThan(plane.fitResidualM, 0.1, name)
        }
    }

    /// The estimator takes the *lowest* dense band, so it must not be fooled
    /// by a scatter of spurious returns below the water.
    func testAFewSpuriousLowReturnsCannotSetTheLevel() throws {
        let fitter = WaterPlaneFitter()
        sweep(fitter) { pose, x, y in
            let spurious = (x * 11 + y * 5) % 33 == 0   // three in a hundred
            return self.surface(spurious ? self.waterY - 0.6 : self.waterY, pose, x, y)
        }
        let plane = try XCTUnwrap(fitter.fit(measuredOnWaterSurface: true))
        XCTAssertEqual(plane.y, waterY, accuracy: 0.01)
        XCTAssertTrue(plane.measuredOnWaterSurface)
    }

    // MARK: - the estimator, on heights alone

    /// Heights drawn uniformly over `range`, plus Gaussian sensor noise.
    private func heights(_ n: Int, over range: ClosedRange<Double>, noise: Double,
                         _ rng: inout GaussianNoise) -> [Double] {
        (0..<n).map { _ in
            range.lowerBound + (range.upperBound - range.lowerBound) * rng.uniform()
                + noise * rng.next()
        }
    }

    /// No water returns at all -- the usual case at a waterline, where the
    /// water half of the box returns nothing the confidence gate passes --
    /// leaves only the bank's heights, from the waterline up.  The level is
    /// their foot, to within the two centimetres the estimator promises.
    func testABankAloneReadsAtItsFoot() throws {
        var rng = GaussianNoise(seed: 21)
        for top in [0.25, 0.4, 1.0] {
            for noise in [0.005, 0.01] {
                for trial in 0..<4 {
                    let level = try XCTUnwrap(WaterPlaneFitter.estimateLevel(
                        heights: heights(6000, over: 0...top, noise: noise, &rng))).level
                    XCTAssertEqual(level, 0, accuracy: 0.02,
                                   "bank 0-\(top) m, noise \(noise) m, trial \(trial)")
                }
            }
        }
    }

    /// A reflection or multipath: three in a hundred returns clustered thirty
    /// centimetres below a bank spread over a metre.  The bank's own band
    /// density is barely higher than the cluster's, so a threshold on density
    /// alone takes the cluster; it holds too small a share of the returns to
    /// be the water.
    func testAFewClusteredLowReturnsUnderADiffuseBankAreIgnored() throws {
        var rng = GaussianNoise(seed: 31)
        for trial in 0..<6 {
            let bank = heights(5820, over: 0...1.0, noise: 0.01, &rng)
            let low = (0..<180).map { _ in -0.3 + 0.01 * rng.next() }
            let level = try XCTUnwrap(WaterPlaneFitter.estimateLevel(heights: bank + low)).level
            XCTAssertEqual(level, 0, accuracy: 0.02, "trial \(trial)")
        }
    }

    /// The water surface's own returns, with noise from a few millimetres to
    /// five centimetres: the level is their centre, to within 5 mm.
    func testSensorNoiseDoesNotBiasTheWaterSurface() throws {
        var rng = GaussianNoise(seed: 41)
        for sigma in [0.005, 0.01, 0.03, 0.05] {
            for trial in 0..<4 {
                let water = (0..<6000).map { _ in sigma * rng.next() }
                let level = try XCTUnwrap(WaterPlaneFitter.estimateLevel(heights: water)).level
                XCTAssertEqual(level, 0, accuracy: 0.005, "sigma \(sigma) m, trial \(trial)")
            }
        }
    }

    /// Water under a bank that fills most of the box, at sensor noise up to
    /// three centimetres: within a centimetre and a half.
    func testWaterUnderAMostlyBankBoxReadsAtTheWater() throws {
        var rng = GaussianNoise(seed: 51)
        for sigma in [0.005, 0.01, 0.02, 0.03] {
            let water = (0..<1800).map { _ in sigma * rng.next() }
            let bank = heights(4200, over: 0...0.4, noise: sigma, &rng)
            let level = try XCTUnwrap(WaterPlaneFitter.estimateLevel(heights: water + bank)).level
            XCTAssertEqual(level, 0, accuracy: 0.015, "sigma \(sigma) m")
        }
    }

    // MARK: - the gates

    func testNotEnoughSamplesMeansNoFit() {
        let fitter = WaterPlaneFitter()
        XCTAssertNil(fitter.fit(measuredOnWaterSurface: false))
        XCTAssertTrue(fitter.guidance().contains("No returns yet"))

        // A single high-confidence return is not a plane.
        add(fitter) { pose, x, y in
            x == 32 && y == 24 ? self.surface(self.waterY, pose, x, y) : (0, 0)
        }
        XCTAssertEqual(fitter.sampleCount, 1)
        XCTAssertNil(fitter.fit(measuredOnWaterSurface: false))
        XCTAssertTrue(fitter.guidance().contains("Keep sweeping"), fitter.guidance())
    }

    /// LiDAR's specified reach is about five metres; beyond it a return is not a
    /// measurement.  Looking straight down from six metres, nothing qualifies.
    func testReturnsBeyondTheSensorsReachAreIgnored() {
        let fitter = WaterPlaneFitter()
        let high = CameraPose.looking(fromHeight: 6, downDegrees: 89.9)
        add(fitter, pose: high) { pose, x, y in self.surface(0, pose, x, y) }
        XCTAssertEqual(fitter.sampleCount, 0)
    }

    /// Medium-confidence returns are common on water -- infrared is absorbed
    /// and the sensor guesses -- and a guess would set the scale.
    func testLowConfidenceReturnsAreIgnored() {
        let fitter = WaterPlaneFitter()
        add(fitter) { pose, x, y in
            var s = self.surface(self.waterY, pose, x, y)
            s.confidence = 1   // ARConfidenceLevel.medium
            return s
        }
        XCTAssertEqual(fitter.sampleCount, 0)
    }

    // MARK: - the sweep

    /// One frame brings a couple of hundred samples from one spot.  A gate on
    /// the sample count alone would open on it, and the progress bar would
    /// jump to full before the user had moved.
    func testOneFrameIsNotASweepWhateverItsSampleCount() {
        let fitter = WaterPlaneFitter()
        // The shipped depth resolution and the whole frame: thousands of
        // samples, from one spot, at one instant.
        let full = (width: 256, height: 192)
        add(fitter, reticleFraction: 1, size: full) { pose, x, y in
            planeDepth(x: x, y: y, depthSize: full,
                       projection: self.projection(planeY: self.waterY, pose: pose))
        }
        XCTAssertGreaterThanOrEqual(fitter.sampleCount, WaterPlaneFitter.minimumSamples,
                                    "the frame must be rich enough to fool a sample count")
        XCTAssertFalse(fitter.isReady)
        XCTAssertLessThanOrEqual(fitter.progress, 1.0 / Double(WaterPlaneFitter.minimumFrames))
        XCTAssertNil(fitter.fit(measuredOnWaterSurface: false))
    }

    func testHoldingStillIsNotASweep() {
        let fitter = WaterPlaneFitter()
        let still: (CameraPose, Int, Int) -> (depth: Float, confidence: UInt8) = { pose, x, y in
            self.surface(self.waterY, pose, x, y)
        }
        for _ in 0..<(2 * WaterPlaneFitter.minimumFrames) { add(fitter, still) }
        XCTAssertFalse(fitter.isReady, "time alone is not a sweep")
        XCTAssertTrue(fitter.guidance().contains("all from one spot"), fitter.guidance())

        sweep(fitter, still)
        XCTAssertTrue(fitter.isReady)
        XCTAssertEqual(fitter.progress, 1)
        XCTAssertTrue(fitter.guidance().hasSuffix("Ready."), fitter.guidance())
    }

    /// Sixty frames a second of a couple of hundred samples each must not grow
    /// without bound, or make `fit` sort it all on the frame queue.
    func testStorageStaysBoundedHoweverLongTheSweep() throws {
        let fitter = WaterPlaneFitter()
        var rng = DeterministicNoise(seed: 5)
        sweep(fitter, frames: 600) { pose, x, y in
            var s = self.surface((x + y) % 4 == 0 ? self.waterY + 0.5 : self.waterY, pose, x, y)
            s.depth += Float(0.004 * rng.next())
            return s
        }
        XCTAssertGreaterThan(fitter.sampleCount, 10_000)
        XCTAssertLessThan(fitter.storedBins, 60,
                          "\(fitter.sampleCount) samples stored as \(fitter.storedBins) bins")
        let plane = try XCTUnwrap(fitter.fit(measuredOnWaterSurface: false))
        XCTAssertEqual(plane.y, waterY, accuracy: 0.01)
    }

    func testResetForgetsEverything() {
        let fitter = WaterPlaneFitter()
        sweep(fitter) { pose, x, y in self.surface(self.waterY, pose, x, y) }
        XCTAssertTrue(fitter.isReady)
        fitter.reset()
        XCTAssertEqual(fitter.sampleCount, 0)
        XCTAssertEqual(fitter.frameCount, 0)
        XCTAssertEqual(fitter.sweepExtentM, 0)
        XCTAssertEqual(fitter.storedBins, 0)
        XCTAssertNil(fitter.fit(measuredOnWaterSurface: true))
    }

    func testATypedHeightBecomesAPlaneBelowTheCamera() {
        let plane = WaterPlaneFitter.manual(cameraHeightAboveWater: 1.5, cameraWorldY: 0.2)
        XCTAssertEqual(plane.y, -1.3, accuracy: 1e-12)
        XCTAssertEqual(plane.fitResidualM, 0.10, "a person's estimate is good to ~10 cm")
        XCTAssertEqual(plane.sampleCount, 0)
        XCTAssertFalse(plane.measuredOnWaterSurface)
    }

    // MARK: - the box the user aims

    /// The reticle is drawn from `reticleRect` and the fit samples
    /// `samplingRegion`: they must be the same square, pixel for pixel of the
    /// depth map.
    func testTheDrawnBoxIsTheSampledRegion() {
        let depth = (width: 256, height: 192)
        let image = CGSize(width: 1920, height: 1440)
        let scale = image.width / CGFloat(depth.width)
        let region = WaterPlaneFitter.samplingRegion(depthWidth: depth.width,
                                                     depthHeight: depth.height)
        let rect = WaterPlaneFitter.reticleRect(imageWidth: image.width,
                                                imageHeight: image.height)
        XCTAssertEqual(rect.width, rect.height, "a square in the image is a square on screen")

        func centre(_ x: Int, _ y: Int) -> CGPoint {
            CGPoint(x: (CGFloat(x) + 0.5) * scale, y: (CGFloat(y) + 0.5) * scale)
        }
        for x in [region.x.lowerBound, region.x.upperBound] {
            for y in [region.y.lowerBound, region.y.upperBound] {
                XCTAssertTrue(rect.contains(centre(x, y)), "sampled (\(x), \(y)) is outside the box")
            }
        }
        XCTAssertFalse(rect.contains(centre(region.x.lowerBound - 1, region.y.lowerBound)))
        XCTAssertFalse(rect.contains(centre(region.x.upperBound + 1, region.y.upperBound)))
        XCTAssertFalse(rect.contains(centre(region.x.lowerBound, region.y.upperBound + 1)))

        // And on screen: about a hundred points square, centred, in either
        // orientation -- not a quarter of the screen, which on an aspect-fill
        // feed is a different and smaller part of the image.
        for (orientation, size) in [(UIInterfaceOrientation.portrait, CGSize(width: 393, height: 852)),
                                    (.landscapeRight, CGSize(width: 852, height: 393))] {
            guard let box = DepthReticle(imageResolution: image, viewportSize: size,
                                         orientation: orientation, active: false).rect else {
                XCTFail("no reticle drawn for \(orientation)")
                continue
            }
            XCTAssertEqual(box.width, box.height, accuracy: 1e-6)
            XCTAssertEqual(box.width, 0.15 * 1440 * 852 / 1920, accuracy: 0.5)
            XCTAssertEqual(box.midX, size.width / 2, accuracy: 1e-6)
            XCTAssertEqual(box.midY, size.height / 2, accuracy: 1e-6)
        }
    }
}

/// Standard normal noise, deterministic: Box-Muller over `DeterministicNoise`.
struct GaussianNoise {
    private var source: DeterministicNoise
    init(seed: UInt64) { source = DeterministicNoise(seed: seed) }
    /// Uniform in (0, 1].
    mutating func uniform() -> Double { 0.5 * (source.next() + 1) + 1e-12 }
    mutating func next() -> Double {
        let u1 = min(uniform(), 1), u2 = uniform()
        return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
    }
}

/// Uniform noise in [-1, 1], deterministic.  splitmix64 again: a flaky fit
/// test would be worse than none.
struct DeterministicNoise {
    private var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }
    mutating func next() -> Double {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        z ^= z >> 31
        return Double(z >> 11) / 9007199254740992.0 * 2 - 1
    }
}