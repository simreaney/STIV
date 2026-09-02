//  PoseResidual.swift
//  Cross-checking ARKit's pose against what the image actually did.
//
//  The desktop pipeline compensates for camera motion by tracking static
//  features and fitting a per-frame image transform (`stiv/stabilise.py`).  This
//  app does not need to: the search-line sample points are fixed *world* points
//  on the water plane, and every frame projects them through its own
//  `ARCamera.transform`.  That is full 6-DoF compensation, for free, and
//  strictly better than a 2-D image warp -- it is also the natural extension of
//  the choice `stiv/geometry.py` already made, to space samples in world metres
//  rather than in pixels.
//
//  But "for free" is doing some work in that sentence.  Visual-inertial odometry
//  drifts, and it drifts worst looking at open water, which offers little
//  parallax and a surface that moves.  If the pose is wrong, the STI is sampling
//  the wrong patch of river and the velocity is wrong with it, silently.  So
//  this file answers one question: **did the pose fully explain the image
//  motion?**  If it did, the residual is zero.
//
//  Three decisions worth defending:
//
//  * **Patches are anchored in 3-D, not in 2-D.**  Predicting where a bank
//    feature moves under a known camera motion requires its depth -- parallax is
//    depth-dependent, and a 2-D tracker sidesteps that only by fitting a global
//    warp a translating camera does not have.  LiDAR supplies the depth, so each
//    patch is a genuine world point and its predicted image position is exact.
//    The cost: only patches inside the sensor's ~5 m reach qualify, so the check
//    validates the *near field* and the far field is assumed to follow.  That is
//    a real limitation, and `MotionReport` says so rather than implying a clean
//    bill of health.
//
//  * **Translation-only Lucas-Kanade**, not the pyramidal six-parameter tracker
//    of `stabilise.py`.  The pose has already removed the bulk motion, so the
//    leftover is sub-pixel to a few pixels; a small integer search plus a
//    parabolic sub-pixel step resolves that without a pyramid, a library, or a
//    Fourier transform.
//
//  * **Every nth frame.**  Aligning 24 patches over a full search grid is tens
//    of millions of operations; at 60 Hz that would compete with the STI
//    sampling that is the actual measurement.  Drift is a slow phenomenon and
//    10 Hz resolves it completely, so the check is decimated and the STI is not.
//
//  The residual is not only diagnosed but optionally corrected: a similarity
//  fitted to the per-patch residuals is composed onto the projected sample
//  points.  Capped, because a large residual means the geometry is wrong rather
//  than shifted, and shifting it further would only hide that.

import ARKit
import Foundation
import StivKit

/// One patch of static ground with a known world position.
private struct ReferencePatch {
    /// World metres, from the reference frame's depth.  Fixed for the recording.
    let world: (x: Double, y: Double, z: Double)
    /// Zero-mean, unit-variance luma template, `(2 * half + 1)` square.
    ///
    /// Normalised once, here, so that per-frame alignment is invariant to the
    /// exposure and white-balance changes the camera makes without being asked
    /// -- on a river in changing light, most of them.
    let template: [Float]
}

/// A 2-D similarity, `q = s R p + t`: the residual motion the pose missed.
struct ResidualSimilarity {
    var scaleCos: Double = 1
    var scaleSin: Double = 0
    var tx: Double = 0
    var ty: Double = 0

    static let identity = ResidualSimilarity()

    func apply(_ p: PixelPoint) -> PixelPoint {
        PixelPoint(u: scaleCos * p.u - scaleSin * p.v + tx,
                   v: scaleSin * p.u + scaleCos * p.v + ty)
    }

    /// How far this transform moves a point `radius` pixels from the origin --
    /// the honest single number for "how big is the correction".
    func magnitude(atRadius radius: Double) -> Double {
        let rotationScale = ((scaleCos - 1) * (scaleCos - 1) + scaleSin * scaleSin).squareRoot()
        return (tx * tx + ty * ty).squareRoot() + rotationScale * radius
    }
}

/// Cross-checks the pose and reports how much it missed by.
final class PoseResidualChecker {
    /// Patch half-width in captured-image pixels: 21 px square.
    private static let half = 10
    /// Integer search radius.  The pose should leave far less than this; the
    /// margin exists so a genuine failure is *measured* rather than clipped.
    private static let searchRadius = 4
    private static let side = 2 * half + 1

    /// Check one frame in this many.  See the note on decimation above.
    let frameInterval: Int
    /// Above this correction, the geometry is wrong rather than shifted: report,
    /// do not correct.
    private let maxCorrectionPx: Double

    private var patches: [ReferencePatch] = []
    /// Where the section centre sat in the reference frame, so the reported
    /// frame shift is a displacement rather than a distance from the lens axis.
    private var referenceSectionCentre: PixelPoint?
    /// Reused so per-frame alignment allocates nothing.
    private var scratch = [Float](repeating: 0,
                                  count: PoseResidualChecker.side * PoseResidualChecker.side)
    private var seen = 0

    private(set) var report = MotionReport()
    private(set) var latest = ResidualSimilarity.identity

    var isArmed: Bool { !patches.isEmpty }

    init(maxCorrectionPx: Double = 6.0, frameInterval: Int = 6) {
        self.maxCorrectionPx = maxCorrectionPx
        self.frameInterval = max(1, frameInterval)
        report.enabled = false
    }

    // MARK: - setup

    /// Choose patches on static ground from the reference frame.
    ///
    /// - Parameter avoid: image points the patches must stay clear of -- the
    ///   projected search lines, which are water by construction.  Same
    ///   reasoning as `stabilise.build_mask`, which excludes a corridor around
    ///   the search lines: a tracker that locks onto the moving water surface
    ///   reports the water's motion as the camera's, and the velocity collapses
    ///   toward zero.
    /// - Parameter waterPolygon: an optional outline of the water in
    ///   captured-image pixels.  Worth asking the user for -- without it only
    ///   the search-line corridor is excluded, and a wide river fills the frame.
    func arm(
        frame: ARFrame,
        avoid: [PixelPoint],
        avoidRadiusPx: Double = 90,
        waterPolygon: [PixelPoint]? = nil,
        sectionCentre: PixelPoint? = nil,
        maxPatches: Int = 24
    ) {
        patches.removeAll()
        report = MotionReport()
        referenceSectionCentre = sectionCentre
        seen = 0

        guard let sceneDepth = frame.sceneDepth ?? frame.smoothedSceneDepth else {
            // No depth: no 3-D anchor, so no honest prediction.  The pose still
            // does the compensating; it simply goes unverified, and the report
            // states that rather than implying a clean check.
            report.enabled = false
            return
        }

        let intrinsics = frame.camera.stivIntrinsics
        let pose = frame.camera.stivPose
        var candidates: [(patch: ReferencePatch, variance: Double)] = []

        _ = withDepthPlane(depthMap: sceneDepth.depthMap,
                           confidenceMap: sceneDepth.confidenceMap) { depthPlane in
            _ = withLumaPlane(of: frame.capturedImage) { luma in
                let scaleU = Double(intrinsics.width) / Double(depthPlane.width)
                let scaleV = Double(intrinsics.height) / Double(depthPlane.height)
                let margin = Double(Self.half + Self.searchRadius + 2)

                // A coarse grid over the depth map: one candidate per cell, so
                // patches spread over the frame instead of clustering on the one
                // high-contrast rock.
                let step = 6
                for dy in Swift.stride(from: 2, to: depthPlane.height - 2, by: step) {
                    for dx in Swift.stride(from: 2, to: depthPlane.width - 2, by: step) {
                        guard let depth = depthPlane.highConfidenceDepth(x: dx, y: dy) else {
                            continue
                        }
                        let range = Double(depth)
                        guard range >= WaterPlaneFitter.minRangeM,
                              range <= WaterPlaneFitter.maxRangeM else { continue }

                        let centre = PixelPoint(u: (Double(dx) + 0.5) * scaleU,
                                                v: (Double(dy) + 0.5) * scaleV)
                        guard centre.u >= margin, centre.v >= margin,
                              centre.u <= Double(luma.width) - margin,
                              centre.v <= Double(luma.height) - margin else { continue }

                        // Keep clear of the water.
                        if avoid.contains(where: {
                            hypot($0.u - centre.u, $0.v - centre.v) < avoidRadiusPx
                        }) { continue }
                        if let waterPolygon, contains(polygon: waterPolygon, point: centre) {
                            continue
                        }

                        let u = Int(centre.u.rounded())
                        let v = Int(centre.v.rounded())
                        // A textureless patch cannot be aligned, and one that
                        // cannot be aligned reports a spurious *zero* residual --
                        // which is worse than reporting nothing.
                        let variance = luma.patchVariance(centreU: u, centreV: v,
                                                          half: Self.half)
                        guard variance > 25 else { continue }

                        let world = worldPoint(
                            depthPixel: (dx, dy),
                            depthSize: (depthPlane.width, depthPlane.height),
                            depth: depth,
                            intrinsics: intrinsics,
                            pose: pose)

                        guard let template = normalisedTemplate(
                            luma: luma, centreU: u, centreV: v) else { continue }
                        candidates.append((ReferencePatch(world: world, template: template),
                                           variance))
                    }
                }
            }
        }

        // Strongest texture first: that is where sub-pixel alignment is sharpest.
        patches = candidates.sorted { $0.variance > $1.variance }
            .prefix(maxPatches).map(\.patch)
        report.patchCount = patches.count
        report.enabled = !patches.isEmpty
    }

    private func normalisedTemplate(luma: LumaPlane, centreU: Int, centreV: Int) -> [Float]? {
        var values = [Float](repeating: 0, count: Self.side * Self.side)
        var sum: Float = 0
        var sumSquares: Float = 0
        for j in 0..<Self.side {
            for i in 0..<Self.side {
                guard let value = luma.sample(u: Double(centreU - Self.half + i),
                                              v: Double(centreV - Self.half + j))
                else { return nil }
                values[j * Self.side + i] = value
                sum += value
                sumSquares += value * value
            }
        }
        let n = Float(values.count)
        let mean = sum / n
        let std = max(sumSquares / n - mean * mean, 0).squareRoot()
        guard std > 1 else { return nil }
        for k in values.indices { values[k] = (values[k] - mean) / std }
        return values
    }

    // MARK: - per-frame check

    /// Measure this frame's residual and return the correction to compose onto
    /// the projected sample points.
    ///
    /// Returns the last accepted correction on frames it skips, so the caller
    /// can apply it unconditionally without knowing about decimation.
    @discardableResult
    func check(frame: ARFrame, sectionCentreNow: PixelPoint?) -> ResidualSimilarity {
        guard report.enabled, !patches.isEmpty else { return .identity }

        if case .normal = frame.camera.trackingState {} else {
            report.trackingDegraded = true
        }

        seen += 1
        guard seen % frameInterval == 0 else { return latest }
        report.frameCount += 1

        let intrinsics = frame.camera.stivIntrinsics
        let pose = frame.camera.stivPose

        var predicted: [PixelPoint] = []
        var measured: [PixelPoint] = []
        predicted.reserveCapacity(patches.count)
        measured.reserveCapacity(patches.count)

        _ = withLumaPlane(of: frame.capturedImage) { luma in
            for patch in patches {
                guard let expected = project(patch.world, intrinsics: intrinsics, pose: pose),
                      let found = align(patch: patch, near: expected, in: luma) else { continue }
                predicted.append(expected)
                measured.append(found)
            }
        }

        // How far the section moved in the image.  Not an error -- the pose
        // removes it -- but it is the thing the user can act on: if the section
        // has slid off the water, no amount of correct geometry helps.
        if let sectionCentreNow, let reference = referenceSectionCentre {
            report.shiftPx.append(hypot(sectionCentreNow.u - reference.u,
                                        sectionCentreNow.v - reference.v))
        }

        guard predicted.count >= 6 else {
            report.failedCount += 1
            report.residualPx.append(.nan)
            report.inlierFraction.append(0)
            latest = .identity
            return .identity
        }

        let fit = fitSimilarity(from: predicted, to: measured)
        report.residualPx.append(fit.rmse)
        report.inlierFraction.append(fit.inlierFraction)

        let radius = 0.5 * hypot(Double(intrinsics.width), Double(intrinsics.height))
        let correction = fit.transform.magnitude(atRadius: radius)
        guard correction <= maxCorrectionPx else {
            // A residual this large is not a shift to be undone: the pose or the
            // plane is wrong, and warping the samples on top of that would hide
            // exactly what the user needs to know.
            report.failedCount += 1
            latest = .identity
            return .identity
        }
        report.maxCorrectionPx = max(report.maxCorrectionPx, correction)
        latest = fit.transform
        return fit.transform
    }

    // MARK: - alignment

    /// Sum of squared differences between the normalised template and a
    /// normalised patch of `luma` centred on `centre`.
    private func cost(_ centre: PixelPoint, template: [Float], luma: LumaPlane) -> Double? {
        var sum: Float = 0
        var sumSquares: Float = 0
        let side = Self.side
        for j in 0..<side {
            for i in 0..<side {
                guard let value = luma.sample(u: centre.u - Double(Self.half) + Double(i),
                                              v: centre.v - Double(Self.half) + Double(j))
                else { return nil }
                scratch[j * side + i] = value
                sum += value
                sumSquares += value * value
            }
        }
        let n = Float(side * side)
        let mean = sum / n
        let std = max(sumSquares / n - mean * mean, 0).squareRoot()
        guard std > 1 else { return nil }

        var total: Float = 0
        for k in 0..<template.count {
            let d = template[k] - (scratch[k] - mean) / std
            total += d * d
        }
        return Double(total)
    }

    private func align(patch: ReferencePatch, near guess: PixelPoint,
                       in luma: LumaPlane) -> PixelPoint? {
        let margin = Double(Self.half + Self.searchRadius + 2)
        guard guess.u >= margin, guess.v >= margin,
              guess.u <= Double(luma.width) - margin,
              guess.v <= Double(luma.height) - margin else { return nil }

        // Two-stage integer search: step 2 over the full radius, then step 1
        // around the winner.  A quarter of the evaluations of an exhaustive
        // grid, on a cost surface smooth enough that it finds the same minimum.
        var best = guess
        var bestCost = cost(guess, template: patch.template, luma: luma) ?? .infinity
        guard bestCost.isFinite else { return nil }

        for coarse in [2, 1] {
            let centre = best
            for dv in Swift.stride(from: -Self.searchRadius, through: Self.searchRadius,
                                   by: coarse) {
                for du in Swift.stride(from: -Self.searchRadius, through: Self.searchRadius,
                                       by: coarse) where du != 0 || dv != 0 {
                    let candidate = PixelPoint(u: centre.u + Double(du),
                                               v: centre.v + Double(dv))
                    if let c = cost(candidate, template: patch.template, luma: luma),
                       c < bestCost {
                        bestCost = c
                        best = candidate
                    }
                }
            }
        }

        // Sub-pixel refinement by a 3-point parabola per axis.  Cheaper than
        // assembling the Gauss-Newton normal equations, and for a cost surface
        // this smooth near its minimum, indistinguishable from them.
        for axis in 0..<2 {
            let minus = axis == 0 ? PixelPoint(u: best.u - 1, v: best.v)
                                  : PixelPoint(u: best.u, v: best.v - 1)
            let plus = axis == 0 ? PixelPoint(u: best.u + 1, v: best.v)
                                 : PixelPoint(u: best.u, v: best.v + 1)
            guard let cMinus = cost(minus, template: patch.template, luma: luma),
                  let cPlus = cost(plus, template: patch.template, luma: luma) else { continue }
            let denominator = cMinus - 2 * bestCost + cPlus
            guard abs(denominator) > 1e-9 else { continue }
            let delta = min(max(0.5 * (cMinus - cPlus) / denominator, -1), 1)
            if axis == 0 { best.u += delta } else { best.v += delta }
        }
        return best
    }

    private func project(_ world: (x: Double, y: Double, z: Double),
                         intrinsics: PinholeIntrinsics,
                         pose: CameraPose) -> PixelPoint? {
        let camera = pose.toCamera(world)
        let depth = -camera.z
        guard depth > 1e-6 else { return nil }
        return PixelPoint(u: intrinsics.fx * (camera.x / depth) + intrinsics.cx,
                          v: intrinsics.fy * (-camera.y / depth) + intrinsics.cy)
    }

    // MARK: - robust similarity fit

    private struct SimilarityFit {
        var transform = ResidualSimilarity.identity
        var rmse = Double.nan
        var inlierFraction = 0.0
    }

    /// Least-squares similarity with one round of outlier rejection.
    ///
    /// A patch that latched onto a moving duck, or onto the water despite the
    /// mask, leaves a residual unlike the rest.  One robust pass removes it, and
    /// the surviving fraction is reported -- a low inlier fraction is precisely
    /// the signal that the patches are on the water, the same diagnosis
    /// `StabilisationReport.warnings` makes on the desktop.
    private func fitSimilarity(from source: [PixelPoint],
                               to destination: [PixelPoint]) -> SimilarityFit {
        var lengths = source.indices.map {
            hypot(destination[$0].u - source[$0].u, destination[$0].v - source[$0].v)
        }
        lengths.sort()
        let threshold = max(2.5 * lengths[lengths.count / 2], 1.0)
        let indices = source.indices.filter {
            hypot(destination[$0].u - source[$0].u, destination[$0].v - source[$0].v) <= threshold
        }

        var fit = SimilarityFit()
        fit.inlierFraction = Double(indices.count) / Double(source.count)
        guard indices.count >= 3 else { return fit }

        var pBar = (u: 0.0, v: 0.0)
        var qBar = (u: 0.0, v: 0.0)
        for i in indices {
            pBar.u += source[i].u; pBar.v += source[i].v
            qBar.u += destination[i].u; qBar.v += destination[i].v
        }
        let n = Double(indices.count)
        pBar = (pBar.u / n, pBar.v / n)
        qBar = (qBar.u / n, qBar.v / n)

        var dot = 0.0, cross = 0.0, norm = 0.0
        for i in indices {
            let p = (u: source[i].u - pBar.u, v: source[i].v - pBar.v)
            let q = (u: destination[i].u - qBar.u, v: destination[i].v - qBar.v)
            dot += p.u * q.u + p.v * q.v
            cross += p.u * q.v - p.v * q.u
            norm += p.u * p.u + p.v * p.v
        }
        guard norm > 1e-9 else { return fit }

        var transform = ResidualSimilarity()
        transform.scaleCos = dot / norm
        transform.scaleSin = cross / norm
        transform.tx = qBar.u - (transform.scaleCos * pBar.u - transform.scaleSin * pBar.v)
        transform.ty = qBar.v - (transform.scaleSin * pBar.u + transform.scaleCos * pBar.v)
        fit.transform = transform

        var sumSquares = 0.0
        for i in indices {
            let mapped = transform.apply(source[i])
            sumSquares += pow(mapped.u - destination[i].u, 2)
                        + pow(mapped.v - destination[i].v, 2)
        }
        fit.rmse = (sumSquares / n).squareRoot()
        return fit
    }
}

/// Even-odd point-in-polygon, for the water outline.
func contains(polygon: [PixelPoint], point: PixelPoint) -> Bool {
    guard polygon.count >= 3 else { return false }
    var inside = false
    var j = polygon.count - 1
    for i in polygon.indices {
        let a = polygon[i], b = polygon[j]
        if (a.v > point.v) != (b.v > point.v) {
            let t = (point.v - a.v) / (b.v - a.v)
            if point.u < a.u + t * (b.u - a.u) { inside.toggle() }
        }
        j = i
    }
    return inside
}
