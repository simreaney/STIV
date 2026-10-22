//  Uncertainty.swift
//  How much to trust the metre.
//
//  This has no counterpart in the Python package, and it exists because the
//  phone gets its scale differently from the drone.  A nadir drone shot has one
//  ground sample distance, known from altitude and optics, uniform across the
//  frame.  The phone instead intersects each pixel's ray with the water-surface
//  plane, and the plane is known from a LiDAR measurement taken *near the
//  camera* -- so scale at the far bank is extrapolated, and the extrapolation
//  gets worse with range.
//
//  Velocity is linear in `Sx` (Eq. 16), so a relative scale error is a relative
//  velocity error of the same size.  It therefore belongs in the reported
//  sigma, next to the angle spread, rather than in a footnote.
//
//  Geometry, once, so the constants below are not magic.  Put the camera at
//  height `h` above the plane, looking at a point at horizontal range `R`, with
//  focal length `f` in pixels.  The along-range ground sample distance is
//
//      dR/dv  ~=  R^2 / (f h)                                            (A)
//
//  -- quadratic in range, which is why a far bank at 30 m is coarse -- and the
//  cross-range one is `R / f`.  Three things perturb it:
//
//  1. **Plane height.**  An error `dh` scales every world distance by
//     `(1 + dh/h)`.  Uniform across the section, and cheap to make small: it is
//     just the LiDAR ranging error plus how well the user placed the waterline.
//
//         relative scale error  =  dh / h                                (B)
//
//  2. **Gravity misalignment.**  ARKit's world Y is gravity from the IMU, good
//     to a fraction of a degree but not exact.  A tilt `eps` moves the
//     intersection point at range `R` by about `eps R^2 / h`, i.e. a relative
//     position error of `eps R / h`; differentiating (A) doubles that for the
//     *scale*:
//
//         relative scale error  ~=  2 eps R / h                          (C)
//
//     This is the term that dominates far-field work.  It grows linearly in
//     `R/h`, and the mitigation is physical, not computational: raise the
//     camera.  A phone held at 1.5 m on a bank has `R/h = 20` at the far side
//     of a 30 m river; the same phone on a 6 m bridge parapet has `R/h = 5`.
//
//     (C) is the radial case.  A search line running cross-range is only linear
//     in `eps R / h`, so (C) is an upper bound -- which is the right side to err
//     on for a reported uncertainty.
//
//  3. **VIO scale drift.**  ARKit's metric scale comes from inertial
//     integration and is typically good to ~1%, worse when the scene offers
//     little parallax -- which open water conspicuously does.
//
//  The three are independent and combine in quadrature.

import Foundation

/// Inputs to the scale-error model.  The defaults are deliberately pessimistic;
/// `planeHeightSigmaM` should be overwritten with the actual water-plane fit
/// residual once it is known.
public struct ScaleSettings: Codable, Sendable, Equatable {
    /// 1-sigma uncertainty in the water-plane height, metres.  Comes from the
    /// LiDAR fit: the scatter of the depth samples the plane was fitted to,
    /// plus the sensor's own ranging error (a few centimetres inside 5 m).
    public var planeHeightSigmaM: Double = 0.03

    /// 1-sigma uncertainty in the gravity direction, degrees.  ARKit does not
    /// publish a figure; 0.3 degrees is a conservative stand-in for a device at
    /// rest, and the app degrades it when the recording was shaky.
    public var gravityTiltSigmaDegrees: Double = 0.3

    /// 1-sigma relative uncertainty of ARKit's metric scale.
    public var vioScaleSigma: Double = 0.01

    /// Value used instead when world tracking reported anything but `.normal`
    /// during the recording.
    public var degradedVioScaleSigma: Double = 0.05

    public init() {}

    private enum CodingKeys: String, CodingKey {
        case planeHeightSigmaM, gravityTiltSigmaDegrees, vioScaleSigma, degradedVioScaleSigma
    }

    // Missing keys take their defaults, so a survey saved before a field
    // existed still loads; synthesized decoding would reject the whole file.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        var d = ScaleSettings()
        d.planeHeightSigmaM = try c.decodeIfPresent(Double.self, forKey: .planeHeightSigmaM) ?? d.planeHeightSigmaM
        d.gravityTiltSigmaDegrees = try c.decodeIfPresent(Double.self, forKey: .gravityTiltSigmaDegrees) ?? d.gravityTiltSigmaDegrees
        d.vioScaleSigma = try c.decodeIfPresent(Double.self, forKey: .vioScaleSigma) ?? d.vioScaleSigma
        d.degradedVioScaleSigma = try c.decodeIfPresent(Double.self, forKey: .degradedVioScaleSigma) ?? d.degradedVioScaleSigma
        self = d
    }
}

/// Relative (fractional) 1-sigma uncertainty of `Sx` at one station.
///
/// - Parameters:
///   - rangeM: horizontal distance from the camera to the station.
///   - cameraHeightM: camera height above the water plane.
///   - trackingDegraded: whether world tracking was ever limited while recording.
public func relativeScaleUncertainty(
    rangeM: Double,
    cameraHeightM: Double,
    settings: ScaleSettings,
    trackingDegraded: Bool = false
) -> Double {
    // A camera at or below the water surface is not a geometry this model
    // describes; report total ignorance rather than a divide-by-zero.
    guard cameraHeightM > 0.05, rangeM.isFinite, rangeM >= 0 else { return .infinity }

    let height = settings.planeHeightSigmaM / cameraHeightM                    // (B)
    let epsilon = settings.gravityTiltSigmaDegrees * .pi / 180
    let tilt = 2.0 * epsilon * rangeM / cameraHeightM                          // (C)
    let vio = trackingDegraded ? settings.degradedVioScaleSigma : settings.vioScaleSigma

    return (height * height + tilt * tilt + vio * vio).squareRoot()
}

/// The advice that goes with a large uncertainty, phrased so it is actionable
/// on the bank rather than in a lab.
///
/// A non-finite `worstRelative` is the model saying it knows nothing -- the
/// camera at, or below, the fitted water plane -- and that deserves a warning
/// of its own rather than the silence an `isFinite` guard used to give it.
public func scaleAdvice(
    worstRelative: Double,
    atRangeM: Double,
    cameraHeightM: Double,
    threshold: Double
) -> String? {
    guard worstRelative.isFinite else {
        if !(cameraHeightM > 0.05) {
            return "the camera is only \(decimals(cameraHeightM, 2)) m above the fitted water "
                 + "plane -- at or below the surface, which the scale-uncertainty model cannot "
                 + "describe. Check the water-plane fit. Velocities are still reported, but their "
                 + "scale uncertainty is unknown."
        }
        return "the scale uncertainty could not be computed (check the scale settings); "
             + "velocities are still reported, but their scale uncertainty is unknown."
    }
    guard worstRelative > threshold else { return nil }
    let suggestedHeight = cameraHeightM * worstRelative / threshold
    return "scale at the far side of the section is uncertain to \(percent(worstRelative)) "
         + "(range \(decimals(atRangeM, 1)) m from a camera \(decimals(cameraHeightM, 2)) m "
         + "above the water). This term grows in proportion to range/height: raising the "
         + "camera to about \(decimals(suggestedHeight, 1)) m -- a bridge parapet, a pole, or "
         + "simply higher ground -- would bring it under \(percent(threshold)). Velocities "
         + "are still reported, with the uncertainty folded into sigma."
}
