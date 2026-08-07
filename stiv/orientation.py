"""Structure-tensor orientation analysis of a space-time image.

This is section 2.2 of Fujita, Watanabe & Tsubaki (2007).  A pattern that is
advected along the search line at speed ``U`` traces a straight streak in the
space-time image (STI); the streak's inclination ``phi`` from the time axis
carries the velocity, via Eq. (16), ``U = (Sx / St) tan phi``.

The orientation is recovered by maximising

.. math::  \\int w(x - x') \\, (\\nabla g(x')^T n)^2 \\, dx' = n^T J n     (Eq. 2)

over unit vectors ``n``, where ``J`` is the structure tensor of Eqs. (3), (4)
and (7)-(9).  Diagonalising ``J`` gives the orientation angle of Eq. (6),

.. math::  \\tan 2\\phi = 2 J_{xt} / (J_{tt} - J_{xx})

and the coherency of Eq. (11),

.. math::  C = \\sqrt{(J_{tt} - J_{xx})^2 + 4 J_{xt}^2} / (J_{xx} + J_{tt})
             = (\\lambda_1 - \\lambda_2) / (\\lambda_1 + \\lambda_2)

which is 1 for a perfectly oriented local pattern and 0 for an isotropic one.
Coherency is what makes the method robust: section 2.4 weights the angle
average by it, so clean streaks dominate and noisy patches are ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

# Antisymmetric central-difference stencils.  The paper's Eqs. (12) and (13)
# are printed with reversed signs; what matters for the structure tensor is
# that the *same* convention is used for d/dx and d/dt, since flipping only
# one of them would flip the sign of J_xt and hence of the velocity.  These
# are the standard forms, applied consistently to both axes.
_STENCILS: dict[int, np.ndarray] = {
    2: np.array([-1.0, 0.0, 1.0]) / 2.0,
    4: np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0,          # Eq. (12)
    6: np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / 60.0,  # Eq. (13)
}


@dataclass
class OrientationField:
    """Per-window orientation angles and coherencies over one STI.

    ``phi`` follows the paper's convention: measured from the time axis toward
    the space axis, positive for downstream motion along the search line.
    """

    phi: np.ndarray           # (nt, nx) radians, at window centres
    coherency: np.ndarray     # (nt, nx) in [0, 1]
    energy: np.ndarray        # (nt, nx) trace of J; gradient power
    centres_t: np.ndarray     # STI row index of each window centre
    centres_x: np.ndarray     # STI column index of each window centre

    @property
    def n_windows(self) -> int:
        return int(self.phi.size)


@dataclass
class AngleEstimate:
    """Result of the Eq. (14) weighted average over one STI."""

    phi: float                    # radians; nan if nothing usable
    tan_phi: float
    #: Coherency-weighted spread of in-range angles, radians.
    phi_std: float
    #: Fraction of windows falling inside [phi_min, phi_max].
    valid_fraction: float
    #: Mean coherency of the in-range windows.
    mean_coherency: float
    n_used: int
    #: Summed histogram of Fig. 11 and its bin centres, for diagnostics.
    hist: np.ndarray
    bin_centres: np.ndarray
    phi_min: float
    phi_max: float


# --------------------------------------------------------------------------- #
# gradients and the structure tensor
# --------------------------------------------------------------------------- #
def central_difference(img: np.ndarray, axis: int, order: int = 4) -> np.ndarray:
    """Central-difference derivative along ``axis`` with the given stencil order.

    Edges are handled by reflection; the affected margin is masked out later by
    :func:`structure_tensor_field`, which trims ``order // 2 + window // 2``
    pixels from every side.
    """
    try:
        k = _STENCILS[order]
    except KeyError:
        raise ValueError(f"gradient order must be one of {sorted(_STENCILS)}, got {order}")

    pad = len(k) // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (pad, pad)
    padded = np.pad(img.astype(np.float64, copy=False), pad_width, mode="reflect")

    out = np.zeros_like(img, dtype=np.float64)
    n = img.shape[axis]
    for i, c in enumerate(k):
        if c == 0.0:
            continue
        sl = [slice(None), slice(None)]
        sl[axis] = slice(i, i + n)
        out += c * padded[tuple(sl)]
    return out


def _window_filter(a: np.ndarray, window_px: int, window_type: str) -> np.ndarray:
    """Apply the window function ``w`` of Eq. (3) over the local area ``A``."""
    if window_type == "box":
        # The paper's rectangular integration area A (31 x 31 px by default).
        return uniform_filter(a, size=window_px, mode="nearest")
    if window_type == "gaussian":
        # Same nominal support, smoothly tapered: better-conditioned tensor.
        return gaussian_filter(a, sigma=window_px / 4.0, mode="nearest")
    raise ValueError(f"window_type must be 'box' or 'gaussian', got {window_type!r}")


def structure_tensor_field(
    sti: np.ndarray,
    window_px: int = 31,
    window_type: str = "box",
    overlap: float = 0.5,
    gradient_order: int = 4,
) -> OrientationField:
    """Compute the orientation angle and coherency over a grid of local windows.

    Parameters
    ----------
    sti
        Space-time image, ``(n_time, n_space)``: rows are frames, columns are
        positions along the search line.
    window_px
        Side length of the local integration area ``A`` of Eqs. (7)-(9).
    overlap
        Fractional overlap between neighbouring windows; 0.5 reproduces the
        50% overlap used in the paper.

    Notes
    -----
    The tensor components are evaluated everywhere by filtering ``gx*gx``,
    ``gx*gt`` and ``gt*gt``, then sampled on the window grid.  That is
    algebraically identical to stepping a window across the image, but far
    faster.
    """
    sti = np.asarray(sti, dtype=np.float64)
    if sti.ndim != 2:
        raise ValueError(f"STI must be 2-D (time, space), got shape {sti.shape}")

    nt, nx = sti.shape
    half = window_px // 2
    margin = half + len(_STENCILS[gradient_order]) // 2
    if nt <= 2 * margin + 1 or nx <= 2 * margin + 1:
        raise ValueError(
            f"STI of shape {sti.shape} is too small for a {window_px}px window "
            f"with an order-{gradient_order} gradient; use more frames, a longer "
            f"search line, or a smaller stiv.window_px"
        )

    # x is the space axis (columns, axis=1); t is the time axis (rows, axis=0).
    gx = central_difference(sti, axis=1, order=gradient_order)
    gt = central_difference(sti, axis=0, order=gradient_order)

    Jxx = _window_filter(gx * gx, window_px, window_type)   # Eq. (7)
    Jxt = _window_filter(gx * gt, window_px, window_type)   # Eq. (8)
    Jtt = _window_filter(gt * gt, window_px, window_type)   # Eq. (9)

    step = max(1, int(round(window_px * (1.0 - overlap))))
    rows = np.arange(margin, nt - margin, step)
    cols = np.arange(margin, nx - margin, step)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("no valid local windows fit inside the STI; reduce stiv.window_px")

    jxx = Jxx[np.ix_(rows, cols)]
    jxt = Jxt[np.ix_(rows, cols)]
    jtt = Jtt[np.ix_(rows, cols)]

    # Streak inclination, measured from the time axis toward the space axis.
    #
    # Mind the eigenvector branch here.  The vector n of Eqs. (1)-(2) is normal
    # to the lines of constant grey value, so maximising n^T J n picks the
    # *major* eigenvector -- the image gradient, which lies across the streaks.
    # Eq. (16), however, needs the inclination of the streaks themselves, i.e.
    # the *minor* eigenvector, 90 degrees away.  Eq. (6) is a statement about
    # tan 2*phi and so cannot distinguish the two branches; taking
    # 0.5*atan2(2 Jxt, Jtt - Jxx) literally returns the gradient direction and
    # yields a velocity of -(Sx/St)/tan(phi_true).  Rotating by 90 degrees --
    # equivalently negating both arguments' roles as below -- gives the streak
    # inclination that Eq. (16) expects.
    #
    # atan2 leaves phi in (-pi/2, pi/2], which spans every distinguishable
    # inclination: an orientation is only defined modulo 180 degrees, and phi
    # and phi + 180 share a tangent, so the sign of the velocity along the
    # search line is unambiguous.
    phi = 0.5 * np.arctan2(-2.0 * jxt, jxx - jtt)

    trace = jxx + jtt
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.sqrt((jtt - jxx) ** 2 + 4.0 * jxt**2) / trace   # Eq. (11)
    coh = np.where(trace > 1e-20, coh, 0.0)
    coh = np.clip(coh, 0.0, 1.0)

    return OrientationField(
        phi=phi, coherency=coh, energy=trace, centres_t=rows, centres_x=cols
    )


# --------------------------------------------------------------------------- #
# Eq. (14): coherency-weighted mean orientation angle
# --------------------------------------------------------------------------- #
def mean_orientation_angle(
    field: OrientationField,
    phi_min: float,
    phi_max: float,
    coherency_levels: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8),
    n_bins: int = 180,
    min_coherency: float = 0.0,
    estimator: str = "eq14",
) -> AngleEstimate:
    """Average the per-window angles into one angle for the STI.

    Implements Eq. (14).  For each coherency threshold ``C`` the histogram
    ``W(phi)`` of angles from windows with ``C_i >= C`` is accumulated over the
    search range ``[phi_min, phi_max]`` (Fig. 10), the histograms are summed
    (Fig. 11), and the weighted mean is taken.  Because a window with high
    coherency enters every threshold's histogram while a low-coherency one
    enters only the lowest, clean streaks are counted more often -- which is
    exactly the intent stated in the paper.

    ``phi_min``/``phi_max`` come from the user's velocity search range, and are
    the mechanism the paper uses to discard vectors corrupted by obstacles in
    the field of view (Fig. 12).

    Parameters
    ----------
    estimator
        ``"eq14"`` for the paper's weighted mean, or ``"peak"`` for the
        parabolically interpolated mode of the summed histogram, which is more
        robust when a strong secondary peak is present.
    """
    if phi_min >= phi_max:
        raise ValueError(f"need phi_min < phi_max, got {phi_min} and {phi_max}")

    phi = field.phi.ravel()
    coh = field.coherency.ravel()

    finite = np.isfinite(phi) & np.isfinite(coh)
    in_range = finite & (phi >= phi_min) & (phi <= phi_max) & (coh >= min_coherency)
    valid_fraction = float(in_range.sum()) / max(1, phi.size)

    edges = np.linspace(phi_min, phi_max, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(n_bins, dtype=np.float64)

    for level in coherency_levels:
        sel = in_range & (coh >= level)
        if not sel.any():
            continue
        h, _ = np.histogram(phi[sel], bins=edges)
        hist += h

    total = hist.sum()
    if total <= 0:
        return AngleEstimate(
            phi=math.nan, tan_phi=math.nan, phi_std=math.nan,
            valid_fraction=valid_fraction, mean_coherency=0.0, n_used=0,
            hist=hist, bin_centres=centres, phi_min=phi_min, phi_max=phi_max,
        )

    mean_phi = float((centres * hist).sum() / total)
    var = float((hist * (centres - mean_phi) ** 2).sum() / total)
    std_phi = math.sqrt(max(var, 0.0))

    if estimator == "peak":
        mean_phi = _parabolic_peak(centres, hist)
    elif estimator != "eq14":
        raise ValueError(f"estimator must be 'eq14' or 'peak', got {estimator!r}")

    return AngleEstimate(
        phi=mean_phi,
        tan_phi=float(math.tan(mean_phi)),
        phi_std=std_phi,
        valid_fraction=valid_fraction,
        mean_coherency=float(coh[in_range].mean()) if in_range.any() else 0.0,
        n_used=int(in_range.sum()),
        hist=hist,
        bin_centres=centres,
        phi_min=phi_min,
        phi_max=phi_max,
    )


def _parabolic_peak(centres: np.ndarray, hist: np.ndarray) -> float:
    """Sub-bin location of the histogram maximum by parabolic interpolation."""
    i = int(np.argmax(hist))
    if i == 0 or i == len(hist) - 1:
        return float(centres[i])
    y0, y1, y2 = hist[i - 1], hist[i], hist[i + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(centres[i])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    step = centres[1] - centres[0]
    return float(centres[i] + delta * step)


# --------------------------------------------------------------------------- #
# velocity conversion
# --------------------------------------------------------------------------- #
def velocity_from_angle(phi: float, sx_m_per_px: float, st_s_per_px: float) -> float:
    """Eq. (16): ``U = (Sx / St) tan phi``.

    ``sx_m_per_px`` is the length of one STI column in metres on the water
    surface; ``st_s_per_px`` is the duration of one STI row in seconds.
    """
    if not np.isfinite(phi):
        return math.nan
    return (sx_m_per_px / st_s_per_px) * math.tan(phi)


def angle_range_from_velocity(
    v_min: float, v_max: float, sx_m_per_px: float, st_s_per_px: float
) -> tuple[float, float]:
    """Invert Eq. (16) to get the ``phi_min``, ``phi_max`` of Eq. (14).

    The result is clipped just inside +/- 90 degrees, where ``tan`` diverges.
    """
    k = st_s_per_px / sx_m_per_px
    limit = 0.5 * math.pi - 1e-4
    lo = float(np.clip(math.atan(v_min * k), -limit, limit))
    hi = float(np.clip(math.atan(v_max * k), -limit, limit))
    return lo, hi
