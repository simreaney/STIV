"""What the 2-D velocity methods -- LS-PIV and PTV -- hand to the analysis.

Both measure a velocity *vector* in world coordinates at each station, where
STIV measures only the component along its search line.  The analysis projects
the vector onto the flow direction it finally settles on, which is why the
estimate carries the vector and its individual samples rather than a speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Scale from the median absolute deviation to a standard deviation, for
#: normally distributed values.
MAD_TO_STD = 1.4826


@dataclass
class VectorEstimate:
    """One station's velocity from a 2-D method."""

    #: Velocity, world m/s, as ``(2,)``; NaN when the station gave none.
    velocity_world: np.ndarray
    #: The individual measurements it summarises -- one per correlated pair
    #: (LS-PIV) or per track (PTV) -- world m/s, ``(n, 2)``.
    samples_world: np.ndarray
    #: Pairs or tracks that went into it.
    n_used: int = 0
    #: LS-PIV: ratio of the correlation peak to the next highest; NaN for PTV.
    peak_ratio: float = math.nan
    #: Typical displacement between the two images of a measurement, in
    #: samples (LS-PIV) or image pixels (PTV): what range advice is based on.
    displacement_px: float = math.nan
    #: Fraction of the samples that fell outside the frame.
    out_of_frame_fraction: float = 0.0
    flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(np.all(np.isfinite(self.velocity_world)))


def robust_spread(values: np.ndarray) -> float:
    """Normal-consistent spread, ``1.4826 * MAD``; NaN for fewer than two values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return math.nan
    return float(MAD_TO_STD * np.median(np.abs(v - np.median(v))))


#: Least-squares design matrix of ``z = a + b x + c y + d x^2 + e x y + f y^2``
#: over the 3 x 3 neighbourhood, rows in ``plane[iy-1:iy+2, ix-1:ix+2]`` order.
_YY, _XX = np.mgrid[-1:2, -1:2]
_DESIGN_PINV = np.linalg.pinv(np.column_stack([
    np.ones(9), _XX.ravel(), _YY.ravel(), _XX.ravel() ** 2, (_XX * _YY).ravel(), _YY.ravel() ** 2,
]))


def subpixel_peak(plane: np.ndarray, iy: int, ix: int) -> tuple[float, float]:
    """Sub-sample offset ``(dy, dx)`` of the maximum of ``plane`` at ``(iy, ix)``.

    A 2-D Gaussian fitted to the 3 x 3 neighbourhood (the nine-point estimator
    of Nobach & Honkanen, 2005).  Fitting each axis on its own -- the usual
    three-point estimator -- is biased when the peak is tilted, as it is in a
    shear flow: the column through the integer maximum then misses the peak's
    centre, and the error shows up as a spurious cross-flow velocity.  Where the
    neighbourhood is not all positive, or the fit has no maximum within one
    sample, a three-point fit along each axis stands in, and a peak on the edge
    of the plane is returned as it stands.
    """
    h, w = plane.shape
    if 0 < iy < h - 1 and 0 < ix < w - 1:
        nb = plane[iy - 1:iy + 2, ix - 1:ix + 2]
        if np.all(nb > 0):
            _, b, c, d, e, f = _DESIGN_PINV @ np.log(nb).ravel()
            det = 4.0 * d * f - e * e
            if d < 0 and det > 0:
                dx = (e * c - 2.0 * f * b) / det
                dy = (e * b - 2.0 * d * c) / det
                if abs(dx) <= 1.0 and abs(dy) <= 1.0:
                    return float(dy), float(dx)
    return _fit_1d(plane[:, ix], iy), _fit_1d(plane[iy, :], ix)


def _fit_1d(line: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(line) - 1:
        return 0.0
    a, b, c = float(line[i - 1]), float(line[i]), float(line[i + 1])
    if a > 0 and b > 0 and c > 0:
        la, lb, lc = math.log(a), math.log(b), math.log(c)
        den = 2.0 * (la - 2.0 * lb + lc)
        if den < 0:
            return float(np.clip((la - lc) / den, -0.5, 0.5))
    den = a - 2.0 * b + c
    if den < 0:
        return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return 0.0
