"""Velocity-area discharge from a STIV surface-velocity profile.

STIV measures velocity *at the water surface*.  Turning that into discharge
takes two further ingredients, neither of which the imagery can supply:

1. **Depth**, from a survey, an ADCP transect or a stage-discharge relation.
2. A **velocity index** ``alpha = U_depth-averaged / U_surface``.  For a
   logarithmic profile in a wide channel this is close to 0.85, the value used
   almost universally in image-velocimetry gauging, though it drifts with
   relative roughness and wind.

The integration is the mid-section method of standard current-meter gauging:
each vertical owns a panel reaching halfway to its neighbours.  Each waterline
is itself a vertical, of zero depth, so the outermost station in the water
owns the ground from halfway to the waterline to halfway to its inner
neighbour, and the half-gap against the bank carries no area -- the depth
falling to zero there.  ``bank_coefficient`` then reduces the velocity of
those two outermost stations, for the velocity falling to zero as well.

"The banks" means the *waterline*, not the ends of the section as clicked.  A
section drawn bank-top to bank-top over a bed survey has stations on dry
ground; those own no panel, and neither does a station of zero depth beyond
the outermost wet ones.  Treating the waterline as a vertical is what keeps
the result continuous: a station sliding across the waterline, or an end
depth passing through zero, changes the area smoothly instead of handing the
next station's full depth a panel all the way to the bank.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .config import ConfigError, DischargeConfig

log = logging.getLogger(__name__)


@dataclass
class DischargeResult:
    """Discharge and the panel-by-panel breakdown behind it."""

    discharge_m3s: float
    area_m2: float
    mean_velocity_ms: float
    mean_surface_velocity_ms: float
    wetted_width_m: float
    max_depth_m: float
    alpha: float
    stations_m: np.ndarray
    depths_m: np.ndarray
    surface_velocity_ms: np.ndarray
    depth_averaged_ms: np.ndarray
    panel_width_m: np.ndarray
    panel_area_m2: np.ndarray
    panel_discharge_m3s: np.ndarray
    #: What the integration had to assume -- gaps it filled, stations it left
    #: out -- for the run's warnings.
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"discharge   {self.discharge_m3s:9.3f} m3/s\n"
            f"area        {self.area_m2:9.3f} m2\n"
            f"mean U      {self.mean_velocity_ms:9.3f} m/s (depth-averaged)\n"
            f"mean Us     {self.mean_surface_velocity_ms:9.3f} m/s (surface)\n"
            f"width       {self.wetted_width_m:9.3f} m\n"
            f"max depth   {self.max_depth_m:9.3f} m\n"
            f"alpha       {self.alpha:9.3f}"
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "station_m": self.stations_m,
                "depth_m": self.depths_m,
                "surface_velocity_ms": self.surface_velocity_ms,
                "depth_averaged_ms": self.depth_averaged_ms,
                "panel_width_m": self.panel_width_m,
                "panel_area_m2": self.panel_area_m2,
                "panel_discharge_m3s": self.panel_discharge_m3s,
            }
        )


# --------------------------------------------------------------------------- #
def load_bathymetry(
    path: str | Path, water_level_m: Optional[float] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read a bathymetry CSV into (distance_m, depth_m).

    Accepts either a ``depth_m`` column or a bed-elevation column together with
    ``discharge.water_level_m``.  Distance is measured along the cross-section
    from ``section.start_px``.

    Depths are returned *unclipped*: a negative depth is ground above the
    water.  That sign is what places the waterline between two survey points
    (see :func:`_wetted_edges`); clip it to zero first and every crossing
    snaps outwards to the last dry point.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"bathymetry file not found: {path}")
    # A row of empty cells, as spreadsheets often export at the end, is not a
    # survey point; a row with only some of its cells empty is refused below.
    df = pd.read_csv(path).dropna(how="all")
    cols = {c.lower().strip(): c for c in df.columns}

    dist_key = next((cols[k] for k in ("distance_m", "distance", "station_m", "station", "chainage_m") if k in cols), None)
    if dist_key is None:
        raise ConfigError(
            f"{path.name} needs a distance column (distance_m); found {list(df.columns)}"
        )

    depth_key = next((cols[k] for k in ("depth_m", "depth") if k in cols), None)
    if depth_key is not None:
        depth = df[depth_key].to_numpy(dtype=float)
    else:
        elev_key = next(
            (cols[k] for k in ("bed_elevation_m", "bed_elevation", "elevation_m", "elevation", "bed_m", "z_m") if k in cols),
            None,
        )
        if elev_key is None:
            raise ConfigError(
                f"{path.name} needs either a depth_m column or a bed-elevation "
                f"column; found {list(df.columns)}"
            )
        if water_level_m is None:
            raise ConfigError(
                f"{path.name} gives bed elevation, so discharge.water_level_m is required"
            )
        depth = float(water_level_m) - df[elev_key].to_numpy(dtype=float)

    dist = df[dist_key].to_numpy(dtype=float)
    # An empty cell reads as NaN, and fails far from here: a missing depth
    # makes Q NaN, and a missing distance sorts last and leaves every station
    # with no depth -- Q = 0, blamed on the bed.
    bad = ~(np.isfinite(dist) & np.isfinite(depth))
    if bad.any():
        row = int(df.index[np.argmax(bad)]) + 2      # 1-based, after the header row
        what = "depth" if depth_key is not None else "bed elevation"
        raise ConfigError(
            f"{path.name} row {row} has an empty or non-numeric cell; every row "
            f"needs a distance and a {what}"
        )
    order = np.argsort(dist)
    return dist[order], depth[order]


def _wetted_edges(
    dist: np.ndarray,
    depth: np.ndarray,
    fallback: tuple[float, float],
    outer: Optional[tuple[float, float]] = None,
) -> tuple[float, float]:
    """Locate the waterline on each side by linear interpolation to depth = 0.

    ``depth`` must be the *unclipped* profile, negative above the water: the
    crossing is interpolated between the last dry point and the first wet one.
    Where the profile is still wet at its end the waterline lies beyond it, and
    is put at ``outer`` (default: the end of the profile).  ``fallback`` is
    returned when nothing is wet at all.
    """
    dist = np.asarray(dist, dtype=float)
    depth = np.asarray(depth, dtype=float)
    wet = depth > 0
    if not wet.any():
        return fallback
    first, last = int(np.argmax(wet)), int(len(wet) - 1 - np.argmax(wet[::-1]))

    left = dist[first] if outer is None or first > 0 else min(outer[0], dist[first])
    if first > 0:
        d0, d1 = depth[first - 1], depth[first]
        if d1 > d0:
            left = dist[first - 1] + (dist[first] - dist[first - 1]) * (0.0 - d0) / (d1 - d0)

    right = dist[last] if outer is None or last < len(dist) - 1 else max(outer[1], dist[last])
    if last < len(dist) - 1:
        d0, d1 = depth[last], depth[last + 1]
        if d0 > d1:
            right = dist[last] + (dist[last + 1] - dist[last]) * (d0 - 0.0) / (d0 - d1)

    return float(left), float(right)


def mid_section_widths(
    stations: np.ndarray, left_edge: float, right_edge: float
) -> np.ndarray:
    """Panel widths for the mid-section method.

    Each vertical owns the ground from halfway to its left neighbour to halfway
    to its right neighbour; the outermost verticals reach to the waterline.
    ``stations`` should all lie between the two edges: see
    :func:`compute_discharge` for how stations on dry ground are left out.
    """
    b = np.asarray(stations, dtype=float)
    n = len(b)
    if n == 1:
        return np.array([max(right_edge - left_edge, 0.0)])

    mid = 0.5 * (b[:-1] + b[1:])
    lo = np.concatenate([[left_edge], mid])
    hi = np.concatenate([mid, [right_edge]])
    return np.maximum(hi - lo, 0.0)


#: Stations within this distance (m) outside a waterline still count as being
#: at it -- a station placed exactly on the edge must not be lost to rounding.
_EDGE_TOLERANCE_M = 1e-6


def compute_discharge(
    cfg: DischargeConfig,
    stations_m: Sequence[float],
    surface_velocity_ms: Sequence[float],
    section_width_m: float,
) -> DischargeResult:
    """Integrate a surface-velocity profile into a discharge.

    Only the stations from the outermost one with a positive depth to the
    other own a panel (an island inside that span keeps its zero-depth
    vertical); every other station gets a zero-width, zero-depth panel and so
    contributes nothing, its row kept so the breakdown still lines up with the
    stations.  The waterlines act as zero-depth verticals, so the panel widths
    sum to the wetted width less the half-gap between each waterline and its
    nearest station.  The bank coefficient goes on the two outermost stations
    with a panel.  A station with a panel but no velocity (one that failed
    quality control and was not interpolated) is filled by interpolating its
    neighbours, and the fill is reported in ``warnings``.

    Raises ``ConfigError`` when no station in the water has a velocity.
    """
    b = np.asarray(stations_m, dtype=float)
    us = np.asarray(surface_velocity_ms, dtype=float)
    if len(b) != len(us):
        raise ConfigError("stations and velocities must have the same length")
    notes: list[str] = []

    # --- depths and waterline -------------------------------------------- #
    # Depths are interpolated *unclipped* and only then clipped, and the
    # waterline is found on the unclipped profile.  Clipping first loses the
    # sign change that places the waterline between two points, so the edge
    # snapped out to the last dry point and dry ground gained a positive depth.
    if cfg.depths_m is not None:
        raw = np.asarray(cfg.depths_m, dtype=float)
        if len(raw) != len(b):
            raise ConfigError(
                f"discharge.depths_m has {len(raw)} entries but there are "
                f"{len(b)} stations; use bathymetry_csv to interpolate instead"
            )
        # The section is drawn waterline to waterline unless the depths say
        # otherwise, so a profile wet at its ends reaches the section ends.
        left, right = _wetted_edges(
            b, raw, (0.0, float(section_width_m)), outer=(0.0, float(section_width_m))
        )
    else:
        bd, bz = load_bathymetry(cfg.bathymetry_csv, cfg.water_level_m)
        if b.min() < bd.min() - 1e-6 or b.max() > bd.max() + 1e-6:
            msg = (
                f"stations span {b.min():.2f}-{b.max():.2f} m but the bathymetry "
                f"covers only {bd.min():.2f}-{bd.max():.2f} m; stations beyond the "
                f"survey are left out of the discharge"
            )
            log.warning(msg)
            notes.append(msg)
        raw = np.interp(b, bd, bz)
        left, right = _wetted_edges(bd, bz, (0.0, float(section_width_m)))

    inside = (b >= left - _EDGE_TOLERANCE_M) & (b <= right + _EDGE_TOLERANCE_M)
    depth = np.where(inside, np.maximum(raw, 0.0), 0.0)
    # The stations that own a panel: from the outermost with a positive depth
    # to the other.  A zero-depth station at the waterline is the waterline
    # vertical itself, and must not take the bank panel from its neighbour.
    positive = np.flatnonzero(depth > 0)
    wet = np.zeros(len(b), bool)
    if positive.size:
        wet[positive[0] : positive[-1] + 1] = True
    n_dry = int((~wet).sum())
    if n_dry:
        log.info(
            "%d station(s) lie outside the wetted width %.2f-%.2f m or have no "
            "depth there, and are left out of the discharge", n_dry, left, right,
        )
    if not wet.any():
        notes.append("no station has a positive depth, so the discharge is zero")
        log.warning(notes[-1])

    # --- velocities ------------------------------------------------------ #
    good = np.isfinite(us)
    if wet.any() and not (good & wet).any():
        raise ConfigError("no finite surface velocities in the wetted width; cannot compute discharge")
    if not good.any():
        raise ConfigError("no finite surface velocities; cannot compute discharge")
    if not good.all():
        gaps = ~good & wet & (depth > 0)
        if gaps.any():
            msg = (
                f"{int(gaps.sum())} station(s) in the water had no velocity "
                f"(at {', '.join(f'{x:.2f}' for x in b[gaps][:6])}"
                f"{'...' if gaps.sum() > 6 else ''} m); the discharge fills them by "
                f"interpolating their neighbours"
            )
            log.warning(msg)
            notes.append(msg)
        us = np.interp(b, b[good], us[good])

    ud = cfg.alpha * us                                  # velocity-index conversion

    # The bank coefficient is a *velocity* reduction -- the near-bank vertical
    # stands in for water that slows to zero at the waterline.  Applying it to
    # the panel width instead gives an identical discharge, since Q is linear
    # in both, but then understates the reported cross-sectional area and so
    # overstates Q/A.  Area is a number people check against an ADCP, so it is
    # kept honest here.  It goes on the outermost stations with a depth: on a
    # dry or zero-depth one it would multiply a zero-area panel and change
    # nothing at all.  Area is continuous as a station crosses the waterline;
    # Q has one small step, when a station gains any depth at all and so takes
    # the coefficient from its neighbour.  That is inherent in giving the
    # coefficient to a single vertical, and the step is at most (1 - c) times
    # that neighbour's share of Q.
    idx = np.flatnonzero(wet)
    if len(idx) > 1:
        ud = ud.copy()
        ud[idx[0]] *= cfg.bank_coefficient
        ud[idx[-1]] *= cfg.bank_coefficient

    # --- integration ----------------------------------------------------- #
    # Each waterline is a vertical of zero depth, owning the ground from the
    # waterline to halfway to the nearest station; so the outermost panels
    # start halfway out, not at the waterline itself.
    width = np.zeros(len(b))
    if len(idx):
        lo = 0.5 * (left + b[idx[0]])
        hi = 0.5 * (b[idx[-1]] + right)
        width[idx] = mid_section_widths(b[idx], lo, hi)

    area = width * depth
    q = area * ud

    total_area = float(area.sum())
    total_q = float(q.sum())

    return DischargeResult(
        discharge_m3s=total_q,
        area_m2=total_area,
        mean_velocity_ms=total_q / total_area if total_area > 0 else float("nan"),
        mean_surface_velocity_ms=float(np.average(us, weights=area)) if total_area > 0 else float(np.mean(us)),
        wetted_width_m=float(right - left),
        max_depth_m=float(depth.max()) if depth.size else 0.0,
        alpha=float(cfg.alpha),
        stations_m=b,
        depths_m=depth,
        surface_velocity_ms=us,
        depth_averaged_ms=ud,
        panel_width_m=width,
        panel_area_m2=area,
        panel_discharge_m3s=q,
        warnings=notes,
    )
