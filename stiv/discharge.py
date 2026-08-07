"""Velocity-area discharge from a STIV surface-velocity profile.

STIV measures velocity *at the water surface*.  Turning that into discharge
takes two further ingredients, neither of which the imagery can supply:

1. **Depth**, from a survey, an ADCP transect or a stage-discharge relation.
2. A **velocity index** ``alpha = U_depth-averaged / U_surface``.  For a
   logarithmic profile in a wide channel this is close to 0.85, the value used
   almost universally in image-velocimetry gauging, though it drifts with
   relative roughness and wind.

The integration is the mid-section method of standard current-meter gauging:
each vertical owns a panel reaching halfway to its neighbours, and the two
panels against the banks are reduced by ``bank_coefficient`` to account for
velocity and depth falling to zero at the waterline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"bathymetry file not found: {path}")
    df = pd.read_csv(path)
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
    order = np.argsort(dist)
    return dist[order], np.maximum(depth[order], 0.0)


def _wetted_edges(dist: np.ndarray, depth: np.ndarray, fallback: tuple[float, float]) -> tuple[float, float]:
    """Locate the waterline on each side by linear interpolation to depth = 0."""
    wet = depth > 0
    if not wet.any():
        return fallback
    first, last = int(np.argmax(wet)), int(len(wet) - 1 - np.argmax(wet[::-1]))

    left = dist[first]
    if first > 0:
        d0, d1 = depth[first - 1], depth[first]
        if d1 > d0:
            left = dist[first - 1] + (dist[first] - dist[first - 1]) * (0.0 - d0) / (d1 - d0)

    right = dist[last]
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
    """
    b = np.asarray(stations, dtype=float)
    n = len(b)
    if n == 1:
        return np.array([max(right_edge - left_edge, 0.0)])

    mid = 0.5 * (b[:-1] + b[1:])
    lo = np.concatenate([[left_edge], mid])
    hi = np.concatenate([mid, [right_edge]])
    return np.maximum(hi - lo, 0.0)


def compute_discharge(
    cfg: DischargeConfig,
    stations_m: Sequence[float],
    surface_velocity_ms: Sequence[float],
    section_width_m: float,
) -> DischargeResult:
    """Integrate a surface-velocity profile into a discharge."""
    b = np.asarray(stations_m, dtype=float)
    us = np.asarray(surface_velocity_ms, dtype=float)
    if len(b) != len(us):
        raise ConfigError("stations and velocities must have the same length")

    # --- depths ---------------------------------------------------------- #
    if cfg.depths_m is not None:
        depth = np.asarray(cfg.depths_m, dtype=float)
        if len(depth) != len(b):
            raise ConfigError(
                f"discharge.depths_m has {len(depth)} entries but there are "
                f"{len(b)} stations; use bathymetry_csv to interpolate instead"
            )
        left, right = 0.0, float(section_width_m)
        depth = np.maximum(depth, 0.0)
    else:
        bd, bz = load_bathymetry(cfg.bathymetry_csv, cfg.water_level_m)
        if b.min() < bd.min() - 1e-6 or b.max() > bd.max() + 1e-6:
            log.warning(
                "stations span %.2f-%.2f m but the bathymetry covers %.2f-%.2f m; "
                "depths outside that range are held at the end values",
                b.min(), b.max(), bd.min(), bd.max(),
            )
        depth = np.maximum(np.interp(b, bd, bz), 0.0)
        left, right = _wetted_edges(bd, bz, (0.0, float(section_width_m)))

    # --- velocities ------------------------------------------------------ #
    if not np.isfinite(us).any():
        raise ConfigError("no finite surface velocities; cannot compute discharge")
    if not np.isfinite(us).all():
        good = np.isfinite(us)
        us = np.interp(b, b[good], us[good])

    ud = cfg.alpha * us                                  # velocity-index conversion

    # --- integration ----------------------------------------------------- #
    width = mid_section_widths(b, left, right)
    if len(width) > 1:
        width = width.copy()
        width[0] *= cfg.bank_coefficient
        width[-1] *= cfg.bank_coefficient

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
    )
