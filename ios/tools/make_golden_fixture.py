#!/usr/bin/env python3
"""Generate the golden fixture that pins the Swift port to the Python original.

The iOS port of ``stiv.orientation`` has to agree with this package number for
number, not merely "look about right": the structure tensor can be wrong in
ways that still produce a plausible velocity (an inverted eigenvector branch, a
flipped gradient sign, a transposed axis, a boundary rule off by one pixel).
A synthetic-slope test catches the gross errors; only a direct comparison
catches the subtle ones.

Run it from the repository root, against the repository's own package -- not
an installed ``stiv``, which may be a stale copy:

    PYTHONPATH=$PWD python3 -I ios/tools/make_golden_fixture.py [--out DIR]

It writes, into ios/StivKit/Tests/StivKitTests/Fixtures/ (or DIR):

    golden_sti.f32     the space-time image, raw little-endian float32,
                       row-major, (rows x cols)
    golden.json        the analysis parameters, what stiv.sti.preprocess makes
                       of that image, and everything stiv.orientation returns:
                       the box-window field and its Eq. (14) and histogram-peak
                       estimates, preprocess on a copy of the image with
                       missing (NaN) samples, the same field with the
                       Gaussian window, the window grids of
                       sizes whose step lands on a rounding half, an STI
                       with featureless windows Eq. (14) must ignore, and the
                       analysis stage -- quality control, the profile median
                       filter, interpolation, the direction check, the
                       frame-step advice and the velocity sigma

The Swift test ``GoldenFixtureTests`` skips itself when these are absent, so the
suite still runs on a machine that has never seen Python; each newer section of
the fixture is likewise optional, and its tests skip until the fixture is
regenerated.  ``STIV_GOLDEN_FIXTURE_DIR`` points the Swift tests at another
directory, so a fixture written with ``--out`` can be tried before it replaces
the committed one.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from stiv import analysis, orientation  # noqa: E402
from stiv.config import Config, StiConfig  # noqa: E402
from stiv.orientation import (  # noqa: E402
    AngleEstimate,
    mean_orientation_angle,
    structure_tensor_field,
)
from stiv.sti import preprocess  # noqa: E402

OUT = REPO / "ios" / "StivKit" / "Tests" / "StivKitTests" / "Fixtures"

# Deliberately awkward: an even-ish size that is not a multiple of the window
# step, so the window grid does not land tidily and any off-by-one in the margin
# arithmetic shows up as a shape mismatch rather than a rounding difference.
ROWS, COLS = 187, 143
TAN_PHI = 1.35
SIGMA = 2.5
NOISE = 0.25
SEED = 20240917

WINDOW_PX = 31
WINDOW_TYPE = "box"
OVERLAP = 0.5
GRADIENT_ORDER = 4
PHI_MIN, PHI_MAX = -0.4, 1.3
COHERENCY_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
N_BINS = 180
MIN_COHERENCY = 0.05

# Window centres sampled individually, so a disagreement can be localised
# instead of merely reported as "the means differ".
PROBES = [(0, 0), (1, 3), (2, 1), (3, 2)]

# STI conditioning is compared too.  It runs *before* the structure tensor, so
# a divergence here moves every angle that follows.  What this pins, in order of
# how easily each is got wrong: the stage order (normalising after smoothing
# rather than before shifts a probe by 0.13), the smoothing boundary rule
# ("nearest" vs "reflect", 0.027), the Gaussian truncation radius (0.00017) and
# sigma itself.  The two mean subtractions are *not* pinned, because double
# centring commutes -- either order gives X - colmean - rowmean + grandmean.
# Cells are sampled at the corners and edges, where a boundary rule differs.
STI_SETTINGS = dict(
    remove_static_background=True,
    normalise=True,
    clahe=False,          # desktop-only; the Swift port has no CLAHE
    smooth_sigma=1.5,
)
STI_PROBES = [
    (0, 0), (0, COLS - 1), (ROWS - 1, 0), (ROWS - 1, COLS - 1),
    (0, COLS // 2), (ROWS // 2, 0), (ROWS // 2, COLS // 2), (137, 29),
]


# The Gaussian window, on the same STI.  Its support is twice the box's, and it
# is the path the Swift port evaluates only at the window centres, so it is
# pinned separately.
GAUSSIAN_WINDOW_PX = 31

# Window sizes whose grid step at 50% overlap lands exactly on a half
# (29 * 0.5 = 14.5, 21 * 0.5 = 10.5).  Python's round() takes halves to even;
# a port that rounds them up gets a different grid.
GRID_CASES = [(29, "box"), (21, "box"), (29, "gaussian"), (21, "gaussian")]

# Samples that left the frame.  The STI with NaN at a few scattered cells and
# down one whole column, conditioned twice: with every other stage off, so the
# output *is* the fill (a scattered NaN takes its column's mean, the all-NaN
# column the grand mean), and with STI_SETTINGS, the pipeline as run.  The first
# is what tells column-mean from grand-mean filling: with the static background
# removed, an all-NaN column comes out zero whatever it was filled with.
MISSING_CELLS = [(0, 0), (5, 17), (5, 18), (93, 70), (ROWS - 1, COLS - 1), (140, 3)]
MISSING_COLUMN = 101
MISSING_FILL_ONLY = dict(
    remove_static_background=False,
    normalise=False,
    clahe=False,
    smooth_sigma=0.0,
)
MISSING_PROBES = MISSING_CELLS + [(0, MISSING_COLUMN), (ROWS // 2, MISSING_COLUMN), (7, 9)]

# Featureless windows.  Columns [0, FLAT_COLS) of the STI are held at a
# constant, so the windows whose support lies inside that block have exactly
# zero gradient energy.  Eq. (6) still gives them phi = 0, and with
# min_coherency 0 the lowest coherency level would admit them as votes for zero
# velocity; stiv.orientation.flat_windows is what keeps them out.
FLAT_COLS = 72
FLAT_FILL = 0.0
FLAT_MIN_COHERENCY = 0.0

# Analysis stage.  Each station: (station_m, velocity_ms, velocity_std_ms,
# tan_phi, coherency, valid_fraction, flags already set).  Chosen so that every
# quality-control branch fires at least once, the finite velocities have an
# even count (np.median averages the middle two), failed stations sit at both
# ends of the profile (np.interp clamps there), and the median filter meets a
# NaN.
NAN = float("nan")
QC_STATIONS = [
    (0.0, NAN, NAN, NAN, 0.0, 0.0, ["no_orientation"]),
    (1.0, 0.9, 0.05, 0.9, 0.6, 0.8, []),
    (2.0, 1.1, 0.05, 1.1, 0.05, 0.8, []),     # low_coherency
    (3.0, 1.4, 0.9, 1.4, 0.6, 0.8, []),       # high_uncertainty
    (4.0, 1.2, 0.1, 9.0, 0.6, 0.8, []),       # tan_phi_high
    (5.0, 0.2, 0.3, 0.2, 0.6, 0.8, []),       # judged against 0.25 * median
    (6.0, 1.0, 0.05, 1.0, 0.6, 0.1, []),      # low_valid_fraction
    (7.0, 0.03, 0.01, 0.03, 0.6, 0.8, []),    # tan_phi_low
    (8.0, 0.95, 0.05, 0.95, 0.6, 0.8, []),
    (9.0, NAN, NAN, NAN, 0.0, 0.0, ["sti_too_small"]),
]
QC_MEDIAN_FILTER = 3

# Frame-step advice: (frame_step, |tan phi| per station).  The second has an
# even count whose np.median (0.195) and upper median (0.21) fall either side
# of the 0.2 threshold; the third puts step / median exactly on a half.
RANGE_ADVICE_CASES = [
    (2, [0.05, 0.06, 0.08]),
    (1, [0.1, 0.18, 0.21, 0.3]),
    (25, [10.0, 10.0, 10.0]),
    (4, [5.0, 6.0, 7.0, 6.5]),
    (2, [0.5, 1.0, 2.0]),
    (1, [5.0, 6.0, 7.0]),     # steep at step 1: no smaller step exists
]

# Velocity sigma: (phi, phi_std, sx, st, negate).  The last is close enough to
# 90 degrees that the cos^2 floor of 1e-6 engages.
VELOCITY_STD_CASES = [
    (0.6, 0.05, 0.02, 1 / 30, False),
    (-0.9, 0.12, 0.015, 1 / 60, False),
    (0.6, 0.05, 0.02, 1 / 30, True),
    (1.5707, 0.01, 0.02, 1 / 30, False),
]

# Direction check: the fixture STI and its mirror image, i.e. the same flow
# seen by a search line pointing upstream.  Two of three upstream makes the
# median negative.
DIRECTION_MIRRORED = [True, True, False]
DIRECTION_SX = 0.02
DIRECTION_ST = 1 / 30


def num(v):
    """A float for JSON, with NaN and infinity as null (JSON has neither)."""
    v = float(v)
    return v if math.isfinite(v) else None


def field_summary(field) -> dict:
    return {
        "shape": list(field.phi.shape),
        "centres_t": field.centres_t.tolist(),
        "centres_x": field.centres_x.tolist(),
        "phi_mean": float(field.phi.mean()),
        "phi_min": float(field.phi.min()),
        "phi_max": float(field.phi.max()),
        "coherency_mean": float(field.coherency.mean()),
        "coherency_min": float(field.coherency.min()),
        "coherency_max": float(field.coherency.max()),
        "energy_mean": float(field.energy.mean()),
        "probes": [
            {
                "i": i,
                "j": j,
                "phi": float(field.phi[i, j]),
                "coherency": float(field.coherency[i, j]),
                "energy": float(field.energy[i, j]),
            }
            for i, j in PROBES
            if i < field.phi.shape[0] and j < field.phi.shape[1]
        ],
    }


def estimate_summary(est) -> dict:
    return {
        "phi": float(est.phi),
        "tan_phi": float(est.tan_phi),
        "phi_std": float(est.phi_std),
        "valid_fraction": float(est.valid_fraction),
        "mean_coherency": float(est.mean_coherency),
        "n_used": int(est.n_used),
        "hist": [float(v) for v in est.hist],
        "bin_centres": [float(v) for v in est.bin_centres],
    }


# The angle spread was renamed velocity_spread_ms, with velocity_std_ms kept as
# a property; construct with whichever the installed package declares.
_SPREAD_FIELD = (
    "velocity_spread_ms"
    if "velocity_spread_ms" in {f.name for f in dataclasses.fields(analysis.StationResult)}
    else "velocity_std_ms"
)


def make_station(i, station_m, v, std, tan, coh, vf, flags, sx=0.01, st=1 / 30):
    return analysis.StationResult(
        index=i, station_m=station_m, x_world=0.0, y_world=0.0, u_px=0.0, v_px=0.0,
        phi_rad=math.atan(tan) if math.isfinite(tan) else NAN, tan_phi=tan,
        velocity_ms=v, velocity_raw_ms=v, coherency=coh, valid_fraction=vf,
        n_windows=10, phi_std_rad=0.0, sx_m_per_px=sx, st_s_per_px=st,
        flags=list(flags), **{_SPREAD_FIELD: std},
    )


def station_out(s) -> dict:
    return {
        "velocity_ms": num(s.velocity_ms),
        "velocity_raw_ms": num(s.velocity_raw_ms),
        "velocity_std_ms": num(s.velocity_std_ms),
        "phi_rad": num(s.phi_rad),
        "tan_phi": num(s.tan_phi),
        "coherency": num(s.coherency),
        "valid_fraction": num(s.valid_fraction),
        "n_windows": int(s.n_windows),
        "flags": list(s.flags),
    }


def quality_settings(cfg) -> dict:
    q = cfg.quality
    return {
        "min_valid_fraction": q.min_valid_fraction,
        "min_mean_coherency": q.min_mean_coherency,
        "max_relative_uncertainty": q.max_relative_uncertainty,
        "tan_phi_range": list(q.tan_phi_range),
        "interpolate_flagged": q.interpolate_flagged,
        "median_filter_stations": q.median_filter_stations,
    }


def analysis_stage(sti: np.ndarray) -> dict:
    """What stiv.analysis does between the Eq. (14) estimates and the report."""
    out: dict = {}

    # Quality control, the median filter and interpolation, together and in
    # the pipeline's order.
    cfg = Config()
    cfg.quality.median_filter_stations = QC_MEDIAN_FILTER
    stations = [make_station(i, *row) for i, row in enumerate(QC_STATIONS)]
    stations_in = [station_out(s) for s in stations]
    warnings: list[str] = []
    analysis._quality_control(cfg, stations, warnings)
    out["quality_control"] = {
        "quality": quality_settings(cfg),
        "stations_in": [dict(d, station_m=row[0]) for d, row in zip(stations_in, QC_STATIONS)],
        "stations_out": [station_out(s) for s in stations],
        "n_warnings": len(warnings),
    }

    # Frame-step advice.  The Python only warns; the suggested step is read
    # back out of the warning.
    cases = []
    for step, tans in RANGE_ADVICE_CASES:
        cfg = Config()
        # Sx = St = 1, so tan phi and the velocity coincide, whichever of the
        # two the advice reads.
        stations = [make_station(i, float(i), t, 0.0, t, 0.6, 0.8, [], sx=1.0, st=1.0)
                    for i, t in enumerate(tans)]
        warnings = []
        analysis._range_advice(cfg, stations, SimpleNamespace(step=step), warnings)
        suggested = None
        for w in warnings:
            m = re.search(r"frame_step: (\d+)", w)
            if m:
                suggested = int(m.group(1))
        cases.append({
            "frame_step": step,
            "tan_phi": tans,
            "median": float(np.median(np.abs(tans))),
            "suggested": suggested,
            "warned": bool(warnings),
        })
    out["range_advice"] = cases

    # Eq. (16) error propagation.
    cases = []
    for phi, phi_std, sx, st, negate in VELOCITY_STD_CASES:
        station =make_station(0, 0.0, NAN, NAN, NAN, 0.0, 0.0, [], sx=sx, st=st)
        est = AngleEstimate(
            phi=phi, tan_phi=math.tan(phi), phi_std=phi_std, valid_fraction=0.9,
            mean_coherency=0.7, n_used=12, hist=np.zeros(1), bin_centres=np.zeros(1),
            phi_min=-1.0, phi_max=1.0,
        )
        analysis._apply_estimate(station, SimpleNamespace(sx_m_per_px=sx, st_s_per_px=st),
                                 est, negate=negate)
        cases.append({
            "phi": phi, "phi_std": phi_std, "sx": sx, "st": st, "negate": negate,
            "velocity_ms": num(station.velocity_ms),
            "velocity_std_ms": num(station.velocity_std_ms),
        })
    out["velocity_std"] = cases

    # The direction check, then quality control on what it leaves -- the
    # order of stiv.analysis.run.
    direction = {
        "sti_settings": STI_SETTINGS,
        "window_px": WINDOW_PX,
        "window_type": WINDOW_TYPE,
        "overlap": OVERLAP,
        "gradient_order": GRADIENT_ORDER,
        "sx": DIRECTION_SX,
        "st": DIRECTION_ST,
        "mirrored": DIRECTION_MIRRORED,
        "runs": [],
    }
    for auto_flip in (True, False):
        cfg = Config()
        for key, value in STI_SETTINGS.items():
            setattr(cfg.stiv, key, value)
        cfg.stiv.window_px = WINDOW_PX
        cfg.stiv.window_type = WINDOW_TYPE
        cfg.stiv.window_overlap = OVERLAP
        cfg.stiv.gradient_order = GRADIENT_ORDER
        cfg.flow.auto_flip = auto_flip
        v_lo, v_hi = cfg.stiv.velocity_range_ms

        stations, stis, fields, estimates = [], [], [], []
        for i, mirrored in enumerate(DIRECTION_MIRRORED):
            data = sti[:, ::-1] if mirrored else sti
            s = SimpleNamespace(
                data=preprocess(np.ascontiguousarray(data, dtype=np.float64), cfg.stiv),
                sx_m_per_px=DIRECTION_SX, st_s_per_px=DIRECTION_ST,
            )
            station = make_station(i, float(i), NAN, NAN, NAN, 0.0, 0.0, [],
                                   sx=DIRECTION_SX, st=DIRECTION_ST)
            field = structure_tensor_field(
                s.data, window_px=WINDOW_PX, window_type=WINDOW_TYPE,
                overlap=OVERLAP, gradient_order=GRADIENT_ORDER,
            )
            est = analysis._estimate_angle(cfg, s, field, v_lo, v_hi)
            analysis._apply_estimate(station, s, est)
            stations.append(station)
            stis.append(s)
            fields.append(field)
            estimates.append(est)

        warnings = []
        analysis._direction_check(cfg, stations, stis, fields, estimates, warnings)
        analysis._quality_control(cfg, stations, warnings)
        direction["runs"].append({
            "auto_flip": auto_flip,
            "quality": quality_settings(cfg),
            "stations_out": [station_out(s) for s in stations],
            "opposite_warning": any("opposite to the assumed" in w for w in warnings),
            "n_warnings": len(warnings),
        })
    out["direction"] = direction
    return out


def synthetic_sti() -> np.ndarray:
    """A broadband pattern advected at exactly TAN_PHI columns per row."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(SEED)
    length = COLS + int(abs(TAN_PHI)) * ROWS + 8 * int(SIGMA) + 10
    base = gaussian_filter(rng.standard_normal(length), SIGMA)
    t = np.arange(ROWS)[:, None]
    x = np.arange(COLS)[None, :]
    idx = np.mod(x - TAN_PHI * t, length)
    i0 = np.floor(idx).astype(int)
    f = idx - i0
    sti = base[i0] * (1 - f) + base[(i0 + 1) % length] * f
    sti = sti + NOISE * sti.std() * rng.standard_normal(sti.shape)
    return np.ascontiguousarray(sti, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT,
                        help="directory to write into (default: the committed fixtures)")
    out_dir = parser.parse_args().out
    out_dir.mkdir(parents=True, exist_ok=True)
    sti = synthetic_sti()

    # The Swift side reads float32 and widens to Double, so hand the analysis
    # the same float32-rounded values rather than the float64 originals.
    field = structure_tensor_field(
        sti.astype(np.float64),
        window_px=WINDOW_PX,
        window_type=WINDOW_TYPE,
        overlap=OVERLAP,
        gradient_order=GRADIENT_ORDER,
    )
    est = mean_orientation_angle(
        field,
        PHI_MIN,
        PHI_MAX,
        coherency_levels=COHERENCY_LEVELS,
        n_bins=N_BINS,
        min_coherency=MIN_COHERENCY,
    )

    conditioned = preprocess(sti.astype(np.float64), StiConfig(**STI_SETTINGS))

    # The histogram-peak estimator, on the same field: its parabolic sub-bin
    # step is a sign away from moving phi by up to two bins.
    peak_est = mean_orientation_angle(
        field,
        PHI_MIN,
        PHI_MAX,
        coherency_levels=COHERENCY_LEVELS,
        n_bins=N_BINS,
        min_coherency=MIN_COHERENCY,
        estimator="peak",
    )

    missing_sti = sti.astype(np.float64)
    for r, c in MISSING_CELLS:
        missing_sti[r, c] = np.nan
    missing_sti[:, MISSING_COLUMN] = np.nan
    missing_runs = []
    for settings in (MISSING_FILL_ONLY, STI_SETTINGS):
        filled = preprocess(missing_sti, StiConfig(**settings))
        missing_runs.append({
            "settings": settings,
            "mean": float(filled.mean()),
            "std": float(filled.std()),
            "min": float(filled.min()),
            "max": float(filled.max()),
            "probes": [{"r": r, "c": c, "value": float(filled[r, c])} for r, c in MISSING_PROBES],
        })

    gaussian = structure_tensor_field(
        sti.astype(np.float64),
        window_px=GAUSSIAN_WINDOW_PX,
        window_type="gaussian",
        overlap=OVERLAP,
        gradient_order=GRADIENT_ORDER,
    )
    gaussian_est = mean_orientation_angle(
        gaussian,
        PHI_MIN,
        PHI_MAX,
        coherency_levels=COHERENCY_LEVELS,
        n_bins=N_BINS,
        min_coherency=MIN_COHERENCY,
    )

    flat_sti = sti.astype(np.float64)
    flat_sti[:, :FLAT_COLS] = FLAT_FILL
    flat_field = structure_tensor_field(
        flat_sti,
        window_px=WINDOW_PX,
        window_type=WINDOW_TYPE,
        overlap=OVERLAP,
        gradient_order=GRADIENT_ORDER,
    )
    flat_est = mean_orientation_angle(
        flat_field,
        PHI_MIN,
        PHI_MAX,
        coherency_levels=COHERENCY_LEVELS,
        n_bins=N_BINS,
        min_coherency=FLAT_MIN_COHERENCY,
    )
    flat_mask = orientation.flat_windows(flat_field)

    grids = []
    for window_px, window_type in GRID_CASES:
        f = structure_tensor_field(
            sti.astype(np.float64),
            window_px=window_px,
            window_type=window_type,
            overlap=OVERLAP,
            gradient_order=GRADIENT_ORDER,
        )
        grids.append(dict(
            {"window_px": window_px, "window_type": window_type, "overlap": OVERLAP,
             "gradient_order": GRADIENT_ORDER},
            **field_summary(f),
        ))

    (out_dir / "golden_sti.f32").write_bytes(sti.tobytes(order="C"))

    payload = {
        "_comment": "Generated by ios/tools/make_golden_fixture.py. Do not edit by hand.",
        "sti": {"rows": ROWS, "cols": COLS, "file": "golden_sti.f32", "dtype": "float32le"},
        "preprocess": {
            "settings": STI_SETTINGS,
            "mean": float(conditioned.mean()),
            "std": float(conditioned.std()),
            "min": float(conditioned.min()),
            "max": float(conditioned.max()),
            "probes": [
                {"r": r, "c": c, "value": float(conditioned[r, c])}
                for r, c in STI_PROBES
            ],
        },
        "parameters": {
            "window_px": WINDOW_PX,
            "window_type": WINDOW_TYPE,
            "overlap": OVERLAP,
            "gradient_order": GRADIENT_ORDER,
            "phi_min": PHI_MIN,
            "phi_max": PHI_MAX,
            "coherency_levels": COHERENCY_LEVELS,
            "n_bins": N_BINS,
            "min_coherency": MIN_COHERENCY,
            "tan_phi_truth": TAN_PHI,
        },
        "field": field_summary(field),
        "estimate": estimate_summary(est),
        "peak_estimate": estimate_summary(peak_est),
        "preprocess_missing": {
            "cells": [list(rc) for rc in MISSING_CELLS],
            "column": MISSING_COLUMN,
            "runs": missing_runs,
        },
        "gaussian_field": dict(
            {"window_px": GAUSSIAN_WINDOW_PX, "window_type": "gaussian", "overlap": OVERLAP,
             "gradient_order": GRADIENT_ORDER},
            **field_summary(gaussian),
            estimate=estimate_summary(gaussian_est),
        ),
        "window_grids": grids,
        "flat_windows": {
            "flat_cols": FLAT_COLS,
            "fill": FLAT_FILL,
            "window_px": WINDOW_PX,
            "window_type": WINDOW_TYPE,
            "overlap": OVERLAP,
            "gradient_order": GRADIENT_ORDER,
            "min_coherency": FLAT_MIN_COHERENCY,
            "rel_energy": orientation.FLAT_WINDOW_REL_ENERGY,
            "n_flat": int(flat_mask.sum()),
            "flat": [bool(v) for v in flat_mask.ravel()],
            "estimate": estimate_summary(flat_est),
        },
        "analysis": analysis_stage(sti.astype(np.float64)),
    }
    (out_dir / "golden.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    print(f"wrote {out_dir / 'golden_sti.f32'} ({sti.nbytes} bytes)")
    print(f"wrote {out_dir / 'golden.json'}")
    print(f"  phi = {est.phi:.9f} rad, tan phi = {est.tan_phi:.9f} (truth {TAN_PHI})")
    print(f"  windows {field.phi.shape}, mean coherency {field.coherency.mean():.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
