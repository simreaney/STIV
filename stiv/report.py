"""Writing a completed run's tables, figures and summary to disk.

Shared by the CLI (``stiv run``) and the interactive notebook UI, so both
front ends produce the same on-disk layout described in the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import __version__
from .analysis import StivResult
from .config import Config


def write_outputs(result: StivResult, cfg: Config) -> Path:
    """Write tables, figures and the run report into ``output.out_dir``."""
    from . import plotting

    out = Path(cfg.output.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = result.to_frame()
    df.to_csv(out / "velocity.csv", index=False)

    if result.discharge is not None:
        result.discharge.to_frame().to_csv(out / "discharge_panels.csv", index=False)

    (out / "report.txt").write_text(result.summary() + "\n", encoding="utf-8")
    cfg.dump_yaml(out / "config_used.yaml")

    summary = {
        "version": __version__,
        "section": cfg.section.name,
        "section_width_m": result.geometry.width_m,
        "n_stations": len(result.stations),
        "n_failed": int(sum(1 for s in result.stations if s.failed)),
        "n_frames": result.n_frames,
        "duration_s": result.duration_s,
        "st_s_per_px": result.st_s_per_px,
        "rectification": result.geometry.rectifier.describe(),
        "flow_bearing_deg": result.geometry.flow_bearing_deg,
        "mean_surface_velocity_ms": float(np.nanmean(result.velocities)),
        "max_surface_velocity_ms": float(np.nanmax(result.velocities)),
        "stabilisation": result.stabilisation.summary(),
        "warnings": result.warnings,
        "elapsed_s": result.elapsed_s,
    }
    if result.discharge is not None:
        summary["discharge_m3s"] = result.discharge.discharge_m3s
        summary["area_m2"] = result.discharge.area_m2
        summary["mean_velocity_ms"] = result.discharge.mean_velocity_ms
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if cfg.output.save_diagnostics:
        plotting.save_all(result, out, dpi=cfg.output.dpi)

    if cfg.output.save_sti_images:
        import cv2

        from .sti import to_display

        sti_dir = out / "sti"
        sti_dir.mkdir(exist_ok=True)
        for s, st in zip(result.stis, result.stations):
            cv2.imwrite(str(sti_dir / f"sti_{st.index:03d}_{st.station_m:07.2f}m.png"),
                        to_display(s.data))

    if cfg.output.save_fields:
        np.savez_compressed(
            out / "orientation_fields.npz",
            **{
                f"phi_{i:03d}": f.phi for i, f in enumerate(result.fields) if f is not None
            },
            **{
                f"coh_{i:03d}": f.coherency for i, f in enumerate(result.fields) if f is not None
            },
        )

    return out
