"""Writing a completed run's tables, figures and summary to disk.

Shared by the CLI (``stiv run``) and the interactive notebook UI, so both
front ends produce the same on-disk layout described in the README.

A survey of several cross-sections (:func:`write_survey_outputs`) writes each
section's usual set of files into a sub-folder named after it, and beside them
the files that compare the sections: ``sections.csv``, the combined
``report.txt``, ``summary.json`` and ``config_used.yaml``, and the overview
and flow map with every section on one frame.
"""

from __future__ import annotations

import copy
import json
import math
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any, Optional

import numpy as np

from . import __version__
from .analysis import StivResult, SurveyResult
from .config import Config, section_folder

#: Every run lists the files it wrote here, in ``output.out_dir``.  The next run
#: into the same folder removes exactly those before writing, so nothing from an
#: earlier run (a discharge table, a fields archive, the STIs of stations that
#: no longer exist) survives into one that no longer produces it.
MANIFEST = ".stiv_outputs.json"

#: Files this module writes at the top of ``output.out_dir``.
OUTPUT_FILES = (
    "velocity.csv",
    "sections.csv",
    "discharge_panels.csv",
    "report.txt",
    "config_used.yaml",
    "summary.json",
    "orientation_fields.npz",
    "overview.png",
    "velocity_map.png",
    "velocity_profile.png",
    "sti_panel.png",
    "discharge.png",
    "report.pdf",
    # A method compared with the reported one writes its own profile.
    "velocity_stiv.csv",
    "velocity_lspiv.csv",
    "velocity_ptv.csv",
    # LS-PIV's and PTV's diagnostics.
    "correlation_panel.png",
    "tracks.png",
)

#: Sub-folders this module writes, and the only files it ever puts in them.
OUTPUT_DIRS = {
    "sti": "sti_[0-9][0-9][0-9]_*m.png",
    "sti_detail": "station_[0-9][0-9][0-9].png",
}


def _is_output_name(rel: PurePosixPath) -> bool:
    """True for a path, relative to the output folder, that STIV could write."""
    if len(rel.parts) == 1:
        return rel.parts[0] in OUTPUT_FILES
    if len(rel.parts) == 2 and rel.parts[0] in OUTPUT_DIRS:
        return fnmatch(rel.parts[1], OUTPUT_DIRS[rel.parts[0]])
    return False


def _is_section_folder(name: Any) -> bool:
    """True for a manifest's section entry that is a plain sub-folder name."""
    return (isinstance(name, str) and bool(name) and section_folder(name) == name
            and name not in OUTPUT_DIRS)


def clear_outputs(out: Path) -> list[Path]:
    """Remove what the previous STIV run into ``out`` wrote, and nothing else.

    An output folder is often a survey folder too, holding the footage, field
    notes -- or an ``overview.png`` of somebody's own.  So only files listed in
    the previous run's :data:`MANIFEST` are removed, and of those only names
    STIV writes, inside ``out``.  A folder with no manifest (never used for
    output, or written by a version that kept none) loses nothing.  A
    sub-folder is removed only if that leaves it empty.  The section folders
    a survey's run listed are cleared the same way, each by its own manifest.
    Returns what was removed.
    """
    manifest = out / MANIFEST
    data = _read_manifest(out)
    listed = data.get("files")
    if not isinstance(listed, list):
        return []

    removed: list[Path] = []
    # A survey's sections each keep their own manifest in their own folder.
    sections = data.get("sections")
    for folder in sections if isinstance(sections, list) else []:
        if not _is_section_folder(folder):
            continue
        sub = out / folder
        if sub.is_dir():
            removed += clear_outputs(sub)
            try:
                sub.rmdir()
            except OSError:  # holds something STIV did not write; leave it be
                pass
    subdirs: set[str] = set()
    for entry in listed:
        if not isinstance(entry, str):
            continue
        rel = PurePosixPath(entry)
        if rel.is_absolute() or ".." in rel.parts or not _is_output_name(rel):
            continue
        f = out.joinpath(*rel.parts)
        if f.is_file():
            f.unlink()
            removed.append(f)
        if len(rel.parts) == 2:
            subdirs.add(rel.parts[0])
    for sub in subdirs:
        try:
            (out / sub).rmdir()
        except OSError:  # not empty, or already gone; leave it be
            pass
    manifest.unlink(missing_ok=True)
    return removed


def add_pdf_report(result: StivResult | SurveyResult, cfg: Config) -> Path:
    """Write ``report.pdf`` into the output folder of a run already written.

    For a report asked for after the fact -- the notebook's results step
    offers one.  It is added to the folder's manifest, so the next run there
    clears it along with everything else this run wrote.  A survey of several
    sections gets one report covering them all, at the top of its folder.
    """
    from .pdf_report import write_pdf_report

    out = Path(cfg.output.out_dir)
    if isinstance(result, SurveyResult):
        from .pdf_report import write_survey_pdf_report

        path = write_survey_pdf_report(result, cfg, out / "report.pdf")
    else:
        path = write_pdf_report(result, cfg, out / "report.pdf")
    data = _read_manifest(out)
    listed = data.get("files", [])
    if not isinstance(listed, list):
        listed = []
    sections = data.get("sections")
    _write_manifest(out, [out / f for f in listed if isinstance(f, str)] + [path],
                    sections=sections if isinstance(sections, list) else None)
    return path


def _read_manifest(out: Path) -> dict:
    try:
        data = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_manifest(out: Path, written: list[Path], sections: Optional[list[str]] = None) -> None:
    files = sorted({p.relative_to(out).as_posix() for p in written if p.is_file()})
    data: dict[str, Any] = {"written_by": f"stiv {__version__}", "files": files}
    if sections is not None:
        data["sections"] = sections
    (out / MANIFEST).write_text(json.dumps(data, indent=1), encoding="utf-8")


def _json_safe(obj: Any) -> Any:
    """Make ``obj`` strict JSON: NaN and infinities become null."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if math.isfinite(x) else None
    return obj


def write_outputs(result: StivResult, cfg: Config) -> Path:
    """Write tables, figures and the run report into ``output.out_dir``.

    Whatever the previous run listed in the folder's :data:`MANIFEST` is
    removed first, and this run's own list is written last.
    """
    out = Path(cfg.output.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clear_outputs(out)
    written: list[Path] = []
    # The manifest is written even if a figure or table fails part-way, so the
    # next run can still clear up whatever this one did manage to write.
    try:
        _write_all(result, cfg, out, written)
    finally:
        _write_manifest(out, written)
    return out


def write_survey_outputs(survey: SurveyResult, cfg: Config) -> Path:
    """Write every cross-section's outputs, and the files comparing them.

    ``cfg`` is the survey's own config, and its ``output`` block decides
    what is written where.  One with a single ``section`` is written exactly
    as :func:`write_outputs` writes it.  Otherwise each section goes into its
    own sub-folder of ``output.out_dir``, named by :func:`section_folder`, as
    if it had been run alone, and the survey-wide files go at the top.  A
    survey's ``report.pdf`` covers every section, so the sections write none
    of their own.
    """
    if cfg.sections is None:
        return write_outputs(survey.results[0], cfg)

    out = Path(cfg.output.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clear_outputs(out)
    written: list[Path] = []
    folders: list[str] = []
    try:
        for r in survey.results:
            folder = section_folder(r.config.section.name)
            # Listed before it is written, so a run that fails part-way still
            # leaves the next one a way to clear what it did write.
            folders.append(folder)
            section_cfg = copy.deepcopy(r.config)
            section_cfg.output = copy.deepcopy(cfg.output)
            section_cfg.output.out_dir = str(out / folder)
            section_cfg.output.pdf_report = False
            write_outputs(r, section_cfg)
        _write_survey_files(survey, cfg, out, written)
    finally:
        _write_manifest(out, written, sections=folders)
    return out


def _write_survey_files(survey: SurveyResult, cfg: Config, out: Path, written: list[Path]) -> None:
    from . import plotting

    def _done(path: Path) -> Path:
        written.append(path)
        return path

    survey.to_frame().to_csv(_done(out / "sections.csv"), index=False)
    _done(out / "report.txt").write_text(survey.summary() + "\n", encoding="utf-8")
    cfg.dump_yaml(_done(out / "config_used.yaml"))

    first = survey.results[0]
    summary = {
        "version": __version__,
        "n_sections": len(survey.results),
        "n_frames": first.n_frames,
        "duration_s": first.duration_s,
        "st_s_per_px": first.st_s_per_px,
        "rectification": first.geometry.rectifier.describe(),
        "stabilisation": first.stabilisation.summary(),
        "warnings": survey.shared_warnings,
        "elapsed_s": survey.elapsed_s,
        "sections": [
            {"folder": section_folder(r.config.section.name), **_summary_dict(r, r.config)}
            for r in survey.results
        ],
    }
    for entry in summary["sections"]:
        for key in ("version", "n_frames", "duration_s", "st_s_per_px", "rectification",
                    "stabilisation", "elapsed_s"):
            entry.pop(key, None)
    _done(out / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
    )

    if cfg.output.save_diagnostics:
        dpi = cfg.output.dpi
        written.append(plotting.plot_survey_overview(survey.results, out / "overview.png", dpi))
        written.append(plotting.plot_survey_velocity_map(survey.results,
                                                         out / "velocity_map.png", dpi))

    if cfg.output.pdf_report:
        from .pdf_report import write_survey_pdf_report

        written.append(write_survey_pdf_report(survey, cfg, out / "report.pdf"))


def _write_all(result: StivResult, cfg: Config, out: Path, written: list[Path]) -> None:
    from . import plotting

    def _done(path: Path) -> Path:
        written.append(path)
        return path

    df = result.to_frame()
    df.to_csv(_done(out / "velocity.csv"), index=False)
    for m in result.comparisons:
        result.to_frame(m).to_csv(_done(out / f"velocity_{m}.csv"), index=False)

    if result.discharge is not None:
        result.discharge.to_frame().to_csv(_done(out / "discharge_panels.csv"), index=False)

    _done(out / "report.txt").write_text(result.summary() + "\n", encoding="utf-8")
    cfg.dump_yaml(_done(out / "config_used.yaml"))

    summary = _summary_dict(result, cfg)
    # allow_nan=False: a bare NaN is not JSON, and strict parsers reject the
    # whole file over it.
    _done(out / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
    )

    if cfg.output.save_diagnostics:
        written.extend(plotting.save_all(result, out, dpi=cfg.output.dpi))

    if cfg.output.pdf_report:
        from .pdf_report import write_pdf_report

        written.append(write_pdf_report(result, cfg, out / "report.pdf"))

    if cfg.output.save_sti_images:
        import cv2

        from .sti import to_display

        sti_dir = out / "sti"
        sti_dir.mkdir(exist_ok=True)
        for s, st in zip(result.stis, result.stations):
            cv2.imwrite(str(_done(sti_dir / f"sti_{st.index:03d}_{st.station_m:07.2f}m.png")),
                        to_display(s.data))

    if cfg.output.save_fields:
        np.savez_compressed(
            _done(out / "orientation_fields.npz"),
            **{
                f"phi_{i:03d}": f.phi for i, f in enumerate(result.fields) if f is not None
            },
            **{
                f"coh_{i:03d}": f.coherency for i, f in enumerate(result.fields) if f is not None
            },
        )


def _summary_dict(result: StivResult, cfg: Config) -> dict[str, Any]:
    """``summary.json`` of one section's run."""
    v = result.velocities
    finite = v[np.isfinite(v)]
    summary = {
        "version": __version__,
        "section": cfg.section.name,
        "section_width_m": result.geometry.width_m,
        "n_stations": len(result.stations),
        "n_failed": int(sum(1 for s in result.stations if s.failed)),
        "n_interpolated": int(sum(1 for s in result.stations
                                  if s.failed and "interpolated" in s.flags)),
        "n_frames": result.n_frames,
        "duration_s": result.duration_s,
        "st_s_per_px": result.st_s_per_px,
        "rectification": result.geometry.rectifier.describe(),
        "flow_bearing_deg": result.geometry.flow_bearing_deg,
        # Over the reported profile only; null when no station reported one.
        "mean_surface_velocity_ms": float(finite.mean()) if finite.size else None,
        "max_surface_velocity_ms": float(finite.max()) if finite.size else None,
        "stabilisation": result.stabilisation.summary(),
        "warnings": result.warnings,
        "elapsed_s": result.elapsed_s,
    }
    if result.discharge is not None:
        summary["discharge_m3s"] = result.discharge.discharge_m3s
        summary["area_m2"] = result.discharge.area_m2
        summary["mean_velocity_ms"] = result.discharge.mean_velocity_ms
    if result.method != "stiv" or result.comparisons:
        summary["method"] = result.method
        summary["comparisons"] = {}
        for m in result.comparisons:
            st = result.comparisons[m]
            v = np.array([s.velocity_ms for s in st], dtype=float)
            v = v[np.isfinite(v)]
            q = result.comparison_discharge.get(m)
            summary["comparisons"][m] = {
                "n_failed": int(sum(1 for s in st if s.failed)),
                "mean_surface_velocity_ms": float(v.mean()) if v.size else None,
                "max_surface_velocity_ms": float(v.max()) if v.size else None,
                **({"discharge_m3s": q.discharge_m3s} if q is not None else {}),
            }
    return summary
