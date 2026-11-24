"""Command line interface.

::

    stiv info      flight.mp4              inspect a video
    stiv init      flight.mp4 -o survey.yaml   write a starter config
    stiv pick      flight.mp4 -o survey.yaml   click the section on a frame
                   (--sections 3 to click three, analysed together)
    stiv run       survey.yaml             analyse and write the outputs
    stiv cameras                           list the DJI field-of-view presets
    stiv demo                              render a synthetic river and analyse it

The scale can be given to info, init and pick as the flight geometry:
``--camera mini-4-pro`` (or ``--hfov-deg``) for the field of view, and
``--altitude-m`` -- or, with the drone's ``.SRT`` flight log beside the video,
``--takeoff-height`` -- for the height above the water.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import __version__
from .config import (
    FRAME_STEP_SPEEDS_MS,
    Config,
    ConfigError,
    RectifyConfig,
    SectionConfig,
    default_duration_s,
    path_for_config,
    suggest_frame_step,
    survey_dict,
    yaml_text,
)
from .report import write_outputs as _write_outputs

log = logging.getLogger("stiv")

#: ``stiv demo`` defaults, shared with the notebook wizard's demo river so the
#: two run exactly the same analysis.
DEMO_DURATION_S = 10.0
DEMO_V_MAX_MS = 1.2
DEMO_NOISE = 0.02
DEMO_DRIFT_PX = 45.0
DEMO_N_STATIONS = 21

#: Validation statistics are taken over stations faster than this, since a
#: relative error is meaningless where the truth approaches zero.
DEMO_MOVING_MS = 0.15


# --------------------------------------------------------------------------- #
def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity < 0 else (logging.DEBUG if verbosity > 0 else logging.INFO)
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _refuse_overwrite(path: Path, force: bool) -> None:
    # A survey file is usually the product of careful clicking and editing, so
    # it is never replaced silently.
    if path.exists() and not force:
        raise ConfigError(f"{path} already exists; pass --force to overwrite it")


def _default_out_dir(config_path: Path) -> str:
    # Relative, so it resolves next to the config file rather than the shell.
    return config_path.with_suffix("").name + "_output"


# --------------------------------------------------------------------------- #
# the scale, from the command line
# --------------------------------------------------------------------------- #
@dataclass
class _Scale:
    """The scale the scale options describe, and how it was arrived at."""

    m_per_px: Optional[float] = None
    altitude_m: Optional[float] = None
    hfov_deg: Optional[float] = None
    #: How each number was arrived at, for the terminal and the YAML header.
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: The flight log beside the video, and what it records over the window.
    srt: Optional[Path] = None
    srt_stats: object = None

    @property
    def complete(self) -> bool:
        return self.m_per_px is not None or (self.altitude_m is not None
                                             and self.hfov_deg is not None)

    def missing(self) -> list[str]:
        """The rectify fields still to fill in, when flight geometry was begun."""
        if self.complete or (self.altitude_m is None and self.hfov_deg is None):
            return []
        if self.altitude_m is None:
            return ["rectify.altitude_m (height above the WATER SURFACE, not take-off)"]
        return ["rectify.hfov_deg (the HORIZONTAL field of view; `stiv cameras` "
                "lists DJI presets)"]


def _camera_preset(key: str):
    from .cameras import DJI_CAMERAS, camera

    preset = camera(key)
    if preset is None:
        raise ConfigError(
            f"unknown --camera {key!r}; choose one of: "
            f"{', '.join(p.key for p in DJI_CAMERAS)} (`stiv cameras` describes them), "
            f"or give the horizontal field of view with --hfov-deg"
        )
    return preset


def _resolve_scale(args, video: str | Path, start_s: float = 0.0,
                   duration_s: Optional[float] = None) -> _Scale:
    """Turn the scale options into a scale, refusing contradictory ones.

    Called before any clicking or writing, so a mistake costs nothing.  The
    flight log's altitude is height above the *take-off point*, so it is used
    only with ``--takeoff-height`` -- the height of that point above the water
    -- and never stands in for height above the water on its own.
    """
    from .srt import altitude_stats, find_srt, parse_srt

    m_per_px = getattr(args, "m_per_px", None)
    key, hfov = getattr(args, "camera", None), getattr(args, "hfov_deg", None)
    altitude, takeoff = getattr(args, "altitude_m", None), getattr(args, "takeoff_height", None)

    if key and hfov is not None:
        raise ConfigError("give --camera or --hfov-deg, not both: the camera preset is "
                          "the horizontal field of view")
    if altitude is not None and takeoff is not None:
        raise ConfigError("give --altitude-m or --takeoff-height, not both: "
                          "--takeoff-height works the altitude out from the flight log")
    if m_per_px is not None and any(x is not None for x in (key, hfov, altitude, takeoff)):
        raise ConfigError("--m-per-px is the scale itself; drop --camera, --hfov-deg, "
                          "--altitude-m and --takeoff-height, or drop --m-per-px")
    for name, value in (("--m-per-px", m_per_px), ("--altitude-m", altitude),
                        ("--hfov-deg", hfov)):
        if value is not None and not value > 0:
            raise ConfigError(f"{name} must be greater than zero")
    if hfov is not None and not hfov < 180:
        raise ConfigError("--hfov-deg must be under 180 degrees")

    sc = _Scale(m_per_px=m_per_px, altitude_m=altitude, hfov_deg=hfov)
    preset = None
    if key:
        preset = _camera_preset(key)
        # One decimal, as the notebook wizard writes it, so a survey set up
        # either way reads the same -- and reopens in the wizard as this drone.
        sc.hfov_deg = round(preset.hfov_deg, 1)
        sc.notes.append(f"hfov_deg {sc.hfov_deg:g} from --camera {preset.key}: {preset.describe()}")

    sc.srt = find_srt(video)
    if sc.srt is not None:
        try:
            sc.srt_stats = altitude_stats(parse_srt(sc.srt), start_s, duration_s)
        except OSError:
            sc.srt_stats = None
    st = sc.srt_stats

    if takeoff is not None:
        if sc.srt is None:
            raise ConfigError(
                f"--takeoff-height needs the drone's flight log beside the video "
                f"({Path(video).with_suffix('.SRT').name}), and there is none; give "
                f"--altitude-m, the height above the water, instead"
            )
        if st is None:
            raise ConfigError(
                f"{sc.srt.name} records no height above take-off, so --takeoff-height "
                f"cannot be used; give --altitude-m, the height above the water, instead"
            )
        sc.altitude_m = round(st.median_m + takeoff, 2)
        spread = (f", {st.min_m:.1f}-{st.max_m:.1f} m" if st.max_m - st.min_m >= 0.05 else "")
        sc.notes.append(
            f"altitude_m {sc.altitude_m:g} = {st.median_m:.2f} m above take-off "
            f"({sc.srt.name}, median over the "
            f"{'analysis window' if duration_s is not None else 'clip'}{spread}) + {takeoff:g} m "
            f"take-off height above the water"
        )
        if st.assumed:
            sc.warnings.append(
                f"{sc.srt.name} logs only a plain \"altitude\", which on the Air 2S and Mavic 2 "
                f"is the height above the take-off point and is used as such here. Check that "
                f"{st.median_m:.1f} m is that, not the height above sea level"
            )

    narrowed = preset.zoomed_hfov_deg(st.focal_len_mm) if preset and st else None
    if narrowed is not None:
        sc.warnings.append(
            f"{sc.srt.name} logs a {st.focal_len_mm:g} mm-equivalent focal length, about "
            f"{st.focal_len_mm / preset.eq_focal_mm:.1f}x digital zoom on the "
            f"{preset.label}'s {preset.eq_focal_mm:g} mm lens, so the view is narrower "
            f"than the preset's: about {narrowed:.1f} deg horizontally. If the zoom was "
            f"on for this clip, use --hfov-deg {narrowed:.1f} instead of --camera"
        )
    return sc


def _report_scale(sc: _Scale) -> None:
    for note in sc.notes:
        print(f"  {note}")
    for warning in sc.warnings:
        print(f"warning: {warning}", file=sys.stderr)


def _header_notes(sc: _Scale) -> str:
    """The scale's provenance, as YAML comments."""
    return "".join(f"# {note}\n" for note in sc.notes)


def _apply_scale(raw: dict, sc: _Scale) -> None:
    """Write the scale into a raw config built without one."""
    if sc.m_per_px is not None:
        raw["rectify"] = {"method": "scale", "m_per_px": sc.m_per_px}
    elif sc.altitude_m is not None or sc.hfov_deg is not None:
        raw["rectify"] = {"method": "scale", "altitude_m": sc.altitude_m,
                          "hfov_deg": sc.hfov_deg}


# --------------------------------------------------------------------------- #
def cmd_cameras(args) -> int:
    from .cameras import DJI_CAMERAS

    kw = max(len(p.key) for p in DJI_CAMERAS) + 2
    lw = max(len(p.label) for p in DJI_CAMERAS) + 2
    print(f"{'key':<{kw}}{'drone':<{lw}}{'horizontal FOV':>14}   {'DJI quotes (diagonal)'}")
    for p in DJI_CAMERAS:
        print(f"{p.key:<{kw}}{p.label:<{lw}}{p.hfov_deg:>10.1f} deg   {p.dfov_deg:g} deg")
    print()
    print("DJI's spec sheets quote the DIAGONAL field of view; the scale needs the\n"
          "horizontal one, which --camera KEY fills in (e.g. `stiv init clip.mp4\n"
          "--camera mini-4-pro --altitude-m 40`). The presets assume the normal\n"
          "landscape video mode, which uses the full sensor width: digital zoom,\n"
          "cropped high-frame-rate modes and vertical shooting narrow the view.")
    return 0


# --------------------------------------------------------------------------- #
def cmd_info(args) -> int:
    from .video import probe

    info = probe(args.video)
    # Over the whole clip: info does not know which part will be analysed.
    sc = _resolve_scale(args, args.video)
    # probe() reports 0 when the file records no frame rate (a raw .h264 or
    # .mjpeg stream, say) rather than guessing one.
    fps_known = info.fps > 0
    print(info)
    if sc.srt is not None:
        title = f"Flight log ({sc.srt.name})"
        print()
        print(title)
        print("-" * len(title))
        st = sc.srt_stats
        if st is None:
            print("  records no height above take-off")
        else:
            spread = f" ({st.min_m:.1f}-{st.max_m:.1f} m)" if st.max_m - st.min_m >= 0.05 else ""
            print(f"  height above take-off   {st.median_m:.1f} m, median over the clip{spread}")
            if st.assumed:
                print('  (logged as plain "altitude": on the Air 2S and Mavic 2 that is the height '
                      "above take-off;\n  check it is not the height above sea level)")
            if st.focal_len_mm:
                print(f"  focal length            {st.focal_len_mm:g} mm (35 mm equivalent)")
            print("  That is NOT the height above the water. Give the take-off point's "
                  "height above the water\n  with --takeoff-height (0 if you took off at "
                  "water level) to use it.")
    print()
    print("Suggested settings")
    print("------------------")
    if not fps_known:
        print("  video.fps_override    ?        (the file records no frame rate: set this "
              "to the rate the footage was recorded at -- every velocity is "
              "proportional to it)")
    print(f"  video.duration_s      {default_duration_s(info.duration_s):.1f}   "
          f"(20-40 s of steady hover is plenty; the paper used 30 s)")
    if sc.altitude_m is not None and sc.hfov_deg is not None:
        rect = RectifyConfig(altitude_m=sc.altitude_m, hfov_deg=sc.hfov_deg)
        gsd = rect.resolve_m_per_px(info.width)
        gw = gsd * info.width
        print(f"  rectify.m_per_px      {gsd:.4f}   "
              f"(from {sc.altitude_m:g} m altitude, {sc.hfov_deg:g} deg HFOV; "
              f"ground width {gw:.1f} m)")
        _report_scale(sc)
        if fps_known:
            for v in FRAME_STEP_SPEEDS_MS:
                print(f"      if U ~ {v:>3.1f} m/s  ->  video.frame_step: "
                      f"{suggest_frame_step(gsd, info.fps, v)}   (puts tan phi near 1)")
        else:
            # Without a frame rate every suggestion would collapse to step 1.
            print("      (no frame_step suggestion without a frame rate; re-run once "
                  "you know it, or work it out as m_per_px * fps / U)")
    else:
        _report_scale(sc)
        print("  pass --altitude-m (or --takeoff-height, with a flight log) and --camera "
              "(or --hfov-deg)\n  to get a ground sample distance and a frame_step "
              "suggestion; `stiv cameras` lists the drones")
    return 0


_INIT_HEADER = """\
# STIV survey configuration (written by `stiv init`)
#
# `stiv run` refuses this file until the null placeholders are filled in:
#
#   rectify.m_per_px    ground sample distance, metres per pixel.  Or replace
#                       it with the flight geometry:
#                         altitude_m: 60.0   # above the WATER SURFACE
#                         hfov_deg: 71.5     # HORIZONTAL FOV, not the diagonal
#                                            # one DJI quotes (see stiv.cameras)
{section_help}#
# flow.downstream_px is optional: any image point downstream of {the_sections},
# which fixes the sign of the flow direction.
#
# stabilise.water_mask_px can be set to `auto`, to outline the water from the
# footage when the analysis starts, rather than to polygons.
#
# Video: {info}

"""

_INIT_HEADER_SCALED = """\
# STIV survey configuration (written by `stiv init`)
#
# `stiv run` refuses this file until the null placeholders are filled in:
#
{missing}{section_help}#
# flow.downstream_px is optional: any image point downstream of {the_sections},
# which fixes the sign of the flow direction.
#
# stabilise.water_mask_px can be set to `auto`, to outline the water from the
# footage when the analysis starts, rather than to polygons.
#
{notes}# Video: {info}

"""


_SECTION_HELP = """\
#   section.start_px    the two ends of the cross-section, in image pixels
#   section.end_px      (`stiv pick` lets you click them on a frame instead)
"""

_SECTIONS_HELP = """\
#   sections: start_px  the two ends of each cross-section, in image pixels
#             end_px    (`stiv pick --sections {n}` lets you click them instead)
#
# The cross-sections are analysed together, in one pass over the video.  Each
# one's name also names the sub-folder its results go in; anything else an
# entry leaves out (n_stations, edge_trim) comes from the `section` block, and
# an entry can have its own downstream_px and bathymetry_csv.
"""


def _check_n_sections(n: int) -> None:
    if n < 1:
        raise ConfigError(f"--sections must be 1 or more, got {n}")


def cmd_init(args) -> int:
    from .video import probe

    _check_n_sections(args.sections)
    out = Path(args.output)
    _refuse_overwrite(out, args.force)
    info = probe(args.video)
    duration = default_duration_s(info.duration_s)
    sc = _resolve_scale(args, args.video, 0.0, duration)
    raw = survey_dict(
        path_for_config(args.video, out),
        duration_s=duration,
        name=Path(args.video).stem,
        out_dir=_default_out_dir(out),
        sections=[{} for _ in range(args.sections)] if args.sections > 1 else None,
    )
    _apply_scale(raw, sc)
    multi = args.sections > 1
    wording = {
        "section_help": _SECTIONS_HELP.format(n=args.sections) if multi else _SECTION_HELP,
        "the_sections": "every cross-section" if multi else "the section",
    }
    if sc.m_per_px is None and sc.altitude_m is None and sc.hfov_deg is None:
        header = _INIT_HEADER.format(info=info, **wording)
    else:
        missing = "".join(f"#   {m}\n" for m in sc.missing())
        header = _INIT_HEADER_SCALED.format(info=info, missing=missing,
                                            notes=_header_notes(sc) + ("#\n" if sc.notes else ""),
                                            **wording)
    out.write_text(header + yaml_text(raw), encoding="utf-8")
    print(f"wrote {out}")
    _report_scale(sc)
    todo = [f"the coordinates of the {args.sections} cross-sections" if multi
            else "the section coordinates"]
    if not sc.complete:
        todo.insert(0, " and ".join(m.split(" ")[0] for m in sc.missing()) or "the scale")
    print(f"Next: fill in {' and '.join(todo)}, then `stiv run {out}`")
    return 0


def cmd_pick(args) -> int:
    from .picker import build_config_interactively

    from .video import probe

    out = Path(args.output)
    # Everything that can be refused is refused before any clicking, not after.
    _check_n_sections(args.sections)
    _refuse_overwrite(out, args.force)
    scale_given = any(getattr(args, k) is not None for k in
                      ("m_per_px", "altitude_m", "hfov_deg", "camera", "takeoff_height"))
    if args.gcps and scale_given:
        raise ConfigError("--gcps takes the scale from ground control points; drop the "
                          "other scale options")
    info = probe(args.video)
    sc = _resolve_scale(args, args.video, args.time,
                        default_duration_s(info.duration_s, args.time))
    raw = build_config_interactively(
        args.video,
        out_dir=_default_out_dir(out),
        t_s=args.time,
        with_gcps=args.gcps,
        n_stations=args.n_stations,
        config_path=out,
        auto_water=not args.no_auto_water,
        n_sections=args.sections,
    )
    _apply_scale(raw, sc)
    header = "# STIV survey configuration (written by `stiv pick`)\n#\n" + _header_notes(sc)
    if args.sections > 1:
        header += ("# The cross-sections are analysed together; rename them in `sections`\n"
                   "# if you like: each name is also its results sub-folder.\n#\n")
    if args.gcps:
        header += ("# Fill in rectify.gcp_world -- the surveyed position of each control\n"
                   "# point in metres, in the order they were clicked -- before running.\n\n")
        todo = ["rectify.gcp_world"]
    elif sc.complete:
        header += "\n"
        todo = []
    elif sc.missing():
        todo = [m.split(" ")[0] for m in sc.missing()]
        header += "".join(f"# Fill in {m} before running.\n" for m in sc.missing()) + "\n"
    else:
        header += ("# Fill in rectify.m_per_px (metres per pixel), or replace it with\n"
                   "# altitude_m plus hfov_deg, before running.\n\n")
        todo = ["rectify.m_per_px"]
    out.write_text(header + yaml_text(raw), encoding="utf-8")
    print(f"wrote {out}")
    _report_scale(sc)
    if todo:
        print(f"Fill in {' and '.join(todo)}, then `stiv run {out}`")
    else:
        print(f"Next: `stiv run {out}`")
    return 0


def cmd_run(args) -> int:
    from .analysis import run_survey
    from .report import write_survey_outputs

    cfg = Config.from_yaml(args.config)

    for override, value in (
        ("video.frame_step", args.frame_step),
        ("video.duration_s", args.duration),
        ("section.n_stations", args.n_stations),
        ("output.out_dir", args.out_dir),
    ):
        if value is not None:
            section, key = override.split(".")
            setattr(getattr(cfg, section), key, value)
    if args.n_stations is not None:
        # Every cross-section, not just those that leave it to `section`.
        for entry in cfg.sections or []:
            entry.n_stations = args.n_stations
    if args.no_stabilise:
        cfg.stabilise.enabled = False
    if args.no_figures:
        cfg.output.save_diagnostics = False
        cfg.output.save_sti_images = False
    if args.pdf:
        cfg.output.pdf_report = True
    cfg.validate()

    survey = run_survey(cfg, progress=not args.quiet)
    out = write_survey_outputs(survey, cfg)

    print()
    print(survey.summary())
    print()
    print(f"outputs written to {out}")
    return 0


def make_demo_river(
    duration_s: float = DEMO_DURATION_S,
    v_max_ms: float = DEMO_V_MAX_MS,
    noise: float = DEMO_NOISE,
    drift_px: float = DEMO_DRIFT_PX,
):
    """The synthetic river ``stiv demo`` renders: a hover with drift, sway and yaw."""
    from .synthetic import SyntheticRiver

    return SyntheticRiver(
        duration_s=duration_s,
        v_max_ms=v_max_ms,
        noise=noise,
        drone_drift_px=drift_px,
        drone_sway_px=drift_px / 8.0 if drift_px else 0.0,
        drone_yaw_deg=1.5 if drift_px else 0.0,
    )


def demo_validation(got: np.ndarray, truth: np.ndarray,
                    moving_ms: float = DEMO_MOVING_MS) -> tuple[list[str], Optional[float]]:
    """Report lines comparing a demo run with its ground truth, and the RMSE.

    Stations with no reported velocity are left out of the statistics and
    counted instead.  If no reported station is moving faster than
    ``moving_ms`` -- a demo river rendered with ``--v-max 0.1``, say -- the
    absolute errors are taken over every reported station and the relative
    error is skipped, rather than taking the maximum of nothing.
    """
    got = np.asarray(got, dtype=float)
    truth = np.asarray(truth, dtype=float)
    err = got - truth
    reported = np.isfinite(got)
    moving = truth > moving_ms
    m = moving & reported

    lines = [f"  stations with U > {moving_ms:.2f} m/s : {int(moving.sum())}"]
    if (moving & ~reported).any():
        lines.append(f"  ...of which without a value : {int((moving & ~reported).sum())}")
    if not m.any():
        if not reported.any():
            lines.append("  no station reported a velocity, so there is nothing to validate")
            return lines, None
        lines.append(f"  (none moving faster than {moving_ms:.2f} m/s with a value, so the "
                     f"errors below are over all {int(reported.sum())} reported stations "
                     f"and the relative error is omitted)")
        m = reported
        rel = None
    else:
        rel = 100 * np.sqrt(np.mean((err[m] / truth[m]) ** 2))

    rmse = float(np.sqrt(np.mean(err[m] ** 2)))
    lines += [
        f"  RMSE                       : {rmse:.4f} m/s",
        f"  bias                       : {err[m].mean():+.4f} m/s",
    ]
    if rel is not None:
        lines.append(f"  relative RMSE              : {rel:.1f} %")
    lines.append(f"  max absolute error         : {np.abs(err[m]).max():.4f} m/s")
    return lines, rmse


def cmd_demo(args) -> int:
    """Render a synthetic river with a known profile and check STIV recovers it."""
    from .analysis import run as run_analysis
    from .synthetic import make_config, render

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    river = make_demo_river(args.duration, args.v_max, args.noise, args.drift_px)
    video = render(river, out_dir / "synthetic_river", progress=not args.quiet)
    print(f"rendered {video}  "
          f"({river.n_frames} frames, peak {river.v_max_ms:.2f} m/s, "
          f"drone drift {river.drone_drift_px:.0f} px)")

    cfg = Config.from_dict(make_config(river, video, out_dir / "analysis",
                                       n_stations=args.n_stations))
    cfg.output.pdf_report = bool(args.pdf)
    result = run_analysis(cfg, progress=not args.quiet)
    _write_outputs(result, cfg)

    _, truth = river.truth_profile(args.n_stations)
    lines, rmse = demo_validation(result.velocities, truth)

    print()
    print(result.summary())
    print()
    print("Validation against the known velocity field")
    print("-------------------------------------------")
    print("\n".join(lines))
    print()
    print(f"outputs written to {out_dir / 'analysis'}")

    return 0 if rmse is not None and rmse < 0.1 else 1


# --------------------------------------------------------------------------- #
def _add_verbosity(parser: argparse.ArgumentParser, default) -> None:
    parser.add_argument("-q", "--quiet", action="store_true", default=default,
                        help="only report warnings")
    parser.add_argument("-v", "--verbose", action="store_true", default=default,
                        help="debug logging")


def _add_scale_options(parser: argparse.ArgumentParser, m_per_px: bool = True) -> None:
    g = parser.add_argument_group(
        "scale", "the ground resolution, directly or from the flight geometry")
    if m_per_px:
        g.add_argument("--m-per-px", type=float, help="ground resolution, metres per pixel")
    g.add_argument("--altitude-m", type=float,
                   help="flight altitude above the WATER SURFACE, metres")
    g.add_argument("--takeoff-height", type=float, metavar="M",
                   help="height of the take-off point above the water, metres: the altitude "
                        "is then read from the drone's .SRT flight log beside the video, "
                        "which records height above take-off")
    g.add_argument("--hfov-deg", type=float,
                   help="the camera's HORIZONTAL field of view, degrees -- not the diagonal "
                        "one DJI spec sheets quote (84 deg for a 24 mm lens is ~71.5 "
                        "horizontally); see --camera")
    g.add_argument("--camera", metavar="KEY",
                   help="a DJI drone, e.g. mini-4-pro, for its horizontal field of view; "
                        "`stiv cameras` lists them")


def build_parser() -> argparse.ArgumentParser:
    # -q/-v work either side of the subcommand.  The subcommand's copies
    # default to SUPPRESS: a subparser writes its defaults over the parent's
    # namespace, so a plain False there would undo `stiv -q run ...`.
    common = argparse.ArgumentParser(add_help=False)
    _add_verbosity(common, argparse.SUPPRESS)

    p = argparse.ArgumentParser(
        prog="stiv",
        description="Space-Time Image Velocimetry for river surface flow "
                    "(Fujita, Watanabe & Tsubaki, 2007).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    _add_verbosity(p, False)
    p.add_argument("--version", action="version", version=f"stiv {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("info", parents=[common],
                       help="inspect a video and suggest settings")
    s.add_argument("video")
    _add_scale_options(s, m_per_px=False)
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("init", parents=[common], help="write a starter config file")
    s.add_argument("video")
    s.add_argument("-o", "--output", default="survey.yaml")
    s.add_argument("--sections", type=int, default=1, metavar="N",
                   help="lay out N cross-sections, analysed together (default 1)")
    s.add_argument("--force", action="store_true", help="overwrite an existing config file")
    _add_scale_options(s)
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("pick", parents=[common],
                       help="click the cross-section on a frame")
    s.add_argument("video")
    s.add_argument("-o", "--output", default="survey.yaml")
    s.add_argument("-t", "--time", type=float, default=0.0,
                   help="seconds into the video to display; also becomes video.start_s, "
                        "since the stabiliser measures drift from the first analysed frame")
    s.add_argument("--gcps", action="store_true",
                   help="also click ground control points for an oblique view")
    s.add_argument("--sections", type=int, default=1, metavar="N",
                   help="click N cross-sections, analysed together in one pass over the "
                        "video (default 1)")
    s.add_argument("-n", "--n-stations", type=int, default=SectionConfig.n_stations)
    s.add_argument("--force", action="store_true", help="overwrite an existing config file")
    s.add_argument("--no-auto-water", action="store_true",
                   help="do not outline the water automatically; click it (or skip it) by hand")
    _add_scale_options(s)
    s.set_defaults(func=cmd_pick)

    s = sub.add_parser("run", parents=[common],
                       help="run the analysis described by a config file")
    s.add_argument("config")
    s.add_argument("--frame-step", type=int, help="override video.frame_step")
    s.add_argument("--duration", type=float, help="override video.duration_s")
    s.add_argument("-n", "--n-stations", type=int,
                   help="override section.n_stations (for every cross-section)")
    s.add_argument("--out-dir", help="override output.out_dir")
    s.add_argument("--no-stabilise", action="store_true",
                   help="skip camera-motion compensation (fixed cameras only)")
    s.add_argument("--no-figures", action="store_true", help="tables only, no images")
    s.add_argument("--pdf", action="store_true",
                   help="also write report.pdf: the summary, figures, station table and config "
                        "(output.pdf_report)")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("cameras", parents=[common],
                       help="list the DJI drones --camera knows, with their fields of view")
    s.set_defaults(func=cmd_cameras)

    s = sub.add_parser("demo", parents=[common], help="render a synthetic river and validate against it")
    s.add_argument("--out-dir", default="stiv_demo")
    s.add_argument("--duration", type=float, default=DEMO_DURATION_S)
    s.add_argument("--v-max", type=float, default=DEMO_V_MAX_MS, help="peak surface velocity, m/s")
    s.add_argument("--noise", type=float, default=DEMO_NOISE)
    s.add_argument("--drift-px", type=float, default=DEMO_DRIFT_PX,
                   help="simulated drone drift in pixels (0 for a fixed camera)")
    s.add_argument("-n", "--n-stations", type=int, default=DEMO_N_STATIONS)
    s.add_argument("--pdf", action="store_true", help="also write report.pdf")
    s.set_defaults(func=cmd_demo)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(1 if args.verbose else (-1 if args.quiet else 0))
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
