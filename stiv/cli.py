"""Command line interface.

::

    stiv info      flight.mp4              inspect a video
    stiv init      flight.mp4 -o survey.yaml   write a starter config
    stiv pick      flight.mp4 -o survey.yaml   click the section on a frame
    stiv run       survey.yaml             analyse and write the outputs
    stiv demo                              render a synthetic river and analyse it
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import __version__
from .config import (
    FRAME_STEP_SPEEDS_MS,
    Config,
    ConfigError,
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
def cmd_info(args) -> int:
    from .video import probe

    info = probe(args.video)
    # probe() reports 0 when the file records no frame rate (a raw .h264 or
    # .mjpeg stream, say) rather than guessing one.
    fps_known = info.fps > 0
    print(info)
    print()
    print("Suggested settings")
    print("------------------")
    if not fps_known:
        print("  video.fps_override    ?        (the file records no frame rate: set this "
              "to the rate the footage was recorded at -- every velocity is "
              "proportional to it)")
    print(f"  video.duration_s      {default_duration_s(info.duration_s):.1f}   "
          f"(20-40 s of steady hover is plenty; the paper used 30 s)")
    if args.altitude_m and args.hfov_deg:
        import math

        gw = 2 * args.altitude_m * math.tan(math.radians(args.hfov_deg) / 2)
        gsd = gw / info.width
        print(f"  rectify.m_per_px      {gsd:.4f}   "
              f"(from {args.altitude_m} m altitude, {args.hfov_deg} deg HFOV; "
              f"ground width {gw:.1f} m)")
        if fps_known:
            for v in FRAME_STEP_SPEEDS_MS:
                print(f"      if U ~ {v:>3.1f} m/s  ->  video.frame_step: "
                      f"{suggest_frame_step(gsd, info.fps, v)}   (puts tan phi near 1)")
        else:
            # Without a frame rate every suggestion would collapse to step 1.
            print("      (no frame_step suggestion without a frame rate; re-run once "
                  "you know it, or work it out as m_per_px * fps / U)")
    else:
        print("  pass --altitude-m and --hfov-deg to get a ground sample distance "
              "and a frame_step suggestion")
    return 0


_INIT_HEADER = """\
# STIV survey configuration (written by `stiv init`)
#
# `stiv run` refuses this file until the null placeholders are filled in:
#
#   rectify.m_per_px    ground sample distance, metres per pixel.  Or replace
#                       it with the flight geometry:
#                         altitude_m: 60.0   # above the WATER SURFACE
#                         hfov_deg: 84.0     # your camera's horizontal FOV
#   section.start_px    the two ends of the cross-section, in image pixels
#   section.end_px      (`stiv pick` lets you click them on a frame instead)
#
# flow.downstream_px is optional: any image point downstream of the section,
# which fixes the sign of the flow direction.
#
# Video: {info}

"""


def cmd_init(args) -> int:
    from .video import probe

    out = Path(args.output)
    _refuse_overwrite(out, args.force)
    info = probe(args.video)
    raw = survey_dict(
        path_for_config(args.video, out),
        duration_s=default_duration_s(info.duration_s),
        name=Path(args.video).stem,
        out_dir=_default_out_dir(out),
    )
    out.write_text(_INIT_HEADER.format(info=info) + yaml_text(raw), encoding="utf-8")
    print(f"wrote {out}")
    print("Next: fill in the scale and the section coordinates, then "
          f"`stiv run {out}`")
    return 0


def cmd_pick(args) -> int:
    from .picker import build_config_interactively

    out = Path(args.output)
    # Checked before any clicking, not after it.
    _refuse_overwrite(out, args.force)
    raw = build_config_interactively(
        args.video,
        out_dir=_default_out_dir(out),
        t_s=args.time,
        with_gcps=args.gcps,
        n_stations=args.n_stations,
        config_path=out,
    )
    header = "# STIV survey configuration (written by `stiv pick`)\n#\n"
    if args.gcps:
        header += ("# Fill in rectify.gcp_world -- the surveyed position of each control\n"
                   "# point in metres, in the order they were clicked -- before running.\n\n")
    else:
        header += ("# Fill in rectify.m_per_px (metres per pixel), or replace it with\n"
                   "# altitude_m plus hfov_deg, before running.\n\n")
    out.write_text(header + yaml_text(raw), encoding="utf-8")
    print(f"wrote {out}")
    what = "rectify.gcp_world" if args.gcps else "rectify.m_per_px"
    print(f"Fill in {what}, then `stiv run {out}`")
    return 0


def cmd_run(args) -> int:
    from .analysis import run as run_analysis

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
    if args.no_stabilise:
        cfg.stabilise.enabled = False
    if args.no_figures:
        cfg.output.save_diagnostics = False
        cfg.output.save_sti_images = False
    cfg.validate()

    result = run_analysis(cfg, progress=not args.quiet)
    out = _write_outputs(result, cfg)

    print()
    print(result.summary())
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
    s.add_argument("--altitude-m", type=float, help="flight altitude above the water")
    s.add_argument("--hfov-deg", type=float, help="camera horizontal field of view")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("init", parents=[common], help="write a starter config file")
    s.add_argument("video")
    s.add_argument("-o", "--output", default="survey.yaml")
    s.add_argument("--force", action="store_true", help="overwrite an existing config file")
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
    s.add_argument("-n", "--n-stations", type=int, default=SectionConfig.n_stations)
    s.add_argument("--force", action="store_true", help="overwrite an existing config file")
    s.set_defaults(func=cmd_pick)

    s = sub.add_parser("run", parents=[common],
                       help="run the analysis described by a config file")
    s.add_argument("config")
    s.add_argument("--frame-step", type=int, help="override video.frame_step")
    s.add_argument("--duration", type=float, help="override video.duration_s")
    s.add_argument("-n", "--n-stations", type=int, help="override section.n_stations")
    s.add_argument("--out-dir", help="override output.out_dir")
    s.add_argument("--no-stabilise", action="store_true",
                   help="skip camera-motion compensation (fixed cameras only)")
    s.add_argument("--no-figures", action="store_true", help="tables only, no images")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("demo", parents=[common], help="render a synthetic river and validate against it")
    s.add_argument("--out-dir", default="stiv_demo")
    s.add_argument("--duration", type=float, default=DEMO_DURATION_S)
    s.add_argument("--v-max", type=float, default=DEMO_V_MAX_MS, help="peak surface velocity, m/s")
    s.add_argument("--noise", type=float, default=DEMO_NOISE)
    s.add_argument("--drift-px", type=float, default=DEMO_DRIFT_PX,
                   help="simulated drone drift in pixels (0 for a fixed camera)")
    s.add_argument("-n", "--n-stations", type=int, default=DEMO_N_STATIONS)
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
