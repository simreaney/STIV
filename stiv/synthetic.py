"""Synthetic drone footage with a known velocity field.

The only honest way to check a velocimetry pipeline end to end is to feed it a
river whose velocity you already know.  This module renders a nadir view of a
straight channel whose surface texture is advected by a prescribed transverse
profile, optionally with drone motion, noise and static bank features, and
writes it as a video the CLI can analyse like any other clip.

Because the advection is steady and purely streamwise, the frame at time ``t``
is an exact resampling of one master texture,

.. math::  I(x, y, t) = T(x - u(y) t / s, y)

with ``s`` the ground sample distance -- so there is no integration error in
the ground truth, only the resampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class SyntheticRiver:
    """Description of the scene to render.

    The channel runs left-to-right across the frame; the cross-section is the
    vertical centre line.  ``bank_top_px`` and ``bank_bottom_px`` bound the
    water, and the velocity profile is zero at those edges.
    """

    width_px: int = 960
    height_px: int = 540
    fps: float = 30.0
    duration_s: float = 12.0
    m_per_px: float = 0.02                 # 2 cm/px, e.g. ~60 m altitude
    bank_top_px: int = 120
    bank_bottom_px: int = 420

    #: Peak surface velocity, m/s, at the channel centreline.
    v_max_ms: float = 1.2
    #: Exponent of the transverse profile ``v = v_max (1 - |2r-1|^p)``.
    profile_exponent: float = 2.0
    #: Flow direction: +1 for left-to-right, -1 for right-to-left.
    flow_sign: int = 1

    #: Correlation length of the surface texture, pixels.
    texture_scale_px: float = 6.0
    texture_contrast: float = 0.35
    #: Per-frame sensor noise as a fraction of full scale.
    noise: float = 0.02

    #: Amplitude of simulated drone drift, pixels, over the whole clip.
    drone_drift_px: float = 0.0
    #: Amplitude of simulated drone sway (sinusoidal), pixels.
    drone_sway_px: float = 0.0
    #: Peak drone yaw over the clip, degrees.
    drone_yaw_deg: float = 0.0

    #: Draw static rocks/vegetation on the banks, so a stabiliser has features.
    draw_banks: bool = True
    seed: int = 12345

    # ------------------------------------------------------------------ #
    @property
    def n_frames(self) -> int:
        return int(round(self.duration_s * self.fps))

    @property
    def channel_width_px(self) -> int:
        return self.bank_bottom_px - self.bank_top_px

    @property
    def channel_width_m(self) -> float:
        return self.channel_width_px * self.m_per_px

    def velocity_at_row(self, rows: np.ndarray) -> np.ndarray:
        """True surface velocity, m/s, at each image row."""
        r = (np.asarray(rows, dtype=float) - self.bank_top_px) / max(self.channel_width_px, 1)
        v = self.v_max_ms * (1.0 - np.abs(2.0 * r - 1.0) ** self.profile_exponent)
        v = np.where((r >= 0.0) & (r <= 1.0), v, 0.0)
        return self.flow_sign * np.maximum(v, 0.0)

    def truth_profile(self, n_stations: int, edge_trim: float = 0.0):
        """Ground truth sampled the way :mod:`stiv.analysis` samples a section.

        Returns ``(station_m, velocity_ms)`` with station measured from the top
        bank downwards, matching a cross-section drawn top-to-bottom.
        """
        fr = np.linspace(edge_trim, 1.0 - edge_trim, n_stations)
        rows = self.bank_top_px + fr * self.channel_width_px
        return fr * self.channel_width_m, np.abs(self.velocity_at_row(rows))


def _make_texture(cfg: SyntheticRiver, pad_px: int, rng: np.random.Generator) -> np.ndarray:
    """Band-limited random field standing in for wind ripples and boils."""
    h = cfg.height_px
    w = cfg.width_px + 2 * pad_px
    field_ = gaussian_filter(rng.standard_normal((h, w)), sigma=cfg.texture_scale_px)
    field_ /= max(field_.std(), 1e-9)
    # A second, larger scale mimics boil vortices convecting with the surface.
    coarse = gaussian_filter(rng.standard_normal((h, w)), sigma=cfg.texture_scale_px * 4)
    coarse /= max(coarse.std(), 1e-9)
    return 0.75 * field_ + 0.25 * coarse


def _bank_appearance(cfg: SyntheticRiver, rng: np.random.Generator) -> np.ndarray:
    """Static background: banks with rocks and vegetation for the stabiliser."""
    img = np.zeros((cfg.height_px, cfg.width_px, 3), np.float32)
    img[:, :] = (0.32, 0.38, 0.30)

    texture = gaussian_filter(rng.standard_normal((cfg.height_px, cfg.width_px)), 2.0)
    texture /= max(texture.std(), 1e-9)
    img += (0.10 * texture)[:, :, None]

    if cfg.draw_banks:
        for _ in range(260):
            y = rng.integers(0, cfg.height_px)
            if cfg.bank_top_px - 10 < y < cfg.bank_bottom_px + 10:
                continue
            x = int(rng.integers(0, cfg.width_px))
            r = int(rng.integers(2, 7))
            shade = float(rng.uniform(0.15, 0.85))
            cv2.circle(img, (x, int(y)), r, (shade, shade * 0.95, shade * 0.8), -1)

    return np.clip(img, 0, 1)


def render(cfg: SyntheticRiver, out_path: str | Path, progress: bool = True) -> Path:
    """Render the scene to an MP4 (falls back to MJPG/AVI if needed)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    max_shift_px = abs(cfg.v_max_ms) * cfg.duration_s / cfg.m_per_px
    pad = int(math.ceil(max_shift_px)) + 32
    texture = _make_texture(cfg, pad, rng)
    background = _bank_appearance(cfg, rng)

    rows = np.arange(cfg.height_px)
    vel_px_per_frame = cfg.velocity_at_row(rows) / cfg.m_per_px / cfg.fps

    water = (rows >= cfg.bank_top_px) & (rows <= cfg.bank_bottom_px)
    water_mask = water[:, None].astype(np.float32)

    base_x = np.arange(cfg.width_px, dtype=np.float32)[None, :] + pad
    grid_y = np.repeat(rows.astype(np.float32)[:, None], cfg.width_px, axis=1)

    writer, path = _open_writer(out_path, cfg.fps, cfg.width_px, cfg.height_px)

    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=cfg.n_frames, unit="frame", desc="render", leave=False)
        except ImportError:
            bar = None

    try:
        for k in range(cfg.n_frames):
            shift = vel_px_per_frame[:, None] * k
            map_x = (base_x - shift).astype(np.float32)
            map_x = np.mod(map_x, texture.shape[1]).astype(np.float32)

            surf = cv2.remap(
                texture.astype(np.float32), map_x, grid_y,
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP,
            )
            water_rgb = 0.30 + cfg.texture_contrast * surf
            frame = background.copy()
            tinted = np.stack([water_rgb * 1.05, water_rgb * 0.98, water_rgb * 0.85], axis=2)
            frame = frame * (1 - water_mask[:, :, None]) + tinted * water_mask[:, :, None]

            if cfg.noise:
                frame = frame + rng.normal(0.0, cfg.noise, frame.shape)

            frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            frame_u8 = _apply_drone_motion(cfg, frame_u8, k)
            writer.write(frame_u8)
            if bar is not None:
                bar.update(1)
    finally:
        writer.release()
        if bar is not None:
            bar.close()

    return path


def _apply_drone_motion(cfg: SyntheticRiver, frame: np.ndarray, k: int) -> np.ndarray:
    """Warp a rendered frame to simulate drift, sway and yaw of the aircraft."""
    if not (cfg.drone_drift_px or cfg.drone_sway_px or cfg.drone_yaw_deg):
        return frame

    f = k / max(cfg.n_frames - 1, 1)
    # Drift is deliberately diagonal so it contaminates both axes.  Every term
    # vanishes at f = 0 so that the first frame -- which the analysis adopts as
    # its reference -- coincides with the scene the ground truth is defined in.
    dx = cfg.drone_drift_px * f + cfg.drone_sway_px * math.sin(2 * math.pi * 1.3 * f)
    dy = 0.6 * cfg.drone_drift_px * f + cfg.drone_sway_px * math.sin(2 * math.pi * 0.9 * f)
    ang = cfg.drone_yaw_deg * f

    centre = (cfg.width_px / 2.0, cfg.height_px / 2.0)
    M = cv2.getRotationMatrix2D(centre, ang, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(
        frame, M, (cfg.width_px, cfg.height_px),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )


def _open_writer(path: Path, fps: float, w: int, h: int):
    """Open a VideoWriter, preferring MP4 but falling back to MJPG/AVI."""
    for fourcc, suffix in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        p = path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if writer.isOpened():
            return writer, p
        writer.release()
    raise RuntimeError(
        "OpenCV could not open any video writer (tried mp4v and MJPG); "
        "the build may lack FFmpeg support"
    )


def make_config(cfg: SyntheticRiver, video_path: str | Path, out_dir: str | Path,
                n_stations: int = 25) -> dict:
    """A ready-to-run config dict for footage produced by :func:`render`."""
    x = cfg.width_px // 2
    return {
        "video": {"path": str(video_path), "frame_step": 1},
        "rectify": {"method": "scale", "m_per_px": cfg.m_per_px},
        "section": {
            "start_px": [x, cfg.bank_top_px],
            "end_px": [x, cfg.bank_bottom_px],
            "n_stations": n_stations,
            "name": "synthetic",
        },
        # The channel runs left-to-right, so the downstream normal points at a
        # marker on the right-hand side of the frame.
        "flow": {"direction": "auto",
                 "downstream_px": [cfg.width_px - 1 if cfg.flow_sign > 0 else 0,
                                   (cfg.bank_top_px + cfg.bank_bottom_px) // 2]},
        "stabilise": {"enabled": bool(cfg.drone_drift_px or cfg.drone_sway_px or cfg.drone_yaw_deg)},
        "stiv": {
            "search_line_length_m": 3.0,
            "window_px": 31,
            "velocity_range_ms": [-0.5, 4.0],
        },
        "output": {"out_dir": str(out_dir)},
    }
