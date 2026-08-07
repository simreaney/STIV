"""Video access: frame iteration and the grey-level field ``g(x, t)``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import ConfigError, VideoConfig


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames_total: int
    width: int
    height: int
    duration_s: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def __str__(self) -> str:
        return (
            f"{Path(self.path).name}: {self.width}x{self.height}, "
            f"{self.fps:.3f} fps, {self.n_frames_total} frames "
            f"({self.duration_s:.1f} s)"
        )


def probe(path: str | Path) -> VideoInfo:
    """Read basic properties without decoding the whole file."""
    path = str(path)
    if not Path(path).is_file():
        raise ConfigError(f"video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ConfigError(f"could not open video (unsupported codec?): {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    return VideoInfo(path, fps, max(n, 0), w, h, (n / fps) if n > 0 and fps > 0 else 0.0)


class FrameSource:
    """Sequential reader that yields the frames selected by :class:`VideoConfig`.

    Decoding is strictly sequential -- random seeking in long-GOP drone footage
    (H.264/H.265) is both slow and, with some builds of FFmpeg, inaccurate --
    so unwanted frames are grabbed and discarded rather than sought past.
    """

    def __init__(self, cfg: VideoConfig):
        self.cfg = cfg
        self.info = probe(cfg.path)
        self.fps = float(cfg.fps_override or self.info.fps)
        if self.fps <= 0:
            raise ConfigError("could not determine frame rate; set video.fps_override")

        self.start_index = max(0, int(round(cfg.start_s * self.fps)))
        self.step = max(1, int(cfg.frame_step))

        if cfg.duration_s is not None:
            span = int(round(cfg.duration_s * self.fps))
        elif self.info.n_frames_total > 0:
            span = self.info.n_frames_total - self.start_index
        else:
            span = 10**9
        n_avail = max(0, span) // self.step
        self.n_frames = int(min(n_avail, cfg.max_frames or n_avail))

        if self.n_frames < 2:
            raise ConfigError(
                f"only {self.n_frames} frame(s) selected from {cfg.path}; check "
                f"video.start_s / duration_s / frame_step"
            )

        w = cfg.channel_weights
        if w is not None:
            w = np.asarray(w, dtype=np.float64)
            if w.shape != (3,):
                raise ConfigError("video.channel_weights must be 3 numbers, in BGR order")
            if abs(w.sum()) < 1e-9:
                raise ConfigError("video.channel_weights must not sum to zero")
            w = w / w.sum()
        self.channel_weights = w

    # ------------------------------------------------------------------ #
    @property
    def st_s_per_px(self) -> float:
        """``S_t`` of Eq. (16): seconds represented by one STI row."""
        return self.step / self.fps

    @property
    def duration_s(self) -> float:
        return self.n_frames * self.st_s_per_px

    @property
    def shape(self) -> tuple[int, int]:
        return self.info.shape

    # ------------------------------------------------------------------ #
    def to_grey(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Collapse a BGR frame to the scalar brightness field ``g``."""
        if self.channel_weights is None:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return (frame_bgr.astype(np.float32) * self.channel_weights[None, None, :]).sum(axis=2)

    def frames(self, colour: bool = False) -> Iterator[np.ndarray]:
        """Yield the selected frames in order, as grey (default) or BGR."""
        cap = cv2.VideoCapture(self.cfg.path)
        if not cap.isOpened():
            raise ConfigError(f"could not open video: {self.cfg.path}")
        try:
            for _ in range(self.start_index):
                if not cap.grab():
                    raise ConfigError(
                        f"video ended while skipping to start_s={self.cfg.start_s}"
                    )
            emitted = 0
            while emitted < self.n_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame if colour else self.to_grey(frame)
                emitted += 1
                if emitted < self.n_frames:
                    for _ in range(self.step - 1):
                        if not cap.grab():
                            return
            self.n_frames = emitted if emitted < self.n_frames else self.n_frames
        finally:
            cap.release()

    def first_frame(self, colour: bool = True) -> np.ndarray:
        """Decode just the first selected frame, for previews and picking."""
        for f in self.frames(colour=colour):
            return f
        raise ConfigError(f"could not read any frame from {self.cfg.path}")


def read_frame_at(path: str | Path, t_s: float = 0.0) -> np.ndarray:
    """Grab a single BGR frame near ``t_s`` seconds; used by the point picker."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ConfigError(f"could not open video: {path}")
    try:
        if t_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            raise ConfigError(f"could not read a frame from {path}")
        return frame
    finally:
        cap.release()
