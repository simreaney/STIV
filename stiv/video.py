"""Video access: frame iteration and the grey-level field ``g(x, t)``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np

from .config import ConfigError, VideoConfig

log = logging.getLogger(__name__)

#: Elementary streams with no container around them.  They carry no timing, so
#: whatever frame rate the decoder reports for one is its own default (FFmpeg
#: assumes 25 fps), not something the camera recorded.
_RAW_STREAM_SUFFIXES = frozenset(
    {".h264", ".264", ".h265", ".265", ".hevc", ".mjpeg", ".mjpg"}
)

#: A single frame interval may differ from the median interval by this fraction
#: before the clip is treated as variable frame rate ...
_VFR_TOLERANCE = 0.10
#: ... or by this many milliseconds, whichever is larger.  Container time bases
#: are coarse (1/600 s is common on phones), and a high-frame-rate clip would
#: otherwise trip the relative test on quantisation alone.
_VFR_MIN_JITTER_MS = 2.0
#: The mean frame interval may differ from ``step / fps`` by this fraction
#: before the frame rate the container reports is called into question.  Every
#: velocity scales with the frame rate, so this is also the bias it would cause.
_FPS_MISMATCH_TOLERANCE = 0.01
#: A clip that ends more than this fraction of its expected length early (and
#: by more than one frame) is reported.  Containers that estimate their frame
#: count are routinely out by a frame or so, which costs nothing; losing more
#: than this usually means a corrupt frame or a truncated file.
_TRUNCATION_TOLERANCE = 0.02


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
        fps = f"{self.fps:.3f} fps" if self.fps > 0 else "unknown fps"
        frames = f"{self.n_frames_total} frames" if self.n_frames_total > 0 else "unknown frame count"
        return (
            f"{Path(self.path).name}: {self.width}x{self.height}, "
            f"{fps}, {frames} ({self.duration_s:.1f} s)"
        )


def probe(path: str | Path) -> VideoInfo:
    """Read basic properties without decoding the whole file.

    A frame rate or frame count the container does not record comes back as
    ``0``, meaning *unknown*.  Neither is ever guessed here: every velocity is
    proportional to the frame rate, so a plausible-looking default would turn
    straight into a plausible-looking wrong answer.
    """
    path = str(path)
    if not Path(path).is_file():
        raise ConfigError(f"video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ConfigError(f"could not open video (unsupported codec?): {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 0:
        log.warning("%s does not record a frame rate", Path(path).name)
        fps = 0.0
    # Raw streams report nonsense here (FFmpeg gives a large negative number).
    n_frames = int(n) if np.isfinite(n) and n > 0 else 0
    duration = n_frames / fps if n_frames > 0 and fps > 0 else 0.0
    return VideoInfo(path, fps, n_frames, w, h, duration)


class FrameSource:
    """Sequential reader that yields the frames selected by :class:`VideoConfig`.

    Decoding is strictly sequential -- random seeking in long-GOP drone footage
    (H.264/H.265) is both slow and, with some builds of FFmpeg, inaccurate --
    so unwanted frames are grabbed and discarded rather than sought past.

    ``n_frames`` is the number of frames that will be yielded, or ``None`` when
    that cannot be known in advance (a raw stream with no frame count, read to
    its end).  After a complete pass it holds the number actually yielded.
    ``warnings`` collects what the pass found wrong with the footage itself.
    """

    def __init__(self, cfg: VideoConfig):
        self.cfg = cfg
        self.info = probe(cfg.path)
        self.fps = float(cfg.fps_override or self.info.fps)
        if not self.fps > 0:
            raise ConfigError(
                f"could not determine the frame rate of {Path(cfg.path).name}: the "
                f"file does not record one. Set video.fps_override to the rate the "
                f"footage was recorded at -- every velocity is proportional to it."
            )
        self._init_warnings: list[str] = []
        self._pass_warnings: list[str] = []
        #: Container timestamp of every yielded frame, ms, from the last pass.
        self.timestamps_ms: list[float] = []

        if cfg.fps_override is None and Path(cfg.path).suffix.lower() in _RAW_STREAM_SUFFIXES:
            self._init_warnings.append(
                f"{Path(cfg.path).name} is a raw stream with no container timing, so "
                f"its reported {self.fps:.3g} fps is probably the decoder's default "
                f"rather than the recording rate; set video.fps_override"
            )

        # start_s is a time on the file's own clock, the one the pickers seek
        # by (read_frame_at), so the analysis starts on the frame they showed
        # even when fps_override says the footage ran at another rate.
        self.start_index = max(0, int(round(cfg.start_s * (self.info.fps or self.fps))))
        self.step = max(1, int(cfg.frame_step))

        # Frames available after start_s, if the container says.
        total = self.info.n_frames_total
        available = total - self.start_index if total > 0 else None
        if available is not None and available <= 0:
            raise ConfigError(
                f"video.start_s = {cfg.start_s} s is beyond the end of {Path(cfg.path).name} "
                f"({self.info.duration_s:.1f} s)"
            )
        if cfg.duration_s is not None:
            span: Optional[int] = int(round(cfg.duration_s * self.fps))
            if available is not None and span > available:
                log.warning(
                    "video.duration_s = %.1f s runs past the end of the video; "
                    "using the %.1f s available", cfg.duration_s, available / self.fps,
                )
                span = available
        else:
            span = available

        if span is None:
            n_frames = cfg.max_frames
        else:
            n_avail = max(0, span) // self.step
            n_frames = int(min(n_avail, cfg.max_frames or n_avail))
        self.n_frames: Optional[int] = None if n_frames is None else int(n_frames)
        #: True when ``n_frames`` is a count the file promised, so that falling
        #: short of it means something went wrong; False when it is only a cap.
        self._count_expected = span is not None

        if self.n_frames is not None and self.n_frames < 2:
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
    def shape(self) -> tuple[int, int]:
        return self.info.shape

    @property
    def warnings(self) -> list[str]:
        """Problems with the footage: set-up ones, plus those the last pass found."""
        return self._init_warnings + self._pass_warnings

    # ------------------------------------------------------------------ #
    def to_grey(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Collapse a BGR frame to the scalar brightness field ``g``.

        The plain conversion is left as ``uint8``.  Promoting it to float here
        would cost a full-frame copy on every frame and buy nothing: the STI
        sampler converts the handful of pixels it actually reads, and the
        stabiliser wants bytes anyway.  Custom ``channel_weights`` can leave the
        0-255 range (a colour-difference channel goes negative), so those stay
        float and the stabiliser scales them once, against the reference frame.
        """
        if self.channel_weights is None:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return (frame_bgr.astype(np.float32) * self.channel_weights[None, None, :]).sum(axis=2)

    def frames(self, colour: bool = False) -> Iterator[np.ndarray]:
        """Yield the selected frames in order, as grey (default) or BGR.

        Once the pass is complete ``n_frames`` holds the number yielded, and
        ``warnings`` says if the video ended early or its timestamps show it
        does not run at the constant rate ``St = frame_step / fps`` assumes.
        """
        cap = cv2.VideoCapture(self.cfg.path)
        if not cap.isOpened():
            raise ConfigError(f"could not open video: {self.cfg.path}")
        self._pass_warnings = []
        self.timestamps_ms = []
        limit = self.n_frames
        emitted = 0
        try:
            for _ in range(self.start_index):
                if not cap.grab():
                    raise ConfigError(
                        f"video ended while skipping to start_s={self.cfg.start_s}"
                    )
            while limit is None or emitted < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                self.timestamps_ms.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
                yield frame if colour else self.to_grey(frame)
                emitted += 1
                if limit is not None and emitted >= limit:
                    break
                # all() stops at the first failed grab: the video has ended.
                if not all(cap.grab() for _ in range(self.step - 1)):
                    break
        finally:
            cap.release()

        # Only reached when the caller consumed the whole pass.
        if limit is not None and self._count_expected and emitted < limit:
            msg = (
                f"the video ended after {emitted} of the {limit} frames expected "
                f"({emitted * self.st_s_per_px:.1f} of {limit * self.st_s_per_px:.1f} s); "
                f"a corrupt frame, a truncated file or a video.duration_s longer than "
                f"the video stops the read early"
            )
            if limit - emitted > max(1.0, _TRUNCATION_TOLERANCE * limit):
                self._pass_warnings.append(msg)
            else:
                log.info(msg)
        self.n_frames = emitted
        self._pass_warnings.extend(
            check_frame_timing(
                self.timestamps_ms,
                self.step,
                self.info.fps,
                fps_overridden=self.cfg.fps_override is not None,
            )
        )


def check_frame_timing(
    timestamps_ms: Sequence[float],
    step: int,
    container_fps: float,
    fps_overridden: bool = False,
) -> list[str]:
    """Check that frames really are ``step / fps`` apart, from their timestamps.

    STIV turns one STI row into ``St = frame_step / fps`` seconds, a constant.
    Phones in particular record *variable* frame rate, and then each row spans
    a different time and a streak is no longer straight.  Nothing downstream
    can see that, so it is checked here, against the container's own clock.

    Two things are tested: that no single interval strays from the median by
    more than ``_VFR_TOLERANCE`` (variable frame rate, or dropped frames), and,
    unless ``video.fps_override`` was set, that the mean interval agrees with
    the frame rate the container reports to within ``_FPS_MISMATCH_TOLERANCE``.
    With an override the container's clock is by definition not the recording
    clock -- slow-motion footage, say -- so only the regularity test applies.

    Resampling the rows onto a uniform time grid is deliberately not attempted:
    interpolating between two frames does not move texture, it double-exposes
    it, which smears the very streaks being measured.

    Timestamps that are missing, non-finite or not increasing (some backends do
    not supply them) are taken as "cannot tell", and nothing is reported.
    """
    t = np.asarray(timestamps_ms, dtype=np.float64)
    if t.size < 3 or not np.isfinite(t).all():
        return []
    dt = np.diff(t)
    if not (dt > 0).all():
        log.debug("frame timestamps are not increasing; frame timing not checked")
        return []

    out = []
    med = float(np.median(dt))
    allowed = max(_VFR_TOLERANCE * med, _VFR_MIN_JITTER_MS)
    irregular = np.abs(dt - med) > allowed
    if irregular.any():
        out.append(
            f"the video's frame timestamps are irregular: {int(irregular.sum())} of "
            f"{dt.size} frame intervals differ from the median {med:.2f} ms by more "
            f"than {allowed:.2f} ms (range {dt.min():.2f}-{dt.max():.2f} ms). STIV "
            f"assumes a constant frame rate, so this is variable-frame-rate footage "
            f"or it has dropped frames; re-encode it to a constant frame rate (or "
            f"record with a fixed frame rate) before analysing it"
        )

    if not fps_overridden and container_fps > 0:
        nominal = step * 1000.0 / container_fps
        mean = float(dt.mean())
        if abs(mean / nominal - 1.0) > _FPS_MISMATCH_TOLERANCE:
            actual = step * 1000.0 / mean
            out.append(
                f"the video reports {container_fps:.3f} fps but its frame timestamps "
                f"average {actual:.3f} fps; velocities scale with the frame rate, so "
                f"they would be out by {100.0 * (container_fps / actual - 1.0):+.1f}%. "
                f"Set video.fps_override to the true recording rate"
            )
    return out


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
