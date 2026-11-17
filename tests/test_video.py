"""Tests for frame selection and the checks on the footage itself.

Everything downstream trusts two numbers from here: which frames were read,
and how far apart in time they are (``St = frame_step / fps``).  Velocity is
proportional to the second, so a wrong frame rate is a wrong answer that looks
like a right one.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from stiv import video as video_mod
from stiv.config import ConfigError, VideoConfig
from stiv.geometry import SearchLine
from stiv.sti import StiCollector
from stiv.video import FrameSource, VideoInfo, check_frame_timing, probe, read_frame_at

W, H = 64, 48
FPS = 30.0
LEVEL = 8          # frame k is a flat grey of LEVEL * k, so its index survives MJPEG


def _frame(k: int) -> np.ndarray:
    return np.full((H, W, 3), LEVEL * k, np.uint8)


def _index_of(frame: np.ndarray) -> int:
    return int(round(float(frame.mean()) / LEVEL))


@pytest.fixture
def clip(tmp_path):
    """30 frames at 30 fps; frame k is grey level 8k."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    assert writer.isOpened()
    for k in range(30):
        writer.write(_frame(k))
    writer.release()
    return path


@pytest.fixture
def raw_stream(tmp_path):
    """A raw MJPEG elementary stream: concatenated JPEGs, no container.

    OpenCV reports a nonsense (negative) frame count for it and FFmpeg's
    default frame rate -- exactly the case of a raw .h264/.mjpeg from a drone.
    """
    path = tmp_path / "clip.mjpeg"
    with path.open("wb") as fh:
        for k in range(20):
            ok, buf = cv2.imencode(".jpg", _frame(k))
            assert ok
            fh.write(buf.tobytes())
    return path


def _read_all(src: FrameSource) -> list[int]:
    return [_index_of(f) for f in src.frames(colour=True)]


# --------------------------------------------------------------------------- #
class TestFrameSelection:
    def test_start_step_and_duration_pick_the_right_frames(self, clip):
        src = FrameSource(VideoConfig(path=str(clip), start_s=0.2, duration_s=0.5, frame_step=3))
        assert src.start_index == 6
        assert src.n_frames == 5                      # 15 frames of span, every 3rd
        assert _read_all(src) == [6, 9, 12, 15, 18]
        assert src.st_s_per_px == pytest.approx(3 / FPS)
        assert src.warnings == []

    def test_max_frames_caps_the_selection(self, clip):
        src = FrameSource(VideoConfig(path=str(clip), frame_step=2, max_frames=4))
        assert _read_all(src) == [0, 2, 4, 6]

    def test_whole_clip_by_default(self, clip):
        src = FrameSource(VideoConfig(path=str(clip)))
        assert src.n_frames == 30
        assert _read_all(src) == list(range(30))
        assert src.warnings == []

    def test_a_duration_past_the_end_is_capped_not_reported_as_truncation(self, clip):
        src = FrameSource(VideoConfig(path=str(clip), duration_s=60.0))
        assert src.n_frames == 30
        assert len(_read_all(src)) == 30
        assert src.warnings == []

    def test_start_beyond_the_end_is_rejected(self, clip):
        with pytest.raises(ConfigError, match="beyond the end"):
            FrameSource(VideoConfig(path=str(clip), start_s=5.0))


class TestFrameRate:
    def test_probe_does_not_invent_a_frame_rate(self, clip, monkeypatch):
        """A file with no frame rate used to be read as 30 fps without a word."""
        real = cv2.VideoCapture

        class NoRate:
            def __init__(self, path):
                self._cap = real(path)

            def isOpened(self):
                return self._cap.isOpened()

            def get(self, prop):
                return 0.0 if prop == cv2.CAP_PROP_FPS else self._cap.get(prop)

            def release(self):
                self._cap.release()

        monkeypatch.setattr(video_mod.cv2, "VideoCapture", NoRate)
        info = probe(clip)
        assert info.fps == 0.0
        assert "unknown fps" in str(info)

    def test_an_unknown_frame_rate_needs_an_override(self, clip, monkeypatch):
        info = VideoInfo(str(clip), 0.0, 30, W, H, 0.0)
        monkeypatch.setattr(video_mod, "probe", lambda _path: info)
        with pytest.raises(ConfigError, match="fps_override"):
            FrameSource(VideoConfig(path=str(clip)))
        src = FrameSource(VideoConfig(path=str(clip), fps_override=25.0))
        assert src.st_s_per_px == pytest.approx(1 / 25.0)

    def test_start_s_is_on_the_files_clock_under_an_override(self, clip):
        """The pickers show the frame at start_s by the file's own clock, and
        the stabiliser measures drift from the first analysed frame, so the
        two must be the same frame whatever rate the analysis assumes."""
        shown = _index_of(read_frame_at(clip, 0.3))
        src = FrameSource(VideoConfig(path=str(clip), start_s=0.3, fps_override=60.0))
        assert _read_all(src)[0] == shown == 9


class TestUnknownFrameCount:
    def test_raw_stream_is_read_to_its_end(self, raw_stream):
        info = probe(raw_stream)
        assert info.n_frames_total == 0                # unknown, not negative

        src = FrameSource(VideoConfig(path=str(raw_stream), fps_override=30.0))
        assert src.n_frames is None
        assert _read_all(src) == list(range(20))
        assert src.n_frames == 20                      # known once read
        assert src.warnings == []                      # reaching the end is no surprise

    def test_raw_stream_frame_rate_is_flagged_as_a_guess(self, raw_stream):
        src = FrameSource(VideoConfig(path=str(raw_stream)))
        assert any("raw stream" in w and "fps_override" in w for w in src.warnings)

    def test_collector_grows_instead_of_preallocating(self):
        """With no frame count the buffer was sized for 10**9 rows: terabytes."""
        n = 40
        line = SearchLine(
            index=0, station_m=0.0, centre_world=np.zeros(2), centre_px=np.zeros(2),
            samples_px=np.column_stack([np.arange(n, dtype=float), np.full(n, 5.0)]),
            samples_world=np.zeros((n, 2)), sx_m_per_px=0.02,
        )
        col = StiCollector([line], None, 1 / 30)
        assert col.buffer is None
        rows = 600                                     # spans three internal blocks
        for k in range(rows):
            col.add_frame(np.full((10, 64), k % 251, np.uint8))
        (sti,) = col.finish()
        assert sti.data.shape == (rows, n)
        assert np.array_equal(sti.data[:, 0], np.arange(rows) % 251)


class TestTruncation:
    def test_an_early_end_is_reported_and_n_frames_corrected(self, clip, monkeypatch):
        """The file promises 60 frames and holds 30.

        With ``frame_step`` > 1 the end is met while skipping, which used to
        ``return`` straight out of the generator -- leaving ``n_frames`` at the
        promised count and saying nothing.
        """
        info = probe(clip)
        info.n_frames_total = 60
        monkeypatch.setattr(video_mod, "probe", lambda _path: info)

        src = FrameSource(VideoConfig(path=str(clip), frame_step=4))
        assert src.n_frames == 15
        got = _read_all(src)
        assert got == list(range(0, 30, 4))
        assert src.n_frames == len(got) == 8
        assert any("ended after 8 of the 15 frames" in w for w in src.warnings)

    def test_one_missing_frame_is_not_worth_a_warning(self, clip, monkeypatch):
        info = probe(clip)
        info.n_frames_total = 31                       # an estimated count, one over
        monkeypatch.setattr(video_mod, "probe", lambda _path: info)
        src = FrameSource(VideoConfig(path=str(clip)))
        assert len(_read_all(src)) == 30
        assert src.n_frames == 30
        assert src.warnings == []


class TestFrameTiming:
    def test_timestamps_are_recorded_for_each_frame(self, clip):
        src = FrameSource(VideoConfig(path=str(clip), frame_step=2))
        _read_all(src)
        assert np.allclose(np.diff(src.timestamps_ms), 2000.0 / FPS, atol=0.01)
        assert src.warnings == []

    def test_constant_rate_passes(self):
        t = np.arange(50) * 1000.0 / 30.0
        assert check_frame_timing(t, 1, 30.0) == []

    def test_timebase_quantisation_is_not_variable_frame_rate(self):
        # 240 fps on a 1/600 s clock: intervals alternate 1 and 2 ticks.
        ticks = np.concatenate([[0], np.cumsum(np.tile([2, 3], 40))])
        t = ticks * 1000.0 / 600.0
        assert check_frame_timing(t, 1, 240.0) == []

    def test_a_dropped_frame_is_variable_frame_rate(self):
        t = np.arange(50) * 1000.0 / 30.0
        t[25:] += 1000.0 / 30.0                         # one frame missing
        (msg,) = [w for w in check_frame_timing(t, 1, 30.0, fps_overridden=True)]
        assert "irregular" in msg and "constant frame rate" in msg

    def test_a_drifting_rate_is_variable_frame_rate(self):
        dt = np.linspace(1000 / 30, 1000 / 24, 60)       # 30 fps sagging to 24
        t = np.concatenate([[0.0], np.cumsum(dt)])
        assert any("irregular" in w for w in check_frame_timing(t, 1, 30.0))

    def test_a_wrong_container_rate_is_reported(self):
        t = np.arange(50) * 1000.0 / 25.0               # really 25 fps
        (msg,) = check_frame_timing(t, 1, 30.0)
        assert "reports 30.000 fps" in msg and "25.000" in msg and "fps_override" in msg
        # With an override the container clock is not the recording clock.
        assert check_frame_timing(t, 1, 30.0, fps_overridden=True) == []

    def test_frame_step_is_allowed_for(self):
        t = np.arange(30) * 3 * 1000.0 / 30.0
        assert check_frame_timing(t, 3, 30.0) == []

    @pytest.mark.parametrize("t", [[], [0.0, 0.0, 0.0, 0.0], [0.0, 33.0, np.nan, 99.0]])
    def test_missing_timestamps_are_not_judged(self, t):
        assert check_frame_timing(t, 1, 30.0) == []
