"""Tests for the DJI camera presets and the DJI flight-log (.SRT) reader."""

from __future__ import annotations

import math

import pytest

from stiv.cameras import DJI_CAMERAS, camera, hfov_from_diagonal
from stiv.config import RectifyConfig
from stiv.srt import altitude_stats, find_srt, parse_srt, parse_srt_text


# --------------------------------------------------------------------------- #
class TestCameras:
    def test_diagonal_fov_of_a_35mm_frame_converts_to_its_width(self):
        # A 24 mm lens on a 36 x 24 mm frame: 84.1 deg diagonal, 73.7 deg across.
        dfov = math.degrees(2 * math.atan(math.hypot(36, 24) / 2 / 24))
        assert hfov_from_diagonal(dfov, (3, 2)) == pytest.approx(
            math.degrees(2 * math.atan(18 / 24)), abs=1e-9)

    def test_quoted_fov_is_not_the_horizontal_one(self):
        """DJI's 84 deg is diagonal; read as horizontal it inflates every
        velocity by about a quarter."""
        mini5 = camera("mini-5-pro")
        assert mini5.hfov_deg == pytest.approx(71.5, abs=0.1)
        wrong = RectifyConfig(altitude_m=50, hfov_deg=84.0).resolve_m_per_px(3840)
        right = RectifyConfig(altitude_m=50, hfov_deg=mini5.hfov_deg).resolve_m_per_px(3840)
        assert wrong / right == pytest.approx(1.25, abs=0.01)

    def test_air_2s_matches_its_3_2_sensor(self):
        assert camera("air-2s").hfov_deg == pytest.approx(77.6, abs=0.1)

    @pytest.mark.parametrize("key", ["air-2s", "mavic-pro", "mavic-2-pro", "mavic-3",
                                     "mini-3-pro", "mini-4-pro", "mini-5-pro"])
    def test_requested_models_are_listed(self, key):
        assert camera(key) is not None

    def test_every_preset_is_sane(self):
        keys = [p.key for p in DJI_CAMERAS]
        assert len(keys) == len(set(keys))
        for p in DJI_CAMERAS:
            assert 40 < p.hfov_deg < p.dfov_deg < 120, p
            assert "horizontal" in p.describe()

    def test_unknown_key_and_bad_input(self):
        assert camera("not-a-drone") is None
        with pytest.raises(ValueError):
            hfov_from_diagonal(180, (4, 3))
        with pytest.raises(ValueError):
            hfov_from_diagonal(80, (4, 0))


# --------------------------------------------------------------------------- #
MODERN = """\
1
00:00:00,000 --> 00:00:00,033
<font size="28">FrameCnt: 1, DiffTime: 33ms
2024-05-12 10:31:23.456
[iso: 100] [shutter: 1/640.0] [fnum: 1.7] [ev: 0] [color_md: default] [focal_len: 24.00] [latitude: 54.775] [longitude: -1.575] [rel_alt: 30.100 abs_alt: 154.200] [ct: 5500] </font>

2
00:00:05,000 --> 00:00:05,033
<font size="28">FrameCnt: 151, DiffTime: 33ms
[iso: 100] [focal_len: 24.00] [rel_alt: 30.500 abs_alt: 154.600] </font>

3
00:00:20,000 --> 00:00:20,033
<font size="28">FrameCnt: 601, DiffTime: 33ms
[iso: 100] [focal_len: 48.00] [rel_alt: 45.000 abs_alt: 169.100] </font>
"""

AIR_2S = """\
1
00:00:00,000 --> 00:00:00,033
<font size="36">SrtCnt : 1, DiffTime : 33ms
[iso : 100] [shutter : 1/640.0] [fnum : 280] [ev : 0] [ct : 5500] [color_md : default] [focal_len : 220] [latitude: 54.1] [longitude: -1.2] [rel_alt: 29.900 abs_alt: 120.000] </font>
"""

MAVIC_PRO = """\
1
00:00:00,000 --> 00:00:01,000
F/2.2, SS 320, ISO 100, EV 0, GPS (-1.5750, 54.7750, 19), D 30.12m, H 29.70m, H.S 0.00m/s, V.S 0.00m/s
"""

PHANTOM = """\
1
00:00:00,000 --> 00:00:01,000
HOME(-1.5750,54.7750) 2018.04.19 11:43:54
GPS(-1.5750,54.7750,13) ISO:100 Shutter:60 EV:0 Fnum:2.2 BAROMETER:10.2
"""

ABSOLUTE_ONLY = """\
1
00:00:00,000 --> 00:00:00,033
[iso : 100] [latitude : 54.1] [longtitude : -1.2] [altitude: 120.5] </font>
"""


class TestSrt:
    def test_modern_format(self):
        recs = parse_srt_text(MODERN)
        assert [r.t_s for r in recs] == [0.0, 5.0, 20.0]
        assert [r.rel_alt_m for r in recs] == [30.1, 30.5, 45.0]
        assert recs[0].focal_len_mm == 24.0

    def test_stats_cover_the_analysis_window_only(self):
        recs = parse_srt_text(MODERN)
        st = altitude_stats(recs, start_s=0.0, duration_s=10.0)
        assert (st.median_m, st.min_m, st.max_m, st.n) == (pytest.approx(30.3), 30.1, 30.5, 2)
        assert st.focal_len_mm == 24.0
        # No caption inside the window: the whole log stands in.
        assert altitude_stats(recs, start_s=100.0, duration_s=5.0).n == 3

    def test_tenths_of_a_millimetre_focal_length(self):
        (rec,) = parse_srt_text(AIR_2S)
        assert rec.rel_alt_m == 29.9 and rec.focal_len_mm == 22.0

    @pytest.mark.parametrize("text,alt", [(MAVIC_PRO, 29.7), (PHANTOM, 10.2)])
    def test_older_formats(self, text, alt):
        (rec,) = parse_srt_text(text)
        assert rec.rel_alt_m == alt

    def test_absolute_altitude_is_not_mistaken_for_height_above_take_off(self):
        (rec,) = parse_srt_text(ABSOLUTE_ONLY)
        assert rec.rel_alt_m is None
        assert altitude_stats([rec]) is None

    def test_found_beside_the_video(self, tmp_path):
        video = tmp_path / "DJI_0001.MP4"
        video.write_bytes(b"")
        assert find_srt(video) is None
        (tmp_path / "DJI_0001.SRT").write_text(MODERN.replace("\n", "\r\n"), encoding="utf-8")
        srt = find_srt(video)
        assert srt is not None and len(parse_srt(srt)) == 3      # Windows line endings too
