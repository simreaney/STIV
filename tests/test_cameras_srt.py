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

#: The Air 2S's own format: a plain ``altitude``, height above take-off on
#: this model.  (Shaped on a real log; the position is made up.)
AIR_2S_PLAIN = """\
1
00:00:00,000 --> 00:00:00,033
<font size="36">SrtCnt : 1, DiffTime : 33ms
2023-08-12 10:52:30,115,185
[iso : 100] [shutter : 1/320.0] [fnum : 280] [ev : 0] [ct : 5881] [color_md : default] [focal_len : 224] [latitude: 54.775000] [longitude: -1.575000] [altitude: 35.900000]
</font>

2
00:00:00,033 --> 00:00:00,066
<font size="36">SrtCnt : 2, DiffTime : 33ms
[iso : 100] [focal_len : 224] [latitude: 54.775000] [longitude: -1.575000] [altitude: 35.800000]
</font>
"""

#: The same field, reading higher than any drone may fly above its take-off.
ABOVE_SEA_LEVEL = """\
1
00:00:00,000 --> 00:00:00,033
[iso : 100] [latitude : 46.5] [longtitude : 7.9] [altitude: 812.5] </font>
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

    def test_plain_altitude_is_used_but_marked_as_assumed(self):
        """The Air 2S logs no rel_alt, only a plain altitude -- which on that
        model is height above take-off, so it is used, flagged for checking."""
        recs = parse_srt_text(AIR_2S_PLAIN)
        assert [r.rel_alt_m for r in recs] == [None, None]
        assert [r.altitude_m for r in recs] == [35.9, 35.8]
        st = altitude_stats(recs)
        assert (st.median_m, st.min_m, st.max_m, st.assumed) == (pytest.approx(35.85), 35.8, 35.9, True)
        assert st.focal_len_mm == 22.4

    def test_a_stated_height_above_take_off_wins_and_is_not_assumed(self):
        (rec,) = parse_srt_text(MODERN.split("\n\n")[0].replace("] </font>", "] [altitude: 99.0] </font>"))
        assert rec.rel_alt_m == 30.1 and rec.altitude_m == 99.0
        st = altitude_stats([rec])
        assert st.median_m == 30.1 and not st.assumed

    def test_altitude_above_sea_level_is_not_mistaken_for_height_above_take_off(self):
        (rec,) = parse_srt_text(ABOVE_SEA_LEVEL)
        assert rec.altitude_m == 812.5
        assert altitude_stats([rec]) is None

    def test_found_whatever_the_case_of_its_name(self, tmp_path):
        """Uploads and copies can change a name's case; on Linux, Colab
        included, FLIGHT.MP4 and flight.SRT are otherwise strangers."""
        video = tmp_path / "FLIGHT.MP4"
        video.write_bytes(b"")
        (tmp_path / "flight.SRT").write_text(MODERN, encoding="utf-8")
        found = find_srt(video)
        assert found is not None and found.name.lower() == "flight.srt"
        assert find_srt(tmp_path / "OTHER.MP4") is None

    def test_found_beside_the_video(self, tmp_path):
        video = tmp_path / "DJI_0001.MP4"
        video.write_bytes(b"")
        assert find_srt(video) is None
        (tmp_path / "DJI_0001.SRT").write_text(MODERN.replace("\n", "\r\n"), encoding="utf-8")
        srt = find_srt(video)
        assert srt is not None and len(parse_srt(srt)) == 3      # Windows line endings too


# --------------------------------------------------------------------------- #
class TestFocalLengthHint:
    """Zoom only lengthens the logged focal length, so a log can rule drones out."""

    @pytest.mark.parametrize("focal", [22.0, 22.4])
    def test_only_the_air_2s_lens_is_that_wide(self, focal):
        from stiv.cameras import preset_for_focal_length

        assert preset_for_focal_length(focal) is camera("air-2s")

    @pytest.mark.parametrize("focal", [24.0, 23.0, 26.0, 28.0, 44.8, 12.0, None, 0.0])
    def test_anything_else_points_nowhere(self, focal):
        """24 mm fits nearly every preset; 44.8 mm is a zoomed Air 2S or a
        zoomed anything; 12 mm is a lens wider than any preset's."""
        from stiv.cameras import preset_for_focal_length

        assert preset_for_focal_length(focal) is None

    def test_a_lens_cannot_log_wider_than_itself(self):
        mini4 = camera("mini-4-pro")
        assert not mini4.can_log_focal_length(22.4)
        assert mini4.can_log_focal_length(23.9) and mini4.can_log_focal_length(48.0)
        assert camera("air-2s").can_log_focal_length(22.4)

    def test_shortest_focal_length_in_a_log(self):
        from stiv.srt import shortest_focal_len

        assert shortest_focal_len(parse_srt_text(MODERN)) == 24.0     # 24, 24, then 48 zoomed
        assert shortest_focal_len(parse_srt_text(AIR_2S)) == 22.0     # logged as tenths: 220
        assert shortest_focal_len(parse_srt_text(MAVIC_PRO)) is None
        assert shortest_focal_len([]) is None
