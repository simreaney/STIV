"""Tests for reading which drone recorded a video from the file's metadata.

No newer-DJI clip ships with the repository, so the files here are built box
by box to the layout :mod:`stiv.videometa` describes: ``moov`` after ``mdat``,
a megabyte-scale ``udta``, and a ``djmd`` track whose first sample is the
protobuf header.
"""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from stiv.cameras import DJI_CAMERAS, camera, preset_for_name
from stiv.videometa import ENCODER_TAG, MODEL_TAG, TRACK, _names, identify_camera

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "DJI_0659.MP4"


# --------------------------------------------------------------------------- #
# MP4 and protobuf, built by hand
def box(typ: bytes, *payload: bytes) -> bytes:
    body = b"".join(payload)
    return struct.pack(">I4s", 8 + len(body), typ) + body


def full(typ: bytes, *payload: bytes) -> bytes:
    """A full box: version and flags (zero) before the payload."""
    return box(typ, bytes(4), *payload)


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def pb(number: int, value) -> bytes:
    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    raw = value.encode() if isinstance(value, str) else value
    return varint(number << 3 | 2) + varint(len(raw)) + raw


def djmd_header(protocol: str, model=None, camera_code=None, at=1) -> bytes:
    """DJI's header sample: 1-1-``at`` the protocol, 1-1-5 the aircraft serial,
    1-1-10 the model, 2-2-1-4 the camera."""
    inner = pb(at, protocol) + pb(2, 3) + pb(5, "1581F6Z0000000000000")
    if model:
        inner += pb(10, model)
    sample = pb(1, pb(1, inner)) + pb(4, 1700000000)
    if camera_code:
        sample += pb(2, pb(2, pb(1, pb(4, camera_code) + pb(2, 7))))
    return sample


def trak(handler: bytes, fmt: bytes, offset: int, size: int, co64=False, udta=b"") -> bytes:
    hdlr = full(b"hdlr", bytes(4), handler, bytes(12), b"Handler\x00")
    stsd = full(b"stsd", struct.pack(">I", 1), box(fmt, bytes(8)))
    stsz = full(b"stsz", struct.pack(">III", 0, 1, size))
    stco = (full(b"co64", struct.pack(">IQ", 1, offset)) if co64
            else full(b"stco", struct.pack(">II", 1, offset)))
    stbl = box(b"stbl", stsd, box(b"stts", bytes(8)), stsz, stco)
    return box(b"trak", full(b"tkhd", bytes(80)), box(b"mdia", full(b"mdhd", bytes(20)), hdlr,
                                                      box(b"minf", stbl)), udta)


def qt_text(typ: bytes, text: str) -> bytes:
    """A QuickTime ©-tag: 16-bit length, 16-bit language, text."""
    raw = text.encode()
    return box(typ, struct.pack(">HH", len(raw), 0x55C4), raw)


def ilst_meta(**items: str) -> bytes:
    tags = {"too": b"\xa9too", "mod": b"\xa9mod"}
    ilst = box(b"ilst", *(box(tags[k], box(b"data", struct.pack(">II", 1, 0), v.encode()))
                          for k, v in items.items()))
    return full(b"meta", full(b"hdlr", bytes(4), b"mdir", b"appl", bytes(8), b"\x00"), ilst)


def mp4(sample: bytes = b"", *, udta: bytes = b"", fmt=b"djmd", handler=b"meta", co64=False,
        moov_first=False, large_mdat=False, thumbnail=200_000) -> bytes:
    """A DJI-shaped file: ftyp, free, mdat (video, then the metadata sample), moov."""
    ftyp = box(b"ftyp", b"isom", struct.pack(">I", 512), b"isomiso2mp41")
    free = box(b"free")
    video = bytes(4096)
    big_udta = box(b"udta", udta, box(b"\xa9dji", b"x" * 40), box(b"snal", bytes(thumbnail)))

    def build(sample_offset: int) -> bytes:
        tracks = trak(b"vide", b"hvc1", len(ftyp) + len(free) + 16, len(video))
        if sample:
            tracks += trak(handler, fmt, sample_offset, len(sample), co64=co64)
        return box(b"moov", full(b"mvhd", bytes(96)), tracks, big_udta)

    if large_mdat:
        mdat = struct.pack(">I4sQ", 1, b"mdat", 16 + len(video) + len(sample)) + video + sample
    else:
        mdat = box(b"mdat", video, sample)
    head = 16 if large_mdat else 8
    if moov_first:
        # The offsets are fixed-width, so the moov's length does not depend on them.
        moov_len = len(build(0))
        offset = len(ftyp) + len(free) + moov_len + head + len(video)
        return ftyp + free + build(offset) + mdat
    offset = len(ftyp) + len(free) + head + len(video)
    return ftyp + free + mdat + build(offset)


def write(tmp_path, data: bytes, name="DJI_0001.MP4") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


MINI_4_PRO = djmd_header("dvtm_Mini4_Pro.proto", "DJI Mini4 Pro", "DJI FC8482")


# --------------------------------------------------------------------------- #
class TestMetadataTrack:
    def test_mini_4_pro(self, tmp_path):
        got = identify_camera(write(tmp_path, mp4(MINI_4_PRO)))
        assert got.preset is camera("mini-4-pro")
        assert got.name == "DJI Mini4 Pro" and got.source == TRACK
        assert got.evidence == 'the video\'s DJI metadata track says "DJI Mini4 Pro"'

    def test_every_name_in_the_header_is_found(self, tmp_path):
        with open(write(tmp_path, mp4(MINI_4_PRO)), "rb") as f:
            names = list(_names(f))
        assert names[:3] == [("DJI Mini4 Pro", TRACK), ("DJI FC8482", TRACK),
                             ("dvtm_Mini4_Pro.proto", TRACK)]

    @pytest.mark.parametrize("protocol,key", [
        ("dvtm_Air3.proto", "air-3"),
        ("dvtm_Air3s.proto", "air-3s"),
        ("dvtm_Mini5Pro.proto", "mini-5-pro"),
        ("dvtm_wm265e.proto", "mavic-3"),           # DJI's code for the Mavic 3
        ("dvtm_wm261.proto", "mavic-3"),            # ... and the Mavic 3 Pro
    ])
    def test_the_protocol_alone_names_the_drone(self, tmp_path, protocol, key):
        got = identify_camera(write(tmp_path, mp4(djmd_header(protocol))))
        assert got.preset is camera(key) and got.name == protocol

    def test_the_protocol_need_not_be_field_1(self, tmp_path):
        sample = djmd_header("dvtm_Air3.proto", at=3)
        assert identify_camera(write(tmp_path, mp4(sample))).preset is camera("air-3")

    def test_a_drone_with_no_preset_is_still_named(self, tmp_path):
        got = identify_camera(write(tmp_path, mp4(djmd_header("dvtm_AVATA2.proto", "DJI Avata2"))))
        assert got.preset is None and got.name == "DJI Avata2" and got.source == TRACK

    def test_an_unknown_protocol_with_no_model(self, tmp_path):
        got = identify_camera(write(tmp_path, mp4(djmd_header("dvtm_wm169.proto"))))
        assert got.preset is None and got.name == "dvtm_wm169.proto"

    @pytest.mark.parametrize("layout", [dict(co64=True), dict(moov_first=True),
                                        dict(large_mdat=True), dict(thumbnail=1_200_000)])
    def test_file_layouts(self, tmp_path, layout):
        got = identify_camera(write(tmp_path, mp4(MINI_4_PRO, **layout)))
        assert got.preset is camera("mini-4-pro")

    def test_a_metadata_track_labelled_otherwise_is_still_read(self, tmp_path):
        got = identify_camera(write(tmp_path, mp4(MINI_4_PRO, fmt=b"mebx")))
        assert got.preset is camera("mini-4-pro")

    def test_someone_elses_metadata_track_is_ignored(self, tmp_path):
        """A timed-metadata track that is not DJI's header names nothing."""
        sample = pb(1, pb(1, "GoPro Hero12")) + pb(2, 5)
        data = mp4(sample, fmt=b"gpmd", udta=qt_text(b"\xa9mdl", "FC3411"))
        got = identify_camera(write(tmp_path, data))
        assert got.preset is camera("air-2s") and got.source == MODEL_TAG


class TestTags:
    def test_dji_bare_model_tag(self, tmp_path):
        """DJI's ©mdl has no length header, just the string."""
        got = identify_camera(write(tmp_path, mp4(udta=box(b"\xa9mdl", b"FC3411"))))
        assert got.preset is camera("air-2s") and got.name == "FC3411" and got.source == MODEL_TAG

    def test_quicktime_model_tag(self, tmp_path):
        got = identify_camera(write(tmp_path, mp4(udta=qt_text(b"\xa9mod", "DJI Mavic 2 Pro"))))
        assert got.preset is camera("mavic-2-pro")

    def test_encoder_tag_with_the_product_name(self, tmp_path):
        got = identify_camera(write(tmp_path, mp4(udta=ilst_meta(too="DJI Mini4 Pro"))))
        assert got.preset is camera("mini-4-pro") and got.source == ENCODER_TAG

    @pytest.mark.parametrize("encoder", ["Lavf56.15.102", "DJI AVC encoder"])
    def test_an_encoder_tag_alone_never_names_a_camera(self, tmp_path, encoder):
        assert identify_camera(write(tmp_path, mp4(udta=ilst_meta(too=encoder)))) is None

    def test_the_track_wins_over_the_tags(self, tmp_path):
        data = mp4(MINI_4_PRO, udta=qt_text(b"\xa9mdl", "FC3411") + ilst_meta(too="Lavf56.15.102"))
        assert identify_camera(write(tmp_path, data)).source == TRACK


class TestNothingToRead:
    def test_missing_empty_and_not_a_video(self, tmp_path):
        assert identify_camera(tmp_path / "nope.mp4") is None
        assert identify_camera(write(tmp_path, b"")) is None
        assert identify_camera(write(tmp_path, b"1\n00:00:00,000 --> 00:00:00,033\n")) is None
        assert identify_camera(write(tmp_path, np.random.default_rng(1).bytes(50_000))) is None

    @pytest.mark.parametrize("cut", [0.5, 0.95, 0.999])
    def test_a_truncated_file(self, tmp_path, cut):
        data = mp4(MINI_4_PRO)
        assert identify_camera(write(tmp_path, data[:int(len(data) * cut)])) is None

    def test_a_sample_offset_past_the_end(self, tmp_path):
        data = bytearray(mp4(MINI_4_PRO))
        i = data.rindex(b"stco") + 12          # the metadata track's first offset
        data[i:i + 4] = struct.pack(">I", len(data) + 1000)
        assert identify_camera(write(tmp_path, bytes(data))) is None

    def test_a_video_opencv_wrote(self, tmp_path):
        path = tmp_path / "plain.mp4"
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for _ in range(5):
            out.write(np.zeros((48, 64, 3), np.uint8))
        out.release()
        assert identify_camera(path) is None

    @pytest.mark.skipif(not EXAMPLE.is_file(), reason="the example Air 2S clip is not checked in")
    def test_the_example_air_2s_clip_names_only_its_muxer(self):
        with open(EXAMPLE, "rb") as f:
            assert list(_names(f)) == [("Lavf56.15.102", ENCODER_TAG)]
        assert identify_camera(EXAMPLE) is None


# --------------------------------------------------------------------------- #
class TestPresetForName:
    @pytest.mark.parametrize("name,key", [
        ("DJI Mini4 Pro", "mini-4-pro"),
        ("dvtm_Mini4_Pro.proto", "mini-4-pro"),
        ("DJI FC8482", "mini-4-pro"),
        ("FC3582", "mini-3-pro"),
        ("DJI Mini 4K", "mini-2"),
        ("FC7303", "mini-2"),
        ("DJI Air 2S", "air-2s"),
        ("FC3411", "air-2s"),
        ("DJI Mavic 3 Pro", "mavic-3"),
        ("L2D-20c", "mavic-3"),
        ("L1D-20c", "mavic-2-pro"),
        ("FC220", "mavic-pro"),
        ("FC6310S", "phantom-4-pro"),
        ("mini-5-pro", "mini-5-pro"),
    ])
    def test_names_dji_writes(self, name, key):
        assert preset_for_name(name) is camera(key)

    @pytest.mark.parametrize("name", ["DJI Mavic 3 Thermal", "DJI Mini 3", "DJI Avata2", "DJI",
                                      "", "Lavf56.15.102", "dvtm_AVATA2.proto", "Mini"])
    def test_whole_names_only(self, name):
        assert preset_for_name(name) is None

    def test_every_preset_name_leads_back_to_its_preset(self):
        for p in DJI_CAMERAS:
            for name in (p.key, p.label, *p.names):
                assert preset_for_name(name) is p, (name, p.key)
