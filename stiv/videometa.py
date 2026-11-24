"""Which drone recorded a video, from the file's own metadata.

DJI drones name the aircraft in one of three places inside the MP4 or MOV,
depending on the generation.

The ``djmd`` metadata track
    The Mini 4 Pro, Mini 5 Pro, Air 3, Air 3S and Mavic 3 family record a
    timed-metadata track beside the video. Its first sample is a protobuf
    header that names the aircraft in three fields:

    * 1-1-1, the protocol (``dvtm_Mini4_Pro.proto``), whose middle part is the
      model or DJI's internal code for it (``wm265e`` for the Mavic 3).
    * 1-1-10, the model (``DJI Mini4 Pro``).
    * 2-2-1-4 or 3-2-1-4, the camera (``DJI FC8482``).

    The field layout follows ExifTool (``Image::ExifTool::DJI``), which lists
    the protocols it has seen.
``udta`` tags
    ``©mdl`` holds a model name; some older DJIs write it without QuickTime's
    usual length header. ``©mod`` is the standard model tag.
The encoder tag, ``©too``
    The newer drones above write their product name here (``DJI Mini4 Pro``).
    Other files carry the muxer's name, e.g. ``Lavf56.15.102``.

Older drones mostly record none of this. The Air 2S clip in ``examples/``
carries only its muxer's name. Footage that has been through an editor or a
converter usually loses all three.

Only box headers, a few small tags and one metadata sample are read: a few KB,
however long the clip. Nothing here raises on a file it cannot make sense of;
the answer is just None.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

from .cameras import CameraPreset, preset_for_name

#: The largest ``moov`` that will be read. DJI's is about a megabyte, most of
#: it a thumbnail; a long clip's sample tables add a little more.
_MAX_MOOV_BYTES = 64 << 20
#: The largest metadata sample that will be read. DJI's header is a few KB.
_MAX_SAMPLE_BYTES = 1 << 20
#: Protobuf nesting deeper than this is not followed.
_MAX_DEPTH = 8
#: A name longer than this is not a model name.
_MAX_NAME = 120

_PROTOCOL = re.compile(r"^dvtm_[\w.-]+\.proto$")
#: Where the djmd header keeps the model and the camera (see the module notes).
_MODEL_PATH = (1, 1, 10)
_PROTOCOL_PATH = (1, 1, 1)
_CAMERA_PATHS = ((2, 2, 1, 4), (3, 2, 1, 4))

#: Where each name came from, as the user is told it.
TRACK = "DJI metadata track"
MODEL_TAG = "model tag"
ENCODER_TAG = "encoder tag"


@dataclass(frozen=True)
class CameraId:
    """A camera that a video file names."""

    #: The name as the file gives it, e.g. "DJI Mini4 Pro" or "FC3411".
    name: str
    #: Where in the file it was found: :data:`TRACK`, :data:`MODEL_TAG` or
    #: :data:`ENCODER_TAG`.
    source: str
    #: The matching preset; None for a camera :mod:`stiv.cameras` does not know.
    preset: Optional[CameraPreset] = None

    @property
    def evidence(self) -> str:
        """E.g. 'the video's DJI metadata track says "DJI Mini4 Pro"'."""
        return f'the video\'s {self.source} says "{self.name}"'


def identify_camera(path: str | Path) -> Optional[CameraId]:
    """The camera the video at ``path`` names in its own metadata, or None.

    A name that matches a preset is preferred: the model in the metadata
    track first, then its camera code and protocol, then the model tag, then
    the encoder tag. Failing that, a DJI name with no preset is returned, with
    ``preset`` None, so the caller can say which drone it was. The encoder tag
    is never returned on its own, because outside the newer DJIs it names
    software, not a camera.
    """
    try:
        with open(path, "rb") as f:
            names = list(_names(f))
    except (OSError, ValueError, struct.error, IndexError, MemoryError):
        return None
    for name, source in names:
        preset = preset_for_name(name)
        if preset is not None:
            return CameraId(name, source, preset)
    for name, source in names:
        if source != ENCODER_TAG and (name.upper().startswith("DJI") or _PROTOCOL.match(name)):
            return CameraId(name, source, None)
    return None


# --------------------------------------------------------------------------- #
def _names(f: BinaryIO) -> Iterator[tuple[str, str]]:
    """Every camera-like name in the file, most specific first."""
    moov = _read_moov(f)
    if moov is None:
        return
    tracks = [trak for typ, trak in _children(moov) if typ == b"trak"]
    # The metadata track: the sample-description format says so; any other
    # timed-metadata track is checked too, in case a model labels it differently.
    for want_djmd in (True, False):
        for trak in tracks:
            if (_sample_format(trak) == b"djmd") != want_djmd:
                continue
            if not want_djmd and _handler(trak) != b"meta":
                continue
            yield from _djmd_names(f, trak)
    udtas = [moov_udta for typ, moov_udta in _children(moov) if typ == b"udta"]
    udtas += [u for trak in tracks for typ, u in _children(trak) if typ == b"udta"]
    tags = [tag for udta in udtas for tag in _udta_tags(udta)]
    yield from ((text, MODEL_TAG) for kind, text in tags if kind == "model")
    yield from ((text, ENCODER_TAG) for kind, text in tags if kind == "encoder")


def _read_moov(f: BinaryIO) -> Optional[bytes]:
    """The ``moov`` box's payload, wherever it is; DJI writes it after ``mdat``."""
    f.seek(0, 2)
    end = f.tell()
    pos = 0
    while pos + 8 <= end:
        f.seek(pos)
        head = f.read(16)
        if len(head) < 8:
            return None
        size, typ = struct.unpack_from(">I4s", head)
        header = 8
        if size == 1:
            if len(head) < 16:
                return None
            size, header = struct.unpack_from(">Q", head, 8)[0], 16
        elif size == 0:
            size = end - pos
        if size < header:
            return None
        if typ == b"moov":
            if size > _MAX_MOOV_BYTES:
                return None
            f.seek(pos + header)
            body = f.read(size - header)
            return body if len(body) == size - header else None     # a truncated file
        pos += size
    return None


def _boxes(buf: bytes) -> Iterator[tuple[bytes, int, int]]:
    """(type, payload start, payload end) of each box laid end to end in ``buf``."""
    pos, end = 0, len(buf)
    while pos + 8 <= end:
        size, typ = struct.unpack_from(">I4s", buf, pos)
        header = 8
        if size == 1:
            if pos + 16 > end:
                return
            size, header = struct.unpack_from(">Q", buf, pos + 8)[0], 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield typ, pos + header, pos + size
        pos += size


def _children(buf: bytes) -> Iterator[tuple[bytes, bytes]]:
    for typ, a, b in _boxes(buf):
        yield typ, buf[a:b]


def _find(buf: Optional[bytes], *path: bytes) -> Optional[bytes]:
    """The payload of the first box along ``path``, e.g. ``(b"mdia", b"hdlr")``."""
    for step in path:
        if buf is None:
            return None
        buf = next((payload for typ, payload in _children(buf) if typ == step), None)
    return buf


def _handler(trak: bytes) -> Optional[bytes]:
    hdlr = _find(trak, b"mdia", b"hdlr")
    # version/flags (4), pre_defined (4), handler_type (4)
    return hdlr[8:12] if hdlr is not None and len(hdlr) >= 12 else None


def _stbl(trak: bytes) -> Optional[bytes]:
    return _find(trak, b"mdia", b"minf", b"stbl")


def _sample_format(trak: bytes) -> Optional[bytes]:
    stsd = _find(_stbl(trak), b"stsd")
    # version/flags (4), entry_count (4), then the first entry: size (4), format (4)
    return stsd[12:16] if stsd is not None and len(stsd) >= 16 else None


def _first_sample(trak: bytes) -> Optional[tuple[int, int]]:
    """(file offset, size) of a track's first sample, which starts its first chunk."""
    stbl = _stbl(trak)
    stsz = _find(stbl, b"stsz")
    if stsz is None or len(stsz) < 12:
        return None
    size, count = struct.unpack_from(">II", stsz, 4)
    if count == 0:
        return None
    if size == 0:
        if len(stsz) < 16:
            return None
        size = struct.unpack_from(">I", stsz, 12)[0]
    stco, co64 = _find(stbl, b"stco"), _find(stbl, b"co64")
    if stco is not None and len(stco) >= 12 and struct.unpack_from(">I", stco, 4)[0] > 0:
        offset = struct.unpack_from(">I", stco, 8)[0]
    elif co64 is not None and len(co64) >= 16 and struct.unpack_from(">I", co64, 4)[0] > 0:
        offset = struct.unpack_from(">Q", co64, 8)[0]
    else:
        return None
    return offset, size


def _djmd_names(f: BinaryIO, trak: bytes) -> Iterator[tuple[str, str]]:
    span = _first_sample(trak)
    if span is None or not 0 < span[1] <= _MAX_SAMPLE_BYTES:
        return
    f.seek(span[0])
    sample = f.read(span[1])
    strings: dict[tuple[int, ...], str] = {}
    for path, text in _pb_strings(sample):
        strings.setdefault(path, text)
    protocol = strings.get(_PROTOCOL_PATH)
    if protocol is None or not _PROTOCOL.match(protocol):
        # ExifTool does not rely on the protocol's position either.
        protocol = next((t for t in strings.values() if _PROTOCOL.match(t)), None)
    if protocol is None:
        return                      # not DJI's header
    for path in (_MODEL_PATH, *_CAMERA_PATHS):
        if path in strings:
            yield strings[path], TRACK
    yield protocol, TRACK


def _udta_tags(udta: bytes) -> Iterator[tuple[str, str]]:
    """("model" | "encoder", text) for each tag of interest in a ``udta``."""
    kinds = {b"\xa9mdl": "model", b"\xa9mod": "model", b"\xa9too": "encoder"}
    for typ, payload in _children(udta):
        if typ in kinds:
            text = _udta_text(payload)
            if text:
                yield kinds[typ], text
        elif typ == b"meta":
            # A full box in MP4 (4 bytes of version and flags), a plain one in QuickTime.
            body = payload[4:] if payload[8:12] == b"hdlr" else payload
            for item_typ, item in _children(_find(body, b"ilst") or b""):
                data = _find(item, b"data")
                if item_typ in kinds and data is not None and len(data) > 8:
                    text = _text(data[8:])          # type (4) and locale (4) first
                    if text:
                        yield kinds[item_typ], text


def _udta_text(payload: bytes) -> Optional[str]:
    # QuickTime's form is a 16-bit length, a 16-bit language code, then the
    # text. DJI's ©mdl is the bare string.
    if len(payload) >= 4:
        n = struct.unpack_from(">H", payload)[0]
        if 0 < n <= len(payload) - 4:
            return _text(payload[4:4 + n])
    return _text(payload)


def _text(raw: bytes) -> Optional[str]:
    """``raw`` as a short, printable string, or None."""
    try:
        text = raw.decode("utf-8").strip("\x00 \t\r\n")
    except UnicodeDecodeError:
        return None
    return text if 0 < len(text) <= _MAX_NAME and text.isprintable() else None


# --------------------------------------------------------------------------- #
# Just enough protobuf to find the strings in a message whose schema is unknown.
def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while pos < len(buf) and shift < 64:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
    raise ValueError("not a varint")


def _pb_fields(buf: bytes) -> list[tuple[int, int, object]]:
    """Every field of a message: (number, wire type, value); ValueError if ``buf`` is not one."""
    fields, pos = [], 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise ValueError("field number 0")
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire in (1, 5):
            n = 8 if wire == 1 else 4
            value, pos = buf[pos:pos + n], pos + n
        elif wire == 2:
            n, pos = _varint(buf, pos)
            value, pos = buf[pos:pos + n], pos + n
        else:
            raise ValueError(f"wire type {wire}")
        if pos > len(buf):
            raise ValueError("runs past the end")
        fields.append((number, wire, value))
    return fields


def _pb_strings(buf: bytes, path: tuple[int, ...] = (), depth: int = 0
                ) -> Iterator[tuple[tuple[int, ...], str]]:
    """(field path, text) of each string in a message, depth first.

    A length-delimited field is a string when it decodes as printable text.
    Otherwise it is followed as a nested message, if it parses as one.
    """
    try:
        fields = _pb_fields(buf)
    except ValueError:
        return
    for number, wire, value in fields:
        if wire != 2:
            continue
        here = path + (number,)
        text = _text(value)
        if text is not None:
            yield here, text
        elif depth < _MAX_DEPTH and value:
            yield from _pb_strings(value, here, depth + 1)
