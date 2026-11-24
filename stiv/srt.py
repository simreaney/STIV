"""Read the flight log DJI drones write next to each video (``.SRT`` captions).

With "video captions" switched on, a DJI drone saves a subtitle file beside
every clip -- ``DJI_0659.MP4`` gets ``DJI_0659.SRT`` -- with one caption per
frame carrying the camera settings and the aircraft's position.  Two things in
it matter here:

``rel_alt`` (or ``H`` / ``BAROMETER`` on older models)
    Height above the **take-off point**, from the barometer.  It is *not* the
    height above the water: a drone launched from a bank two metres above the
    river reads two metres low.  The caller has to add that offset.
``focal_len``
    The 35 mm-equivalent focal length actually in use; larger than the lens's
    native value when digital zoom was on, which narrows the field of view.

Formats differ by model and firmware, so only fields whose meaning is
unambiguous are read; an absolute (GPS / sea-level) altitude is ignored.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->")
#: Height above take-off, in the order tried.  ``rel_alt: 30.1`` (Mini 3/4/5
#: Pro, Air 2S/3/3S, Mavic 3), ``H 29.70m`` / ``H:29.7m`` (Mavic Pro), and
#: ``BAROMETER:10.2`` (Phantom 4).
_REL_ALT = (
    re.compile(r"rel_alt\s*:\s*(-?\d+(?:\.\d+)?)", re.I),
    re.compile(r"(?<![A-Za-z.])H\s*[: ]\s*(-?\d+(?:\.\d+)?)\s*m\b"),
    re.compile(r"BAROMETER\s*:\s*(-?\d+(?:\.\d+)?)", re.I),
)
_FOCAL = re.compile(r"focal_len\s*:\s*(\d+(?:\.\d+)?)", re.I)


@dataclass
class SrtRecord:
    t_s: float
    rel_alt_m: Optional[float] = None
    #: 35 mm-equivalent focal length in use, mm.
    focal_len_mm: Optional[float] = None


@dataclass
class AltitudeStats:
    median_m: float
    min_m: float
    max_m: float
    n: int
    #: Median 35 mm-equivalent focal length over the same records, if logged.
    focal_len_mm: Optional[float] = None


def find_srt(video_path: str | Path) -> Optional[Path]:
    """The ``.SRT`` beside ``video_path`` with the same name, if there is one."""
    p = Path(video_path)
    for suffix in (".SRT", ".srt", ".Srt"):
        q = p.with_suffix(suffix)
        if q.is_file():
            return q
    return None


def parse_srt(path: str | Path) -> list[SrtRecord]:
    """Every caption that carries a time, with whatever could be read from it."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_srt_text(text)


def parse_srt_text(text: str) -> list[SrtRecord]:
    records: list[SrtRecord] = []
    for block in re.split(r"\r?\n\s*\r?\n", text):
        m = _TIME.search(block)
        if not m:
            continue
        h, mi, s, ms = m.groups()
        t = int(h) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0
        body = block[m.end():]
        rec = SrtRecord(t_s=t)
        for pattern in _REL_ALT:
            a = pattern.search(body)
            if a:
                rec.rel_alt_m = float(a.group(1))
                break
        f = _FOCAL.search(body)
        if f:
            focal = float(f.group(1))
            # Air 2S and Mavic 2 firmware log tenths of a millimetre ("220").
            rec.focal_len_mm = focal / 10.0 if focal > 150 else focal
        records.append(rec)
    return records


def altitude_stats(records: Sequence[SrtRecord], start_s: float = 0.0,
                   duration_s: Optional[float] = None) -> Optional[AltitudeStats]:
    """Height above take-off over the analysis window; None if not logged.

    Falls back to the whole log when no caption falls inside the window.
    """
    with_alt = [r for r in records if r.rel_alt_m is not None and math.isfinite(r.rel_alt_m)]
    if not with_alt:
        return None
    end = math.inf if duration_s is None else start_s + duration_s
    window = [r for r in with_alt if start_s <= r.t_s <= end] or with_alt
    alt = np.array([r.rel_alt_m for r in window], float)
    focal = [r.focal_len_mm for r in window if r.focal_len_mm]
    return AltitudeStats(
        median_m=float(np.median(alt)), min_m=float(alt.min()), max_m=float(alt.max()),
        n=len(window), focal_len_mm=float(np.median(focal)) if focal else None,
    )
