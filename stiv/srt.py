"""Read the flight log DJI drones write next to each video (``.SRT`` captions).

With "video captions" switched on, a DJI drone saves a subtitle file beside
every clip -- ``DJI_0659.MP4`` gets ``DJI_0659.SRT`` -- with one caption per
frame carrying the camera settings and the aircraft's position.  Two things in
it matter here:

``rel_alt`` (or ``H`` / ``BAROMETER`` on older models)
    Height above the **take-off point**, from the barometer.  It is *not* the
    height above the water: a drone launched from a bank two metres above the
    river reads two metres low.  The caller has to add that offset.
``altitude``
    What the Air 2S and the Mavic 2 log instead, with no reference stated.  On
    those models it is the same height above take-off -- the number the
    controller shows as H -- so it is used, but marked as an assumption
    (:attr:`AltitudeStats.assumed`) for the user to check.  DJI firmware will
    not fly more than 500 m above the take-off point, so a log reading higher
    than that is height above sea level and is not used.
``focal_len``
    The 35 mm-equivalent focal length actually in use; larger than the lens's
    native value when digital zoom was on, which narrows the field of view.

A field that says what it is measured from (``rel_alt``) is preferred to one
that does not (``altitude``); an altitude above sea level is never used.
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
#: Pro, Air 3/3S, Mavic 3), ``H 29.70m`` / ``H:29.7m`` (Mavic Pro), and
#: ``BAROMETER:10.2`` (Phantom 4).
_REL_ALT = (
    re.compile(r"rel_alt\s*:\s*(-?\d+(?:\.\d+)?)", re.I),
    re.compile(r"(?<![A-Za-z.])H\s*[: ]\s*(-?\d+(?:\.\d+)?)\s*m\b"),
    re.compile(r"BAROMETER\s*:\s*(-?\d+(?:\.\d+)?)", re.I),
)
#: ``[altitude: 35.9]`` (Air 2S, Mavic 2): no reference stated; see above.
_ALTITUDE = re.compile(r"(?<![A-Za-z_])altitude\s*:\s*(-?\d+(?:\.\d+)?)", re.I)
_FOCAL = re.compile(r"focal_len\s*:\s*(\d+(?:\.\d+)?)", re.I)
#: DJI's ceiling above the take-off point, m.  A logged altitude above it can
#: only be height above sea level.
MAX_ABOVE_TAKEOFF_M = 500.0


@dataclass
class SrtRecord:
    t_s: float
    #: Height above take-off, from a field that says so.
    rel_alt_m: Optional[float] = None
    #: A bare ``altitude``, whose reference the log does not state.
    altitude_m: Optional[float] = None
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
    #: True when the heights are a bare ``altitude``, read as height above
    #: take-off because that is what it is on the models that log it -- but
    #: the log does not say so, and the user should check.
    assumed: bool = False


def find_srt(video_path: str | Path) -> Optional[Path]:
    """The ``.SRT`` beside ``video_path`` with the same name, if there is one.

    The name is matched ignoring case -- ``DJI_0659.MP4`` finds
    ``dji_0659.srt`` -- since a file copied or uploaded can change case, and on
    Linux (Colab included) names that differ only in case are different files.
    """
    p = Path(video_path)
    for suffix in (".SRT", ".srt", ".Srt"):
        q = p.with_suffix(suffix)
        if q.is_file():
            return q
    wanted = (p.stem + ".srt").lower()
    try:
        for q in sorted(p.parent.iterdir()):
            if q.name.lower() == wanted and q.is_file():
                return q
    except OSError:
        pass
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
        bare = _ALTITUDE.search(body)
        if bare:
            rec.altitude_m = float(bare.group(1))
        f = _FOCAL.search(body)
        if f:
            focal = float(f.group(1))
            # Air 2S and Mavic 2 firmware log tenths of a millimetre ("220").
            rec.focal_len_mm = focal / 10.0 if focal > 150 else focal
        records.append(rec)
    return records


def shortest_focal_len(records: Sequence[SrtRecord]) -> Optional[float]:
    """The shortest 35 mm-equivalent focal length the log records, mm; None if none.

    Over the whole log, not an analysis window: it is the lens's own focal
    length unless the zoom was on throughout, and so says which lens it was
    (:func:`stiv.cameras.preset_for_focal_length`).
    """
    focal = [r.focal_len_mm for r in records
             if r.focal_len_mm is not None and math.isfinite(r.focal_len_mm) and r.focal_len_mm > 0]
    return min(focal) if focal else None


def altitude_stats(records: Sequence[SrtRecord], start_s: float = 0.0,
                   duration_s: Optional[float] = None) -> Optional[AltitudeStats]:
    """Height above take-off over the analysis window; None if not logged.

    From ``rel_alt`` (or ``H`` / ``BAROMETER``) when the log has it; otherwise
    from a bare ``altitude``, marked ``assumed``, unless that reads higher than
    a drone can fly above its take-off point.  Falls back to the whole log when
    no caption falls inside the window.
    """
    end = math.inf if duration_s is None else start_s + duration_s
    for field, assumed in (("rel_alt_m", False), ("altitude_m", True)):
        with_alt = [r for r in records
                    if getattr(r, field) is not None and math.isfinite(getattr(r, field))]
        if not with_alt:
            continue
        window = [r for r in with_alt if start_s <= r.t_s <= end] or with_alt
        alt = np.array([getattr(r, field) for r in window], float)
        if assumed and alt.max() > MAX_ABOVE_TAKEOFF_M:
            return None                     # above sea level: no use here
        focal = [r.focal_len_mm for r in window if r.focal_len_mm]
        return AltitudeStats(
            median_m=float(np.median(alt)), min_m=float(alt.min()), max_m=float(alt.max()),
            n=len(window), focal_len_mm=float(np.median(focal)) if focal else None,
            assumed=assumed,
        )
    return None
