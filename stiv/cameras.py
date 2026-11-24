"""Field-of-view presets for common DJI drones.

DJI's spec sheets quote one "FOV" per camera, and it is the **diagonal** field
of view: 84 degrees for a 24 mm-equivalent lens is exactly the diagonal of a
35 mm frame.  The scale calculation (:meth:`stiv.config.RectifyConfig.
resolve_m_per_px`) needs the **horizontal** one -- the angle spanned by the
image width -- which is narrower: about 71.5 degrees for that same lens on a
4:3 sensor.  Using the quoted figure as if it were horizontal overstates the
ground width, and so every velocity, by about a quarter.

Each preset therefore stores what DJI publishes (the diagonal FOV and the
sensor's aspect ratio) and derives the horizontal FOV from it.  The derivation
assumes the video uses the **full sensor width**, which is true of the normal
16:9 video modes on these drones: a 16:9 frame is cut from the 4:3 or 3:2
sensor by dropping rows, not columns.  It does not hold for digital zoom,
cropped high-frame-rate modes, or the vertical (portrait) shooting mode of the
Mini Pro models; for those, measure the scale on the ground instead.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional


def hfov_from_diagonal(dfov_deg: float, aspect: tuple[float, float]) -> float:
    """Horizontal FOV, degrees, of a rectilinear lens with diagonal FOV ``dfov_deg``.

    ``aspect`` is the sensor's (width, height) ratio, e.g. ``(4, 3)``.  The
    image half-diagonal and half-width are in the ratio ``sqrt(w^2 + h^2) : w``
    on the sensor, and the tangents of the half-angles keep that ratio.
    """
    if not 0.0 < dfov_deg < 180.0:
        raise ValueError("the diagonal field of view must be in (0, 180) degrees")
    w, h = (float(x) for x in aspect)
    if not (w > 0 and h > 0):
        raise ValueError("aspect must be two positive numbers, (width, height)")
    half = math.atan(math.tan(math.radians(dfov_deg) / 2.0) * w / math.hypot(w, h))
    return math.degrees(2.0 * half)


#: How far short of a lens's quoted focal length a logged one may read and
#: still be that lens (see :meth:`CameraPreset.can_log_focal_length`).
_FOCAL_TOLERANCE = 1.05


@dataclass(frozen=True)
class CameraPreset:
    """One camera, as its manufacturer specifies it."""

    key: str
    label: str
    #: The field of view the spec sheet quotes -- the diagonal one.
    dfov_deg: float
    #: Sensor (width, height) ratio.
    aspect: tuple[int, int]
    #: 35 mm-format-equivalent focal length, mm, as quoted.
    eq_focal_mm: float
    sensor: str = ""
    #: What DJI writes into a video to say which drone shot it, besides the
    #: key and label: product names, DJI's internal model codes (``wm265e``),
    #: and the camera's model number (the "FC" code a still's EXIF gives).
    #: See :func:`preset_for_name` and :mod:`stiv.videometa`.
    names: tuple[str, ...] = ()

    @property
    def hfov_deg(self) -> float:
        """Horizontal field of view of full-sensor-width video, degrees."""
        return hfov_from_diagonal(self.dfov_deg, self.aspect)

    def zoomed_hfov_deg(self, focal_len_mm: Optional[float]) -> Optional[float]:
        """The horizontal FOV at a logged focal length, when it shows digital zoom.

        ``focal_len_mm`` is the 35 mm-equivalent focal length a DJI flight log
        records (:mod:`stiv.srt`).  Returns None unless it is more than 10%
        longer than this lens's own -- i.e. unless the view was zoomed.
        """
        if not focal_len_mm or focal_len_mm <= 1.1 * self.eq_focal_mm:
            return None
        zoom = focal_len_mm / self.eq_focal_mm
        return math.degrees(2.0 * math.atan(math.tan(math.radians(self.hfov_deg) / 2.0) / zoom))

    def can_log_focal_length(self, focal_len_mm: float) -> bool:
        """Whether this camera could have logged ``focal_len_mm``.

        Digital zoom only lengthens the focal length a flight log records, so
        a lens can log its own focal length or anything longer, but nothing
        wider. Logs round a little (the Air 2S's 22 mm lens logs 22.4 mm), so
        a reading up to 5% short of the quoted figure still counts as this lens.
        """
        return focal_len_mm * _FOCAL_TOLERANCE >= self.eq_focal_mm

    def describe(self) -> str:
        return (
            f"{self.hfov_deg:.1f}° horizontal (DJI quotes {self.dfov_deg:g}° diagonal; "
            f"{self.sensor + ' sensor, ' if self.sensor else ''}{self.eq_focal_mm:g} mm equivalent)"
        )


#: Main (wide) cameras, from DJI's published specifications.  The model codes
#: are those DJI's EXIF and metadata give for these drones, and the internal
#: codes those ExifTool's DJI module lists for their metadata tracks.
DJI_CAMERAS: tuple[CameraPreset, ...] = (
    CameraPreset("mini-5-pro", "DJI Mini 5 Pro", 84.0, (4, 3), 24, '1"'),
    CameraPreset("mini-4-pro", "DJI Mini 4 Pro", 82.1, (4, 3), 24, '1/1.3"', names=("FC8482",)),
    CameraPreset("mini-3-pro", "DJI Mini 3 Pro", 82.1, (4, 3), 24, '1/1.3"', names=("FC3582",)),
    CameraPreset("mini-2", "DJI Mini 2 / Mini 4K", 83.0, (4, 3), 24, '1/2.3"',
                 names=("Mini 4K", "FC7303")),
    CameraPreset("air-3s", "DJI Air 3S (wide camera)", 84.0, (4, 3), 24, '1"'),
    CameraPreset("air-3", "DJI Air 3 (wide camera)", 82.0, (4, 3), 24, '1/1.3"'),
    CameraPreset("air-2s", "DJI Air 2S", 88.0, (3, 2), 22, '1"', names=("FC3411",)),
    CameraPreset("mavic-3", "DJI Mavic 3 / 3 Classic / 3 Pro (Hasselblad camera)", 84.0, (4, 3), 24, "4/3",
                 names=("Mavic 3 Classic", "Mavic 3 Pro", "Mavic 3 Cine",
                        "wm265e", "wm261", "wm260", "L2D-20c")),
    CameraPreset("mavic-2-pro", "DJI Mavic 2 Pro", 77.0, (3, 2), 28, '1"', names=("L1D-20c",)),
    CameraPreset("mavic-pro", "DJI Mavic Pro", 78.8, (4, 3), 26, '1/2.3"',
                 names=("Mavic Pro Platinum", "FC220")),
    CameraPreset("phantom-4-pro", "DJI Phantom 4 Pro / Pro V2.0", 84.0, (3, 2), 24, '1"',
                 names=("Phantom 4 Pro V2.0", "FC6310", "FC6310S")),
)


def camera(key: str) -> Optional[CameraPreset]:
    """The preset with this key, or None."""
    for preset in DJI_CAMERAS:
        if preset.key == key:
            return preset
    return None


def _normalise(name: str) -> str:
    """``name`` reduced for matching: 'DJI Mini4 Pro', 'dvtm_Mini4_Pro.proto'
    and 'mini-4-pro' all become 'mini4pro'."""
    text = re.sub(r"(?i)^\s*dvtm_|\.proto\s*$", "", name)
    text = re.sub(r"[^a-z0-9]", "", text.lower())
    return text[3:] if text.startswith("dji") else text


def preset_for_name(name: str) -> Optional[CameraPreset]:
    """The preset for a drone or camera name as a DJI file records it, or None.

    Case, spacing, punctuation and a leading "DJI" are ignored. ``name`` has
    to match a preset's key, label or one of its :attr:`~CameraPreset.names`
    whole; "DJI Mavic 3 Thermal" does not match the Mavic 3.
    """
    want = _normalise(name)
    if not want:
        return None
    for preset in DJI_CAMERAS:
        if any(_normalise(n) == want for n in (preset.key, preset.label, *preset.names)):
            return preset
    return None


def preset_for_focal_length(focal_len_mm: Optional[float]) -> Optional[CameraPreset]:
    """The only preset that could have logged ``focal_len_mm``, or None.

    ``focal_len_mm`` should be the shortest 35 mm-equivalent focal length a
    flight log records (:func:`stiv.srt.shortest_focal_len`). Zoom only
    lengthens it, so a reading could come from any lens no longer than it. When
    exactly one preset's lens is that wide, the log points to that drone. In
    practice that is the Air 2S: its 22 mm lens is wider than every other
    preset's 24 mm or more. A 24 mm reading fits nearly every drone listed and
    picks out none.

    This is only a hint. A drone the presets do not list could log the same
    figure.
    """
    if not focal_len_mm or not focal_len_mm > 0:
        return None
    able = [p for p in DJI_CAMERAS if p.can_log_focal_length(focal_len_mm)]
    return able[0] if len(able) == 1 else None
