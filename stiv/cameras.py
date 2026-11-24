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

    def describe(self) -> str:
        return (
            f"{self.hfov_deg:.1f}° horizontal (DJI quotes {self.dfov_deg:g}° diagonal; "
            f"{self.sensor + ' sensor, ' if self.sensor else ''}{self.eq_focal_mm:g} mm equivalent)"
        )


#: Main (wide) cameras, from DJI's published specifications.
DJI_CAMERAS: tuple[CameraPreset, ...] = (
    CameraPreset("mini-5-pro", "DJI Mini 5 Pro", 84.0, (4, 3), 24, '1"'),
    CameraPreset("mini-4-pro", "DJI Mini 4 Pro", 82.1, (4, 3), 24, '1/1.3"'),
    CameraPreset("mini-3-pro", "DJI Mini 3 Pro", 82.1, (4, 3), 24, '1/1.3"'),
    CameraPreset("mini-2", "DJI Mini 2 / Mini 4K", 83.0, (4, 3), 24, '1/2.3"'),
    CameraPreset("air-3s", "DJI Air 3S (wide camera)", 84.0, (4, 3), 24, '1"'),
    CameraPreset("air-3", "DJI Air 3 (wide camera)", 82.0, (4, 3), 24, '1/1.3"'),
    CameraPreset("air-2s", "DJI Air 2S", 88.0, (3, 2), 22, '1"'),
    CameraPreset("mavic-3", "DJI Mavic 3 / 3 Classic / 3 Pro (Hasselblad camera)", 84.0, (4, 3), 24, "4/3"),
    CameraPreset("mavic-2-pro", "DJI Mavic 2 Pro", 77.0, (3, 2), 28, '1"'),
    CameraPreset("mavic-pro", "DJI Mavic Pro", 78.8, (4, 3), 26, '1/2.3"'),
    CameraPreset("phantom-4-pro", "DJI Phantom 4 Pro / Pro V2.0", 84.0, (3, 2), 24, '1"'),
)


def camera(key: str) -> Optional[CameraPreset]:
    """The preset with this key, or None."""
    for preset in DJI_CAMERAS:
        if preset.key == key:
            return preset
    return None
