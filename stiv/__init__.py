"""Space-Time Image Velocimetry (STIV) for river surface flow from drone video.

An implementation of

    Fujita, I., Watanabe, H. & Tsubaki, R. (2007), "Development of a
    non-intrusive and efficient flow monitoring technique: The space-time image
    velocimetry (STIV)", Intl. J. River Basin Management 5(2), 105-114.

Quick start::

    from stiv import Config, run

    result = run(Config.from_yaml("survey.yaml"))
    print(result.summary())
    result.to_frame().to_csv("velocity.csv", index=False)

or from the command line::

    stiv pick flight.mp4 -o survey.yaml     # click the cross-section
    stiv run survey.yaml                    # analyse and write outputs

A config listing several cross-sections under ``sections`` is analysed in one
pass over the video by :func:`run_survey`, which returns a result for each.
"""

# The single source of the package version: pyproject.toml reads it from here
# (``[tool.setuptools.dynamic]``).  Defined before the imports below so that a
# submodule can import it while the package is still initialising.
__version__ = "1.1.1"

from .analysis import StationResult, StivResult, SurveyResult, run, run_survey
from .config import Config, ConfigError
from .discharge import DischargeResult, compute_discharge
from .geometry import (
    HomographyRectifier,
    Rectifier,
    ScaleRectifier,
    SearchLine,
    SectionGeometry,
    build_section,
)
from .orientation import (
    AngleEstimate,
    OrientationField,
    angle_range_from_velocity,
    mean_orientation_angle,
    structure_tensor_field,
    velocity_from_angle,
)
from .sti import SpaceTimeImage, preprocess
from .video import FrameSource, VideoInfo, probe


__all__ = [
    "Config",
    "ConfigError",
    "run",
    "run_survey",
    "StivResult",
    "SurveyResult",
    "StationResult",
    "DischargeResult",
    "compute_discharge",
    "build_section",
    "SectionGeometry",
    "SearchLine",
    "Rectifier",
    "ScaleRectifier",
    "HomographyRectifier",
    "structure_tensor_field",
    "mean_orientation_angle",
    "velocity_from_angle",
    "angle_range_from_velocity",
    "OrientationField",
    "AngleEstimate",
    "SpaceTimeImage",
    "preprocess",
    "FrameSource",
    "VideoInfo",
    "probe",
    "__version__",
]
