# STIV — Space-Time Image Velocimetry for river flow from drone video

A Python implementation of

> Fujita, I., Watanabe, H. & Tsubaki, R. (2007). *Development of a non-intrusive
> and efficient flow monitoring technique: The space-time image velocimetry
> (STIV).* International Journal of River Basin Management **5**(2), 105–114.

extended for UAV footage: camera-motion compensation, world-space rectification,
quality control, and velocity–area discharge. LS-PIV and particle tracking can
run in place of STIV or beside it, for comparison
([other velocity methods](#other-velocity-methods)).

Give it a drone video of a river and the two ends of a cross-section, and it
returns the streamwise surface velocity at every station across that section —
and, with a bed profile, the discharge.

```
stiv pick flight.mp4 -o survey.yaml     # click the cross-section on a frame
stiv run survey.yaml                    # analyse, plot, tabulate
```

---

## How it works

STIV reads velocity from the *slope of a streak*, not from pattern matching.

1. **Search lines.** A line is laid along the flow direction at each station
   across the cross-section.
2. **Space-time image (STI).** The brightness along that line is sampled from
   every frame and stacked — space across, time down. Ripples, boils and foam
   drifting along the line draw inclined streaks.
3. **Orientation analysis.** The streak inclination `φ` is measured with the
   structure tensor (Eqs. 1–11): local gradients are integrated over a 31×31
   window, and the tensor's eigenvectors give both the orientation and a
   **coherency** `C = (λ₁−λ₂)/(λ₁+λ₂)` saying how streak-like the patch is.
4. **Averaging.** Per-window angles are combined by the coherency-weighted
   histogram of Eq. (14): angles are histogrammed at several coherency
   thresholds and the histograms summed, so clean streaks are counted more often
   than noisy ones.
5. **Velocity.** Eq. (16): `U = (Sx / St) · tan φ`, where `Sx` is metres per STI
   column and `St` is seconds per STI row.

The whole cross-section is built in a single streaming pass over the video.

### What is different from the 2007 paper

| | Paper | Here | Why |
|---|---|---|---|
| Camera | fixed tower / bridge | hovering drone | a drifting aircraft writes its own motion into every STI |
| Search-line sampling | uniform in image pixels | uniform in **world** metres | keeps `Sx` truly constant in an oblique view — the paper notes STI length varies with obliqueness |
| Angle branch | `tan 2φ = 2J_xt/(J_tt−J_xx)` | minor eigenvector | see the note below |
| STI conditioning | — | static-background removal | a rock or a fixed glint draws a *vertical* streak, i.e. a vote for zero velocity |
| Output | velocity | velocity + QC + discharge | gauging needs an uncertainty and a number in m³/s |

**A note on Eq. (6).** The vector `n` in Eqs. (1)–(2) is defined *normal to the
lines of constant grey value*, so maximising `nᵀJn` selects the image gradient —
which runs **across** the streaks. Eq. (16) needs the inclination **of** the
streaks, ninety degrees away. Because Eq. (6) is a statement about `tan 2φ` it
cannot distinguish the two branches, and taking it literally returns
`U = −(Sx/St)/tan φ_true` — wrong sign for half the cases and wrong magnitude for
all of them. This implementation takes the minor eigenvector, and
[`tests/test_orientation.py`](tests/test_orientation.py) pins the behaviour.

---

## Install

```bash
pip install -e .            # or: pip install -e '.[test,gui]'
```

Needs Python ≥ 3.10, numpy, scipy, opencv-python-headless, matplotlib, pandas,
PyYAML, tqdm. The `gui` extra is only for the interactive `stiv pick`.

OpenCV is the *headless* build: nothing here opens an OpenCV window (the point
picker is a matplotlib window), and the headless wheel avoids clashing with
the Qt libraries a GUI backend brings in. If something else in your environment
needs the full `opencv-python`, keep just one of the two -- both install the
same `cv2` module and overwrite each other.

## Interactive notebook UI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simreaney/STIV/blob/main/notebooks/STIV_Interactive.ipynb)

For a guided, click-through alternative to the command line -- aimed at
students setting up their first survey -- there is a step-by-step Jupyter
wizard in [`notebooks/STIV_Interactive.ipynb`](notebooks/STIV_Interactive.ipynb).
It wraps the exact same pipeline as the CLI, one decision per screen, with a
side panel that keeps every choice in view.

The notebook installs what it needs itself: its first cell runs

```bash
pip install "stiv[widgets] @ git+https://github.com/simreaney/STIV.git"
```

if STIV is not installed yet, so it can be opened straight from GitHub --
in Google Colab with the badge above (the cell also connects Google Drive,
the easiest place for large videos and for the results), or downloaded and
opened in JupyterLab or VS Code. From a clone of the repository:

```bash
pip install -e '.[notebook]'
jupyter lab notebooks/STIV_Interactive.ipynb
```

1. **Video** -- browse to a file or upload one, generate a synthetic demo
   river with a *known* velocity profile to try the workflow risk-free, or
   reopen a saved survey (any run's `config_used.yaml`) to change a setting
   and run again without re-clicking anything.
2. **Scale** -- enter a ground resolution directly, or let it derive from
   flight altitude and camera field of view: pick a DJI drone from the list
   (Mini 3/4/5 Pro, Air 2S/3/3S, Mavic Pro, Mavic 2 Pro, Mavic 3, ...) and its
   horizontal FOV is filled in. When the video names its drone in its own
   metadata, as the Mini 4/5 Pro, Air 3/3S and Mavic 3 do, it is chosen for
   you (older DJIs mostly do not). If the drone saved a flight log (`.SRT`, same
   name as the video) its altitude is offered too, once you say how far the
   take-off point was above the water, and the focal length it records is
   checked against the drone chosen: a log reading about 22 mm can only be an
   Air 2S, so it is suggested. Nothing here is assumed: every velocity
   is proportional to the scale. Also sets which part of the clip to analyse.
3. **Cross-section** -- click the section and a downstream point on the frame
   at the analysis start time (the frame the drone-motion correction measures
   drift from). The water is then outlined automatically, from where the image
   moves once the drone's drift is removed; check it and confirm, or draw your
   own. It also warns if the water seems to flow away from your downstream
   point. To measure in more than one place, **Add another cross-section**
   once that is done: every section is analysed together, in one pass over
   the video. Loading a different video starts the survey afresh.
4. **Settings** -- station count, expected velocity range, drone-motion
   compensation, and optional discharge, with sensible defaults and the fussier
   options tucked under "Advanced". A PDF report is written with the results
   unless you untick it. With several cross-sections you name each one here,
   and give each its own depth file (those without one report velocities only).
5. **Run** -- reviews the choices, then analyses the video with a progress bar.
6. **Results** -- headline numbers, then tabs with the flow map (an arrow per
   station, its length proportional to the velocity), the velocity profile,
   the STIs and the table, then the PDF report (or a button to make one) and
   a zip of the results folder. With several cross-sections, a row of buttons
   switches between all of them side by side -- a table, and the flow map with
   every section on one frame -- and each one's own results.

**About the field of view.** DJI spec sheets quote the *diagonal* field of
view -- 84° for a 24 mm-equivalent lens -- but `hfov_deg` is the *horizontal*
one, about 71.5° for that lens on a 4:3 sensor. Using the quoted figure as the
horizontal one makes every velocity about 25% too high. The drone list (in
`stiv.cameras`) does the conversion; for another camera, convert with
`stiv.cameras.hfov_from_diagonal`.

It covers the common nadir-drone case; oblique views with ground control
points, and scripting a batch of surveys, still want the CLI below.

## Check it works

```bash
stiv demo
```

renders a synthetic river with a known parabolic velocity profile — including
simulated drone drift, sway and yaw — analyses it, and reports the error against
ground truth. Expect **≈0.6% relative RMSE**:

```
stations       19/21 clean, 2 failed (2 interpolated)
surface U      mean 0.783  median 0.902  max 1.209 m/s
stabilisation: 1200 features, median residual 0.24 px, max frame shift 59.2 px,
               0/299 frames fell back to the previous transform, 18 feature top-up(s)

  RMSE           : 0.0033 m/s
  bias           : +0.0015 m/s
  relative RMSE  : 0.6 %
```

The relative RMSE is taken over the stations moving faster than 0.15 m/s, since
a relative error is meaningless where the truth approaches zero; the run prints
how many stations that was.

---

## Using it on real footage

### 1. Look at the video

```bash
stiv info flight.mp4 --camera mini-4-pro --altitude-m 60
```

Reports resolution and frame rate, converts the flight geometry into a ground
sample distance, and suggests a `frame_step` for the velocity you expect.

`--camera` takes a DJI drone and fills in its *horizontal* field of view;
`stiv cameras` lists the ones it knows (Mini 2 to Mini 5 Pro, Air 2S/3/3S,
Mavic Pro, Mavic 2 Pro, Mavic 3, Phantom 4 Pro). For any other camera give
`--hfov-deg` -- the horizontal field of view, not the diagonal one most spec
sheets quote (see [About the field of view](#interactive-notebook-ui)).
Give neither and the drone is read from the video file when it names one:
the Mini 4 Pro, Mini 5 Pro, Air 3, Air 3S and Mavic 3 record it in a metadata
track, though older DJIs (the Air 2S among them) mostly do not, and editing
software usually strips it. The output says which drone was found and where;
a `--camera` that disagrees with the file is warned about.

If the drone saved its flight log beside the video (`.SRT`, same name --
"video captions" in the DJI app), `stiv info` reports the height it recorded.
That is the height above the **take-off point**, not above the water, so it is
used only when you say how far the take-off point was above the water surface:

```bash
stiv info flight.mp4 --camera mini-4-pro --takeoff-height 1.5
```

(`0` if you took off at water level.) When the log shows the camera was
zoomed, you are told the narrower field of view to use instead of the preset.
The log's focal length also says which lenses could have recorded it, since
zoom only ever lengthens it. The Air 2S's 22 mm lens is the widest of the
presets (all the others are 24 mm or longer), so a log reading about 22 mm
suggests `--camera air-2s`, and a `--camera` whose lens is longer than the log
shows is warned about.
`--altitude-m` and `--takeoff-height` are alternatives, as are `--camera` and
`--hfov-deg`; `stiv init` and `stiv pick` take the same options, and write the
scale into the survey, with a comment saying where each number came from.

### 2. Define the survey

```bash
stiv pick flight.mp4 -o survey.yaml
```

Opens a frame and asks you to click: the two ends of the cross-section, and a
point downstream (which fixes the flow direction). The water is then outlined
automatically, from what still moves once the drone's drift is taken out, and
shown on the frame: enter accepts it, `n` turns it down and lets you click the
outline yourself (or press enter at once to go without one). If the outline
cannot be found -- still water, say -- the terminal says why and you are asked
to click it; it also warns when the water seems to flow away from your
downstream point. `--no-auto-water` goes straight to clicking. `-t SECONDS`
shows a later frame, and that time also becomes `video.start_s`, since drift is
measured from the first analysed frame.

Unless the scale options above are given, the config it writes leaves the
scale as `m_per_px: null` (with `--gcps`, the world coordinates of the control
points as `null`s), and `stiv run` refuses the file until you fill it in --
there is no safe default for a number every velocity is proportional to. `stiv init flight.mp4` writes the same skeleton
with the section ends also `null`, for filling in by hand, or start from
[`examples/nadir_drone.yaml`](examples/nadir_drone.yaml). Neither command
overwrites an existing file unless given `--force`. The video path is written
relative to the config file's folder (or absolute, if the video is elsewhere),
because that is what relative paths in a config are resolved against.

#### Several cross-sections

One clip can be measured at several places at once:

```bash
stiv pick flight.mp4 --sections 3 -o survey.yaml
```

asks for each cross-section in turn (each window shows those already clicked),
then for one point downstream of all of them. The config lists them under
`sections`, and the `section` block then holds only what they share:

```yaml
section:   {n_stations: 40, edge_trim: 0.02}    # shared by every section
sections:
  - {name: gauge, start_px: [330, 880], end_px: [1608, 830],
     bathymetry_csv: gauge_depths.csv}
  - {name: riffle, start_px: [340, 1250], end_px: [1640, 1210], n_stations: 30}
flow:      {direction: auto, downstream_px: [1000, 2100]}
```

An entry needs only its ends; whatever else it leaves out -- `n_stations`,
`edge_trim`, `downstream_px`, the depths -- comes from the shared blocks. A
section can have its own `downstream_px` where no one point is downstream of
them all, and its own `bathymetry_csv` (or `depths_m`, `water_level_m`), which
turns its discharge on. [`examples/multi_section.yaml`](examples/multi_section.yaml)
has three; `stiv init --sections 3` writes the skeleton with the ends as `null`.

`stiv run` analyses them together, in **one pass over the video**: the frames
are read and the drone's drift tracked once, and every section's search lines
are sampled from each frame. Each section is then analysed as it would be on
its own, with its own flow direction, quality control and discharge. The one
difference is that the drift correction keeps off the water around every
section's search lines, not one section's, so it follows slightly different
bank features: on the synthetic test river the velocities move by a couple of
mm/s, and with `stabilise.enabled: false` they are identical.

### 3. Run

```bash
stiv run survey.yaml
stiv run survey.yaml --pdf     # ...and report.pdf as well
```

Outputs, in `output.out_dir`:

| File | Contents |
|---|---|
| `velocity.csv` | per station: distance, φ, tan φ, velocity, σ, coherency, flags (σ is explained below) |
| `summary.json` | machine-readable run summary |
| `report.txt` | the console summary, including warnings |
| `velocity_profile.png` | velocity across the section with its uncertainty band, and coherency below |
| `overview.png` | the frame with the section, search lines and flow direction drawn |
| `velocity_map.png` | the frame with one flow arrow per station, its length proportional to the velocity |
| `sti_panel.png` | contact sheet of up to 24 STIs, evenly spaced across the section (cf. Fig. 4) |
| `sti/*.png` | every station's STI, full resolution (`output.save_sti_images`) |
| `sti_detail/*.png` | orientation vectors, coherency map and angle histogram (Figs. 8, 9, 11) for the first, middle and last stations |
| `discharge.png`, `discharge_panels.csv` | if depths were supplied |
| `orientation_fields.npz` | per-window orientation and coherency fields, if `output.save_fields` |
| `config_used.yaml` | the fully resolved config, for reproducibility |
| `report.pdf` | one document to keep or hand in: the headline velocities, the survey details and every warning, then each figure, the station table and the config used -- with `--pdf` or `output.pdf_report: true` (the notebook wizard writes it by default) |
| `velocity_<method>.csv` | each method in `velocimetry.compare`, station by station ([other velocity methods](#other-velocity-methods)) |
| `correlation_panel.png` | LS-PIV's correlation plane at each station, on velocity axes |
| `tracks.png` | particle tracking's tracks over the frame, coloured by speed |

With several cross-sections, each one's files above go in a sub-folder named
after it -- its own `config_used.yaml` re-runs that section alone -- and the
top of `output.out_dir` holds what compares them: `sections.csv` (a row per
section: width, stations passing, mean and maximum velocity, discharge),
`report.txt`, `summary.json`, the survey's `config_used.yaml`, `overview.png`
and `velocity_map.png` with every section on one frame (one arrow scale for
all, so arrows compare between sections), and a single `report.pdf` covering
them all.

Each run lists the files it wrote in `.stiv_outputs.json`, and a re-run into
the same folder first removes exactly those, so a run with fewer stations, or
without discharge, leaves nothing stale behind. Nothing else is ever deleted: a
folder with no such list -- a survey folder holding its own `overview.png`, say
-- keeps everything, and same-named files are simply overwritten.
In `summary.json` a value that could not be computed is `null`.

The σ column of `velocity.csv` (`velocity_std_ms`) is the *spread* of the
individual window velocities about the station's value -- how much the streaks
in that STI disagree -- not the standard error of the station's velocity, which
is smaller by a factor nothing here can estimate, since the overlapping windows
are not independent. It is empty wherever the reported velocity is not the
station's own measurement: interpolated, median-filtered or failed. In code it
is `StationResult.velocity_spread_ms` (`velocity_std_ms` still works).

---

## Getting good numbers

Much of this is decided before take-off. For which drone to use and how to set it
up, see [`droneRecommendations.md`](droneRecommendations.md): frame rate and
resolution against the speeds you expect, codec and bit rate (with tests on
compressed footage), hover precision, and a field checklist.

**The water must have visible texture.** STIV measures the motion of a *pattern*.
Ripples, boils, foam, floating debris and suspended sediment all work; a glassy
pool does not. This is a physical limit of the method, not of the code.

**Hold the aircraft as still as you can, but let the software finish the job.**
Camera motion is the dominant error source in UAV gauging: at 2 cm/px and 30 fps
a drift of one pixel per frame is a spurious **0.6 m/s** on every station.
Stabilisation is on by default. Supply `stabilise.water_mask_px` — without it the
tracker may lock onto the moving water surface, and the run report will warn you
when fewer than half the tracked features agree with the fitted motion. `stiv
pick` and the notebook wizard outline the water for you; in a survey written by
hand, `water_mask_px: auto` does the same when the run starts. Nobody checks
that outline, so it is used only when the detection is clear-cut; otherwise the
run carries on without one, as it would have anyway, and the warnings and the
summary's `water mask` line say what happened. `auto` stays in
`config_used.yaml`, so a re-run looks again rather than reusing the outline. If it
finds fewer than 6 trackable features on static ground it cannot fit the motion
at all and switches itself off: the summary then reads `DISABLED AUTOMATICALLY`,
the warnings say so in capitals, and every velocity includes whatever the
aircraft drifted. Give it `stabilise.static_mask_px` around textured bank, or
set `stabilise.enabled: false` if the camera really was fixed.

**Keep `tan φ` near 1.** Accuracy is best when a surface feature moves about one
STI column per row, and the paper only validated `tan φ` from 0.1 to 7. Since
`tan φ = U·St/Sx`, the knob is `video.frame_step` — the run warns you and names a
better value when the median drifts out of range. If it is still above about 4
at `frame_step: 1`, `St` cannot get any shorter, so make `Sx` coarser instead:
record at a higher frame rate, fly higher, or set a smaller `stiv.n_samples` so
the samples along each search line are further apart. A longer search line does
not help: with the default automatic `n_samples` it only adds columns at the
same spacing, and leaves `tan φ` where it was.

**Use constant-frame-rate footage.** `St = frame_step/fps` is taken to be the
same for every STI row; if it is not, the streaks bend and nothing downstream
can tell. The run reads every frame's timestamp and warns when the intervals are
irregular (variable frame rate, which phones often record, or dropped frames) or
when they disagree with the frame rate the container reports -- velocities scale
with it. Raw `.h264`/`.mjpeg` streams carry no timing at all, so set
`video.fps_override` for them; a file that records no frame rate is an error
until you do.

**Leave `stiv.smooth_sigma` on.** It defaults to 1.5 px, and it is doing more
than tidying up noise: a feature advancing `tan φ` columns per STI row *aliases*
once `tan φ` approaches a third of the finest wavelength in the surface texture,
and aliasing biases the velocity **low** by tens of percent while the coherency
stays high enough that nothing flags it. On fine texture (finest wavelength ≈ 6
STI columns) the error at `tan φ` = 2.5 is −23% unsmoothed and −1.6% at
`smooth_sigma: 1.5`; where the pattern is already well sampled the smoothing
costs nothing measurable. This is also section 2.3 of the paper, which reports
the same for its "Case C" smooth patterns.

**Bracket the velocity, but not too tightly.** `stiv.velocity_range_ms` sets the
`φ_min`/`φ_max` of Eq. (14) and is the mechanism the paper uses to throw away
vectors corrupted by obstacles. Too wide admits noise; too narrow biases the
answer toward the range you guessed.

**Fly nadir if you can.** A straight-down view needs only a scale, and its
resolution is uniform across the frame. Oblique views need ground control points
and lose resolution with range. Survey at least 5 (6 or more is better): a
homography has exactly enough freedom to pass through 4, so with 4 their RMSE
is zero whatever their errors, a blunder goes undetected, and the run warns you
of this.

**Give the analysis enough to work with.** Each STI must be at least as large
as the `stiv.window_px` analysis window, in rows (frames) and columns (samples
along the search line). A clip or search line too short for that is refused
before any frames are processed, with a message naming what to change, rather
than failing every station as `sti_too_small`.

### Reading the quality flags

| Flag | Meaning |
|---|---|
| `no_orientation` | no window gave a usable streak angle at all |
| `sti_too_small` | this station's STI is smaller than the `window_px` analysis window — lengthen its search line (if every station would be, the run stops up front) |
| `low_coherency` | no streak-like structure — check the surface has texture |
| `high_uncertainty` | the angle histogram is flat; usually a line straddling the bank |
| `low_valid_fraction` | most windows fell outside `velocity_range_ms` — widen it |
| `tan_phi_high` / `tan_phi_low` | outside the validated range; adjust `frame_step` |
| `out_of_frame` | the search line left the image, often after camera drift |
| `interpolated` | this station failed and was filled from its neighbours |
| `median_filtered` | `quality.median_filter_stations` moved this station's velocity |
| `no_correlation_peak` | LS-PIV: nothing correlated -- featureless water, or the patch off the frame |
| `low_peak_ratio` | LS-PIV: the correlation peak hardly stands above the next one (`quality.min_peak_ratio`) |
| `displacement_at_search_edge` | LS-PIV: the texture moved as far as the search reaches -- raise `stiv.velocity_range_ms`'s upper bound |
| `too_few_tracks` | PTV: fewer tracks crossed the station than `quality.min_tracks` |
| `displacement_low` | LS-PIV: under half a sample per pair; raise `lspiv.pair_step` |
| `cross_flow_high` | LS-PIV, PTV: the water runs more than 30° off the flow direction -- an eddy, or the direction is set wrong |

Advisory flags (`tan_phi_*`, `median_filtered`, `displacement_low`,
`cross_flow_high`) never trigger interpolation: a
genuinely slow patch near the bank is measuring correctly, and pulling it toward
its faster neighbours would add error rather than remove it.

With `quality.interpolate_flagged: false`, a failed station is reported with no
velocity (empty in `velocity.csv`, a gap in `velocity_profile.png`) rather than
filled in; its untrusted estimate stays in the `velocity_raw_ms` column, and is
drawn hollow on the plot. Summary statistics are only ever taken over the
reported profile.

---

## Other velocity methods

STIV is one of three ways to read the surface velocity, and they can run side
by side on the same footage. Each uses the same stations, rectification,
camera-motion compensation, quality control and discharge, and all of them
share the one pass over the video.

```yaml
velocimetry:
  method: stiv            # stiv | lspiv | ptv: reported, and integrated into the discharge
  compare: [lspiv, ptv]   # also run these, and report them alongside
```

or `stiv run survey.yaml --method lspiv --compare stiv`. `--compare` works on
`stiv init`, `stiv pick` and `stiv demo` too; `stiv demo --compare lspiv,ptv`
holds every method to the synthetic river's known profile.

| Method | What it measures | Works best on | Struggles with |
|---|---|---|---|
| **STIV** (`stiv`) | the slope of streaks in a space-time image along a search line: streamwise only | sparse or patchy texture, oblique views | flow far off the search lines |
| **FFT-STIV** (`stiv.estimator: fft`) | the same STIs, read from their 2-D Fourier spectrum instead of the structure tensor | noisy, low-contrast STIs | short clips (a few seconds) |
| **LS-PIV** (`lspiv`) | cross-correlation of a small ortho-image patch at each station between frames: a 2-D vector | dense, lasting texture -- ripples, boils | glassy water, very sparse tracers |
| **Particle tracking** (`ptv`) | corners on the water followed by Lucas-Kanade optical flow (as KLT-IV): a 2-D vector | distinct tracers -- foam, leaves, debris | smooth water with no corners |

LS-PIV and particle tracking report the velocity along the flow direction in
`velocity_ms`, as STIV does, and the component across it in
`velocity_cross_ms`. That makes a wrongly set flow direction, or an eddy, show
up directly (`cross_flow_high`), and their arrows on `velocity_map.png` point
where the water went.

With `compare`, the reported method goes in `velocity.csv` with a
`velocity_<method>_ms` column for each compared one; each compared method's
full table goes in `velocity_<method>.csv`; `report.txt` and `summary.json`
give each method's clean stations, mean and maximum velocity and discharge; and
the velocity profile draws them all. Methods that disagree on which way the
water flows are warned about.

On the synthetic river of `stiv demo` (10 s, drifting drone, 21 stations,
1.2 m/s at the centre) the RMS errors against the known profile are 0.004 m/s
for STIV, 0.011 for FFT-STIV, 0.019 for LS-PIV and 0.022 for particle
tracking. Most of LS-PIV's and the tracker's error is at the stations beside
the banks, where the velocity changes fastest across their interrogation areas
and boxes, while a search line samples one line of water. That river's texture
suits STIV: it is a single, never-renewing pattern, sheared ever further by the
velocity profile. Real water renews its texture, and which method wins on your
footage is what `compare` is for.

**LS-PIV** (`lspiv` block). At each station a square of water surface is
sampled through the rectifier, square to the flow. The central *interrogation
area* of one frame's patch is correlated against the whole patch `pair_step`
frames later, and the correlations of every pair are summed (*ensemble*
correlation) before the peak is found, so intermittent texture still adds up.
The clip's mean patch is subtracted first (`background: mean`), so a rock or a
fixed glint does not correlate at zero velocity. That is exact even though the
mean is known only at the end, because the correlation expands into running
sums. The peak is located to a fraction of a sample by a 2-D Gaussian fit,
which stays unbiased when shear tilts the peak.

- `ia_size_m` defaults to the station spacing, at least 17 samples.
- `gsd_m` defaults to the image's own resolution.
- `pair_step` defaults to the largest step at which the top of
  `stiv.velocity_range_ms` still moves the texture no more than the
  interrogation area. Too tight an upper bound wastes resolution, as it does
  for STIV.

The `±` of a station is the scatter of one-second sub-ensembles.

**Particle tracking** (`ptv` block). Corners are detected inside each
station's box, which is `stiv.search_line_length_m` long down the flow and one
station spacing wide. Each corner is tracked for `track_frames` frames,
checked by tracking back (`fb_error_px`), and mapped through the inverse of the
camera-motion transform before it is rectified. Tracks that are too fast for
`stiv.velocity_range_ms` are dropped, and so are those more than
`max_angle_deg` off the flow axis or moving less than `min_displacement_px`
(static glare). The station's velocity is a robust straight-line fit across its
box, read at the centre, so tracks crowded on one side of a velocity gradient
do not bias it.

Other methods that would work but are not implemented:

- dense optical flow, OpenCV's DIS, sampled in the station boxes;
- two-direction STIV, for 2-D vectors from STIs;
- optical tracking velocimetry;
- deep optical flow such as RAFT, which is too heavy a dependency for Colab or a
  phone.

The iPhone app runs STIV only for now. A survey it exports has no `velocimetry`
block and so runs as STIV on the desktop, where `--method` and `--compare`
apply as to any other survey.

---

## Discharge

Surface velocity is not discharge. Two things must come from outside the imagery:

```yaml
discharge:
  enabled: true
  bathymetry_csv: bed.csv     # distance_m,depth_m  (or a bed-elevation column
                              # plus water_level_m)
  alpha: 0.85                 # velocity index: depth-averaged / surface
  bank_coefficient: 0.7
```

Integration is the mid-section method of standard current-meter gauging. **`alpha`
is the largest single uncertainty in the result** — 0.85 suits a logarithmic
profile in a wide channel, but it varies with relative roughness and wind, and is
worth calibrating against an ADCP transect if the site matters.

"The banks" here means the waterline found from the bed survey, not the ends of
the section as clicked. Stations outside it own no panel and add nothing, and
`bank_coefficient` applies to the outermost stations that are in the water. A
station in the water with no velocity (failed, and not interpolated) is filled
from its neighbours for the discharge, with a warning. If no station has a
velocity at all, the run warns and reports no discharge, and still writes the
velocity results.

With [several cross-sections](#several-cross-sections) each has its own depths:
`bathymetry_csv` (or `depths_m`) in its entry under `sections`. A section given
depths gets a discharge and the others report velocities only;
`discharge.enabled: true` asks for one at every section, so each then needs
depths, its own or the `discharge` block's.

---

## On an iPhone

There is a native app in [`ios/`](ios/README.md) that runs the same method from
the built-in camera, with the **LiDAR scanner supplying the scale** -- no
altitude, no ground control points, no tape measure.

The trick is that scale only needs one number. ARKit's world frame is
gravity-aligned, so still water is the plane `y = y0`; LiDAR reaches about five
metres, which is enough to range the *waterline* -- at water level by definition
-- and gravity carries that scale out to a far bank no depth sensor could reach.
Camera motion is handled by anchoring the search-line samples as world points
and reprojecting them through each frame's 6-DoF pose, which replaces
[`stiv/stabilise.py`](stiv/stabilise.py) entirely.

Far-field scale is extrapolated, so its error budget is derived, reported per
station and flagged; the dominant term goes as range / camera height, which is
why the app tells you to stand higher. Measurements export as `velocity.csv`
plus a `survey.yaml` that `stiv run` can re-analyse, since the phone's ray-plane
mapping *is* the homography `HomographyRectifier` fits.

Needs Xcode 16 and an iPhone 12 Pro or later; other iPhones fall back to a typed
camera height. Discharge is desktop-only for now.

## Library use

```python
from stiv import Config, run

result = run(Config.from_yaml("survey.yaml"))

print(result.summary())
result.to_frame().to_csv("velocity.csv", index=False)

for s in result.stations:
    if s.failed:  # interpolated from its neighbours, or no value at all
        print(f"{s.station_m:6.2f} m  not measured ({', '.join(s.flags)})")
    elif s.ok:
        print(f"{s.station_m:6.2f} m  {s.velocity_ms:.3f} m/s "
              f"(window spread {s.velocity_spread_ms:.3f} m/s)")
    else:  # a real measurement with a caveat, e.g. tan_phi_low
        print(f"{s.station_m:6.2f} m  {s.velocity_ms:.3f} m/s ({', '.join(s.flags)})")
```

A config with several `sections` is analysed with `run_survey`, which reads the
video once and returns a result per section:

```python
from stiv import Config, run_survey
from stiv.report import write_survey_outputs

cfg = Config.from_yaml("survey.yaml")
survey = run_survey(cfg)
print(survey.to_frame())                 # a row per section
for result in survey:                    # each a StivResult, as above
    print(result.config.section.name, result.velocities)
write_survey_outputs(survey, cfg)        # the folder layout `stiv run` writes
```

The pieces are usable on their own — `structure_tensor_field` and
`mean_orientation_angle` work on any 2-D array, so you can run the orientation
analysis on STIs produced by other software.

## Tests

```bash
pytest                     # 291 tests, ~20 s
```

The end-to-end tests render synthetic footage with a known velocity field and
assert the pipeline recovers it — with a fixed camera, with drone motion, across
all three stabilisation models, and at several `frame_step` values.

Notebooks are committed without outputs; CI fails if one still has them. Running
the wizard embeds a frame of whatever footage was loaded, so set git up once per
clone to strip notebooks on commit (your working copy keeps its outputs):

```bash
git config filter.strip-notebook.clean "python3 tools/strip_notebooks.py --filter"
```

## Layout

```
stiv/
  config.py       schema + validation for the survey YAML
  geometry.py     rectification (scale / GCP homography), cross-section, search lines
  video.py        frame iteration, the brightness field g(x,t)
  stabilise.py    drone-motion tracking and compensation
  sti.py          space-time image construction and conditioning
  orientation.py  structure tensor, coherency, Eq. (14), Eq. (16)   <- the core
                  (and the FFT angle estimator, stiv.estimator: fft)
  lspiv.py        LS-PIV: ortho-image patches, ensemble correlation
  ptv.py          particle tracking: KLT tracks through each station's box
  vectors.py      what LS-PIV and PTV hand the analysis; sub-pixel peak fit
  analysis.py     the pipeline, quality control, warnings
  discharge.py    velocity-area integration
  report.py       writing tables/figures/summary for a completed run
  plotting.py     diagnostic figures
  picker.py       interactive point selection (matplotlib window)
  notebook_ui.py  interactive point selection + full workflow (Jupyter wizard)
  cameras.py      field-of-view presets for DJI drones (diagonal -> horizontal)
  srt.py          reads DJI flight logs (.SRT) for the altitude
  videometa.py    reads which drone recorded a video from its own metadata
  watermask.py    automatic water outline for the drone-motion correction
  synthetic.py    synthetic river with known ground truth
  cli.py          command line

notebooks/
  STIV_Interactive.ipynb   launches the notebook_ui wizard

tools/
  strip_notebooks.py       keeps run outputs out of committed notebooks
  codec_test/              compression tests behind droneRecommendations.md
```

## Limitations

- **1-D by construction.** STIV measures the velocity component along the search
  line. It suits unidirectional flow, which is where discharge gauging happens;
  it cannot resolve recirculation or vortex shedding. Use PIV/LSPIV for those
  (Table 1 of the paper compares the two).
- Assumes a **flat, horizontal water surface** at a known level. Steep or
  strongly superelevated reaches break the planar mapping.
- Wind waves travelling at an angle to the flow, heavy rain, and specular glare
  all degrade the pattern, as the paper's own conclusions note.
- `alpha` is an assumption, not a measurement.

## License

MIT.
