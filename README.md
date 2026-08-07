# STIV — Space-Time Image Velocimetry for river flow from drone video

A Python implementation of

> Fujita, I., Watanabe, H. & Tsubaki, R. (2007). *Development of a non-intrusive
> and efficient flow monitoring technique: The space-time image velocimetry
> (STIV).* International Journal of River Basin Management **5**(2), 105–114.

extended for UAV footage: camera-motion compensation, world-space rectification,
quality control, and velocity–area discharge.

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

Needs Python ≥ 3.10, numpy, scipy, opencv-python, matplotlib, pandas, PyYAML,
tqdm. The `gui` extra is only for the interactive `stiv pick`.

## Interactive notebook UI

For a guided, click-through alternative to the command line -- aimed at
students setting up their first survey -- there is a step-by-step Jupyter
wizard in [`notebooks/STIV_Interactive.ipynb`](notebooks/STIV_Interactive.ipynb).
It wraps the exact same pipeline as the CLI, one decision per screen:

```bash
pip install -e '.[notebook]'
jupyter lab notebooks/STIV_Interactive.ipynb
```

1. **Video** -- browse to a file, or generate a synthetic demo river with a
   *known* velocity profile to try the workflow risk-free first.
2. **Scale** -- enter a ground resolution directly, or let it derive from
   flight altitude and camera field of view; suggests a `frame_step`.
3. **Cross-section** -- click the section, a downstream point, and (optionally)
   outline the water, directly on the video frame.
4. **Settings** -- station count, expected velocity range, drone-motion
   compensation, and optional discharge, with sensible defaults and the fussier
   options tucked under "Advanced".
5. **Run** -- reviews the choices, then analyses the video with a progress bar.
6. **Results** -- the summary, plots and velocity table inline, plus the folder
   everything was saved to.

It covers the common nadir-drone case; oblique views with ground control
points, and scripting a batch of surveys, still want the CLI below.

## Check it works

```bash
stiv demo
```

renders a synthetic river with a known parabolic velocity profile — including
simulated drone drift, sway and yaw — analyses it, and reports the error against
ground truth. Expect **≈3% relative RMSE**:

```
stations       19/21 clean, 2 failed and interpolated
surface U      mean 0.788  median 0.903  max 1.205 m/s
stabilisation: 1200 features, median residual 0.26 px, max frame shift 59.2 px

  RMSE           : 0.0099 m/s
  bias           : +0.0053 m/s
  relative RMSE  : 2.9 %
```

---

## Using it on real footage

### 1. Look at the video

```bash
stiv info flight.mp4 --altitude-m 60 --hfov-deg 84
```

Reports resolution and frame rate, converts the flight geometry into a ground
sample distance, and suggests a `frame_step` for the velocity you expect.

### 2. Define the survey

```bash
stiv pick flight.mp4 -o survey.yaml
```

Opens a frame and asks you to click: the two ends of the cross-section, a point
downstream (which fixes the flow direction), and optionally an outline of the
water. Writes a config you then edit. Or start from
[`examples/nadir_drone.yaml`](examples/nadir_drone.yaml) and fill in the
coordinates by hand.

### 3. Run

```bash
stiv run survey.yaml
```

Outputs, in `output.out_dir`:

| File | Contents |
|---|---|
| `velocity.csv` | per station: distance, φ, tan φ, velocity, σ, coherency, flags |
| `summary.json` | machine-readable run summary |
| `report.txt` | the console summary, including warnings |
| `velocity_profile.png` | velocity across the section with its uncertainty band, and coherency below |
| `overview.png` | the frame with the section, search lines and flow direction drawn |
| `sti_panel.png` | contact sheet of every STI (cf. Fig. 4) |
| `sti_detail/*.png` | orientation vectors, coherency map and angle histogram (Figs. 8, 9, 11) |
| `discharge.png`, `discharge_panels.csv` | if depths were supplied |
| `config_used.yaml` | the fully resolved config, for reproducibility |

---

## Getting good numbers

**The water must have visible texture.** STIV measures the motion of a *pattern*.
Ripples, boils, foam, floating debris and suspended sediment all work; a glassy
pool does not. This is a physical limit of the method, not of the code.

**Hold the aircraft as still as you can, but let the software finish the job.**
Camera motion is the dominant error source in UAV gauging: at 2 cm/px and 30 fps
a drift of one pixel per frame is a spurious **0.6 m/s** on every station.
Stabilisation is on by default. Supply `stabilise.water_mask_px` — without it the
tracker may lock onto the moving water surface, and the run report will warn you
when fewer than half the tracked features agree with the fitted motion.

**Keep `tan φ` near 1.** Accuracy is best when a surface feature moves about one
STI column per row, and the paper only validated `tan φ` from 0.1 to 7. Since
`tan φ = U·St/Sx`, the knob is `video.frame_step` — the run warns you and names a
better value when the median drifts out of range.

**Bracket the velocity, but not too tightly.** `stiv.velocity_range_ms` sets the
`φ_min`/`φ_max` of Eq. (14) and is the mechanism the paper uses to throw away
vectors corrupted by obstacles. Too wide admits noise; too narrow biases the
answer toward the range you guessed.

**Fly nadir if you can.** A straight-down view needs only a scale, and its
resolution is uniform across the frame. Oblique views need ground control points
and lose resolution with range.

### Reading the quality flags

| Flag | Meaning |
|---|---|
| `low_coherency` | no streak-like structure — check the surface has texture |
| `high_uncertainty` | the angle histogram is flat; usually a line straddling the bank |
| `low_valid_fraction` | most windows fell outside `velocity_range_ms` — widen it |
| `tan_phi_high` / `tan_phi_low` | outside the validated range; adjust `frame_step` |
| `out_of_frame` | the search line left the image, often after camera drift |
| `interpolated` | this station failed and was filled from its neighbours |

Advisory flags (`tan_phi_*`) never trigger interpolation: a genuinely slow patch
near the bank is measuring correctly, and pulling it toward its faster neighbours
would add error rather than remove it.

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

---

## Library use

```python
from stiv import Config, run

result = run(Config.from_yaml("survey.yaml"))

print(result.summary())
result.to_frame().to_csv("velocity.csv", index=False)

for s in result.stations:
    if s.failed:
        print(f"{s.station_m:6.2f} m  no data ({', '.join(s.flags)})")
    else:
        print(f"{s.station_m:6.2f} m  {s.velocity_ms:.3f} ± {s.velocity_std_ms:.3f} m/s")
```

The pieces are usable on their own — `structure_tensor_field` and
`mean_orientation_angle` work on any 2-D array, so you can run the orientation
analysis on STIs produced by other software.

## Tests

```bash
pytest                     # 98 tests, ~20 s
```

The end-to-end tests render synthetic footage with a known velocity field and
assert the pipeline recovers it — with a fixed camera, with drone motion, across
all three stabilisation models, and at several `frame_step` values.

## Layout

```
stiv/
  config.py       schema + validation for the survey YAML
  geometry.py     rectification (scale / GCP homography), cross-section, search lines
  video.py        frame iteration, the brightness field g(x,t)
  stabilise.py    drone-motion tracking and compensation
  sti.py          space-time image construction and conditioning
  orientation.py  structure tensor, coherency, Eq. (14), Eq. (16)   <- the core
  analysis.py     the pipeline, quality control, warnings
  discharge.py    velocity-area integration
  report.py       writing tables/figures/summary for a completed run
  plotting.py     diagnostic figures
  picker.py       interactive point selection (matplotlib window)
  notebook_ui.py  interactive point selection + full workflow (Jupyter wizard)
  synthetic.py    synthetic river with known ground truth
  cli.py          command line

notebooks/
  STIV_Interactive.ipynb   launches the notebook_ui wizard
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
