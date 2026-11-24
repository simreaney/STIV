# STIV for iPhone

Space-Time Image Velocimetry from the built-in camera, with the LiDAR scanner
supplying real-world scale. Same method as the Python package in the repository
root — Fujita, Watanabe & Tsubaki (2007), Eqs. (1)–(16) — with the two
drone-specific parts replaced by what a phone actually has.

```
ios/
  StivKit/          the numerics: a Swift port of the paper. No ARKit, no UIKit.
  StivApp/          the app: ARKit capture, the six-screen survey, results.
  StivAppTests/     the app's tests, run in the simulator or on a phone.
  StivApp.xcodeproj hand-authored, Xcode 16 synchronized folders.
  project.yml       XcodeGen fallback, if the project above will not open.
  tools/            the test runner and the golden-fixture generator.
  TESTING.md        what each layer of testing proves, and the field protocol.
  DEPLOYING.md      getting the app onto a test iPhone.
```

## How the scale works

This is the part worth understanding before trusting a number.

Apple's LiDAR reaches about five metres. A river cross-section is ten to fifty.
Depth alone cannot scale a river, so it is not asked to.

ARKit's world frame is **gravity-aligned**: +Y is up, to within a fraction of a
degree, for free, from the IMU. Still water is horizontal. So the water surface
is the plane `y = y₀`, and `y₀` is a single scalar — well within what LiDAR can
measure at close range.

What gets measured is the **waterline**: the contour where water meets bank. It
sits at water level by definition, it is texture the sensor handles well, and
from a bank it is the nearest thing in the scene. The water itself returns
little that passes the confidence gate, so most samples in the aiming box are
bank, all of them above the water; the fit therefore takes the *bottom* of the
returns rather than the median of everything, which would sit ten or twenty
centimetres high: the water's own cluster of heights when there is one, the foot
of the bank's spread of heights when there is not, ignoring a small cluster of
reflections below. Everything after that is
ray–plane intersection: unproject any pixel through the camera intrinsics,
intersect `y = y₀`, and read off world metres — including at a far bank thirty
metres away where no depth measurement is possible at all.

(Close in — a flume, a culvert, a footbridge — the sensor reaches the surface
itself and the same fit runs on it directly, more accurately, since nothing is
being inferred from a contour. The app records which case it was.)

**The price.** Far-field scale is extrapolated, and the error grows with range:

| term | size | fix |
|---|---|---|
| water level, `δh/h` | 1.5% at `h` = 2 m, `δh` = 3 cm | tidier sweep |
| gravity tilt, `2εR/h` | ~10% at `ε` = 0.3°, `R` = 20 m, `h` = 2 m | **raise the camera** |
| VIO scale drift | ~1%, worse when tracking is limited | hold still |

The tilt term dominates, it scales with range ÷ camera height, and its
mitigation is physical rather than computational: a phone at 1.5 m on a bank has
`R/h = 20` at the far side of a 30 m river; the same phone on a 6 m bridge
parapet has `R/h = 5`. Velocity is linear in `Sx`, so all of this is folded into
the reported σ and flagged per station rather than left in a footnote. See
`StivKit/Sources/StivKit/Uncertainty.swift` for the derivation.

## Camera motion

The search-line sample points are fixed **world** points on the water plane, and
every frame projects them through its own `ARCamera.transform`. Camera motion
therefore never enters the space-time image: this is 6-DoF compensation, and it
replaces `stiv/stabilise.py` outright. It is also strictly better — a 2-D image
warp cannot represent what a translating camera does to a scene with depth.

Because visual-inertial odometry drifts, and drifts worst looking at open water,
a cross-check runs alongside: patches of static ground with LiDAR-known depth
are aligned by translation-only Lucas–Kanade, and the residual is the part of the
image motion the pose failed to explain. It validates the near field; the far
field is assumed to follow, and the report says so.

A small residual is also corrected, as a correction to the *pose* — three
rotations and three translations, fitted with each patch's depth — rather than
as an image warp. The patches are a couple of metres away and the river tens of
metres, and a translation error moves the two by different amounts where a
rotation error moves them alike; correcting the pose gives each sample the shift
its own depth implies. When the patches' depths are too alike to tell a turn
from a slip, it corrects rotation only. A correction is applied only when it
stands out from the alignment noise, so a correct pose is left alone; it is
ramped in rather than applied in one frame, held rather than dropped when a
check fails, and refused above a 6 px cap.

### The water outline

The cross-check's patches must stay off the moving water, so the section screen
asks for an outline of it -- and finds one itself. Straight after the downstream
tap, while the phone is still aimed at the section, a second or so of frames
is copied out, reduced about six times, and compared: each grid cell is
followed as a point on the water plane through every frame's own pose, so the
camera's motion is already accounted for, and whatever *still* changes is the
water. The threshold is calibrated from the section just tapped -- its middle
is water, the ground just beyond its ends is bank -- and a scene where the two
do not differ clearly gets no outline rather than a guessed one. The outline is
drawn on the feed for the user to accept, or replace by tapping their own. It
also warns when the water seems to flow away from the downstream point, with a
button to move the point across. This is the phone's version of
`stiv/watermask.py`: `StivKit/Sources/StivKit/WaterOutline.swift`.

## Building

Requires **Xcode 16+** and an **iPhone 12 Pro or later** for LiDAR. Non-Pro
iPhones run the app with a manually entered camera height — there is no
`UIRequiredDeviceCapabilities` value for a depth sensor, so LiDAR can only be
checked at runtime.

```bash
open ios/StivApp.xcodeproj
```

[DEPLOYING.md](DEPLOYING.md) walks through signing, installing on a test
iPhone, Release builds for the field, and TestFlight.

The simulator cannot run a survey — no camera, no world tracking, no depth —
but it does run the app's tests, which stand in for all three. To look at the
screens without Xcode:

```bash
ios/tools/run.sh           # build, then launch in the newest iPhone simulator
ios/tools/run.sh --logs    # the same, streaming the app's console output
```

In VS Code the same is *Terminal → Run Task → iOS: run app in simulator*.

### Testing

```bash
ios/tools/test.sh          # StivKit on the Mac, then the app in a simulator
ios/tools/test.sh device   # the same on a connected iPhone, plus the ARKit checks
```

The first runs in about a minute and should pass before every commit. It covers
the numerics against the paper and the Python, the pixel-buffer readers, the
ARKit conversions, the water-level fit, the pose cross-check, and an end-to-end
test that films a rendered river with a swaying hand-held camera and checks the
velocity comes back. Two field checks — a null test on still water and a timed
float — are worth more than all of it. [TESTING.md](TESTING.md) has what each
layer proves, what it cannot, and the field protocol.

## What this version does not do

- **Discharge.** Surface velocity is not discharge; converting needs depths and
  a velocity index. `stiv/discharge.py` does it on the desktop.
- **Save the video.** Only the space-time images are kept, so a recording cannot
  be re-analysed with different geometry — though the frame step *can* be
  retuned afterwards, since every frame is recorded and `St` is applied at
  analysis time.
- **Record for as long as you like.** Every frame is kept in memory, which is
  what makes the frame step retunable, and a recording is held to a 300 MB
  budget. The defaults fit comfortably; many long search lines do not, and the
  app then shortens the recording, or thins the lines, and says so before it
  starts.
- **Glassy water.** STIV measures the motion of a pattern. No pattern, no
  measurement, in any implementation.
- **Rolling shutter.** In an oblique view a search line spans image rows, each
  read at a slightly different time, which tilts streaks and biases `φ`. iOS
  exposes no readout-time API, so it is not corrected; the null test in
  [TESTING.md](TESTING.md#4-in-the-field) is how to see whether it matters.

## Exports

`velocity.csv`, `summary.json`, `report.txt` — columns matching the desktop's —
`velocity_map.png`, and `survey.yaml`, which expresses the run as a
`stiv.Config`. The flow map is the results screen's: an arrow per station, its
length proportional to the speed, as the desktop's `velocity_map.png` -- but
drawn in plan, true to scale on the water, rather than over a frame, because a
phone looks across a river at a shallow angle and in its own view a far-bank
arrow would come out several times shorter than the same speed near by. They are saved
as soon as each analysis finishes, in a folder per recording that the Files app
shows under *On My iPhone → STIV*, and the share button sends the same files. The phone's
ray–plane mapping is a plane-to-plane projective map, which is exactly what
`stiv.geometry.HomographyRectifier` fits, so it exports losslessly as ground
control points. Fill in `video.path` with footage of the same section and
`stiv run survey.yaml` reproduces the analysis on the desktop.
