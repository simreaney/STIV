# STIV for iPhone

Space-Time Image Velocimetry from the built-in camera, with the LiDAR scanner
supplying real-world scale. Same method as the Python package in the repository
root — Fujita, Watanabe & Tsubaki (2007), Eqs. (1)–(16) — with the two
drone-specific parts replaced by what a phone actually has.

```
ios/
  StivKit/          the numerics: a Swift port of the paper. No ARKit, no UIKit.
  StivApp/          the app: ARKit capture, the six-screen survey, results.
  StivApp.xcodeproj hand-authored, Xcode 16 synchronized folders.
  project.yml       XcodeGen fallback, if the project above will not open.
  tools/            the golden-fixture generator.
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
from a bank it is the nearest thing in the scene. Everything after that is
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

## Building

Requires **Xcode 16+** and an **iPhone 12 Pro or later** for LiDAR. Non-Pro
iPhones run the app with a manually entered camera height — there is no
`UIRequiredDeviceCapabilities` value for a depth sensor, so LiDAR can only be
checked at runtime.

```bash
open ios/StivApp.xcodeproj
```

The simulator gives you nothing useful: no camera, no world tracking, no depth.

### Testing the numerics

```bash
cd ios/StivKit && swift test
```

Runs on a Mac without a simulator. Three of the tests exist specifically to
catch the ways a structure-tensor port fails while still producing plausible
numbers: an inverted eigenvector branch (returns `-1/tan φ`), a flipped gradient
sign (`-tan φ`), and transposed axes (`1/tan φ`).

For the strongest check, generate the golden fixture once — it compares the Swift
structure tensor against the Python one to 1e-9, which is what catches a
boundary rule off by one pixel or a histogram bin edge counted the other way:

```bash
pip install -e .                          # repository root
python ios/tools/make_golden_fixture.py
cd ios/StivKit && swift test
```

Without the fixture those tests skip rather than fail.

### On-device checks

`ProjectionParity.swift` runs once per session in debug builds and compares the
hand-written world-to-image projection against `ARCamera.projectPoint`. It is
the only way to verify the *conventions* match ARKit's — the unit tests prove the
geometry is self-consistent, not that its axes point the same way ARKit's do.
Watch the console for `projection parity`.

Two field checks are worth more than any of the above:

1. **Static-scene null test.** Point at still water, or a wall, and record. The
   velocity must come out ≈ 0. This catches a sign error, a rolling-shutter
   bias, and pose noise leaking into the STI, all at once.
2. **Known velocity.** Time a float over a measured distance and compare. The
   desktop package hits ~3% relative RMSE on synthetic footage; a phone on a
   real bank will be worse, and the point is to find out by how much.

## What this version does not do

- **Discharge.** Surface velocity is not discharge; converting needs depths and
  a velocity index. `stiv/discharge.py` does it on the desktop.
- **Save the video.** Only the space-time images are kept, so a recording cannot
  be re-analysed with different geometry — though the frame step *can* be
  retuned afterwards, since every frame is recorded and `St` is applied at
  analysis time.
- **Glassy water.** STIV measures the motion of a pattern. No pattern, no
  measurement, in any implementation.
- **Rolling shutter.** In an oblique view a search line spans image rows, each
  read at a slightly different time, which tilts streaks and biases `φ`. iOS
  exposes no readout-time API; `CaptureSettings.rollingShutterRowSeconds` is
  there to be calibrated, and defaults to off.

## Exports

`velocity.csv`, `summary.json`, `report.txt` — columns matching the desktop's —
and `survey.yaml`, which expresses the run as a `stiv.Config`. The phone's
ray–plane mapping is a plane-to-plane projective map, which is exactly what
`stiv.geometry.HomographyRectifier` fits, so it exports losslessly as ground
control points. Fill in `video.path` with footage of the same section and
`stiv run survey.yaml` reproduces the analysis on the desktop.
