# Testing the iPhone app

Four layers, each catching what the one before it cannot. The first two run on
a Mac in about a minute and should pass before every commit; the third needs a
phone on a cable; the fourth needs a river.

| layer | runs on | command | time | proves |
|---|---|---|---|---|
| 1. numerics | Mac | `ios/tools/test.sh kit` | ~10 s | the Swift port matches the paper and the Python |
| 2. app | iOS Simulator | `ios/tools/test.sh app` | ~1 min | the app turns posed frames into the right velocity |
| 3. device | iPhone | `ios/tools/test.sh device` | ~1 min | ARKit behaves the way layers 1–2 assume |
| 4. field | iPhone, outdoors | by hand, below | an hour | the whole thing works on real water |

`ios/tools/test.sh` with no argument runs layers 1 and 2. Or open
`ios/StivApp.xcodeproj`, pick a simulator, and press ⌘U for layer 2.

## 1. Numerics — StivKit on the Mac

```bash
cd ios/StivKit && swift test
```

The paper's method, ported to Swift with no ARKit and no UIKit, so it runs
without a simulator. Three of the tests exist to catch the ways a
structure-tensor port fails while still producing plausible numbers: an
inverted eigenvector branch (returns `-1/tan φ`), a flipped gradient sign
(`-tan φ`), and transposed axes (`1/tan φ`).

`GoldenFixtureTests` compares the Swift structure tensor and STI preprocessing
against a recorded Python run to 1e-9 — which is what catches a boundary rule
off by one pixel, or a histogram bin edge counted the other way. The fixture is
committed. Regenerate it whenever `stiv/sti.py` or `stiv/orientation.py`
changes:

```bash
pip install -e .                          # repository root
python ios/tools/make_golden_fixture.py
```

What it cannot see: anything the app does before the space-time images exist.
The pipeline tests synthesise STIs directly.

## 2. The app — in the simulator

```bash
ios/tools/test.sh app
```

`StivAppTests` is an app-hosted test bundle: it runs inside `STIV.app` and
reaches the app's internals with `@testable import STIV`. The simulator has no
camera, no world tracking and no depth, so the tests stand in for all three.
They build the CoreVideo buffers ARKit would deliver and render posed camera
frames of a scene whose answer is known.

| file | covers |
|---|---|
| `FrameBufferTests` | luma and depth readers: padded strides, bilinear sampling, out-of-frame as *missing* not black, confidence gating, refusing an unexpected depth format |
| `ARAdapterTests` | `simd` → StivKit conversions read column-major (a transpose would mirror everything), depth back-projection as Z-depth through the pixel centre, water-outline containment, the recording plan, the pose correction's composition, and the one shared projection being StivKit's |
| `ViewportTests` | the camera feed, overlay and taps share the *whole* screen, not the safe area (hosted in a real window); the aspect-fill mapping round-trips, crops as ARKit does, and turns the image the right way in every orientation |
| `WaterPlaneFitterTests` | the water-level fit: a bank alone reads at its foot (within 2 cm), water noise up to 5 cm does not bias it (within 5 mm), a bank filling most of the box does not lift it, nor does a 25% contamination or a small cluster of low reflections; out-of-range and low-confidence returns are dropped; readiness needs time, a sweep and two thousand samples, not one rich frame; storage stays bounded; the drawn box is the sampled region |
| `PoseResidualTests` | the drift check, on rendered static ground: a correct pose leaves nothing to correct, an injected 2 px yaw error is measured and undone, a sideways slip is corrected on the bank without being carried into the far field, an error over the cap is reported not applied, corrections ramp in and are held then ramped out rather than snapped, patches stay off the water and clear of the whole search line, the patch search is the documented size, lost tracking is recorded even when the check is not armed; and on synthetic patches, a correct pose is left alone whether the patches sit at one depth (where the fit falls back to rotation) or many, while a 2 px yaw is still corrected |
| `SyntheticRiverTests` | end to end: a textured surface moving at a known speed, filmed by a swaying hand-held camera, through `RecordingPlan.sampleRow` and the full analysis |
| `RecordingPreflightTests` | before Start: the memory budget shortens, then thins, using the layout's own sample counts, and the defaults fit a typical survey; a view facing away, held flat, or with the section mostly off screen is refused |
| `CaptureSessionTests` | the frame queue ends a recording at its duration, progress is published at a fraction of the frame rate, an interruption drops the recording and its end never overtakes its start, a refused camera is reported as a permission problem |
| `SurveyModelTests` | the tap sequence and undo; and, from camera views the tests supply, a whole recording: the layout rebuilt from the view at Start, the budget applied, the double-stop and stop-then-start races, an abandoned hand-over dropped, retunes run one at a time and capped where the STIs would be too short (at the median frame period, as StivKit grids them), every result saved to its recording's folder in Documents and overwritten by a retune, interruptions and failures sending the user back to measure, Results → New and re-measuring starting clean, the idle timer and the heat warning |

`SyntheticRiverTests` is the one to understand. The app's central claim is
that sampling fixed *world* points through each frame's own pose removes camera
motion entirely, and this is where that claim is tested. The camera sways by
about a degree and a couple of centimetres, as a hand does. The river must come
back within 3% RMS across the section, with no station flagged. (On this ideal
scene it comes back within 0.3%; the slack is for legitimate changes to the
analysis, not for a few-percent bias.) As a control, the same sway sampled
without the pose must ruin the profile, which shows the sway was large enough
to test anything. A still-water version is the simulated twin of the field null
test below.

What the simulator cannot model: rolling shutter, motion blur, exposure
changes, lens distortion, real VIO drift, and real LiDAR on real water. Those
are what layers 3 and 4 are for.

To run one test:

```bash
xcodebuild test -project ios/StivApp.xcodeproj -scheme STIV \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -only-testing:StivAppTests/SyntheticRiverTests
```

### Adding a test

Numerics with no ARKit type in sight go in `StivKit/Tests`. Anything that
touches a pixel buffer, a `simd` matrix or the survey flow goes in
`ios/StivAppTests/`. It is a synchronized folder, so a new file needs no
project edit.

An `ARFrame` cannot be constructed outside ARKit, so code that needs testing
takes the planes, intrinsics and pose instead, and the `ARFrame` overload stays
a thin wrapper that unpacks them. `RecordingPlan.sampleRow`,
`WaterPlaneFitter.add(depth:…)` and `PoseResidualChecker.arm(depth:…)` /
`check(luma:…)` are the pattern. `SyntheticFrames.swift` has the buffer
builders, the renderer, the texture and a hand-held pose.

## 3. On a phone

```bash
ios/tools/test.sh device                  # first connected iPhone
TEAM=ABCDE12345 ios/tools/test.sh device  # if signing needs a team
```

The same bundle, on hardware. The phone needs to be set up for development
first — signing, Developer Mode, trust — as in [DEPLOYING.md](DEPLOYING.md).
`OnDeviceTests` skips in the simulator and runs here:

- **Projection parity.** StivKit's hand-written world-to-image projection
  against `ARCamera.projectPoint`, to under a pixel. The unit tests prove the
  geometry is self-consistent; only this proves its axes point the same way
  ARKit's do. A sign or a transpose wrong here mirrors every measurement while
  every other test passes.
- **Viewport parity.** The aspect-fill mapping that sizes the depth reticle,
  against `ARCamera.projectPoint` in each supported orientation. The reticle is
  a centred square, so only its scale is at stake, but this is the check that
  says the mapping is ARKit's.
- **Buffer formats.** The captured image is bi-planar 4:2:0 YCbCr. The depth map
  is Float32 with the same aspect ratio as the image. Apple documents neither
  as a contract, so this is the test to rerun on a new iPhone or iOS version.
- **LiDAR returns.** High-confidence depth actually arrives (Pro models only).

Keep the phone unlocked. Allow camera access on the first run. Then move the
phone slowly over a textured surface within a couple of metres until the run
finishes. A test that never sees normal tracking skips rather than fails.

The parity check also runs once per session in any debug build of the app
itself; watch the Xcode console for `projection parity`.

Run this layer after any change under `ios/StivApp/AR/`, and on each new iPhone
model or major iOS release.

## 4. In the field

Two checks are worth more than everything above, because they are the only ones
that see real water, real light and a real hand. Do both at each new site and
with each new phone, and keep the exports. Use a Release build, and go through
the field-day checklist in [DEPLOYING.md](DEPLOYING.md#before-a-field-day)
first.

**Null test.** Point the phone at still water, or at a textured wall, and
record the full duration. No station may report a confident non-zero
velocity. Either the stations are flagged, or they read within a few cm/s
of zero. This catches a sign error, a rolling-shutter bias and pose noise
leaking into the STI, all at once. If it fails, nothing else measured that day
can be trusted.

**Known velocity.** Measure out a reach parallel to the flow: two marks on the
bank 10 m apart, say, either side of the section. Drop a float well upstream:
an orange sits low enough to follow the surface, not the wind. Time it between
the marks, and note where across the channel it passed. Repeat at least five
times across the width. Record the section from the bank as usual. Compare each
float's speed with the station it passed. STIV measures surface velocity, so no
velocity index applies. The desktop package reaches ~3% relative RMSE on
synthetic footage. A phone on a real bank will be worse, and the point of this
check is to find out by how much.

Worth noting alongside each run:

| | |
|---|---|
| site, date, time, weather | surface texture depends on wind and light |
| phone model, iOS version | |
| camera height above water | the app's figure, and a tape measure's if you can reach |
| water level method | waterline sweep, on-surface, or typed height |
| frame step used | and whether the results screen suggested another |
| null-test result | max \|v\| over unflagged stations |
| float speeds and stations | each run, not just the mean |
| exports | `summary.json`, `velocity.csv`, `survey.yaml`, `report.txt` |

`survey.yaml` reruns the same analysis on the desktop. Film the section with
any camera at the same time, fill in `video.path`, and run
`stiv run survey.yaml` for an independent comparison against the Python
pipeline.
