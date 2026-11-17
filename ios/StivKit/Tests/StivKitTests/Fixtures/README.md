# Golden fixtures

Generated, not authored. From the repository root, against the repository's
own `stiv` package (not an installed copy, which may be stale):

```bash
PYTHONPATH=$PWD python3 -I ios/tools/make_golden_fixture.py
```

That writes `golden_sti.f32` and `golden.json` here, and `GoldenFixtureTests`
and `GoldenAnalysisTests` compare the Swift port against the Python one, mostly
to ~1e-9:

- `stiv.sti.preprocess`, the STI conditioning, also on an STI with missing
  (NaN) samples, which are filled with their column's mean;
- the structure tensor and the Eq. (14) estimate, with the box window and with
  the Gaussian one, and the histogram-peak estimator on the box field;
- the window grids of sizes whose step lands on a rounding half (Python's
  `round` goes to even);
- an STI with featureless windows, which Eq. (14) must leave out
  (`stiv.orientation.flat_windows`);
- the analysis stage: quality-control flags, the profile median filter,
  interpolation and blanking, in their pipeline order; the direction check;
  the frame-step advice; and the velocity spread.

Without the files those tests skip, so `swift test` still passes on a machine
that has never seen numpy — but the port is then only checked against
synthetic slopes, which is a weaker guarantee.

**Keep these files committed.** They are generated, but they are *reference*
data: their whole purpose is to pin the Swift port to a known Python run, and
while they are untracked the parity tests skip everywhere except the one machine
that happened to run the generator -- which is the same as not having them.
They are ~120 KB together.

Regenerate whenever `stiv/orientation.py`, `stiv/sti.py` or the analysis
stage of `stiv/analysis.py` (quality control, the median filter, interpolation,
the direction check, the frame-step advice) changes.

Sections added after the fixture was first committed -- `gaussian_field`,
`window_grids`, `analysis`, `peak_estimate` and `preprocess_missing` -- are optional: their tests skip, with a message
naming the missing section, until the fixture is regenerated. To try a new
fixture before replacing these files, write it elsewhere and point the tests at
it:

```bash
PYTHONPATH=$PWD python3 -I ios/tools/make_golden_fixture.py --out /tmp/golden
STIV_GOLDEN_FIXTURE_DIR=/tmp/golden swift test --package-path ios/StivKit
```

## What the fixture does not pin

STI *construction* is not compared, and on one point the two pipelines differ
on purpose. The desktop stacks one STI row per decoded frame and assumes
`St = frame_step / fps`, only warning when the container's timestamps show an
irregular frame rate (`stiv.video.check_frame_timing`). The phone resamples
its rows onto a uniform time grid using `ARFrame.timestamp`
(`StiRecording.finish`), because those timestamps are precise and a throttled
phone does drop frames. On a recording with dropped frames the phone's STI is
therefore not the one the desktop would build from the same footage. The
fixture starts from a finished STI, after that step.
