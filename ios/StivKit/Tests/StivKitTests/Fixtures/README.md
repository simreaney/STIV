# Golden fixtures

Generated, not authored. Run once on a machine with the Python package
installed:

```bash
pip install -e .                          # from the repository root
python ios/tools/make_golden_fixture.py
```

That writes `golden_sti.f32` and `golden.json` here, and
`GoldenFixtureTests` starts comparing the Swift port against the Python one to
~1e-9: `stiv.sti.preprocess` (the STI conditioning) as well as the structure
tensor and the Eq. (14) estimate. Without them those tests skip, so `swift test` still
passes on a machine that has never seen numpy — but the port is then only
checked against synthetic slopes, which is a weaker guarantee.

**Keep these files committed.** They are generated, but they are *reference*
data: their whole purpose is to pin the Swift port to a known Python run, and
while they are untracked the parity tests skip everywhere except the one machine
that happened to run the generator -- which is the same as not having them.
They are ~120 KB together.

Regenerate whenever `stiv/orientation.py` or `stiv/sti.py` changes.
