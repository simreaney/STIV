# Golden fixtures

Generated, not authored. Run once on a machine with the Python package
installed:

```bash
pip install -e .                          # from the repository root
python ios/tools/make_golden_fixture.py
```

That writes `golden_sti.f32` and `golden.json` here, and
`GoldenFixtureTests` starts comparing the Swift structure tensor against the
Python one to ~1e-9. Without them those tests skip, so `swift test` still
passes on a machine that has never seen numpy — but the port is then only
checked against synthetic slopes, which is a weaker guarantee.

Regenerate whenever `stiv/orientation.py` changes.
