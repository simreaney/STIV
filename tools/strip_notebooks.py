"""Keep the committed notebooks free of run outputs and per-machine metadata.

Running the wizard saves into the .ipynb a frame of whatever footage was loaded
(often a megabyte or more of base64, and footage that is kept out of the repo
on purpose), widget views that are dead once the kernel stops, and the local
kernel's name and exact Python version.  None of it belongs in git.

    python3 tools/strip_notebooks.py            # strip notebooks/*.ipynb in place
    python3 tools/strip_notebooks.py --check    # fail if any needs stripping (CI)
    python3 tools/strip_notebooks.py --filter   # stdin -> stdout, for git

Set up once per clone, git then commits the stripped notebook while your
working copy keeps its outputs (.gitattributes routes *.ipynb through it):

    git config filter.strip-notebook.clean "python3 tools/strip_notebooks.py --filter"

Standard library only, so it runs before (or without) installing the package.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))

#: The generic kernel, rather than whatever environment last ran it.
KERNELSPEC = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
#: Notebook-level metadata worth keeping; the rest (widget state, editor
#: settings, interpreter hashes) is written by the machine that ran it.
KEEP_NB_METADATA = ("kernelspec", "language_info")
#: Cell metadata worth keeping; execution timings, scroll and collapse state go.
KEEP_CELL_METADATA = ("tags",)


def strip(nb: dict) -> dict:
    """The notebook with outputs, execution counts and local metadata removed."""
    nb = copy.deepcopy(nb)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell["metadata"] = {k: v for k, v in cell.get("metadata", {}).items() if k in KEEP_CELL_METADATA}

    meta = {k: v for k, v in nb.get("metadata", {}).items() if k in KEEP_NB_METADATA}
    meta["kernelspec"] = dict(KERNELSPEC)
    if "language_info" in meta:
        meta["language_info"] = {k: v for k, v in meta["language_info"].items() if k != "version"}
    nb["metadata"] = meta
    return nb


def dumps(nb: dict) -> str:
    """Serialised the way Jupyter and VS Code write it, so saves don't churn."""
    return json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False, separators=(",", ": ")) + "\n"


def _filter() -> int:
    text = sys.stdin.read()
    try:
        out = dumps(strip(json.loads(text)))
    except (ValueError, AttributeError, TypeError):
        out = text  # not a notebook we understand (e.g. mid-merge): leave it to git
    sys.stdout.write(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report unstripped notebooks and exit 1")
    mode.add_argument("--filter", action="store_true", help="strip stdin to stdout (git clean filter)")
    args = ap.parse_args(argv)

    if args.filter:
        return _filter()

    dirty = []
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text(encoding="utf-8"))
        clean = strip(nb)
        if clean == nb:
            continue
        dirty.append(path.relative_to(ROOT))
        if not args.check:
            path.write_text(dumps(clean), encoding="utf-8")

    for path in dirty:
        print(f"{'needs stripping' if args.check else 'stripped'}: {path}", flush=True)
    if args.check and dirty:
        print("Run `python3 tools/strip_notebooks.py` and commit the result.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
