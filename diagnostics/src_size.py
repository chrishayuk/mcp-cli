#!/usr/bin/env python
"""
src_size.py  <path-or-vcs-url>
================================
Measure how much space a Python package (plus its **runtime** dependencies)
occupies when built **from source**, without counting the extra build tools we
install inside a temporary virtual-env (``pip``, ``setuptools``, ``wheel``,
etc.).

What you get
------------
1. **Raw source** - size of the directory / checkout you point at.
2. **Built artefacts** - combined size of the sdist + wheel produced by
   ``python -m build``.
3. **Runtime tree** - bytes taken by the package **and its runtime deps** once
   installed, *excluding* build-time tooling.
4. *(optional)* a **breakdown** of every runtime distribution, sorted
   large → small.

CLI flags
---------
```
--no-deps        Skip installing dependencies (handy for library-only size)
--breakdown, -b  Show per-package size table
--include-tools  Include build tools (pip/setuptools/wheel) in totals + table
```

> **Note**The script still seeds pip inside the venv so it works on
> pip-less interpreters - those files are just ignored by default in the
> final numbers.
"""

from __future__ import annotations

import argparse
import ensurepip
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import List, Tuple

BYTES_IN_MB = 1_048_576
BUILD_TOOLS = {"pip", "setuptools", "wheel"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def du(path: pathlib.Path) -> int:
    """Recursive size of *path* in bytes."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run(cmd: List[str]) -> None:
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def build_artefacts(src: pathlib.Path, out: pathlib.Path) -> List[pathlib.Path]:
    run([sys.executable, "-m", "pip", "install", "-q", "build"])
    run([sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", out, src])
    return list(out.glob("*.*"))


def pick_one(artefacts: List[pathlib.Path]) -> pathlib.Path:
    return next((a for a in artefacts if a.suffix == ".whl"), artefacts[0])


def create_venv(venv: pathlib.Path) -> pathlib.Path:
    run([sys.executable, "-m", "venv", venv])
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run([str(py), "-m", "ensurepip", "--upgrade"])
    run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"])
    return py


def dist_sizes(py: pathlib.Path) -> List[Tuple[str, int]]:
    """Return list of (dist_name, size_bytes) for every distribution in venv."""
    code = r'''
import importlib.metadata as m, pathlib, json, os
sizes = {}
for dist in m.distributions():
    total = 0
    for entry in dist.files or []:
        p = pathlib.Path(dist.locate_file(entry))
        if p.is_file():
            try:
                total += p.stat().st_size
            except FileNotFoundError:
                pass
    sizes[dist.metadata['Name']] = total
print(json.dumps(sizes))
'''
    out = subprocess.check_output([str(py), "-c", code], text=True)
    data = json.loads(out)
    return sorted(((k, v) for k, v in data.items()), key=lambda kv: kv[1], reverse=True)


def canonical(name: str) -> str:
    return name.lower().replace("_", "-")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(src: str, include_deps: bool, show_breakdown: bool, include_tools: bool) -> None:
    ensurepip.bootstrap()  # ensure pip for outer interpreter

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = pathlib.Path(tmp_s)

        # 1. Obtain source ---------------------------------------------------
        if pathlib.Path(src).is_dir():
            src_dir = pathlib.Path(src).resolve()
        else:
            src_dir = tmp / "clone"
            run(["git", "clone", "--depth", "1", src, src_dir])

        print(f"Raw source:     {du(src_dir)/BYTES_IN_MB:.2f} MB")

        # 2. Build artefacts -------------------------------------------------
        artefacts = build_artefacts(src_dir, tmp)
        print(f"Sdist+wheel:    {sum(p.stat().st_size for p in artefacts)/BYTES_IN_MB:.2f} MB")
        artefact = pick_one(artefacts)

        # 3. Install into temp venv -----------------------------------------
        py = create_venv(tmp / "venv")
        install_cmd = [str(py), "-m", "pip", "install", "-q", str(artefact)]
        if not include_deps:
            install_cmd.insert(5, "--no-deps")
        run(install_cmd)

        dists = dist_sizes(py)
        # Filter build tools unless user asked to keep them
        runtime_dists = [(n, sz) for n, sz in dists if include_tools or canonical(n) not in {canonical(t) for t in BUILD_TOOLS}]

        total_runtime = sum(sz for _, sz in runtime_dists)
        print(f"Runtime tree:   {total_runtime/BYTES_IN_MB:.2f} MB" + (" (includes build tools)" if include_tools else ""))

        if show_breakdown:
            print("\nBreakdown (descending):")
            for name, sz in runtime_dists:
                print(f"  {name:<25} {sz/BYTES_IN_MB:7.2f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Measure on-disk size of a Python package built from source.")
    p.add_argument("source", help="Path, git/https URL, or anything pip understands.")
    p.add_argument("--no-deps", action="store_true", help="Skip installing dependencies inside the tmp venv.")
    p.add_argument("--breakdown", "-b", action="store_true", help="Show per-package size contribution (runtime only).")
    p.add_argument("--include-tools", action="store_true", help="Include build tools (pip/setuptools/wheel) in totals and table.")
    args = p.parse_args()
    main(args.source, include_deps=not args.no_deps, show_breakdown=args.breakdown, include_tools=args.include_tools)
