# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Dump the corpus programs as SDFG JSON files for the Lean tools.

Usage (from the repository root, with DaCe importable)::

    python formal/scripts/dump_corpus.py [--out formal/tests/corpus] [--no-simplify]

For every program in ``corpus_programs.PROGRAMS`` this writes
``<name>.sdfg.json`` (simplified unless ``--no-simplify``) and, for a few
programs, transformed variants (``<name>.tiled.sdfg.json``,
``<name>.expanded.sdfg.json``).  Inputs/outputs for differential testing are
produced by ``difftest.py``.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dace  # noqa: E402
from corpus_programs import PROGRAMS  # noqa: E402
from dace.transformation.dataflow import MapTiling  # noqa: E402


def save(sdfg: dace.SDFG, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, f"{name}.sdfg.json")
    sdfg.save(path, compress=False)
    # Keep the committed files friendly to the end-of-file pre-commit hook.
    with open(path, "a") as f:
        f.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "tests", "corpus"))
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="subset of program names")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, prog, _ in PROGRAMS:
        if args.only and name not in args.only:
            continue
        sdfg = prog.to_sdfg(simplify=not args.no_simplify)
        sdfg.validate()
        print("wrote", save(sdfg, args.out, name))

        if name == "matmul":
            tiled = prog.to_sdfg(simplify=True)
            tiled.apply_transformations(MapTiling, options={"tile_sizes": (2, 2, 3)})
            tiled.validate()
            print("wrote", save(tiled, args.out, f"{name}.tiled"))
        if name == "reduce_rows":
            expanded = prog.to_sdfg(simplify=True)
            expanded.expand_library_nodes()
            expanded.validate()
            print("wrote", save(expanded, args.out, f"{name}.expanded"))


if __name__ == "__main__":
    main()
