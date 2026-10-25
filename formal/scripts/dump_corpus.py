# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Dump the corpus programs as SDFG JSON files for the Lean tools.

Usage (from the repository root, with DaCe importable)::

    python formal/scripts/dump_corpus.py [--out formal/tests/corpus] [--no-simplify]

For every program in ``corpus_programs.PROGRAMS`` this writes
``<name>.sdfg.json`` (simplified unless ``--no-simplify``) and, for a few
programs, transformed variants (``<name>.tiled.sdfg.json``,
``<name>.expanded.sdfg.json``), followed by the SDFGs of ``tests/schedule_tree/roundtrip_test.py``
(``rt.<test>.sdfg.json``, see ``roundtrip_corpus.py``; ``--no-roundtrip`` skips them).  Alongside each SDFG it also writes the schedule
tree of the same SDFG as ``<name>.stree.json`` (machine-readable, see
``dace.sdfg.analysis.schedule_tree.serialize``) and ``<name>.stree.txt``
(``as_string()`` for human review and Lean-side pretty-printer checks).
Inputs/outputs for differential testing are produced by ``difftest.py``.
The corpus is not committed; CI regenerates it with this script followed by
``difftest.py --record --seed 0``.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus_programs import PROGRAMS  # noqa: E402
from roundtrip_corpus import collect as collect_roundtrip  # noqa: E402

import dace  # noqa: E402
from dace.sdfg.analysis.schedule_tree import serialize as stree_serialize  # noqa: E402
from dace.transformation.dataflow import MapTiling  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _scrub_paths(obj):
    """Make absolute paths under the repository root relative (debug info, ``source_files``).

    The generated corpus must not depend on the machine it was generated on.
    """
    if isinstance(obj, dict):
        return {k: _scrub_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_paths(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(REPO_ROOT + os.sep):
        return os.path.relpath(obj, REPO_ROOT)
    return obj


def save(sdfg: dace.SDFG, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, f"{name}.sdfg.json")
    with open(path, "w") as f:
        f.write(dace.serialize.dumps(_scrub_paths(sdfg.to_json())))
        # End with a newline like the other generated JSON files.
        f.write("\n")
    # The tree of the SDFG as stored: reloading fixes the edge order to the file's (in-memory edge order may differ,
    # and the formal pair checker matches the tree against the stored SDFG).
    _save_stree(dace.SDFG.from_file(path), out_dir, name)
    return path


def _save_stree(sdfg: dace.SDFG, out_dir: str, name: str) -> None:
    """Write ``<name>.stree.json`` and ``<name>.stree.txt`` for the given (stored) SDFG.

    The schedule tree is built from a copy (``in_place=False``) so that the SDFG that was dumped as ``<name>.sdfg.json``
    is left unchanged.
    """
    tree = sdfg.as_schedule_tree(in_place=False)
    with open(os.path.join(out_dir, f"{name}.stree.json"), "w") as f:
        json.dump(_scrub_paths(stree_serialize.to_json(tree)), f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(out_dir, f"{name}.stree.txt"), "w") as f:
        f.write(tree.as_string())
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "tests", "corpus"))
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="subset of program names")
    parser.add_argument(
        "--no-roundtrip", action="store_true", help="skip the SDFGs of tests/schedule_tree/roundtrip_test.py"
    )
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

    if args.no_roundtrip:
        return
    for name, sdfg, note in collect_roundtrip():
        if args.only and name not in args.only:
            continue
        if sdfg is None:
            print("skipped", name, "--", note)
            continue
        print("wrote", save(sdfg, args.out, name))


if __name__ == "__main__":
    main()
