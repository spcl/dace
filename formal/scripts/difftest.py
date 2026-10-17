# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Differential testing harness for the DaCe formal semantics (milestone M3).

For every ``formal/tests/corpus/<base>[.variant].sdfg.json`` this script either

* ``--record``  loads the SDFG from the JSON file, takes the inputs from the base
  program's argument factory in :mod:`corpus_programs`, compiles and runs the
  SDFG, and writes ``<name>.inputs.json`` and ``<name>.expected.json``; or
* ``--check`` (alias ``--lean``) runs ``lake exe dace-interp <ir> <ir.json>
  <inputs.json>`` inside ``formal/``, parses its stdout as JSON and compares the
  resulting containers against ``<name>.expected.json`` with dtype-aware
  tolerances.  ``--ir`` selects the intermediate representation: ``sdfg``
  (default, uses ``<name>.sdfg.json``) or ``stree`` (uses ``<name>.stree.json``,
  the schedule tree of the same SDFG).  Both must compute the same outputs, so
  they are compared against the same ``<name>.expected.json``.

When neither mode is given, the script records if any expected file is missing
and checks otherwise.

JSON format (fixed; the Lean reader is written against it)::

    {
      "program": "matmul",
      "sdfg": "matmul.tiled.sdfg.json",
      "symbols": {"M": 4, "K": 6, "N": 5},
      "containers": {
        "A": {"dtype": "float64", "shape": [4, 6], "data": [/* row-major */]},
        "a": {"dtype": "float64", "shape": [], "data": [2.0]}
      }
    }

Scalars (DaCe ``Scalar`` descriptors) use ``shape: []`` with a one-element
``data``.  ``expected.json`` has the same shape but only ``program``/``sdfg``/
``containers``.  Keys are sorted, indentation is two spaces, files end with a
trailing newline.
"""

import argparse
import copy
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dace  # noqa: E402
from corpus_programs import PROGRAMS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.normpath(os.path.join(HERE, "..", "tests", "corpus"))
FORMAL_DIR = os.path.normpath(os.path.join(HERE, ".."))
FACTORIES = {name: factory for name, _prog, factory in PROGRAMS}

# Default np.allclose rtol per NumPy dtype kind/name.
RTOL = {"float32": 1e-5, "float64": 1e-9}


# --------------------------------------------------------------------------- #
# JSON (de)serialization of containers
# --------------------------------------------------------------------------- #
def _json_scalar(value, npdt: np.dtype):
    """Convert one NumPy value to a JSON-native number/bool with full precision."""
    kind = npdt.kind
    if kind == "b":
        return bool(value)
    if kind in ("i", "u"):
        return int(value)
    # Floating point: Python's json emits the round-trippable repr of a float.
    return float(value)


def container_entry(desc, value) -> dict:
    """Serialize a container value against its DaCe descriptor."""
    npdt = np.dtype(desc.dtype.type)
    arr = np.asarray(value, dtype=npdt)
    flat = arr.reshape(-1, order="C")
    if isinstance(desc, dace.data.Scalar):
        shape = []
        data = [_json_scalar(flat[0], npdt)] if flat.size else [_json_scalar(npdt.type(0), npdt)]
    else:
        shape = [int(d) for d in arr.shape]
        data = [_json_scalar(v, npdt) for v in flat]
    return {"dtype": npdt.name, "shape": shape, "data": data}


def entry_to_array(entry: dict) -> np.ndarray:
    """Rebuild a NumPy array from a serialized container entry."""
    npdt = np.dtype(entry["dtype"])
    shape = tuple(entry["shape"])
    arr = np.array(entry["data"], dtype=npdt)
    if shape == ():
        return arr.reshape(())
    return arr.reshape(shape, order="C")


def dump_json(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(obj, sort_keys=True, indent=2))
        f.write("\n")


# --------------------------------------------------------------------------- #
# Corpus discovery
# --------------------------------------------------------------------------- #
def discover(corpus: str, only):
    """Yield (name, base, sdfg_path) for each committed corpus SDFG JSON."""
    out = []
    for path in sorted(glob.glob(os.path.join(corpus, "*.sdfg.json"))):
        name = os.path.basename(path)[: -len(".sdfg.json")]
        base = name.split(".")[0]
        if only and name not in only and base not in only:
            continue
        out.append((name, base, path))
    return out


def symbol_values(sdfg: dace.SDFG, args: dict) -> dict:
    """Free symbols of the SDFG plus explicit symbol args, valued from ``args``."""
    names = {str(s) for s in sdfg.free_symbols}
    names.update(k for k in args if k not in sdfg.arrays)
    syms = {}
    for n in names:
        if n not in args:
            raise KeyError(f"symbol {n!r} has no value in the argument factory")
        syms[n] = int(args[n])
    return syms


# --------------------------------------------------------------------------- #
# NumPy sanity references (a few programs only)
# --------------------------------------------------------------------------- #
def sanity_reference(base: str, args: dict):
    """Return {container: expected_np} for programs we can cheaply cross-check."""
    if base == "axpy":
        return {"y": args["a"] * np.asarray(args["x"]) + np.asarray(args["y"])}
    if base == "matmul":
        return {"C": np.asarray(args["A"]) @ np.asarray(args["B"])}
    if base == "reduce_rows":
        return {"out": np.sum(np.asarray(args["A"]), axis=1)}
    return None


# --------------------------------------------------------------------------- #
# Record mode
# --------------------------------------------------------------------------- #
def record(corpus: str, only) -> int:
    failures = 0
    for name, base, sdfg_path in discover(corpus, only):
        if base not in FACTORIES:
            print(f"SKIP   {name}: no argument factory for base {base!r}")
            continue
        try:
            sdfg = dace.SDFG.from_file(sdfg_path)
            args = FACTORIES[base]()
            syms = symbol_values(sdfg, args)

            nonT = [n for n, d in sdfg.arrays.items() if not d.transient]
            inputs_containers = {n: container_entry(sdfg.arrays[n], args[n]) for n in nonT}

            call = copy.deepcopy(args)
            res = sdfg(**call)

            expected_containers = {n: container_entry(sdfg.arrays[n], call[n]) for n in nonT}
            ret_names = sorted(n for n in sdfg.arrays if n.startswith("__return"))
            if ret_names:
                rets = list(res) if isinstance(res, (list, tuple)) else [res]
                for rn, rv in zip(ret_names, rets):
                    expected_containers[rn] = container_entry(sdfg.arrays[rn], rv)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL   {name}: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        sdfg_fname = os.path.basename(sdfg_path)
        dump_json(
            os.path.join(corpus, f"{name}.inputs.json"),
            {"program": base, "sdfg": sdfg_fname, "symbols": syms, "containers": inputs_containers},
        )
        dump_json(
            os.path.join(corpus, f"{name}.expected.json"),
            {"program": base, "sdfg": sdfg_fname, "containers": expected_containers},
        )

        # Cheap NumPy cross-check for a few programs.
        ref = sanity_reference(base, args)
        note = ""
        if ref is not None:
            bad = []
            for cname, expected_np in ref.items():
                got = entry_to_array(expected_containers[cname])
                if not np.allclose(got, expected_np, rtol=1e-9, atol=0.0):
                    bad.append(cname)
            note = "  sanity OK" if not bad else f"  sanity MISMATCH {bad}"
            if bad:
                failures += 1
        print(f"RECORD {name}: symbols={syms} containers={sorted(expected_containers)}{note}")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# Check mode (Lean interpreter)
# --------------------------------------------------------------------------- #
def have_lake() -> bool:
    try:
        subprocess.run(["lake", "--version"], cwd=FORMAL_DIR, capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def compare_containers(got: dict, expected: dict, tol_scale: float):
    """Return a list of human-readable mismatch descriptions (empty == PASS)."""
    problems = []
    got_keys, exp_keys = set(got), set(expected)
    if got_keys != exp_keys:
        missing = exp_keys - got_keys
        extra = got_keys - exp_keys
        if missing:
            problems.append(f"missing containers {sorted(missing)}")
        if extra:
            problems.append(f"extra containers {sorted(extra)}")
    for cname in sorted(exp_keys & got_keys):
        exp_entry = expected[cname]
        got_entry = got[cname]
        try:
            exp_arr = entry_to_array(exp_entry)
            got_arr = entry_to_array({"dtype": exp_entry["dtype"], "shape": got_entry.get("shape", exp_entry["shape"]), "data": got_entry["data"]})
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{cname}: cannot parse ({exc})")
            continue
        if list(exp_entry["shape"]) != list(got_entry.get("shape", exp_entry["shape"])):
            problems.append(f"{cname}: shape {got_entry.get('shape')} != {exp_entry['shape']}")
            continue
        npdt = np.dtype(exp_entry["dtype"])
        if npdt.kind in ("i", "u", "b"):
            if not np.array_equal(exp_arr, got_arr):
                problems.append(f"{cname}: integer/bool values differ")
        else:
            rtol = RTOL.get(npdt.name, 1e-6) * tol_scale
            if not np.allclose(got_arr, exp_arr, rtol=rtol, atol=0.0, equal_nan=True):
                problems.append(f"{cname}: not close (rtol={rtol})")
    return problems


def check(corpus: str, only, tol_scale: float, ir: str = "sdfg") -> int:
    if not have_lake():
        print("SKIP all: `lake` not found on PATH; cannot run the Lean interpreter.")
        return 0
    failures = 0
    for name, base, sdfg_path in discover(corpus, only):
        inputs_path = os.path.join(corpus, f"{name}.inputs.json")
        expected_path = os.path.join(corpus, f"{name}.expected.json")
        if not (os.path.exists(inputs_path) and os.path.exists(expected_path)):
            print(f"SKIP   {name}: no recorded inputs/expected (run --record first)")
            continue

        # Select the IR file and the matching ``dace-interp`` subcommand. Both the SDFG and its schedule tree must
        # compute the same outputs, so they are checked against the same ``<name>.expected.json``.
        if ir == "stree":
            ir_path = os.path.join(corpus, f"{name}.stree.json")
            if not os.path.exists(ir_path):
                print(f"SKIP   {name}: no schedule tree ({name}.stree.json); run dump_corpus.py first")
                continue
        else:
            ir_path = sdfg_path

        proc = subprocess.run(
            ["lake", "exe", "dace-interp", ir, os.path.abspath(ir_path), os.path.abspath(inputs_path)],
            cwd=FORMAL_DIR,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 2:
            print(f"SKIP   {name}: dace-interp {ir} not implemented (exit 2)")
            continue
        if proc.returncode != 0:
            print(f"FAIL   {name}: dace-interp exit {proc.returncode}: {proc.stderr.strip()[:200]}")
            failures += 1
            continue
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            print(f"FAIL   {name}: dace-interp stdout is not JSON ({exc})")
            failures += 1
            continue

        with open(expected_path) as f:
            expected = json.load(f)

        if "error" in result:
            print(f"FAIL   {name}: dace-interp reported semantic error {result['error']!r} but a value was expected")
            failures += 1
            continue
        if "containers" not in result:
            print(f"FAIL   {name}: dace-interp output missing 'containers'")
            failures += 1
            continue

        problems = compare_containers(result["containers"], expected["containers"], tol_scale)
        if problems:
            print(f"FAIL   {name}: " + "; ".join(problems))
            failures += 1
        else:
            print(f"PASS   {name}")
    return 2 if failures else 0


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true", help="record inputs/expected by running DaCe")
    mode.add_argument("--check", "--lean", dest="check", action="store_true", help="run the Lean interpreter and compare")
    parser.add_argument("--only", nargs="*", default=None, help="subset of corpus names or program bases")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help="corpus directory (default: formal/tests/corpus)")
    parser.add_argument("--tol-scale", type=float, default=1.0, help="multiply all rtol tolerances in --check")
    parser.add_argument(
        "--ir",
        choices=["sdfg", "stree"],
        default="sdfg",
        help="intermediate representation to check: 'sdfg' (default) or 'stree' (schedule tree)",
    )
    args = parser.parse_args()

    corpus = os.path.abspath(args.corpus)
    only = set(args.only) if args.only else None

    if args.record:
        do_record = True
    elif args.check:
        do_record = False
    else:
        # Default: record if any expected file is missing, otherwise check.
        missing = [
            name
            for name, _base, _p in discover(corpus, only)
            if not os.path.exists(os.path.join(corpus, f"{name}.expected.json"))
        ]
        do_record = bool(missing)

    rc = record(corpus, only) if do_record else check(corpus, only, args.tol_scale, args.ir)
    sys.exit(rc)


if __name__ == "__main__":
    main()
