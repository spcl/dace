"""One-shot dumper for every HLFIR test fixture's SDFG.

Walks every ``tests/hlfir/*.f90`` file, builds its SDFG via the same
``SDFGBuilder`` the tests use, and saves the result to an inspection
directory (``/tmp/hlfir_sdfgs/`` by default).  Prints one status line
per fixture so failures are visible at a glance.

Usage:
    python3 tools/dump_hlfir_sdfgs.py [out_dir]

Pipelines:
    select_case.f90 needs the minimal ``hlfir-propagate-shapes`` pipeline
    (lift-cf-to-scf refuses to walk past fir.select_case).  Everything
    else gets the default pipeline.
"""

import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HLFIR_DIR = REPO / "dace" / "frontend" / "hlfir"
FIXTURES = REPO / "tests" / "hlfir"
# Put this checkout ahead of any other dace install on sys.path.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HLFIR_DIR))
sys.path.insert(0, str(HLFIR_DIR / "build"))

from hlfir_to_sdfg import SDFGBuilder, DEFAULT_PIPELINE  # noqa: E402

# Per-fixture pipeline overrides.  Most HLFIR tests use the minimal
# ``hlfir-propagate-shapes`` pipeline; only a handful want the full chain.
MINIMAL = "hlfir-propagate-shapes"
FIXTURE_PIPELINES: dict[str, str] = {
    # DO WHILE / EXIT needs cf→scf lifting before scf.while is visible.
    "do_loop_exit.f90": DEFAULT_PIPELINE,
}

# Fixtures that are not full-SDFG-buildable today — their real tests
# exercise IR or classifier state, not a constructed SDFG.
SKIP_FIXTURES: dict[str, str] = {
    "complex_struct.f90": "struct flattening produces dotted array names "
    "that NestedDict rejects in SDFG.add_array",
    "velocity_struct.f90": "flatten_structs_test only inspects IR text, "
    "not a built SDFG",
    "jagged_struct.f90": "flatten_structs_test only inspects IR text, "
    "not a built SDFG",
}


def subroutine_name(src: str) -> str:
    m = re.search(r"^\s*subroutine\s+(\w+)", src, re.M | re.I)
    if not m:
        raise RuntimeError("no `subroutine` found")
    return m.group(1)


def flang() -> str:
    for name in ("flang-new-21", "flang-new-20"):
        p = shutil.which(name)
        if p:
            return p
    raise RuntimeError("flang-new not on PATH")


def compile_hlfir(src_path: Path, work: Path) -> Path:
    hlfir = work / (src_path.stem + ".hlfir")
    subprocess.check_call([flang(), "-fc1", "-emit-hlfir", str(src_path), "-o", str(hlfir)])
    return hlfir


def dump_one(src_path: Path, out_dir: Path) -> tuple[str, str]:
    """Returns ``("ok" | "skip" | "fail", message)``."""
    if src_path.name in SKIP_FIXTURES:
        return ("skip", SKIP_FIXTURES[src_path.name])
    source = src_path.read_text()
    name = subroutine_name(source)
    pipeline = FIXTURE_PIPELINES.get(src_path.name, MINIMAL)
    with tempfile.TemporaryDirectory(prefix="hlfir_dump_") as td:
        work = Path(td)
        staged = work / src_path.name
        staged.write_text(source)
        hlfir = compile_hlfir(staged, work)
        builder = SDFGBuilder(str(hlfir), pipeline=pipeline)
        sdfg = builder.build()
        out_path = out_dir / f"{src_path.stem}.sdfg"
        sdfg.save(str(out_path))
        return ("ok", f"{name} -> {out_path}  ({len(list(sdfg.all_control_flow_blocks()))} blocks)")


def main(argv: list[str]) -> int:
    out_dir = Path(argv[1]) if len(argv) > 1 else Path("/tmp/hlfir_sdfgs")
    out_dir.mkdir(parents=True, exist_ok=True)

    fixtures = sorted(FIXTURES.glob("*.f90"))
    if not fixtures:
        print(f"No fixtures found under {FIXTURES}", file=sys.stderr)
        return 1

    print(f"dumping {len(fixtures)} SDFGs -> {out_dir}")
    print()
    n_ok = n_skip = n_fail = 0
    for src in fixtures:
        try:
            status, msg = dump_one(src, out_dir)
            print(f"[{status:>4}] {src.name:<30s} {msg}")
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
        except Exception:
            print(f"[fail] {src.name:<30s}")
            traceback.print_exc()
            n_fail += 1
            # stash the traceback in a sibling .err file for offline reading
            err_path = out_dir / f"{src.stem}.err"
            err_path.write_text(traceback.format_exc())
    print()
    print(f"{n_ok} ok, {n_skip} skipped, {n_fail} failed  "
          f"(out of {len(fixtures)}) under {out_dir}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
