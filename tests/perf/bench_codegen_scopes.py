# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Optimization and code-generation wall time, for comparing two checkouts.

Three workloads:
  1. ``opt``            -- ``sdfg.simplify()`` over every npbench/polybench kernel
  2. ``codegen``        -- ``generate_code`` on the same kernels, after simplify
  3. ``cloudsc``        -- initial simplify, then loop unrolling, then codegen on cloudsc

Writes one CSV. Run it once per checkout and give the two files to plot_codegen_scopes.py.

    python tests/perf/bench_codegen_scopes.py --label pr --out pr.csv --reps 5

Only the Python half is timed; the C++ toolchain is untouched by these passes and would bury the
signal. Setup (building and deep-copying SDFGs) stays outside every timer.
"""

import argparse
import copy
import csv
import importlib
import importlib.util
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import dace
from dace.codegen import codegen
from dace.transformation.interstate import LoopUnroll

REPO = Path(__file__).resolve().parents[2]

#: The kernel corpus is the workload, so it must be identical on both arms of a comparison. The
#: A/B driver checks tests/corpus out of a fixed ref into every worktree; this script just reads it.
CORPUS = REPO / 'tests' / 'corpus'


def load_module(path: Path):
    """Import a corpus file by path, so no assumption is made about package layout."""
    name = 'dacebench_' + '_'.join(path.relative_to(CORPUS).with_suffix('').parts)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def discover_programs(subdir: str) -> List[Tuple[str, 'dace.frontend.python.parser.DaceProgram']]:
    """Every ``DaceProgram`` under ``tests/corpus/<subdir>``, as (kernel, program)."""
    found: List[Tuple[str, object]] = []
    root = CORPUS / subdir
    if not root.is_dir():
        return found
    if str(CORPUS) not in sys.path:
        sys.path.insert(0, str(CORPUS))
    for path in sorted(root.rglob('*.py')):
        if path.name.startswith('_') or 'generate_data' in path.name:
            continue
        try:
            module = load_module(path)
        except Exception:
            continue
        if module is None:
            continue
        for attr in sorted(dir(module)):
            obj = getattr(module, attr, None)
            if isinstance(obj, dace.frontend.python.parser.DaceProgram):
                found.append((f'{path.stem}:{attr}', obj))
    return found


def timed(fn: Callable[[], None], reps: int) -> Optional[Dict[str, float]]:
    """Median/min of ``reps`` timings, or None if the workload raised."""
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        try:
            fn()
        except Exception:
            return None
        samples.append(time.perf_counter() - start)
    return {'median_s': statistics.median(samples), 'min_s': min(samples), 'reps': len(samples)}


def bench_kernels(rows: List[dict], label: str, reps: int, limit: int = 0) -> None:
    """simplify() and generate_code() per kernel. Each rep gets a pristine deepcopy."""
    programs = discover_programs('npbench') + discover_programs('polybench')
    if limit:
        programs = programs[:limit]
    if not programs:
        raise SystemExit(f'[{label}] no kernels under {CORPUS} -- refusing to write an empty CSV '
                         f'that would look like a clean run. Check tests/corpus out of a branch that '
                         f'has it (see run_codegen_scopes_ab.sh --corpus-ref).')
    print(f'[{label}] discovered {len(programs)} kernels', flush=True)

    for name, program in programs:
        try:
            pristine = program.to_sdfg(simplify=False)
        except Exception:
            print(f'[{label}] SKIP build {name}', flush=True)
            continue

        clones = [copy.deepcopy(pristine) for _ in range(reps)]
        it = iter(clones)
        opt = timed(lambda: next(it).simplify(validate=False), reps)
        if opt:
            rows.append({'workload': 'opt', 'kernel': name, **opt})

        try:
            simplified = copy.deepcopy(pristine)
            simplified.simplify(validate=False)
        except Exception:
            print(f'[{label}] SKIP simplify {name}', flush=True)
            continue

        # generate_code mutates (preprocess pads regions), so each rep needs its own copy
        cg_clones = [copy.deepcopy(simplified) for _ in range(reps)]
        cit = iter(cg_clones)
        cg = timed(lambda: codegen.generate_code(next(cit)), reps)
        if cg:
            rows.append({'workload': 'codegen', 'kernel': name, **cg})

        done = [r for r in rows if r['kernel'] == name]
        print(f'[{label}] {name}: ' + '  '.join(f"{r['workload']}={r['median_s'] * 1000:.1f}ms" for r in done),
              flush=True)


def bench_cloudsc(rows: List[dict], label: str, reps: int) -> None:
    """cloudsc: initial simplify, then loop unrolling, then codegen on the unrolled graph.

    Each stage feeds the next, so they are timed on progressively transformed graphs rather than
    all on the pristine one -- that is the order a real compile runs them in.
    """
    if str(CORPUS) not in sys.path:
        sys.path.insert(0, str(CORPUS))
    try:
        from cloudsc.generate_data_for_cloudsc import build_cloudsc_sdfg
        build = lambda: build_cloudsc_sdfg(simplify=False)
    except Exception as ex:
        # cloudsc imports dace.transformation.layout, which only exists on branches carrying that
        # subsystem -- so it cannot be built against an upstream base ref no matter where the corpus
        # comes from. Warn rather than abort: the kernel workloads above are still valid.
        print(
            f'[{label}] WARNING: cloudsc unavailable on this checkout ({type(ex).__name__}: {ex}). '
            f'The three cloudsc stages are ABSENT from this CSV. For cloudsc numbers, both refs '
            f'must carry dace.transformation.layout.',
            flush=True)
        return

    print(f'[{label}] building cloudsc SDFG...', flush=True)
    start = time.perf_counter()
    pristine = build()
    print(
        f'[{label}] cloudsc built in {time.perf_counter() - start:.1f}s '
        f'({len(pristine.arrays)} arrays, {len(list(pristine.states()))} states)',
        flush=True)

    stages = []

    clones = [copy.deepcopy(pristine) for _ in range(reps)]
    it = iter(clones)
    stages.append(('cloudsc_simplify', timed(lambda: next(it).simplify(validate=False), reps)))

    simplified = copy.deepcopy(pristine)
    simplified.simplify(validate=False)

    clones = [copy.deepcopy(simplified) for _ in range(reps)]
    it = iter(clones)
    stages.append(
        ('cloudsc_unroll', timed(lambda: next(it).apply_transformations_repeated(LoopUnroll, validate=False), reps)))

    unrolled = copy.deepcopy(simplified)
    unrolled.apply_transformations_repeated(LoopUnroll, validate=False)

    clones = [copy.deepcopy(unrolled) for _ in range(reps)]
    it = iter(clones)
    stages.append(('cloudsc_codegen_after_unroll', timed(lambda: codegen.generate_code(next(it)), reps)))

    for workload, result in stages:
        if result:
            rows.append({'workload': workload, 'kernel': 'cloudsc', **result})
            print(f'[{label}] {workload}: {result["median_s"] * 1000:.1f}ms', flush=True)
        else:
            print(f'[{label}] {workload}: FAILED', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True, help='name for this checkout, e.g. "pr" or "main"')
    parser.add_argument('--out', required=True, help='CSV path')
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--skip-cloudsc', action='store_true')
    parser.add_argument('--cloudsc-only', action='store_true', help='skip the kernel corpus')
    parser.add_argument('--limit', type=int, default=0, help='only the first N kernels (smoke test)')
    args = parser.parse_args()

    os.environ.setdefault('DACE_compiler_use_cache', '0')
    rows: List[dict] = []
    if not args.cloudsc_only:
        bench_kernels(rows, args.label, args.reps, args.limit)
    if not args.skip_cloudsc:
        bench_cloudsc(rows, args.label, args.reps)

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['label', 'workload', 'kernel', 'median_s', 'min_s', 'reps'])
        writer.writeheader()
        for row in rows:
            writer.writerow({'label': args.label, **row})
    print(f'[{args.label}] wrote {len(rows)} rows to {args.out}', flush=True)


if __name__ == '__main__':
    main()
