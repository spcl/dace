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
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def timed(setup: Callable[[], Any], fn: Callable[[Any], None], reps: int) -> Optional[Dict[str, float]]:
    """Median/min over ``reps`` runs of ``fn(setup())``, or None if it raised.

    ``setup`` runs inside the loop but outside the timer, and its result is dropped before the next
    iteration. Materializing every clone up front would hold ``reps`` copies of the graph alive at
    once -- on a 5890-state SDFG that is gigabytes, and the allocator pressure alone would distort
    what is being measured.
    """
    samples = []
    for _ in range(reps):
        subject = setup()
        start = time.perf_counter()
        try:
            fn(subject)
        except Exception:
            return None
        samples.append(time.perf_counter() - start)
        del subject
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

        opt = timed(lambda: copy.deepcopy(pristine), lambda g: g.simplify(validate=False), reps)
        if opt:
            rows.append({'workload': 'opt', 'kernel': name, **opt})

        try:
            simplified = copy.deepcopy(pristine)
            simplified.simplify(validate=False)
        except Exception:
            print(f'[{label}] SKIP simplify {name}', flush=True)
            continue

        # generate_code mutates (preprocess pads regions), so each rep needs its own copy
        cg = timed(lambda: copy.deepcopy(simplified), codegen.generate_code, reps)
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
    # Both roots: `cloudsc.*` resolves against tests/corpus, while generate_data_for_cloudsc
    # imports `tests.corpus.cloudsc.cloudsc` absolutely, which needs the repo root.
    for root in (str(CORPUS), str(REPO)):
        if root not in sys.path:
            sys.path.insert(0, root)
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

    # The build is the INPUT, not what is measured, and it is identical for every ref -- so a
    # shared cache lets the second arm skip a frontend parse that takes minutes on cloudsc.
    cache = os.environ.get('DACE_BENCH_SDFG_CACHE')
    pristine = None
    if cache and os.path.exists(cache):
        try:
            start = time.perf_counter()
            pristine = dace.SDFG.from_file(cache)
            print(f'[{label}] loaded cloudsc SDFG from {cache} in {time.perf_counter() - start:.1f}s', flush=True)
        except Exception as ex:
            print(f'[{label}] cache at {cache} unusable ({type(ex).__name__}), rebuilding', flush=True)
            pristine = None

    if pristine is None:
        print(f'[{label}] building cloudsc SDFG (several minutes -- frontend parse)...', flush=True)
        start = time.perf_counter()
        pristine = build()
        if cache:
            pristine.save(cache, compress=True)
            print(f'[{label}] cached the built SDFG at {cache}', flush=True)
    print(
        f'[{label}] cloudsc built in {time.perf_counter() - start:.1f}s '
        f'({len(pristine.arrays)} arrays, {len(list(pristine.states()))} states)',
        flush=True)

    stages = []

    stages.append(('cloudsc_simplify', timed(lambda: copy.deepcopy(pristine), lambda g: g.simplify(validate=False),
                                             reps)))

    simplified = copy.deepcopy(pristine)
    simplified.simplify(validate=False)

    stages.append(('cloudsc_unroll',
                   timed(lambda: copy.deepcopy(simplified),
                         lambda g: g.apply_transformations_repeated(LoopUnroll, validate=False), reps)))

    unrolled = copy.deepcopy(simplified)
    unrolled.apply_transformations_repeated(LoopUnroll, validate=False)

    stages.append(('cloudsc_codegen_after_unroll', timed(lambda: copy.deepcopy(unrolled), codegen.generate_code, reps)))

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
