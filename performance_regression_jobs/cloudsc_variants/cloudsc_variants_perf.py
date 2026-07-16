#!/usr/bin/env python3
"""Codegen-backend study on the CLOUDSC kernel (ECMWF cloud microphysics -- a huge
Fortran-derived SDFG) over the four DaCe build variants:
``{build_mode: cmake, native} x {compiler.cpu.implementation: legacy, experimental_readable}``.

Same two orthogonal axes as ``codegen_variants``: ``build_mode`` picks how the SAME
generated code is built (CMake configure+build vs the direct g++/link native back-end,
moves the compile wall only); ``cpu.implementation`` picks WHICH C++ is emitted (legacy
connector-based tasklets vs the readable per-array ``_idx()`` generator), so it moves
codegen time, compile time AND runtime. CloudSC is the stress case the NPBench corpus
never reaches: thousands of tasklets, hundreds of maps, one enormous translation unit.

Two phases:

* Phase A (cache, EXPENSIVE -- minutes): build the CloudSC SDFG (``simplify=False``
  parse) and run the parallelize chain validated end-to-end by
  ``tests/corpus/cloudsc/cloudsc_parallelize_chain_test.py`` (simplify + ShortLoopUnroll
  + PrivatizeScalars + PCIA + AugAssignToWCR + LoopToMap + LoopToScan -- the ``_chain()``
  the test asserts is numerically faithful). The result is saved COMPRESSED to
  ``<cache-dir>/cloudsc_parallel.sdfgz`` and reused by every later invocation, so the
  loop2map+unroll cost is paid once, on the login node (``--build-cache-only``).
* Phase B (measure, debug-window sized): LOAD the cached .sdfgz (fast), run one
  sequential reference (``make_sequential`` + cmake/legacy), then per variant time
  codegen_ms + compile_total_ms (cold ``sdfg.compile()`` wall, unique names) and
  runtime_ms (MEDIAN of ``--run-reps``, default 25) with correctness vs the sequential
  reference (``compare_outputs``). Each cell is crash-isolated (engine.run_isolated).

Deviations from the codegen_variants pattern (single-kernel job, huge SDFG):

* runtime is the MEDIAN of the reps, not best-of-N (one kernel -- report the
  distribution's center, not its lucky tail).
* correctness comes off the same instrumented build that is timed (the buffers hold
  the last rep's outputs) instead of a separate correctness build -- one fewer
  multi-minute compile per cell.
* ``-ffast-math`` is stripped and ``-fno-fast-math -ffp-contract=off`` appended: the
  cloudsc harness can only bound transform error under IEEE-respecting builds (see
  generate_data_for_cloudsc); ``-O3 -march=native`` are kept, so the perf regime is
  the chain test's validated ``o3`` regime.
* ``--compile-reps`` defaults to 1 (a cold cloudsc compile is minutes, and the whole
  sweep must fit a 30-min debug window).

    python3 cloudsc_variants/cloudsc_variants_perf.py --build-cache-only   # login node, once
    python3 cloudsc_variants/cloudsc_variants_perf.py                      # the 4-cell sweep
    python3 cloudsc_variants/cloudsc_variants_perf.py --variants cmake_legacy
    python3 cloudsc_variants/cloudsc_variants_perf.py --tables-only        # rebuild the tables
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import contextlib
import csv
import statistics
import sys
import time

# This job lives one level below the flat performance_regression_jobs framework (engine.py),
# and two levels below the repo root (tests.corpus.cloudsc -- the SDFG builder + input
# generator + chain live there). Neither is an installed package; put both on the path.
_JOB_DIR = os.path.dirname(os.path.abspath(__file__))
_PERF_DIR = os.path.dirname(_JOB_DIR)
_REPO_ROOT = os.path.dirname(_PERF_DIR)
for _p in (_PERF_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dace.config import set_temporary

import engine

# Same guard as codegen_variants: the cmake variants run a real CMake configure inside the
# measurement subprocess; under srun that deadlocks unless DaCe carries the SIGCHLD-mask +
# rank-env fixes. Refuse early rather than hanging the debug job.
from dace.codegen import compiler as dace_compiler

missing_sigchld_fix = [
    n for n in ('_build_subprocess_env', '_build_subprocess_sigmask') if n not in vars(dace_compiler)
]
if missing_sigchld_fix:
    raise SystemExit(f'cloudsc_variants: this DaCe lacks {missing_sigchld_fix}; the cmake variants would hang under '
                     f'srun (SIGCHLD-blocked mask). Run on the extended branch, which carries the fix.')

CORPUS = 'cloudsc'
KERNEL = 'cloudsc'
CACHE_FILENAME = 'cloudsc_parallel.sdfgz'
DEFAULT_CACHE_DIR = os.path.join(_JOB_DIR, 'cache')
#: The four variants: (build_mode, cpu.implementation) -- same axis as codegen_variants.
VARIANTS = (
    ('cmake', 'legacy'),
    ('native', 'legacy'),
    ('cmake', 'experimental_readable'),
    ('native', 'experimental_readable'),
)
#: Thread count for the parallel runs, read from the launch environment (slurm sets 72).
MULTI_THREADS = max(1, int(os.environ.get('OMP_NUM_THREADS', '4')))
#: Same row schema as codegen_variants, so the existing table/plot tooling reads both.
FIELDS = ('kernel', 'build_mode', 'implementation', 'codegen_ms', 'codegen_bytes', 'compile_total_ms', 'build_ms',
          'run_ms', 'correct')
SEED = 0
#: Parallel candidate vs sequential reference under -O3 -fno-fast-math -ffp-contract=off:
#: the chain test bounds this regime at 1e-12; 1e-9 adds headroom for -march=native and
#: OMP reduction-order jitter without masking a real miscompile.
TOL = 1e-9


# --------------------------------------------------------------------------
# Phase A: build + parallelize + cache. Runs ONCE (login node); everything in
# Phase B only ever loads the .sdfgz.
# --------------------------------------------------------------------------
def build_cache(cache_path):
    """Build the CloudSC SDFG and run the validated parallelize chain, saving the
    result compressed to ``cache_path``. Pure transform work -- no compilation."""
    from tests.corpus.cloudsc.generate_data_for_cloudsc import build_cloudsc_sdfg
    from tests.corpus.cloudsc.cloudsc_parallelize_chain_test import _chain
    t_total = time.perf_counter()
    print('phase A: building the CloudSC SDFG (simplify=False parse -- minutes)...', flush=True)
    t0 = time.perf_counter()
    sdfg = build_cloudsc_sdfg(simplify=False)
    print(f'  build_cloudsc_sdfg: {time.perf_counter() - t0:.1f}s', flush=True)
    for label, apply_fn in _chain():
        t0 = time.perf_counter()
        # The loop transforms log every refused loop; keep the job log readable.
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            apply_fn(sdfg)
        sdfg.validate()
        print(f'  {label}: {time.perf_counter() - t0:.1f}s', flush=True)
    sdfg.name = 'cloudsc_parallel'
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    sdfg.save(cache_path, compress=True)
    size_mb = os.path.getsize(cache_path) / 1e6
    print(f'phase A: cached {cache_path} ({size_mb:.1f} MB) in {time.perf_counter() - t_total:.1f}s total',
          flush=True)


# --------------------------------------------------------------------------
# Phase B helpers. All of these run inside the isolated measurement subprocess.
# --------------------------------------------------------------------------
def _ieee_o3_flags():
    """Replace configure_dace_process's -ffast-math with -fno-fast-math -ffp-contract=off,
    keeping -O3/-march=native/-fopenmp and the OpenMP rpath / gcc-install-dir flags it set.
    This is the chain test's validated ``o3`` regime -- fast-math reassociates cloudsc's
    flux prefix sums and rewrites transcendentals, so no tolerance can bound it."""
    import dace
    toks = [t for t in dace.Config.get('compiler', 'cpu', 'args').split() if t != '-ffast-math']
    for flag in ('-fno-fast-math', '-ffp-contract=off'):
        if flag not in toks:
            toks.append(flag)
    dace.Config.set('compiler', 'cpu', 'args', value=' '.join(toks))


def _load_variant(cache_path, tag):
    """The cached parallel SDFG under a fresh unique name. The name is the DaCe cache key
    (engine.configure_dace_process sets cache='name'), so each cell/rep maps to its own
    build folder and every ``.compile()`` is a real cold build."""
    import dace
    sdfg = dace.SDFG.from_file(cache_path)
    sdfg.name = f'{KERNEL}_{tag}'
    return sdfg


def _variant_inputs(sdfg, seed=SEED):
    """Physically-realistic inputs, filtered to what the chained SDFG still takes (the
    chain's ``specialize`` bakes the species symbols out -- same filter as the test)."""
    from tests.corpus.cloudsc.generate_data_for_cloudsc import generate_cloudsc_inputs
    inputs = generate_cloudsc_inputs(sdfg, seed)
    needed = set(sdfg.arglist().keys()) | {str(s) for s in sdfg.free_symbols}
    return {k: v for k, v in inputs.items() if k in needed}


def run_reference(cache_path, ref_npz):
    """Isolated subprocess: run the cached SDFG SEQUENTIALLY (make_sequential) once under
    cmake/legacy and save every array buffer to ``ref_npz`` -- the deterministic oracle
    every variant cell is compared against. Returned via a FILE, not the result queue:
    run_isolated joins the child before draining the queue, and a multi-MB payload would
    block the queue's feeder thread (pipe capacity) and deadlock the cell into a timeout."""
    engine.configure_dace_process()
    _ieee_o3_flags()
    import numpy as np
    from tests.corpus.cloudsc.generate_data_for_cloudsc import make_sequential
    with set_temporary('compiler', 'build_mode', value='cmake'):
        with set_temporary('compiler', 'cpu', 'implementation', value='legacy'):
            sdfg = _load_variant(cache_path, 'ref_seq')
            make_sequential(sdfg)
            kwargs = _variant_inputs(sdfg)
            sdfg(**kwargs)
    arrays = {k: v for k, v in kwargs.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(ref_npz, **arrays)
    return len(arrays)


def bench_variant(cache_path, ref_npz, mode, impl, compile_reps, run_reps, timeout):
    """Isolated subprocess (bad codegen can crash): one (build_mode, implementation) cell.
    Returns one row."""
    engine.configure_dace_process()
    _ieee_o3_flags()
    import numpy as np
    from dace.codegen import codegen
    from tests.corpus.cloudsc.generate_data_for_cloudsc import compare_outputs

    reference = dict(np.load(ref_npz))
    os.environ['OMP_NUM_THREADS'] = str(MULTI_THREADS)
    with set_temporary('compiler', 'build_mode', value=mode):
        with set_temporary('compiler', 'cpu', 'implementation', value=impl):
            # codegen time + emitted-C++ size, per implementation (measured in context so
            # each variant is timed with its own generator selected).
            codegen_samples = []
            generated_bytes = 0
            for rep in range(compile_reps):
                sdfg = _load_variant(cache_path, f'{mode}_{impl}_cg{rep}')
                t0 = time.perf_counter()
                objects = codegen.generate_code(sdfg)
                codegen_samples.append((time.perf_counter() - t0) * 1000.0)
                generated_bytes = sum(len(obj.clean_code) for obj in objects)

            # full compile wall, cold each rep via a unique name, under this (mode, impl).
            compile_samples = []
            for rep in range(compile_reps):
                sdfg = _load_variant(cache_path, f'{mode}_{impl}_c{rep}')
                t0 = time.perf_counter()
                sdfg.compile()
                compile_samples.append((time.perf_counter() - t0) * 1000.0)

            # runtime + correctness on ONE instrumented build (deviation from
            # codegen_variants: no separate correctness build -- a cloudsc compile is
            # minutes). time_sdfg resets every array in place before each call, so after
            # it returns the buffers hold exactly one run's outputs.
            sdfg = _load_variant(cache_path, f'{mode}_{impl}_run')
            call_kwargs = _variant_inputs(sdfg)
            run_samples = engine.time_sdfg(sdfg, call_kwargs, run_reps, time_budget_s=0.4 * timeout)
            report = compare_outputs(reference, call_kwargs, rtol=TOL, atol=TOL)
            correct = all(ok for _, _, ok in report.values())
            worst = max(((ma, mr) for ma, mr, _ in report.values()), default=(0.0, 0.0))
            print(f'[{mode}/{impl}] reps={len(run_samples)} worst |abs|={worst[0]:.3e} '
                  f'|rel|={worst[1]:.3e} (tol={TOL:.0e}) {"ok" if correct else "FAIL"}', flush=True)

    codegen_ms = min(codegen_samples)
    compile_ms = min(compile_samples)
    return {
        'kernel': KERNEL,
        'build_mode': mode,
        'implementation': impl,
        'codegen_ms': round(codegen_ms, 3),
        'codegen_bytes': generated_bytes,
        'compile_total_ms': round(compile_ms, 3),
        'build_ms': round(max(0.0, compile_ms - codegen_ms), 3),
        'run_ms': round(statistics.median(run_samples), 4) if run_samples else '',
        'correct': int(bool(correct)),
    }


# --------------------------------------------------------------------------
# Results: same CSV schema + table style as codegen_variants (single kernel).
# --------------------------------------------------------------------------
def results_csv(results_dir):
    return os.path.join(results_dir, CORPUS, 'cloudsc_variants.csv')


def append_rows(results_dir, rows):
    path = results_csv(results_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.isfile(path)
    with open(path, 'a', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def num(row, key):
    try:
        return float(row[key])
    except (ValueError, KeyError, TypeError):
        return None


def write_tables(results_dir):
    """One variant table plus the two headline ratios: runtime legacy vs
    experimental_readable (per build_mode) and the native-vs-cmake build speedup."""
    path = results_csv(results_dir)
    if not os.path.isfile(path):
        print(f'no rows at {path}')
        return
    with open(path, newline='') as fp:
        rows = list(csv.DictReader(fp))

    cells = {}  # (mode, impl) -> row; a re-run's later row wins.
    for row in rows:
        cells[(row['build_mode'], row['implementation'])] = row

    lines = [
        f'# CloudSC codegen variants (klev=klon=32, OMP_NUM_THREADS as launched)',
        '',
        'Phase-A-cached parallel SDFG (the chain validated by cloudsc_parallelize_chain_test), '
        'compiled at -O3 -march=native -fno-fast-math -ffp-contract=off. run_ms is the MEDIAN '
        'of the reps; correct = matched the sequential reference within 1e-9.',
        '',
        '| variant | codegen ms | codegen B | compile ms | build ms | run ms | correct |',
        '|---|--:|--:|--:|--:|--:|:-:|',
    ]
    for mode, impl in VARIANTS:
        row = cells.get((mode, impl))
        if not row:
            continue
        lines.append(f"| {mode}/{impl} | {row['codegen_ms']} | {row.get('codegen_bytes', '')} | "
                     f"{row['compile_total_ms']} | {row['build_ms']} | {row['run_ms']} | "
                     f"{'yes' if row['correct'] == '1' else 'NO'} |")

    # Runtime: legacy vs experimental_readable, per build_mode (build_mode should not move
    # runtime -- listing both doubles as that sanity check on this one kernel).
    for mode in ('cmake', 'native'):
        leg, exp = cells.get((mode, 'legacy')), cells.get((mode, 'experimental_readable'))
        if not leg or not exp:
            continue
        both_ok = leg['correct'] == '1' and exp['correct'] == '1'
        rl, re_ = num(leg, 'run_ms'), num(exp, 'run_ms')
        if rl and re_ and both_ok:
            lines += ['', f'**runtime experimental vs legacy ({mode})**: {rl / re_:.3f}x '
                      f'(>1 = the readable generator produced faster code).']
        elif not both_ok:
            lines += ['', f'**runtime experimental vs legacy ({mode})**: n/a (a variant failed correctness).']

    # Code-size + build-mode levers, as in codegen_variants.
    leg, exp = cells.get(('cmake', 'legacy')), cells.get(('cmake', 'experimental_readable'))
    bl, be = (num(leg, 'codegen_bytes') if leg else None), (num(exp, 'codegen_bytes') if exp else None)
    if bl and be:
        lines.append(f'**codegen size experimental vs legacy**: {be / bl:.3f}x '
                     f'(<1 = readable generator emits less C++).')
    for impl in ('legacy', 'experimental_readable'):
        cm, nat = cells.get(('cmake', impl)), cells.get(('native', impl))
        bc, bn = (num(cm, 'build_ms') if cm else None), (num(nat, 'build_ms') if nat else None)
        if bc and bn:
            lines.append(f'**native vs cmake build speedup ({impl})**: {bc / bn:.2f}x.')

    failing = sorted(f'{m}/{i}' for (m, i), r in cells.items() if r.get('correct') != '1')
    lines += ['', '## Correctness audit', '']
    lines.append(f'Variants that did NOT match the sequential reference: {", ".join(failing)}'
                 if failing else 'All variants matched the sequential reference.')

    out = os.path.join(os.path.dirname(path), 'cloudsc_variants.md')
    with open(out, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')
    print(f'wrote {out}')


def parse_variants(spec):
    if not spec:
        return VARIANTS
    by_id = {f'{m}_{i}': (m, i) for m, i in VARIANTS}
    picked = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in by_id:
            raise SystemExit(f'--variants: unknown {tok!r}; choose from: {", ".join(by_id)}')
        picked.append(by_id[tok])
    return tuple(picked)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default='results', help='results root (default: results)')
    ap.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR,
                    help=f'directory for the Phase-A .sdfgz cache (default: {DEFAULT_CACHE_DIR})')
    ap.add_argument('--rebuild-cache', action='store_true', help='force Phase A even if the cache exists')
    ap.add_argument('--build-cache-only', action='store_true',
                    help='run Phase A (if needed) and exit -- for pre-building on the login node')
    ap.add_argument('--compile-reps', type=int, default=1,
                    help='cold-compile samples per cell (default: 1 -- a cloudsc compile is minutes)')
    ap.add_argument('--run-reps', type=int, default=25, help='runtime samples per cell (default: 25)')
    ap.add_argument('--variants', default=None,
                    help='comma-separated subset, e.g. cmake_legacy,native_experimental_readable (default: all 4)')
    ap.add_argument('--tables-only', action='store_true', help='skip measurement, just rebuild the markdown tables')
    ap.add_argument('--timeout', type=float, default=900.0, help='per-cell subprocess timeout, seconds')
    ap.add_argument('--cxx', default=None, help='C++ compiler for DaCe codegen (default: clang++ on PATH, else g++)')
    args = ap.parse_args()

    if args.tables_only:
        write_tables(args.results_dir)
        return
    if args.cxx:
        os.environ['DACE_PERF_CXX'] = args.cxx  # relayed to the spawned measurement subprocesses

    cache_path = os.path.abspath(os.path.join(args.cache_dir, CACHE_FILENAME))
    if args.rebuild_cache or not os.path.isfile(cache_path):
        build_cache(cache_path)
    else:
        print(f'phase A: cache hit ({cache_path}, {os.path.getsize(cache_path) / 1e6:.1f} MB)')
    if args.build_cache_only:
        return

    variants = parse_variants(args.variants)

    print('phase B: sequential reference run (one compile + one run, crash-isolated)...', flush=True)
    ref_npz = os.path.join(os.path.dirname(cache_path), 'cloudsc_reference_outputs.npz')
    ok, n_ref = engine.run_isolated(run_reference, (cache_path, ref_npz), timeout=args.timeout)
    if not ok or not os.path.isfile(ref_npz):
        raise SystemExit(f'cloudsc_variants: reference run failed: {n_ref}')
    print(f'phase B: reference saved ({n_ref} arrays -> {ref_npz})', flush=True)

    for mode, impl in variants:
        t0 = time.perf_counter()
        ok, payload = engine.run_isolated(
            bench_variant, (cache_path, ref_npz, mode, impl, args.compile_reps, args.run_reps, args.timeout),
            timeout=args.timeout)
        if ok:
            append_rows(args.results_dir, [payload])
            print(f'[{mode}/{impl}] done in {time.perf_counter() - t0:.1f}s')
        else:
            print(f'[{mode}/{impl}] failed: {payload}')

    write_tables(args.results_dir)


if __name__ == '__main__':
    main()
