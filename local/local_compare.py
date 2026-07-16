#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Local 4-variant codegen comparison: {cmake, native} x {oldcpu (legacy), newcpu (experimental)}.

Compiles all FOUR variants of every selected kernel through the REAL build path
(``sdfg.compile()``, which honours ``compiler.build_mode`` -- unlike engine's direct timed
compile, which always bypasses CMake), then times each with ``--reps`` repetitions and reports
the MEDIAN. The headline comparison is ``cmake-newcpu`` vs ``cmake-oldcpu``:

    speedup = median(cmake-oldcpu) / median(cmake-newcpu)     (>1 => the new codegen is faster)

Every variant runs the identical pipeline (dace + simplify + LoopToMap + MapFusion +
ConvertLengthOneArraysToScalars) -- the only things that vary are ``compiler.build_mode`` and
``compiler.cpu.implementation``.

Each variant runs in a forked/spawned subprocess (engine.run_isolated), so a segfault or a
compile failure in one variant fails only that row.

    python3 local_compare.py --preset paper --reps 25 --out results.tsv
"""
import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PERF_DIR = os.path.join(REPO_ROOT, 'cpu_codegen_perf_jobs')
PERF_JOBS_DIR = os.path.join(REPO_ROOT, 'performance_regression_jobs')
for _p in (PERF_DIR, PERF_JOBS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import engine  # noqa: E402
import npbench_polybench_perf as base  # noqa: E402
from run_readable_perf import pipelined_sdfg, select_kernels, kernel_corpus, set_implementation  # noqa: E402

#: The four variants: (label, build_mode, codegen). ``oldcpu`` = the legacy C++ generator,
#: ``newcpu`` = the experimental readable one.
VARIANTS = (
    ('cmake-oldcpu', 'cmake', 'legacy'),
    ('cmake-newcpu', 'cmake', 'experimental'),
    ('native-oldcpu', 'native', 'legacy'),
    ('native-newcpu', 'native', 'experimental'),
)
#: The pair the speedup is computed from.
BASELINE, CANDIDATE = 'cmake-oldcpu', 'cmake-newcpu'

COLUMNS = ('kernel', 'corpus', 'variant', 'build_mode', 'codegen', 'preset', 'threads', 'cxx', 'reps', 'codegen_ms',
           'compile_ms', 'median_ms', 'speedup', 'status', 'error')


def variant_job(name, preset, build_mode, codegen, reps, timeout):
    """Build + compile ONE variant through the configured build_mode, then time it.

    Returns {'codegen_ms', 'compile_ms', 'median_ms', 'reps'}. ``sdfg.compile()`` (not engine's
    direct compile) is used on purpose: it is the only path that honours ``compiler.build_mode``,
    which is exactly what the cmake-vs-native axis measures.
    """
    import dace
    engine.configure_dace_process()
    dace.Config.set('compiler', 'build_mode', value=build_mode)
    os.environ['DACE_compiler_build_mode'] = build_mode
    set_implementation(codegen)

    info = base.load_bench_info(name)
    params = info['parameters'][preset]
    program, arrays, params = base.build_program_and_data(name, info, params)
    # Name carries the variant so the four builds never share a cache folder.
    sdfg = pipelined_sdfg(program, f'local_{name}_{build_mode}_{codegen}', 'cpu')

    t0 = time.perf_counter()
    sdfg.generate_code()
    codegen_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    sdfg.compile()
    compile_ms = (time.perf_counter() - t0) * 1000.0

    kwargs = base._dace_call_kwargs(sdfg, arrays, params)
    times = engine.time_sdfg(sdfg, kwargs, reps, time_budget_s=0.8 * timeout)
    return dict(codegen_ms=codegen_ms,
                compile_ms=compile_ms,
                median_ms=(statistics.median(times) if times else None),
                reps=len(times))


def cell(value, fmt='%.3f'):
    return '' if value is None else (fmt % value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', choices=('S', 'paper'), default='paper', help='dataset size (default: paper)')
    ap.add_argument('--corpus', choices=('npbench', 'polybench', 'both'), default='both')
    ap.add_argument('--reps', type=int, default=25, help='timed repetitions per variant (median; default: 25)')
    ap.add_argument('--only', default=None, help='substring filter on kernel name')
    ap.add_argument('--kernels', default=None, help='comma-separated explicit kernel list')
    ap.add_argument('--timeout', type=float, default=900.0, help='per-variant subprocess timeout, seconds')
    ap.add_argument('--out', default='local_compare.tsv', help='TSV output path')
    ap.add_argument('--list-kernels', action='store_true')
    args = ap.parse_args()

    explicit = [k.strip() for k in args.kernels.split(',')] if args.kernels else None
    names = select_kernels(args.corpus, args.preset, args.only, explicit)
    if args.list_kernels:
        print('\n'.join(names))
        return

    threads = os.environ.get('OMP_NUM_THREADS', '1')
    cxx = os.environ.get('DACE_PERF_CXX', 'g++')
    print(f'local 4-variant compare: {len(names)} kernel(s)  preset={args.preset}  reps={args.reps} (median)  '
          f'threads={threads}  cxx={cxx}')
    print(f'  variants: {", ".join(v[0] for v in VARIANTS)}   speedup = median({BASELINE}) / median({CANDIDATE})')

    with open(args.out, 'w') as fh:
        fh.write('\t'.join(COLUMNS) + '\n')
        fh.flush()
        for name in names:
            medians = {}
            rows = []
            for label, build_mode, codegen in VARIANTS:
                # run_isolated spawns (not forks), so the callable is pickled: it must be a
                # module-level function taking a plain args tuple -- never a local/lambda closure.
                ok, payload = engine.run_isolated(variant_job,
                                                  (name, args.preset, build_mode, codegen, args.reps, args.timeout),
                                                  timeout=args.timeout)
                if ok and payload:
                    medians[label] = payload['median_ms']
                    rows.append(
                        dict(variant=label,
                             build_mode=build_mode,
                             codegen=codegen,
                             reps=payload['reps'],
                             codegen_ms=payload['codegen_ms'],
                             compile_ms=payload['compile_ms'],
                             median_ms=payload['median_ms'],
                             status='ok',
                             error=''))
                    print(f'  [{name}/{label}] codegen={payload["codegen_ms"]:.1f}ms '
                          f'compile={payload["compile_ms"]:.1f}ms median={cell(payload["median_ms"], "%.6f")}ms '
                          f'({payload["reps"]} reps)')
                else:
                    err = str(payload).replace('\t', ' ').replace('\n', ' ')[:400] if payload else 'failed'
                    rows.append(
                        dict(variant=label,
                             build_mode=build_mode,
                             codegen=codegen,
                             reps=0,
                             codegen_ms=None,
                             compile_ms=None,
                             median_ms=None,
                             status='ERROR',
                             error=err))
                    print(f'  [{name}/{label}] ERROR: {err[:120]}')

            # speedup lands on the candidate row: baseline / candidate (>1 => candidate faster)
            base_ms, cand_ms = medians.get(BASELINE), medians.get(CANDIDATE)
            speedup = (base_ms / cand_ms) if (base_ms and cand_ms) else None
            for r in rows:
                fh.write('\t'.join([
                    name,
                    kernel_corpus(name),
                    r['variant'],
                    r['build_mode'],
                    r['codegen'],
                    args.preset,
                    str(threads),
                    cxx,
                    str(r['reps']),
                    cell(r['codegen_ms'], '%.3f'),
                    cell(r['compile_ms'], '%.3f'),
                    cell(r['median_ms'], '%.6f'),
                    cell(speedup if r['variant'] == CANDIDATE else None, '%.4f'),
                    r['status'],
                    r['error'],
                ]) + '\n')
                fh.flush()
            if speedup:
                print(f'  [{name}] speedup ({BASELINE} / {CANDIDATE}) = {speedup:.3f}x')

    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
