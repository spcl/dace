#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""External-TU build/perf sweep: codegen + REAL build time + runtime across the grid

    codegen   in {legacy, experimental}                (compiler.cpu.implementation)
    split     in {off, on}   single-TU vs external-TU  (codegen_params.split_nsdfg_translation_units)
    builder   in {cmake, native}                        (compiler.build_mode)

over the NPBench + PolyBench corpora, CPU only. Unlike ``run_buildperf.py`` -- whose
``compile_ms`` is a DIRECT ``g++`` invocation on the generated sources (deliberately builder-agnostic)
-- this driver times the REAL cold ``sdfg.compile()`` under each ``build_mode``, so the cmake configure
overhead vs the native no-cmake path is exactly what the number reflects. The external-TU split is a
build-parallelism tactic (N top-level nests -> N ``.cpp`` the cmake/Ninja generator compiles
concurrently), so builder x split is the axis of interest.

New CPU codegen is measured at its DEFAULT flags, which are the intended-optimal ones (explicit copy,
by-reference scalar binding, ...); nothing here overrides them.

The output TSV carries raw ``codegen_ms`` / ``compile_ms`` / ``runtime_ms`` per grid point; ratios and
the requested plots are derived by ``plot_external_tu.py`` -- so the sweep stays a pure measurement and
you can ``python plot_external_tu.py results.tsv`` straight after.

Reuses the corpus loader, pipeline and crash-isolation of the codegen_buildperf_jobs suite
(``run_buildperf`` / ``npbench_polybench_perf`` / ``engine``).

Example:
    python3 run_external_tu_perf.py --preset S --only atax --out probe.tsv
    python3 run_external_tu_perf.py --preset paper --corpus both --out paper_exttu.tsv
"""
import argparse
import copy
import csv
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PERF_JOBS_DIR = os.path.join(REPO_ROOT, 'performance_regression_jobs')
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if PERF_JOBS_DIR not in sys.path:
    sys.path.append(PERF_JOBS_DIR)

import engine  # noqa: E402
import npbench_polybench_perf as base  # noqa: E402
import run_buildperf as rb  # noqa: E402 -- reuse pipeline, implementation selection, kernel selection

CODEGENS = ('legacy', 'experimental')
SPLITS = ('off', 'on')
BUILD_MODES = ('cmake', 'native')

#: TSV columns -- the exact header plot_external_tu.py reads.
TSV_FIELDS = [
    'kernel', 'corpus', 'codegen', 'split', 'build_mode', 'preset', 'threads', 'codegen_ms', 'compile_ms',
    'runtime_ms', 'error'
]


def set_split(on):
    """Enable/disable the per-nest external translation-unit split for THIS process (config + env, so a
    nested spawn inherits it)."""
    import dace
    dace.Config.set('compiler', 'cpu', 'codegen_params', 'split_nsdfg_translation_units', value=on)
    os.environ['DACE_compiler_cpu_codegen_params_split_nsdfg_translation_units'] = 'true' if on else 'false'


def set_build_mode(build_mode):
    """Select the DaCe builder (cmake | native) for THIS process (config + env)."""
    import dace
    dace.Config.set('compiler', 'build_mode', value=build_mode)
    os.environ['DACE_compiler_build_mode'] = build_mode


def ext_tu_job(name, preset, codegen, split, build_mode, reps, timeout):
    """One grid point, run inside engine.run_isolated (a codegen/compile crash fails only this point).

    Returns ``{'codegen_ms', 'compile_ms', 'runtime_ms'}``:
      codegen_ms  DaCe C++/nest generation time (includes the outline pass when split is on).
      compile_ms  REAL cold ``sdfg.compile()`` under this build_mode -- cmake configure+build, or the
                  native direct-build path -- into a fresh /dev/shm folder (never a cache no-op).
      runtime_ms  best-of-`reps` execution time.
    """
    import dace
    from dace.codegen import codegen as cg
    engine.configure_dace_process()
    rb.set_implementation(codegen)
    set_split(split == 'on')
    set_build_mode(build_mode)

    info = base.load_bench_info(name)
    params = info['parameters'][preset]
    program, arrays, params = base.build_program_and_data(name, info, params)
    uname = f'exttu_{name}_{preset}_{codegen}_{split}_{build_mode}'
    sdfg = rb.pipelined_sdfg(program, uname)

    # Codegen time on a deepcopy: generate_code mutates (library expansion, outline), so timing it must
    # not perturb the graph the build below compiles.
    sdfg_cg = copy.deepcopy(sdfg)
    t0 = time.perf_counter()
    cg.generate_code(sdfg_cg)
    codegen_ms = (time.perf_counter() - t0) * 1000.0

    # Real cold build under the selected builder. Fresh /dev/shm build folder so this is never a
    # cache-hit no-op and stays off any shared/network scratch FS.
    shm = '/dev/shm'
    build_root = tempfile.mkdtemp(prefix='exttu_build_', dir=shm if os.path.isdir(shm) and os.access(shm, os.W_OK) else None)
    prev_folder = os.environ.get('DACE_default_build_folder')
    os.environ['DACE_default_build_folder'] = build_root
    dace.Config.set('default_build_folder', value=build_root)
    try:
        t1 = time.perf_counter()
        sdfg.compile()
        compile_ms = (time.perf_counter() - t1) * 1000.0
        kwargs = base._dace_call_kwargs(sdfg, arrays, params)
        times = engine.time_sdfg(sdfg, kwargs, reps, time_budget_s=0.8 * timeout)
    finally:
        if prev_folder is None:
            os.environ.pop('DACE_default_build_folder', None)
        else:
            os.environ['DACE_default_build_folder'] = prev_folder
        shutil.rmtree(build_root, ignore_errors=True)
    return dict(codegen_ms=codegen_ms, compile_ms=compile_ms, runtime_ms=(min(times) if times else None))


def grid(codegens, splits, build_modes):
    return [(c, s, b) for c in codegens for s in splits for b in build_modes]


def process_kernel(writer, out_handle, name, args, threads):
    corpus = rb.kernel_corpus(name)
    for codegen, split, build_mode in grid(rb.codegen_list(args.codegen), args.splits, args.build_modes):
        ok, payload = engine.run_isolated(ext_tu_job, (name, args.preset, codegen, split, build_mode, args.reps,
                                                       args.timeout), timeout=args.timeout)
        codegen_ms = f"{payload['codegen_ms']:.3f}" if ok and payload.get('codegen_ms') is not None else ''
        compile_ms = f"{payload['compile_ms']:.3f}" if ok and payload.get('compile_ms') is not None else ''
        runtime_ms = f"{payload['runtime_ms']:.6f}" if ok and payload.get('runtime_ms') is not None else ''
        error = '' if ok else str(payload)
        writer.writerow(dict(kernel=name, corpus=corpus, codegen=codegen, split=split, build_mode=build_mode,
                             preset=args.preset, threads=threads, codegen_ms=codegen_ms, compile_ms=compile_ms,
                             runtime_ms=runtime_ms, error=error))
        out_handle.flush()
        print(f"[{name}/{corpus}/{args.preset}] {codegen}/{split}/{build_mode}: "
              f"codegen={codegen_ms or '-'}ms compile={compile_ms or '-'}ms runtime={runtime_ms or '-'}ms"
              + (f' ({error})' if error else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', choices=('S', 'paper'), default='S', help='size preset (default: S)')
    ap.add_argument('--codegen', choices=('legacy', 'experimental', 'both'), default='both')
    ap.add_argument('--corpus', choices=('npbench', 'polybench', 'both'), default='both')
    ap.add_argument('--splits', default='off,on', help='comma list of {off,on} (default: off,on)')
    ap.add_argument('--build-modes', default='cmake,native', help='comma list of {cmake,native} (default: cmake,native)')
    ap.add_argument('--out', default='exttu_results.tsv', help='TSV output path')
    ap.add_argument('--only', default=None, help='substring filter on kernel name')
    ap.add_argument('--kernels', default=None, help='comma-separated explicit kernel list (overrides --corpus/--only)')
    ap.add_argument('--reps', type=int, default=10, help='timed repetitions per grid point (best-of)')
    ap.add_argument('--timeout', type=float, default=900.0, help='per-measurement subprocess timeout, seconds')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', '1')))
    ap.add_argument('--list-kernels', action='store_true')
    args = ap.parse_args()
    args.splits = tuple(s for s in args.splits.split(',') if s in SPLITS)
    args.build_modes = tuple(b for b in args.build_modes.split(',') if b in BUILD_MODES)

    kernels = rb.select_kernels(args.corpus, args.preset, args.only, args.kernels)
    if args.list_kernels:
        for k in kernels:
            print(k)
        return
    if not kernels:
        print('no kernels selected', file=sys.stderr)
        sys.exit(2)

    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    print(f"external-TU sweep: {len(kernels)} kernels x {len(grid(rb.codegen_list(args.codegen), args.splits, args.build_modes))} "
          f"grid points -> {args.out}")
    with open(args.out, 'w', newline='') as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=TSV_FIELDS, delimiter='\t')
        writer.writeheader()
        for name in kernels:
            try:
                process_kernel(writer, out_handle, name, args, args.threads)
            except Exception as ex:  # noqa: BLE001 - never let one kernel abort the sweep
                writer.writerow(dict(kernel=name, corpus=rb.kernel_corpus(name), codegen='', split='', build_mode='',
                                     preset=args.preset, threads=args.threads, codegen_ms='', compile_ms='',
                                     runtime_ms='', error=f'kernel-level: {ex}'))
                out_handle.flush()
    print(f"done -> {args.out}\n  plot: python3 {os.path.join(HERE, 'plot_external_tu.py')} {args.out}")


if __name__ == '__main__':
    main()
