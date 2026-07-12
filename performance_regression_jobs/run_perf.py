#!/usr/bin/env python3
"""Unified performance-benchmark runner: ONE driver for two experiments across
four corpora (npbench, polybench, tsvc2, tsvc2_5). One invocation = one
(experiment, corpus) job; kernels self-partition across ranks (SLURM_PROCID /
NTASKS via engine.my_slice), so 1 node / 4 ranks distributes the corpus.

Experiments (both CPU; vector_vs runs its dace lanes through the multi-dim
vectorizer, native lanes are compiled fast-math either way):

  canon_vs   dace-autoopt      engine.pipeline_auto_opt
             dace-canon        engine.pipeline_canon
             dace-parallel     engine.pipeline_parallel
             compiler-seq      native single-core -O3 -march=native -ffast-math
             compiler-autopar  native gcc auto-parallelized C
             numpy             timed numpy oracle -- npbench/polybench ONLY

  vector_vs  dace-canon-vec    engine.pipeline_canon_vectorize
             dace-parallel-vec engine.pipeline_parallel_vectorize
             compiler-seq      native single-core -O3 -march=native -ffast-math
             compiler-autopar  native gcc auto-parallelized C
             numpy             timed numpy oracle -- npbench/polybench ONLY

Both experiments use the SAME two native lanes (compiler-seq is autovectorized
single-thread; compiler-autopar, at -O3, also autovectorizes). Baseline speedups
are reported against: numpy (npbench/polybench) or compiler-seq (tsvc2/tsvc2_5).
tsvc2/tsvc2_5 have no numpy oracle, so the numpy lane is omitted for them.
npbench/polybench have no native C source, so the native lanes record 'compiler
not available' and are skipped there.

    python3 run_perf.py --experiment canon_vs  --corpus tsvc2  --reps 25
    python3 run_perf.py --experiment vector_vs --corpus npbench --kernels gemm --reps 2
    python3 run_perf.py --experiment canon_vs  --corpus polybench --tables-only
"""
import os

# MPI anti-hang: set BEFORE importing dace / mpi4py / any corpus module (a dace
# script otherwise blocks on MPI_Init and the sweep looks hung). setdefault so an
# explicit value (a real MPI launch) still wins.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adapters
import engine
import native_harness as nh

#: experiment -> its dace + native lane sets (numpy is added per-corpus below).
EXPERIMENTS = {
    'canon_vs': dict(dace=['dace-autoopt', 'dace-canon', 'dace-parallel'], native=['compiler-seq', 'compiler-autopar']),
    'vector_vs': dict(dace=['dace-canon-vec', 'dace-parallel-vec'], native=['compiler-seq', 'compiler-autopar']),
}

#: single-device sweep; every result folder uses this preset token.
PRESET = 'default'


def corpus_lanes(experiment, corpus):
    """Ordered lane list for (experiment, corpus): dace, native, then numpy
    (npbench/polybench only). This is also the tables' column order."""
    spec = EXPERIMENTS[experiment]
    lanes = list(spec['dace']) + list(spec['native'])
    if adapters.adapter(corpus)['has_numpy']:
        lanes.append('numpy')
    return lanes


def baseline_lane(corpus):
    return 'numpy' if adapters.adapter(corpus)['has_numpy'] else 'compiler-seq'


def _kdir(lane, kdir_dace, kdir_native):
    """Native lanes are namespaced separately (engine.host_tag) so they aren't
    invalidated by a --cxx change; dace and numpy lanes share the dace-tag folder."""
    return kdir_native if lane in nh.LANES else kdir_dace


def _check_lane(corpus, kernel_name, recipe, lane, native_libs, has_numpy_ref, timeout):
    if lane in adapters.DACE_PIPELINE:
        return engine.run_isolated(adapters.check_dace_job, (corpus, kernel_name, recipe, lane), timeout=timeout)
    if lane in nh.LANES:
        entry = native_libs.get(lane)
        if entry is None:
            return False, 'compiler not available'
        so_path, c_name, sig = entry
        return engine.run_isolated(adapters.check_native_job, (corpus, kernel_name, recipe, so_path, c_name, sig),
                                   timeout=timeout)
    if lane == 'numpy':
        if not has_numpy_ref:
            return False, 'no numpy ref'
        return engine.run_isolated(adapters.check_numpy_job, (corpus, kernel_name, recipe), timeout=timeout)
    return False, f'unknown lane {lane}'


def _time_lane(corpus, kernel_name, recipe, lane, reps, native_libs, timeout):
    if lane in adapters.DACE_PIPELINE:
        return engine.run_isolated(adapters.time_dace_job, (corpus, kernel_name, recipe, lane, reps), timeout=timeout)
    if lane in nh.LANES:
        so_path, c_name, sig = native_libs[lane]
        return engine.run_isolated(adapters.time_native_job, (corpus, kernel_name, recipe, so_path, c_name, sig, reps),
                                   timeout=timeout)
    if lane == 'numpy':
        return engine.run_isolated(adapters.time_numpy_job, (corpus, kernel_name, recipe, reps), timeout=timeout)
    return False, f'unknown lane {lane}'


def kernel_complete(lanes, kdir_dace, kdir_native, reps):
    """Fast per-kernel resume check (no build/compile/check): a kernel is done
    when every lane has a status entry AND every CORRECT lane already has >=reps
    recorded reps. A known-incorrect lane (vectorizer refused / compiler absent)
    never reaches reps, so it only needs a status entry -- otherwise a single
    failing lane would make the kernel un-skippable forever."""
    for lane in lanes:
        kd = _kdir(lane, kdir_dace, kdir_native)
        status = engine.read_status(kd, lane)
        if status is None:
            return False
        if status.get('correct') == 'True' and engine.existing_reps(kd, lane) < reps:
            return False
    return True


def process_kernel(corpus, kernel_name, recipe, lanes, args, rank, native_libs, has_numpy_ref):
    kdir_dace = engine.kernel_dir(args.results_dir, corpus, kernel_name, PRESET)
    kdir_native = engine.native_kernel_dir(args.results_dir, corpus, kernel_name, PRESET)

    if not args.force and kernel_complete(lanes, kdir_dace, kdir_native, args.reps):
        print(f'[{kernel_name}] already complete, skipping')
        return

    for lane in lanes:
        ldir = _kdir(lane, kdir_dace, kdir_native)
        status = None if args.force else engine.read_status(ldir, lane)
        if status is None:
            ok, correct = _check_lane(corpus, kernel_name, recipe, lane, native_libs, has_numpy_ref, args.timeout)
            engine.write_status(ldir, lane, bool(ok and correct), '' if ok and correct else str(correct))
            correct_now = bool(ok and correct)
        else:
            correct_now = status['correct'] == 'True'
        if not correct_now:
            continue

        have = 0 if args.force else engine.existing_reps(ldir, lane)
        remaining = args.reps - have
        if remaining <= 0:
            continue
        print(f'[{kernel_name}] {lane}: measuring {remaining} more rep(s)')
        ok, payload = _time_lane(corpus, kernel_name, recipe, lane, remaining, native_libs, args.timeout)
        if ok:
            engine.append_results(ldir, lane, payload, have)
        else:
            engine.write_status(ldir, lane, False, str(payload))

    engine.write_run_meta(kdir_dace, rank=rank, reps_target=args.reps, experiment=args.experiment, recipe=recipe)
    engine.write_run_meta(kdir_native, rank=rank, reps_target=args.reps, experiment=args.experiment)


def prepare_native_libs(corpus, experiment, args, rank):
    """Compile the experiment's native lanes once per rank. Returns (sigs, {lane:
    so_path}); ({}, {}) if the corpus has no native C source (npbench/polybench).
    Each lane finds its own vendor's compiler; a missing one is skipped."""
    cpp_path, sigs = adapters.adapter(corpus)['native_info']()
    if cpp_path is None:
        return {}, {}
    build_dir = engine.native_build_dir(args.results_dir, rank, corpus)
    compiled = {}
    for lane in EXPERIMENTS[experiment]['native']:
        so_path = os.path.join(build_dir, f'lib_{lane}.so')
        ok, err = nh.compile_lane(cpp_path, so_path, lane)
        if ok:
            compiled[lane] = so_path
            print(f'compiled {lane}: {so_path}')
        else:
            print(f'{lane}: {err}')
    return sigs, compiled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--experiment', required=True, choices=sorted(EXPERIMENTS), help='which experiment to run')
    ap.add_argument('--corpus', required=True, choices=sorted(adapters.ADAPTERS), help='which corpus to sweep')
    engine.add_common_args(ap)
    args = ap.parse_args()

    corpus, experiment = args.corpus, args.experiment
    a = adapters.adapter(corpus)
    lanes = corpus_lanes(experiment, corpus)
    base = baseline_lane(corpus)

    if args.list_kernels:
        print('\n'.join(a['kernel_list']()))
        return
    if args.tables_only:
        engine.write_tables(args.results_dir, corpus, lanes, base)
        engine.write_summary_csv(args.results_dir, corpus, base)
        return

    engine.export_cxx_override(args)
    cxx = engine.pick_cxx_compiler(args.cxx)  # fails fast on a bad --cxx, before any work starts
    print(f'C++ compiler (DaCe codegen): {cxx or "(none found; DaCe default)"}')

    rank, world = engine.get_world_rank(), engine.get_world_size()
    all_kernels = a['kernel_list']()
    if args.only:
        all_kernels = [k for k in all_kernels if args.only in k]
    explicit = engine.explicit_kernel_list(args)
    mine = explicit if explicit is not None else engine.my_slice(all_kernels, rank, world)
    print(f'rank {rank}/{world}: {len(mine)}/{len(all_kernels)} kernels; experiment={experiment} corpus={corpus}')
    print(f'lanes: {lanes} | baseline: {base}')

    sigs, compiled = prepare_native_libs(corpus, experiment, args, rank)

    for name in mine:
        recipe = a['recipe'](name)
        has_numpy_ref = a['has_numpy'] and adapters.numpy_ref_available(name)
        native_libs = {}
        if compiled:
            c_name, sig = a['native_call'](name, sigs)
            native_libs = {lane: (so_path, c_name, sig) for lane, so_path in compiled.items()}
        process_kernel(corpus, name, recipe, lanes, args, rank, native_libs, has_numpy_ref)

    engine.write_tables(args.results_dir, corpus, lanes, base)
    engine.write_summary_csv(args.results_dir, corpus, base)


if __name__ == '__main__':
    main()
