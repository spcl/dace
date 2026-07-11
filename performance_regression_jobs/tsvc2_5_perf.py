#!/usr/bin/env python3
"""Performance regression (CPU only): the DaCe pipelines (parallel and
canonicalize / fast-canonicalize, each in a -par and a -seq schedule, plus
auto_opt) vs. two native C baselines -- single-core (native-clang) and
multi-core compiler auto-parallelization (native-clang-polly-autopar and gcc
native-gcc-autopar -- see native_harness.LANES) -- over the 65-kernel TSVC2.5
corpus. Standalone (only needs dace+numpy importable); multi-rank via
OMPI_COMM_WORLD_RANK/SLURM_PROCID (see engine.py).

    python3 tsvc2_5_perf.py                          # this rank's slice, 100 reps
    python3 tsvc2_5_perf.py --only heat3d --reps 3   # quick smoke test
    python3 tsvc2_5_perf.py --save-sdfg-only         # just dump canon/fast-canon SDFGs
    python3 tsvc2_5_perf.py --tables-only            # rebuild correctness.md/speedup.md
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import inspect
import math
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import dace
import engine
import native_harness as nh
import tsvc_2_5_corpus as tsvc25

CORPUS = 'tsvc2_5'
CPP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tsvc_2_5_core.cpp')
DACE_LANES = [f'{p}-{v}' for p in engine.PIPELINES for v in ('par', 'seq')]
ALL_LANES = DACE_LANES + list(nh.LANES)
#: Speedups in speedup.md are reported vs. the single-core native C baseline
#: (the boxplot additionally shows vs. the multi-core auto-par native lanes).
BASELINE_LANE = nh.SINGLE_CORE_LANE

_TARGET_BYTES = 2 * 1024**3
_SCALABLE = ('LEN_1D', 'LEN_2D', 'LEN_3D', 'LEN_R7')
_MAX_SCALE, _MIN_SCALE, _MAX_ITERS = 4096.0, 1.0, 8


def _program(name):
    # DaceProgram.name prefixes the enclosing module name (tsvc_2_5_corpus_...);
    # .f.__name__ is the bare function name, matching tsvc_2_5_core.cpp's symbols.
    return next(p for p in tsvc25.collect() if p.f.__name__ == name)


# --------------------------------------------------------------------------
# Sizing: symbolic per-kernel byte-sum (no real allocation) + geometric search
# over a single multiplier applied only to the LEN_* (size-like) symbols --
# the small structural symbols (S, SSYM, K, M, T, T1, T2) constrain loop/tile
# validity, not just memory, so they stay fixed at their corpus defaults.
# --------------------------------------------------------------------------
def _bytes_for_sizes(program, sizes):
    total = 0
    for _, par in inspect.signature(program.f).parameters.items():
        ann = par.annotation
        if getattr(ann, 'shape', None):
            shape = [int(dace.symbolic.evaluate(d, sizes)) for d in ann.shape]
            total += int(np.prod(shape)) * ann.dtype.bytes
    return total


def size_scale_for_kernel(program, target_bytes=_TARGET_BYTES):
    base = dict(tsvc25.SIZES)
    lo, hi = _MIN_SCALE, _MAX_SCALE
    best = dict(base)
    for _ in range(_MAX_ITERS):
        mid = math.sqrt(lo * hi)
        trial = {k: (max(1, int(v * mid)) if k in _SCALABLE else v) for k, v in base.items()}
        if 'LEN_R7' in trial:
            trial['LEN_R7'] = max(7, round(trial['LEN_R7'] / 7) * 7)
        b = _bytes_for_sizes(program, trial)
        best = trial
        if b == 0 or abs(b - target_bytes) / target_bytes < 0.15:
            break
        lo, hi = (mid, hi) if b < target_bytes else (lo, mid)
    # Final safety net (never allocates, only computes shape products): a
    # kernel combining several SCALABLE symbols non-multiplicatively could
    # still overshoot after MAX_ITERS -- shrink further until sane.
    while _bytes_for_sizes(program, best) > 4 * target_bytes:
        best = {k: (max(1, v // 2) if k in _SCALABLE else v) for k, v in best.items()}
    return best


# --------------------------------------------------------------------------
# Build one pipeline's SDFG (par or seq variant). The name -- unique per
# (corpus, kernel, pipeline, par/seq) -- is also its cache key (dace
# cache='name' in engine.configure_dace_process): the exact same variant,
# however many times it's independently rebuilt (another lane's reference
# check, a timing run right after its own correctness check, a resumed
# invocation), always lands on the same compiled binary instead of
# recompiling. SDFG names must be valid identifiers, no hyphens.
# --------------------------------------------------------------------------
def _build_sdfg(kernel_name, pipeline, seq):
    engine.configure_dace_process()
    program = _program(kernel_name)
    sdfg = program.to_sdfg(simplify=False)
    sdfg.name = f"{CORPUS}_{sdfg.name}_{pipeline.replace('-', '_')}_{'seq' if seq else 'par'}"
    sdfg = engine.PIPELINES[pipeline](sdfg)
    if seq:
        sdfg = engine.make_sequential(sdfg)
    return sdfg


def _symbol_values(sdfg, sizes):
    free = {str(s) for s in sdfg.free_symbols}
    for s in free:
        if s not in sdfg.symbols:
            sdfg.add_symbol(s, dace.int64)
    return {s: sizes[s] for s in sizes if s in free}


def _inputs(kernel_name, sizes):
    """make_inputs() reads the module-global SIZES dict, not a parameter --
    swap it in for the scaled sizes, then restore (this always runs inside its
    own spawned subprocess, but restoring keeps repeated in-process calls, e.g.
    the ref+candidate pair in _check_dace_job, from surprising each other)."""
    program = _program(kernel_name)
    original = dict(tsvc25.SIZES)
    tsvc25.SIZES.clear()
    tsvc25.SIZES.update(sizes)
    try:
        arrays, scalars = tsvc25.make_inputs(program, seed=1234)
    finally:
        tsvc25.SIZES.clear()
        tsvc25.SIZES.update(original)
    return program, arrays, scalars


# --------------------------------------------------------------------------
# Jobs run inside the isolated subprocess (recipe in, small result out).
# --------------------------------------------------------------------------
def _check_dace_job(kernel_name, sizes, pipeline, seq):
    _, ref_arrays, ref_scalars = _inputs(kernel_name, sizes)
    ref_sdfg = _build_sdfg(kernel_name, 'parallel', False)
    ref_sym = _symbol_values(ref_sdfg, sizes)
    ref_call = {**{n: a.copy() for n, a in ref_arrays.items()}, **ref_scalars, **ref_sym}
    ref_sdfg.compile()(**ref_call)

    _, cand_arrays, cand_scalars = _inputs(kernel_name, sizes)
    cand_sdfg = _build_sdfg(kernel_name, pipeline, seq)
    cand_sym = _symbol_values(cand_sdfg, sizes)
    cand_call = {**{n: a.copy() for n, a in cand_arrays.items()}, **cand_scalars, **cand_sym}
    cand_sdfg.compile()(**cand_call)

    return engine.arrays_close(ref_call, cand_call)


def _time_dace_job(kernel_name, sizes, pipeline, seq, reps):
    _, arrays, scalars = _inputs(kernel_name, sizes)
    sdfg = _build_sdfg(kernel_name, pipeline, seq)
    sym = _symbol_values(sdfg, sizes)
    call_kwargs = {**{n: a.copy() for n, a in arrays.items()}, **scalars, **sym}
    return engine.time_sdfg(sdfg, call_kwargs, reps)


def _check_native_job(kernel_name, sizes, so_path, c_name, sig):
    _, ref_arrays, ref_scalars = _inputs(kernel_name, sizes)
    ref_sdfg = _build_sdfg(kernel_name, 'parallel', False)
    ref_sym = _symbol_values(ref_sdfg, sizes)
    ref_call = {**{n: a.copy() for n, a in ref_arrays.items()}, **ref_scalars, **ref_sym}
    ref_sdfg.compile()(**ref_call)

    _, cand_arrays, cand_scalars = _inputs(kernel_name, sizes)
    lib = nh.load_library(so_path)
    nh.call_kernel(lib, c_name, sig, arrays=cand_arrays, len_1d=sizes.get('LEN_1D', 0), len_2d=sizes.get('LEN_2D', 0),
                   scalar_params=cand_scalars, symbols=sizes)
    return engine.arrays_close(ref_call, cand_arrays)


def _time_native_job(kernel_name, sizes, so_path, c_name, sig, reps, warmup=1):
    _, arrays, scalars = _inputs(kernel_name, sizes)
    lib = nh.load_library(so_path)
    times = []
    for i in range(warmup + reps):
        ns = nh.call_kernel(lib, c_name, sig, arrays=arrays, len_1d=sizes.get('LEN_1D', 0),
                            len_2d=sizes.get('LEN_2D', 0), scalar_params=scalars, symbols=sizes)
        if i >= warmup:
            times.append(ns / 1e6)
    return times


# --------------------------------------------------------------------------
# Per-kernel driver (same shape as tsvc2_perf.py).
# --------------------------------------------------------------------------


def _save_sdfg_job(kernel_name, pipeline, seq):
    """Runs isolated: returns the built (and validated) SDFG's JSON so the
    parent can write it to disk -- validate() happens in here too, so an
    InvalidSDFGError never escapes to the parent's sweep either."""
    sdfg = _build_sdfg(kernel_name, pipeline, seq)
    sdfg.validate()
    return sdfg.to_json()


def process_kernel(kernel_name, sizes, args, rank, native_libs):
    # DaCe lanes and native lanes are namespaced separately (engine.host_tag's
    # docstring) since native lanes shouldn't be invalidated by a --cxx change.
    kdir_dace = engine.kernel_dir(args.results_dir, CORPUS, kernel_name, 'default')
    kdir_native = engine.native_kernel_dir(args.results_dir, CORPUS, kernel_name, 'default')
    kdir = lambda lane: kdir_native if engine.lane_kind(lane) == 'native' else kdir_dace

    for lane in ALL_LANES:
        ldir = kdir(lane)
        status = None if args.force else engine.read_status(ldir, lane)
        if status is None:
            if engine.lane_kind(lane) == 'dace':
                pipeline, seq = lane.rsplit('-', 1)
                ok, correct = engine.run_isolated(_check_dace_job, (kernel_name, sizes, pipeline, seq == 'seq'),
                                                  timeout=args.timeout)
            else:
                so_path, c_name, sig = native_libs.get(lane, (None, None, None))
                if so_path is None:
                    ok, correct = False, 'compiler not available'
                else:
                    ok, correct = engine.run_isolated(_check_native_job, (kernel_name, sizes, so_path, c_name, sig),
                                                      timeout=args.timeout)
            engine.write_status(ldir, lane, bool(ok and correct), '' if ok and correct else str(correct))
            if args.save_sdfg and ok and engine.lane_kind(lane) == 'dace':
                pipeline, seq = lane.rsplit('-', 1)
                ok2, sdfg_json = engine.run_isolated(_save_sdfg_job, (kernel_name, pipeline, seq == 'seq'),
                                                     timeout=args.timeout)
                if ok2:
                    engine.save_sdfg(ldir, dace.SDFG.from_json(sdfg_json), lane)
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
        if engine.lane_kind(lane) == 'dace':
            pipeline, seq = lane.rsplit('-', 1)
            ok, payload = engine.run_isolated(_time_dace_job, (kernel_name, sizes, pipeline, seq == 'seq', remaining),
                                              timeout=args.timeout)
        else:
            so_path, c_name, sig = native_libs[lane]
            ok, payload = engine.run_isolated(_time_native_job, (kernel_name, sizes, so_path, c_name, sig, remaining),
                                              timeout=args.timeout)
        if ok:
            engine.append_results(ldir, lane, payload, have)
        else:
            engine.write_status(ldir, lane, False, str(payload))
    engine.write_run_meta(kdir_dace, rank=rank, reps_target=args.reps, sizes=sizes)
    engine.write_run_meta(kdir_native, rank=rank, reps_target=args.reps, sizes=sizes)


def save_sdfg_only(kernel_name, args):
    kdir = engine.kernel_dir(args.results_dir, CORPUS, kernel_name, 'default')
    if not args.force and all(os.path.exists(os.path.join(kdir, f'{p}-par.sdfg')) for p in ('canon', 'fast-canon')):
        print(f'[{kernel_name}] SDFGs already saved, skip')
        return
    for pipeline in ('canon', 'fast-canon'):
        ok, payload = engine.run_isolated(_save_sdfg_job, (kernel_name, pipeline, False), timeout=args.timeout)
        if not ok:
            print(f'[{kernel_name}] {pipeline}: {payload}')
            continue
        engine.save_sdfg(kdir, dace.SDFG.from_json(payload), f'{pipeline}-par')


def kernel_list(args):
    names = sorted(p.f.__name__ for p in tsvc25.collect())
    if args.only:
        names = [n for n in names if args.only in n]
    return names


def prepare_native_libs(results_dir, rank):
    """Each lane finds its own vendor's compiler independently (see
    native_harness.compile_lane) -- a vendor with no compiler installed is
    just skipped for that lane, not the whole corpus."""
    sigs = nh.parse_signatures(CPP_FILE)
    build_dir = engine.native_build_dir(results_dir, rank)
    out = {}
    for lane in nh.LANES:
        so_path = os.path.join(build_dir, f'lib_{lane}.so')
        ok, err = nh.compile_lane(CPP_FILE, so_path, lane)
        if ok:
            out[lane] = so_path
            print(f'compiled {lane}: {so_path}')
        else:
            print(f'{lane}: {err}')
    return sigs, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    engine.add_common_args(ap)
    ap.add_argument('--target-bytes', type=int, default=_TARGET_BYTES, help='workspace target in bytes per kernel')
    ap.add_argument('--len1d', type=int, default=None, help='global LEN_1D override (skips the sizing search)')
    ap.add_argument('--len2d', type=int, default=None, help='global LEN_2D override (skips the sizing search)')
    ap.add_argument('--len3d', type=int, default=None, help='global LEN_3D override (skips the sizing search)')
    args = ap.parse_args()

    if args.list_kernels:
        print('\n'.join(kernel_list(args)))
        return
    if args.tables_only:
        engine.write_tables(args.results_dir, CORPUS, ALL_LANES, BASELINE_LANE)
        engine.write_summary_csv(args.results_dir, CORPUS, BASELINE_LANE)
        return

    engine.export_cxx_override(args)
    cxx = engine.pick_cxx_compiler(args.cxx)  # fails fast on a bad --cxx, before any work starts
    print(f'C++ compiler (DaCe codegen): {cxx or "(none found; DaCe default)"}')

    rank, world = engine.get_world_rank(), engine.get_world_size()
    all_kernels = kernel_list(args)
    explicit = engine.explicit_kernel_list(args)
    mine = explicit if explicit is not None else engine.my_slice(all_kernels, rank, world)
    print(f'rank {rank}/{world}: {len(mine)}/{len(all_kernels)} kernels')

    sigs, compiled = ({}, {}) if args.save_sdfg_only else prepare_native_libs(args.results_dir, rank)

    overrides = {k: v for k, v in (('LEN_1D', args.len1d), ('LEN_2D', args.len2d), ('LEN_3D', args.len3d))
                if v is not None}
    for name in mine:
        if args.save_sdfg_only:
            save_sdfg_only(name, args)
            continue
        program = _program(name)
        if overrides:
            sizes = {**tsvc25.SIZES, **overrides}  # global override: skip the search, keep other stock defaults
        else:
            sizes = size_scale_for_kernel(program, target_bytes=args.target_bytes)
        c_name = name + '_run_timed'
        per_kernel_libs = {lane: (so_path, c_name, sigs.get(name, [])) for lane, so_path in compiled.items()}
        process_kernel(name, sizes, args, rank, per_kernel_libs)

    if not args.save_sdfg_only:
        engine.write_tables(args.results_dir, CORPUS, ALL_LANES, BASELINE_LANE)
        engine.write_summary_csv(args.results_dir, CORPUS, BASELINE_LANE)


if __name__ == '__main__':
    main()
