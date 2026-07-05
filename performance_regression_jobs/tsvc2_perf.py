#!/usr/bin/env python3
"""Performance regression: DaCe canonicalize/fast-canonicalize vs. a DaCe
loop2map+mapfusion baseline vs. native C++ built with 4 vendor toolchains
(GCC, Clang/LLVM, Intel oneAPI, NVIDIA HPC SDK), each in a plain-serial and
that vendor's own auto-parallelized form (see native_harness.LANES), over
the 151-kernel TSVC2 corpus. Standalone (only needs dace+numpy importable);
multi-rank via OMPI_COMM_WORLD_RANK/SLURM_PROCID (see engine.py).

    python3 tsvc2_perf.py                       # this rank's slice, 100 reps
    python3 tsvc2_perf.py --only s000 --reps 3  # quick smoke test
    python3 tsvc2_perf.py --save-sdfg-only      # just dump canon/fast-canon SDFGs
    python3 tsvc2_perf.py --tables-only         # rebuild correctness.md/speedup.md
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dace
import numpy as np

import engine
import native_harness as nh
import tsvc_corpus as tsvc

CORPUS = 'tsvc2'
CPP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tsvc2_core.cpp')
DACE_LANES = [f'{p}-{v}' for p in engine.PIPELINES for v in ('par', 'seq')]
ALL_LANES = DACE_LANES + list(nh.LANES)
BASELINE_LANE = 'baseline-par'

_TARGET_BYTES = 2 * 1024**3
_REF_1D, _REF_2D = 64, tsvc.LEN_2D_FIXED
_MAX_LEN_1D, _MAX_LEN_2D = 2_000_000, 20_000
_MAX_ACCEPTABLE_BYTES = 4 * _TARGET_BYTES  # hard ceiling regardless of what the search proposes


# --------------------------------------------------------------------------
# Sizing: the documented linear (1d-regime) / quadratic (2d-regime) scaling
# law, refined by an ANALYTICAL byte count (shape x dtype size). Never
# allocate a candidate size for real -- a kernel that doesn't actually match
# the assumed regime's scaling law (e.g. a 1d-regime kernel with a hidden
# LEN_1D**2 array) can propose a trial size many orders of magnitude too
# large, and actually allocating it would crash with an ArrayMemoryError
# before the loop gets a chance to correct downward.
# --------------------------------------------------------------------------
def _bytes_for_len(kernel, l1, l2):
    total = 0
    for _, desc in kernel.program.to_sdfg(simplify=False).arglist().items():
        if isinstance(desc, dace.data.Array):
            shape = tsvc._concrete_shape(desc, l1, l2)
            total += int(np.prod(shape)) * desc.dtype.bytes
    return total


def size_for_kernel(kernel):
    if kernel.regime == '1d':
        l1_ref, l2_ref = tsvc.regime_sizes('1d', _REF_1D)
    else:
        l1_ref, l2_ref = tsvc.regime_sizes('2d', _REF_2D)
    bytes_ref = _bytes_for_len(kernel, l1_ref, l2_ref)
    if bytes_ref == 0:
        return l1_ref, l2_ref
    swept_ref = l1_ref if kernel.regime == '1d' else l2_ref
    scale = (_TARGET_BYTES / bytes_ref) if kernel.regime == '1d' else math.sqrt(_TARGET_BYTES / bytes_ref)
    swept = int(swept_ref * scale)
    cap = _MAX_LEN_1D if kernel.regime == '1d' else _MAX_LEN_2D
    for _ in range(8):
        swept = max(1, min(swept, cap))
        l1, l2 = tsvc.regime_sizes(kernel.regime, swept)
        bytes_now = _bytes_for_len(kernel, l1, l2)
        if bytes_now == 0 or abs(bytes_now - _TARGET_BYTES) / _TARGET_BYTES < 0.15:
            break
        ratio = _TARGET_BYTES / bytes_now
        swept = int(swept * (ratio if kernel.regime == '1d' else math.sqrt(ratio)))
    # Final safety net: the assumed linear/quadratic law can still be wrong
    # for an unusual kernel: shrink further, geometrically, until the real
    # (analytical) byte count is sane -- this can never crash since it never
    # allocates, only computes shape products.
    swept = max(1, min(swept, cap))
    l1, l2 = tsvc.regime_sizes(kernel.regime, swept)
    while _bytes_for_len(kernel, l1, l2) > _MAX_ACCEPTABLE_BYTES and swept > 1:
        swept = max(1, swept // 2)
        l1, l2 = tsvc.regime_sizes(kernel.regime, swept)
    return l1, l2


# --------------------------------------------------------------------------
# Build one pipeline's SDFG (par or seq variant). Cache isolation is
# per-process (dace cache=unique, keyed on PID + sdfg.name) rather than a
# manually constructed path -- see engine.configure_dace_process.
# --------------------------------------------------------------------------
def _build_sdfg(kernel_name, pipeline, seq):
    engine.configure_dace_process()
    kernel = tsvc.collect(name=kernel_name)[0]
    tag = f'{kernel_name}_{pipeline}'
    sdfg = tsvc.to_sdfg(kernel, tag, simplify=False)
    sdfg = engine.PIPELINES[pipeline](sdfg)
    if seq:
        sdfg = engine.make_sequential(sdfg)
    return sdfg


def _inputs(kernel_name, l1, l2, seed_extra=''):
    kernel = tsvc.collect(name=kernel_name)[0]
    rng = np.random.default_rng(tsvc.stable_seed((kernel_name, l1, l2, seed_extra)))
    arrays = tsvc.allocate(kernel, l1, l2, rng)
    sym = tsvc.symbols(kernel, l1, l2)
    sparams = tsvc.scalar_params(kernel, l1)
    return kernel, arrays, sym, sparams


# --------------------------------------------------------------------------
# Jobs run inside the isolated subprocess (engine.run_isolated forks each of
# these; a segfault or exception anywhere in here -- transform, compile, or
# run -- only takes down this one job, never the rank's remaining sweep).
# Top-level + picklable; inputs are rebuilt from a small recipe (never
# shipped as arrays across the process boundary), matching the
# deterministic-seed requirement.
# --------------------------------------------------------------------------
def _check_dace_job(kernel_name, l1, l2, pipeline, seq):
    _, ref_arrays, ref_sym, ref_sparams = _inputs(kernel_name, l1, l2)
    ref_sdfg = _build_sdfg(kernel_name, 'baseline', False)
    try:
        ref_call = {**{n: a.copy() for n, a in ref_arrays.items()}, **ref_sparams, **ref_sym}
        ref_sdfg.compile()(**ref_call)
    finally:
        engine.cleanup_build_folder(ref_sdfg)

    _, cand_arrays, cand_sym, cand_sparams = _inputs(kernel_name, l1, l2)
    cand_sdfg = _build_sdfg(kernel_name, pipeline, seq)
    try:
        cand_call = {**{n: a.copy() for n, a in cand_arrays.items()}, **cand_sparams, **cand_sym}
        cand_sdfg.compile()(**cand_call)
    finally:
        engine.cleanup_build_folder(cand_sdfg)

    return _compare(ref_call, cand_call)


def _time_dace_job(kernel_name, l1, l2, pipeline, seq, reps):
    _, arrays, sym, sparams = _inputs(kernel_name, l1, l2)
    sdfg = _build_sdfg(kernel_name, pipeline, seq)
    try:
        call_kwargs = {**{n: a.copy() for n, a in arrays.items()}, **sparams, **sym}
        return engine.time_sdfg(sdfg, call_kwargs, reps)
    finally:
        engine.cleanup_build_folder(sdfg)


def _check_native_job(kernel_name, l1, l2, so_path, c_name, sig):
    _, ref_arrays, ref_sym, ref_sparams = _inputs(kernel_name, l1, l2)
    ref_sdfg = _build_sdfg(kernel_name, 'baseline', False)
    try:
        ref_call = {**{n: a.copy() for n, a in ref_arrays.items()}, **ref_sparams, **ref_sym}
        ref_sdfg.compile()(**ref_call)
    finally:
        engine.cleanup_build_folder(ref_sdfg)

    _, cand_arrays, cand_sym, cand_sparams = _inputs(kernel_name, l1, l2)
    lib = nh.load_library(so_path)
    nh.call_kernel(lib, c_name, sig, arrays=cand_arrays, len_1d=l1, len_2d=l2, scalar_params=cand_sparams,
                   symbols=cand_sym)
    return _compare(ref_call, cand_arrays)


def _time_native_job(kernel_name, l1, l2, so_path, c_name, sig, reps, warmup=1):
    _, arrays, sym, sparams = _inputs(kernel_name, l1, l2)
    lib = nh.load_library(so_path)
    times = []
    for i in range(warmup + reps):
        ns = nh.call_kernel(lib, c_name, sig, arrays=arrays, len_1d=l1, len_2d=l2, scalar_params=sparams,
                            symbols=sym)
        if i >= warmup:
            times.append(ns / 1e6)
    return times


def _compare(ref, got, tol=1e-9):
    for name, r in ref.items():
        g = got.get(name)
        if g is None or not np.issubdtype(np.asarray(r).dtype, np.floating):
            continue
        if not np.allclose(r, g, rtol=tol, atol=tol, equal_nan=True):
            return False
    return True


# --------------------------------------------------------------------------
# Per-kernel driver.
# --------------------------------------------------------------------------
def _lane_kind(lane):
    return 'native' if lane in nh.LANES else 'dace'


def _save_sdfg_job(kernel_name, pipeline, seq):
    """Runs isolated: returns the built (and validated) SDFG's JSON so the
    parent can write it to disk -- validate() happens in here too, so an
    InvalidSDFGError never escapes to the parent's sweep either."""
    sdfg = _build_sdfg(kernel_name, pipeline, seq)
    sdfg.validate()
    return sdfg.to_json()


def process_kernel(kernel_name, l1, l2, args, rank, native_libs):
    # DaCe lanes (depend on --cxx) and native lanes (each pick their own vendor
    # compiler, independent of --cxx) are namespaced separately -- see
    # engine.host_tag()'s docstring for why -- so results.csv/status.csv for
    # the two kinds of lane live in two different folders.
    kdir_dace = engine.kernel_dir(args.results_dir, CORPUS, kernel_name, 'default')
    kdir_native = engine.native_kernel_dir(args.results_dir, CORPUS, kernel_name, 'default')
    kdir = lambda lane: kdir_native if _lane_kind(lane) == 'native' else kdir_dace

    # Every lane must have a status entry already (not just "some folder has
    # some results") before considering the kernel done -- a fresh dace-tag
    # folder (new --cxx against a --results-dir whose native-tag folder is
    # already fully measured) must NOT be mistaken for "fully populated": its
    # own lanes have no status yet, so `all(...)` on an empty/no-status
    # generator would otherwise be vacuously true and skip real work.
    if not args.force and all(engine.read_status(kdir(lane), lane) is not None for lane in ALL_LANES) and all(
            engine.existing_reps(kdir(lane), lane) >= args.reps for lane in ALL_LANES
            if (engine.read_status(kdir(lane), lane) or {}).get('correct') == 'True'):
        print(f'[{kernel_name}] fully populated, skip')
        return

    for lane in ALL_LANES:
        ldir = kdir(lane)
        status = None if args.force else engine.read_status(ldir, lane)
        if status is None:
            if _lane_kind(lane) == 'dace':
                pipeline, seq = lane.rsplit('-', 1)
                ok, correct = engine.run_isolated(_check_dace_job, (kernel_name, l1, l2, pipeline, seq == 'seq'),
                                                  timeout=args.timeout)
            else:
                so_path, c_name, sig = native_libs.get(lane, (None, None, None))
                if so_path is None:
                    ok, correct = False, 'compiler not available'
                else:
                    ok, correct = engine.run_isolated(_check_native_job, (kernel_name, l1, l2, so_path, c_name, sig),
                                                      timeout=args.timeout)
            engine.write_status(ldir, lane, bool(ok and correct), '' if ok and correct else str(correct))
            if args.save_sdfg and ok and _lane_kind(lane) == 'dace':
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
        if _lane_kind(lane) == 'dace':
            pipeline, seq = lane.rsplit('-', 1)
            ok, payload = engine.run_isolated(_time_dace_job, (kernel_name, l1, l2, pipeline, seq == 'seq', remaining),
                                              timeout=args.timeout)
        else:
            so_path, c_name, sig = native_libs[lane]
            ok, payload = engine.run_isolated(_time_native_job, (kernel_name, l1, l2, so_path, c_name, sig, remaining),
                                              timeout=args.timeout)
        if ok:
            engine.append_results(ldir, lane, payload, have)
        else:
            engine.write_status(ldir, lane, False, str(payload))
    engine.write_run_meta(kdir_dace, rank=rank, reps_target=args.reps, len_1d=l1, len_2d=l2)
    engine.write_run_meta(kdir_native, rank=rank, reps_target=args.reps, len_1d=l1, len_2d=l2)


def save_sdfg_only(kernel_name, l1, l2, args, rank):
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


def cpp_base_name(kernel_name):
    """tsvc2_core.cpp has no _d_single suffix (e.g. 's000_run_timed', not 's000_d_single_run_timed')."""
    return kernel_name[:-len('_d_single')] if kernel_name.endswith('_d_single') else kernel_name


def kernel_list(args):
    names = sorted(k.name for k in tsvc.collect())
    if args.only:
        names = [n for n in names if args.only in n]
    return names


def prepare_native_libs(results_dir, rank, nthreads=4):
    """Compile every native lane once (per rank); returns lane -> (so_path, prefix, sigs).

    Each lane finds its own vendor's compiler independently (see
    native_harness.compile_lane) -- a vendor with no compiler installed is
    just skipped for that lane, not the whole corpus."""
    sigs = nh.parse_signatures(CPP_FILE)
    build_dir = engine.native_build_dir(results_dir, rank)
    out = {}
    for lane in nh.LANES:
        so_path = os.path.join(build_dir, f'lib_{lane}.so')
        ok, err = nh.compile_lane(CPP_FILE, so_path, lane, nthreads=nthreads)
        if ok:
            out[lane] = (so_path, sigs)
            print(f'compiled {lane}: {so_path}')
        else:
            print(f'{lane}: {err}')
    return sigs, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    engine.add_common_args(ap)
    ap.add_argument('--len1d', type=int, default=None, help='override LEN_1D (skips the sizing search)')
    ap.add_argument('--len2d', type=int, default=None, help='override LEN_2D (skips the sizing search)')
    ap.add_argument('--native-threads', type=int, default=4, help='GCC -ftree-parallelize-loops thread count')
    args = ap.parse_args()

    if args.list_kernels:
        print('\n'.join(kernel_list(args)))
        return
    if args.tables_only:
        engine.write_tables(args.results_dir, CORPUS, ALL_LANES, BASELINE_LANE)
        return

    engine.export_cxx_override(args)
    cxx = engine.pick_cxx_compiler(args.cxx)  # fails fast on a bad --cxx, before any work starts
    print(f'C++ compiler (DaCe codegen): {cxx or "(none found; DaCe default)"}')

    rank, world = engine.get_world_rank(), engine.get_world_size()
    all_kernels = kernel_list(args)
    explicit = engine.explicit_kernel_list(args)
    mine = explicit if explicit is not None else engine.my_slice(all_kernels, rank, world)
    print(f'rank {rank}/{world}: {len(mine)}/{len(all_kernels)} kernels')

    native_libs = {}
    if not args.save_sdfg_only:
        sigs, compiled = prepare_native_libs(args.results_dir, rank, nthreads=args.native_threads)
        for lane, (so_path, _) in compiled.items():
            native_libs[lane] = None  # filled per-kernel below with the kernel's own signature
        _sigs, _compiled = sigs, compiled

    for name in mine:
        kernel = tsvc.collect(name=name)[0]
        l1, l2 = (args.len1d, args.len2d) if args.len1d and args.len2d else size_for_kernel(kernel)
        if args.save_sdfg_only:
            save_sdfg_only(name, l1, l2, args, rank)
            continue
        base = cpp_base_name(name)
        c_name = base + '_run_timed'
        per_kernel_libs = {
            lane: (so_path, c_name, _sigs.get(base, []))
            for lane, (so_path, _) in _compiled.items()
        }
        process_kernel(name, l1, l2, args, rank, per_kernel_libs)

    if not args.save_sdfg_only:
        engine.write_tables(args.results_dir, CORPUS, ALL_LANES, BASELINE_LANE)


if __name__ == '__main__':
    main()
