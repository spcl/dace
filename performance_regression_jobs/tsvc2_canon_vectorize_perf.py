#!/usr/bin/env python3
"""4th job type: canonicalize -> VectorizeCPU on TSVC2, vs. the DaCe
simplify+loop2map+mapfusion baseline (same reference/sizing/native-compare
machinery as tsvc2_perf.py, reused directly rather than duplicated).

    python3 tsvc2_canon_vectorize_perf.py --only s000 --reps 3
    python3 tsvc2_canon_vectorize_perf.py --tables-only
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine
import native_harness as nh
import tsvc2_perf as base  # reuse sizing, inputs, native-lib prep, cpp_base_name

CORPUS = 'tsvc2_canon_vectorize'
LANES = ('baseline-par', 'canon-vectorize') + tuple(nh.LANES)
BASELINE_LANE = 'baseline-par'


def _canon_vectorize(sdfg):
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
    canonicalize(sdfg, validate=True)
    VectorizeCPU(vector_width=8, fail_on_unvectorizable=False, remainder_strategy='scalar', use_fp_factor=False,
                branch_normalization=True).apply_pass(sdfg, {})
    return sdfg


def _build_sdfg(kernel_name, pipeline):
    engine.configure_dace_process()
    kernel = base.tsvc.collect(name=kernel_name)[0]
    sdfg = base.tsvc.to_sdfg(kernel, f'{kernel_name}_{pipeline}', simplify=False)
    return _canon_vectorize(sdfg) if pipeline == 'canon-vectorize' else engine.pipeline_baseline(sdfg)


def _check_dace_job(kernel_name, l1, l2, pipeline):
    _, ref_arrays, ref_sym, ref_sparams = base._inputs(kernel_name, l1, l2)
    ref_sdfg = _build_sdfg(kernel_name, 'baseline-par')
    try:
        ref_call = {**{n: a.copy() for n, a in ref_arrays.items()}, **ref_sparams, **ref_sym}
        ref_sdfg.compile()(**ref_call)
    finally:
        engine.cleanup_build_folder(ref_sdfg)

    _, cand_arrays, cand_sym, cand_sparams = base._inputs(kernel_name, l1, l2)
    cand_sdfg = _build_sdfg(kernel_name, pipeline)
    try:
        cand_call = {**{n: a.copy() for n, a in cand_arrays.items()}, **cand_sparams, **cand_sym}
        cand_sdfg.compile()(**cand_call)
    finally:
        engine.cleanup_build_folder(cand_sdfg)
    return base._compare(ref_call, cand_call)


def _time_dace_job(kernel_name, l1, l2, pipeline, reps):
    _, arrays, sym, sparams = base._inputs(kernel_name, l1, l2)
    sdfg = _build_sdfg(kernel_name, pipeline)
    try:
        call_kwargs = {**{n: a.copy() for n, a in arrays.items()}, **sparams, **sym}
        return engine.time_sdfg(sdfg, call_kwargs, reps)
    finally:
        engine.cleanup_build_folder(sdfg)


def _check_native_job(kernel_name, l1, l2, so_path, c_name, sig):
    _, ref_arrays, ref_sym, ref_sparams = base._inputs(kernel_name, l1, l2)
    ref_sdfg = _build_sdfg(kernel_name, 'baseline-par')
    try:
        ref_call = {**{n: a.copy() for n, a in ref_arrays.items()}, **ref_sparams, **ref_sym}
        ref_sdfg.compile()(**ref_call)
    finally:
        engine.cleanup_build_folder(ref_sdfg)
    _, cand_arrays, cand_sym, cand_sparams = base._inputs(kernel_name, l1, l2)
    lib = nh.load_library(so_path)
    nh.call_kernel(lib, c_name, sig, arrays=cand_arrays, len_1d=l1, len_2d=l2, scalar_params=cand_sparams,
                   symbols=cand_sym)
    return base._compare(ref_call, cand_arrays)


def _time_native_job(kernel_name, l1, l2, so_path, c_name, sig, reps, warmup=1):
    _, arrays, sym, sparams = base._inputs(kernel_name, l1, l2)
    lib = nh.load_library(so_path)
    times = []
    for i in range(warmup + reps):
        ns = nh.call_kernel(lib, c_name, sig, arrays=arrays, len_1d=l1, len_2d=l2, scalar_params=sparams,
                            symbols=sym)
        if i >= warmup:
            times.append(ns / 1e6)
    return times


def _lane_kind(lane):
    return 'native' if lane in nh.LANES else 'dace'


def process_kernel(kernel_name, l1, l2, args, native_libs):
    # DaCe lanes (depend on --cxx) and native lanes (each pick their own vendor
    # compiler, independent of --cxx) are namespaced separately -- see
    # engine.host_tag()'s docstring for why.
    kdir_dace = engine.kernel_dir(args.results_dir, CORPUS, kernel_name, 'default')
    kdir_native = engine.native_kernel_dir(args.results_dir, CORPUS, kernel_name, 'default')
    kdir = lambda lane: kdir_native if _lane_kind(lane) == 'native' else kdir_dace

    for lane in LANES:
        ldir = kdir(lane)
        status = None if args.force else engine.read_status(ldir, lane)
        if status is None:
            if _lane_kind(lane) == 'dace':
                ok, correct = engine.run_isolated(_check_dace_job, (kernel_name, l1, l2, lane), timeout=args.timeout)
            else:
                so_path, c_name, sig = native_libs.get(lane, (None, None, None))
                if so_path is None:
                    ok, correct = False, 'compiler not available'
                else:
                    ok, correct = engine.run_isolated(_check_native_job, (kernel_name, l1, l2, so_path, c_name, sig),
                                                      timeout=args.timeout)
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
        if _lane_kind(lane) == 'dace':
            ok, payload = engine.run_isolated(_time_dace_job, (kernel_name, l1, l2, lane, remaining),
                                              timeout=args.timeout)
        else:
            so_path, c_name, sig = native_libs[lane]
            ok, payload = engine.run_isolated(_time_native_job, (kernel_name, l1, l2, so_path, c_name, sig, remaining),
                                              timeout=args.timeout)
        if ok:
            engine.append_results(ldir, lane, payload, have)
        else:
            engine.write_status(ldir, lane, False, str(payload))
    engine.write_run_meta(kdir_dace, reps_target=args.reps, len_1d=l1, len_2d=l2)
    engine.write_run_meta(kdir_native, reps_target=args.reps, len_1d=l1, len_2d=l2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    engine.add_common_args(ap)
    ap.add_argument('--len1d', type=int, default=None)
    ap.add_argument('--len2d', type=int, default=None)
    ap.add_argument('--native-threads', type=int, default=4, help='GCC -ftree-parallelize-loops thread count')
    args = ap.parse_args()

    if args.list_kernels:
        print('\n'.join(base.kernel_list(args)))
        return
    if args.tables_only:
        engine.write_tables(args.results_dir, CORPUS, list(LANES), BASELINE_LANE)
        return

    engine.export_cxx_override(args)
    cxx = engine.pick_cxx_compiler(args.cxx)  # fails fast on a bad --cxx, before any work starts
    print(f'C++ compiler (DaCe codegen): {cxx or "(none found; DaCe default)"}')

    rank, world = engine.get_world_rank(), engine.get_world_size()
    all_kernels = base.kernel_list(args)
    explicit = engine.explicit_kernel_list(args)
    mine = explicit if explicit is not None else engine.my_slice(all_kernels, rank, world)
    print(f'rank {rank}/{world}: {len(mine)}/{len(all_kernels)} kernels')

    sigs, compiled = base.prepare_native_libs(args.results_dir, rank, nthreads=args.native_threads)

    for name in mine:
        kernel = base.tsvc.collect(name=name)[0]
        l1, l2 = (args.len1d, args.len2d) if args.len1d and args.len2d else base.size_for_kernel(kernel)
        cname = base.cpp_base_name(name) + '_run_timed'
        native_libs = {
            lane: (so_path, cname, sigs.get(base.cpp_base_name(name), []))
            for lane, (so_path, _) in compiled.items()
        }
        process_kernel(name, l1, l2, args, native_libs)

    engine.write_tables(args.results_dir, CORPUS, list(LANES), BASELINE_LANE)


if __name__ == '__main__':
    main()
