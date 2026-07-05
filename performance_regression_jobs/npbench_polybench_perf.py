#!/usr/bin/env python3
"""Performance regression: DaCe canonicalize/fast-canonicalize vs. a DaCe
simplify+loop2map+mapfusion baseline vs. DaCe CPU auto_optimize vs. a timed
numpy baseline, over the combined NPBench+PolyBench corpus. Fully
self-contained: the kernel definitions are copied from this dace repo's own
tests/corpus/{npbench,polybench} (npbench_corpus/, polybench_corpus/), and
the "paper"-preset dataset sizes + numpy reference implementations are
vendored from the real upstream npbench repo (bench_info/, npbench_numpy_refs/)
since this dace-repo port has no "paper" preset and no numpy oracle at all
for polybench kernels. Nothing here imports the external npbench package.

    python3 npbench_polybench_perf.py                    # this rank's slice, 100 reps
    python3 npbench_polybench_perf.py --only gemm --reps 3
    python3 npbench_polybench_perf.py --save-sdfg-only   # just dump canon/fast-canon SDFGs
    python3 npbench_polybench_perf.py --tables-only      # rebuild correctness.md/speedup.md
"""
import argparse
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import dace
import engine
import npbench_corpus.npbench as npb
import polybench_corpus.polybench as pb

CORPUS = 'npbench_polybench'
_HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_INFO_DIR = os.path.join(_HERE, 'bench_info')
PRESET = 'paper'

DACE_LANES = ('baseline', 'auto-opt', 'canon', 'fast-canon')
ALL_LANES = DACE_LANES + ('numpy', )
BASELINE_LANE = 'baseline'
#: The 3 comparisons the grid plot shows, canon against each.
SPEEDUP_VS = ('baseline', 'auto-opt', 'numpy')


def _dace_pipeline(label, sdfg):
    if label == 'baseline':
        return engine.pipeline_baseline(sdfg)
    if label == 'auto-opt':
        from dace.transformation.auto.auto_optimize import auto_optimize
        return auto_optimize(sdfg, dace.DeviceType.CPU)
    if label == 'canon':
        return engine.pipeline_canon(sdfg)
    if label == 'fast-canon':
        return engine.pipeline_fast_canon(sdfg)
    raise ValueError(label)


# --------------------------------------------------------------------------
# bench_info: which suite (poly vs np) an entry belongs to, and its paper-preset
# parameters. module_name is the join key against npbench_corpus/polybench_corpus
# (short_name differs for several kernels, e.g. go_fast vs npgofast -- verified).
# --------------------------------------------------------------------------
def load_bench_info(name):
    with open(os.path.join(BENCH_INFO_DIR, f'{name}.json')) as f:
        return json.load(f)['benchmark']


def _is_polybench(info):
    return info['relative_path'].startswith('polybench/')


def _numpy_ref(info):
    mod_path = 'npbench_numpy_refs.' + info['relative_path'].replace('/', '.') + f".{info['module_name']}_numpy"
    return getattr(importlib.import_module(mod_path), info['func_name'])


# --------------------------------------------------------------------------
# Build (dace_program, arrays, params) at the paper preset, for either suite.
# --------------------------------------------------------------------------
def _poly_kernel(name):
    return next(k for k in pb.collect() if k.name == name)


def _np_kernel(name):
    return next(c for c in npb.collect() if c['name'] == name)


def kernel_exists(name):
    """Not every upstream bench_info entry made it into this dace repo's own
    npbench/polybench port (e.g. conv2d_bias) -- skip those cleanly instead
    of letting every lane fail independently with a StopIteration."""
    return any(k.name == name for k in pb.collect()) or any(c['name'] == name for c in npb.collect())


def _poly_data(kernel, params):
    mod = pb._module(kernel)
    program = pb._program(mod)
    arrays = []
    for shape, dtype in mod.args:
        concrete = [params[str(s)] if isinstance(s, dace.symbol) else s for s in shape]
        arrays.append(dace.ndarray(concrete, dtype))
    mod.init_array(*arrays, **{k.lower(): v for k, v in params.items()})
    call_arrays = {n: a for n, a in zip(program.argnames, arrays)}
    return program, call_arrays


def _np_data(c, params):
    args = [params[a] for a in c['input_args']]
    rets = c['initialize'](*args)
    rets = rets if isinstance(rets, tuple) else (rets, )
    arrays = dict(zip(c['array_args'], rets))
    return c['program'], arrays


def build_program_and_data(name, info, params):
    if _is_polybench(info):
        return _poly_data(_poly_kernel(name), params)
    return _np_data(_np_kernel(name), params)


def _collect_outputs(output_args, ret, kwargs):
    rets = ret if isinstance(ret, tuple) else ((ret, ) if ret is not None else ())
    out = {}
    for i, oname in enumerate(output_args):
        out[oname] = np.asarray(rets[i]) if i < len(rets) and rets[i] is not None else kwargs[oname]
    return out


def _dace_call_kwargs(info, arrays, params):
    kwargs = {n: (arrays[n].copy() if isinstance(arrays[n], np.ndarray) else arrays[n]) for n in info['input_args']}
    symbols = {k: v for k, v in params.items() if k not in kwargs}
    return {**kwargs, **symbols}


def _numpy_call_kwargs(info, arrays):
    return {n: (arrays[n].copy() if isinstance(arrays[n], np.ndarray) else arrays[n]) for n in info['input_args']}


def _compare(ref, got, rtol=1e-3, atol=1e-4):
    for oname, r in ref.items():
        g = got.get(oname)
        if g is None or not np.allclose(np.asarray(r), np.asarray(g), rtol=rtol, atol=atol, equal_nan=True):
            return False
    return True


# --------------------------------------------------------------------------
# Jobs run inside the isolated subprocess (recipe in, small result out) --
# a segfault or exception in canonicalize/auto_optimize/compile/run only
# takes down this one job, never the rank's remaining sweep.
# --------------------------------------------------------------------------
def _run_numpy(info, arrays):
    fn = _numpy_ref(info)
    kwargs = _numpy_call_kwargs(info, arrays)
    ret = fn(**kwargs)
    return _collect_outputs(info['output_args'], ret, kwargs)


def _run_dace(sdfg, info, arrays, params):
    kwargs = _dace_call_kwargs(info, arrays, params)
    ret = sdfg.compile()(**kwargs)
    return _collect_outputs(info['output_args'], ret, kwargs)


def _check_job(name, pipeline):
    engine.configure_dace_process()
    info = load_bench_info(name)
    params = info['parameters'][PRESET]
    program, arrays = build_program_and_data(name, info, params)
    ref = _run_numpy(info, arrays)
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = f"{sdfg.name}_{pipeline.replace('-', '_')}"
    sdfg = _dace_pipeline(pipeline, sdfg)
    try:
        got = _run_dace(sdfg, info, arrays, params)
    finally:
        engine.cleanup_build_folder(sdfg)
    return _compare(ref, got)


def _time_dace_job(name, pipeline, reps):
    engine.configure_dace_process()
    info = load_bench_info(name)
    params = info['parameters'][PRESET]
    program, arrays = build_program_and_data(name, info, params)
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = f"{sdfg.name}_{pipeline.replace('-', '_')}"
    sdfg = _dace_pipeline(pipeline, sdfg)
    try:
        call_kwargs = _dace_call_kwargs(info, arrays, params)
        return engine.time_sdfg(sdfg, call_kwargs, reps)
    finally:
        engine.cleanup_build_folder(sdfg)


def _time_numpy_job(name, reps, warmup=1):
    info = load_bench_info(name)
    params = info['parameters'][PRESET]
    _, arrays = build_program_and_data(name, info, params)
    fn = _numpy_ref(info)
    times = []
    for i in range(warmup + reps):
        kwargs = _numpy_call_kwargs(info, arrays)
        t0 = time.perf_counter()
        fn(**kwargs)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    return times


def _save_sdfg_job(name, pipeline):
    engine.configure_dace_process()
    info = load_bench_info(name)
    params = info['parameters'][PRESET]
    program, _ = build_program_and_data(name, info, params)
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = f"{sdfg.name}_{pipeline.replace('-', '_')}"
    sdfg = _dace_pipeline(pipeline, sdfg)
    sdfg.validate()
    return sdfg.to_json()


# --------------------------------------------------------------------------
# Per-kernel driver.
# --------------------------------------------------------------------------
def process_kernel(name, args, rank):
    kdir = engine.kernel_dir(args.results_dir, CORPUS, name, PRESET)
    for lane in ALL_LANES:
        status = None if args.force else engine.read_status(kdir, lane)
        if status is None:
            if lane == 'numpy':
                ok, correct = True, True  # numpy IS the correctness reference; trivially "correct"
            else:
                ok, correct = engine.run_isolated(_check_job, (name, lane), timeout=args.timeout)
            engine.write_status(kdir, lane, bool(ok and correct), '' if ok and correct else str(correct))
            if args.save_sdfg and ok and lane not in ('numpy', ):
                ok2, sdfg_json = engine.run_isolated(_save_sdfg_job, (name, lane), timeout=args.timeout)
                if ok2:
                    engine.save_sdfg(kdir, dace.SDFG.from_json(sdfg_json), lane)
            correct_now = bool(ok and correct)
        else:
            correct_now = status['correct'] == 'True'
        if not correct_now:
            continue

        have = 0 if args.force else engine.existing_reps(kdir, lane)
        remaining = args.reps - have
        if remaining <= 0:
            continue
        print(f'[{name}/{PRESET}] {lane}: measuring {remaining} more rep(s)')
        if lane == 'numpy':
            ok, payload = engine.run_isolated(_time_numpy_job, (name, remaining), timeout=args.timeout)
        else:
            ok, payload = engine.run_isolated(_time_dace_job, (name, lane, remaining), timeout=args.timeout)
        if ok:
            engine.append_results(kdir, lane, payload, have)
        else:
            engine.write_status(kdir, lane, False, str(payload))
    engine.write_run_meta(kdir, rank=rank, reps_target=args.reps, preset=PRESET)


def save_sdfg_only(name, args):
    kdir = engine.kernel_dir(args.results_dir, CORPUS, name, PRESET)
    if not args.force and all(os.path.exists(os.path.join(kdir, f'{p}.sdfg')) for p in ('canon', 'fast-canon')):
        print(f'[{name}/{PRESET}] SDFGs already saved, skip')
        return
    for pipeline in ('canon', 'fast-canon'):
        ok, payload = engine.run_isolated(_save_sdfg_job, (name, pipeline), timeout=args.timeout)
        if not ok:
            print(f'[{name}] {pipeline}: {payload}')
            continue
        engine.save_sdfg(kdir, dace.SDFG.from_json(payload), pipeline)


def kernel_list(args):
    names = sorted(f[:-len('.json')] for f in os.listdir(BENCH_INFO_DIR) if f.endswith('.json'))
    if args.only:
        names = [n for n in names if args.only in n]
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    engine.add_common_args(ap)
    args = ap.parse_args()

    if args.list_kernels:
        print('\n'.join(kernel_list(args)))
        return
    if args.tables_only:
        engine.write_tables(args.results_dir, CORPUS, list(ALL_LANES), BASELINE_LANE)
        return

    engine.export_cxx_override(args)
    cxx = engine.pick_cxx_compiler(args.cxx)  # fails fast on a bad --cxx, before any work starts
    print(f'C++ compiler (DaCe codegen): {cxx or "(none found; DaCe default)"}')

    rank, world = engine.get_world_rank(), engine.get_world_size()
    all_kernels = kernel_list(args)
    explicit = engine.explicit_kernel_list(args)
    mine = explicit if explicit is not None else engine.my_slice(all_kernels, rank, world)
    print(f'rank {rank}/{world}: {len(mine)}/{len(all_kernels)} kernels (preset={PRESET})')

    for name in mine:
        if not kernel_exists(name):
            print(f'[{name}] no matching kernel in the vendored corpus, skip')
            continue
        info = load_bench_info(name)
        if PRESET not in info.get('parameters', {}):
            print(f'[{name}] no {PRESET} preset data, skip')
            continue
        if args.save_sdfg_only:
            save_sdfg_only(name, args)
        else:
            process_kernel(name, args, rank)

    if not args.save_sdfg_only:
        engine.write_tables(args.results_dir, CORPUS, list(ALL_LANES), BASELINE_LANE)


if __name__ == '__main__':
    main()
