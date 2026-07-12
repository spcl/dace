#!/usr/bin/env python3
"""Performance regression: DaCe canonicalize/fast-canonicalize and a light
simplify+loop2map+mapfusion pipeline (parallel), each vs. DaCe's own
auto_optimize baseline (auto_opt), on BOTH cpu and gpu, over the combined
NPBench+PolyBench corpus. A timed numpy run is kept as an extra CPU reference.
Fully self-contained: the kernel definitions are copied from this dace repo's
own tests/corpus/{npbench,polybench} (npbench_corpus/, polybench_corpus/), and
the "paper"-preset dataset sizes + numpy reference implementations are
vendored from the real upstream npbench repo (bench_info/, npbench_numpy_refs/)
since this dace-repo port has no "paper" preset and no numpy oracle at all
for polybench kernels. Nothing here imports the external npbench package.

GPU degrades gracefully: with no CUDA toolchain/device the gpu device is
skipped entirely (a one-time crash-isolated probe), never crashing the sweep.

    python3 npbench_polybench_perf.py                    # this rank's slice, 100 reps, cpu+gpu
    python3 npbench_polybench_perf.py --only gemm --reps 3 --devices cpu
    python3 npbench_polybench_perf.py --save-sdfg-only   # just dump canon/fast-canon SDFGs
    python3 npbench_polybench_perf.py --tables-only      # rebuild correctness.md/speedup.md
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import importlib
import json
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
DEVICES = ('cpu', 'gpu')

DACE_LANES = tuple(engine.PIPELINES)  # auto_opt, parallel, canon, fast-canon
#: numpy is a CPU-only extra reference lane (see process_kernel); not a baseline.
BASELINE_LANE = 'auto_opt'
#: Candidates the grid plot shows, each as a speedup vs. auto_opt per device.
CANDIDATES = ('parallel', 'canon', 'fast-canon')


def lanes_for_device(device):
    """Every pipeline on either device, plus a timed numpy run on cpu only."""
    return DACE_LANES + (('numpy', ) if device == 'cpu' else ())


def preset_tag(device):
    """Per-device result folder token, e.g. 'paper-cpu' / 'paper-gpu'. Keeps the
    2-underscore result_tag invariant (device is a hyphen suffix, not a '_')."""
    return f'{PRESET}-{device}'


# --------------------------------------------------------------------------
# bench_info: which suite (poly vs np) an entry belongs to, and its paper-preset
# parameters. module_name is the join key against npbench_corpus/polybench_corpus
# (short_name differs for several kernels, e.g. go_fast vs npgofast -- verified).
# --------------------------------------------------------------------------
def load_bench_info(name):
    with open(os.path.join(BENCH_INFO_DIR, f'{name}.json')) as f:
        return json.load(f)['benchmark']


def _is_polybench(name, info):
    """Whether `name` should be built via the polybench or npbench-native
    local corpus. Prefers actual local corpus membership over the upstream
    bench_info's declared relative_path: a couple of entries (cholesky2,
    covariance2) are declared 'polybench/...' upstream but this repo's own
    port of them only exists in npbench_corpus -- trusting the declared
    path for those raises a StopIteration deep inside _poly_kernel/_np_kernel
    when it searches the wrong local corpus."""
    in_poly = any(k.name == name for k in pb.collect())
    in_np = any(c['name'] == name for c in npb.collect())
    if in_poly != in_np:
        return in_poly
    # ambiguous or absent from both; kernel_exists() already filters the latter.
    return info['relative_path'].startswith('polybench/')


def _numpy_ref(info):
    rel = info['relative_path']
    if rel.startswith('polybench/'):
        # polybench numpy oracles live in their own top-level polybench_numpy_refs/
        # dir (grouped with the polybench corpus), NOT nested under npbench_numpy_refs.
        mod_path = f"polybench_numpy_refs.{info['module_name']}.{info['module_name']}_numpy"
    else:
        mod_path = 'npbench_numpy_refs.' + rel.replace('/', '.') + f".{info['module_name']}_numpy"
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
    if _is_polybench(name, info):
        return _poly_data(_poly_kernel(name), params)
    return _np_data(_np_kernel(name), params)


def _collect_outputs(output_args, ret, kwargs):
    rets = ret if isinstance(ret, tuple) else ((ret, ) if ret is not None else ())
    out = {}
    for i, oname in enumerate(output_args):
        out[oname] = np.asarray(rets[i]) if i < len(rets) and rets[i] is not None else kwargs[oname]
    return out


def _resolve_args(names, arrays, params):
    """Each name in `names` (info['input_args']) is an actual array (present
    in `arrays`, copied so repeated calls never mutate the shared input), a
    scalar/symbol value that only lives in `params` (e.g. a grid size like
    'nx' or a loop bound like 'TMAX'), or -- a handful of PolyBench kernels'
    upstream convention -- a float-cast dimension like 'float_n' that's
    never itself in bench_info, only its integer counterpart 'N' is. Any
    other name not covered by these three (e.g. a real algorithm coefficient
    like deriche's 'alpha') is intentionally left to raise KeyError rather
    than guess a numeric value with no authoritative source."""
    out = {}
    for n in names:
        if n in arrays:
            v = arrays[n]
            out[n] = v.copy() if isinstance(v, np.ndarray) else v
        elif n in params:
            out[n] = params[n]
        elif n.startswith('float_') and n[len('float_'):].upper() in params:
            out[n] = float(params[n[len('float_'):].upper()])
        else:
            out[n] = params[n]
    return out


def _dace_call_kwargs(info, arrays, params):
    # The compiled SDFG's call signature needs every array `arrays` actually
    # holds -- _poly_data/_np_data key `arrays` by the program's real
    # argnames, so its keys ARE the ground truth -- plus whatever
    # scalar/symbol names info['input_args'] lists that aren't arrays.
    # info['output_args'] isn't a reliable source for the array names: some
    # kernels (e.g. correlation) declare it empty even though the DaCe
    # program needs pre-allocated workspace arrays ('corr', 'mean',
    # 'stddev') that the numpy reference allocates/returns internally
    # instead of taking as parameters.
    names = set(info['input_args']) | set(arrays)
    kwargs = _resolve_args(names, arrays, params)
    symbols = {k: v for k, v in params.items() if k not in kwargs}
    return {**kwargs, **symbols}


def _numpy_call_kwargs(info, arrays, params):
    return _resolve_args(info['input_args'], arrays, params)


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
def _run_numpy(info, arrays, params):
    fn = _numpy_ref(info)
    kwargs = _numpy_call_kwargs(info, arrays, params)
    ret = fn(**kwargs)
    return _collect_outputs(info['output_args'], ret, kwargs)


def _run_dace(sdfg, info, arrays, params):
    kwargs = _dace_call_kwargs(info, arrays, params)
    ret = sdfg.compile()(**kwargs)
    return _collect_outputs(info['output_args'], ret, kwargs)


def _dace_name(sdfg_name, pipeline, device):
    # name (unique per corpus+kernel+pipeline+device) is also its cache key (dace
    # cache='name' in engine.configure_dace_process): the exact same variant,
    # however many times it's independently rebuilt (a check, the timing run
    # right after, a resumed invocation), always lands on the same compiled
    # binary instead of recompiling. The device suffix keeps the cpu and gpu
    # builds of one kernel/pipeline from colliding on one cache folder.
    return f"{CORPUS}_{sdfg_name}_{pipeline.replace('-', '_')}_{device}"


def _check_job(name, pipeline, device):
    engine.configure_dace_process()
    info = load_bench_info(name)
    params = info['parameters'][PRESET]
    program, arrays = build_program_and_data(name, info, params)
    ref = _run_numpy(info, arrays, params)  # numpy oracle is host-side on either device
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = _dace_name(sdfg.name, pipeline, device)
    sdfg = engine.PIPELINES[pipeline](sdfg, device)
    got = _run_dace(sdfg, info, arrays, params)  # host arrays; DaCe inserts device copies on gpu
    return _compare(ref, got)


def _time_dace_job(name, pipeline, device, reps):
    engine.configure_dace_process()
    info = load_bench_info(name)
    params = info['parameters'][PRESET]
    program, arrays = build_program_and_data(name, info, params)
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = _dace_name(sdfg.name, pipeline, device)
    sdfg = engine.PIPELINES[pipeline](sdfg, device)
    call_kwargs = _dace_call_kwargs(info, arrays, params)
    return engine.time_sdfg(sdfg, call_kwargs, reps)


def _time_numpy_job(name, reps, warmup=1):
    info = load_bench_info(name)
    params = info['parameters'][PRESET]
    _, arrays = build_program_and_data(name, info, params)
    fn = _numpy_ref(info)
    times = []
    for i in range(warmup + reps):
        kwargs = _numpy_call_kwargs(info, arrays, params)
        t0 = time.perf_counter()
        fn(**kwargs)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    return times


def _save_sdfg_job(name, pipeline, device):
    engine.configure_dace_process()
    info = load_bench_info(name)
    params = info['parameters'][PRESET]
    program, _ = build_program_and_data(name, info, params)
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = _dace_name(sdfg.name, pipeline, device)
    sdfg = engine.PIPELINES[pipeline](sdfg, device)
    sdfg.validate()
    return sdfg.to_json()


# --------------------------------------------------------------------------
# Per-kernel driver.
# --------------------------------------------------------------------------
def process_kernel(name, args, rank, devices):
    for device in devices:
        tag = preset_tag(device)
        kdir = engine.kernel_dir(args.results_dir, CORPUS, name, tag)
        for lane in lanes_for_device(device):
            status = None if args.force else engine.read_status(kdir, lane)
            if status is None:
                if lane == 'numpy':
                    ok, correct = True, True  # numpy IS the correctness reference; trivially "correct"
                else:
                    ok, correct = engine.run_isolated(_check_job, (name, lane, device), timeout=args.timeout)
                engine.write_status(kdir, lane, bool(ok and correct), '' if ok and correct else str(correct))
                if args.save_sdfg and ok and lane != 'numpy':
                    ok2, sdfg_json = engine.run_isolated(_save_sdfg_job, (name, lane, device), timeout=args.timeout)
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
            print(f'[{name}/{tag}] {lane}: measuring {remaining} more rep(s)')
            if lane == 'numpy':
                ok, payload = engine.run_isolated(_time_numpy_job, (name, remaining), timeout=args.timeout)
            else:
                ok, payload = engine.run_isolated(_time_dace_job, (name, lane, device, remaining), timeout=args.timeout)
            if ok:
                engine.append_results(kdir, lane, payload, have)
            else:
                engine.write_status(kdir, lane, False, str(payload))
        engine.write_run_meta(kdir, rank=rank, reps_target=args.reps, preset=tag, device=device)


def save_sdfg_only(name, args, devices):
    for device in devices:
        kdir = engine.kernel_dir(args.results_dir, CORPUS, name, preset_tag(device))
        if not args.force and all(os.path.exists(os.path.join(kdir, f'{p}.sdfg')) for p in ('canon', 'fast-canon')):
            print(f'[{name}/{preset_tag(device)}] SDFGs already saved, skip')
            continue
        for pipeline in ('canon', 'fast-canon'):
            ok, payload = engine.run_isolated(_save_sdfg_job, (name, pipeline, device), timeout=args.timeout)
            if not ok:
                print(f'[{name}/{device}] {pipeline}: {payload}')
                continue
            engine.save_sdfg(kdir, dace.SDFG.from_json(payload), pipeline)


def kernel_list(args):
    names = sorted(f[:-len('.json')] for f in os.listdir(BENCH_INFO_DIR) if f.endswith('.json'))
    if args.only:
        names = [n for n in names if args.only in n]
    return names


#: All lanes that can appear in any device folder (cpu adds numpy), for the tables.
TABLE_LANES = list(DACE_LANES) + ['numpy']


def resolve_devices(requested):
    """Filter the requested devices down to what this machine supports: gpu is
    kept only if the one-time crash-isolated probe succeeds (no CUDA -> skipped,
    with a printed notice, never a crash)."""
    out = []
    for d in requested:
        if d == 'gpu' and not engine.gpu_supported():
            print('gpu: no CUDA toolchain/device detected (probe failed); skipping gpu device')
            continue
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    engine.add_common_args(ap, default_timeout=900.0)  # 15 min/kernel; bump --timeout for the largest paper builds
    ap.add_argument('--devices', default=','.join(DEVICES),
                    help=f'comma-separated devices to sweep (default: {",".join(DEVICES)}); gpu auto-skips if no CUDA')
    args = ap.parse_args()

    if args.list_kernels:
        print('\n'.join(kernel_list(args)))
        return
    if args.tables_only:
        engine.write_tables(args.results_dir, CORPUS, TABLE_LANES, BASELINE_LANE)
        engine.write_summary_csv(args.results_dir, CORPUS, BASELINE_LANE)
        return

    engine.export_cxx_override(args)
    cxx = engine.pick_cxx_compiler(args.cxx)  # fails fast on a bad --cxx, before any work starts
    print(f'C++ compiler (DaCe codegen): {cxx or "(none found; DaCe default)"}')

    requested = [d.strip() for d in args.devices.split(',') if d.strip()]
    devices = requested if args.save_sdfg_only else resolve_devices(requested)
    print(f'devices: {devices or "(none available)"}')

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
            save_sdfg_only(name, args, devices)
        else:
            process_kernel(name, args, rank, devices)

    if not args.save_sdfg_only:
        engine.write_tables(args.results_dir, CORPUS, TABLE_LANES, BASELINE_LANE)
        engine.write_summary_csv(args.results_dir, CORPUS, BASELINE_LANE)


if __name__ == '__main__':
    main()
