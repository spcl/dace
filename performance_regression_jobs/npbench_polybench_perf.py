#!/usr/bin/env python3
"""Performance regression: DaCe canonicalize and a light
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
    python3 npbench_polybench_perf.py --save-sdfg-only   # just dump canon SDFGs
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
import inspect
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

DACE_LANES = tuple(engine.PIPELINES)  # auto_opt, parallel, canon
#: numpy is a CPU-only extra reference lane (see process_kernel); not a baseline.
BASELINE_LANE = 'auto_opt'
#: Candidates the grid plot shows, each as a speedup vs. auto_opt per device.
#: auto_opt is included as its own bar (a constant 1.0 reference line) so every
#: run visibly reports dace-autoopt alongside dace-parallel and dace-canon.
CANDIDATES = ('auto_opt', 'parallel', 'canon')

#: GPU-only lanes (e.g. 'canon-gpu' = canonicalize -> apply_gpu_transformations ->
#: compile). They run solely on the gpu device; resolve_devices() already drops the
#: gpu device entirely on a CPU-only box, so these never build there.
GPU_ONLY_LANES = tuple(engine.GPU_ONLY_PIPELINES)


def lanes_for_device(device):
    """The standard pipelines on either device; plus a timed numpy run on cpu only,
    and the gpu-only lanes (canon-gpu) on gpu only."""
    if device == 'cpu':
        return DACE_LANES + ('numpy', )
    return DACE_LANES + GPU_ONLY_LANES


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


def _np_data(c, info, params):
    """Initialize the npbench kernel's inputs and split the returned values into
    call arrays and derived scalar params. ``initialize`` returns a mix of arrays
    and derived scalars the kernel/reference/SDFG need but that live in no dataset
    preset -- nbody's step count ``Nt = ceil(tEnd/dt)``, vadv's ``dtr_stage``,
    cavity/channel's grid spacings ``dx/dy/dt``. The dace corpus's own
    ``array_args`` names those returns positionally when the counts line up (it
    lists the inline scalars too, e.g. cavity's dx/dy/dt); when it drops a scalar
    it omitted (nbody's trailing ``Nt``, vadv's leading ``dtr_stage``) the counts
    differ, so fall back to bench_info's canonical ``init.output_args`` list. Each
    value is classed array-vs-scalar by its runtime type, never by position."""
    args = [params[a] for a in c['input_args']]
    rets = c['initialize'](*args)
    rets = rets if isinstance(rets, tuple) else (rets, )
    names = list(c['array_args']) if len(rets) == len(c['array_args']) else info['init']['output_args']
    arrays, scalars = {}, {}
    for nm, val in zip(names, rets):
        (arrays if isinstance(val, np.ndarray) else scalars)[nm] = val
    return c['program'], arrays, scalars


def build_program_and_data(name, info, params):
    """Return ``(program, arrays, params)`` where ``params`` is augmented with any
    derived scalars the initializer produced (npbench); polybench initializers
    mutate their arrays in place and add no scalars, so ``params`` is unchanged."""
    if _is_polybench(name, info):
        program, arrays = _poly_data(_poly_kernel(name), params)
        return program, arrays, params
    program, arrays, scalars = _np_data(_np_kernel(name), info, params)
    return program, arrays, {**params, **scalars}


def _collect_outputs(output_args, ret, kwargs):
    rets = ret if isinstance(ret, tuple) else ((ret, ) if ret is not None else ())
    out = {}
    for i, oname in enumerate(output_args):
        out[oname] = np.asarray(rets[i]) if i < len(rets) and rets[i] is not None else kwargs[oname]
    # Many "returning" kernels declare output_args=[] in bench_info (resnet, mlp,
    # nbody, k3mm, ...): without this, _compare({}, {}) is vacuously True and a
    # miscompile (all-zeros) would pass. Capture the returned value(s) so numpy's
    # oracle and DaCe's output land under the same key and _compare runs a real
    # allclose. (Kernels whose numpy ref instead mutates a workspace arg and
    # returns None still need output_args populated in bench_info -- a separate
    # pre-existing gap: for those this adds nothing and the check stays weak.)
    if not output_args:
        for i, r in enumerate(rets):
            if r is not None:
                out[f'__return_{i}'] = np.asarray(r)
    return out


def lookup_scalar(name, params):
    """Resolve one scalar/symbol value the SDFG or numpy reference asks for by
    name. Matches `params` (dataset preset + derived scalars) exactly first, then
    case-insensitively -- bench_info spells timestep/size scalars upper-case
    (TSTEPS, TMAX) while several DaCe kernels declare the symbol lower-case
    (tsteps). Falls back to the PolyBench `float_n` convention (a float-cast of the
    integer dimension `N`). Any name still unresolved (e.g. deriche's real
    coefficient `alpha`, which has no authoritative dataset value) raises KeyError
    rather than guessing."""
    if name in params:
        return params[name]
    lowered = name.lower()
    for k, v in params.items():
        if k.lower() == lowered:
            return v
    if name.startswith('float_') and name[len('float_'):].upper() in params:
        return float(params[name[len('float_'):].upper()])
    raise KeyError(f'no scalar/symbol value for {name!r}; have {sorted(params)}')


def _bind_array(name, arrays):
    v = arrays[name]
    return v.copy() if isinstance(v, np.ndarray) else v  # never mutate the shared input


def _dace_call_kwargs(sdfg, arrays, params):
    """Bind exactly what the compiled SDFG asks for. Its `arglist()` (data args +
    the free symbols it was parametrized over) is the ground truth for both the
    set of names AND their spelling: bench_info lists some scalars the SDFG derives
    itself and must NOT receive (stockham's `N = R**K`), and spells others in a
    different case than the kernel's symbol (`TSTEPS` vs `tsteps`). Arrays bind by
    exact name; scalars/symbols resolve from the param pool (case-insensitively).
    `__return*` entries are SDFG-allocated outputs, not inputs to pass."""
    required = set(sdfg.arglist()) | {str(s) for s in sdfg.free_symbols}
    out = {}
    for name in required:
        if name.startswith('__return'):
            continue
        if name in arrays:
            out[name] = _bind_array(name, arrays)
            continue
        try:
            out[name] = lookup_scalar(name, params)
        except KeyError:
            # A required symbol with no dataset value is one DaCe infers from an
            # array argument's shape at call time (lenet's C_before_fc1 =
            # fc1w.shape[0]); leave it unbound. A genuinely un-inferable one
            # (deriche's coefficient alpha) then surfaces at the compiled call.
            pass
    return out


def _numpy_call_kwargs(fn, arrays, params):
    """Bind what the numpy reference's own signature asks for (not bench_info's
    input_args, which for some kernels lists derived quantities the reference
    computes internally). Parameters with a default that we can't resolve are left
    to the reference's default (e.g. an optional `datatype`)."""
    out = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name in arrays:
            out[name] = _bind_array(name, arrays)
            continue
        try:
            out[name] = lookup_scalar(name, params)
        except KeyError:
            if param.default is inspect.Parameter.empty:
                raise
    return out


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
    kwargs = _numpy_call_kwargs(fn, arrays, params)
    ret = fn(**kwargs)
    return _collect_outputs(info['output_args'], ret, kwargs)


def _run_dace(sdfg, info, arrays, params):
    kwargs = _dace_call_kwargs(sdfg, arrays, params)
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
    program, arrays, params = build_program_and_data(name, info, params)
    ref = _run_numpy(info, arrays, params)  # numpy oracle is host-side on either device
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = _dace_name(sdfg.name, pipeline, device)
    sdfg = engine.ALL_PIPELINES[pipeline](sdfg, device)
    got = _run_dace(sdfg, info, arrays, params)  # host arrays; DaCe inserts device copies on gpu
    return _compare(ref, got)


def _time_dace_job(name, pipeline, device, reps, timeout):
    engine.configure_dace_process()
    info = load_bench_info(name)
    params = info['parameters'][PRESET]
    program, arrays, params = build_program_and_data(name, info, params)
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = _dace_name(sdfg.name, pipeline, device)
    sdfg = engine.ALL_PIPELINES[pipeline](sdfg, device)
    call_kwargs = _dace_call_kwargs(sdfg, arrays, params)
    # Budget the rep loop under the subprocess timeout so a slow paper-size kernel
    # (cholesky/lu/gemm) yields fewer reps instead of a hard-killed total loss.
    return engine.time_sdfg(sdfg, call_kwargs, reps, time_budget_s=0.8 * timeout)


def _time_numpy_job(name, reps, warmup=1, timeout=None):
    info = load_bench_info(name)
    params = info['parameters'][PRESET]
    _, arrays, params = build_program_and_data(name, info, params)
    fn = _numpy_ref(info)
    # Same wall-clock budget as the dace lanes: a slow numpy oracle at paper size
    # yields fewer reps rather than hard-timing-out the whole measurement.
    budget = 0.8 * timeout if timeout is not None else None
    times = []
    start = None
    for i in range(warmup + reps):
        kwargs = _numpy_call_kwargs(fn, arrays, params)
        t0 = time.perf_counter()
        fn(**kwargs)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
            if start is None:
                start = t0
            if budget is not None and time.perf_counter() - start >= budget:
                break
    return times


def _save_sdfg_job(name, pipeline, device):
    engine.configure_dace_process()
    info = load_bench_info(name)
    params = info['parameters'][PRESET]
    program, _, _ = build_program_and_data(name, info, params)
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = _dace_name(sdfg.name, pipeline, device)
    sdfg = engine.ALL_PIPELINES[pipeline](sdfg, device)
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
                ok, payload = engine.run_isolated(_time_numpy_job, (name, remaining, 1, args.timeout), timeout=args.timeout)
            else:
                ok, payload = engine.run_isolated(_time_dace_job, (name, lane, device, remaining, args.timeout),
                                                  timeout=args.timeout)
            if ok:
                engine.append_results(kdir, lane, payload, have)
            else:
                engine.write_status(kdir, lane, False, str(payload))
        engine.write_run_meta(kdir, rank=rank, reps_target=args.reps, preset=tag, device=device)


def save_sdfg_only(name, args, devices):
    for device in devices:
        kdir = engine.kernel_dir(args.results_dir, CORPUS, name, preset_tag(device))
        if not args.force and all(os.path.exists(os.path.join(kdir, f'{p}.sdfg')) for p in ('canon', )):
            print(f'[{name}/{preset_tag(device)}] SDFGs already saved, skip')
            continue
        for pipeline in ('canon', ):
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


#: All lanes that can appear in any device folder (cpu adds numpy; gpu adds the
#: gpu-only lanes like canon-gpu), for the tables. A lane with no results in a
#: given device folder simply shows blank there.
TABLE_LANES = list(DACE_LANES) + ['numpy'] + list(GPU_ONLY_LANES)


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
    # 1h/kernel: the largest paper builds (cholesky/lu/gemm at N=2000) exceed the
    # old 900s. The rep loop is also wall-clock-budgeted to ~0.8x this (time_sdfg),
    # so a slow kernel yields fewer reps well before the hard kill.
    engine.add_common_args(ap, default_timeout=3600.0)
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
