"""Uniform per-corpus adapter layer consumed by run_perf.py.

Every corpus (npbench, polybench, tsvc2, tsvc2_5) exposes the SAME set of
picklable, top-level job functions -- so run_perf.py never special-cases a
corpus. The heavy corpus machinery (kernel discovery, byte-target sizing,
deterministic input generation, native-lib compilation, numpy oracles) is
NOT reimplemented here: it is imported wholesale from the existing corpus
modules (tsvc2_perf, tsvc2_5_perf, npbench_polybench_perf) and only wired into
the unified lane model.

The dace lanes are the four engine pipelines:
    dace-canon        engine.pipeline_canon
    dace-parallel     engine.pipeline_parallel
    dace-canon-vec    engine.pipeline_canon_vectorize
    dace-parallel-vec engine.pipeline_parallel_vectorize
Correctness ground truth for EVERY lane (dace, native, numpy) is an
unvectorized ``engine.pipeline_parallel`` build of the same kernel.

Job functions run inside engine.run_isolated's spawned subprocess: they take a
small picklable recipe (never live arrays) and rebuild everything, so a crash
in transform/compile/run takes down only that one job.
"""
import importlib
import os
import re

import numpy as np

import dace
import engine
import native_harness as nh
import tsvc2_perf as t2
import tsvc2_5_perf as t25
import npbench_polybench_perf as npp

#: experiment-facing dace lane -> engine pipeline callable (all take (sdfg, device)).
DACE_PIPELINE = {
    'dace-autoopt': engine.pipeline_auto_opt,
    'dace-canon': engine.pipeline_canon,
    'dace-parallel': engine.pipeline_parallel,
    'dace-canon-vec': engine.pipeline_canon_vectorize,
    'dace-parallel-vec': engine.pipeline_parallel_vectorize,
}

#: name_tag used to build the correctness ground-truth SDFG (unvectorized parallel).
_REF_TAG = 'ref_parallel'


def _safe(s):
    """Identifier-safe token for an SDFG name (its dace cache='name' key)."""
    return re.sub(r'[^0-9A-Za-z_]', '_', str(s))


# ==========================================================================
# tsvc2
# ==========================================================================
def tsvc2_kernel_list():
    return sorted(k.name for k in t2.tsvc.collect())


def tsvc2_recipe(kernel_name):
    kernel = t2.tsvc.collect(name=kernel_name)[0]
    l1, l2 = t2.size_for_kernel(kernel)
    return {'l1': int(l1), 'l2': int(l2)}


def tsvc2_prepare(kernel_name, recipe, name_tag, pipeline_fn, device='cpu'):
    """Returns (sdfg, call_kwargs, collect): a built+pipelined SDFG (on `device`,
    'cpu' or 'gpu'), its call kwargs, and collect(call_kwargs, ret) -> comparison
    dict. On 'gpu' the pipeline's auto_optimize inserts the H2D/D2H copies, so the
    host numpy call_kwargs are unchanged."""
    l1, l2 = recipe['l1'], recipe['l2']
    kernel = t2.tsvc.collect(name=kernel_name)[0]
    sdfg = t2.tsvc.to_sdfg(kernel, f'{name_tag}', simplify=False)  # names it <kernel>_<name_tag>
    sdfg = pipeline_fn(sdfg, device)
    _, arrays, sym, sparams = t2._inputs(kernel_name, l1, l2)
    call_kwargs = {**{n: a.copy() for n, a in arrays.items()}, **sparams, **sym}
    return sdfg, call_kwargs, lambda ck, ret: ck


def tsvc2_run_native(kernel_name, recipe, so_path, c_name, sig):
    l1, l2 = recipe['l1'], recipe['l2']
    _, arrays, sym, sparams = t2._inputs(kernel_name, l1, l2)
    lib = nh.load_library(so_path)
    nh.call_kernel(lib, c_name, sig, arrays=arrays, len_1d=l1, len_2d=l2, scalar_params=sparams, symbols=sym)
    return arrays


def tsvc2_time_native(kernel_name, recipe, so_path, c_name, sig, reps, warmup=1):
    l1, l2 = recipe['l1'], recipe['l2']
    _, arrays, sym, sparams = t2._inputs(kernel_name, l1, l2)
    lib = nh.load_library(so_path)
    times = []
    for i in range(warmup + reps):
        ns = nh.call_kernel(lib, c_name, sig, arrays=arrays, len_1d=l1, len_2d=l2, scalar_params=sparams, symbols=sym)
        if i >= warmup:
            times.append(ns / 1e6)
    return times


def tsvc2_native_info():
    """(cpp_path, sigs) or (None, None) if this corpus has no native C source."""
    return t2.CPP_FILE, nh.parse_signatures(t2.CPP_FILE)


def tsvc2_native_call(kernel_name, sigs):
    base = t2.cpp_base_name(kernel_name)
    return base + '_run_timed', sigs.get(base, [])


# ==========================================================================
# tsvc2_5
# ==========================================================================
def tsvc2_5_kernel_list():
    return sorted(p.f.__name__ for p in t25.tsvc25.collect())


def tsvc2_5_recipe(kernel_name):
    program = t25._program(kernel_name)
    return {'sizes': t25.size_scale_for_kernel(program)}


def tsvc2_5_prepare(kernel_name, recipe, name_tag, pipeline_fn, device='cpu'):
    sizes = recipe['sizes']
    program = t25._program(kernel_name)
    sdfg = program.to_sdfg(simplify=False)
    sdfg.name = f'{_safe(name_tag)}_{_safe(kernel_name)}'
    sdfg = pipeline_fn(sdfg, device)
    _, arrays, scalars = t25._inputs(kernel_name, sizes)
    sym = t25._symbol_values(sdfg, sizes)
    call_kwargs = {**{n: a.copy() for n, a in arrays.items()}, **scalars, **sym}
    return sdfg, call_kwargs, lambda ck, ret: ck


def tsvc2_5_run_native(kernel_name, recipe, so_path, c_name, sig):
    sizes = recipe['sizes']
    _, arrays, scalars = t25._inputs(kernel_name, sizes)
    lib = nh.load_library(so_path)
    nh.call_kernel(lib, c_name, sig, arrays=arrays, len_1d=sizes.get('LEN_1D', 0), len_2d=sizes.get('LEN_2D', 0),
                   scalar_params=scalars, symbols=sizes)
    return arrays


def tsvc2_5_time_native(kernel_name, recipe, so_path, c_name, sig, reps, warmup=1):
    sizes = recipe['sizes']
    _, arrays, scalars = t25._inputs(kernel_name, sizes)
    lib = nh.load_library(so_path)
    times = []
    for i in range(warmup + reps):
        ns = nh.call_kernel(lib, c_name, sig, arrays=arrays, len_1d=sizes.get('LEN_1D', 0),
                            len_2d=sizes.get('LEN_2D', 0), scalar_params=scalars, symbols=sizes)
        if i >= warmup:
            times.append(ns / 1e6)
    return times


def tsvc2_5_native_info():
    return t25.CPP_FILE, nh.parse_signatures(t25.CPP_FILE)


def tsvc2_5_native_call(kernel_name, sigs):
    return kernel_name + '_run_timed', sigs.get(kernel_name, [])


# ==========================================================================
# npbench / polybench (shared machinery; split by _is_polybench)
# ==========================================================================
def _np_kernel_list(want_poly):
    names = []
    for f in sorted(f for f in os.listdir(npp.BENCH_INFO_DIR) if f.endswith('.json')):
        name = f[:-len('.json')]
        if not npp.kernel_exists(name):
            continue
        info = npp.load_bench_info(name)
        if npp.PRESET not in info.get('parameters', {}):
            continue
        if bool(npp._is_polybench(name, info)) == bool(want_poly):
            names.append(name)
    return sorted(names)


def npbench_kernel_list():
    return _np_kernel_list(want_poly=False)


def polybench_kernel_list():
    return _np_kernel_list(want_poly=True)


def np_recipe(kernel_name):
    return {}  # everything is rebuilt from bench_info by name inside the subprocess


def numpy_ref(info):
    """The numpy oracle for a bench_info entry. polybench oracles are vendored in
    their own top-level ``polybench_numpy_refs/`` dir (not under npbench_numpy_refs);
    npbench ones live under ``npbench_numpy_refs/<relative_path>/``. Tries the
    relative-path form, then both FLAT ``<pkg>.<module>.<module>_numpy`` forms.
    Raises ImportError/AttributeError only when NONE resolves."""
    module = info['module_name']
    candidates = [
        'npbench_numpy_refs.' + info['relative_path'].replace('/', '.') + f'.{module}_numpy',
        f'polybench_numpy_refs.{module}.{module}_numpy',
        f'npbench_numpy_refs.{module}.{module}_numpy',
    ]
    last = None
    for mp in candidates:
        try:
            return getattr(importlib.import_module(mp), info['func_name'])
        except Exception as e:  # ImportError or missing func_name
            last = e
    raise last


def numpy_ref_available(kernel_name):
    """Parent-side probe: is a numpy oracle importable for this kernel?"""
    try:
        numpy_ref(npp.load_bench_info(kernel_name))
        return True
    except Exception:
        return False


def np_prepare(kernel_name, recipe, name_tag, pipeline_fn, device='cpu'):
    info = npp.load_bench_info(kernel_name)
    params = info['parameters'][npp.PRESET]
    # build_program_and_data returns the params dict AUGMENTED with any derived
    # scalars the npbench initializer produced (a new dict, not mutated in place);
    # the call-kwargs below need that augmented copy, not the raw preset params.
    program, arrays, params = npp.build_program_and_data(kernel_name, info, params)
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = f'{_safe(name_tag)}_{_safe(kernel_name)}'
    sdfg = pipeline_fn(sdfg, device)
    call_kwargs = npp._dace_call_kwargs(sdfg, arrays, params)
    return sdfg, call_kwargs, lambda ck, ret: npp._collect_outputs(info['output_args'], ret, ck)


def np_run_numpy(kernel_name):
    info = npp.load_bench_info(kernel_name)
    params = info['parameters'][npp.PRESET]
    _, arrays, params = npp.build_program_and_data(kernel_name, info, params)
    fn = numpy_ref(info)
    kwargs = npp._numpy_call_kwargs(fn, arrays, params)
    ret = fn(**kwargs)
    return npp._collect_outputs(info['output_args'], ret, kwargs)


def np_time_numpy(kernel_name, reps, warmup=1):
    import time
    info = npp.load_bench_info(kernel_name)
    params = info['parameters'][npp.PRESET]
    _, arrays, params = npp.build_program_and_data(kernel_name, info, params)
    fn = numpy_ref(info)
    times = []
    for i in range(warmup + reps):
        kwargs = npp._numpy_call_kwargs(fn, arrays, params)
        t0 = time.perf_counter()
        fn(**kwargs)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    return times


def np_native_info():
    return None, None  # npbench/polybench have NO native C source -> native lanes unavailable


# ==========================================================================
# registry: corpus -> primitive functions (all top-level -> picklable)
# ==========================================================================
def _arrays_close(ref, got):
    return engine.arrays_close(ref, got)


def _np_compare(ref, got):
    return npp._compare(ref, got)  # rtol=1e-3, atol=1e-4


ADAPTERS = {
    'tsvc2': dict(kernel_list=tsvc2_kernel_list, recipe=tsvc2_recipe, prepare=tsvc2_prepare,
                  run_native=tsvc2_run_native, time_native=tsvc2_time_native, native_info=tsvc2_native_info,
                  native_call=tsvc2_native_call, compare=_arrays_close, has_numpy=False),
    'tsvc2_5': dict(kernel_list=tsvc2_5_kernel_list, recipe=tsvc2_5_recipe, prepare=tsvc2_5_prepare,
                    run_native=tsvc2_5_run_native, time_native=tsvc2_5_time_native, native_info=tsvc2_5_native_info,
                    native_call=tsvc2_5_native_call, compare=_arrays_close, has_numpy=False),
    'npbench': dict(kernel_list=npbench_kernel_list, recipe=np_recipe, prepare=np_prepare, run_native=None,
                    time_native=None, native_info=np_native_info, native_call=None, compare=_np_compare,
                    has_numpy=True),
    'polybench': dict(kernel_list=polybench_kernel_list, recipe=np_recipe, prepare=np_prepare, run_native=None,
                      time_native=None, native_info=np_native_info, native_call=None, compare=_np_compare,
                      has_numpy=True),
}


def adapter(corpus):
    return ADAPTERS[corpus]


# ==========================================================================
# Generic job functions (run in the isolated subprocess). Dispatch by corpus.
# ==========================================================================
def _run_dace_variant(corpus, kernel_name, recipe, name_tag, pipeline_fn, device='cpu'):
    a = ADAPTERS[corpus]
    sdfg, call_kwargs, collect = a['prepare'](kernel_name, recipe, name_tag, pipeline_fn, device)
    call_kwargs = engine.to_device_args(sdfg, call_kwargs, device)  # np -> cupy per-arg (GPU-storage only)
    ret = sdfg.compile()(**call_kwargs)
    call_kwargs, ret = engine.args_to_host(call_kwargs, ret, device)  # cupy -> np for the comparison
    return collect(call_kwargs, ret)


def check_dace_job(corpus, kernel_name, recipe, dace_lane, device='cpu'):
    engine.configure_dace_process()
    # Ground-truth reference is ALWAYS the CPU parallel build (the trusted oracle); the candidate runs
    # on the target device, so a GPU candidate is validated against the CPU result.
    ref = _run_dace_variant(corpus, kernel_name, recipe, _REF_TAG, engine.pipeline_parallel, 'cpu')
    cand = _run_dace_variant(corpus, kernel_name, recipe, dace_lane, DACE_PIPELINE[dace_lane], device)
    return ADAPTERS[corpus]['compare'](ref, cand)


def time_dace_job(corpus, kernel_name, recipe, dace_lane, reps, device='cpu'):
    engine.configure_dace_process()
    a = ADAPTERS[corpus]
    sdfg, call_kwargs, _ = a['prepare'](kernel_name, recipe, dace_lane, DACE_PIPELINE[dace_lane], device)
    call_kwargs = engine.to_device_args(sdfg, call_kwargs, device)  # resident device buffers (GPU-storage args)
    return engine.time_sdfg(sdfg, call_kwargs, reps)


def check_native_job(corpus, kernel_name, recipe, so_path, c_name, sig):
    engine.configure_dace_process()  # the reference SDFG needs the per-process dace config
    a = ADAPTERS[corpus]
    ref = _run_dace_variant(corpus, kernel_name, recipe, _REF_TAG, engine.pipeline_parallel)
    got = a['run_native'](kernel_name, recipe, so_path, c_name, sig)
    return a['compare'](ref, got)


def time_native_job(corpus, kernel_name, recipe, so_path, c_name, sig, reps):
    return ADAPTERS[corpus]['time_native'](kernel_name, recipe, so_path, c_name, sig, reps)


def check_numpy_job(corpus, kernel_name, recipe):
    engine.configure_dace_process()
    ref = _run_dace_variant(corpus, kernel_name, recipe, _REF_TAG, engine.pipeline_parallel)
    got = np_run_numpy(kernel_name)
    return ADAPTERS[corpus]['compare'](ref, got)


def time_numpy_job(corpus, kernel_name, recipe, reps):
    return np_time_numpy(kernel_name, reps)
