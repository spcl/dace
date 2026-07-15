#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Performance + code-quality comparison of the experimental *readable* CPU code
generator against the *legacy* one, over the full corpus (npbench, polybench,
tsvc, tsvc_2_5) and, optionally, CloudSC.

The toggle under test is the ``compiler.cpu.implementation`` config knob
(``legacy`` vs ``experimental``). "pre vs post optimization" == these two
generators: legacy is the classic connector-based codegen (pre), experimental is
the readable form (post) that emits per-array ``_idx`` index functions,
connector-free tasklets, ``const``/``constexpr`` write-once initialization, and
which additionally runs clang-format + clang-tidy + duplicate-``#include``
collapse for readability. The transformation pipeline is identical for both
lanes -- ``simplify + LoopToMap + MapFusion`` (reused verbatim as
``run_full_corpus_sweep.pipeline``); only the codegen differs.

Per (kernel, mode, codegen) it measures:

  * ``codegen_ms``  -- ALL work before the C++ compiler runs: the experimental
                       preprocessing passes + code emission (both timed inside
                       ``sdfg.generate_code()``) PLUS the clang-tidy readability
                       pass. The readable path does more here, so it is EXPECTED to
                       be higher; this is the key codegen-cost metric.
  * ``tidy_ms``     -- the clang-tidy portion of ``codegen_ms`` (0 for legacy),
                       broken out so the readable-codegen cost splits cleanly.
  * ``compile_ms``  -- the PURE C++/CMake build (``sdfg.compile()`` with clang-tidy
                       forced off so its cost does not leak in; clang-format, part
                       of the readable emission-to-disk, stays).
  * ``loc``         -- line count of the generated CPU frame ``.cpp``.
  * ``code_bytes``  -- byte size of that file (LoC + bytes ARE the readability
                       metric).
  * ``runtime_ms``  -- median kernel runtime over ``--reps`` reps (one warm-up).
  * ``correct``     -- experimental output bit-exact vs legacy, on IDENTICAL
                       inputs (generated ONCE per kernel and deep-copied per run,
                       so np.empty scratch outputs cannot diverge from garbage).

Modes:

  * ``single_core`` -- every map forced to ``ScheduleType.Sequential`` before
                       codegen, run with ``OMP_NUM_THREADS=1``.
  * ``multi_core``  -- default (CPU_Multicore) schedules, full node threads.

Every measurement runs in a forked child (repo rule: always fork when running a
compiled kernel -- a segfault in an experimental build fails only that one row,
never the sweep). A per-kernel failure is caught and recorded as an ERROR row.

The corpus is distributed across ranks on a single node: rank ``r`` of ``N``
processes ``cases[r::N]`` and writes ``<out>.rank<r>``; ``--merge`` concatenates
the per-rank files into ``<out>`` and prints the summary.

    python3 run_readable_compare.py --mode both --codegen both --corpus all --out compare.tsv
    python3 run_readable_compare.py --mode single_core --corpus polybench --only atax --out probe.tsv
    python3 run_readable_compare.py --cloudsc --out cloudsc.tsv
    python3 run_readable_compare.py --merge --out compare.tsv --nranks 8
"""
import os

# MPI/OpenMP anti-hang defaults BEFORE importing dace (a dace script otherwise
# blocks on MPI_Init and the sweep looks hung). setdefault so an explicit launch
# value or the per-mode thread override in main() wins.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import copy
import csv
import glob
import pickle
import shutil
import statistics
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Kernel enumeration + build helpers are reused, not duplicated. build_case below
# mirrors run_full_corpus_sweep.run_impl's per-corpus SDFG construction because
# run_impl compiles + runs a kernel in one shot and never exposes the SDFG object,
# whereas this job must time generate_code() and compile() separately -- so the
# construction is factored out here while all_cases/make_base/pipeline are reused.
from run_full_corpus_sweep import all_cases, make_base, pipeline
# waitpid_with_timeout is the repo's tested fork-timeout + SIGKILL helper.
from tests.codegen.readable.conftest import waitpid_with_timeout
from tests.corpus.npbench import npbench
from tests.corpus.polybench import polybench
from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc_2_5 import tsvc_2_5

import dace
from dace.sdfg import nodes

#: CLI --corpus value -> the corpus tag used by run_full_corpus_sweep.all_cases.
CORPUS_MAP = {'npbench': 'np', 'polybench': 'poly', 'tsvc': 'tsvc', 'tsvc_2_5': 'tsvc25'}
CODEGENS = ('legacy', 'experimental')
MODES = ('single_core', 'multi_core')

#: TSV schema. ``codegen_ms`` is the full pre-C++-compiler cost (experimental
#: passes + emission + clang-tidy); ``tidy_ms`` breaks out the clang-tidy portion
#: of it; ``compile_ms`` is the pure C++/CMake build (clang-tidy forced off).
TSV_FIELDS = [
    'kernel', 'corpus', 'mode', 'codegen', 'threads', 'codegen_ms', 'compile_ms', 'tidy_ms', 'runtime_ms', 'loc',
    'code_bytes', 'tidy', 'correct', 'error'
]
#: Reserved keys carrying scalar metrics through the isolated child's payload.
METRIC_KEYS = ('codegen_ms', 'compile_ms', 'tidy_ms', 'runtime_ms', 'loc', 'code_bytes', 'tidy')


# --------------------------------------------------------------------------
# Codegen selection + SDFG construction / measurement (runs inside the child).
# --------------------------------------------------------------------------
def set_implementation(codegen):
    """Select the CPU code generator for THIS process (also mirrored to the env for
    any nested spawn). The experimental generator now auto-runs clang-format,
    duplicate-``#include`` collapse and clang-tidy with no config flag; clang-tidy is
    suppressed for the timed compile via :func:`compile_without_tidy` instead."""
    dace.Config.set('compiler', 'cpu', 'implementation', value=codegen)
    os.environ['DACE_compiler_cpu_implementation'] = codegen


def compile_without_tidy(sdfg):
    """``sdfg.compile()`` with the readable clang-tidy pass suppressed, returning
    ``(compiled, compile_ms)`` for the PURE C++/CMake build.

    The experimental generator auto-invokes ``apply_clang_tidy`` inside
    ``generate_program_folder`` (there is no longer a config flag to gate it), which
    would leak clang-tidy's cost into ``compile_ms``. Monkeypatch it to a no-op for
    the duration so tidy can be timed separately (see :func:`tidy_generated_cpp`) and
    folded into ``codegen_ms``. The duplicate-include collapse and clang-format stay,
    as they are part of the readable emission-to-disk.
    """
    import dace.codegen.compiler as compiler_module
    real_tidy = compiler_module.apply_clang_tidy
    compiler_module.apply_clang_tidy = lambda code_path: None
    try:
        t0 = time.perf_counter()
        compiled = sdfg.compile()
        return compiled, (time.perf_counter() - t0) * 1000.0
    finally:
        compiler_module.apply_clang_tidy = real_tidy


def tidy_generated_cpp(sdfg):
    """clang-tidy every generated ``.cpp`` / ``.cu`` for ``sdfg`` in place; return
    ``(elapsed_ms, ran)``.

    This is the readable generator's clang-tidy readability pass, timed on its own.
    ``apply_clang_tidy`` is standalone (no CMake / compilation database), so it is
    cheap and safe to run after the build -- the ``.so`` is already compiled, so
    tidying the source is purely cosmetic and does not affect ``runtime_ms``. It
    DOES change loc/code_bytes, so loc/bytes are read AFTER this runs (final form).
    """
    import dace.codegen.compiler as compiler_module
    if not shutil.which('clang-tidy'):
        return 0.0, False
    src = os.path.join(sdfg.build_folder, 'src')
    files = (glob.glob(os.path.join(src, '**', '*.cpp'), recursive=True) +
             glob.glob(os.path.join(src, '**', '*.cu'), recursive=True))
    if not files:
        return 0.0, False
    t0 = time.perf_counter()
    for path in files:
        compiler_module.apply_clang_tidy(path)
    return (time.perf_counter() - t0) * 1000.0, True


def force_sequential(sdfg):
    """Force every map and library node to a sequential schedule, in place -- the
    single_core mode (deterministic, no OpenMP)."""
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            node.map.schedule = dace.ScheduleType.Sequential
        elif isinstance(node, nodes.LibraryNode) and hasattr(node, 'schedule'):
            node.schedule = dace.ScheduleType.Sequential


def build_case(corpus, name, base, tag, single_core):
    """Parse + pipeline the kernel and return ``(sdfg, arrays, call)``.

    Mirrors run_full_corpus_sweep.run_impl's per-corpus construction, but returns
    the SDFG (uniquely named by ``tag``) and its call kwargs instead of running it,
    so codegen/compile can be timed separately. ``arrays`` is a private deep copy
    of the shared inputs (so both lanes see identical data); for single_core every
    map is forced sequential before codegen.
    """
    arrays = copy.deepcopy(base['arrays'])
    extra = base['extra']
    if corpus == 'poly':
        sdfg = polybench.fresh_sdfg(base['kernel'])
        pipeline(sdfg, 'cpu')
        sdfg.name = f'{sdfg.name}_{tag}'
        call = {**arrays, **extra}
    elif corpus == 'np':
        sdfg = npbench.fresh_sdfg(base['descriptor'])
        pipeline(sdfg, 'cpu')
        sdfg.name = f'{sdfg.name}_{tag}'
        params = extra
        call = npbench._map_call(base['descriptor']['program'], arrays, params)
        symbols = {k: v for k, v in params.items() if not isinstance(v, float)}
        call.update({k: v for k, v in symbols.items() if k not in call})
    elif corpus == 'tsvc':
        sdfg = tsvc.to_sdfg(base['kernel'], tag, simplify=True)
        pipeline(sdfg, 'cpu')
        call = {**arrays, **extra}
    else:  # tsvc25
        sdfg = base['program'].to_sdfg(simplify=True)
        pipeline(sdfg, 'cpu')
        sdfg.name = f'{sdfg.name}_{tag}'
        free = {str(s) for s in sdfg.free_symbols}
        for s in free:
            if s not in sdfg.symbols:
                sdfg.add_symbol(s, dace.int64)
        symbols = {n: v for n, v in tsvc_2_5.SIZES.items() if n in free}
        call = {**arrays, **extra, **symbols}
    if single_core:
        force_sequential(sdfg)
    return sdfg, arrays, call


def collect_outputs(corpus, base, arrays, ret):
    """Return the kernel's output arrays as a name -> ndarray dict. npbench outputs
    may be returned values (``_collect_outputs``); the others mutate ``arrays`` in
    place, so every ndarray argument is an output."""
    if corpus == 'np':
        return {k: np.asarray(v) for k, v in npbench._collect_outputs(base['descriptor'], ret, arrays).items()}
    return {k: v for k, v in arrays.items() if isinstance(v, np.ndarray)}


def frame_cpp(sdfg):
    """Path to the generated CPU frame ``.cpp`` for ``sdfg`` (the ``<name>.cpp`` under
    ``.dacecache/<name>/src/cpu``; falls back to the largest cpp there)."""
    cpp_dir = os.path.join(sdfg.build_folder, 'src', 'cpu')
    files = sorted(glob.glob(os.path.join(cpp_dir, '*.cpp')))
    if not files:
        return None
    named = [f for f in files if os.path.basename(f) == f'{sdfg.name}.cpp']
    return named[0] if named else max(files, key=os.path.getsize)


def measure(corpus, name, base, codegen, single_core, reps, warmup_codegen, tag):
    """Build + measure one (kernel, codegen, mode) lane. Runs INSIDE the forked
    child; returns a payload dict of metric scalars + output ndarrays."""
    set_implementation(codegen)
    sdfg, arrays, call = build_case(corpus, name, base, tag, single_core)

    # emission_ms: time generate_code() -- the experimental preprocessing passes
    # (NSDFG inlining, MarkConstInit, InlineTaskletConnectors) run INSIDE it, so this
    # captures passes + code emission. generate_code() deep-copies self internally,
    # so `sdfg` stays pristine for the compile below; a warm-up first pays the
    # one-time lazy-import / JIT cost so the timed number is the actual pass cost.
    if warmup_codegen:
        sdfg.generate_code()
    t0 = time.perf_counter()
    sdfg.generate_code()
    emission_ms = (time.perf_counter() - t0) * 1000.0

    # compile_ms: pure C++/CMake build with clang-tidy suppressed (compile_without_tidy).
    # sdfg.compile() deep-copies internally, so `sdfg` stays pristine and the generated
    # files land under sdfg.build_folder. Clear the build folder first so this is a true
    # COLD build every time -- a stale cache would report a no-op link, and the
    # post-compile tidy below dirties the readable lane's cached source, which would
    # otherwise give legacy an unfair cache advantage on a rerun.
    shutil.rmtree(sdfg.build_folder, ignore_errors=True)
    compiled, compile_ms = compile_without_tidy(sdfg)

    # clang-tidy (readable lane only), timed separately and folded into codegen_ms.
    tidy_ms, tidy_ran = (0.0, False)
    if codegen == 'experimental':
        tidy_ms, tidy_ran = tidy_generated_cpp(sdfg)
    codegen_ms = emission_ms + tidy_ms

    frame = frame_cpp(sdfg)  # read AFTER tidy -> loc/bytes reflect the final readable form
    loc = float(sum(1 for _ in open(frame)) if frame else 0)
    code_bytes = float(os.path.getsize(frame) if frame else 0)

    # Warm-up run doubles as the correctness output capture (a single clean run on
    # the private inputs -- identical to what run_full_corpus_sweep compares). The
    # binary was built before the source tidy, so runtime is unaffected by tidy.
    ret = compiled(**call)
    outputs = collect_outputs(corpus, base, arrays, ret)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        compiled(**call)
        times.append(time.perf_counter() - t0)
    runtime_ms = statistics.median(times) * 1000.0 if times else float('nan')

    payload = dict(codegen_ms=codegen_ms,
                   compile_ms=compile_ms,
                   tidy_ms=tidy_ms,
                   runtime_ms=runtime_ms,
                   loc=loc,
                   code_bytes=code_bytes,
                   tidy=1.0 if tidy_ran else 0.0)
    payload.update(outputs)
    return payload


# --------------------------------------------------------------------------
# Fork isolation: run `fn` in a child, marshal its result (or error) back.
# --------------------------------------------------------------------------
def run_isolated_result(fn, timeout):
    """Run ``fn() -> dict`` in a forked child and return ``(ok, payload_or_error)``.

    Repo rule: always fork when running a compiled kernel. The child pickles its
    result to a temp file and exits 0; a raised exception is pickled as an error
    payload (so the row carries a real message), and a hard crash / timeout is
    caught via the repo's tested ``waitpid_with_timeout`` and reported as an error.
    """
    handle, path = tempfile.mkstemp(suffix='.pkl')
    os.close(handle)
    pid = os.fork()
    if pid == 0:  # child
        try:
            result = fn()
            with open(path, 'wb') as f:
                pickle.dump({'ok': True, 'payload': result}, f)
        except BaseException as ex:  # noqa: BLE001 - report, never raise past fork
            import traceback
            traceback.print_exc()
            try:
                with open(path, 'wb') as f:
                    pickle.dump({'ok': False, 'error': f'{type(ex).__name__}: {str(ex).splitlines()[0][:200]}'}, f)
            except BaseException:  # noqa: BLE001 - unpicklable state; parent sees empty file
                pass
        finally:
            os._exit(0)
    try:
        waitpid_with_timeout(pid, timeout)
        with open(path, 'rb') as f:
            record = pickle.load(f)
        if record.get('ok'):
            return True, record['payload']
        return False, record.get('error', 'child reported failure')
    except Exception as ex:  # noqa: BLE001 - crash / timeout / empty payload
        return False, f'{type(ex).__name__}: {str(ex).splitlines()[0][:200]}' if str(ex) else type(ex).__name__
    finally:
        if os.path.exists(path):
            os.remove(path)


def split_payload(payload):
    """Split a child payload into (metrics dict, outputs dict of ndarrays)."""
    metrics = {k: payload[k] for k in METRIC_KEYS if k in payload}
    outputs = {k: v for k, v in payload.items() if k not in METRIC_KEYS}
    return metrics, outputs


def bit_exact(legacy_outputs, experimental_outputs):
    """(equal, max_abs_diff) over the arrays shared by both output dicts. Bit-exact
    is the correctness bar: on CPU the two lanes share inputs, host compiler and
    (single_core) schedule, so any difference is a real lowering bug."""
    worst = 0.0
    for k, lv in legacy_outputs.items():
        ev = experimental_outputs.get(k)
        if isinstance(lv, np.ndarray) and isinstance(ev, np.ndarray) and lv.size:
            worst = max(worst, float(np.max(np.abs(lv.astype(np.float64) - ev.astype(np.float64)))))
    return worst == 0.0, worst


# --------------------------------------------------------------------------
# CloudSC (optional, single-core only, large -> long timeout).
# --------------------------------------------------------------------------
def cloudsc_base():
    """Build the un-simplified CloudSC SDFG + one input set ONCE (identical inputs
    for both lanes via fork COW). Returns None if the loader/data is unavailable."""
    try:
        from tests.corpus.cloudsc.generate_data_for_cloudsc import build_cloudsc_sdfg, generate_cloudsc_inputs
    except Exception as ex:  # noqa: BLE001 - loader not importable -> skip cleanly
        print(f'cloudsc: loader unavailable ({type(ex).__name__}: {ex}); skipping')
        return None
    try:
        sdfg = build_cloudsc_sdfg(simplify=False)
        inputs = generate_cloudsc_inputs(sdfg, seed=0)
    except Exception as ex:  # noqa: BLE001 - data/build unavailable -> skip cleanly
        print(f'cloudsc: build/inputs unavailable ({type(ex).__name__}: {ex}); skipping')
        return None
    return dict(sdfg=sdfg, inputs=inputs)


def measure_cloudsc(base, codegen, tag):
    """Measure one CloudSC lane (single-core). Deep-copies + simplifies the shared
    base SDFG, forces sequential schedules, then times codegen/compile/runtime.
    No codegen warm-up (a second full CloudSC codegen would be minutes)."""
    from tests.corpus.cloudsc.generate_data_for_cloudsc import make_sequential
    set_implementation(codegen)
    sdfg = copy.deepcopy(base['sdfg'])
    sdfg.simplify()
    sdfg.name = f'cloudsc_{tag}'
    make_sequential(sdfg)

    # No generate_code() warm-up: a second full CloudSC emission would be minutes.
    t0 = time.perf_counter()
    sdfg.generate_code()
    emission_ms = (time.perf_counter() - t0) * 1000.0

    shutil.rmtree(sdfg.build_folder, ignore_errors=True)  # cold build
    compiled, compile_ms = compile_without_tidy(sdfg)  # tidy suppressed -> pure build

    tidy_ms, tidy_ran = (0.0, False)
    if codegen == 'experimental':
        tidy_ms, tidy_ran = tidy_generated_cpp(sdfg)
    codegen_ms = emission_ms + tidy_ms

    frame = frame_cpp(sdfg)  # after tidy -> final readable form
    loc = float(sum(1 for _ in open(frame)) if frame else 0)
    code_bytes = float(os.path.getsize(frame) if frame else 0)

    args = copy.deepcopy(base['inputs'])
    compiled(**args)
    outputs = {k: v for k, v in args.items() if isinstance(v, np.ndarray)}
    times = []
    for _ in range(3):  # CloudSC runs are expensive: 3 timed reps
        rep_args = copy.deepcopy(base['inputs'])
        t0 = time.perf_counter()
        compiled(**rep_args)
        times.append(time.perf_counter() - t0)
    runtime_ms = statistics.median(times) * 1000.0

    payload = dict(codegen_ms=codegen_ms,
                   compile_ms=compile_ms,
                   tidy_ms=tidy_ms,
                   runtime_ms=runtime_ms,
                   loc=loc,
                   code_bytes=code_bytes,
                   tidy=1.0 if tidy_ran else 0.0)
    payload.update(outputs)
    return payload


# --------------------------------------------------------------------------
# Case selection + per-case driver + TSV assembly.
# --------------------------------------------------------------------------
def select_cases(corpus, only):
    """The (corpus_tag, name) cases for the requested --corpus, optionally filtered
    by an --only substring on ``corpus/name``."""
    cases = all_cases()
    if corpus != 'all':
        tag = CORPUS_MAP[corpus]
        cases = [(c, n) for (c, n) in cases if c == tag]
    if only:
        cases = [(c, n) for (c, n) in cases if only in f'{c}/{n}']
    return cases


def rank_and_size(args):
    """(rank, nranks): explicit overrides win, then SLURM, then mpi4py, else 0/1."""
    if args.rank is not None and args.nranks is not None:
        return args.rank, args.nranks
    if 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS'])
    try:
        from mpi4py import MPI
        # Query Is_initialized() first: with MPI4PY_RC_INITIALIZE=0 (our anti-hang
        # default) MPI_Init is never auto-called, and touching a communicator before
        # init hard-aborts the process. Only trust mpi4py when it is truly running.
        if MPI.Is_initialized():
            comm = MPI.COMM_WORLD
            if comm.Get_size() > 1:
                return comm.Get_rank(), comm.Get_size()
    except Exception:  # noqa: BLE001 - mpi4py absent / single process
        pass
    return 0, 1


def tidy_cell(codegen, metrics):
    """Render the ``tidy`` column: na for legacy, yes/missing for experimental."""
    if codegen == 'legacy':
        return 'na'
    return 'yes' if metrics.get('tidy', 0.0) >= 1.0 else 'missing'


def emit_row(writer, handle, kernel, corpus, mode, codegen, threads, ok, payload, correct):
    """Write one TSV row (a measured lane or an ERROR row) and echo it."""
    if ok:
        metrics, _ = split_payload(payload)
        row = dict(kernel=kernel,
                   corpus=corpus,
                   mode=mode,
                   codegen=codegen,
                   threads=threads,
                   codegen_ms=f"{metrics['codegen_ms']:.3f}",
                   compile_ms=f"{metrics['compile_ms']:.3f}",
                   tidy_ms=f"{metrics['tidy_ms']:.3f}",
                   runtime_ms=f"{metrics['runtime_ms']:.6f}",
                   loc=int(metrics['loc']),
                   code_bytes=int(metrics['code_bytes']),
                   tidy=tidy_cell(codegen, metrics),
                   correct=correct,
                   error='')
    else:
        row = dict(
            kernel=kernel,
            corpus=corpus,
            mode=mode,
            codegen=codegen,
            threads=threads,
            codegen_ms='',
            compile_ms='',
            tidy_ms='',
            runtime_ms='',
            loc='',
            code_bytes='',
            tidy='',  # the build never reached the tidy pass; leave blank rather than imply "missing"
            correct='error',
            error=str(payload))
    writer.writerow(row)
    handle.flush()  # keep partial results on disk if a later kernel crashes the sweep
    print(f"[{corpus}/{kernel}] {mode}/{codegen}: codegen={row['codegen_ms'] or '-'}ms "
          f"(tidy={row['tidy_ms'] or '-'}ms) compile={row['compile_ms'] or '-'}ms "
          f"runtime={row['runtime_ms'] or '-'}ms loc={row['loc']} tidy={row['tidy']} correct={row['correct']}" +
          (f" ({row['error']})" if row['error'] else ''),
          flush=True)


def process_kernel(writer, handle, corpus, name, mode, codegens, threads, reps, timeout, warmup_codegen):
    """Measure the requested codegen lane(s) for one kernel in one mode, isolated
    per lane, and emit their rows (experimental correctness == bit-exact vs legacy
    when both lanes ran)."""
    single_core = mode == 'single_core'
    base = make_base(corpus, name)  # inputs generated ONCE -> identical for both lanes
    results = {}
    for codegen in codegens:
        tag = f'{mode}_{codegen}'
        ok, payload = run_isolated_result(
            lambda cg=codegen, t=tag: measure(corpus, name, base, cg, single_core, reps, warmup_codegen, t),
            timeout=timeout)
        results[codegen] = (ok, payload)

    legacy_outputs = None
    if 'legacy' in results and results['legacy'][0]:
        _, legacy_outputs = split_payload(results['legacy'][1])

    for codegen in codegens:
        ok, payload = results[codegen]
        correct = 'na'
        if ok and codegen == 'legacy' and 'experimental' in codegens:
            correct = 'ref'
        elif ok and codegen == 'experimental' and legacy_outputs is not None:
            _, exp_outputs = split_payload(payload)
            equal, worst = bit_exact(legacy_outputs, exp_outputs)
            correct = 'pass' if equal else f'fail({worst:.2e})'
        emit_row(writer, handle, name, corpus, mode, codegen, threads, ok, payload, correct)


def process_cloudsc(writer, handle, codegens, reps, timeout):
    """Measure CloudSC (single_core) for the requested codegen lane(s), or skip
    cleanly if unavailable."""
    base = cloudsc_base()
    if base is None:
        return
    results = {}
    for codegen in codegens:
        ok, payload = run_isolated_result(lambda cg=codegen: measure_cloudsc(base, cg, cg), timeout=timeout)
        results[codegen] = (ok, payload)
    legacy_outputs = None
    if 'legacy' in results and results['legacy'][0]:
        _, legacy_outputs = split_payload(results['legacy'][1])
    for codegen in codegens:
        ok, payload = results[codegen]
        correct = 'na'
        if ok and codegen == 'legacy' and 'experimental' in codegens:
            correct = 'ref'
        elif ok and codegen == 'experimental' and legacy_outputs is not None:
            _, exp_outputs = split_payload(payload)
            equal, worst = bit_exact(legacy_outputs, exp_outputs)
            correct = 'pass' if equal else f'fail({worst:.2e})'
        emit_row(writer, handle, 'cloudsc', 'cloudsc', 'single_core', codegen, 1, ok, payload, correct)


# --------------------------------------------------------------------------
# Threads, summary, merge.
# --------------------------------------------------------------------------
def mode_threads(mode):
    """OMP width for a mode: single_core is always 1; multi_core honors an exported
    OMP_NUM_THREADS (the sbatch's node width), else the host CPU count."""
    if mode == 'single_core':
        return 1
    exported = os.environ.get('OMP_NUM_THREADS')
    if exported and exported.isdigit() and int(exported) > 1:
        return int(exported)
    return os.cpu_count() or 1


def read_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_or_none(values):
    return statistics.mean(values) if values else None


def summarize(rows):
    """Print the readability-vs-cost summary: mean LoC / code_bytes per lane, the
    mean codegen_ms (with its clang-tidy portion) and runtime ratios, per mode and
    overall. Pairs a legacy and experimental row by (corpus, kernel, mode). The
    headline story: experimental spends more codegen_ms (extra passes + tidy) for
    readability, at ~equal runtime_ms."""
    pairs = {}
    for r in rows:
        pairs.setdefault((r['corpus'], r['kernel'], r['mode']), {})[r['codegen']] = r

    print('\n==== SUMMARY (readability gained vs codegen-time cost) ====')
    modes = sorted({k[2] for k in pairs})
    for mode in modes + (['ALL'] if len(modes) > 1 else []):
        loc_l, loc_e, by_l, by_e = [], [], [], []
        cg_l, cg_e, tidy_e, cg_ratio, rt_ratio = [], [], [], [], []
        for (corpus, kernel, m), lanes in pairs.items():
            if mode != 'ALL' and m != mode:
                continue
            leg, exp = lanes.get('legacy'), lanes.get('experimental')
            if not leg or not exp:
                continue
            ll, le = to_float(leg['loc']), to_float(exp['loc'])
            bl, be = to_float(leg['code_bytes']), to_float(exp['code_bytes'])
            cl, ce = to_float(leg['codegen_ms']), to_float(exp['codegen_ms'])
            te = to_float(exp['tidy_ms'])
            rl, re = to_float(leg['runtime_ms']), to_float(exp['runtime_ms'])
            if ll is not None and le is not None:
                loc_l.append(ll)
                loc_e.append(le)
            if bl is not None and be is not None:
                by_l.append(bl)
                by_e.append(be)
            if cl is not None and ce is not None:
                cg_l.append(cl)
                cg_e.append(ce)
                if cl:
                    cg_ratio.append(ce / cl)
            if te is not None:
                tidy_e.append(te)
            if rl and re:
                rt_ratio.append(re / rl)
        if not loc_l:
            continue
        line = (f'  [{mode}] n={len(loc_l)}  '
                f'LoC legacy={mean_or_none(loc_l):.1f} exp={mean_or_none(loc_e):.1f} '
                f'({mean_or_none(loc_e) / mean_or_none(loc_l):.2f}x)  '
                f'bytes legacy={mean_or_none(by_l):.0f} exp={mean_or_none(by_e):.0f}')
        if cg_l:
            line += (f'  codegen_ms legacy={mean_or_none(cg_l):.1f} exp={mean_or_none(cg_e):.1f} '
                     f'(tidy={mean_or_none(tidy_e):.1f}) ratio={mean_or_none(cg_ratio):.2f}x')
        if rt_ratio:
            line += f'  runtime ratio(exp/leg)={mean_or_none(rt_ratio):.2f}'
        print(line)
    npass = sum(1 for r in rows if r['correct'] == 'pass')
    nfail = sum(1 for r in rows if r['correct'].startswith('fail'))
    nerr = sum(1 for r in rows if r['correct'] == 'error')
    print(f'  correctness: {npass} pass, {nfail} fail, {nerr} error (experimental vs legacy, bit-exact)')


def merge(out):
    """Concatenate ``<out>.rank*`` into ``<out>`` (single header) and summarize."""
    parts = sorted(glob.glob(f'{out}.rank*'))
    if not parts:
        print(f'merge: no per-rank files matching {out}.rank*')
        return
    all_rows = []
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter='\t')
        writer.writeheader()
        for part in parts:
            for row in read_rows(part):
                writer.writerow(row)
                all_rows.append(row)
    print(f'merged {len(parts)} rank file(s) -> {out} ({len(all_rows)} rows)')
    summarize(all_rows)


# --------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=('single_core', 'multi_core', 'both'), default='both')
    ap.add_argument('--codegen', choices=('legacy', 'experimental', 'both'), default='both')
    ap.add_argument('--corpus', choices=('all', 'npbench', 'polybench', 'tsvc', 'tsvc_2_5'), default='all')
    ap.add_argument('--cloudsc', action='store_true', help='also measure CloudSC (single-core, optional, large)')
    ap.add_argument('--only', default=None, help='substring filter on corpus/kernel (handy for smoke tests)')
    ap.add_argument('--reps', type=int, default=10, help='timed repetitions per lane (median; default 10)')
    ap.add_argument('--timeout', type=float, default=900.0, help='per-measurement subprocess timeout, seconds')
    ap.add_argument('--cloudsc-timeout', type=float, default=7200.0, help='CloudSC per-measurement timeout, seconds')
    ap.add_argument('--no-codegen-warmup',
                    action='store_true',
                    help='skip the generate_code() warm-up (faster, noisier codegen_ms)')
    ap.add_argument('--out', default='compare.tsv', help='TSV output path')
    ap.add_argument('--rank', type=int, default=None, help='override this rank index')
    ap.add_argument('--nranks', type=int, default=None, help='override the total rank count')
    ap.add_argument('--merge', action='store_true', help='merge <out>.rank* into <out> and summarize, then exit')
    args = ap.parse_args()

    if args.merge:
        merge(args.out)
        return

    rank, nranks = rank_and_size(args)
    modes = list(MODES) if args.mode == 'both' else [args.mode]
    codegens = list(CODEGENS) if args.codegen == 'both' else [args.codegen]
    cases = select_cases(args.corpus, args.only)
    my_cases = cases[rank::nranks]
    out_path = f'{args.out}.rank{rank}' if nranks > 1 else args.out

    print(f'rank {rank}/{nranks}: {len(my_cases)}/{len(cases)} kernels  modes={modes} codegens={codegens} '
          f'corpus={args.corpus} reps={args.reps} -> {out_path}')
    if not shutil.which('clang-tidy'):
        print('WARNING: clang-tidy not found; experimental tidy pass will be skipped (tidy=missing)')
    if not shutil.which('clang-format'):
        print('WARNING: clang-format not found; experimental formatting will be skipped')

    all_rows = []
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter='\t')
        writer.writeheader()
        f.flush()
        for mode in modes:
            threads = mode_threads(mode)
            os.environ['OMP_NUM_THREADS'] = str(threads)  # inherited by every forked measurement
            for corpus_tag, name in my_cases:
                process_kernel(writer, f, corpus_tag, name, mode, codegens, threads, args.reps, args.timeout,
                               not args.no_codegen_warmup)
        # CloudSC (single-core only) runs on rank 0 -- it is one large case.
        if args.cloudsc and rank == 0:
            os.environ['OMP_NUM_THREADS'] = '1'
            process_cloudsc(writer, f, codegens, args.reps, args.cloudsc_timeout)
        all_rows = read_rows(out_path)

    print(f'wrote {out_path} ({len(all_rows)} rows)')
    if nranks == 1:
        summarize(all_rows)
    else:
        print(f'rank {rank} done; run `--merge --out {args.out} --nranks {nranks}` after all ranks finish')


if __name__ == '__main__':
    main()
