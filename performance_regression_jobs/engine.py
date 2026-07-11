"""Shared engine for the performance_regression_jobs scripts: plain functions only.

Reuses the exact patterns already established in dace/optimization/utils.py:
world-rank detection, subprocess-isolated measurement (so a segfaulting SDFG or
native binary never kills the whole sweep), and instrumentation-report-based
timing (sdfg.instrument + get_latest_report instead of wall-clock wrapping).
"""
import os

# Set BEFORE any dace import (dace is imported lazily below, but every
# entry-point script that imports this module must set the same defaults at its
# own top too, ahead of its top-level `import dace`): DaCe scripts otherwise
# block on MPI_Init and the sweep looks like it hangs. setdefault so an explicit
# value in the environment (e.g. a real MPI launch) still wins.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import copy
import csv
import glob
import json
import multiprocessing as mp
import re
import shutil
import socket
import subprocess
import tempfile
import time

_RESULTS_CSV = 'results.csv'
_STATUS_CSV = 'status.csv'
_COMPILE_CSV = 'compile.csv'


# --------------------------------------------------------------------------
# Multi-rank: identical semantics to dace/optimization/utils.py's
# get_world_rank/get_world_size (falls back to a single sequential rank).
# --------------------------------------------------------------------------
def get_world_rank():
    for var in ('OMPI_COMM_WORLD_RANK', 'SLURM_PROCID', 'MV2_COMM_WORLD_RANK'):
        if var in os.environ:
            return int(os.environ[var])
    return 0


def get_world_size():
    for var in ('OMPI_COMM_WORLD_SIZE', 'SLURM_NTASKS', 'MV2_COMM_WORLD_SIZE'):
        if var in os.environ:
            return int(os.environ[var])
    return 1


def my_slice(items, rank, world):
    """Deterministic, even round-robin slice of a stably-ordered list."""
    return items[rank::world]


def pick_cxx_compiler(explicit=None):
    """C++ compiler for DaCe's own codegen (native lanes pick their own
    vendor's compiler independently -- see native_harness.compile_lane).

    `explicit` (--cxx / DACE_PERF_CXX) may be a path or bare name; an
    unresolvable value raises immediately rather than silently falling back.
    Absent an override: clang++ if on PATH, else g++, else None (DaCe's own
    default compiler config is left alone)."""
    if explicit:
        resolved = shutil.which(explicit)
        if not resolved:
            raise FileNotFoundError(f'--cxx {explicit!r}: not found or not executable')
        return resolved
    import native_harness as nh
    return nh.find_best_cpp_compiler()


def configure_dace_process():
    """Per-process DaCe setup: Config state doesn't survive a spawn boundary,
    so this must run inside every measurement subprocess, not just once at
    the parent's startup.

    - cache='name' keys each SDFG's build folder purely on sdfg.name (no
      content hash, no PID) -- so as long as every caller gives each distinct
      (corpus, kernel, pipeline, par/seq) variant its own stable, unique name
      (see each script's _build_sdfg), the exact same variant always maps to
      the exact same build folder. Combined with compiler.use_cache below,
      that means a variant is only ever actually compiled once: every other
      lane-check that needs the same 'baseline' reference, and the timing
      run that follows a candidate's own correctness check, both just load
      the already-compiled binary instead of recompiling.
      ('hash' mode looks like the obvious choice for this but doesn't work:
      DaCe stamps a fresh random guid onto every SDFG node/state on every
      construction, and that guid is part of what gets hashed, so two
      separate builds of the "same" SDFG never hash equal. 'name' mode has no
      such problem since it never inspects SDFG content at all.)
      Safe here because each kernel is owned by exactly one rank (my_slice()
      partitions disjointly) and one rank's lane-checks run strictly
      sequentially (run_isolated blocks until each subprocess exits) -- so
      the same name is never being compiled by two processes at once. This
      is also why job functions don't call a build-folder cleanup: deleting
      the artifact right after use would defeat the point of reusing it.
    - default_build_folder is redirected to /dev/shm when available: DaCe's
      own default ('.dacecache', a relative path under the current working
      directory) lands every CMake configure + compile on whatever
      filesystem the job runs from. On an HPC cluster that's typically a
      shared, possibly network-backed scratch filesystem -- with many ranks
      per node all compiling concurrently, that contention alone can make a
      normally sub-second compile take minutes (observed: kernels that
      compile in under a second standalone timed out at 300s under a real
      multi-rank sweep). /dev/shm (RAM-backed, node-local) is the same fix
      dace/optimization/utils.py already uses for exactly this reason.
    - compiler.cpu.args is guaranteed to contain native_harness.OPT_FLAGS
      (-O3 -march=native -ffast-math) -- see that constant's docstring for
      why a DaCe lane and a native lane must be compiled at the same
      optimization level.
    - compiler.cpu.executable comes from DACE_PERF_CXX (exported by the
      parent from --cxx) or pick_cxx_compiler()'s own autodetection.
    - If that compiler needs it (native_harness.needs_gcc_install_dir:
      clang++/icpx/icpc), --gcc-install-dir is appended to compiler.cpu.args
      -- DaCe's CMake build never adds this itself, so on a machine with
      several GCC versions it can pick one with mismatched libstdc++ headers
      and fail to link. Idempotent (checks the flag isn't already present):
      this can run more than once per process (e.g. _check_dace_job builds
      both a reference and a candidate SDFG)."""
    import dace
    import native_harness as nh
    # Warm the transformation import graph before any to_sdfg() runs in this
    # process. Parsing a nested @dace.program makes to_sdfg import
    # dace.transformation.interstate first, which can trip a lazy-import cycle;
    # importing canonicalize up front loads the modules in an order that resolves
    # cleanly. Harmless (a no-op re-import) once the graph is cycle-free.
    import dace.transformation.passes.canonicalize  # noqa: F401
    dace.Config.set('cache', value='name')
    dace.Config.set('compiler', 'use_cache', value=True)
    shm_root = '/dev/shm'
    if os.path.isdir(shm_root) and os.access(shm_root, os.W_OK):
        dace.Config.set('default_build_folder', value=os.path.join(shm_root, f'dace_perf_jobs_{os.getuid()}'))

    # Set explicitly rather than trusting DaCe's own schema default (which
    # already happens to match): a stray ~/.dace.conf on some machine could
    # override compiler.cpu.args to something that drops these, and then a
    # "canon is faster than native" (or the reverse) result would just be
    # reflecting a flags mismatch, not a real difference. Appended rather
    # than replaced so nothing else already in args (e.g. -Wall -fPIC) is lost.
    args = dace.Config.get('compiler', 'cpu', 'args')
    for flag in nh.OPT_FLAGS:
        if flag not in args:
            args = f'{args} {flag}'
    dace.Config.set('compiler', 'cpu', 'args', value=args)

    cxx = pick_cxx_compiler(os.environ.get('DACE_PERF_CXX'))
    if cxx:
        dace.Config.set('compiler', 'cpu', 'executable', value=cxx)
        if nh.needs_gcc_install_dir(cxx):
            gcc_dir = nh.find_gcc_install_dir()
            if gcc_dir:
                flag = f'--gcc-install-dir={gcc_dir}'
                args = dace.Config.get('compiler', 'cpu', 'args')
                if flag not in args:
                    dace.Config.set('compiler', 'cpu', 'args', value=f'{args} {flag}')


# --------------------------------------------------------------------------
# Crash isolation: run fn(*args) in a spawned subprocess. A segfault leaves
# no exception to catch -- only a nonzero/None exitcode -- so that is checked
# explicitly in addition to a normal Python exception.
# --------------------------------------------------------------------------
def _run_and_queue(fn, args, q):
    try:
        q.put(('ok', fn(*args)))
    except Exception as e:
        q.put(('error', f'{type(e).__name__}: {e}'))


def run_isolated(fn, args=(), timeout=120):
    """Returns (ok, payload_or_error). Never raises; a crash/timeout is (False, msg)."""
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_run_and_queue, args=(fn, args, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False, f'timeout after {timeout}s'
    if p.exitcode != 0:
        return False, f'crashed (exit code {p.exitcode})'
    try:
        status, payload = q.get(timeout=5)
    except Exception:
        return False, 'crashed (no result returned)'
    return (True, payload) if status == 'ok' else (False, payload)


# --------------------------------------------------------------------------
# DaCe-instrumentation timing. Call from inside the isolated subprocess (the
# SDFG must already be built + pipelined + build_folder-assigned by the caller).
# --------------------------------------------------------------------------
def _flatten_durations(d):
    """report.durations is UUID -> name -> thread-id -> list[ms]; walk to any depth."""
    times = []
    if isinstance(d, dict):
        for v in d.values():
            times.extend(_flatten_durations(v))
    elif isinstance(d, list):
        times.extend(d)
    return times


def time_sdfg(sdfg, call_kwargs, reps, warmup=1):
    """Best-of-`reps` timing via sdfg.instrument + get_latest_report (ms per call).

    The first `warmup` call(s) are executed (and instrumented, same as any
    other call -- simplest way to keep one accumulated report) but sliced off
    before returning, so they never reach the CSV."""
    import dace
    sdfg.instrument = dace.InstrumentationType.Timer
    # Instrumented codegen is different C++ than the plain correctness-check
    # build of the exact same variant -- needs its own cache-key (name) or
    # cache='name' mode would find and silently reuse the uninstrumented
    # binary, leaving get_latest_report() with nothing recorded.
    sdfg.name = f'{sdfg.name}_timed'
    with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
        csdfg = sdfg.compile()
        for _ in range(warmup + reps):
            csdfg(**call_kwargs)
        csdfg.finalize()
    return _flatten_durations(sdfg.get_latest_report().durations)[warmup:]


def _direct_compile_cmd(sdfg, folder):
    """Our OWN compiler command line (not sdfg.compile() / CMake) to build the
    generated frame into a .so: the configured C++ compiler + its OPT flags,
    the generated include/ + DaCe's runtime include, -fopenmp for the parallel
    maps, every src/cpu/*.cpp. Mirrors how nest-forge owns its build."""
    import dace
    cxx = dace.Config.get('compiler', 'cpu', 'executable') or shutil.which('g++') or 'g++'
    args = (dace.Config.get('compiler', 'cpu', 'args') or '').split()
    folder = str(folder)
    srcs = sorted(glob.glob(os.path.join(folder, 'src', 'cpu', '*.cpp')))
    runtime_inc = os.path.join(os.path.dirname(dace.__file__), 'runtime', 'include')
    inc = [f"-I{os.path.join(folder, 'include')}", f'-I{runtime_inc}']
    so = os.path.join(folder, f'lib{sdfg.name}.so')
    # Don't pin a low standard (DaCe codegen may use newer C++); assume the toolchain supports >=23.
    std = os.environ.get('DACE_PERF_CXX_STD', 'c++23')
    return [cxx, *args, f'-std={std}', '-fPIC', '-shared', '-fopenmp', *inc, *srcs, '-o', so], srcs


def compile_sdfg_timed(sdfg):
    """Compile `sdfg` from scratch, returning (codegen_ms, cxx_ms): DaCe C++
    codegen time, then a DIRECT compiler invocation timed on its own.

    Does NOT call sdfg.compile() / DaCe's CMake build. That path pays a CMake
    configure cost (seconds, dominated by CMake not the compiler) and, under
    cache='name', would be a ~0ms no-op on an already-built variant -- neither
    is a useful "compile speed" number. Instead: generate the C++, lay out the
    program folder (fast file writes, not timed as compile), then invoke OUR OWN
    `<cxx> <opt-flags> -fopenmp ... -shared` command (the same compiler + flags
    the runtime lane uses) and time just that subprocess. Fresh temp folder each
    call, so every sample is a real cold compile. Call from an isolated
    subprocess (a bad SDFG can crash codegen)."""
    from dace.codegen import codegen
    from dace.codegen import compiler as dace_compiler
    # Build on node-local RAM (/dev/shm) when available, same reason
    # configure_dace_process redirects DaCe's own builds there: on a cluster the
    # default temp dir is often a shared/network scratch FS, and many ranks
    # compiling there at once turns a sub-second compile into minutes.
    shm = '/dev/shm'
    tmp_parent = shm if os.path.isdir(shm) and os.access(shm, os.W_OK) else None
    build_root = tempfile.mkdtemp(prefix='dace_compilebench_', dir=tmp_parent)
    try:
        t0 = time.perf_counter()
        code_objects = codegen.generate_code(sdfg)
        codegen_ms = (time.perf_counter() - t0) * 1000.0
        folder = dace_compiler.generate_program_folder(sdfg, code_objects, build_root)
        cmd, srcs = _direct_compile_cmd(sdfg, folder)
        if not srcs:
            raise RuntimeError(f'no src/cpu/*.cpp generated for {sdfg.name}')
        t1 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        cxx_ms = (time.perf_counter() - t1) * 1000.0
        if p.returncode != 0:
            raise RuntimeError(f'direct compile failed ({os.path.basename(cmd[0])}): {p.stderr[-1500:]}')
    finally:
        shutil.rmtree(build_root, ignore_errors=True)
    return codegen_ms, cxx_ms


def arrays_close(ref, got, tol=1e-9):
    """True if every floating-point array/scalar in `ref` matches its `got`
    counterpart within `tol` -- non-floating entries (ints, symbol values)
    are skipped, since only numerical drift indicates a real correctness bug
    for these kernels. Shared by the TSVC2/TSVC2.5 scripts (both DaCe and
    native lanes); NPBench+PolyBench uses its own looser default tolerance."""
    import numpy as np
    for name, r in ref.items():
        g = got.get(name)
        if g is None or not np.issubdtype(np.asarray(r).dtype, np.floating):
            continue
        if not np.allclose(r, g, rtol=tol, atol=tol, equal_nan=True):
            return False
    return True


# --------------------------------------------------------------------------
# Shared pipelines: existing passes/transformations only, nothing new. Every
# pipeline takes (sdfg, device) where device is 'cpu' or 'gpu' -- so the exact
# same four comparison points can be measured on either target. "-seq" lanes
# are never a separate pipeline -- always make_sequential(par result).
# --------------------------------------------------------------------------
def _device_type(device):
    import dace
    return dace.DeviceType.GPU if device == 'gpu' else dace.DeviceType.CPU


def _set_tree_reduction(enabled):
    """Set the codegen ``compiler.tree_reduction`` flag for THIS process.

    ON  -> a parallel WCR reduction lowers to privatize-and-tree-reduce (CPU OpenMP
           ``reduction(op:var)`` clause / GPU per-block ``cub::BlockReduce``).
    OFF -> the same reduction lowers to a plain atomic WCR (CPU per-iteration atomic /
           GPU ``atomicAdd`` per thread) -- correct but contended.

    Only the canonicalize pipelines turn it on; auto_opt and parallel turn it off, so
    their reductions emit plain atomic WCR and canon is the only lane that tree-reduces.
    Each pipeline runs in its own measurement subprocess right before that subprocess
    generates code (see the drivers' _check_job / _time_dace_job), so this process-global
    Config set reaches codegen and never leaks across lanes."""
    import dace
    dace.Config.set('compiler', 'tree_reduction', value=bool(enabled))


def pipeline_auto_opt(sdfg, device='cpu'):
    """DaCe's own auto_optimize for the target device -- the speedup baseline
    every other pipeline is reported against. Emits atomic WCR reductions (no
    tree reduction -- that is the canonicalize lanes' distinguishing lowering)."""
    from dace.transformation.auto.auto_optimize import auto_optimize
    _set_tree_reduction(False)
    return auto_optimize(sdfg, _device_type(device))


def pipeline_parallel(sdfg, device='cpu'):
    """The light pipeline: simplify + LoopToMap + MapFusion + simplify. On GPU
    the non-transient (I/O) arrays are moved to GPU_Global storage FIRST via
    ``apply_gpu_storage`` (the same helper ``auto_optimize`` uses -- arrays and
    written scalars go on-device, read-only scalar args stay host kernel args),
    then ``apply_gpu_transformations`` moves the parallel maps onto the device.
    So the measured program keeps its data resident on the GPU rather than
    round-tripping every buffer host<->device per call."""
    from dace.transformation.interstate import LoopToMap
    from dace.transformation.dataflow import MapFusionHorizontal, MapFusionVertical
    from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
    from dace.transformation.auto.auto_optimize import apply_gpu_storage
    _set_tree_reduction(False)  # atomic WCR, like auto_opt -- only canon tree-reduces
    sdfg.simplify(validate=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    PatternMatchAndApplyRepeated([MapFusionVertical(), MapFusionHorizontal()]).apply_pass(sdfg, {})
    sdfg.simplify(validate=True)
    if device == 'gpu':
        apply_gpu_storage(sdfg)  # non-transient arrays -> GPU_Global (data resident on device)
        sdfg.apply_gpu_transformations()
        sdfg.simplify(validate=True)
    return sdfg


#: The canonicalize knobs used everywhere (mirrors the sibling gate's _CPU
#: preset). Passed verbatim for both targets -- only `target` differs.
_CANON_KNOBS = dict(peel_limit=4,
                    break_anti_dependence=True,
                    interchange_carry_with_map=True,
                    scatter_to_guarded_maps=True)


def pipeline_canon(sdfg, device='cpu'):
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target
    _set_tree_reduction(True)  # canon is the only lane that tree-reduces its WCR
    return finalize_for_target(canonicalize(sdfg, validate=True, target=device, **_CANON_KNOBS), device)


def pipeline_fast_canon(sdfg, device='cpu'):
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target
    _set_tree_reduction(True)  # canon is the only lane that tree-reduces its WCR
    return finalize_for_target(canonicalize(sdfg, validate=True, fast=True, target=device, **_CANON_KNOBS), device)


#: The 4 DaCe-side comparison points. auto_opt is the BASELINE speedups are
#: reported against; parallel (light simplify+loop2map+mapfusion) and
#: canonicalize / canonicalize(fast=True) are the candidates. The key is also
#: the SDFG-name suffix each variant is cache-keyed on (with a _cpu/_gpu device
#: tail), so e.g. canon->'..._canon_cpu', parallel->'..._parallel_gpu',
#: auto_opt->'..._auto_opt_cpu' -- distinct build folders, never colliding.
PIPELINES = {
    'auto_opt': pipeline_auto_opt,
    'parallel': pipeline_parallel,
    'canon': pipeline_canon,
    'fast-canon': pipeline_fast_canon,
}


# --------------------------------------------------------------------------
# GPU availability probe: run once per process (cached). A machine with no CUDA
# toolchain / no device must DEGRADE GRACEFULLY -- the caller skips the gpu
# device entirely instead of recording a per-kernel compile error for every
# kernel. The probe itself is crash-isolated (run_isolated), so a missing nvcc
# or driver is just (False, msg), never an exception in the sweep.
# --------------------------------------------------------------------------
_GPU_SUPPORTED = None


def _probe_gpu():
    import dace
    import numpy as np

    @dace.program
    def _probe(a: dace.float64[32]):
        a[:] = a + 1.0

    sdfg = _probe.to_sdfg()
    pipeline_auto_opt(sdfg, 'gpu')
    a = np.ones(32, dtype=np.float64)
    sdfg(a=a)
    return bool(np.allclose(a, 2.0))


def gpu_supported(timeout=600):
    global _GPU_SUPPORTED
    if _GPU_SUPPORTED is None:
        ok, payload = run_isolated(_probe_gpu, (), timeout=timeout)
        _GPU_SUPPORTED = bool(ok and payload)
    return _GPU_SUPPORTED


def make_sequential(sdfg):
    """Deep-copy + force every map to Sequential (dace/cli/daceprof.py:222-232's logic)."""
    import dace
    s = copy.deepcopy(sdfg)
    for sd in s.all_sdfgs_recursive():
        sd.openmp_sections = False
    for n, _ in s.all_nodes_recursive():
        if isinstance(n, dace.nodes.EntryNode) and getattr(
                n, 'schedule', False) in (dace.ScheduleType.CPU_Multicore, dace.ScheduleType.CPU_Persistent,
                                          dace.ScheduleType.Default):
            n.schedule = dace.ScheduleType.Sequential
    return s


# --------------------------------------------------------------------------
# Per-kernel results: one CSV row per timing sample (resumable), one row per
# pipeline for correctness (checked once, never re-checked on resume).
# --------------------------------------------------------------------------
_COMPILER_HOST_TAG_CACHE = None


def _slug(s):
    return re.sub(r'[^A-Za-z0-9.+-]', '-', s)


def compiler_host_tag():
    """`<compiler>_<hostname>` -- namespace for DaCe-lane results (baseline/
    auto-opt/canon/fast-canon): a different --cxx (or a different node's
    autodetected default) produces timings that aren't comparable, so this
    keeps them in separate result folders instead of corrupting one shared
    results.csv/status.csv. Native lanes use host_tag() instead -- they pick
    their own compiler regardless of --cxx, so they shouldn't be invalidated
    every time it changes. Cached per process (autodetection does a PATH
    scan, and this is stable for the process's lifetime)."""
    global _COMPILER_HOST_TAG_CACHE
    if _COMPILER_HOST_TAG_CACHE is None:
        cxx = pick_cxx_compiler(os.environ.get('DACE_PERF_CXX'))
        compiler_slug = _slug(os.path.basename(cxx)) if cxx else 'unknown-cxx'
        _COMPILER_HOST_TAG_CACHE = f'{compiler_slug}_{_slug(socket.gethostname())}'
    return _COMPILER_HOST_TAG_CACHE


def host_tag():
    """`<hostname>` -- namespace for native lanes: each picks its own vendor
    compiler independently of --cxx (native_harness.compile_lane), so unlike
    compiler_host_tag() this excludes the DaCe compiler -- otherwise every
    native lane's measurement would be redone every time --cxx changes.
    Hostname alone still prevents two hosts sharing one --results-dir from
    racing on the same compiled .so (see native_build_dir)."""
    return _slug(socket.gethostname())


def result_tag(preset):
    """`<compiler>_<hostname>_<preset>` -- folder DaCe-lane results land in.
    `preset` is slugged so it can never contain the `_` write_tables() counts
    on to tell a DaCe-tag folder (2 underscores) from a native-tag one (1)."""
    return f'{compiler_host_tag()}_{_slug(str(preset))}'


def native_result_tag(preset):
    """`<hostname>_<preset>` -- folder native-lane results land in."""
    return f'{host_tag()}_{_slug(str(preset))}'


def lane_kind(lane):
    """'native' if `lane` is one of native_harness.LANES, else 'dace' -- every
    entry-point script uses this to route a lane to kernel_dir() vs.
    native_kernel_dir()."""
    import native_harness as nh
    return 'native' if lane in nh.LANES else 'dace'


def kernel_dir(results_root, corpus, kernel, preset):
    return os.path.join(results_root, corpus, kernel, result_tag(preset))


def native_kernel_dir(results_root, corpus, kernel, preset):
    return os.path.join(results_root, corpus, kernel, native_result_tag(preset))


def native_build_dir(results_root, rank):
    """Per-rank native .so build directory, namespaced by hostname (see
    host_tag()): two runs sharing one --results-dir from different hosts
    must not race on or clobber the same compiled library -- see
    native_harness.compile_lane's non-atomic `-o so_path` write."""
    return os.path.join(results_root, 'native_build', host_tag(), f'rank{rank}')


def existing_reps(kdir, pipeline):
    path = os.path.join(kdir, _RESULTS_CSV)
    if not os.path.exists(path):
        return 0
    with open(path, newline='') as f:
        return sum(1 for row in csv.DictReader(f) if row['pipeline'] == pipeline)


def append_results(kdir, pipeline, times_ms, start_index):
    os.makedirs(kdir, exist_ok=True)
    path = os.path.join(kdir, _RESULTS_CSV)
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['pipeline', 'rep_index', 'time_ms'])
        for i, t in enumerate(times_ms):
            w.writerow([pipeline, start_index + i, f'{t:.6f}'])


def existing_compile_reps(kdir, pipeline):
    path = os.path.join(kdir, _COMPILE_CSV)
    if not os.path.exists(path):
        return 0
    with open(path, newline='') as f:
        return sum(1 for row in csv.DictReader(f) if row['pipeline'] == pipeline)


def append_compile_results(kdir, pipeline, samples, start_index):
    """`samples` is a list of (codegen_ms, cxx_ms) pairs; one CSV row per sample
    (resumable, exactly like append_results for runtime)."""
    os.makedirs(kdir, exist_ok=True)
    path = os.path.join(kdir, _COMPILE_CSV)
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['pipeline', 'rep_index', 'codegen_ms', 'cxx_ms', 'total_ms'])
        for i, (cg, cx) in enumerate(samples):
            w.writerow([pipeline, start_index + i, f'{cg:.6f}', f'{cx:.6f}', f'{cg + cx:.6f}'])


def read_status(kdir, pipeline):
    path = os.path.join(kdir, _STATUS_CSV)
    if not os.path.exists(path):
        return None
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row['pipeline'] == pipeline:
                return row
    return None


def write_status(kdir, pipeline, correct, error=''):
    os.makedirs(kdir, exist_ok=True)
    path = os.path.join(kdir, _STATUS_CSV)
    rows = []
    if os.path.exists(path):
        with open(path, newline='') as f:
            rows = [r for r in csv.DictReader(f) if r['pipeline'] != pipeline]
    rows.append(dict(pipeline=pipeline, correct=correct, error=error))
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pipeline', 'correct', 'error'])
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def write_run_meta(kdir, **meta):
    os.makedirs(kdir, exist_ok=True)
    meta = dict(meta, host=socket.gethostname(), timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'))
    with open(os.path.join(kdir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def save_sdfg(kdir, sdfg, tag):
    os.makedirs(kdir, exist_ok=True)
    sdfg.save(os.path.join(kdir, f'{tag}.sdfg'))


# --------------------------------------------------------------------------
# Aggregate markdown tables, built by re-scanning the results tree (this is
# also the cross-rank aggregation step -- every rank writes into the same
# <results-root>/<corpus>/ tree, so one final --tables-only pass sees all of it).
# --------------------------------------------------------------------------
def _median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _read_kernel(kdir):
    """pipeline -> {'correct': bool, 'median_ms': float|None, 'min_ms': float|None}
    for one kernel/preset dir. `min_ms` is the best-of-N used for speedups (the
    spec's autoopt_min / pipeline_min); `median_ms` is kept for display."""
    out = {}
    for row in [] if not os.path.exists(os.path.join(kdir, _STATUS_CSV)) else csv.DictReader(
            open(os.path.join(kdir, _STATUS_CSV), newline='')):
        out[row['pipeline']] = dict(correct=row['correct'] == 'True', median_ms=None, min_ms=None)
    times = {}
    if os.path.exists(os.path.join(kdir, _RESULTS_CSV)):
        for row in csv.DictReader(open(os.path.join(kdir, _RESULTS_CSV), newline='')):
            times.setdefault(row['pipeline'], []).append(float(row['time_ms']))
    for pipeline, ts in times.items():
        e = out.setdefault(pipeline, dict(correct=True, median_ms=None, min_ms=None))
        e['median_ms'] = _median(ts)
        e['min_ms'] = min(ts) if ts else None
    return out


def write_tables(results_root, corpus, lanes, baseline_label):
    """Scan <results_root>/<corpus>/**/ and write correctness.md + speedup.md.

    Each kernel folder holds result_tag() folders ('<compiler>_<host>_<preset>',
    2 underscores) and/or native_result_tag() folders ('<host>_<preset>', 1
    underscore -- see result_tag()'s docstring for why the count is reliable).
    A DaCe-tag row is merged with its matching native-tag folder (same host +
    preset), since native results are shared across every --cxx that produced
    a DaCe-tag folder for that host."""
    corpus_dir = os.path.join(results_root, corpus)
    rows = []  # (kernel_label, {pipeline: {'correct':, 'median_ms':}})
    if os.path.isdir(corpus_dir):
        for kernel in sorted(os.listdir(corpus_dir)):
            kpath = os.path.join(corpus_dir, kernel)
            if not os.path.isdir(kpath):
                continue
            tags = sorted(t for t in os.listdir(kpath) if os.path.isdir(os.path.join(kpath, t)))
            dace_tags = [t for t in tags if t.count('_') == 2]
            native_tags = {t for t in tags if t.count('_') == 1}
            matched_natives = set()
            for tag in dace_tags:
                _compiler, host, preset = tag.split('_')
                entries = dict(_read_kernel(os.path.join(kpath, tag)))
                native_tag = f'{host}_{preset}'
                if native_tag in native_tags:
                    entries.update(_read_kernel(os.path.join(kpath, native_tag)))
                    matched_natives.add(native_tag)
                rows.append((f'{kernel} ({tag})', entries))
            # A native-tag folder with no matching DaCe-tag folder yet (e.g. only
            # native lanes have run so far) is still shown on its own.
            for tag in sorted(native_tags - matched_natives):
                rows.append((f'{kernel} ({tag})', _read_kernel(os.path.join(kpath, tag))))

    def _cell_correct(entry):
        if entry is None:
            return '-'
        return ':white_check_mark:' if entry.get('correct') else ':x:'

    def _cell_speedup(entry, base):
        if not entry or not base or not entry.get('correct') or not base.get('correct'):
            return ''
        if not entry.get('min_ms') or not base.get('min_ms'):
            return ''
        return f"{base['min_ms'] / entry['min_ms']:.2f}x"

    os.makedirs(corpus_dir, exist_ok=True)
    with open(os.path.join(corpus_dir, 'correctness.md'), 'w') as f:
        f.write('| kernel | ' + ' | '.join(lanes) + ' |\n')
        f.write('|---' * (len(lanes) + 1) + '|\n')
        for label, entries in rows:
            f.write(f'| {label} | ' + ' | '.join(_cell_correct(entries.get(l)) for l in lanes) + ' |\n')

    other_lanes = [l for l in lanes if l != baseline_label]
    with open(os.path.join(corpus_dir, 'speedup.md'), 'w') as f:
        f.write('| kernel | ' + ' | '.join(other_lanes) + ' |\n')
        f.write('|---' * (len(other_lanes) + 1) + '|\n')
        for label, entries in rows:
            base = entries.get(baseline_label)
            f.write(f'| {label} | ' + ' | '.join(_cell_speedup(entries.get(l), base) for l in other_lanes) + ' |\n')

    print(f'wrote {corpus_dir}/correctness.md and speedup.md ({len(rows)} kernel/preset rows)')


def device_of_tag(tag):
    """The device a result folder belongs to, read off its preset token: a
    '...-gpu' preset (e.g. 'clang++_host_paper-gpu') is a GPU folder, everything
    else (paper-cpu, default, ...) is CPU. Devices are namespaced by folder --
    the CPU and GPU results.csv for one kernel never share a file."""
    return 'gpu' if tag.rsplit('_', 1)[-1].endswith('-gpu') else 'cpu'


def write_summary_csv(results_root, corpus, baseline_label=None):
    """The flat 'listing all' summary CSV: one row per (kernel, tag, pipeline)
    across the whole <results_root>/<corpus>/ tree (so it is also the cross-rank
    + cross-device aggregation). Columns carry `device` and the pipeline/baseline
    names; `speedup_vs_baseline` is filled when baseline_label is given and both
    it and the row's pipeline are correct with a median in the same tag."""
    corpus_dir = os.path.join(results_root, corpus)
    fields = ['corpus', 'kernel', 'tag', 'device', 'preset', 'pipeline', 'correct', 'min_ms', 'median_ms',
              'speedup_vs_baseline']
    rows = []
    if os.path.isdir(corpus_dir):
        for kernel in sorted(os.listdir(corpus_dir)):
            kpath = os.path.join(corpus_dir, kernel)
            if not os.path.isdir(kpath):
                continue
            for tag in sorted(t for t in os.listdir(kpath) if os.path.isdir(os.path.join(kpath, t))):
                entries = _read_kernel(os.path.join(kpath, tag))
                base = entries.get(baseline_label) if baseline_label else None
                base_ms = base['min_ms'] if base and base.get('correct') and base.get('min_ms') else None
                for pipeline, e in entries.items():
                    speedup = ''  # spec's autoopt_min / pipeline_min (best-of-N)
                    if base_ms and e.get('correct') and e.get('min_ms'):
                        speedup = f"{base_ms / e['min_ms']:.4f}"
                    rows.append(
                        dict(corpus=corpus, kernel=kernel, tag=tag, device=device_of_tag(tag),
                             preset=tag.rsplit('_', 1)[-1], pipeline=pipeline, correct=e.get('correct'),
                             min_ms='' if e.get('min_ms') is None else f"{e['min_ms']:.6f}",
                             median_ms='' if e.get('median_ms') is None else f"{e['median_ms']:.6f}",
                             speedup_vs_baseline=speedup))
    os.makedirs(corpus_dir, exist_ok=True)
    path = os.path.join(corpus_dir, 'summary.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path} ({len(rows)} rows)')
    return path


def _read_kernel_compile(kdir):
    """(pipeline -> {'codegen': med_ms, 'cxx': med_ms, 'total': med_ms}) for one kernel/tag dir."""
    path = os.path.join(kdir, _COMPILE_CSV)
    if not os.path.exists(path):
        return {}
    cg, cx, tot = {}, {}, {}
    for row in csv.DictReader(open(path, newline='')):
        p = row['pipeline']
        cg.setdefault(p, []).append(float(row['codegen_ms']))
        cx.setdefault(p, []).append(float(row['cxx_ms']))
        tot.setdefault(p, []).append(float(row['total_ms']))
    return {p: dict(codegen=_median(cg[p]), cxx=_median(cx[p]), total=_median(tot[p])) for p in cg}


def write_compile_tables(results_root, corpus, lanes):
    """Scan <results_root>/<corpus>/**/ and write compile_total.md + compile_codegen.md +
    compile_cxx.md -- median compile time (ms) per pipeline per kernel. Compile time is a DaCe-codegen
    metric, so only DaCe-tag folders ('<compiler>_<host>_<preset>', 2 underscores) are read; this is also
    the cross-rank aggregation step, exactly like write_tables (every rank wrote into the same tree)."""
    corpus_dir = os.path.join(results_root, corpus)
    rows = []  # (label, {pipeline: {'codegen','cxx','total'}})
    if os.path.isdir(corpus_dir):
        for kernel in sorted(os.listdir(corpus_dir)):
            kpath = os.path.join(corpus_dir, kernel)
            if not os.path.isdir(kpath):
                continue
            for tag in sorted(t for t in os.listdir(kpath)
                              if os.path.isdir(os.path.join(kpath, t)) and t.count('_') == 2):
                entries = _read_kernel_compile(os.path.join(kpath, tag))
                if entries:
                    rows.append((f'{kernel} ({tag})', entries))

    def _cell(entries, lane, key):
        e = entries.get(lane)
        if not e or e.get(key) is None:
            return ''
        return f'{e[key]:.1f}'

    os.makedirs(corpus_dir, exist_ok=True)
    for key, fname in (('total', 'compile_total.md'), ('codegen', 'compile_codegen.md'), ('cxx', 'compile_cxx.md')):
        with open(os.path.join(corpus_dir, fname), 'w') as f:
            f.write('| kernel | ' + ' | '.join(lanes) + ' |\n')
            f.write('|---' * (len(lanes) + 1) + '|\n')
            for label, entries in rows:
                f.write(f'| {label} | ' + ' | '.join(_cell(entries, l, key) for l in lanes) + ' |\n')
    print(f'wrote {corpus_dir}/compile_total.md + compile_codegen.md + compile_cxx.md '
          f'({len(rows)} kernel/preset rows, ms)')


# --------------------------------------------------------------------------
# CLI flags shared by every entry-point script.
# --------------------------------------------------------------------------
def add_common_args(ap, default_timeout=3600.0):
    ap.add_argument('--results-dir', default='results', help='results root (default: results)')
    ap.add_argument('--reps', type=int, default=100, help='target repetitions per pipeline (default: 100)')
    ap.add_argument('--only', default=None, help='substring filter on kernel name')
    ap.add_argument('--kernels', default=None, help='comma-separated explicit kernel list (overrides rank slicing)')
    ap.add_argument('--kernels-file', default=None, help='file of kernel names, one per line (overrides rank slicing)')
    ap.add_argument('--force', action='store_true', help='ignore existing results, re-measure from scratch')
    ap.add_argument('--save-sdfg', action='store_true', help='save each pipeline\'s SDFG into the kernel folder')
    ap.add_argument('--save-sdfg-only', action='store_true', help='save canon/fast-canon SDFGs, skip all timing')
    ap.add_argument('--list-kernels', action='store_true', help='print this corpus\'s kernel identifiers and exit')
    ap.add_argument('--tables-only', action='store_true', help='skip measurement, just rebuild the markdown tables')
    ap.add_argument('--timeout',
                    type=float,
                    default=default_timeout,
                    help='per-measurement subprocess timeout, seconds')
    ap.add_argument('--cxx',
                    default=None,
                    help='C++ compiler for DaCe\'s own codegen only -- native lanes each find their own '
                    'vendor compiler independently (default: clang++ on PATH, else g++)')
    return ap


def export_cxx_override(args):
    """Call once in main() after parsing args: Config state doesn't survive a
    spawn boundary, so an explicit --cxx is relayed to every subprocess via
    the environment instead (see configure_dace_process)."""
    if args.cxx:
        os.environ['DACE_PERF_CXX'] = args.cxx


def explicit_kernel_list(args):
    if args.kernels:
        return [k.strip() for k in args.kernels.split(',') if k.strip()]
    if args.kernels_file:
        with open(args.kernels_file) as f:
            return [line.strip() for line in f if line.strip()]
    return None
