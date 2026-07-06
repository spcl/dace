"""Shared engine for the performance_regression_jobs scripts: plain functions only.

Reuses the exact patterns already established in dace/optimization/utils.py:
world-rank detection, subprocess-isolated measurement (so a segfaulting SDFG or
native binary never kills the whole sweep), and instrumentation-report-based
timing (sdfg.instrument + get_latest_report instead of wall-clock wrapping).
"""
import copy
import csv
import json
import multiprocessing as mp
import os
import re
import shutil
import socket
import time

_RESULTS_CSV = 'results.csv'
_STATUS_CSV = 'status.csv'


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
    """Resolve the C++ compiler used for DaCe's own codegen (every native
    lane picks its own vendor's compiler independently -- see
    native_harness.compile_lane -- and never consults this).

    `explicit` (--cxx, or DACE_PERF_CXX inherited from it) may be an absolute
    path or a bare executable name -- shutil.which resolves either (a path
    containing a separator is checked directly for executability, a bare
    name is searched on PATH). An explicit value that doesn't resolve raises
    immediately rather than silently substituting a different compiler than
    what was asked for.

    Absent an override, picks clang++ if it's on PATH, else g++ (plain PATH
    lookup -- trusts whatever module/spack/venv setup already put the
    intended version on PATH under its bare name). Returns None if neither is
    found (compiler config is then left at DaCe's own default)."""
    if explicit:
        resolved = shutil.which(explicit)
        if not resolved:
            raise FileNotFoundError(f'--cxx {explicit!r}: not found or not executable')
        return resolved
    import native_harness as nh
    return nh.find_best_cpp_compiler()


def configure_dace_process():
    """Per-process DaCe setup: must run inside every measurement subprocess
    (Config state doesn't survive across a spawn boundary), not just once at
    the parent script's startup.

    - cache='unique' keys every SDFG's build folder on
      `{sdfg.name}_{md5(pid)}` (dace/sdfg/sdfg.py's `build_folder` property)
      -- distinct ranks are distinct OS processes, so distinct PIDs, so no
      two ranks (or two pipelines within one rank, since sdfg.name also
      differs per pipeline) ever share a build folder.
    - compiler.cpu.executable is set to DACE_PERF_CXX (exported into the
      environment by the parent from --cxx, and inherited by every spawned
      child process) or, absent that, pick_cxx_compiler()'s own LLVM-first
      autodetection.
    - If that resolved compiler needs it (native_harness.needs_gcc_install_dir:
      clang++ or Intel's LLVM-based icpx/icpc), --gcc-install-dir is appended
      to compiler.cpu.args: DaCe's own CMake-driven build never adds this by
      itself, so on a machine with several GCC versions installed side by
      side (the same situation native_harness.compile_lane's clang/icpx lanes
      handle) it can otherwise pick a GCC install with no matching libstdc++
      headers and fail with 'cannot find -lstdc++' -- observed concretely
      when clang++ auto-wins over a GCC whose newest version is C-only (no
      libstdc++-dev installed for it). Idempotent (checks the flag isn't
      already present) since this runs once per pipeline build and can be
      called more than once in the same process (e.g. _check_dace_job builds
      both a reference and a candidate SDFG).
    """
    import dace
    dace.Config.set('cache', value='unique')
    cxx = pick_cxx_compiler(os.environ.get('DACE_PERF_CXX'))
    if cxx:
        dace.Config.set('compiler', 'cpu', 'executable', value=cxx)
        import native_harness as nh
        if nh.needs_gcc_install_dir(cxx):
            gcc_dir = nh.find_gcc_install_dir()
            if gcc_dir:
                flag = f'--gcc-install-dir={gcc_dir}'
                args = dace.Config.get('compiler', 'cpu', 'args')
                if flag not in args:
                    dace.Config.set('compiler', 'cpu', 'args', value=f'{args} {flag}')


def cleanup_build_folder(sdfg):
    """'unique' mode never reuses or reclaims a build folder (by design --
    that's what makes it collision-free), so a long sweep leaves one
    permanent directory per compiled SDFG unless the caller removes it
    itself once done with it. Call at the end of each job, after any timing
    the caller needed the compiled binary for."""
    import shutil
    shutil.rmtree(sdfg.build_folder, ignore_errors=True)


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
    with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
        csdfg = sdfg.compile()
        for _ in range(warmup + reps):
            csdfg(**call_kwargs)
        csdfg.finalize()
    return _flatten_durations(sdfg.get_latest_report().durations)[warmup:]


# --------------------------------------------------------------------------
# Shared pipelines: existing passes/transformations only, nothing new.
# "-seq" lanes are never a separate pipeline -- always make_sequential(par result).
# --------------------------------------------------------------------------
def pipeline_baseline(sdfg):
    from dace.transformation.interstate import LoopToMap
    from dace.transformation.dataflow import MapFusionHorizontal, MapFusionVertical
    from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
    sdfg.simplify(validate=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    PatternMatchAndApplyRepeated([MapFusionVertical(), MapFusionHorizontal()]).apply_pass(sdfg, {})
    return sdfg


def pipeline_canon(sdfg):
    from dace.transformation.passes.canonicalize import canonicalize
    return canonicalize(sdfg, validate=True)


def pipeline_fast_canon(sdfg):
    from dace.transformation.passes.canonicalize import canonicalize
    return canonicalize(sdfg, validate=True, fast=True)


def pipeline_auto_opt(sdfg):
    import dace
    from dace.transformation.auto.auto_optimize import auto_optimize
    return auto_optimize(sdfg, dace.DeviceType.CPU)


#: The 4 DaCe-side comparison points: two baselines (plain simplify+
#: loop2map+mapfusion, and DaCe's own auto_optimize) vs. canonicalize and
#: canonicalize(fast=True).
PIPELINES = {
    'baseline': pipeline_baseline,
    'auto-opt': pipeline_auto_opt,
    'canon': pipeline_canon,
    'fast-canon': pipeline_fast_canon,
}


def make_sequential(sdfg):
    """Deep-copy + force every map to Sequential (dace/cli/daceprof.py:222-232's logic)."""
    import dace
    s = copy.deepcopy(sdfg)
    for sd in s.all_sdfgs_recursive():
        sd.openmp_sections = False
    for n, _ in s.all_nodes_recursive():
        if isinstance(n, dace.nodes.EntryNode) and getattr(n, 'schedule', False) in (
                dace.ScheduleType.CPU_Multicore, dace.ScheduleType.CPU_Persistent, dace.ScheduleType.Default):
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
    """`<compiler>_<hostname>` -- namespace for anything that depends on
    DaCe's own C++ codegen compiler (--cxx / DACE_PERF_CXX): different
    compilers (a different --cxx, a different node's autodetected
    latest-clang/gcc) produce DaCe-lane timings that are not comparable, and
    must never share (and silently corrupt) the same results.csv/status.csv.
    Keying on compiler+hostname means a multi-node sweep where nodes differ
    in their default toolchain, or separate --cxx runs against the same
    --results-dir, land in distinct result folders instead of conflicting.

    Only the DaCe-lane (baseline/canon/fast-canon) results use this tag --
    see native_result_tag() for the native lanes, which each pick their own
    vendor's compiler independently of --cxx and so must not be needlessly
    invalidated every time --cxx changes. Computed once per process and
    cached -- it's stable for the process's lifetime and
    pick_cxx_compiler()'s autodetection does a PATH scan."""
    global _COMPILER_HOST_TAG_CACHE
    if _COMPILER_HOST_TAG_CACHE is None:
        cxx = pick_cxx_compiler(os.environ.get('DACE_PERF_CXX'))
        compiler_slug = _slug(os.path.basename(cxx)) if cxx else 'unknown-cxx'
        _COMPILER_HOST_TAG_CACHE = f'{compiler_slug}_{_slug(socket.gethostname())}'
    return _COMPILER_HOST_TAG_CACHE


def host_tag():
    """`<hostname>` -- namespace for native lanes (native-gcc, native-clang,
    native-icpx, native-nvc and their autopar variants): each finds its own
    vendor's compiler independently (native_harness.compile_lane), never
    depending on --cxx, so unlike compiler_host_tag() this must NOT include
    the DaCe codegen compiler -- doing so would force every native lane's
    100-rep measurement to be redone from scratch every time --cxx changes,
    even though nothing about those lanes' own compilers changed. Hostname
    alone is still needed: two different hosts (possibly with different
    installed toolchain versions) sharing one --results-dir must not race on
    or clobber the same compiled .so (see native_build_dir)."""
    return _slug(socket.gethostname())


def result_tag(preset):
    """`<compiler>_<hostname>_<preset>` -- folder DaCe-lane results land in.

    `preset` is slugged like every other component: write_tables() tells a
    DaCe-tag folder (2 underscores) apart from a native-tag folder (1
    underscore) by counting underscores, which only works if every component
    -- preset included -- is guaranteed underscore-free."""
    return f'{compiler_host_tag()}_{_slug(str(preset))}'


def native_result_tag(preset):
    """`<hostname>_<preset>` -- folder native-lane results land in (see
    host_tag() for why this excludes the DaCe compiler, and result_tag() for
    why preset is slugged)."""
    return f'{host_tag()}_{_slug(str(preset))}'


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
    """(pipeline -> {'correct': bool, 'median_ms': float or None}) for one kernel/preset dir."""
    out = {}
    for row in [] if not os.path.exists(os.path.join(kdir, _STATUS_CSV)) else csv.DictReader(
            open(os.path.join(kdir, _STATUS_CSV), newline='')):
        out[row['pipeline']] = dict(correct=row['correct'] == 'True', median_ms=None)
    times = {}
    if os.path.exists(os.path.join(kdir, _RESULTS_CSV)):
        for row in csv.DictReader(open(os.path.join(kdir, _RESULTS_CSV), newline='')):
            times.setdefault(row['pipeline'], []).append(float(row['time_ms']))
    for pipeline, ts in times.items():
        out.setdefault(pipeline, dict(correct=True, median_ms=None))['median_ms'] = _median(ts)
    return out


def write_tables(results_root, corpus, lanes, baseline_label):
    """Scan <results_root>/<corpus>/**/ and write correctness.md + speedup.md.

    Each kernel folder can hold two kinds of tag subfolder: result_tag()'s
    '<compiler>_<hostname>_<preset>' (DaCe lanes, 2 underscores) and
    native_result_tag()'s '<hostname>_<preset>' (native lanes, 1 underscore --
    _slug() never emits a literal underscore, so this count is unambiguous).
    A DaCe-tag row is merged with its matching native-tag folder (same host
    + preset) so one row shows every lane measured for that host/preset,
    regardless of how many different --cxx values were used to produce the
    DaCe-tag folders (native results are shared across all of them)."""
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
        if not entry.get('median_ms') or not base.get('median_ms'):
            return ''
        return f"{base['median_ms'] / entry['median_ms']:.2f}x"

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


# --------------------------------------------------------------------------
# CLI flags shared by every entry-point script.
# --------------------------------------------------------------------------
def add_common_args(ap):
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
    ap.add_argument('--timeout', type=float, default=120.0, help='per-measurement subprocess timeout, seconds')
    ap.add_argument('--cxx', default=None,
                     help='C++ compiler for DaCe\'s own codegen only -- native lanes each find their own '
                          'vendor compiler independently (default: clang++ on PATH, else g++)')
    return ap


def export_cxx_override(args):
    """Call once in main(), right after parsing args: an explicit --cxx must
    reach every spawned measurement subprocess, and Config state (which is
    where configure_dace_process() would otherwise set it) doesn't survive
    a spawn boundary -- the process environment does, so that's the channel."""
    if args.cxx:
        os.environ['DACE_PERF_CXX'] = args.cxx


def explicit_kernel_list(args):
    if args.kernels:
        return [k.strip() for k in args.kernels.split(',') if k.strip()]
    if args.kernels_file:
        with open(args.kernels_file) as f:
            return [line.strip() for line in f if line.strip()]
    return None
