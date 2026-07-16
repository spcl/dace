#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Build-and-perf comparison of the experimental *readable* C++ code generator against
the legacy one, over the NPBench + PolyBench corpora, on the daint.alps cluster. CPU only.

This is the npbench/polybench driver of the codegen_buildperf_jobs suite (its tsvc2 /
tsvc2_5 siblings are ``run_buildperf_tsvc2.py`` / ``run_buildperf_tsvc2_5.py``). It is the
readable-vs-legacy driver from ``cpu_codegen_perf_jobs`` reduced to the CPU-only build/perf
scope and extended to record, per (kernel x codegen), the generated-code BUILD metrics inline
in the same TSV row -- so one merged TSV carries codegen time, compile time, runtime,
generated-code size and the readability metrics together, with no second build pass.

Per (kernel x codegen) it measures and records:
  codegen_ms    DaCe SDFG -> C++ generation time (cold).
  compile_ms    C++ -> binary time (cold, a direct compiler invocation timed on its own).
  runtime_ms    best-of-N kernel runtime.
  code_bytes    generated .cpp size on disk.
  nloc          lizard non-comment/non-blank lines of the generated .cpp (LoC).
  max_nesting   max brace-nesting depth (a size-independent readability discriminator).
  tokens        lizard token count (size-sensitive; reported with nloc as its normalizer).
  max_ccn       max cyclomatic complexity (a CONTROL: both lanes emit the same loop
                structure, so it is expected UNCHANGED -- proof readability did not come
                from dropping semantics).
  speedup       legacy_runtime / this_runtime (legacy row == 1.000; experimental row is the
                new-vs-legacy ratio -- the same baseline/candidate convention the sibling
                harness uses). Readability is expected to be perf-neutral, so this is a
                regression guard, not a claimed win.
  correctness   experimental output compared against the LEGACY output; legacy compared
                against the numpy oracle when one exists.

The pipeline under test is the light one the sibling ``performance_regression_jobs`` uses --
``dace + simplify + LoopToMap + MapFusion + (len-1 -> scalar)`` -- and measurement always
happens AFTER it; the ONLY thing that varies between the two lanes is the code generator
(``compiler.cpu.implementation`` = legacy | experimental). Everything else -- kernel
discovery, dataset presets, SDFG build, data init, isolated (crash-safe) measurement,
cold-compile timing, numpy correctness oracle -- is reused verbatim from the sibling harness
(``engine.py`` + ``npbench_polybench_perf.py``), not reinvented. The generated-code metrics
come from this folder's ``readability_metrics.py``, scored on the frame the correctness build
persists (``<build_folder>/src/cpu/<name>.cpp``) -- so no extra build is done to get them.

Two presets:
  S      single-core (OMP_NUM_THREADS=1), the S dataset size -- a quick single-core signal.
  paper  multi-core (full-node threads), the paper dataset sizes -- the writeup numbers.

A per-kernel build/compile/run failure is caught, recorded as an ERROR row, and the sweep
continues.

    python3 run_buildperf.py --preset S --codegen both --corpus npbench --out s.tsv
    python3 run_buildperf.py --preset paper --codegen both --corpus both --out paper.tsv
    python3 run_buildperf.py --preset S --only arc_distance --cxx g++ --out probe.tsv
    python3 run_buildperf.py --list-kernels --corpus polybench --preset S
"""
import os

# MPI/OpenMP anti-hang defaults BEFORE importing dace / the sibling harness (a dace
# script otherwise blocks on MPI_Init and the sweep looks hung). setdefault so an
# explicit value (a real srun launch, or the --preset thread override in main) wins.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import copy
import csv
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# This folder's readability_metrics.py is authoritative, so keep this dir first on sys.path.
# The sibling performance_regression_jobs supplies the shared infra we deliberately do NOT
# duplicate (engine.py = pipelines, isolated timing, cold-compile timing; and
# npbench_polybench_perf.py = kernel discovery, dataset presets, SDFG build, data init, numpy
# oracle, output collection + comparison). It is APPENDED so it can never shadow this dir's
# local copies. Both paths derive from this file's own location -- no hardcoded absolute paths.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PERF_JOBS_DIR = os.path.join(REPO_ROOT, 'performance_regression_jobs')
if PERF_JOBS_DIR not in sys.path:
    sys.path.append(PERF_JOBS_DIR)

import engine
import npbench_polybench_perf as base
import readability_metrics

DEVICE = 'cpu'  # this suite is CPU-only (both codegens, both host compilers, no GPU)
CODEGENS = ('legacy', 'experimental')

#: TSV columns -- the EXACT header plot_buildperf.py reads (order matters). codegen_ms (DaCe
#: C++ code generation) and compile_ms (the cxx compiler, on its own) are kept SEPARATE so a
#: stacked plot can show each. The five build/size/readability columns (code_bytes .. max_ccn)
#: are recorded inline per row -- see readability_metrics.FRAME_METRIC_FIELDS -- so one merged
#: TSV carries build cost, size and readability together. `corpus` is the kernel's
#: npbench/polybench membership; `threads` the OMP width the row was measured at; `cxx` the
#: HOST C++ compiler it was built with (g++ / clang++); `phase` which sweep phase produced it
#: (single_core / multi_core), so the merged TSV stays groupable; `error` carries a failure
#: message so an ERROR row is self-explanatory.
TSV_FIELDS = [
    'kernel', 'corpus', 'codegen', 'preset', 'threads', 'cxx', 'phase', 'codegen_ms', 'compile_ms', 'runtime_ms',
    'code_bytes', 'nloc', 'max_nesting', 'tokens', 'max_ccn', 'speedup', 'correctness', 'error'
]


# --------------------------------------------------------------------------
# Codegen selection + SDFG construction (shared by the isolated jobs below).
# --------------------------------------------------------------------------
def set_implementation(codegen):
    """Select the CPU code generator for THIS process via the config knob the readable
    codegen reads (compiler.cpu.implementation). Also mirrored into the environment so
    any nested spawn inherits the same selection."""
    import dace
    value = codegen
    if codegen == 'experimental':
        # The readable generator's flag value was renamed 'experimental' -> 'experimental_readable';
        # probe which this dace build recognizes so the readable path activates against either tree.
        from dace.codegen.targets import cpp
        value = 'experimental_readable'
        dace.Config.set('compiler', 'cpu', 'implementation', value=value)
        if not cpp.readable_cpu_codegen_active():
            value = 'experimental'
    dace.Config.set('compiler', 'cpu', 'implementation', value=value)
    os.environ['DACE_compiler_cpu_implementation'] = value


def sdfg_name(name, preset, codegen):
    """Stable, unique per (kernel, preset, codegen) -- also its cache key
    (configure_dace_process sets cache='name'), so the legacy and experimental builds of
    one kernel never collide on a build folder, and the persisted frame each build leaves
    is uniquely addressable for the inline readability score."""
    return f'buildperf_{name}_{preset}_{codegen}'


def pipelined_sdfg(program, name):
    """Parse `program` to an SDFG and run the pipeline under test on it:
    simplify + LoopToMap + MapFusion + length-1-array -> scalar. The scalarization
    (transient single-element scratch, incl. (1, 1) MapFusion buffers) runs for BOTH code
    generators so the comparison isolates the code generator."""
    from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    sdfg = engine.pipeline_parallel(sdfg, DEVICE)
    ConvertLengthOneArraysToScalars(transient_only=True).apply_pass(sdfg, {})
    return sdfg


def run_dace_output(sdfg, info, arrays, params):
    """Compile + run once and return the kernel's output arrays as host numpy. CPU only, so
    no host<->device marshalling. sdfg.compile() persists the generated frame under
    sdfg.build_folder, which frame_metrics_for then scores (the readable/legacy source is
    host-compiler-independent, so one scored frame per (kernel, codegen) covers both cxx)."""
    kwargs = base._dace_call_kwargs(sdfg, arrays, params)
    ret = sdfg.compile()(**kwargs)
    return base._collect_outputs(info['output_args'], ret, kwargs)


def frame_metrics_for(sdfg):
    """Build/size/readability metrics of the frame `sdfg` just persisted (a plain
    sdfg.compile(), NOT the ``_timed`` instrumented rename), addressed by name via
    cache='name'. Never raises."""
    path = readability_metrics.frame_source_path(sdfg.build_folder, sdfg.name)
    return readability_metrics.frame_metrics(path)


# --------------------------------------------------------------------------
# Isolated jobs (run inside engine.run_isolated's spawned subprocess: a segfault or
# exception in codegen/compile/run only fails this one job, never the sweep).
# --------------------------------------------------------------------------
def perf_job(name, preset, codegen, reps, timeout):
    """Build+pipeline under `codegen`, then measure the cold codegen and cxx-compile times
    (separately) and the best-of-`reps` runtime. Returns
    {'codegen_ms': float, 'compile_ms': float, 'runtime_ms': float|None}."""
    engine.configure_dace_process()
    set_implementation(codegen)
    info = base.load_bench_info(name)
    params = info['parameters'][preset]
    program, arrays, params = base.build_program_and_data(name, info, params)
    sdfg = pipelined_sdfg(program, sdfg_name(name, preset, codegen))
    codegen_ms, compile_ms = engine.compile_sdfg_timed(sdfg)  # (codegen_ms, cxx_ms), a real cold compile
    kwargs = base._dace_call_kwargs(sdfg, arrays, params)
    times = engine.time_sdfg(sdfg, kwargs, reps, time_budget_s=0.8 * timeout)
    return dict(codegen_ms=codegen_ms, compile_ms=compile_ms, runtime_ms=(min(times) if times else None))


def correctness_job(name, preset, want_experimental):
    """Legacy is the reference. Build+run legacy once (comparing it against the numpy oracle
    when one is importable -> the legacy pass/fail); if `want_experimental`, build+run
    experimental on the SAME inputs and compare its output against legacy's -- the
    'vs legacy output' check the spec asks for. Both SDFGs are deepcopies of one parsed base
    graph so the two lanes see identical inputs even for a randomly-initialized kernel.

    Also scores each built frame's generated-code metrics inline (readability_metrics), so the
    build metrics ride out on the same isolated pass that already compiled each lane -- no
    extra build. Returns {'legacy': bool|None, 'experimental': bool|None, 'error': str,
    'metrics': {codegen: {code_bytes, nloc, max_nesting, tokens, max_ccn}}}."""
    engine.configure_dace_process()
    info = base.load_bench_info(name)
    params = info['parameters'][preset]
    program, arrays, params = base.build_program_and_data(name, info, params)
    base_sdfg = program.to_sdfg(simplify=True)

    out = dict(legacy=None, experimental=None, error='', metrics={})

    # numpy oracle is optional -- a few kernels have no importable reference module.
    oracle = None
    try:
        oracle = base._run_numpy(info, arrays, params)
    except Exception:
        oracle = None

    set_implementation('legacy')
    legacy = pipelined_sdfg_copy(base_sdfg, sdfg_name(name, preset, 'legacy'))
    out_legacy = run_dace_output(legacy, info, arrays, params)
    out['metrics']['legacy'] = frame_metrics_for(legacy)
    out['legacy'] = base._compare(oracle, out_legacy) if oracle is not None else True

    if want_experimental:
        set_implementation('experimental')
        try:
            experimental = pipelined_sdfg_copy(base_sdfg, sdfg_name(name, preset, 'experimental'))
            out_exp = run_dace_output(experimental, info, arrays, params)
            out['metrics']['experimental'] = frame_metrics_for(experimental)
            out['experimental'] = base._compare(out_legacy, out_exp)  # vs legacy output
        except Exception as e:
            out['experimental'] = False
            out['error'] = f'{type(e).__name__}: {e}'
    return out


def pipelined_sdfg_copy(base_sdfg, name):
    """Run the pipeline under test on a fresh deepcopy of `base_sdfg` named `name` (so both
    lanes start from the identically parsed graph)."""
    sdfg = copy.deepcopy(base_sdfg)
    sdfg.name = name
    sdfg = engine.pipeline_parallel(sdfg, DEVICE)
    from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
    ConvertLengthOneArraysToScalars(transient_only=True).apply_pass(sdfg, {})
    return sdfg


# --------------------------------------------------------------------------
# Kernel selection (parent process).
# --------------------------------------------------------------------------
def select_kernels(corpus, preset, only, explicit):
    """Kernels (from the vendored bench_info) that (a) have a local corpus port,
    (b) have the requested dataset preset, and (c) fall in the requested corpus.
    npbench vs polybench membership is decided by the sibling harness's own
    _is_polybench (local-corpus membership over the declared upstream path)."""
    names = sorted(f[:-len('.json')] for f in os.listdir(base.BENCH_INFO_DIR) if f.endswith('.json'))
    if explicit is not None:
        names = [n for n in names if n in explicit]
    if only:
        names = [n for n in names if only in n]
    out = []
    for n in names:
        info = base.load_bench_info(n)
        if preset not in info.get('parameters', {}):
            continue
        if not base.kernel_exists(n):
            continue
        poly = base._is_polybench(n, info)
        if corpus == 'npbench' and poly:
            continue
        if corpus == 'polybench' and not poly:
            continue
        out.append(n)
    return out


def kernel_corpus(name):
    """The kernel's corpus membership ('npbench' | 'polybench') for the TSV `corpus` column,
    decided the same way select_kernels partitions -- base._is_polybench over local-corpus
    membership. Recorded per row (not the --corpus filter) so a 'both' sweep stays groupable."""
    return 'polybench' if base._is_polybench(name, base.load_bench_info(name)) else 'npbench'


def resolve_threads(preset):
    """OMP width for the preset: S is always single-core; paper honors an OMP_NUM_THREADS
    already exported (the sbatch sets it to the node width), else falls back to cpu count."""
    if preset == 'S':
        return 1
    exported = os.environ.get('OMP_NUM_THREADS')
    if exported and exported.isdigit() and int(exported) > 1:
        return int(exported)
    return os.cpu_count() or 1


def resolve_cxx_label(explicit):
    """The `cxx` column: the HOST C++ compiler DaCe's codegen is built with, as a bare name
    (g++ / clang++). Mirrors what engine.configure_dace_process actually honors -- an explicit
    --cxx, else the exported DACE_PERF_CXX -- so the label can never disagree with the compiler
    that ran. 'default' when neither is set: engine then autodetects (clang++ on PATH, else g++)."""
    cxx = explicit or os.environ.get('DACE_PERF_CXX', '')
    return os.path.basename(cxx) if cxx else 'default'


def resolve_phase(explicit, threads):
    """The `phase` column: which sweep phase produced the row. An explicit --phase (what the
    sbatch passes) wins; otherwise derive it from the OMP width, so a standalone invocation
    still labels itself correctly: single OMP thread -> 'single_core', else 'multi_core'."""
    if explicit:
        return explicit
    return 'single_core' if threads == 1 else 'multi_core'


def codegen_list(codegen):
    return list(CODEGENS) if codegen == 'both' else [codegen]


# --------------------------------------------------------------------------
# Per-kernel driver + TSV assembly.
# --------------------------------------------------------------------------
def correctness_cell(codegen, corr_ok, corr, perf_ok):
    """Render the correctness column for one (codegen) row from the isolated results."""
    if not perf_ok or not corr_ok:
        return 'ERROR'  # the build/compile/run failed; correctness is moot (error carries why)
    value = corr.get(codegen)
    if value is True:
        return 'pass'
    if value is False:
        return 'fail'
    return 'unknown'


def metric_cell(corr_ok, corr, codegen, field):
    """A build/size/readability cell for one (codegen) row from the correctness pass's inline
    metrics -- blank when the metrics are missing (correctness crashed) or the metric is None
    (e.g. tokens/max_ccn without lizard)."""
    if not corr_ok:
        return ''
    value = corr.get('metrics', {}).get(codegen, {}).get(field)
    return '' if value is None else str(value)


def process_kernel(writer, out_handle, name, args, threads, cxx, phase):
    preset, reps, timeout = args.preset, args.reps, args.timeout
    corpus = kernel_corpus(name)
    codegens = codegen_list(args.codegen)

    # One correctness pass covers both lanes (legacy is always built as the reference) and
    # also carries each built frame's inline generated-code metrics.
    corr_ok, corr = engine.run_isolated(correctness_job, (name, preset, 'experimental' in codegens), timeout=timeout)
    if not corr_ok:
        corr = {}

    # Time each requested lane independently (crash-isolated).
    perf = {}
    for codegen in codegens:
        ok, payload = engine.run_isolated(perf_job, (name, preset, codegen, reps, timeout), timeout=timeout)
        perf[codegen] = (ok, payload)

    legacy_rt = None
    if 'legacy' in perf and perf['legacy'][0]:
        legacy_rt = perf['legacy'][1]['runtime_ms']

    for codegen in codegens:
        ok, payload = perf[codegen]
        codegen_ms = f"{payload['codegen_ms']:.3f}" if ok and payload.get('codegen_ms') is not None else ''
        compile_ms = f"{payload['compile_ms']:.3f}" if ok and payload.get('compile_ms') is not None else ''
        runtime_ms = f"{payload['runtime_ms']:.6f}" if ok and payload.get('runtime_ms') is not None else ''
        speedup = ''
        row_rt = payload['runtime_ms'] if ok else None
        if legacy_rt and row_rt:
            speedup = f'{legacy_rt / row_rt:.3f}'  # baseline(legacy) / candidate -- legacy row == 1.000
        error = ''
        if not ok:
            error = str(payload)
        elif codegen == 'experimental' and corr_ok and corr.get('error'):
            error = corr['error']
        row = dict(kernel=name,
                   corpus=corpus,
                   codegen=codegen,
                   preset=preset,
                   threads=threads,
                   cxx=cxx,
                   phase=phase,
                   codegen_ms=codegen_ms,
                   compile_ms=compile_ms,
                   runtime_ms=runtime_ms,
                   code_bytes=metric_cell(corr_ok, corr, codegen, 'code_bytes'),
                   nloc=metric_cell(corr_ok, corr, codegen, 'nloc'),
                   max_nesting=metric_cell(corr_ok, corr, codegen, 'max_nesting'),
                   tokens=metric_cell(corr_ok, corr, codegen, 'tokens'),
                   max_ccn=metric_cell(corr_ok, corr, codegen, 'max_ccn'),
                   speedup=speedup,
                   correctness=correctness_cell(codegen, corr_ok, corr, ok),
                   error=error)
        writer.writerow(row)
        out_handle.flush()  # keep partial results on disk if a later kernel crashes the sweep
        print(f"[{name}/{corpus}/{preset}/{phase}/{cxx}] {codegen}: codegen={codegen_ms or '-'}ms "
              f"compile={compile_ms or '-'}ms runtime={runtime_ms or '-'}ms nloc={row['nloc'] or '-'} "
              f"nest={row['max_nesting'] or '-'} speedup={speedup or '-'} correct={row['correctness']}" +
              (f' ({error})' if error else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset',
                    choices=('S', 'paper'),
                    default='S',
                    help='S = single-core, S dataset (quick signal); paper = full-node threads, paper dataset')
    ap.add_argument('--codegen',
                    choices=('legacy', 'experimental', 'both'),
                    default='both',
                    help='which code generator(s) to measure')
    ap.add_argument('--corpus', choices=('npbench', 'polybench', 'both'), default='both', help='which corpus to sweep')
    ap.add_argument('--out', default='results.tsv', help='TSV output path (default: results.tsv)')
    ap.add_argument('--only', default=None, help='substring filter on kernel name')
    ap.add_argument('--kernels', default=None, help='comma-separated explicit kernel list (overrides --corpus/--only)')
    ap.add_argument('--reps', type=int, default=10, help='timed repetitions per lane (best-of; default: 10)')
    ap.add_argument('--timeout', type=float, default=900.0, help='per-measurement subprocess timeout, seconds')
    ap.add_argument('--list-kernels', action='store_true', help='print the selected kernels and exit')
    # Host C++ compiler + phase label. Both are recorded per row so the phases of one merged
    # sweep (submit_codegen_buildperf.sbatch) stay groupable; --cxx additionally SELECTS the
    # compiler, relayed to every measurement subprocess via DACE_PERF_CXX (Config state does
    # not survive a spawn boundary -- see engine.configure_dace_process).
    ap.add_argument('--cxx',
                    default=None,
                    help="host C++ compiler for DaCe's codegen, also recorded in the `cxx` column "
                    '(default: $DACE_PERF_CXX, else engine autodetection)')
    ap.add_argument('--phase',
                    choices=('single_core', 'multi_core'),
                    default=None,
                    help='value for the `phase` column (default: derived from the OMP width)')
    # Multi-rank kernel partitioning: one MPI rank per Grace socket, each measuring a
    # disjoint slice ``kernels[rank::num_ranks]``. Defaults come from the SLURM launch
    # (``SLURM_PROCID`` / ``SLURM_NTASKS``), so a plain single-process invocation stays
    # rank 0 of 1 (the whole corpus). Each rank writes its own ``<out>.rank<r>`` TSV.
    ap.add_argument('--rank',
                    type=int,
                    default=int(os.environ.get('SLURM_PROCID', '0')),
                    help='this rank index (default: $SLURM_PROCID, else 0)')
    ap.add_argument('--num-ranks',
                    type=int,
                    default=int(os.environ.get('SLURM_NTASKS', '1')),
                    help='total ranks partitioning the kernels (default: $SLURM_NTASKS, else 1)')
    args = ap.parse_args()

    # An explicit --cxx is relayed to every measurement subprocess through the environment
    # (the same knob the sbatch exports); engine.configure_dace_process turns it into
    # compiler.cpu.executable inside each spawn.
    if args.cxx:
        os.environ['DACE_PERF_CXX'] = args.cxx

    explicit = [k.strip() for k in args.kernels.split(',') if k.strip()] if args.kernels else None
    kernels = select_kernels(args.corpus, args.preset, args.only, explicit)

    # Split the kernels across ranks (round-robin balances uneven per-kernel cost
    # better than contiguous blocks) and give each rank a distinct output file.
    if args.num_ranks > 1:
        kernels = kernels[args.rank::args.num_ranks]
        stem, ext = os.path.splitext(args.out)
        args.out = f'{stem}.rank{args.rank}{ext or ".tsv"}'

    if args.list_kernels:
        print('\n'.join(kernels))
        return

    threads = resolve_threads(args.preset)
    os.environ['OMP_NUM_THREADS'] = str(threads)  # inherited by every spawned measurement subprocess
    cxx = resolve_cxx_label(args.cxx)
    phase = resolve_phase(args.phase, threads)

    rank_note = f' rank={args.rank}/{args.num_ranks}' if args.num_ranks > 1 else ''
    print(f'preset={args.preset} codegen={args.codegen} corpus={args.corpus} '
          f'threads={threads} cxx={cxx} phase={phase} reps={args.reps}{rank_note}')
    print(f'{len(kernels)} kernel(s) -> {args.out}')

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter='\t')
        writer.writeheader()
        f.flush()
        for name in kernels:
            process_kernel(writer, f, name, args, threads, cxx, phase)

    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
