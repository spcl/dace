#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Build-and-perf comparison of the experimental *readable* C++ code generator against the
legacy one over the TSVC2.5 corpus, on the daint.alps cluster -- the tsvc2_5 sibling of
``run_buildperf.py`` (which covers NPBench+PolyBench). CPU only. Same structure, same TSV,
same crash-isolated measurement, same inline generated-code metrics; the only differences
are the corpus adapter (the vendored ``tsvc2_5_perf`` + ``tsvc_2_5_corpus`` harness, for
kernel discovery, deterministic input allocation and per-kernel symbolic sizing) and the
correctness oracle (TSVC has no numpy reference, so the LEGACY output is the reference the
experimental output is compared against, via ``engine.arrays_close``).

The pipeline under test is identical to run_buildperf.py's:
``dace + simplify + LoopToMap + MapFusion + (len-1 -> scalar)``; the ONLY thing that varies
between the two lanes is ``compiler.cpu.implementation`` (legacy vs experimental). Readability
is expected to be perf-neutral, so the runtime comparison is a regression guard, not a claimed
win -- the win is in generated size / readability.

Per (kernel x codegen) it records the SAME columns run_buildperf.py does: codegen_ms
(DaCe C++ codegen), compile_ms (the cxx compiler on its own), runtime_ms (best-of-N),
code_bytes / nloc / max_nesting / tokens / max_ccn (generated-code size + readability, scored
inline on the frame the correctness build persists), speedup, correctness, error.

Two presets:
  S      single-core, the corpus's stock (small) SIZES defaults -- a quick single-core signal.
  paper  full-node threads, the ~2GB working-set sizing tsvc2_5_perf.py measures at
         (base.size_scale_for_kernel, which scales only the LEN_* symbols).

A per-kernel build/compile/run failure is caught, recorded as an ERROR row, and the sweep
continues.

    python3 run_buildperf_tsvc2_5.py --preset S --codegen both --out tsvc2_5.tsv
    python3 run_buildperf_tsvc2_5.py --preset paper --only heat3d --out probe.tsv
    python3 run_buildperf_tsvc2_5.py --list-kernels
"""
import os

# MPI/OpenMP anti-hang defaults BEFORE importing dace / the vendored harness (a dace
# script otherwise blocks on MPI_Init and the sweep looks hung). setdefault so an
# explicit value (a real srun launch, or the --preset thread override in main) wins.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import csv
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PERF_JOBS_DIR = os.path.join(REPO_ROOT, 'performance_regression_jobs')
# The tsvc harness vendored into THIS folder (tsvc2_5_perf, tsvc_2_5_corpus) + readability_metrics
# are authoritative, so keep this dir ahead of the sibling on sys.path. The sibling
# performance_regression_jobs supplies the shared infra we deliberately do NOT duplicate
# (engine.py = pipelines, isolated timing, cold-compile timing); it is APPENDED so it can
# never shadow this dir's local copies. Both paths derive from this file's own location.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if PERF_JOBS_DIR not in sys.path:
    sys.path.append(PERF_JOBS_DIR)

import engine
import readability_metrics
import tsvc_2_5_corpus as tsvc25
import tsvc2_5_perf as base

CORPUS = base.CORPUS  # 'tsvc2_5'
DEVICE = 'cpu'        # the TSVC corpus is CPU-only; the readable comparison is CPU codegen
CODEGENS = ('legacy', 'experimental')

#: TSV columns -- the EXACT header plot_buildperf.py reads (order matters), identical to
#: run_buildperf.py. codegen_ms (DaCe C++ code generation) and compile_ms (the cxx compiler,
#: on its own) are kept SEPARATE; code_bytes .. max_ccn are the inline generated-code metrics
#: (readability_metrics.FRAME_METRIC_FIELDS); `corpus` is literally "tsvc2_5".
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
    (configure_dace_process sets cache='name'); a valid identifier (no hyphens)."""
    return f'buildperf_{CORPUS}_{name}_{preset}_{codegen}'


def pipelined_sdfg(program, name):
    """Parse `program` to an SDFG and run the pipeline under test on it:
    simplify + LoopToMap + MapFusion + length-1-array -> scalar. The scalarization runs
    for BOTH code generators so the comparison isolates the code generator."""
    from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    sdfg = engine.pipeline_parallel(sdfg, DEVICE)
    ConvertLengthOneArraysToScalars(single_element=True, transient_only=True).apply_pass(sdfg, {})
    return sdfg


def sizes_for(program, preset):
    """The symbol-size dict for the preset: paper = base.size_scale_for_kernel (the ~2GB
    target, scaling only the LEN_* symbols); S = the corpus's stock small SIZES defaults."""
    if preset == 'paper':
        return base.size_scale_for_kernel(program)
    return dict(tsvc25.SIZES)


def call_kwargs(arrays, scalars, sym):
    """The SDFG call dict: fresh array copies (each run mutates in place) + scalar
    parameters + the free-symbol values the built SDFG needs."""
    return {**{n: a.copy() for n, a in arrays.items()}, **scalars, **sym}


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
def perf_job(name, sizes, preset, codegen, reps, timeout):
    """Build+pipeline under `codegen`, then measure the cold codegen and cxx-compile
    times (separately) and the best-of-`reps` runtime. Returns
    {'codegen_ms': float, 'compile_ms': float, 'runtime_ms': float|None}."""
    engine.configure_dace_process()
    set_implementation(codegen)
    program, arrays, scalars = base._inputs(name, sizes)
    sdfg = pipelined_sdfg(program, sdfg_name(name, preset, codegen))
    sym = base._symbol_values(sdfg, sizes)  # free-symbol values on the FINAL (post-pipeline) SDFG
    codegen_ms, compile_ms = engine.compile_sdfg_timed(sdfg)
    times = engine.time_sdfg(sdfg, call_kwargs(arrays, scalars, sym), reps, time_budget_s=0.8 * timeout)
    return dict(codegen_ms=codegen_ms, compile_ms=compile_ms, runtime_ms=(min(times) if times else None))


def correctness_job(name, sizes, preset, want_experimental):
    """Legacy is the reference (TSVC has no numpy oracle). Build+run legacy once; if
    `want_experimental`, build+run experimental on IDENTICAL inputs (base._inputs reseeds
    deterministically from the same recipe) and compare its mutated arrays against legacy's
    via engine.arrays_close. Also scores each built frame's generated-code metrics inline, so
    the build metrics ride out on the pass that already compiled each lane -- no extra build.
    Returns {'legacy': bool|None, 'experimental': bool|None, 'error': str,
    'metrics': {codegen: {code_bytes, nloc, max_nesting, tokens, max_ccn}}}."""
    engine.configure_dace_process()
    out = dict(legacy=None, experimental=None, error='', metrics={})

    set_implementation('legacy')
    program, arrays, scalars = base._inputs(name, sizes)
    legacy = pipelined_sdfg(program, sdfg_name(name, preset, 'legacy'))
    legacy_sym = base._symbol_values(legacy, sizes)
    legacy_call = call_kwargs(arrays, scalars, legacy_sym)
    legacy.compile()(**legacy_call)
    out['metrics']['legacy'] = frame_metrics_for(legacy)
    out['legacy'] = True  # no numpy oracle for TSVC: the legacy output IS the reference

    if want_experimental:
        set_implementation('experimental')
        try:
            program2, arrays2, scalars2 = base._inputs(name, sizes)  # identical inputs (deterministic seed)
            experimental = pipelined_sdfg(program2, sdfg_name(name, preset, 'experimental'))
            exp_sym = base._symbol_values(experimental, sizes)
            exp_call = call_kwargs(arrays2, scalars2, exp_sym)
            experimental.compile()(**exp_call)
            out['metrics']['experimental'] = frame_metrics_for(experimental)
            out['experimental'] = engine.arrays_close(legacy_call, exp_call)  # vs legacy output
        except Exception as e:
            out['experimental'] = False
            out['error'] = f'{type(e).__name__}: {e}'
    return out


# --------------------------------------------------------------------------
# Kernel selection (parent process).
# --------------------------------------------------------------------------
def select_kernels(only, explicit):
    """The TSVC2.5 corpus kernel names (bare function names, from the vendored
    tsvc_2_5_corpus), optionally filtered by an explicit list or a substring."""
    names = sorted(p.f.__name__ for p in tsvc25.collect())
    if explicit is not None:
        names = [n for n in names if n in explicit]
    if only:
        names = [n for n in names if only in n]
    return names


def resolve_threads(preset):
    """OMP width for the preset: S is always single-core; paper honors an OMP_NUM_THREADS
    already exported (the sbatch sets it to the node/socket width), else the cpu count."""
    if preset == 'S':
        return 1
    exported = os.environ.get('OMP_NUM_THREADS')
    if exported and exported.isdigit() and int(exported) > 1:
        return int(exported)
    return os.cpu_count() or 1


def resolve_cxx_label(explicit):
    """The `cxx` column: the host C++ compiler DaCe's codegen is built with, as a bare name
    (g++ / clang++). Mirrors what engine.configure_dace_process actually honors -- an explicit
    --cxx, else the exported DACE_PERF_CXX -- so the label can never disagree with the compiler
    that ran. 'default' when neither is set: engine then autodetects (clang++ on PATH, else g++)."""
    cxx = explicit or os.environ.get('DACE_PERF_CXX', '')
    return os.path.basename(cxx) if cxx else 'default'


def resolve_phase(explicit, threads):
    """The `phase` column: which sweep phase produced the row. An explicit --phase (what the
    sbatch passes) wins; otherwise derive it from the OMP width, so a standalone invocation
    still labels itself correctly."""
    if explicit:
        return explicit
    return 'single_core' if threads == 1 else 'multi_core'


def codegen_list(codegen):
    return list(CODEGENS) if codegen == 'both' else [codegen]


# --------------------------------------------------------------------------
# Per-kernel driver + TSV assembly (mirrors run_buildperf.py).
# --------------------------------------------------------------------------
def correctness_cell(codegen, corr_ok, corr, perf_ok):
    """Render the correctness column for one (codegen) row from the isolated results."""
    if not perf_ok or not corr_ok:
        return 'ERROR'
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
    codegens = codegen_list(args.codegen)
    program = base._program(name)
    sizes = sizes_for(program, preset)

    # One correctness pass covers both lanes (legacy is always built as the reference) and
    # also carries each built frame's inline generated-code metrics.
    corr_ok, corr = engine.run_isolated(correctness_job, (name, sizes, preset, 'experimental' in codegens),
                                        timeout=timeout)
    if not corr_ok:
        corr = {}

    # Time each requested lane independently (crash-isolated).
    perf = {}
    for codegen in codegens:
        ok, payload = engine.run_isolated(perf_job, (name, sizes, preset, codegen, reps, timeout), timeout=timeout)
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
                   corpus=CORPUS,
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
        print(f"[{name}/{CORPUS}/{preset}/{phase}/{cxx}] {codegen}: codegen={codegen_ms or '-'}ms "
              f"compile={compile_ms or '-'}ms runtime={runtime_ms or '-'}ms nloc={row['nloc'] or '-'} "
              f"nest={row['max_nesting'] or '-'} speedup={speedup or '-'} correct={row['correctness']}" +
              (f' ({error})' if error else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset',
                    choices=('S', 'paper'),
                    default='S',
                    help='S = single-core, stock SIZES (quick signal); paper = full-node threads, ~2GB scaled sizes')
    ap.add_argument('--codegen',
                    choices=('legacy', 'experimental', 'both'),
                    default='both',
                    help='which code generator(s) to measure')
    ap.add_argument('--out', default='results.tsv', help='TSV output path (default: results.tsv)')
    ap.add_argument('--only', default=None, help='substring filter on kernel name')
    ap.add_argument('--kernels', default=None, help='comma-separated explicit kernel list (overrides --only)')
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
    kernels = select_kernels(args.only, explicit)

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
    print(f'preset={args.preset} corpus={CORPUS} codegen={args.codegen} '
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
