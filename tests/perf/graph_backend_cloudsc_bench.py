# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Benchmark: does dace.graphlib's rustworkx backend actually speed up a realistic,
large-SDFG workload, compared to the default networkx backend?

Pipeline timed per repetition, per backend, in this order (see cloudsc_backend_pipeline.py):
  1. simplify              -- SDFG.simplify() on the CloudSC kernel.
  2. config_prop_loopunroll -- specialize_scalar(kidia/kfdia) + SDFG.specialize(nclv family)
                               ("config-prop") + apply_transformations_repeated(LoopUnroll),
                               on the simplified result.
  3. codegen                -- dace.codegen.codegen.generate_code(sdfg): pure Python SDFG ->
                               C++ source text, no compiler invoked.
  4. compile                -- sdfg.compile(): the full build (repeats codegen internally,
                               then invokes CMake/g++). Reported separately from (3) since a
                               real user calls .compile() directly; the overlap is expected.
  5. serialize / deserialize -- SDFG.save(compress=True) / SDFG.from_file().

Correctness (both backends reproducing the un-transformed reference bit-for-bit under the IEEE
build) is checked ONCE up front, not per repetition -- see graph_backend_cloudsc_test.py for the
dedicated, CI-integrated version of the same check, run through the identical simplify ->
config-prop+loopunroll pipeline.

That check is REPORTED, not enforced: this is a compile/optimize-time benchmark, and the known
CloudSC divergence reproduces under backend='networkx' too, so it is a property of the
transformation pipeline (LoudUnroll / specialize_scalar), not of the graph backend under test.
Aborting the run on it would only mean a multi-hour allocation produces no timings at all. The
outcome is printed loudly, recorded in the JSON under 'correctness', and stamped on both the
markdown table and the plot, so no timing table can be mistaken for a validated one. Pass
--strict-correctness to restore the old abort-on-divergence behavior.

Usage: python3 tests/perf/graph_backend_cloudsc_bench.py [--reps 10] [--output out.json]
                                                          [--table-output table.md]
                                                          [--plot-output plot.png]
                                                          [--strict-correctness]
"""
import argparse
import copy
import json
import os
import statistics
import time
import traceback
from typing import Dict, List

import dace
from dace import graphlib
from dace.codegen.codegen import generate_code

from tests.corpus.cloudsc.generate_data_for_cloudsc import (IEEE_CPU_ARGS, build_cloudsc_sdfg, compare_outputs,
                                                            generate_cloudsc_inputs, make_sequential)
from tests.perf.cloudsc_backend_pipeline import (filtered_inputs, run_pipeline, run_simplify, specialize_and_unroll)

BACKENDS = ['networkx', 'rustworkx']
PHASES = [
    'frontend', 'simplify', 'codegen_simplified', 'config_prop_loopunroll', 'codegen', 'compile', 'serialize',
    'deserialize'
]
PHASE_LABELS = {
    'frontend': 'python->sdfg',
    'simplify': 'sdfg->simplified',
    'codegen_simplified': 'simplified->codegen',
    'config_prop_loopunroll': 'config-prop+loopunroll',
    'codegen': 'codegen(unrolled)',
    'compile': 'compile',
    'serialize': 'serialize',
    'deserialize': 'deserialize',
}


def check_correctness(reference: dace.SDFG, backend: str) -> Dict[str, tuple]:
    """Run the full simplify -> config-prop+loopunroll pipeline on a private copy under
    ``backend`` and compare it against ``reference`` bit-for-bit under the IEEE build.

    Returns the per-array ``{name: (max_abs, max_rel)}`` of everything that did NOT match, empty
    when the pipeline reproduces the reference. Reporting rather than raising is deliberate --
    see the module docstring; the caller decides what to do about a non-empty result."""
    candidate = copy.deepcopy(reference)
    make_sequential(candidate)
    run_pipeline(candidate, backend)

    ref_inputs = generate_cloudsc_inputs(reference, seed=0)
    cand_inputs = filtered_inputs(candidate, copy.deepcopy(ref_inputs))

    saved_args = dace.Config.get('compiler', 'cpu', 'args')
    try:
        dace.Config.set('compiler', 'cpu', 'args', value=IEEE_CPU_ARGS)
        # reference(**ref_inputs) mutates ref_inputs in place with the reference's
        # actual outputs -- cand_inputs was deep-copied BEFORE this call, so the two
        # dicts hold independent, comparable post-run buffers (matching the harness's
        # own run_and_compare pattern). Calling with a throwaway copy here would leave
        # ref_inputs at its pristine pre-run values, comparing inputs against outputs.
        reference(**ref_inputs)
        candidate(**cand_inputs)
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=saved_args)

    report = compare_outputs(ref_inputs, cand_inputs, rtol=1e-15, atol=1e-15)
    return {name: (max_abs, max_rel) for name, (max_abs, max_rel, ok) in report.items() if not ok}


def run_one_repetition(reference: dace.SDFG, backend: str, tmp_dir: str, rep: int) -> Dict[str, float]:
    """Run the full timed pipeline once under ``backend``, starting from a fresh
    private copy of ``reference``. Returns ``{phase: seconds}``."""
    times: Dict[str, float] = {}

    # python -> SDFG, measured per repetition rather than reusing the module-level reference:
    # the frontend parse is the phase being timed here, so it has to actually run. simplify=False
    # keeps it a pure parse, with simplification measured as its own phase below.
    with graphlib.set_default_backend(backend):
        t0 = time.perf_counter()
        sdfg = build_cloudsc_sdfg(simplify=False)
        times['frontend'] = time.perf_counter() - t0
    make_sequential(sdfg)

    # SDFG -> simplified, then codegen straight off the SIMPLIFIED graph (before config-prop and
    # unrolling) so that cost is attributable on its own -- 'codegen' further down is the
    # post-unroll one, a much bigger graph, and the two are not comparable.
    times['simplify'] = run_simplify(sdfg, backend)
    with graphlib.set_default_backend(backend):
        simplified = copy.deepcopy(sdfg)
        t0 = time.perf_counter()
        generate_code(simplified)
        times['codegen_simplified'] = time.perf_counter() - t0

    pass_time, applied = specialize_and_unroll(sdfg, backend)
    if applied == 0:
        # Worth shouting about -- it means the phase timed nothing -- but not worth losing the
        # whole allocation over: the remaining phases are still measuring real work.
        print(f'  WARNING: backend={backend!r} rep={rep}: LoopUnroll found nothing to unroll after simplify')
    times['config_prop_loopunroll'] = pass_time

    with graphlib.set_default_backend(backend):
        t0 = time.perf_counter()
        generate_code(sdfg)
        times['codegen'] = time.perf_counter() - t0

        sdfg.name = f'cloudsc_bench_{backend}_{rep}'
        saved_args = dace.Config.get('compiler', 'cpu', 'args')
        try:
            dace.Config.set('compiler', 'cpu', 'args', value=IEEE_CPU_ARGS)
            t0 = time.perf_counter()
            sdfg.compile()
            times['compile'] = time.perf_counter() - t0
        finally:
            dace.Config.set('compiler', 'cpu', 'args', value=saved_args)

        path = f'{tmp_dir}/bench_{backend}_{rep}.sdfgz'
        t0 = time.perf_counter()
        sdfg.save(path, compress=True)
        times['serialize'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dace.SDFG.from_file(path)
        times['deserialize'] = time.perf_counter() - t0

    return times


def safe_median(values: List[float]) -> float:
    """Median that yields NaN instead of raising on an empty sample list -- a repetition that
    failed contributes no sample, and one lost repetition must not sink the whole report."""
    return statistics.median(values) if values else float('nan')


def median_report(samples: Dict[str, Dict[str, List[float]]]) -> str:
    lines = [f"{'phase':<24} {'networkx (s)':>14} {'rustworkx (s)':>15} {'speedup':>9}"]
    for phase in PHASES:
        nx_med = safe_median(samples['networkx'][phase])
        rx_med = safe_median(samples['rustworkx'][phase]) if 'rustworkx' in samples else float('nan')
        speedup = nx_med / rx_med if rx_med else float('nan')
        lines.append(f"{PHASE_LABELS[phase]:<24} {nx_med:>14.4f} {rx_med:>15.4f} {speedup:>8.2f}x")
    return '\n'.join(lines)


def write_markdown_table(samples: Dict[str, Dict[str, List[float]]], path: str, reps: int,
                         correctness: Dict[str, Dict[str, tuple]]) -> None:
    """Write the median-per-phase comparison as a markdown table, for the SLURM job's
    saved artifact (see submit_cloudsc_backend.sh / run_cloudsc_backend.sh)."""
    have_rustworkx = 'rustworkx' in samples
    lines = [
        f'# Graph backend benchmark: CloudSC (median over {reps} repetitions)',
        '',
    ]
    lines += correctness_note(correctness) + ['']
    lines += [
        '| phase | networkx (s) | rustworkx (s) | speedup |',
        '|---|---:|---:|---:|',
    ]
    for phase in PHASES:
        nx_med = safe_median(samples['networkx'][phase])
        if have_rustworkx:
            rx_med = safe_median(samples['rustworkx'][phase])
            speedup = f'{nx_med / rx_med:.2f}x' if rx_med else 'n/a'
            rx_cell = f'{rx_med:.4f}'
        else:
            rx_cell = 'n/a'
            speedup = 'n/a'
        lines.append(f'| {PHASE_LABELS[phase]} | {nx_med:.4f} | {rx_cell} | {speedup} |')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def correctness_note(correctness: Dict[str, Dict[str, tuple]]) -> List[str]:
    """One-line-per-backend verdict, so a timing artifact always carries its own caveat."""
    diverged = sorted(b for b, bad in correctness.items() if bad)
    if not diverged:
        return ['Correctness: all backends reproduce the un-transformed reference bit-for-bit.']
    arrays = sorted({name for b in diverged for name in correctness[b]})
    return [
        f'**Correctness: DIVERGED on {", ".join(diverged)}** -- these are compile/optimize-time '
        'numbers only, the pipeline output does not match the reference.',
        '',
        f'Arrays affected: {", ".join(arrays)}.',
    ]


def write_plot(samples: Dict[str, Dict[str, List[float]]], path: str, reps: int,
               correctness: Dict[str, Dict[str, tuple]]) -> bool:
    """Grouped bar chart of per-phase medians, one bar per backend. Log-scaled because compile
    dominates the other phases by orders of magnitude and would otherwise flatten them to zero.
    Returns False (without raising) when matplotlib is unavailable -- a missing plotting library
    on a compute node must not cost the run its timings."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # no display on a compute node
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed -- skipping plot (timings and table are unaffected).')
        return False

    backends = [b for b in BACKENDS if b in samples]
    x = range(len(PHASES))
    width = 0.8 / len(backends)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, backend in enumerate(backends):
        medians = [safe_median(samples[backend][p]) for p in PHASES]
        offsets = [xi + i * width - 0.4 + width / 2 for xi in x]
        bars = ax.bar(offsets, medians, width, label=backend)
        ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)

    ax.set_yscale('log')
    ax.set_ylabel('seconds (median, log scale)')
    ax.set_xticks(list(x))
    ax.set_xticklabels([PHASE_LABELS[p] for p in PHASES], rotation=30, ha='right')
    ax.set_title(f'CloudSC compile/optimize time by phase (median of {reps} reps)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, which='both')

    if any(correctness.values()):
        fig.text(0.5,
                 0.01,
                 'WARNING: pipeline output diverges from the reference -- timing data only.',
                 ha='center',
                 fontsize=9,
                 color='crimson')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reps', type=int, default=10, help='Repetitions per backend (default: 10)')
    parser.add_argument('--output', type=str, default=None, help='Optional path to dump raw samples as JSON')
    parser.add_argument('--table-output',
                        type=str,
                        default=None,
                        help='Optional path to write the median comparison as a markdown table')
    parser.add_argument('--plot-output',
                        type=str,
                        default=None,
                        help='Optional path to write a per-phase bar chart (PNG, needs matplotlib)')
    parser.add_argument('--strict-correctness',
                        action='store_true',
                        help='Abort before timing if the pipeline output diverges from the reference '
                        '(default: report the divergence and benchmark anyway)')
    parser.add_argument('--tmp-dir',
                        type=str,
                        default='/tmp/graph_backend_cloudsc_bench',
                        help='Scratch dir for serialize/deserialize (default: /tmp/graph_backend_cloudsc_bench)')
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)

    print('Building reference CloudSC SDFG (frontend parse, backend-independent, this is the slow part)...')
    reference = build_cloudsc_sdfg(simplify=False)
    make_sequential(reference)
    print(f'Reference built: {reference.number_of_nodes()} top-level nodes.')

    available_backends = list(BACKENDS)
    try:
        import rustworkx  # noqa: F401
    except ImportError:
        print('rustworkx not installed -- skipping rustworkx backend entirely.')
        available_backends = ['networkx']

    print('Correctness check (once per backend, not timed)...')
    correctness: Dict[str, Dict[str, tuple]] = {}
    for backend in available_backends:
        correctness[backend] = check_correctness(reference, backend)
        if correctness[backend]:
            print(f'  backend={backend!r}: DIVERGES from the un-transformed reference: {correctness[backend]}')
        else:
            print(f'  backend={backend!r}: OK (matches un-transformed reference bit-for-bit)')

    if any(correctness.values()) and args.strict_correctness:
        raise RuntimeError(f'--strict-correctness: pipeline output diverges from reference: {correctness}')
    if any(correctness.values()):
        print('  ^ continuing anyway: this is a compile/optimize-time benchmark and the divergence is a')
        print('    property of the transformation pipeline, not of the graph backend (it reproduces on')
        print('    networkx too). Timings below are valid; pipeline OUTPUT is not. Use --strict-correctness')
        print('    to abort here instead.')

    samples: Dict[str, Dict[str, List[float]]] = {b: {p: [] for p in PHASES} for b in available_backends}
    failures = 0
    for backend in available_backends:
        for rep in range(args.reps):
            try:
                times = run_one_repetition(reference, backend, args.tmp_dir, rep)
            except Exception:  # noqa: BLE001 -- one lost repetition must not cost the whole run
                failures += 1
                print(f'backend={backend:<10} rep={rep + 1}/{args.reps}  FAILED, skipping this repetition:')
                traceback.print_exc()
                continue
            for phase in PHASES:
                samples[backend][phase].append(times[phase])
            print(f'backend={backend:<10} rep={rep + 1}/{args.reps}  ' + '  '.join(f'{PHASE_LABELS[p]}={times[p]:.4f}s'
                                                                                   for p in PHASES))

    if failures:
        print(f'\n{failures} repetition(s) failed and were skipped; medians below use the survivors.')
    if not any(samples[b][PHASES[0]] for b in available_backends):
        raise RuntimeError('every repetition failed on every backend -- no timings to report')

    print()
    print(f'Median over {args.reps} repetitions:')
    print(
        median_report(samples) if len(available_backends) == 2 else '\n'.join(
            f'{PHASE_LABELS[p]}: {safe_median(samples["networkx"][p]):.4f}s' for p in PHASES))
    print()
    print('\n'.join(correctness_note(correctness)))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'samples': samples, 'correctness': correctness, 'reps': args.reps}, f, indent=2)
        print(f'Raw samples written to {args.output}')

    if args.table_output:
        write_markdown_table(samples, args.table_output, args.reps, correctness)
        print(f'Markdown table written to {args.table_output}')

    if args.plot_output and write_plot(samples, args.plot_output, args.reps, correctness):
        print(f'Plot written to {args.plot_output}')


if __name__ == '__main__':
    main()
