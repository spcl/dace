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

Correctness (both backends must reproduce the un-transformed reference bit-for-bit under the
IEEE build) is checked ONCE up front, not per repetition -- see graph_backend_cloudsc_test.py
for the dedicated, CI-integrated version of the same check, run through the identical
simplify -> config-prop+loopunroll pipeline. A benchmark whose backends disagree numerically
is not reporting a real speedup, so this script refuses to print a timing table if that
check fails.

Usage: python3 tests/perf/graph_backend_cloudsc_bench.py [--reps 10] [--output out.json]
                                                          [--table-output table.md]
"""
import argparse
import copy
import json
import os
import statistics
import time
from typing import Dict, List

import dace
from dace.codegen.codegen import generate_code

from tests.corpus.cloudsc.generate_data_for_cloudsc import (IEEE_CPU_ARGS, build_cloudsc_sdfg, compare_outputs,
                                                    generate_cloudsc_inputs, make_sequential)
from tests.perf.cloudsc_backend_pipeline import filtered_inputs, run_pipeline

BACKENDS = ['networkx', 'rustworkx']
PHASES = ['simplify', 'config_prop_loopunroll', 'codegen', 'compile', 'serialize', 'deserialize']
PHASE_LABELS = {
    'simplify': 'simplify',
    'config_prop_loopunroll': 'config-prop+loopunroll',
    'codegen': 'codegen',
    'compile': 'compile',
    'serialize': 'serialize',
    'deserialize': 'deserialize',
}


def check_correctness(reference: dace.SDFG, backend: str) -> None:
    """Run the full simplify -> config-prop+loopunroll pipeline on a private copy under
    ``backend`` and assert it reproduces ``reference`` bit-for-bit under the IEEE build.
    Raises on mismatch."""
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
    bad = {name: (max_abs, max_rel) for name, (max_abs, max_rel, ok) in report.items() if not ok}
    if bad:
        raise RuntimeError(f'backend={backend!r}: pipeline output diverges from reference: {bad}')


def run_one_repetition(reference: dace.SDFG, backend: str, tmp_dir: str, rep: int) -> Dict[str, float]:
    """Run the full timed pipeline once under ``backend``, starting from a fresh
    private copy of ``reference``. Returns ``{phase: seconds}``."""
    sdfg = copy.deepcopy(reference)
    make_sequential(sdfg)

    times: Dict[str, float] = run_pipeline(sdfg, backend)

    with dace.config.set_temporary('graph', 'backend', value=backend):
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


def median_report(samples: Dict[str, Dict[str, List[float]]]) -> str:
    lines = [f"{'phase':<24} {'networkx (s)':>14} {'rustworkx (s)':>15} {'speedup':>9}"]
    for phase in PHASES:
        nx_med = statistics.median(samples['networkx'][phase])
        rx_med = statistics.median(samples['rustworkx'][phase]) if 'rustworkx' in samples else float('nan')
        speedup = nx_med / rx_med if rx_med else float('nan')
        lines.append(f"{PHASE_LABELS[phase]:<24} {nx_med:>14.4f} {rx_med:>15.4f} {speedup:>8.2f}x")
    return '\n'.join(lines)


def write_markdown_table(samples: Dict[str, Dict[str, List[float]]], path: str, reps: int) -> None:
    """Write the median-per-phase comparison as a markdown table, for the SLURM job's
    saved artifact (see submit_cloudsc_backend.sh / run_cloudsc_backend.sh)."""
    have_rustworkx = 'rustworkx' in samples
    lines = [
        f'# Graph backend benchmark: CloudSC (median over {reps} repetitions)',
        '',
        '| phase | networkx (s) | rustworkx (s) | speedup |',
        '|---|---:|---:|---:|',
    ]
    for phase in PHASES:
        nx_med = statistics.median(samples['networkx'][phase])
        if have_rustworkx:
            rx_med = statistics.median(samples['rustworkx'][phase])
            speedup = f'{nx_med / rx_med:.2f}x' if rx_med else 'n/a'
            rx_cell = f'{rx_med:.4f}'
        else:
            rx_cell = 'n/a'
            speedup = 'n/a'
        lines.append(f'| {PHASE_LABELS[phase]} | {nx_med:.4f} | {rx_cell} | {speedup} |')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reps', type=int, default=10, help='Repetitions per backend (default: 10)')
    parser.add_argument('--output', type=str, default=None, help='Optional path to dump raw samples as JSON')
    parser.add_argument('--table-output', type=str, default=None,
                        help='Optional path to write the median comparison as a markdown table')
    parser.add_argument('--tmp-dir', type=str, default='/tmp/graph_backend_cloudsc_bench',
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
    for backend in available_backends:
        check_correctness(reference, backend)
        print(f'  backend={backend!r}: OK (matches un-transformed reference bit-for-bit)')

    samples: Dict[str, Dict[str, List[float]]] = {b: {p: [] for p in PHASES} for b in available_backends}
    for backend in available_backends:
        for rep in range(args.reps):
            times = run_one_repetition(reference, backend, args.tmp_dir, rep)
            for phase in PHASES:
                samples[backend][phase].append(times[phase])
            print(f'backend={backend:<10} rep={rep + 1}/{args.reps}  ' +
                 '  '.join(f'{PHASE_LABELS[p]}={times[p]:.4f}s' for p in PHASES))

    print()
    print(f'Median over {args.reps} repetitions:')
    print(median_report(samples) if len(available_backends) == 2 else
         '\n'.join(f'{PHASE_LABELS[p]}: {statistics.median(samples["networkx"][p]):.4f}s' for p in PHASES))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f'Raw samples written to {args.output}')

    if args.table_output:
        write_markdown_table(samples, args.table_output, args.reps)
        print(f'Markdown table written to {args.table_output}')


if __name__ == '__main__':
    main()
