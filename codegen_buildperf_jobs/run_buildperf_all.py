#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Combined single-pass build-and-perf driver over ALL FOUR corpora at once (npbench +
polybench + tsvc2 + tsvc2_5), CPU only, for submit_codegen_buildperf.sbatch.

This is the orchestrator the sbatch launches once per (phase, cxx). Instead of a separate
per-corpus pass, it builds ONE combined kernel list spanning every corpus, slices it
round-robin across the MPI ranks (``combined[rank::num_ranks]`` off SLURM_PROCID/SLURM_NTASKS),
and measures each rank's own disjoint slice -- so all four corpora are distributed across the
4 ranks *together*, in a single ordered pass, into one per-rank TSV.

No measurement logic is reimplemented: each entry is dispatched to the matching per-corpus
driver's own ``process_kernel`` (``run_buildperf`` for npbench+polybench, ``run_buildperf_tsvc2``
for tsvc2, ``run_buildperf_tsvc2_5`` for tsvc2_5). Those three modules already share an
identical TSV schema and crash-isolated measurement (codegen/compile time, runtime, inline
generated-code size + readability, correctness), so their rows concatenate under one header.

    python3 run_buildperf_all.py --preset S --codegen both --phase single_core --out s.tsv
    python3 run_buildperf_all.py --preset paper --phase multi_core --out paper.tsv
    python3 run_buildperf_all.py --list-kernels --preset S
"""
import os

# MPI/OpenMP anti-hang defaults BEFORE importing dace / the per-corpus drivers (a dace script
# otherwise blocks on MPI_Init and the sweep looks hung). setdefault so an explicit value (a
# real srun launch, or the --preset thread override in main) wins. The imported drivers set the
# same defaults at their own import time; setting them here first keeps it explicit.
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
# This dir first (its per-corpus drivers + readability_metrics + vendored tsvc harness are
# authoritative); the sibling performance_regression_jobs is APPENDED so it can never shadow a
# local copy. Both paths derive from this file's own location -- no hardcoded absolute paths.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if PERF_JOBS_DIR not in sys.path:
    sys.path.append(PERF_JOBS_DIR)

import run_buildperf as npb
import run_buildperf_tsvc2 as t2
import run_buildperf_tsvc2_5 as t25

#: The per-corpus drivers, in the order their kernels are concatenated into the combined list.
#: run_buildperf covers BOTH npbench and polybench (it labels each row's corpus per kernel).
MODULES = (npb, t2, t25)

#: The shared TSV header, identical across the three drivers.
TSV_FIELDS = npb.TSV_FIELDS


def entry_corpus(module, name):
    """Display corpus for a combined-list entry (per-kernel for npbench/polybench)."""
    return module.kernel_corpus(name) if module is npb else module.CORPUS


def build_combined(preset, only, explicit):
    """The ONE combined kernel list across all four corpora, as [(module, name)] in a stable
    order (npbench+polybench, then tsvc2, then tsvc2_5, each sorted). Slicing this list
    ``[rank::num_ranks]`` distributes every corpus across the ranks together."""
    entries = []
    for name in npb.select_kernels('both', preset, only, explicit):
        entries.append((npb, name))
    for name in t2.select_kernels(only, explicit):
        entries.append((t2, name))
    for name in t25.select_kernels(only, explicit):
        entries.append((t25, name))
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset',
                    choices=('S', 'paper'),
                    default='S',
                    help='S = single-core, S/stock dataset (quick signal); paper = full-node threads, paper dataset')
    ap.add_argument('--codegen',
                    choices=('legacy', 'experimental', 'both'),
                    default='both',
                    help='which code generator(s) to measure')
    ap.add_argument('--out', default='results.tsv', help='TSV output path (default: results.tsv)')
    ap.add_argument('--only', default=None, help='substring filter on kernel name (applied within every corpus)')
    ap.add_argument('--reps', type=int, default=10, help='timed repetitions per lane (best-of; default: 10)')
    ap.add_argument('--timeout', type=float, default=900.0, help='per-measurement subprocess timeout, seconds')
    ap.add_argument('--list-kernels', action='store_true', help='print the combined kernel list and exit')
    # Host C++ compiler + phase label -- recorded per row so the phases of one merged sweep stay
    # groupable; --cxx additionally SELECTS the compiler (relayed via DACE_PERF_CXX). Both are
    # handled by the per-corpus drivers' own resolve_* helpers, reused here verbatim.
    ap.add_argument('--cxx',
                    default=None,
                    help="host C++ compiler for DaCe's codegen, also recorded in the `cxx` column "
                    '(default: $DACE_PERF_CXX, else engine autodetection)')
    ap.add_argument('--phase',
                    choices=('single_core', 'multi_core'),
                    default=None,
                    help='value for the `phase` column (default: derived from the OMP width)')
    # Multi-rank kernel partitioning over the COMBINED list: one MPI rank per Grace socket, each
    # measuring a disjoint slice ``combined[rank::num_ranks]``. Defaults come from the SLURM
    # launch (``SLURM_PROCID`` / ``SLURM_NTASKS``), so a plain single-process invocation stays
    # rank 0 of 1 (the whole list). Each rank writes its own ``<out>.rank<r>`` TSV.
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

    entries = build_combined(args.preset, args.only, None)

    # Slice the ONE combined list across ranks (round-robin balances uneven per-kernel cost
    # better than contiguous blocks) and give each rank a distinct output file.
    if args.num_ranks > 1:
        entries = entries[args.rank::args.num_ranks]
        stem, ext = os.path.splitext(args.out)
        args.out = f'{stem}.rank{args.rank}{ext or ".tsv"}'

    if args.list_kernels:
        print('\n'.join(f'{entry_corpus(m, n)}\t{n}' for m, n in entries))
        return

    # Reuse the per-corpus drivers' identical resolve_* helpers (single source of truth).
    threads = npb.resolve_threads(args.preset)
    os.environ['OMP_NUM_THREADS'] = str(threads)  # inherited by every spawned measurement subprocess
    cxx = npb.resolve_cxx_label(args.cxx)
    phase = npb.resolve_phase(args.phase, threads)

    rank_note = f' rank={args.rank}/{args.num_ranks}' if args.num_ranks > 1 else ''
    counts = {}
    for m, n in entries:
        counts[entry_corpus(m, n)] = counts.get(entry_corpus(m, n), 0) + 1
    print(f'preset={args.preset} codegen={args.codegen} threads={threads} cxx={cxx} phase={phase} '
          f'reps={args.reps}{rank_note}')
    print(f'{len(entries)} kernel(s) across all corpora {counts} -> {args.out}')

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter='\t')
        writer.writeheader()
        f.flush()
        for module, name in entries:
            module.process_kernel(writer, f, name, args, threads, cxx, phase)

    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
