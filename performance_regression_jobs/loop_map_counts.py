#!/usr/bin/env python3
"""Counts loops (LoopRegion) and maps (MapEntry) before vs. after canonicalize,
across all 3 corpora. Purely structural (canonicalize needs no input data, no
compilation, no execution), so this is much lighter than the *_perf.py scripts
-- just build the SDFG, count, transform, count again.

    python3 loop_map_counts.py                       # all 3 corpora
    python3 loop_map_counts.py --corpus tsvc2 --only s000
    python3 loop_map_counts.py --csv counts.csv
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import csv
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dace
import dace.transformation.passes.canonicalize  # noqa: F401  (warm import graph; see engine.configure_dace_process)
from dace.sdfg.state import LoopRegion

import tsvc_corpus as tsvc
import tsvc_2_5_corpus as tsvc25
import npbench_polybench_perf as npbp

PIPELINES = ('canon', )


def count_loops_and_maps(sdfg):
    loops = sum(1 for cfr in sdfg.all_control_flow_regions(recursive=True) if isinstance(cfr, LoopRegion))
    maps = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))
    return loops, maps


def _canonicalize(sdfg):
    from dace.transformation.passes.canonicalize import canonicalize
    return canonicalize(sdfg, validate=True)


def _row(corpus, name, sdfg_before, build_after):
    loops_before, maps_before = count_loops_and_maps(sdfg_before)
    row = dict(corpus=corpus, kernel=name, loops_before=loops_before, maps_before=maps_before)
    label = 'canon'
    try:
        after = build_after()
        loops_after, maps_after = count_loops_and_maps(after)
        row[f'loops_{label}'] = loops_after
        row[f'maps_{label}'] = maps_after
    except Exception as e:
        row[f'loops_{label}'] = row[f'maps_{label}'] = ''
        row[f'error_{label}'] = f'{type(e).__name__}: {str(e)[:120]}'
    return row


# --------------------------------------------------------------------------
# Per-corpus kernel iteration, reusing the existing corpus modules as-is.
# --------------------------------------------------------------------------
def tsvc2_rows(only):
    for kernel in tsvc.collect():
        if only and only not in kernel.name:
            continue
        before = tsvc.to_sdfg(kernel, f'{kernel.name}_before', simplify=True)

        def build(kernel=kernel):
            sdfg = tsvc.to_sdfg(kernel, f'{kernel.name}_canon', simplify=False)
            return _canonicalize(sdfg)

        yield _row('tsvc2', kernel.name, before, build)


def tsvc2_5_rows(only):
    for program in tsvc25.collect():
        name = program.f.__name__
        if only and only not in name:
            continue
        before = program.to_sdfg(simplify=True)

        def build(program=program):
            sdfg = program.to_sdfg(simplify=False)
            return _canonicalize(sdfg)

        yield _row('tsvc2_5', name, before, build)


def npbench_polybench_rows(only):
    class _Args:
        pass

    args = _Args()
    args.only = only
    for name in npbp.kernel_list(args):
        if not npbp.kernel_exists(name):
            continue
        info = npbp.load_bench_info(name)
        if npbp.PRESET not in info.get('parameters', {}):
            continue
        params = info['parameters'][npbp.PRESET]
        program, _arrays = npbp.build_program_and_data(name, info, params)
        before = program.to_sdfg(simplify=True)

        def build(program=program):
            sdfg = program.to_sdfg(simplify=False)
            return _canonicalize(sdfg)

        yield _row('npbench_polybench', name, before, build)


CORPORA = {'tsvc2': tsvc2_rows, 'tsvc2_5': tsvc2_5_rows, 'npbench_polybench': npbench_polybench_rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', choices=list(CORPORA), default=None, help='default: all 3')
    ap.add_argument('--only', default=None, help='substring filter on kernel name')
    ap.add_argument('--csv', default=None, help='write results to this CSV path (default: print a table)')
    args = ap.parse_args()

    corpora = [args.corpus] if args.corpus else list(CORPORA)
    fields = ['corpus', 'kernel', 'loops_before', 'maps_before']
    for label in PIPELINES:
        fields += [f'loops_{label}', f'maps_{label}', f'error_{label}']

    rows = []
    for corpus in corpora:
        for row in CORPORA[corpus](args.only):
            rows.append(row)
            print(f"[{row['corpus']}] {row['kernel']}: "
                  f"loops {row['loops_before']}->{row.get('loops_canon')} "
                  f"maps {row['maps_before']}->{row.get('maps_canon')}")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or '.', exist_ok=True)
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {len(rows)} rows to {args.csv}')


if __name__ == '__main__':
    main()
