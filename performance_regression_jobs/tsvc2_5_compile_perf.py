#!/usr/bin/env python3
"""Compile-speed regression over TSVC2.5: DaCe codegen time + C++ compile time
for the 4 DaCe pipelines -- baseline (plain simplify+loop2map+mapfusion),
auto-opt, canon, fast-canon.

Companion to tsvc2_5_perf.py (post-compile RUNTIME of the same lanes); writes
compile_total.md / compile_codegen.md / compile_cxx.md into the SAME results
tree as its speedup.md, so one job compares BOTH compile speed and post-compile
performance. Multi-rank + isolation + --cxx are reused unchanged (see engine.py
and tsvc2_compile_perf.py).

    python3 tsvc2_5_compile_perf.py                       # this rank's slice, 3 compile samples/lane
    python3 tsvc2_5_compile_perf.py --only s000 --compile-reps 1
    python3 tsvc2_5_compile_perf.py --tables-only         # rebuild compile_*.md
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dace  # noqa: F401

import engine
import tsvc2_5_perf as base

CORPUS = base.CORPUS
COMPILE_LANES = [f'{p}-par' for p in engine.PIPELINES]
BASELINE_LANE = 'auto_opt-par'


def _compile_job(kernel_name, pipeline, creps):
    samples = []
    for _ in range(creps):
        sdfg = base._build_sdfg(kernel_name, pipeline, False)
        samples.append(engine.compile_sdfg_timed(sdfg))
    return samples


def process_kernel(kernel_name, args, rank):
    kdir = engine.kernel_dir(args.results_dir, CORPUS, kernel_name, 'default')
    for lane in COMPILE_LANES:
        pipeline = lane.rsplit('-', 1)[0]
        have = 0 if args.force else engine.existing_compile_reps(kdir, lane)
        remaining = args.compile_reps - have
        if remaining <= 0:
            continue
        print(f'[{kernel_name}] {lane}: {remaining} compile sample(s)')
        ok, payload = engine.run_isolated(_compile_job, (kernel_name, pipeline, remaining), timeout=args.timeout)
        if ok:
            engine.append_compile_results(kdir, lane, payload, have)
        else:
            print(f'[{kernel_name}] {lane}: compile-bench failed: {payload}')
    engine.write_run_meta(kdir, rank=rank, compile_reps=args.compile_reps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    engine.add_common_args(ap)
    ap.add_argument('--compile-reps', type=int, default=3, help='cold-compile samples per pipeline (default: 3)')
    args = ap.parse_args()

    if args.list_kernels:
        print('\n'.join(base.kernel_list(args)))
        return
    if args.tables_only:
        engine.write_compile_tables(args.results_dir, CORPUS, COMPILE_LANES)
        return

    engine.export_cxx_override(args)
    cxx = engine.pick_cxx_compiler(args.cxx)
    print(f'C++ compiler (DaCe codegen): {cxx or "(none found; DaCe default)"}')

    rank, world = engine.get_world_rank(), engine.get_world_size()
    all_kernels = base.kernel_list(args)
    explicit = engine.explicit_kernel_list(args)
    mine = explicit if explicit is not None else engine.my_slice(all_kernels, rank, world)
    print(f'rank {rank}/{world}: {len(mine)}/{len(all_kernels)} kernels (compile-bench)')

    for name in mine:
        process_kernel(name, args, rank)

    engine.write_compile_tables(args.results_dir, CORPUS, COMPILE_LANES)


if __name__ == '__main__':
    main()
