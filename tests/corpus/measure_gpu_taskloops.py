# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A/B ``optimizer.gpu_taskloop_heuristics`` over npbench + polybench on the GPU.

Every kernel goes down the canon-GPU path twice, knob off and on, checked against its reference.
Each case runs in its OWN subprocess -- DaCe's parse state is process-global, and a CUDA compile
that dies takes the process with it. Rows stream as they land, so a sweep that is killed halfway
still leaves what it measured.

Run::

    python -m tests.corpus.measure_gpu_taskloops --preset paper --csv out.csv
    python -m tests.corpus.measure_gpu_taskloops --suites poly --only gemm,doitgen
"""
import os

# Pin the run before DaCe/OpenMP load, as the CPU sweep does.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse
import csv
import json
import subprocess
import sys
import time
from typing import Dict, List

import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import (finalize_for_target, recompute_fuse_for_gpu,
                                                              select_gpu_device_block_size)
from dace.transformation.passes.offloading.taskloop import taskloop_maps

from tests.corpus import corpus_suite as suite

#: Seconds per (kernel, setting) subprocess: a big CUDA compile is minutes, past this is a hang.
CASE_TIMEOUT = 1800


def offloaded_sdfg(ctx: Dict, heuristics: bool) -> dace.SDFG:
    """The kernel canonicalized, offloaded and finalized for the GPU at one setting of the knob."""
    taskloops: List[str] = []

    def transform(sdfg: dace.SDFG) -> None:
        canonicalize(sdfg, validate=False, validate_all=False, target='gpu')
        # ``offload_to_gpu`` minus ``apply_gpu_storage``: the corpus hands out numpy arrays, and a
        # host-side signature puts the copies the pass places inside the measurement.
        with dace.config.set_temporary('optimizer', 'gpu_taskloop_heuristics', value=heuristics), \
                dace.config.set_temporary('compiler', 'cuda', 'max_concurrent_streams', value=-1):
            recompute_fuse_for_gpu(sdfg)
            taskloops.extend(entry.map.label for entry in taskloop_maps(sdfg))
            sdfg.apply_gpu_transformations()
            select_gpu_device_block_size(sdfg)
        finalize_for_target(sdfg, 'gpu', validate=False)

    sdfg = suite.build(ctx, transform, f"taskloop_{int(heuristics)}")
    sdfg.taskloop_labels = taskloops
    return sdfg


def measure_one(kind: str, name: str, heuristics: bool, repeats: int, preset: str) -> Dict:
    """Correctness + median time for one kernel at one setting; runs in its own process."""
    ctx = suite.make(kind, name, preset=preset)
    sdfg = offloaded_sdfg(ctx, heuristics)
    taskloops = list(sdfg.taskloop_labels)
    correct = bool(suite.run_matches(ctx, sdfg))

    compiled, kwargs = suite.compiled_call(ctx, sdfg)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        compiled(**kwargs)
        times.append(time.perf_counter() - start)
    times.sort()
    return dict(name=name, heuristics=heuristics, correct=correct, taskloops=taskloops, median=times[len(times) // 2])


def run_case(kind: str, name: str, heuristics: bool, repeats: int, preset: str) -> Dict:
    """Spawn ``measure_one`` and read back its JSON; a dead case reports itself incorrect."""
    cmd = [
        sys.executable, '-m', 'tests.corpus.measure_gpu_taskloops', '--one', name, '--kind', kind, '--heuristics',
        str(int(heuristics)), '--repeats',
        str(repeats), '--preset', preset
    ]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=CASE_TIMEOUT, cwd=os.getcwd())
    except subprocess.TimeoutExpired:
        return dict(kind=kind,
                    name=name,
                    heuristics=heuristics,
                    correct=False,
                    taskloops=[],
                    median=float('nan'),
                    error=f'timeout after {CASE_TIMEOUT}s')
    for line in done.stdout.splitlines():
        if line.startswith('RESULT '):
            return json.loads(line[len('RESULT '):])
    tail = [line for line in (done.stderr or done.stdout).strip().splitlines() if line.strip()]
    return dict(kind=kind,
                name=name,
                heuristics=heuristics,
                correct=False,
                taskloops=[],
                median=float('nan'),
                error=tail[-1][:200] if tail else 'no output')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suites', default='np,poly', help='comma-separated: np, poly')
    parser.add_argument('--only', help='comma-separated kernel names')
    parser.add_argument('--preset', default='paper', choices=suite.PRESETS)
    parser.add_argument('--repeats', type=int, default=5, help='timed invocations per case')
    parser.add_argument('--csv', help='write the table here as it goes')
    parser.add_argument('--one', help='internal: measure this one kernel in this process')
    parser.add_argument('--kind', default='np', help='internal: the suite for --one')
    parser.add_argument('--heuristics', type=int, default=0, help='internal: the knob for --one')
    args = parser.parse_args()

    if args.one:
        print('RESULT ' +
              json.dumps(measure_one(args.kind, args.one, bool(args.heuristics), args.repeats, args.preset)),
              flush=True)
        return 0

    wanted = args.only.split(',') if args.only else None
    suites = args.suites.split(',')
    cases = [(kind, name) for kind, name in suite.kernels() if kind in suites and (wanted is None or name in wanted)]

    rows: List[Dict] = []
    print(f"preset={args.preset} repeats={args.repeats} cases={len(cases)}", flush=True)
    print(f"{'suite':6s} {'kernel':28s} {'off':>10s} {'on':>10s} {'ratio':>7s}  taskloops", flush=True)
    for kind, name in cases:
        off = run_case(kind, name, False, args.repeats, args.preset)
        on = run_case(kind, name, True, args.repeats, args.preset)
        ratio = on['median'] / off['median'] if off['median'] > 0 else float('nan')
        note = '' if (off['correct'] and on['correct']) else '  ' + (on.get('error') or off.get('error') or 'WRONG')
        rows.append(
            dict(suite=kind,
                 kernel=name,
                 off=off['median'],
                 on=on['median'],
                 ratio=ratio,
                 off_correct=off['correct'],
                 on_correct=on['correct'],
                 taskloops=' '.join(on['taskloops']),
                 note=note.strip()))
        print(
            f"{kind:6s} {name:28s} {off['median']:10.5f} {on['median']:10.5f} {ratio:7.2f}  "
            f"{' '.join(on['taskloops'])[:36]}{note[:90]}",
            flush=True)
        if args.csv:
            with open(args.csv, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    fired = [r['kernel'] for r in rows if r['taskloops']]
    print(f"heuristics fired on {len(fired)}/{len(rows)}: {', '.join(fired)}", flush=True)
    wrong = [r['kernel'] for r in rows if not (r['off_correct'] and r['on_correct'])]
    if wrong:
        print(f"INCORRECT or FAILED: {', '.join(wrong)}", flush=True)
    return 1 if wrong else 0


if __name__ == '__main__':
    sys.exit(main())
