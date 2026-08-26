# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A/B ``optimizer.gpu_taskloop_heuristics`` over polybench, npbench and tsvc on the GPU.

Every kernel goes down the canon-GPU path twice, knob off and on, checked against its reference.
Each case runs in its OWN subprocess -- DaCe's parse state is process-global, and a CUDA compile
that dies takes the process with it. Rows stream as they land, so a sweep that is killed halfway
still leaves what it measured.

The two arms are SCREENED before they are timed: each is offloaded and emitted (seconds, no compiler,
no device) and the two finalized graphs compared. The knob is inert on most of the corpus, and where
the graph handed to codegen is identical so is its runtime -- so those kernels are compiled and timed
once and the ratio reported as exactly 1. Only a kernel the knob actually rewrites pays for two CUDA
builds. That is what makes a 281-kernel sweep finish: the screen costs seconds, a build costs
minutes.

``--stage codegen`` stops after the screen. It needs no GPU at all, which is also the only way to
cover the offloading pass on a device-less runner.

Run::

    python -m tests.corpus.measure_gpu_taskloops --preset paper --csv out.csv
    python -m tests.corpus.measure_gpu_taskloops --suites poly --only gemm,doitgen
    python -m tests.corpus.measure_gpu_taskloops --stage codegen --suites tsvc --shard 1/4
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
import hashlib
import json
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import (finalize_for_target, recompute_fuse_for_gpu,
                                                              select_gpu_device_block_size)
from dace.transformation.passes.offloading.taskloop import taskloop_maps

from tests.corpus import corpus_suite as suite

#: Seconds per (kernel, setting) subprocess: a big CUDA compile is minutes, past this is a hang.
CASE_TIMEOUT = 1800
#: Untimed calls before the timed ones: the first invocation pays lazy device init and a cold
#: instruction cache, and on a power-capped part it also runs at the idle clock.
WARMUP = 2
#: Seconds for a screen case. It parses, canonicalizes and emits -- no compiler, no device.
SCREEN_TIMEOUT = 900


def offloaded_sdfg(ctx: Dict, heuristics: bool, tag: str) -> Tuple[dace.SDFG, List[str]]:
    """The kernel canonicalized, offloaded and finalized for the GPU at one setting of the knob.

    ``tag`` uniquifies the SDFG name, hence the build folder. The screen hands both arms the SAME
    tag on purpose: the name reaches the emitted text, so differing tags would make every kernel
    look rewritten.
    """
    taskloops: List[str] = []

    def transform(sdfg: dace.SDFG) -> None:
        canonicalize(sdfg, validate=False, validate_all=False, target='gpu')
        # ``offload_to_gpu`` minus ``apply_gpu_storage``: the corpus hands out numpy arrays, and a
        # host-side signature puts the copies the pass places inside the measurement.
        with dace.config.set_temporary('optimizer', 'gpu_taskloop_heuristics', value=heuristics), \
                dace.config.set_temporary('compiler', 'cuda', 'max_concurrent_streams', value=-1):
            recompute_fuse_for_gpu(sdfg)
            # ``launch_only`` IS the knob: without it the classifier reports every candidate, so a
            # column built from the bare call is identical in both arms and says nothing.
            taskloops.extend(entry.map.label for entry in taskloop_maps(sdfg, launch_only=heuristics))
            sdfg.apply_gpu_transformations()
            select_gpu_device_block_size(sdfg)
        finalize_for_target(sdfg, 'gpu', validate=False)

    return suite.build(ctx, transform, tag), taskloops


#: Fields an SDFG remints on every rebuild. Two structurally identical graphs disagree on them, so
#: a digest that keeps them calls every kernel rewritten and screens nothing.
VOLATILE_JSON_FIELDS = ('guid', 'sdfg_list_id', 'cfg_list_id', 'hash')


def stable_json(node: Any) -> Any:
    """``node`` with every per-object identity field dropped, recursively."""
    if isinstance(node, dict):
        return {k: stable_json(v) for k, v in node.items() if k not in VOLATILE_JSON_FIELDS}
    if isinstance(node, list):
        return [stable_json(item) for item in node]
    return node


def codegen_one(kind: str, name: str, heuristics: bool, preset: str) -> Dict:
    """Offload one kernel at one setting and digest the graph; no compiler and no device involved.

    The digest is taken from the SDFG, not from the emitted text, even though the text is what runs.
    ``generate_code`` is not reproducible on this branch -- the same finalized SDFG emits one of two
    programs run to run -- so a text digest would report kernels as rewritten at random. It is still
    called, because a kernel the offloading pass leaves unemittable has to fail here.

    ⛔ Order is load-bearing: emitting MUTATES the graph, so the digest has to precede it.
    """
    ctx = suite.make(kind, name, preset=preset)
    sdfg, taskloops = offloaded_sdfg(ctx, heuristics, 'ab')
    # Digest BEFORE emitting. ``generate_code`` rewrites the graph it is handed -- it pads every
    # region boundary with a landing state and resets the CFG list -- and it does so
    # nondeterministically, so a digest taken afterwards reports kernels as rewritten at random.
    digest = hashlib.sha256(json.dumps(stable_json(sdfg.to_json()), sort_keys=True, default=str).encode())
    sdfg.generate_code()
    return dict(kind=kind,
                name=name,
                heuristics=heuristics,
                correct=True,
                taskloops=taskloops,
                median=float('nan'),
                digest=digest.hexdigest())


def measure_one(kind: str, name: str, heuristics: bool, repeats: int, preset: str) -> Dict:
    """Correctness + median time for one kernel at one setting; runs in its own process."""
    ctx = suite.make(kind, name, preset=preset)
    sdfg, taskloops = offloaded_sdfg(ctx, heuristics, f"taskloop_{int(heuristics)}")
    correct = bool(suite.run_matches(ctx, sdfg))

    compiled, kwargs = suite.compiled_call(ctx, sdfg)
    for _ in range(WARMUP):
        compiled(**kwargs)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        compiled(**kwargs)
        times.append(time.perf_counter() - start)
    # The BEST run, not the median. This box reports ``SW Power Cap: Active`` with the SM clock at
    # two thirds of its maximum, and the cap moves during a sweep -- a median tracks whatever the
    # clock was doing, and the two arms of an A/B are minutes apart, so it charges one arm for the
    # other's throttling. The minimum is the one estimator the cap can only spoil in one direction.
    return dict(name=name, heuristics=heuristics, correct=correct, taskloops=taskloops, median=min(times))


def run_case(kind: str, name: str, heuristics: bool, repeats: int, preset: str, stage: str = 'full') -> Dict:
    """Spawn ``measure_one``/``codegen_one`` and read back its JSON; a dead case reports itself
    incorrect. ``stage`` picks which, and with it the timeout: emitting cannot take minutes, so a
    screen that runs long is a hang and must not hold the sweep for the build budget."""
    cmd = [
        sys.executable, '-m', 'tests.corpus.measure_gpu_taskloops', '--one', name, '--kind', kind, '--heuristics',
        str(int(heuristics)), '--repeats',
        str(repeats), '--preset', preset, '--stage', stage
    ]
    budget = CASE_TIMEOUT if stage == 'full' else SCREEN_TIMEOUT
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=budget, cwd=os.getcwd())
    except subprocess.TimeoutExpired:
        return dict(kind=kind,
                    name=name,
                    heuristics=heuristics,
                    correct=False,
                    taskloops=[],
                    median=float('nan'),
                    error=f'timeout after {budget}s')
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
    parser.add_argument('--suites', default='poly,np,tsvc,tsvc25', help='comma-separated: poly, np, tsvc, tsvc25')
    parser.add_argument('--only', help='comma-separated kernel names')
    parser.add_argument('--preset', default='paper', choices=suite.PRESETS)
    parser.add_argument('--repeats', type=int, default=5, help='timed invocations per case')
    parser.add_argument('--stage', default='full', choices=('full', 'codegen'), help='codegen: screen only, no GPU')
    parser.add_argument('--shard', metavar='I/N', default='0/1', help='only every N-th selected kernel starting at I')
    parser.add_argument('--no-screen',
                        action='store_true',
                        help='time both arms even where the screen says the graph is identical')
    parser.add_argument('--csv', help='write the table here as it goes')
    parser.add_argument('--one', help='internal: measure this one kernel in this process')
    parser.add_argument('--kind', default='np', help='internal: the suite for --one')
    parser.add_argument('--heuristics', type=int, default=0, help='internal: the knob for --one')
    args = parser.parse_args()

    if args.one:
        if args.stage == 'codegen':
            row = codegen_one(args.kind, args.one, bool(args.heuristics), args.preset)
        else:
            row = measure_one(args.kind, args.one, bool(args.heuristics), args.repeats, args.preset)
        print('RESULT ' + json.dumps(row), flush=True)
        return 0

    wanted = args.only.split(',') if args.only else None
    suites = args.suites.split(',')
    cases = [(kind, name) for kind, name in suite.kernels() if kind in suites and (wanted is None or name in wanted)]
    index, total = (int(part) for part in args.shard.split('/'))
    cases = cases[index::total]

    rows: List[Dict] = []
    print(f"preset={args.preset} repeats={args.repeats} stage={args.stage} cases={len(cases)}", flush=True)
    print(f"{'suite':6s} {'kernel':28s} {'off':>10s} {'on':>10s} {'ratio':>7s}  taskloops", flush=True)
    for kind, name in cases:
        off_cg = run_case(kind, name, False, args.repeats, args.preset, stage='codegen')
        on_cg = run_case(kind, name, True, args.repeats, args.preset, stage='codegen')
        screened = off_cg.get('digest') is not None and off_cg.get('digest') == on_cg.get('digest')
        if args.stage == 'codegen':
            off, on = off_cg, on_cg
        else:
            off = run_case(kind, name, False, args.repeats, args.preset)
            # Byte-identical emitted programs cannot differ in runtime, so one build answers both.
            on = off if (screened and not args.no_screen) else run_case(kind, name, True, args.repeats, args.preset)
        ratio = 1.0 if on is off else (on['median'] / off['median'] if off['median'] > 0 else float('nan'))
        note = '' if (off['correct'] and on['correct']) else '  ' + (on.get('error') or off.get('error') or 'WRONG')
        if not note and screened:
            note = '  identical graph' + ('' if on is off else ', timed anyway')
        rows.append(
            dict(suite=kind,
                 kernel=name,
                 off=off['median'],
                 on=on['median'],
                 ratio=ratio,
                 off_correct=off['correct'],
                 on_correct=on['correct'],
                 screened=screened,
                 taskloops=' '.join(on_cg['taskloops']),
                 note=note.strip()))
        print(
            f"{kind:6s} {name:28s} {off['median']:10.5f} {on['median']:10.5f} {ratio:7.2f}  "
            f"{' '.join(on_cg['taskloops'])[:36]}{note[:90]}",
            flush=True)
        if args.csv:
            with open(args.csv, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    fired = [r['kernel'] for r in rows if r['taskloops']]
    print(f"heuristics fired on {len(fired)}/{len(rows)}: {', '.join(fired)}", flush=True)
    rewritten = [r['kernel'] for r in rows if not r['screened']]
    print(f"knob rewrites the emitted program for {len(rewritten)}/{len(rows)}: {', '.join(rewritten)}", flush=True)
    wrong = [r['kernel'] for r in rows if not (r['off_correct'] and r['on_correct'])]
    if wrong:
        print(f"INCORRECT or FAILED: {', '.join(wrong)}", flush=True)
    return 1 if wrong else 0


if __name__ == '__main__':
    sys.exit(main())
