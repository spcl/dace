#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Cavity-flow-only codegen comparison: legacy vs experimental (readable) C++
code generator, compiled with BOTH g++ and clang++, run single-core (S preset).

For each (compiler x codegen) lane it reports:
  * codegen_ms  -- DaCe C++ code generation time (generate_code)
  * compile_ms  -- a direct, timed compiler invocation on the generated .cpp
                   (no CMake configure cost -- see engine.compile_sdfg_timed)
  * runtime_ms  -- best-of-N single-core runtime (OMP_NUM_THREADS=1)
  * loc         -- non-blank lines of the generated CPU .cpp

The only thing that differs between the two codegen lanes is
``compiler.cpu.implementation`` (legacy vs experimental); the pipeline is the
project-standard ``dace + simplify + LoopToMap + MapFusion``. Speedup is the
legacy-runtime / experimental-runtime ratio for each compiler (>1 means the
readable codegen is faster) -- readability is expected to be perf-neutral, so
this is a regression guard, not a win.

    python3 run_cavity_compare.py                 # gcc + clang, 10 reps, S preset
    python3 run_cavity_compare.py --reps 20 --cxx g++,clang++,clang++-18
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')  # single-core: the S preset signal
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PERF_JOBS_DIR = os.path.join(REPO_ROOT, 'performance_regression_jobs')
if PERF_JOBS_DIR not in sys.path:
    sys.path.insert(0, PERF_JOBS_DIR)

from dace.config import Config, set_temporary  # noqa: E402
import engine  # noqa: E402
import npbench_polybench_perf as base  # noqa: E402
from run_readable_perf import pipelined_sdfg, sdfg_name, set_implementation  # noqa: E402

KERNEL = 'cavity_flow'
CODEGENS = ('legacy', 'experimental')


def generated_loc(codegen):
    """Non-blank line count of the generated CPU .cpp for one codegen lane."""
    value = codegen
    if codegen == 'experimental':
        # Flag value renamed 'experimental' -> 'experimental_readable'; probe which this build wants.
        from dace.codegen.targets import cpp as cpp_target
        value = 'experimental_readable'
        Config.set('compiler', 'cpu', 'implementation', value=value)
        if not cpp_target.readable_cpu_codegen_active():
            value = 'experimental'
    with set_temporary('compiler', 'cpu', 'implementation', value=value):
        info = base.load_bench_info(KERNEL)
        params = info['parameters']['S']
        program, _arrays, _params = base.build_program_and_data(KERNEL, info, params)
        sdfg = pipelined_sdfg(program, sdfg_name(KERNEL, 'cpu', 'S', codegen) + '_loc', 'cpu')
        cpp = '\n'.join((o.clean_code or o.code) for o in sdfg.generate_code() if o.language == 'cpp')
    return len([ln for ln in cpp.splitlines() if ln.strip()])


def measure(codegen, cxx, reps):
    """(codegen_ms, compile_ms, best_runtime_ms) for one (codegen, compiler) lane,
    at the S dataset size and single core. Runs isolated so a crash fails one lane."""

    def work():
        engine.configure_dace_process()
        Config.set('compiler', 'cpu', 'executable', value=cxx)  # gcc vs clang for THIS lane
        set_implementation(codegen)
        info = base.load_bench_info(KERNEL)
        params = info['parameters']['S']
        program, arrays, params = base.build_program_and_data(KERNEL, info, params)
        sdfg = pipelined_sdfg(program, sdfg_name(KERNEL, 'cpu', 'S', codegen) + '_' + os.path.basename(cxx), 'cpu')
        codegen_ms, cxx_ms = engine.compile_sdfg_timed(sdfg)
        kwargs = base._dace_call_kwargs(sdfg, arrays, params)
        times = engine.time_sdfg(sdfg, kwargs, reps)
        return dict(codegen_ms=codegen_ms, compile_ms=cxx_ms, runtime_ms=(min(times) if times else None))

    # run_isolated returns (ok, payload_or_error); main() wants the payload dict (or None on a
    # crash/timeout), so unpack here rather than leak the tuple into the table formatting.
    ok, payload = engine.run_isolated(work, timeout=900)
    return payload if ok else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reps', type=int, default=10, help='timed single-core reps per lane (best-of; default 10)')
    ap.add_argument('--cxx', default='g++,clang++', help='comma-separated compilers to compare (default g++,clang++)')
    args = ap.parse_args()

    compilers = [c.strip() for c in args.cxx.split(',') if c.strip()]
    resolved = [(c, shutil.which(c)) for c in compilers]
    missing = [c for c, p in resolved if p is None]
    if missing:
        print(f'WARNING: compiler(s) not found, skipping: {", ".join(missing)}')
    compilers = [c for c, p in resolved if p is not None]
    if not compilers:
        print('no usable compiler found; nothing to measure')
        return

    loc = {cg: generated_loc(cg) for cg in CODEGENS}

    print(f'\ncavity_flow  |  preset=S  single-core (OMP_NUM_THREADS={os.environ["OMP_NUM_THREADS"]})  reps={args.reps}')
    print(f'pipeline: dace + simplify + LoopToMap + MapFusion   |   LoC(non-blank): '
          f'legacy={loc["legacy"]}  experimental={loc["experimental"]}\n')
    header = f'{"compiler":10} {"codegen":13} {"codegen_ms":>11} {"compile_ms":>11} {"runtime_ms":>11}'
    print(header)
    print('-' * len(header))

    results = {}
    for cxx in compilers:
        for cg in CODEGENS:
            r = measure(cg, cxx, args.reps)
            results[(cxx, cg)] = r
            if r is None:
                print(f'{os.path.basename(cxx):10} {cg:13} {"FAILED":>11}')
                continue
            rt = f'{r["runtime_ms"]:.3f}' if r['runtime_ms'] is not None else 'n/a'
            print(f'{os.path.basename(cxx):10} {cg:13} {r["codegen_ms"]:11.1f} {r["compile_ms"]:11.1f} {rt:>11}')

    print('\nexperimental-vs-legacy speedup (legacy_runtime / experimental_runtime):')
    for cxx in compilers:
        leg, exp = results.get((cxx, 'legacy')), results.get((cxx, 'experimental'))
        if leg and exp and leg['runtime_ms'] and exp['runtime_ms']:
            print(f'  {os.path.basename(cxx):10} {leg["runtime_ms"] / exp["runtime_ms"]:.3f}x')
        else:
            print(f'  {os.path.basename(cxx):10} n/a')


if __name__ == '__main__':
    main()
