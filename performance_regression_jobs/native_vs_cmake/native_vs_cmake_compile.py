#!/usr/bin/env python3
"""DaCe build back-end study: ``compiler.build_mode = cmake`` (BEFORE) vs
``native`` (AFTER), over the whole NPBench+PolyBench corpus at the paper preset,
single-core and multi-core.

Native mode (``dace/codegen/native_compiler.py``) emits the ``g++``/``nvcc``/link
commands directly and runs them via subprocess, skipping the CMake configure step
that dominates a small kernel's build wall-clock. This job quantifies that on the
SAME generated code: codegen time is back-end independent, the compile wall is the
win, and runtime must be unchanged -- so every measured build is also checked
against the kernel's NumPy oracle.

Per kernel, four builds are timed -- {single, multi} core x {cmake, native}:
  * multi-core  = the paper CPU lane: simplify + LoopToMap + MapFusion (OpenMP maps)
  * single-core = the same SDFG with every map forced to Sequential (serial C++)
Each row records codegen_ms, compile_total_ms, build_ms (= compile_total - codegen),
run_ms (best of N) and whether the result matched NumPy.

Kernels self-partition by rank (SLURM_PROCID / SLURM_NTASKS); the reference layout
is 1 node x 4 ranks. Reuses the performance_regression_jobs engine + corpus
unchanged (engine.py, npbench_polybench_perf.py) -- nothing about the corpus,
sizes, oracle or isolation is duplicated here.

    python3 native_vs_cmake_compile.py                       # this rank's slice
    python3 native_vs_cmake_compile.py --only gemm --compile-reps 1
    python3 native_vs_cmake_compile.py --tables-only         # rebuild native_vs_cmake.md
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import csv
import statistics
import sys
import time

# The performance_regression_jobs framework is a flat directory of standalone job
# scripts (not an installed package); its own drivers put that directory on the
# path the same way. This job lives one level down, so add the parent.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dace  # noqa: F401
from dace import ScheduleType
from dace.config import set_temporary
from dace.sdfg import nodes as dace_nodes

import engine
import npbench_polybench_perf as base

CORPUS = base.CORPUS
PRESET = base.PRESET
#: Paper CPU lane. The corpus's own "parallel" pipeline == simplify + LoopToMap + MapFusion.
PIPELINE = 'parallel'
#: BEFORE (status quo, CMake configure + build) vs AFTER (direct g++/nvcc/link).
BUILD_MODES = ('cmake', 'native')
CORES = ('single', 'multi')
#: Thread count for the multi-core lane, taken from the launch environment before the per-cell loop
#: rewrites OMP_NUM_THREADS (single-core forces 1). Same knob the runtime and OpenMP maps honor.
MULTI_THREADS = max(1, int(os.environ.get('OMP_NUM_THREADS', '4')))
FIELDS = ('kernel', 'cores', 'build_mode', 'codegen_ms', 'compile_total_ms', 'build_ms', 'run_ms', 'correct')


def force_sequential(sdfg):
    """Turn the multi-core (OpenMP) SDFG into its single-core form: every map runs
    on one thread, so codegen emits plain serial loops (no ``#pragma omp``)."""
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace_nodes.MapEntry):
            node.map.schedule = ScheduleType.Sequential
    return sdfg


def build_variant(program, cores, mode, tag):
    """A freshly named SDFG for one (cores, build_mode) cell. The unique name is
    the DaCe cache key (engine.configure_dace_process sets cache='name'), so each
    cell maps to its own build folder and every ``.compile()`` here is a real cold
    build rather than a no-op reuse of another cell's artifact."""
    sdfg = program.to_sdfg(simplify=True)
    sdfg = engine.PIPELINES[PIPELINE](sdfg)
    if cores == 'single':
        force_sequential(sdfg)
    sdfg.name = f"{CORPUS}_{sdfg.name}_{PIPELINE}_{cores}_{mode}_{tag}"
    return sdfg


def bench_kernel(name, compile_reps, run_reps, timeout):
    """Runs in an isolated subprocess (bad codegen can crash). Returns a list of
    result rows, one per (cores, build_mode)."""
    engine.configure_dace_process()
    from dace.codegen import codegen

    info = base.load_bench_info(name)
    params = info['parameters'][PRESET]
    program, arrays, params = base.build_program_and_data(name, info, params)
    reference = base._run_numpy(info, arrays, params)

    rows = []
    for cores in CORES:
        os.environ['OMP_NUM_THREADS'] = '1' if cores == 'single' else str(MULTI_THREADS)
        for mode in BUILD_MODES:
            with set_temporary('compiler', 'build_mode', value=mode):
                # codegen time (back-end independent; measured in-context for a fair pairing)
                codegen_samples = []
                for _ in range(compile_reps):
                    sdfg = build_variant(program, cores, mode, f'cg{_}')
                    t0 = time.perf_counter()
                    codegen.generate_code(sdfg)
                    codegen_samples.append((time.perf_counter() - t0) * 1000.0)

                # full compile wall (cold each rep via a unique name) under this build_mode
                compile_samples = []
                for rep in range(compile_reps):
                    sdfg = build_variant(program, cores, mode, f'c{rep}')
                    t0 = time.perf_counter()
                    sdfg.compile()
                    compile_samples.append((time.perf_counter() - t0) * 1000.0)

                # runtime + correctness on a compiled build of this exact variant
                sdfg = build_variant(program, cores, mode, 'run')
                call_kwargs = base._dace_call_kwargs(sdfg, arrays, params)
                got = base._collect_outputs(info['output_args'], sdfg.compile()(**call_kwargs), call_kwargs)
                correct = base._compare(reference, got)
                run_samples = engine.time_sdfg(sdfg, call_kwargs, run_reps, time_budget_s=0.4 * timeout)

            codegen_ms = min(codegen_samples)
            compile_ms = min(compile_samples)
            rows.append({
                'kernel': name,
                'cores': cores,
                'build_mode': mode,
                'codegen_ms': round(codegen_ms, 3),
                'compile_total_ms': round(compile_ms, 3),
                'build_ms': round(max(0.0, compile_ms - codegen_ms), 3),
                'run_ms': round(min(run_samples), 4) if run_samples else '',
                'correct': int(bool(correct)),
            })
    return rows


def results_csv(results_dir):
    return os.path.join(results_dir, CORPUS, 'native_vs_cmake.csv')


def append_rows(results_dir, rows):
    path = results_csv(results_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.isfile(path)
    with open(path, 'a', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_table(results_dir):
    """Aggregate all ranks' rows into a before/after markdown table: native speedup
    = cmake / native for compile, build and codegen; runtime ratio as a sanity check."""
    path = results_csv(results_dir)
    if not os.path.isfile(path):
        print(f'no rows at {path}')
        return
    with open(path, newline='') as fp:
        rows = list(csv.DictReader(fp))

    def num(row, key):
        try:
            return float(row[key])
        except (ValueError, KeyError):
            return None

    cells = {}  # (kernel, cores) -> {mode: row}
    for row in rows:
        cells.setdefault((row['kernel'], row['cores']), {})[row['build_mode']] = row

    lines = [
        f'# Native vs CMake build back-end -- {CORPUS} @ {PRESET} preset',
        '',
        'compile / build / codegen columns are BEFORE(cmake) -> AFTER(native) in ms; '
        'speedup = cmake / native (>1 = native faster). run_x = native/cmake runtime ratio '
        '(~1 expected: same code). correct = both back-ends matched NumPy.',
        '',
        '| kernel | cores | compile cmake | compile native | compile x | build cmake | build native | '
        'build x | codegen ms | run x | correct |',
        '|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|',
    ]
    compile_speedups, build_speedups = [], []
    for (kernel, cores), by_mode in sorted(cells.items()):
        cm, nat = by_mode.get('cmake'), by_mode.get('native')
        if not cm or not nat:
            continue
        cc, cn = num(cm, 'compile_total_ms'), num(nat, 'compile_total_ms')
        bc, bn = num(cm, 'build_ms'), num(nat, 'build_ms')
        rc, rn = num(cm, 'run_ms'), num(nat, 'run_ms')
        cx = cc / cn if cc and cn else None
        bx = bc / bn if bc and bn else None
        rx = rn / rc if rc and rn else None
        if cx:
            compile_speedups.append(cx)
        if bx:
            build_speedups.append(bx)
        both_ok = 'yes' if (cm.get('correct') == '1' and nat.get('correct') == '1') else 'NO'
        lines.append(f"| {kernel} | {cores} | {cc:.0f} | {cn:.0f} | {cx:.2f} | {bc:.0f} | {bn:.0f} | "
                     f"{bx:.2f} | {num(nat, 'codegen_ms'):.0f} | {rx:.2f} | {both_ok} |" if all(v is not None for v in (
                         cc, cn, cx, bc, bn, bx,
                         rx)) else f"| {kernel} | {cores} | {cc} | {cn} | {cx} | {bc} | {bn} | {bx} | | | {both_ok} |")

    if compile_speedups:
        lines += [
            '',
            f'**compile speedup (native vs cmake)**: geomean '
            f'{statistics.geometric_mean(compile_speedups):.2f}x over {len(compile_speedups)} variants, '
            f'median {statistics.median(compile_speedups):.2f}x.',
            f'**build-only speedup**: geomean {statistics.geometric_mean(build_speedups):.2f}x.'
            if build_speedups else '',
        ]
    out = os.path.join(os.path.dirname(path), 'native_vs_cmake.md')
    with open(out, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    engine.add_common_args(ap, default_timeout=900.0)
    ap.add_argument('--compile-reps', type=int, default=3, help='cold-compile samples per cell (default: 3)')
    ap.add_argument('--run-reps', type=int, default=10, help='runtime samples per cell (default: 10)')
    args = ap.parse_args()

    if args.list_kernels:
        print('\n'.join(base.kernel_list(args)))
        return
    if args.tables_only:
        write_table(args.results_dir)
        return

    engine.export_cxx_override(args)
    rank, world = engine.get_world_rank(), engine.get_world_size()
    all_kernels = base.kernel_list(args)
    explicit = engine.explicit_kernel_list(args)
    mine = explicit if explicit is not None else engine.my_slice(all_kernels, rank, world)
    print(f'rank {rank}/{world}: {len(mine)}/{len(all_kernels)} kernels (native-vs-cmake)')

    for name in mine:
        ok, payload = engine.run_isolated(bench_kernel, (name, args.compile_reps, args.run_reps, args.timeout),
                                          timeout=args.timeout)
        if ok:
            append_rows(args.results_dir, payload)
            print(f'[{name}] done ({len(payload)} rows)')
        else:
            print(f'[{name}] failed: {payload}')

    write_table(args.results_dir)


if __name__ == '__main__':
    main()
