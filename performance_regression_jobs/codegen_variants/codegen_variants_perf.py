#!/usr/bin/env python3
"""Codegen-backend study over the four DaCe build variants:
``{build_mode: cmake, native} x {compiler.cpu.implementation: legacy, experimental_readable}``.

Two orthogonal axes meet here:
  * ``build_mode`` picks how the SAME generated code is built -- CMake configure+build
    vs the direct ``g++``/``nvcc``/link native back-end. It affects the compile wall,
    not the emitted code, so it changes neither codegen time nor runtime.
  * ``cpu.implementation`` picks WHICH C++ is emitted -- ``legacy`` (connector-based
    tasklets) vs ``experimental_readable`` (per-array ``_idx()`` index functions,
    connector-free tasklets). It changes the generated source, so it moves codegen
    time, compile time AND runtime.

So each of the four variants is timed for codegen + compile (all four rows carry
``codegen_ms``/``compile_total_ms``/``build_ms``), and the runtime question -- does the
readable generator emit faster or slower code? -- is answered by holding ``build_mode``
fixed at cmake and comparing ``legacy`` vs ``experimental_readable`` runtime. Every
build is checked against the kernel's NumPy oracle, so a codegen variant that miscompiles
a kernel shows up as ``correct = 0`` rather than a bogus speedup.

Per kernel, four cells are timed (paper CPU lane = simplify + LoopToMap + MapFusion, OpenMP):
  cmake_legacy, native_legacy, cmake_experimental_readable, native_experimental_readable

Kernels self-partition by rank (SLURM_PROCID / SLURM_NTASKS); the reference layout is
1 node x 4 ranks. Reuses the performance_regression_jobs engine + corpus unchanged
(engine.py, npbench_polybench_perf.py) -- nothing about the corpus, sizes, oracle or
isolation is duplicated here; only the (build_mode, implementation) axis is new.

    python3 codegen_variants/codegen_variants_perf.py                    # this rank's slice
    python3 codegen_variants/codegen_variants_perf.py --only gemm --compile-reps 1
    python3 codegen_variants/codegen_variants_perf.py --tables-only      # rebuild the tables
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

# The performance_regression_jobs framework is a flat directory of standalone job scripts
# (not an installed package); this job lives one level down, so add the parent to the path the
# same way the sibling jobs do.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dace.config import set_temporary

import engine
import npbench_polybench_perf as base

# The cmake variants run a real CMake configure inside each MPI rank. srun/mpirun start tasks with
# SIGCHLD BLOCKED, which children inherit; CMake (KWSys) then spins in select() reaping its
# compiler-id helpers, and any child touching a PMI/PMIx client hangs in init. DaCe fixes both
# (compiler._build_subprocess_sigmask unblocks SIGCHLD around the build Popen,
# compiler._build_subprocess_env strips the rank identity) and wires them into the cmake path.
# Refuse to run against a DaCe predating the fix rather than deadlocking the sweep.
from dace.codegen import compiler as dace_compiler

missing_sigchld_fix = [
    n for n in ('_build_subprocess_env', '_build_subprocess_sigmask') if n not in vars(dace_compiler)
]
if missing_sigchld_fix:
    raise SystemExit(f'codegen_variants: this DaCe lacks {missing_sigchld_fix}; the cmake variants would hang under '
                     f'srun (SIGCHLD-blocked mask). Run on the extended branch, which carries the fix.')

CORPUS = base.CORPUS
PRESET = base.PRESET
#: Paper CPU lane. The corpus's own "parallel" pipeline == simplify + LoopToMap + MapFusion.
PIPELINE = 'parallel'
#: The four variants: (build_mode, cpu.implementation). build_mode moves the compile wall only;
#: implementation moves the emitted code (hence codegen time, compile time and runtime).
VARIANTS = (
    ('cmake', 'legacy'),
    ('native', 'legacy'),
    ('cmake', 'experimental_readable'),
    ('native', 'experimental_readable'),
)
#: Thread count for the paper (multi-core) lane, read from the launch environment.
MULTI_THREADS = max(1, int(os.environ.get('OMP_NUM_THREADS', '4')))
FIELDS = ('kernel', 'build_mode', 'implementation', 'codegen_ms', 'codegen_bytes', 'compile_total_ms', 'build_ms',
          'run_ms', 'correct')


def build_variant(program, tag):
    """A freshly named SDFG for one cell. The unique name is the DaCe cache key
    (engine.configure_dace_process sets cache='name'), so each cell maps to its own build
    folder and every ``.compile()`` here is a real cold build, not a no-op reuse. The codegen
    implementation is chosen by the caller's set_temporary, so the emitted code -- and this
    SDFG's generated sources -- reflect the variant under test."""
    sdfg = program.to_sdfg(simplify=True)
    sdfg = engine.PIPELINES[PIPELINE](sdfg)
    sdfg.name = f"{CORPUS}_{sdfg.name}_{PIPELINE}_{tag}"
    return sdfg


def bench_kernel(name, compile_reps, run_reps, timeout):
    """Runs in an isolated subprocess (bad codegen can crash). Returns one row per variant."""
    engine.configure_dace_process()
    from dace.codegen import codegen

    os.environ['OMP_NUM_THREADS'] = str(MULTI_THREADS)
    info = base.load_bench_info(name)
    params = info['parameters'][PRESET]
    program, arrays, params = base.build_program_and_data(name, info, params)
    reference = base._run_numpy(info, arrays, params)

    rows = []
    for index, (mode, impl) in enumerate(VARIANTS):
        with set_temporary('compiler', 'build_mode', value=mode):
            with set_temporary('compiler', 'cpu', 'implementation', value=impl):
                # codegen time -- depends on the implementation (the emitted code), measured in
                # context so each variant is timed with its own generator selected. The size of the
                # emitted C++ (summed over all code objects) is captured too: translation-unit size
                # is a codegen-quality lever in its own right -- it moves compile time and can move
                # runtime -- and it is exactly where legacy and experimental_readable differ.
                codegen_samples = []
                generated_bytes = 0
                for rep in range(compile_reps):
                    sdfg = build_variant(program, f'{index}_cg{rep}')
                    t0 = time.perf_counter()
                    objects = codegen.generate_code(sdfg)
                    codegen_samples.append((time.perf_counter() - t0) * 1000.0)
                    generated_bytes = sum(len(obj.clean_code) for obj in objects)

                # full compile wall, cold each rep via a unique name, under this (mode, impl).
                compile_samples = []
                for rep in range(compile_reps):
                    sdfg = build_variant(program, f'{index}_c{rep}')
                    t0 = time.perf_counter()
                    sdfg.compile()
                    compile_samples.append((time.perf_counter() - t0) * 1000.0)

                # runtime + correctness on a compiled build of this exact variant.
                sdfg = build_variant(program, f'{index}_run')
                call_kwargs = base._dace_call_kwargs(sdfg, arrays, params)
                got = base._collect_outputs(info['output_args'], sdfg.compile()(**call_kwargs), call_kwargs)
                correct = base._compare(reference, got)
                run_samples = engine.time_sdfg(sdfg, call_kwargs, run_reps, time_budget_s=0.4 * timeout)

        codegen_ms = min(codegen_samples)
        compile_ms = min(compile_samples)
        rows.append({
            'kernel': name,
            'build_mode': mode,
            'implementation': impl,
            'codegen_ms': round(codegen_ms, 3),
            'codegen_bytes': generated_bytes,
            'compile_total_ms': round(compile_ms, 3),
            'build_ms': round(max(0.0, compile_ms - codegen_ms), 3),
            'run_ms': round(min(run_samples), 4) if run_samples else '',
            'correct': int(bool(correct)),
        })
    return rows


def results_csv(results_dir):
    return os.path.join(results_dir, CORPUS, 'codegen_variants.csv')


def append_rows(results_dir, rows):
    """Append this kernel's rows, refusing to write into a file whose columns are not FIELDS.

    csv.DictWriter writes values positionally and does not check them against the header already on
    disk, so appending to a results file written by an older column set silently shifts every field
    (a codegen_bytes landing under compile_total_ms, a run_ms under correct) -- the tables then report
    numbers that are confidently wrong, and the correctness audit calls a fine kernel miscompiled.
    Fail loudly instead; the fix is to move the stale file aside.
    """
    path = results_csv(results_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    if exists:
        with open(path, newline='') as fp:
            header = next(csv.reader(fp), [])
        if tuple(header) != FIELDS:
            raise SystemExit(f'{path} has columns {tuple(header)} but this job writes {FIELDS}; appending would '
                             f'misalign every row. Move the stale file aside and re-run.')
    with open(path, 'a', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def num(row, key):
    try:
        return float(row[key])
    except (ValueError, KeyError, TypeError):
        return None


def ratio(numerator, denominator):
    """``numerator / denominator``, or None when either is missing or the divisor is zero.

    Deliberately not a truthiness test: a measured 0.0 is data, not absence, and ``if a and b``
    would silently drop it from the aggregate instead of counting it (or reporting it as
    unmeasurable). Only a genuinely absent value (None) or a zero divisor yields None.
    """
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def write_tables(results_dir):
    """Two tables: (1) codegen + compile wall for all four variants per kernel; (2) the runtime
    question -- cmake legacy vs cmake experimental_readable, the codegen-implementation effect on
    the produced binary's speed."""
    path = results_csv(results_dir)
    if not os.path.isfile(path):
        print(f'no rows at {path}')
        return
    with open(path, newline='') as fp:
        rows = list(csv.DictReader(fp))

    cells = {}  # kernel -> {(mode, impl): row}
    for row in rows:
        cells.setdefault(row['kernel'], {})[(row['build_mode'], row['implementation'])] = row

    lines = [
        f'# Codegen variants -- {CORPUS} @ {PRESET} preset',
        '',
        '## Codegen + compile wall (ms) per variant',
        '',
        'codegen_ms and codegen_bytes (size of the emitted C++) depend on the implementation; build_ms '
        '(= compile_total - codegen) depends on build_mode (cmake configure vs native) and on code size. '
        'correct = the variant matched NumPy.',
        '',
        '| kernel | variant | codegen ms | codegen B | compile ms | build ms | run ms | correct |',
        '|---|---|--:|--:|--:|--:|--:|:-:|',
    ]
    for kernel in sorted(cells):
        for mode, impl in VARIANTS:
            row = cells[kernel].get((mode, impl))
            if not row:
                continue
            lines.append(f"| {kernel} | {mode}/{impl} | {row['codegen_ms']} | {row.get('codegen_bytes', '')} | "
                         f"{row['compile_total_ms']} | {row['build_ms']} | {row['run_ms']} | "
                         f"{'yes' if row['correct'] == '1' else 'NO'} |")

    # Runtime: legacy vs experimental_readable under cmake (build_mode-independent, so cmake is the
    # clean pairing the user asked for). speedup = legacy / experimental (>1 = experimental faster).
    lines += [
        '',
        '## Runtime: cmake legacy vs cmake experimental_readable',
        '',
        'run_ms of the two codegen implementations, built under cmake. speedup = legacy / experimental '
        '(>1 = the readable generator produced faster code). Only rows where BOTH variants matched NumPy.',
        '',
        '| kernel | legacy run ms | experimental run ms | speedup | both correct |',
        '|---|--:|--:|--:|:-:|',
    ]
    speedups = []
    for kernel in sorted(cells):
        leg = cells[kernel].get(('cmake', 'legacy'))
        exp = cells[kernel].get(('cmake', 'experimental_readable'))
        if not leg or not exp:
            continue
        both_ok = leg['correct'] == '1' and exp['correct'] == '1'
        rl, re_ = num(leg, 'run_ms'), num(exp, 'run_ms')
        sx = ratio(rl, re_) if both_ok else None
        if sx is not None:
            speedups.append(sx)
        lines.append(f"| {kernel} | {leg['run_ms']} | {exp['run_ms']} | "
                     f"{sx:.3f} | {'yes' if both_ok else 'NO'} |" if sx is not None else
                     f"| {kernel} | {leg['run_ms']} | {exp['run_ms']} | - | {'yes' if both_ok else 'NO'} |")

    # codegen + compile speedups across the two build_modes, for the record (native vs cmake, per impl).
    def build_speedup(impl):
        vals = []
        for kernel in sorted(cells):
            cm = cells[kernel].get(('cmake', impl))
            nat = cells[kernel].get(('native', impl))
            bc, bn = (num(cm, 'build_ms') if cm else None), (num(nat, 'build_ms') if nat else None)
            sx = ratio(bc, bn)
            if sx is not None:
                vals.append(sx)
        return vals

    if speedups:
        lines += [
            '', f'**runtime experimental vs legacy (cmake)**: geomean '
            f'{statistics.geometric_mean(speedups):.3f}x over {len(speedups)} kernels, '
            f'median {statistics.median(speedups):.3f}x (>1 = experimental faster).'
        ]
    # Code-size lever: how much smaller/larger the readable generator's C++ is than legacy's.
    size_ratios = []
    for kernel in sorted(cells):
        leg, exp = cells[kernel].get(('cmake', 'legacy')), cells[kernel].get(('cmake', 'experimental_readable'))
        bl, be = (num(leg, 'codegen_bytes') if leg else None), (num(exp, 'codegen_bytes') if exp else None)
        sx = ratio(be, bl)
        if sx is not None:
            size_ratios.append(sx)
    if size_ratios:
        lines.append(f'**codegen size experimental vs legacy**: geomean {statistics.geometric_mean(size_ratios):.3f}x '
                     f'over {len(size_ratios)} kernels (<1 = readable generator emits less C++).')
    for impl in ('legacy', 'experimental_readable'):
        bs = build_speedup(impl)
        if bs:
            lines.append(f'**native vs cmake build speedup ({impl})**: geomean '
                         f'{statistics.geometric_mean(bs):.2f}x over {len(bs)} kernels.')

    # Correctness audit: the readable generator is newer, so flag every kernel where any variant
    # missed NumPy -- the first full sweep doubles as an experimental-codegen correctness check.
    failing = sorted(k for k, by_mode in cells.items() if any(r.get('correct') != '1' for r in by_mode.values()))
    lines += ['', '## Correctness audit', '']
    if failing:
        lines.append(f'{len(failing)} kernel(s) with a variant that did NOT match NumPy:')
        for kernel in failing:
            bad = [f"{m}/{i}" for (m, i), r in sorted(cells[kernel].items()) if r.get('correct') != '1']
            lines.append(f'- `{kernel}`: {", ".join(bad)}')
    else:
        lines.append('All variants matched NumPy on every kernel.')

    out = os.path.join(os.path.dirname(path), 'codegen_variants.md')
    with open(out, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    engine.add_common_args(ap, default_timeout=900.0)
    ap.add_argument('--compile-reps', type=int, default=3, help='cold-compile samples per cell (default: 3)')
    ap.add_argument('--run-reps', type=int, default=10, help='runtime samples per cell (default: 10)')
    args = ap.parse_args()

    # Every cell reports min() over its samples, so a zero rep count would die on an empty sequence
    # deep inside the isolated subprocess rather than here.
    if args.compile_reps < 1:
        ap.error('--compile-reps must be >= 1 (each cell reports the minimum over its samples)')
    if args.run_reps < 1:
        ap.error('--run-reps must be >= 1')

    if args.list_kernels:
        print('\n'.join(base.kernel_list(args)))
        return
    if args.tables_only:
        write_tables(args.results_dir)
        return

    engine.export_cxx_override(args)
    rank, world = engine.get_world_rank(), engine.get_world_size()
    all_kernels = base.kernel_list(args)
    explicit = engine.explicit_kernel_list(args)
    mine = explicit if explicit is not None else engine.my_slice(all_kernels, rank, world)
    print(f'rank {rank}/{world}: {len(mine)}/{len(all_kernels)} kernels (codegen-variants)')

    for name in mine:
        ok, payload = engine.run_isolated(bench_kernel, (name, args.compile_reps, args.run_reps, args.timeout),
                                          timeout=args.timeout)
        if ok:
            append_rows(args.results_dir, payload)
            print(f'[{name}] done ({len(payload)} rows)')
        else:
            print(f'[{name}] failed: {payload}')

    write_tables(args.results_dir)


if __name__ == '__main__':
    main()
