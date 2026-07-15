# native_vs_cmake

`compiler.build_mode = cmake` (**before**) vs `native` (**after**) — how much the
no-CMake native build back-end (`dace/codegen/native_compiler.py`) speeds up
DaCe's build, on the same generated code.

Reuses the parent `performance_regression_jobs` framework unchanged (`engine.py`,
`npbench_polybench_perf.py`): same NPBench+PolyBench corpus, same paper preset,
same NumPy oracle, same rank self-partition and subprocess isolation. Only the
build back-end axis is new here.

## What it measures

Per kernel, four cells — `{single, multi} core × {cmake, native}`:

| cores        | SDFG                                                        |
|--------------|------------------------------------------------------------|
| `multi`      | paper CPU lane: `simplify + LoopToMap + MapFusion` (OpenMP) |
| `single`     | same SDFG, every map forced `Sequential` (serial C++)      |

Each cell records `codegen_ms`, `compile_total_ms`, `build_ms`
(`= compile_total − codegen`), `run_ms` (best of N) and whether the result
matched NumPy. codegen is back-end independent; **`build_ms` cmake vs native is
the win**; runtime must be unchanged (the correctness check guards that).

## Run

1 node × 4 ranks (kernels self-partition by `SLURM_PROCID`/`SLURM_NTASKS`):

```bash
sbatch slurm_native_vs_cmake.sh          # adjust the SBATCH header for your account
```

Without a scheduler (one rank):

```bash
python3 native_vs_cmake/native_vs_cmake_compile.py --compile-reps 3 --run-reps 10
python3 native_vs_cmake/native_vs_cmake_compile.py --only gemm      # single kernel
python3 native_vs_cmake/native_vs_cmake_compile.py --tables-only    # rebuild the table
```

`native` mode needs `nvcc` + `g++` on `PATH` exactly like the CMake path; the
slurm script sets the same toolchain env as `slurm_npbench_polybench_compile.sh`.

## Output

```
results/npbench_polybench/native_vs_cmake.csv    one row per (kernel, cores, build_mode)
results/npbench_polybench/native_vs_cmake.md      before→after table + geomean compile speedup
```
