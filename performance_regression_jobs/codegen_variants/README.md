# codegen_variants

The four DaCe build variants, `{build_mode: cmake, native} × {compiler.cpu.implementation:
legacy, experimental_readable}` — codegen + C++ compile wall for all four, and the runtime
effect of the codegen implementation.

Two orthogonal axes:

| axis                     | what it changes                                    | affects            |
|--------------------------|----------------------------------------------------|--------------------|
| `build_mode`             | how the SAME code is built (CMake vs direct g++)   | compile wall only  |
| `cpu.implementation`     | WHICH C++ is emitted (legacy vs readable `_idx()`) | codegen, compile, runtime |

Because `build_mode` never touches the emitted code, the runtime question — *does the readable
generator produce faster or slower code?* — is answered by holding `build_mode` at cmake and
comparing `legacy` vs `experimental_readable`. Every build is checked against the kernel's NumPy
oracle, so a codegen variant that miscompiles a kernel shows up as `correct = 0`, not a fake speedup.

Reuses the parent `performance_regression_jobs` framework unchanged (`engine.py`,
`npbench_polybench_perf.py`): same NPBench+PolyBench corpus, paper preset, NumPy oracle, rank
self-partition and subprocess isolation. Only the (build_mode, implementation) axis is new.

## What it measures

Per kernel, four cells (paper CPU lane = `simplify + LoopToMap + MapFusion`, OpenMP):
`cmake/legacy`, `native/legacy`, `cmake/experimental_readable`, `native/experimental_readable`.
Each records `codegen_ms`, `compile_total_ms`, `build_ms` (`= compile_total − codegen`),
`run_ms` (best of N) and `correct`.

## Run

1 node × 4 ranks (kernels self-partition by `SLURM_PROCID`/`SLURM_NTASKS`):

```bash
sbatch slurm_codegen_variants.sh          # adjust the SBATCH header for your account
```

Without a scheduler (one rank):

```bash
python3 codegen_variants/codegen_variants_perf.py --compile-reps 3 --run-reps 10
python3 codegen_variants/codegen_variants_perf.py --only gemm       # single kernel
python3 codegen_variants/codegen_variants_perf.py --tables-only     # rebuild the tables
```

`native` mode and the cmake variants both need `nvcc` + `g++` (+ `cmake` for the cmake variants)
on `PATH`; the slurm script sets the same toolchain env as `slurm_npbench_polybench_compile.sh`.

## Output

```
results/npbench_polybench/codegen_variants.csv   one row per (kernel, build_mode, implementation)
results/npbench_polybench/codegen_variants.md    (1) codegen+compile per variant  (2) legacy vs
                                                 experimental runtime + native-vs-cmake build speedup
```
