# local/ — run the readable-codegen perf jobs on THIS machine

Single-node, single-rank, direct (`./script.sh`) runners for the daint.alps sbatch jobs in
`../cpu_codegen_perf_jobs` and `../codegen_buildperf_jobs`. No SLURM, no `srun`, no modules/spack:
they use this machine's compilers and system BLAS, one rank.

Each script mirrors the sbatch's CPU phases:

| phase | preset | dataset | OpenMP threads |
|-------|--------|---------|----------------|
| `single_core` | `S` | small | 1 |
| `multi_core` | `paper` | full | `THREADS` (default **8**) |

Both phases run by default and land in ONE merged TSV. Every phase runs the same pipeline
(`dace + simplify + LoopToMap + MapFusion + ConvertLengthOneArraysToScalars`) and varies only
`compiler.cpu.implementation` (legacy | experimental).

## Scripts

- **`./run_local_job.sh`** — THE job. Compiles all FOUR variants of every kernel
  (`cmake-oldcpu`, `cmake-newcpu`, `native-oldcpu`, `native-newcpu` = `{compiler.build_mode}` ×
  `{legacy, experimental_readable}`), times each with **25 reps reporting the MEDIAN** on **8
  threads**, then plots the headline speedup:

      speedup = median(cmake-oldcpu) / median(cmake-newcpu)      (>1 => the new codegen is faster)

  → `local_compare.tsv` + `plots/speedup_cmake_newcpu_vs_oldcpu.{png,pdf}`.
  Cold-builds by default (`KEEP_CACHE=1` to reuse the `/dev/shm` build cache).
- `./run_local.sh` — RUNTIME + readability comparison over npbench/polybench/tsvc2/tsvc2_5 →
  `perfresults_local.tsv` + `codegen_metrics_local.csv`.
- `./run_local_buildperf.sh` — codegen-time + compile-time + generated-size + readability, INLINE
  per row, plus runtime, CPU-only → `buildperfresults_local.tsv`.

### `run_local_job.sh` knobs

| var | default | meaning |
|-----|---------|---------|
| `THREADS` | `8` | OpenMP threads |
| `REPS` | `25` | timed reps per variant (reduced by **median**) |
| `PRESET` | `paper` | `paper` (full sizes) or `S` (small) |
| `CORPUS` | `both` | `npbench` \| `polybench` \| `both` |
| `ONLY` / `KERNELS` | _(none)_ | substring filter / explicit comma-separated list |
| `CXX` | `g++` | host compiler |
| `CPP_STD` | `20` | C++ standard digits |
| `CONST_SCALAR_ABI` | `by_ref` | read-only scalar binding: `by_ref` (legacy ABI) or `by_value` |
| `KEEP_CACHE` | `0` | `1` reuses the build cache (skips the cold rebuild) |
| `TIMEOUT` | `900` | per-variant subprocess timeout (s) |

```bash
cd local
./run_local_job.sh                        # 4 variants, 25 reps median, 8 threads, paper, + speedup plot
ONLY=atax REPS=5 PRESET=S ./run_local_job.sh   # quick single-kernel check
CONST_SCALAR_ABI=by_value ./run_local_job.sh   # measure the const-value scalar binding instead
```

## Usage

```bash
cd local
./run_local.sh                                   # single_core (S) + multi_core (paper, 8 threads), both codegens, g++
PHASES=multi_core ./run_local.sh                 # only the 8-thread paper lane
PHASES=single_core ONLY=atax ./run_local.sh      # quick single-kernel single-core check
THREADS=16 ./run_local.sh                         # 16-thread multi_core
CXX=clang++ ./run_local.sh                        # clang instead of g++
TARGET=gpu ./run_local.sh                         # GPU lane (paper only; needs nvcc + a CUDA device)
./run_local_buildperf.sh                          # codegen/compile-time suite
```

## Knobs (env vars)

| var | default | meaning |
|-----|---------|---------|
| `PHASES` | `single_core multi_core` | which phases to run (space-separated) |
| `THREADS` | `8` | OpenMP threads for the **multi_core** lane (single_core is always 1) |
| `CODEGEN` | `both` | `legacy` \| `experimental` \| `both` |
| `CORPUS` | `both` | `npbench` \| `polybench` \| `both` (run_local.sh) |
| `TARGET` | `cpu` | `cpu` \| `gpu` — gpu runs the paper lane only (run_local.sh) |
| `REPS` | `5` | timed reps per lane (best-of) |
| `CXX` | `g++` | host compiler (`g++`, `clang++`, ...) |
| `CPP_STD` | `20` | C++ standard digits (`20` → consteval; `17` → constexpr) |
| `ONLY` | _(none)_ | substring filter on kernel name (e.g. `ONLY=atax`) |
| `PYTHON` | `python3` | interpreter |
| `OUT` | `local/*_local.tsv` | merged TSV path |

## Notes

- **Single rank, single node.** Runs pass `--rank 0 --num-ranks 1` (and clear any stray `SLURM_*`),
  so one process sweeps every kernel — no `srun` / MPI.
- **Local paths.** `PYTHONPATH` = repo root (DaCe finds its own `dace` package; the runners add the
  sibling `performance_regression_jobs` themselves). `DACE_PERF_CXX` picks the host compiler;
  `DACE_PERF_CXX_STD` / `DACE_compiler_cpp_standard` pin the C++ standard for both the timed direct
  compile and the CMake correctness build. System OpenBLAS/LAPACK (on the ldconfig cache) link by
  default — the SDFG's own BLAS/LAPACK environments carry the flags (a Debian soname like
  `liblapacke.so.3` is resolved to a path / `-l<stem>` by the harness).
- Requires the sibling `../performance_regression_jobs` harness with its submodules initialised:
  `git submodule update --init dace/external/moodycamel dace/external/cub`.
- Plot the results with the `plot_*.py` scripts printed at the end of each run.
