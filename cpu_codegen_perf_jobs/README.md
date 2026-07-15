# cpu_codegen_perf_jobs

Submittable daint.alps jobs comparing the **legacy** vs **experimental (readable)**
C++/CUDA code generator (`compiler.cpu.implementation = legacy | experimental`), over the
NPBench + PolyBench + TSVC2 + TSVC2.5 corpora and on `cavity_flow` specifically.

The only thing that differs between the two lanes is the code generator; the pipeline
is the project-standard **`dace + simplify + LoopToMap + MapFusion`** (`+ len-1 →
scalar` scalarization), and measurement always happens **after** it. Readability is
expected to be perf-neutral, so the runtime comparison is a regression guard, not a
claimed win — the win is in generated lines-of-code / readability.

## Layout

| file | what |
|------|------|
| `submit_daint_readable.sbatch` | **THE job** — every phase, one node, one submission, one merged TSV. See below. |
| `run_readable_perf.py` | npbench+polybench sweep driver: per (kernel × codegen) → codegen time, cold compile time, best-of-N runtime, correctness (experimental compared against legacy). CPU **and** GPU (`--target`). Self-partitions across MPI ranks via `SLURM_PROCID` / `SLURM_NTASKS`. |
| `run_readable_tsvc2_perf.py` | the same, for the TSVC2 corpus (CPU only). |
| `run_readable_tsvc2_5_perf.py` | the same, for the TSVC2.5 corpus (CPU only). |
| `run_readable_compare.py` | codegen A/B on generated sources (readability / LoC side). |
| `run_cavity_compare.py` | `cavity_flow`-only: legacy vs experimental, compiled with **both g++ and clang++**, single-core (S preset) — reports codegen / compile / runtime + speedup + LoC. |
| `plot_codegen_perf.py` | figures from the merged TSV (filters on `phase` / `cxx`). |

Reuses the sibling `../performance_regression_jobs` harness (`engine.py` +
`npbench_polybench_perf.py`) for kernel discovery, dataset presets, isolated
(crash-safe) timing and the numpy correctness oracle. Keep that folder next to this
one, with its submodules initialized:

```bash
git submodule update --init dace/external/moodycamel dace/external/cub
```

## The job

`submit_daint_readable.sbatch` runs **5 sweeps** on 1 node, in this order, each over all
four corpora with **both** codegens:

| # | phase | preset | layout | host cxx |
|---|-------|--------|--------|----------|
| 1 | `single_core` | S | 4 ranks (1/Grace socket), `OMP_NUM_THREADS=1`, kernels split round-robin | `g++` |
| 2 | `single_core` | S | ″ | `clang++` |
| 3 | `multi_core` | paper | 1 rank, all 72 cores of one socket (`OMP_NUM_THREADS=72`) | `g++` |
| 4 | `multi_core` | paper | ″ | `clang++` |
| 5 | `gpu` | paper | 1 rank + 1 GPU; **nvcc** always compiles the device code | `g++-14` (nvcc's host compiler) |

Phase 5 runs once: nvcc dictates its host compiler (CUDA rejects the spack gcc 16 the CPU
phases use), so there is no g++/clang++ sweep there — the host compiler is still recorded.
It covers npbench+polybench only, as the two tsvc runners are CPU-only.

Every row of every phase lands in **one** merged TSV (`$OUT`, default `perfresults.tsv`):

```
kernel  corpus  codegen  preset  threads  cxx  phase  codegen_ms  compile_ms  runtime_ms  speedup  correctness  error
```

`phase` (`single_core|multi_core|gpu`), `cxx` (`g++|clang++|g++-14`) and `corpus`
(`npbench|polybench|tsvc2|tsvc2_5`) are what keep the merged rows groupable.

```bash
sbatch submit_daint_readable.sbatch

# knobs (all optional):
sbatch --export=ALL,REPS=3,CODEGEN=experimental,OUT=quick.tsv submit_daint_readable.sbatch
sbatch --export=ALL,CPU_CXX_LIST="g++" submit_daint_readable.sbatch   # skip the clang++ sweeps
```

Every `#SBATCH --account` / `--partition` / `--chdir` and the `REPO_ROOT` / venv /
spack lines are marked `TODO` — set them to your daint.alps account and synced repo
path before submitting.

## Presets

- **S** — dataset size `S`, **single core** (`OMP_NUM_THREADS=1`): a quick regression signal.
- **paper** — paper dataset sizes, **multi-core**: the writeup numbers.

## Run locally (no SLURM)

```bash
cd cpu_codegen_perf_jobs
PYTHONPATH=<repo> OMP_NUM_THREADS=1 python3 run_cavity_compare.py --reps 10 --cxx g++,clang++
PYTHONPATH=<repo> python3 run_readable_perf.py --preset S --corpus npbench --out s.tsv
PYTHONPATH=<repo> python3 run_readable_perf.py --preset S --only arc_distance --cxx g++ --out probe.tsv
```

`--cxx` selects the host compiler (relayed to every measurement subprocess as
`DACE_PERF_CXX` → `compiler.cpu.executable`) and is what the `cxx` column records;
`--phase` overrides the `phase` column, which otherwise derives from `--target` and the
OpenMP width.
