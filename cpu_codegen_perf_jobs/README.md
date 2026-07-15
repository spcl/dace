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
| `readability_metrics.py` | lizard/local readability metrics (`nloc`, `max_nesting`, `tokens`, `tokens_per_stmt`, `max_ccn`) over generated `.cpp`/`.cu`; writes the `--csv` that feeds FIGURE B. |
| `plot_codegen_perf.py` | the two figures from the merged TSV (+ the metrics CSV); groups by `phase` / `cxx`. |

**Standalone tools (NOT part of the one SLURM job).** These use a *different* stack — the
in-tree `tests/corpus/*` + `run_full_corpus_sweep`, not the sibling `performance_regression_jobs`
harness — and a *different* TSV schema, so they are neither merged into `$OUT` nor read by
`plot_codegen_perf.py`. Keep or retire them independently of the job:

| file | what |
|------|------|
| `run_readable_compare.py` | self-contained legacy-vs-experimental perf+quality driver over the in-tree corpora (+ optional CloudSC). Emits its OWN schema (`kernel corpus mode codegen threads codegen_ms compile_ms tidy_ms runtime_ms loc code_bytes tidy correct error`) and prints a readability-vs-cost summary; breaks out the clang-tidy cost (`tidy_ms`) and reads LoC straight from the generated frame. |
| `run_cavity_compare.py` | `cavity_flow`-only: legacy vs experimental, compiled with **both g++ and clang++**, single-core (S preset) — reports codegen / compile / runtime + speedup + LoC. Uses this folder's `run_readable_perf` helpers + the sibling harness. |

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
sbatch --export=ALL,CPP_STANDARD=17 submit_daint_readable.sbatch      # constexpr instead of consteval
```

Extra knobs: `CPP_STANDARD` (default `20`) and `METRICS_CSV` (default `codegen_metrics.csv`).

### C++ standard

`CPP_STANDARD` is pinned from one place onto BOTH builds a lane measures: DaCe's CMake build
(`compiler.cpp_standard`, the runtime/correctness binaries) and `engine.compile_sdfg_timed`'s
direct compile (`DACE_PERF_CXX_STD`, the `codegen_ms`/`compile_ms` numbers). Left unset these two
default to *different* standards (c++20 vs c++23), which would compare compile time and runtime at
mismatched standards. The readable generator emits `consteval` size/index helpers under **C++20+**
and degrades to `constexpr` under **C++17**, so `CPP_STANDARD=17` exercises the constexpr path.

### Plots

After the job, `submit_daint_readable.sbatch` harvests the generated CPU frames (from the
node-local `/dev/shm/dace_perf_jobs_<uid>_rank*` build roots) into `$METRICS_CSV` and prints the
plot command. Produce the two figures with:

```bash
python3 plot_codegen_perf.py --tsv perfresults.tsv --metrics-csv codegen_metrics.csv --out-dir plots
```

- **FIGURE A** `plots/runtime_<phase>.{png,pdf}` — total-runtime comparison, legacy vs experimental,
  one figure per phase with a panel per host compiler: `runtime_multi_core` is g++ vs clang++ CPU
  (before/after), `runtime_gpu` is the GPU lane, `runtime_single_core` is the quick S signal.
- **FIGURE B** `plots/build_and_quality.{png,pdf}` — stacked codegen+compile time, generated LoC,
  and the readability panel (`nloc` + `max_nesting` headline, `tokens_per_stmt` normalized,
  `max_ccn` control). The LoC/readability panels need `--metrics-csv` (or `--srcdir <sources>`);
  without either they are skipped and only the build-time panel is drawn.

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
