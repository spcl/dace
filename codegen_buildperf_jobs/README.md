# codegen_buildperf_jobs

A submittable daint.alps job comparing the **legacy** vs **experimental (readable)** C++ code
generator (`compiler.cpu.implementation = legacy | experimental`) on **build cost** and
**runtime**, over the NPBench + PolyBench + TSVC2 + TSVC2.5 corpora, with **both** host
compilers (`g++` and `clang++`). **CPU only** — there is no GPU phase in this job.

The only thing that differs between the two lanes is the code generator; the pipeline is the
project-standard **`dace + simplify + LoopToMap + MapFusion`** (`+ len-1 → scalar`
scalarization), and measurement always happens **after** it. Readability is expected to be
perf-neutral, so the runtime comparison is a regression guard, not a claimed win — the win is
in generated size / readability.

This folder is self-contained: it vendors the TSVC corpus harness and `readability_metrics.py`,
and reuses (imports, never modifies) the sibling `../performance_regression_jobs` harness
(`engine.py` + `npbench_polybench_perf.py` + the npbench/polybench corpora) for kernel
discovery, dataset presets, isolated crash-safe timing, cold-compile timing, and the numpy
oracle. Keep that folder next to this one, with its submodules initialized:

```bash
git submodule update --init dace/external/moodycamel dace/external/cub
```

## Layout

| file | what |
|------|------|
| `submit_codegen_buildperf.sbatch` | **THE job** — every phase, one node, one submission, one merged TSV. |
| `run_buildperf_all.py` | **the driver the job launches** — builds ONE combined kernel list across all four corpora, slices it round-robin across ranks (`combined[rank::num_ranks]`), and dispatches each kernel to the matching per-corpus driver's `process_kernel`. One pass, one per-rank TSV. |
| `run_buildperf.py` | npbench+polybench driver: per (kernel × codegen) → codegen time, cold compile time, best-of-N runtime, **inline** generated-code size + readability metrics, correctness. CPU only. Reused by `run_buildperf_all.py`; also runnable standalone. |
| `run_buildperf_tsvc2.py` | the same, for the TSVC2 corpus. |
| `run_buildperf_tsvc2_5.py` | the same, for the TSVC2.5 corpus. |
| `readability_metrics.py` | lizard/local readability metrics (`nloc`, `max_nesting`, `tokens`, `max_ccn`) + `frame_metrics()` the drivers call inline on each generated `.cpp`. |
| `plot_buildperf.py` | the two figures, straight from the merged TSV (build/size/readability are inline, so no separate metrics file). |
| `tsvc2_perf.py`, `tsvc2_5_perf.py`, `tsvc_corpus.py`, `tsvc_2_5_corpus.py` | vendored TSVC2 / TSVC2.5 corpus harness (kernel discovery, sizing, deterministic input allocation). |

## What it measures

For **{g++, clang++} × {legacy, experimental} × {npbench, polybench, tsvc2, tsvc2_5}**, per
kernel:

1. **Build metrics** (recorded inline in every row):
   - `codegen_ms` — SDFG → C++ generation time (cold).
   - `compile_ms` — C++ → binary time (cold, a direct compiler invocation timed on its own).
   - `code_bytes` — generated `.cpp` size on disk; `nloc` — lizard non-comment/non-blank LoC.
   - readability: `nloc` and `max_nesting` are the **discriminators**, `tokens` is
     size-sensitive (reported with `nloc` as its normalizer), `max_ccn` is a **control**
     (both lanes emit the same loop structure, so it should be unchanged — proof readability
     did not come from dropping semantics).
2. **Single-core** runtime, preset **S**, `OMP_NUM_THREADS=1`, all 4 corpora.
3. **Multi-core** runtime, preset **paper** (full node, 72 cores), all 4 corpora.

The generated C++ frame is host-compiler-independent (g++ and clang++ build the *same*
source), so `codegen_ms`, `code_bytes`, `nloc`, `max_nesting`, `tokens`, `max_ccn` agree
across the two `cxx` rows of a (kernel, codegen); only `compile_ms` and the runtimes depend on
the compiler. The C++ *standard* does affect the generated code (`consteval` under C++20,
`constexpr` under C++17), so it is pinned once for the whole job (see below).

## Run layout — 4 ranks, 2 ordered phases

1 node, 4 ranks, phases that never overlap (each `srun` sets its own `--ntasks`):

| # | phase | preset | layout | host cxx |
|---|-------|--------|--------|----------|
| 1 | `single_core` | S | 4 ranks (1/Grace socket), `OMP_NUM_THREADS=1`, combined list split round-robin `combined[rank::4]` | `g++` |
| 2 | `single_core` | S | ″ | `clang++` |
| 3 | `multi_core` | paper | 1 rank, all 72 cores of one socket (`OMP_NUM_THREADS=72`) | `g++` |
| 4 | `multi_core` | paper | ″ | `clang++` |

Each sweep is **one** `srun python3 run_buildperf_all.py`: it builds a **single combined kernel
list across all four corpora** (npbench + polybench + tsvc2 + tsvc2_5) and each rank measures
its own disjoint round-robin slice — the corpora are distributed across the ranks *together*,
not as a separate per-corpus pass. Build metrics are gathered **as part of the run** (scored on
the frame each correctness build persists), so there is no second build or `/dev/shm` harvest
pass. Every phase’s every rank is merged into **one** TSV.

## Unified TSV schema

```
kernel  corpus  codegen  preset  threads  cxx  phase  codegen_ms  compile_ms  runtime_ms \
  code_bytes  nloc  max_nesting  tokens  max_ccn  speedup  correctness  error
```

- `corpus` ∈ `npbench|polybench|tsvc2|tsvc2_5`; `codegen` ∈ `legacy|experimental`;
  `cxx` ∈ `g++|clang++`; `phase` ∈ `single_core|multi_core`.
- `codegen_ms` / `compile_ms` are **separate** (a stacked plot can show each); blank on error.
- `code_bytes` = generated-source bytes; `nloc` = generated LoC; `max_nesting` / `tokens` /
  `max_ccn` = readability metrics (`tokens`/`max_ccn` are blank if `lizard` is not installed;
  `nloc` then falls back to a local line count, `max_nesting` and `code_bytes` are always
  available).
- `speedup` = `legacy_runtime / this_runtime` (legacy row = 1.000; experimental row = the
  new-vs-legacy ratio).
- `correctness` ∈ `pass|fail|unknown|ERROR` (experimental vs legacy output; legacy vs numpy
  oracle where one exists). `error` carries a failure message on an ERROR row.

## Submit

```bash
sbatch submit_codegen_buildperf.sbatch

# knobs (all optional):
sbatch --export=ALL,REPS=3,CODEGEN=experimental,OUT=quick.tsv submit_codegen_buildperf.sbatch
sbatch --export=ALL,CPU_CXX_LIST="g++" submit_codegen_buildperf.sbatch   # skip the clang++ sweeps
sbatch --export=ALL,CPP_STANDARD=17 submit_codegen_buildperf.sbatch      # constexpr instead of consteval
```

Knobs: `CODEGEN` (default `both`), `REPS` (default `10`), `OUT` (default `perfresults.tsv`),
`CPU_CXX_LIST` (default `"g++ clang++"`), `CPP_STANDARD` (default `20`).

### C++ standard consistency

`CPP_STANDARD` is pinned from **one** place onto BOTH builds a lane measures:
`DACE_compiler_cpp_standard` drives DaCe’s CMake build (the runtime/correctness binaries) **and**
the readable generator’s `consteval`/`constexpr` choice (`experimental_cpu.size_qualifier`
reads `compiler.cpp_standard`: `consteval` under C++20+, `constexpr` under C++17), while
`DACE_PERF_CXX_STD` gives `engine.compile_sdfg_timed`’s direct compile the matching `-std`
(the `codegen_ms`/`compile_ms` numbers). Left unset these two default to *different* standards
(c++20 vs c++23), which would compare compile time and runtime at mismatched standards. Default
**20** keeps `consteval` on. clang-tidy also reads `compiler.cpp_standard`, so it stays
consistent automatically — nothing here hardcodes a conflicting `-std`.

### cmake SIGCHLD hang (do not reintroduce)

`srun` starts tasks with `SIGCHLD` blocked; cmake’s configure reaps helpers via `select()` and
spins forever if `SIGCHLD` stays blocked. DaCe unblocks it only around the build in
`dace/codegen/compiler.py::_build_subprocess_sigmask()` (defined ~L375, used ~L907). So every
build runs under a **plain** `srun … python3 …` — **no** `preexec_fn`, no `SIGCHLD`
trap/handler, no `setsid`/`stdbuf` shim, no nested `srun`. See the header comment in the sbatch.

## Plot

```bash
python3 plot_buildperf.py --tsv perfresults.tsv --out-dir plots
```

- **`plots/build_and_quality.{png,pdf}`** — four panels, legacy vs experimental, per corpus, at
  one phase (`--phase`, default `multi_core`): (i) DaCe codegen time, (ii) C++ compile time
  (g++ vs clang++), (iii) generated LoC, (iv) readability discriminators (per-corpus geomean of
  the experimental/legacy `nloc` and `max_nesting` ratios; <1 = experimental smaller/flatter).
- **`plots/performance.{png,pdf}`** — two panels (single-core S, multi-core paper) of the
  experimental-vs-legacy runtime speedup (geomean `legacy/experimental` per corpus), g++ vs
  clang++, with a parity line at 1.0. Absolute runtimes span orders of magnitude across
  kernels, so the per-corpus ratio is the readable axis.

## Presets

- **S** — dataset size `S`, **single core** (`OMP_NUM_THREADS=1`): a quick regression signal.
- **paper** — paper dataset sizes, **multi-core**: the writeup numbers.

## Placeholders the owner must fill in

Every `#SBATCH --account` / `--partition` / `--chdir` and the `REPO_ROOT` / `PYTHONUSERBASE` /
venv / `spack load` lines in `submit_codegen_buildperf.sbatch` are marked **TODO** — set them
to your daint.alps account, a CPU-only partition, and the synced repo path before submitting.
The job assumes `g++` (spack gcc) and `clang++` (spack llvm) are both on `PATH`, that
`lizard` is `pip`-installed in the venv for the token/complexity metrics (it degrades
gracefully without it), and that OpenBLAS is available for the BLAS/LAPACK kernels.

## Run locally (no SLURM — at most 4 threads on a shared machine)

```bash
cd codegen_buildperf_jobs
PYTHONPATH=<repo> OMP_NUM_THREADS=1 python3 run_buildperf.py --preset S --only arc_distance --out probe.tsv
PYTHONPATH=<repo> python3 run_buildperf_tsvc2.py --preset S --only s000 --out t2.tsv
python3 plot_buildperf.py --tsv probe.tsv --out-dir plots
```
