# Static perf analysis: parallel regions + vectorization capability

Compile-time COUNTS only, no timing. One entry point: `run_analysis.py` -> `report.md`.

## Kernel groups

* Group A: `poly` + `np` (polybench, npbench) -- 54 kernels.
* Group B: `tsvc` + `tsvc25` -- 223 kernels.

Kernels come from `tests/corpus/corpus_suite.py` (`CS.kernels()`, `CS.build`). No new drivers.
No dataset is built and no reference is computed: only the SDFG is needed.

## Pipelines (reused from `tests/passes/canonicalize/canonicalize_perf_corpus_test.py`)

`untransformed`, `serialize`, `canonicalize(+finalize_for_target)`, `auto_optimize` are IMPORTED
from that test module. Its pytest entry points are never invoked (they own `perf_results/`).

| variant | SDFG pipeline | codegen (`compiler.cpu.implementation`) |
|---|---|---|
| `serialize` | simplify, expand lib nodes + all maps Sequential to a fixed point | experimental_readable |
| `simplify_new` | simplify only | experimental_readable |
| `simplify_old` | simplify only | legacy (OLD) |
| `canon` | canonicalize + finalize_for_target | experimental_readable (NEW) |
| `autoopt` | auto_optimize(CPU) | legacy |

Library nodes lower to `pure` in EVERY variant wherever a pure expansion exists (the timing harness
uses OpenBLAS for the DaCe arms). Deliberate: a vendor call is not a loop, so an OpenBLAS lowering
would delete the very loops both metrics count and the configs would stop being comparable. A node
with no pure expansion (Cholesky) keeps its vendor lowering in every variant alike.

Code is emitted with `sdfg.generate_code()` + `generate_program_folder()`; each variant is then
compiled standalone with `-c` (no linking, no DaCe build system).

## Base flags (identical for every arm; from the timing harness)

`-std=c++20 -fPIC -Wall -Wextra -O3 -march=native -fno-math-errno -fno-trapping-math
-fno-signed-zeros -ffp-contract=fast -fopenmp` plus `-I dace/runtime/include -I <folder>/include`.

## Metric 1 -- parallel loops/regions per arm

| arm | input | flags added | counted by |
|---|---|---|---|
| (a) gcc-autopar | `serialize` | `-ftree-parallelize-loops=8 -floop-parallelize-all -fgraphite-identity` | `-fopt-info-loop-optimized` lines matching `parallelizing (outer\|inner) loop` |
| (b) llvm-autopar | `serialize` | `-mllvm -polly -mllvm -polly-parallel -mllvm -polly-omp-backend=LLVM -mllvm -polly-process-unprofitable -mllvm -polly-parallel-force` | `__kmpc_fork_call` relocation sites (Polly emits NO remark for parallel codegen; verified) |
| (c) dace-simplify | `simplify_new` | none | `#pragma omp parallel` occurrences in the generated `.cpp` |
| (d) dace-canon | `canon` | none | same |
| (e) dace-autoopt | `autoopt` | none | same |

Arms (a) and (b) must be handed genuinely SEQUENTIAL code. `serialize` alone does not achieve that:
codegen expands COPY library nodes internally, after any pipeline can reach them, and their
default-scheduled maps are emitted as `#pragma omp parallel for` (2 of them on polybench
covariance, which inflated its Polly count from 5 to 8). So the serialize source additionally has
every `#pragma omp` line deleted before compiling -- the same loops, sequentially -- and the report
proves the input was clean by reporting the residual pragma count, which must be 0.

Counting rule `if (cond) parallel else sequential` == ONE parallel holds in all three counters:
only the parallel branch carries a pragma; gcc reports one `parallelizing` line per parallelized
loop even when parloops versions it; one versioned construct is one runtime fork site.
Cross-check recorded for (a): `GOMP_parallel` relocation sites (agreed with opt-info on the probe).

## Metric 2 -- vectorization capability, cost model neutralized

Configs: (i) `canon` (NEW codegen), (ii) `simplify_old` (OLD codegen). Each built with gcc AND
clang.

* gcc: `-fvect-cost-model=unlimited` plus ONE `-fopt-info-vec-all=<file>` request. One, not two:
  gcc warns `ignoring possibly conflicting option` and silently drops the second when
  `-fopt-info-vec-optimized=` and `-fopt-info-vec-missed=` name different files, so `-all` is the
  only way to get both the vectorized and the missed halves out of a single compile.
* clang: no `-fvect-cost-model` equivalent exists; `#pragma clang loop vectorize(enable)` is
  per-loop, so the global stand-in used is `-mllvm -force-vector-width=4`, which sets `UserVF` and
  bypasses the cost model's VF/profitability decision. Reported by `-Rpass=loop-vectorize`. Every
  clang config is ALSO built with the stock cost model, and the report states how many loops the
  forcing gained and how many (if any) it lost -- forcing a width can in principle refuse a loop
  clang would otherwise take, so the report must show that it did not.

Normalization (SAME for both compilers): count DISTINCT `file:line:col` source locations reported
vectorized, keeping only locations inside the generated `.cpp`. Required, not cosmetic: gcc emits
several lines per loop (main body plus epilogue, 64-byte and 32-byte), so raw line counts would
inflate gcc by ~2x. Filter drops `dace/runtime/include/**` and system headers, i.e. everything
that is not USER kernel code.

## Execution

One worker process per kernel under `systemd-run --user --scope -p MemoryMax=5G`, serialized, one
JSON per kernel (resumable). Per-kernel timeout. Every python invocation carries
`PYTHONHASHSEED=0 MPI4PY_RC_INITIALIZE=0 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_VFS_ENABLE=n
DACE_cache=unique DACE_default_build_folder=$HOME/.cache/dace-build-perfanalysis`.
A kernel that fails a pipeline or a compile is EXCLUDED from that arm's column and listed in the
report footnote; it is never silently dropped.
