# Corpus sweep state — GCC arms

Jobs 4381745 (`results_loops`, 220 kernels) and 4381746 (`results_array`, 45 kernels).
72 threads, g++ 16.1.0, LEN_1D=589824, LEN_2D=768, polybench size index 3.

## Where we are

| group | denominator | autoopt-gcc | canon-gcc |
|:--|:--|--:|--:|
| tsvc + tsvc_2_5 | sequential C++ | 3.39x | **7.24x** |
| npbench + polybench | threaded numpy | 6.58x | **12.18x** |

Canonicalize is ~2x auto_optimize, and the gain is coverage, not tuning: kernels left
effectively serial (<1.2x) drop from 53% to 24%, kernels above 8x rise from 43% to 61%.
Best external auto-parallelizer is forced gcc autopar at 3.18x; Polly at its own cost
model is inert (1.15x, 83% serial).

## What to improve

Ranked by payoff. A guard that merely never loses to sequential is worth
**+24%** geomean on loops (7.24 -> 8.96x) and **+16%** on array (12.18 -> 14.18x).

1. **Lost parallelism under canon** — four distinct mechanisms, all confirmed in source:
   - `np:stockham_fft` 88 -> 3869 ms (44x). Dominant cause is codegen, `experimental_readable`
     ONLY: the `Dv[:,:,k]` strided copy falls through to a mapped-tasklet expansion whose
     schedule is set unconditionally to `Default` -> `CPU_Multicore`
     (`libraries/standard/nodes/copy_node.py:318` — it tests GPU and thread-local scope, never
     CPU enclosing scope). It sits in a 349,525-iteration loop, so 349,525 fork/joins x 11.1 us
     = essentially the whole runtime. autoopt runs ~50 regions total. Secondary: gemm and
     TensorTranspose pinned `Sequential` by `libnode_is_sequential`
     (`transformation/auto/auto_optimize.py:464`), where ANY enclosing `LoopRegion` returns
     True — conflating "inside a parallel map" (real hazard) with "inside a sequential loop"
     (10 iterations around 4.2M complex MACs).
   - `tsvc:va` 0.011 -> 0.252 ms (22x). `a[i] = b[i]` is lifted to a `CopyLibraryNode`, then
     lowered to a single-threaded `memcpy` because `is_parallel_cpu_transfer_size`
     (`libraries/standard/helper.py:83`) returns False for ANY symbolic count. LEN_1D is a
     symbol, so a 9.4 MB copy is treated like a 4-element one. Measured 38 GB/s = one core.
   - `t25:ext_floordiv_offset_mod`. `LoopToScan` computes a NEGATIVE `scan_stride` for the
     read-ahead form `a[i] = a[i+d] + b[i]`, emits the guard `-floor(LEN_1D,M) >= 1` which is
     provably false, and the entire parallel branch is dead code — every run takes the
     sequential fallback. Fix: compare against `|scan_stride|` in
     `loop_to_scan.py:2426`, and refuse to emit a guard provably unsatisfiable.
   - `t25:reduce_inner_carry` (4.9x). A thread-private stack scalar gets `reduce_atomic` per
     element (590k atomics) because `_collect_omp_reductions` (`codegen/targets/cpu.py:2910`)
     only recognises WCR targets OUTSIDE the map scope. Blocks vectorization too.
   NOT a regression: `t25:ext_scatter_store` (2.8x) — canon pays for the
   `ScatterToGuardedMaps` runtime permutation proof; autoopt emits an UNGUARDED parallel
   scatter that is correct only because this dataset's `idx` happens to be a permutation.
   Report it as soundness cost, or add a `scatter_to_guarded_maps=False` variant.
2. **Parallel region inside the outer loop of 2-D nests** — exactly six kernels
   (`s115 s119 s125 s126 s2233 s233`) sit at a flat 7.7-8.0 ms = 768 x 10.4 us,
   i.e. one 72-thread fork/join per outer row. `s115` is 116x slower than sequential
   and is ~100% team startup (empty-region floor 7.98 ms vs 8.15 ms for its real body).
   CONFIRMED root cause: `sequentialize_nested_parallel_scopes` in
   `dace/transformation/passes/canonicalize/finalize.py:336` applies the `in_loop`
   guard to library nodes but not to maps — its docstring states the policy outright.
   `LoopToMap` also lifts only the innermost loop and never revisits the outer one, so
   `s125`/`s126` lose a fully independent outer dimension permanently.
   Fixes: (a) lift/collapse the outer loop where the nest is perfect — measured
   `s125` 8.03 -> 0.019 ms, `s126` 8.30 -> 0.027 ms; (b) extend the `in_loop` guard to
   maps, the only fix available for `s115`/`s119`, break-even ~50k iterations.
   Hoisting to a persistent team is only 2.2x and needs new machinery — skip it.
   Schedule/chunk tuning is measurably NEGATIVE (`dynamic,8` 10.6 ms, `guided` 24.4 ms).
   `s119` has the same defect under autoopt, so canonicalize alone will not move it.
3. **Anti-dependence snapshot is a single-threaded memcpy** — seven WAR recurrences
   (`s112 s121 s151 s162 s421 ext_war_unit ext_war_sym`) cluster at 0.61 ms vs
   0.167 ms sequential. NOT a missing cost model: `BreakAntiDependence` stages the
   window as a state-level array-to-array copy (`break_anti_dependence.py:1034`) that
   codegen lowers to a bare `memcpy`. That copy is 87% of the runtime and costs more
   than the whole sequential kernel (3N words at ~74 GB/s vs 2N words to a cold
   destination at ~33 GB/s, so T_copy ~= 1.5 x T_seq) — a loss at EVERY trip count,
   no crossover. Fork/join is ~2%; allocation is hoisted to init, so it is not that.
   `ChunkAntiDependence` already implements the fix and is wired in at
   `canonicalize/pipeline.py:997`, but NEVER MATCHES: it requires the copy and the
   MapEntry in the same state (`chunk_anti_dependence.py:110,128`) while
   `BreakAntiDependence` uses `add_state_before`, and state fusion runs later than the
   `cpu_specialize` band. Applied by hand it works and is numerically correct:
   `s151` 0.328 -> 0.0216 ms, from 1.9x slower than sequential to 7.8x faster.
   Broader fix, covering what chunking cannot reach (`s112` reverse, `s162` symbolic
   offset, `s175` strided, both staging copies in `s3112`): lower large state-level
   array-to-array copies as parallel copies. Only add a cost gate if the copy stays
   sequential — break-even is ~5 flops/element and this family is at 1.
4. **Array corpus: parallelism placed at the wrong nest level.**
   - `np:scattering_self_energies` autoopt 60,832 ms = 1540x its own sequential build.
     An 8-deep fully sequential nest with the ONLY `parallel for` at the bottom, trip
     count **4** (`Norb`), entered 622,080 times. canon instead parallelizes the outer
     `Nkz` loop and is 468x faster. Both arms then lose to sequential anyway because a
     4x4 `zgemm` is a library call at all (622k `cblas_zgemm`), and canon heap-allocates
     three temporaries per innermost body = 933,120 alloc/free pairs.
   - `poly:seidel_2d` 0.10x — 78,804 regions x ~11 us = 869 ms, the ENTIRE gap
     (960.4 - 91.5). The Gauss-Seidel `j` scan is correctly serialized; the only parallel
     construct is a 398-element row sum, i.e. 5 doubles per thread. Correct fix: decline.
   - `poly:gramschmidt` autoopt 205.7 ms — parallelized the innermost length-240 map,
     20,100 regions x 9.96 us = 200.2 ms, exactly the gap. **canon already fixes this**
     by hoisting to the outer `j` loop (4.82 ms, below the 5.45 ms sequential build).
   - `poly:covariance` canon 39.2 -> 63.6 ms — a DEAD copy: `cov` row is copied to a
     heap temp and straight back, 2,400 extra regions x 10.15 us = 24.36 ms, and the
     measured regression is 24.36 ms exactly.
   - `np:hdiff` canon 11.2 -> 13.1 ms — a lost stencil fusion, one extra materialized
     42.0 M-element pass at 509 GB/s (Grace bandwidth). Small, bandwidth-bound.
   - `poly:nussinov`, `np:crc16` — ZERO regions in all arms. Correct: triangular DP and
     a bit-serial chain. canon is actually 3.2x AHEAD on nussinov by privatizing the
     inner `max` WCR into a register.
5. **`poly:trisolv` is not a dependence story — it is OpenBLAS threading.** Neither DaCe
   arm emits any OpenMP region (correct: forward substitution). Both lower `L[i,:i]@x[:i]`
   to `cblas_ddot`, called 16,000 times at average length 8,000, and
   `submit_corpus_perf.sh:95` exports `OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS`, so each
   short dot inside a serial chain is dispatched over 72 threads. Measured: 95.6 ms at 72
   OpenBLAS threads vs **26.2 ms at 1**. At one thread trisolv would be 3.1x FASTER than
   its sequential build instead of 20% slower. Best ratio-per-effort fix in the set.
6. **One miscompile** — `tsvc25:scan_multi_5carry` fails its value check on both
   canon arms. Excluded from the geomeans, not counted as 1.0.

Baseline overhead on this machine: one 72-thread parallel region costs ~10 us,
which is the floor every trivial kernel bottoms out at.

**The unifying gap.** Nothing on the canon path asks whether a parallel region is worth
its fork/join given trip count and enclosing loop depth. That single missing cost model
produces class 2 (768 regions), stockham_fft's 349,525 regions, and `tsvc:s4116` (15x —
canon correctly lifts a reduction autoopt declined, then pays 13.7 us of fork/join for
767 iterations of work). A trip-count gate in `_generate_MapEntry` — constant counts
below a threshold emit `Sequential`, symbolic ones emit `if(count > N)` — would have
prevented all three.

## Harness caveats found during the triage

- **`channel_flow`'s reported time is ONE while-iteration, not the kernel.**
  `_time_all_reps` builds the call args once and never resets them (deliberate, and right
  for straight-line kernels), but `channel_flow` converges on `while udiff > 0.001` over
  arrays it mutates, so the untimed sizing call runs the full 989 iterations and every
  timed rep starts already converged and exits after one. Arm-to-arm ratios stay sound;
  absolute figures are ~1/989 of the real workload. It is the ONLY corpus kernel with a
  data-dependent convergence loop over mutated inputs (checked all `while` sites).
- **`seq-cpp` keeps `blas='pure'` while the DaCe arms use OpenBLAS** — deliberate and
  documented (`Arm.blas`, line ~303: a threads=openmp OpenBLAS would run on
  OMP_NUM_THREADS regardless of schedule and destroy the meaning of "one thread"). The
  published figure is unaffected — poly/np divide by numpy, not by seq-cpp — but any
  "slower than its own sequential build" claim on a BLAS-carrying kernel (trisolv,
  gramschmidt, covariance, scattering) crosses a BLAS boundary and must say so.
- Preset actually measured is `paper_sizes`, not polybench index 3: seidel_2d ran at
  tsteps=100 N=400, trisolv N=16000, covariance M=1200 N=1400, gramschmidt M=240 N=200.
- **Arms are averaged over different kernel sets** (canon n=219, everyone else n=220):
  `scan_multi_5carry` is dropped only from the arms that miscompile it, while scoring
  ~1.00x for every competitor. Canon is credited by omission for a kernel it gets wrong.
  Quantified on the common 219-kernel set: autoopt-gcc 3.384 -> 3.403x, canon-gcc
  unchanged at 7.243x, so the ratio moves 2.14x -> 2.13x. Immaterial, but the geomeans
  should intersect the kernel sets across arms before publication.
- Unresolved: `scattering` and `channel_flow` imply ~90-98 us per region against the
  ~10 us measured four other ways in the same job. Region counts are exact; the 8x
  inflation reproduces on neither the login node nor any microbenchmark. Both are
  kernels where 4 MPI ranks x 72 threads run concurrently with non-trivial sequential
  code between regions — cross-rank affinity or libgomp wait policy is where to look.
  Changes no conclusion above.

## Not covered here

LLVM arms are excluded on purpose — canon-llvm is 9.90x on loops but only 1.06x
against numpy on the array corpus, a separate problem to be fixed later.
Array coverage holes: Cholesky/Solve have no `pure` expansion, lenet/resnet time out.
