# Static perf analysis: parallel regions and vectorization capability

> PARTIAL RUN: sweep interrupted at 206/277 kernels; regenerate with run_analysis.py for full tables.

Compile-time counts, no timing. Method, flags and counting rules: `plan.md`. Regenerate: `python dace/perf_analysis/run_analysis.py`.

## Group A -- npbench + polybench (2 kernels)

### Metric 1 -- parallel loops/regions

| arm | kernels | parallel loops/regions | kernels with >=1 | mean |
|---|---|---|---|---|
| (a) gcc autopar | 2 | 2 | 1 | 1.00 |
| (b) llvm autopar (Polly) | 2 | 16 | 2 | 8.00 |
| (c) dace-simplify | 2 | 7 | 2 | 3.50 |
| (d) dace-canon | 2 | 9 | 2 | 4.50 |
| (e) dace-autoopt | 2 | 7 | 2 | 3.50 |

### Metric 2 -- vectorized loops, cost model neutralized

| config | compiler | kernels | vectorized loops | kernels with >=1 | mean | gcc missed |
|---|---|---|---|---|---|---|
| (i) canon + new codegen | gcc | 2 | 1 | 1 | 0.50 | 43 |
| (i) canon + new codegen | clang | 2 | 2 | 1 | 1.00 |  |
| (ii) simplify + old codegen | gcc | 2 | 1 | 1 | 0.50 | 51 |
| (ii) simplify + old codegen | clang | 2 | 0 | 0 | 0.00 |  |

## Takeaways

* Group A metric 1: (b) llvm autopar (Polly) leads with 16 parallel regions; external autopar reaches 2 (gcc) / 16 (llvm) against DaCe's 7 (simplify), 9 (canon), 7 (autoopt).
* Group A metric 2: canon+new codegen vectorizes 1 loops (gcc) / 2 (clang) versus simplify+old codegen 1 / 0.
* 0 kernels leave canonicalize with no parallel region at all -- the remaining parallelization gap, and the shortlist worth reading kernel by kernel.

## Notes

**Group A**: clang forced-VF vs stock cost model: +0 loops gained, -0 lost.
**Group A**: gcc opt-info vs GOMP_parallel call sites disagree on 0/2 kernels.
**Group A**: 0/2 kernels had residual DaCe pragmas in the autopar input (must be 0).
**Group A exclusions** (0):
* none
