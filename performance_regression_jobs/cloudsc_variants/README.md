# cloudsc_variants

The CLOUDSC kernel (ECMWF cloud microphysics — a huge Fortran-derived SDFG) under the four
DaCe build variants, `{build_mode: cmake, native} × {compiler.cpu.implementation: legacy,
experimental_readable}`. Same axes and row schema as the sibling `codegen_variants` job,
but on the single stress-case SDFG the NPBench corpus never reaches (thousands of tasklets,
one enormous translation unit) — and with the expensive SDFG preparation **cached**.

## Two phases

| phase | what | cost | where |
|---|---|---|---|
| A (cache) | parse CloudSC (`simplify=False`) + the parallelize chain validated by `tests/corpus/cloudsc/cloudsc_parallelize_chain_test.py` (simplify, ShortLoopUnroll, PrivatizeScalars, PCIA, AugAssignToWCR, LoopToMap, LoopToScan), saved compressed to `cache/cloudsc_parallel.sdfgz` | minutes, ONCE | **login node** |
| B (measure) | load the `.sdfgz`, one sequential reference run, then per variant: `codegen_ms` + `compile_total_ms` (cold `sdfg.compile()` wall) + `run_ms` (median of `--run-reps`) + correctness vs the reference; each cell crash-isolated (`engine.run_isolated`) | fits a 30-min debug window | debug job |

**Pre-build the cache on the login node first** (transform work, allowed there):

```bash
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
python3 cloudsc_variants/cloudsc_variants_perf.py --build-cache-only
```

The sbatch script warns (but still proceeds) if the cache is missing — a cold Phase A can
exceed the 30-min debug window.

## Run

```bash
sbatch cloudsc_variants/slurm_cloudsc_variants.sh    # debug partition, 1 node x 1 rank x 72 cpus
```

Manually (Phase B only, cache present):

```bash
python3 cloudsc_variants/cloudsc_variants_perf.py                       # all 4 cells
python3 cloudsc_variants/cloudsc_variants_perf.py --variants cmake_legacy,native_legacy
python3 cloudsc_variants/cloudsc_variants_perf.py --rebuild-cache --build-cache-only
python3 cloudsc_variants/cloudsc_variants_perf.py --tables-only         # rebuild the tables
```

## Deviations from codegen_variants

* `run_ms` is the **median** of the reps (single kernel — report the center, not best-of-N).
* correctness is read off the same instrumented build that is timed (buffers hold the last
  rep's outputs) — saves one multi-minute compile per cell.
* compiled at `-O3 -march=native -fno-fast-math -ffp-contract=off` (fast-math stripped):
  cloudsc correctness can only be bounded under IEEE-respecting builds; this is the chain
  test's validated `o3` regime, tolerance 1e-9 vs the `make_sequential` reference.
* `--compile-reps` defaults to 1 (a cold cloudsc compile is minutes).

## Output

```
cloudsc_variants/cache/cloudsc_parallel.sdfgz   Phase A cache (gitignored; delete or
                                                --rebuild-cache after chain changes)
cloudsc_variants/cache/cloudsc_reference_outputs.npz
                                                sequential-reference outputs, rewritten by
                                                every Phase B run (file, not queue payload --
                                                run_isolated joins before draining the queue)
results/cloudsc/cloudsc_variants.csv            one row per (build_mode, implementation),
                                                same columns as codegen_variants.csv
results/cloudsc/cloudsc_variants.md             variant table + legacy-vs-experimental
                                                runtime + native-vs-cmake build speedup
```
