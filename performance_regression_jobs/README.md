# performance_regression_jobs

Times three DaCe pipelines per kernel and reports their speedups. The three
canonicalize corpora are NPBench+PolyBench, TSVC2 and TSVC2.5.

## Pipelines

| name        | what it is                                   |
|-------------|----------------------------------------------|
| `auto_opt`  | DaCe `auto_optimize` (auto-parallelization) — **baseline** |
| `parallel`  | light `simplify + LoopToMap + MapFusion` (dace-par) |
| `canon`     | `canonicalize`                               |

Speedup = `auto_opt_min / pipeline_min` (best-of-N). `>1` = faster than auto-par.

## Run with clang++

On the cluster (each script already pins `--cxx=clang++`; edit its `#SBATCH`
header for your account/partition/nodes):

```bash
./submit_main.sh                     # EVERYTHING: runtime + compile-speed (all corpora), GPU, canon-vectorize
./submit_all.sh                      # runtime ONLY (no compile timing)
# or individually:
sbatch slurm_npbench_polybench.sh    # CPU  (GPU is a separate job: slurm_npbench_polybench_gpu.sh)
sbatch slurm_tsvc2.sh
sbatch slurm_tsvc2_5.sh
```

`submit_main.sh` submits the three `*_compile.sh` jobs (each runs the runtime sweep
AND the compile-speed sweep), plus the GPU and canon-vectorize sweeps — one command,
no double node allocation. For a cross-compiler run: `CXXES="g++ clang++" ./submit_main.sh`.

Without a scheduler — one rank (timing uses 25 reps; buffers are reused across reps and
reset in place, never reallocated):

```bash
python3 npbench_polybench_perf.py --reps 25 --cxx=clang++ --devices cpu
python3 tsvc2_perf.py             --reps 25 --cxx=clang++
python3 tsvc2_5_perf.py           --reps 25 --cxx=clang++
```

N local ranks (kernels self-partition by rank): `./run_distributed.sh tsvc2_perf.py 8 --cxx=clang++`.

## Where the data goes

```
results/<corpus>/<kernel>/<compiler>_<host>_<preset>/
    results.csv     one row per timed rep (ms)
    status.csv      correctness per pipeline
    run_meta.json   rank / host / sizes
```

`<corpus>` ∈ `npbench_polybench` | `tsvc2` | `tsvc2_5`; `<preset>` is
`paper-cpu`/`paper-gpu` (npbench) or `default` (tsvc). Aggregated per corpus
(written at the end of each run; rebuild any time without re-measuring with
`python3 <corpus>_perf.py --tables-only`):

```
results/<corpus>/speedup.md       speedup table
results/<corpus>/correctness.md   per-pipeline correctness
results/<corpus>/summary.csv      one row per kernel x pipeline, speedup_vs_baseline column
```

## Getting canon vs. dace-par vs. auto-par speedups

- **NPBench+PolyBench**: baseline is `auto_opt`, so `speedup.md` / `summary.csv`
  give `parallel`, `canon` speedups over auto-par directly.
- **TSVC2 / TSVC2.5**: `speedup.md` is vs. the native-C baseline; for the
  DaCe-vs-DaCe numbers read `min_ms` per lane in `summary.csv`
  (`auto_opt-par`, `parallel-par`, `canon-par`).

## Plot

```bash
python3 plot_npbench_speedup_grid.py --results-dir results --out npbench_speedup_grid.png   # canon/parallel vs auto_opt, cpu+gpu
python3 plot_tsvc_boxplot.py         --results-dir results --out tsvc_boxplot.png           # DaCe vs native single/multi-core
```
