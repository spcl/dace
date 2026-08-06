# Corpus perf job

One job, four in-repo corpora (polybench, npbench, tsvc, tsvc_2_5), `paper` preset.
`X` nodes x 4 ranks. Default 1 node = 4 ranks = one corpus per rank.

## Submit

```bash
sbatch tests/perf/submit_corpus_perf.sh          # 1 node,  4 ranks
sbatch tests/perf/submit_corpus_perf.sh 4        # 4 nodes, 16 ranks
```

Arg 1 = node count. Everything after it goes to the rank script:
`--preset` `--facet` `--limit` `--check` `--force` `--out`.
The script re-submits itself with `--nodes`, so `--nodes` is never an `#SBATCH` line.

No Slurm? Same script, `mpirun` fallback, then single-rank fallback:

```bash
bash tests/perf/submit_corpus_perf.sh 1 --preset S --limit 2       # smoke run
```

## Rank -> work

Each corpus is cut into `size // 4` shards (min 1). The `4 * shards` slots are laid out
suite-minor and rank `r` takes `slots[r::size]`. One rule, no special cases:

| ranks | rank 0 | rank 1 | rank 2 | rank 3 | rank 4 |
|:--|:--|:--|:--|:--|:--|
| 1  | all four, whole | - | - | - | - |
| 4  | poly | np | tsvc | tsvc25 | - |
| 8  | poly `0/2` | np `0/2` | tsvc `0/2` | tsvc25 `0/2` | poly `1/2` |

`i/n` is passed straight through as the drivers' own `--shard i/n` (per corpus, round-robin).

## What runs

Two facets per slot, both existing drivers, one kernel at a time (no xdist -- it would make the
timings meaningless):

* `tests.corpus.measure_parallelization` -- residual sequential loops, `--check` adds compile+run
  value-preservation.
* `tests.passes.canonicalize.canonicalize_perf_corpus_test` -- wall-clock canon vs auto-opt.

## Env

Pinned by the submit script, do not vary them between compared pipelines:

| var | why |
|:--|:--|
| `DACE_compiler_cpu_args` | `-ffp-contract=off`; bit-exact comparison dies under FP contraction |
| `OMP_NUM_THREADS` | cores / 4, so 4 ranks do not oversubscribe. `OMP_PROC_BIND=close`, `OMP_PLACES=cores` |
| `PYTHONHASHSEED=0` | DaCe determinism |
| `MPI4PY_RC_INITIALIZE=0` `OMPI_MCA_pml=ob1` `OMPI_MCA_btl=self,vader` `UCX_VFS_ENABLE=n` | every python invocation |
| `DACE_cache_distaware=1` | plus a per-rank build folder; cmake wedges under `srun` with SIGCHLD blocked |
| `PYTHON` | interpreter. A batch script does not inherit the interactive PATH; the job dies loudly if `import dace` fails |
| `OUT_DIR` | results (default `tests/perf/corpus_perf_results`) |
| `DACE_JOB_SCRATCH` | build scratch root, overrides the auto choice |

Scratch root is chosen at start and printed: `/dev/shm` (in memory) when it has 4 GiB free per
rank on the node, else `$HOME/.cache`. Each rank gets its own `rank<NNNN>/` subdir under it.
Never `/tmp` -- small shared tmpfs, filling it fakes corpus failures.

## Results

Under `$OUT_DIR`:

```
parallelism_rank<NNNN>.csv   per-rank, per-kernel loop/map counts
perf_json/<suite>_<kernel>.json   per-kernel timings, ALL repetitions
speedup.csv  speedup_table.md     the aggregate, written by the post-step
ranks/                            per-rank partial tables
```

The post-step runs automatically after the ranks exit (`--summarize` + a `--no-run` re-export).
Re-run it alone with:

```bash
python tests/perf/corpus_perf_job.py --out "$OUT_DIR" --aggregate
```

Exit code is non-zero if any kernel errored or miscompiled.

## Resume

The timing facet writes one JSON per kernel and skips kernels that already have one, so a killed
sweep resumes by re-submitting. Delete a kernel's JSON to re-measure just it, or pass `--force`
to re-measure everything. The parallelism facet is cheap and always re-runs (it appends, so
delete `parallelism_rank*.csv` before a clean re-run).

---
The `*_cloudsc_backend.*` files in this directory are a separate job.
