# Corpus perf job

ONE job: four in-repo corpora, six arms, TWO figures (the corpora split into two denominator
groups, see below). Three files, nothing else.

| file | what it does |
|:--|:--|
| `submit_corpus_perf.sh` | sbatch submission. Pins the environment, probes every arm, fans out `X` nodes x 4 ranks, aggregates |
| `corpus_perf_job.py` | the rank. Picks its kernels, takes a private build folder, unblocks SIGCHLD, drives the measurement |
| `plot_corpus_perf.py` | the two figures, from the aggregated results |

(`cloudsc/` next door is an unrelated job.)

## The six arms

One row per column of the figure. Every arm is built from the SAME base flag string
(`-ffp-contract=fast` included) and adds only its own flags, runs at the same `OMP_NUM_THREADS`, is
value-compared against the corpus reference, and is timed on the same rank as every other arm of
the same kernel.

| arm | SDFG it compiles | compiler | parallelized by | speedup baseline |
|:--|:--|:--|:--|:--|
| `dace-autoopt-gcc` | `auto_optimize` | `g++` | DaCe | poly/np: corpus reference. tsvc/tsvc25: `seq-cpp` |
| `dace-autoopt-llvm` | `auto_optimize` | `clang++` | DaCe | poly/np: corpus reference. tsvc/tsvc25: `seq-cpp` |
| `dace-canon-gcc` | `canonicalize` | `g++` | DaCe | poly/np: corpus reference. tsvc/tsvc25: `seq-cpp` |
| `dace-canon-llvm` | `canonicalize` | `clang++` | DaCe | poly/np: corpus reference. tsvc/tsvc25: `seq-cpp` |
| `dace-simplify+gcc-autopar` | post-`SimplifyPass`, emitted SEQUENTIAL | `g++` | GCC autopar + Graphite | poly/np: corpus reference. tsvc/tsvc25: `seq-cpp` |
| `dace-simplify+llvm-autopar` | post-`SimplifyPass`, emitted SEQUENTIAL | `clang++` | Polly autopar | poly/np: corpus reference. tsvc/tsvc25: `seq-cpp` |
| `seq-cpp` | post-`SimplifyPass`, emitted SEQUENTIAL | `g++` | nothing (`-O3`, no autopar, no polyhedral flags) | not a comparison arm: it IS the tsvc/tsvc25 denominator |

Graphite is folded into the gcc autopar arm and Polly into the llvm one. A polyhedral pass that
restructures a nest nobody then parallelizes answers no question the figure asks, so each external
arm is one column that has to prove BOTH of its passes ran.

The per-kernel JSON also carries `speedup_vs_baseline`, against `dace-autoopt-gcc`. That is the
pipeline-vs-pipeline number and the CI regression assertion; the figure uses the per-corpus
denominator below.

## Corpora and baselines

`tests/corpus/corpus_suite.py`, `paper` preset, in this fixed order:

| corpus | kernels | denominator | what the denominator is |
|:--|--:|:--|:--|
| `poly` | 30 | corpus reference | timed numpy reference (`polybench_numpy`). `seidel_2d`'s numpy form is a scalar Python loop: labelled `python-scalar`, never timed, never divided by |
| `np` | 24 | corpus reference | timed numpy reference |
| `tsvc` | 151 | `seq-cpp` | sequential C++ from the same post-simplify SDFG |
| `tsvc25` | 72 | `seq-cpp` | sequential C++ from the same post-simplify SDFG |
| | 277 | | |

The tsvc and tsvc_2_5 oracles are scalar Python loops. They are correctness oracles and MUST NEVER
be a timing denominator -- dividing by them yields speedups in the hundreds and silently invalidates
the figure. The two groups answer different questions, are labelled by `kind` in every record, and
are never pooled into one geomean.

## Speedup scale

With `r = t_baseline / t_arm`, the plotted number is signed and symmetric:

```
s = r - 1        if r >= 1      # 2x faster -> +1
s = -(1/r - 1)   if r <  1      # 2x slower -> -1, parity -> 0
```

Geomeans are computed on the RAW ratio `r` and only then converted, never on `s`.

## Rank -> work

Round-robin over the POOLED kernel list: `corpus_suite.kernels()` in its fixed order, rank `r` of
`n` takes `kernels[r::n]`, and that rank runs EVERY arm of every kernel it owns.

| ranks (nodes x 4) | kernels per rank |
|:--|:--|
| 1 (bare, no launcher) | 277 |
| 4 (1 node) | 70, 69, 69, 69 |
| 16 (4 nodes) | 18 x 5, 17 x 11 |

Two reasons, both load-bearing:

* **Load balance.** One corpus per rank leaves the tsvc rank with 151 kernels and the npbench rank
  with 24, so the job lasts as long as its slowest corpus.
* **Attribution.** All arms of a kernel must be timed on the SAME node. Arms split across nodes
  differ by thermal/DVFS state and memory locality as well as by pipeline, and the speedup stops
  being attributable to the pipeline.

The order is fixed, so a re-submission lines up with the per-kernel JSONs already on disk.

## Submit

```bash
sbatch canon_corpus_perf_job/submit_corpus_perf.sh          # 1 node,  4 ranks
sbatch canon_corpus_perf_job/submit_corpus_perf.sh 4        # 4 nodes, 16 ranks
```

Arg 1 = node count. Everything after it goes to the rank script: `--preset` `--facet` `--limit`
`--check` `--force` `--out`. The script re-submits itself with `--nodes`, so `--nodes` is never an
`#SBATCH` line. The `#SBATCH` defaults are CSCS values; set `ACCOUNT` / `PARTITION` / `TIMELIMIT`
for any other site.

No Slurm? The same script falls back to `mpirun`, then to a single rank, so it is also the local
smoke test:

```bash
bash canon_corpus_perf_job/submit_corpus_perf.sh 1 --preset S --limit 2       # smoke run
python canon_corpus_perf_job/corpus_perf_job.py --preset S --limit 2          # bare: rank 0 of 1
```

Before any rank starts, the submit script runs one ~2 s preflight on the node that will do the
work: it resolves the interpreter, proves `DACE_cache_distaware` reached DaCe, and probes every
arm. Each external pass must report itself on a known SCoP (`Adding SCoP`, `SCoP begins here`) and
emit a parallel region (`GOMP_parallel` as an undefined symbol of the probe object), and gcc's
compile-time `-ftree-parallelize-loops=N` must equal `OMP_NUM_THREADS`. A missing binary, a
rejected flag or a silent pass aborts the job there, with the reason, instead of publishing plain
`-O3` timings under an autopar label hours later. The evidence lines go to the job log and into
every result JSON under `arms`.

## Resume

The perf facet writes one JSON per kernel and skips kernels that already have a complete one, so a
killed sweep resumes by re-submitting. Delete a kernel's JSON to re-measure just it, or pass
`--force` to re-measure everything. Records written by the pre-six-arm labels are recognized as
stale and re-measured. The parallelism facet is cheap and always re-runs; it appends, so delete
`parallelism_rank*.csv` before a clean re-run.

## Env

Pinned by the submit script and exported, so the value in the log is the value every rank and every
cmake/compiler child inherits. Do not vary them between compared arms.

| var | why |
|:--|:--|
| `DACE_compiler_cpu_args` | the ONE base flag string. `-ffp-contract=` must be explicit (gcc defaults to `fast`, clang to `on`, so omitting it makes the compiler columns incomparable); `-ffast-math` is rejected, since associative math lets each compiler reassociate reductions differently |
| `OMP_NUM_THREADS` | cores / 4, so 4 ranks do not oversubscribe. Identical for every arm, and gcc's `-ftree-parallelize-loops` is derived from it. `OMP_PROC_BIND=close`, `OMP_PLACES=cores` |
| `CANON_PERF_ARMS` | `1` = the six arms (the job). `0` before submitting drops to the two g++ DaCe pipelines, for a smoke run on a box without a polyhedral clang |
| `DACE_cache_distaware` | `1`, plus a per-rank build folder; without it ranks load libraries other ranks are still writing |
| `DACE_JOB_SCRATCH` | build scratch root. Defaults to `/dev/shm/dace-corpus-perf` when /dev/shm holds 4 GiB per rank on the node, else `$HOME/.cache/dace-corpus-perf` with a loud warning that compiles are now disk-bound. `TMPDIR` is placed under it. NEVER `/tmp` -- the small shared tmpfs, and filling it has faked corpus failures here; a `/tmp` scratch is refused outright |
| `PYTHONHASHSEED=0` | DaCe determinism |
| `MPI4PY_RC_INITIALIZE=0` `OMPI_MCA_pml=ob1` `OMPI_MCA_btl=self,vader` `UCX_VFS_ENABLE=n` | every python invocation |
| `REPS` | best-of repetitions, default 50 (exported as `CANON_PERF_REPS`) |
| `PYTHON` | interpreter. A batch script does not inherit the interactive PATH; the preflight dies loudly if `import dace` fails |
| `OUT_DIR` | results (default `canon_corpus_perf_job/corpus_perf_results`) |
| `ACCOUNT` `PARTITION` `TIMELIMIT` | sbatch settings, CSCS defaults (`g34` / `normal` / `04:00:00`) |

## Time budget

Measured on the dev box: `poly/lu`, the slowest kernel, costs 15.2 s fixed (build, reference,
correctness) plus 0.766 s per repetition. Summed over all 277 kernels, the timed region at 50 reps
is about 8.6 minutes.

So COMPILING dominates and repetitions are nearly free: quadrupling `REPS` to 200 adds roughly half
an hour to a job whose length is set by builds, and buys tighter medians. The 4 h `#SBATCH` default
is generous for one node; `--limit` bounds a smoke run.

## Results

Under `$OUT_DIR`:

```
perf_json/<suite>_<kernel>.json   per-kernel: every arm, every repetition, denominator, arm provenance
speedup.csv  speedup_table.md     the aggregate, written by the post-step
ranks/                            per-rank partial tables
parallelism_rank<NNNN>.csv        per-rank, per-kernel loop/map counts
```

The post-step runs automatically once the ranks exit. Re-run it alone with:

```bash
python canon_corpus_perf_job/corpus_perf_job.py --out "$OUT_DIR" --aggregate
```

Exit code is non-zero if any kernel errored or miscompiled.

## Plot

`plot_corpus_perf.py` reads the per-kernel JSON above and NEVER re-measures anything. It writes
**TWO figures**, because the four corpora split into two groups with different denominators:

| figure file | corpora | denominator |
|:--|:--|:--|
| `<prefix>_numpy-reference.png` | `np` + `poly` | the timed **parallel** numpy reference |
| `<prefix>_seq-cpp.png` | `tsvc` + `tsvc25` | the `seq-cpp` arm: **sequential** C++ |

They are separate files on purpose. 3x over threaded numpy and 3x over single-core C++ are
different claims: the numpy reference dispatches gemm / 2mm / 3mm / syrk / cholesky into a threaded
OpenBLAS, while `seq-cpp` is single-core by construction. One shared axis would invite a comparison
that is not valid, so each figure carries a caption naming its denominator in words, gets its own
geomean table, and there is NO geomean anywhere that pools the two.

```bash
python canon_corpus_perf_job/plot_corpus_perf.py --results "$OUT_DIR"
python canon_corpus_perf_job/plot_corpus_perf.py --results "$OUT_DIR" --suite tsvc,tsvc25
python canon_corpus_perf_job/plot_corpus_perf.py --results "$OUT_DIR" --arm dace-canon-gcc,dace-canon-llvm
```

Default prefix is `<results>/corpus_speedup_<preset>`; each figure appends its own slug and writes
`.png` plus `.md` beside it. `--suite` selecting one group writes that figure only.

* Each `.md` holds: coverage per corpus, geomean per (corpus, arm) and per arm over THAT figure,
  `n` beside every geomean, then the exclusions by category and by reason.
* Geomeans are over the RAW ratio and only then converted to the signed scale, which spans zero and
  negatives, where a geometric mean is undefined.
* Every y tick prints the signed value AND its raw ratio (`+2 (3x)`), so `+2` cannot be read as 2x.
  `--yscale linear` turns off the default symlog, which keeps `+-1` linear but stops a 1000x
  outlier from flattening the panel.
* `python-scalar` references are never a denominator: the tsvc/tsvc25 oracles (their bars divide by
  `seq-cpp`, so they still plot) and polybench `seidel_2d` (dropped, and counted in the `.md`).
* Miscompiled arms, errored arms and kernels whose denominator cannot be verified are dropped and
  counted, never replaced by 1.0. A partial or killed sweep plots what exists and reports what is
  absent; nothing plottable writes no figure, prints the reason and exits non-zero.
* Result files predating the six-arm labels (e.g. `tests/passes/canonicalize/perf_results/`) are
  reported as STALE and refused, not mixed in. `--denominator baseline` is the mode that reads them,
  against their own baseline arm, and says so in the figure.
* `--preset` `--suite` `--arm` `--min-ms` `--sort` `--yscale` subset or rescale; `--results` is
  repeatable (duplicate kernels resolve to the newest timestamp, reported).
