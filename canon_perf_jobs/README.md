# Corpus perf job

Four in-repo corpora, eight comparison arms plus the `seq-cpp` denominator, TWO figures (the
corpora split into two denominator groups, see below).

| file | what it does |
|:--|:--|
| `submit_daint.sh` | **the entry point.** Preflight, then one sbatch per corpus: 1 node = 4 ranks x 72 threads |
| `submit_corpus_perf.sh` | the job body. Pins the environment, probes every arm, fans the ranks out, aggregates |
| `corpus_perf_job.py` | the rank. Picks its kernels, takes a private build folder, unblocks SIGCHLD, drives the measurement |
| `run_local.sh` | the same measurement on this machine, no Slurm |
| `plot_corpus_perf.py` | the two figures, from the aggregated results |

There is exactly ONE submitter. The old per-group ones (`submit_daint_array.sh`,
`submit_daint_loops.sh`, `run_sweep.sh`, `run_local_{array,loops}.sh`) are deleted -- they differed
only in a suite string, and a second copy of a submitter is how two sweeps end up measuring
different things.

(`cloudsc/` next door is an unrelated job.)

## The eight arms

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
| `dace-simplify+gcc-autopar` | post-`SimplifyPass`, emitted SEQUENTIAL | `g++` | GCC autopar + Graphite, FORCED (`-floop-parallelize-all`) | poly/np: corpus reference. tsvc/tsvc25: `seq-cpp` |
| `dace-simplify+gcc-autopar-default` | post-`SimplifyPass`, emitted SEQUENTIAL | `g++` | GCC autopar + Graphite, gcc's own cost model | poly/np: corpus reference. tsvc/tsvc25: `seq-cpp` |
| `dace-simplify+llvm-autopar` | post-`SimplifyPass`, emitted SEQUENTIAL | `clang++` | Polly autopar, FORCED (`-polly-parallel-force`) | poly/np: corpus reference. tsvc/tsvc25: `seq-cpp` |
| `dace-simplify+llvm-autopar-default` | post-`SimplifyPass`, emitted SEQUENTIAL | `clang++` | Polly autopar, Polly's own cost model | poly/np: corpus reference. tsvc/tsvc25: `seq-cpp` |
| `seq-cpp` | post-`SimplifyPass`, emitted SEQUENTIAL | `g++` | nothing (`-O3`, no autopar, no polyhedral flags) | not a comparison arm: it IS the tsvc/tsvc25 denominator |

Graphite is folded into the gcc autopar arm and Polly into the llvm one. A polyhedral pass that
restructures a nest nobody then parallelizes answers no question the figure asks, so each external
arm is one column that has to prove BOTH of its passes ran.

Each auto-parallelizer appears twice because the gap between forced and default is itself a result:
forcing takes Polly from 1.209x to 7.572x (unforced it declines flat 1-D loops and times sequential
code) while gcc barely moves, 8.058x to 8.088x.

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
| `tsvc25` | 76 | `seq-cpp` | sequential C++ from the same post-simplify SDFG |
| | 281 | | |

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
| 1 (bare, no launcher) | 281 |
| 4 (1 node) | 71, 70, 70, 70 |
| 16 (4 nodes) | 18 x 9, 17 x 7 |

Two reasons, both load-bearing:

* **Load balance.** One corpus per rank leaves the tsvc rank with 151 kernels and the npbench rank
  with 24, so the job lasts as long as its slowest corpus.
* **Attribution.** All arms of a kernel must be timed on the SAME node. Arms split across nodes
  differ by thermal/DVFS state and memory locality as well as by pipeline, and the speedup stops
  being attributable to the pipeline.

The order is fixed, so a re-submission lines up with the per-kernel JSONs already on disk.

## How to run

```bash
# 0. ALWAYS FIRST: 2 kernels per corpus per rank on the debug partition, ~3 min.
#    Proves the toolchain, the arms and the SIGCHLD/cmake chain before a real allocation is spent.
bash canon_perf_jobs/submit_daint.sh smoke

# 1. One corpus, one node (4 ranks x 72 threads):
bash canon_perf_jobs/submit_daint.sh tsvc
bash canon_perf_jobs/submit_daint.sh poly 2          # ...or 2 nodes = 8 ranks

# 2. All four corpora, as four independent jobs:
bash canon_perf_jobs/submit_daint.sh all

squeue --me
```

Corpora are `poly`, `np` (array kernels, divided by the corpus numpy reference) and `tsvc`,
`tsvc25` (loop kernels, divided by `seq-cpp`). One job each rather than one job per group: every
corpus then gets its own queue slot, time limit and results directory, so a slow polybench sweep
cannot consume the wall clock npbench needed. Results land in `results_<corpus>_<preset>/`.

A Grace-Hopper node is four modules, each a 72-core Grace CPU with its own H100, so one node is
4 ranks x 72 threads with nothing left over. `corpus_perf_job.py` shards that corpus's kernels
across the ranks positionally (`SLURM_PROCID` of `SLURM_NTASKS`) and gives each rank a private
build scratch, so no two ranks share a `.dacecache`.

Locally, without Slurm:

```bash
bash canon_perf_jobs/run_local.sh tsvc                 # PRESET=S by default
PRESET=paper OMP_NUM_THREADS=8 bash canon_perf_jobs/run_local.sh poly
```

### The preflight is not optional courtesy

`submit_daint.sh` resolves the spack toolchain, pins `gcc@16.1.0 +graphite` and `llvm@22.1.5 +mlir`
(both are installed twice, and the other variant kills the autopar arms), pins a per-compiler
OpenBLAS, and PROVES Polly emits a parallel region on two probe shapes. A missing package degrades
a column **silently**: no clang kills four arms, a Polly-less clang takes the flags and measures
plain `-O3` under an autopar label, and a missing OpenBLAS drops the DaCe arms to `pure` BLAS
(-20-40x) while still printing a number. `PREFLIGHT=0` skips it -- only for a site where these
spack specs do not resolve.

### Do not replace the sbatch with a bare srun

`slurmstepd` starts every task with **SIGCHLD blocked**. The mask survives fork+exec, reaches cmake,
and cmake's KWSys -- which learns its helpers exited by *receiving* SIGCHLD -- then waits in
`select()` forever. It looks like a stuck configure; it is a lost wakeup. The only place the unblock
works is `corpus_perf_job.py`'s module scope, because `srun` execs it directly with no shell in
between, and `submit_corpus_perf.sh` preserves that chain. A wrapper shell, `setsid`, `nohup`,
`timeout` or `env --ignore-signal=CHLD` all bring the hang back. The rank re-checks the mask at
entry and refuses to start if it is still set, so a broken chain fails in the first second rather
than after an hour of apparently-running cmake.

## Resume

The perf facet writes one JSON per kernel and skips kernels that already have a complete one, so a
killed sweep resumes by re-submitting. Delete a kernel's JSON to re-measure just it, or pass
`--force` to re-measure everything. Records written by superseded arm labels are recognized as
stale and re-measured. The parallelism facet is cheap and always re-runs; it appends, so delete
`parallelism_rank*.csv` before a clean re-run.

## Env

Pinned by the submit script and exported, so the value in the log is the value every rank and every
cmake/compiler child inherits. Do not vary them between compared arms.

| var | why |
|:--|:--|
| `DACE_compiler_cpu_args` | the ONE base flag string. `-ffp-contract=` must be explicit (gcc defaults to `fast`, clang to `on`, so omitting it makes the compiler columns incomparable); `-ffast-math` is rejected, since associative math lets each compiler reassociate reductions differently |
| `OMP_NUM_THREADS` | cores / 4, so 4 ranks do not oversubscribe. Identical for every arm, and gcc's `-ftree-parallelize-loops` is derived from it. `OMP_PROC_BIND=close`, `OMP_PLACES=cores` |
| `CANON_PERF_ARMS` | `1` = all nine arms (the job). `0` before submitting drops to the two g++ DaCe pipelines, for a smoke run on a box without a polyhedral clang. The aggregate step is run with `0` on purpose: it only re-exports JSON, and re-probing nine toolchains there would let a late probe failure cost a finished sweep its outputs |
| `DACE_cache_distaware` | `1`, plus a per-rank build folder; without it ranks load libraries other ranks are still writing |
| `DACE_JOB_SCRATCH` | build scratch root. Defaults to `/dev/shm/dace-corpus-perf` when /dev/shm holds 4 GiB per rank on the node, else `$HOME/.cache/dace-corpus-perf` with a loud warning that compiles are now disk-bound. `TMPDIR` is placed under it. NEVER `/tmp` -- the small shared tmpfs, and filling it has faked corpus failures here; a `/tmp` scratch is refused outright |
| `PYTHONHASHSEED=0` | DaCe determinism |
| `MPI4PY_RC_INITIALIZE=0` `OMPI_MCA_pml=ob1` `OMPI_MCA_btl=self,vader` `UCX_VFS_ENABLE=n` | every python invocation |
| `REPS` | best-of repetitions, default 50 (exported as `CANON_PERF_REPS`) |
| `PYTHON` | interpreter. A batch script does not inherit the interactive PATH; the preflight dies loudly if `import dace` fails |
| `OUT_DIR` | results (default `canon_perf_jobs/corpus_perf_results`) |
| `ACCOUNT` `PARTITION` `TIMELIMIT` | sbatch settings, CSCS defaults (`g34` / `normal` / `04:00:00`) |

## Time budget

Measured on the dev box: `poly/lu`, the slowest kernel, costs 15.2 s fixed (build, reference,
correctness) plus 0.766 s per repetition. Summed over all 281 kernels, the timed region at 50 reps
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
python canon_perf_jobs/corpus_perf_job.py --out "$OUT_DIR" --aggregate
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
python canon_perf_jobs/plot_corpus_perf.py --results "$OUT_DIR"
python canon_perf_jobs/plot_corpus_perf.py --results "$OUT_DIR" --suite tsvc,tsvc25
python canon_perf_jobs/plot_corpus_perf.py --results "$OUT_DIR" --arm dace-canon-gcc,dace-canon-llvm
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
* Result files predating the current arm labels (e.g. `tests/passes/canonicalize/perf_results/`) are
  reported as STALE and refused, not mixed in. `--denominator baseline` is the mode that reads them,
  against their own baseline arm, and says so in the figure.
* `--preset` `--suite` `--arm` `--min-ms` `--sort` `--yscale` subset or rescale; `--results` is
  repeatable (duplicate kernels resolve to the newest timestamp, reported).
