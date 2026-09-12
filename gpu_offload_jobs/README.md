# GPU offloading job

`OffloadToAccelerator` over the four in-repo corpora, comparing two GPU pipelines. Two jobs, because
correctness and timing cost three orders of magnitude apart.

`--arms` names the pair, baseline first, from `autoopt` (DaCe's `auto_optimize`), `canon`
(canonicalize then offload) and `canon+taskloop` (that path with
`optimizer.gpu_taskloop_heuristics` on). The default, `canon,canon+taskloop`, is the knob A/B;
`--arms autoopt,canon` is the GPU half of the canon perf comparison whose CPU half lives in
`canon_perf_jobs/`. Every flag this submitter does not recognize is passed straight to the driver,
so no script here changes to run a different pair.

| file | what it does |
|:--|:--|
| `submit_daint.sh` | **the entry point.** One sbatch per corpus, for either job |
| `submit_gpu_offload.sh` | the job body. Pins the environment, sruns one rank per module |
| `gpu_offload_job.py` | the rank. Binds its GPU, takes a private build folder, runs its shard |
| `run_local.sh` | the same two sweeps on one machine, no Slurm |

The measurement itself is `tests/corpus/measure_gpu_arms.py`; these scripts only place it on a
cluster. Running that module directly is always a valid way to reproduce a row.

## The two jobs

| job | what a case does | needs a GPU | cost |
|:--|:--|:--:|:--|
| `cpu` | canonicalize for GPU, offload at both knob settings, `validate()`, emit | no | seconds/kernel |
| `gpu` | the above, then compile, run against the corpus reference, time both arms | yes | minutes/kernel |

The `cpu` job is not a weaker version of the `gpu` one. It is where the placement failures actually
show: a descriptor the pass claims for the device while host code still reads it fails `validate()`
with the container named, which is how the BLAS-alpha family (`symm`, `syr2k`, `syrk`, `trisolv`,
`gramschmidt`) reports itself. It is also the only sweep that runs on a device-less runner.

    ./submit_daint.sh smoke              # debug partition, preset S, ~15 min. Run this first.
    ./submit_daint.sh cpu all            # 281 kernels, both arms, no device
    ./submit_daint.sh gpu poly           # one corpus, timed
    ./submit_daint.sh gpu all 2          # four jobs, 2 nodes each
    ./submit_daint.sh gpu all 2 --arms autoopt,canon   # canon against auto_optimize on the GPU

## Why the sweep finishes at all

281 kernels x 2 arms x a CUDA build is not a job that ends. Every kernel is therefore SCREENED
first: it is offloaded at both settings, and the two finalized graphs are compared. The knob is inert
on most of the corpus, and where the graph handed to codegen is identical so is its runtime, so
those kernels are built and timed ONCE and the ratio reported as exactly 1. Only a kernel the knob
actually rewrites pays for two builds.

The digest is taken from the graph, not from the emitted text, and BEFORE `generate_code` is called.
Emitting rewrites what it is handed -- it pads every region boundary and resets the CFG list -- and
it is not reproducible on this branch: the same finalized SDFG emits one of two programs, differing
in block-id suffixes and one `const double x = e;` against `double x; x = e;`. Digesting the text,
or the graph afterwards, reports kernels as rewritten at random.

`--no-screen` forces both arms to be built and timed even where they agree. That is how the screen
itself gets tested, and it turns the agreeing kernels into a null control -- see below.

## Reading the numbers

⛔ **Ratios are only comparable within a rank, and only against the floor.** The two arms of a
kernel are built and run minutes apart, so anything that moves the clock in between lands on the
ratio. On a power-capped part this is the dominant term: the same polybench `cholesky` measured
1.75x, then 0.91x, then 1.12x with a median estimator. The driver now discards warm-up calls and
reports the BEST of N, which a moving cap can only spoil in one direction.

The kernels whose two arms compile the same graph are deliberately left in the table. Their ratios
cannot be anything but 1, so whatever they read IS the floor. On one RTX 4050 at the `paper` preset
they came in at 0.99-1.05, and nothing within +-5% of 1 elsewhere in that table is a result.

The submitted jobs time each arm **20 times** (`REPEATS`, and `--repeats` on the driver). The count
buys robustness against the cap moving part-way through an arm, not a tighter average -- the
reported number is the best of the 20, so a longer run can only find a less-throttled sample.

Each rank binds `CUDA_VISIBLE_DEVICES` to its own module's H100. Without that binding four ranks
drive device 0 and time each other; the numbers still come out, they are just not about the compiler.

## Results as of 2026-08-26

One RTX 4050, `paper` preset, warm-up plus best-of-15 (the jobs now use 20), both arms forced. The taskloop rule now
declines a map whose body extent is written in its own parameter, which is what the last two columns
are about.

| kernel | before the extent rule | after | why |
|:--|--:|--:|:--|
| poly `cholesky` | 1.12 | 1.00 | inner extent `0:i + 1`; no longer a taskloop |
| np `cholesky2` | 1.14 | 1.00 | inner extent `i + 1:N`; no longer a taskloop |
| np `resnet` | 1.08 | 1.00 | fires on a device-wide library node, which is a requirement |
| poly `covariance` | 1.05 | 1.04 | null control -- both arms are the same graph |
| poly `doitgen` | 0.99 | 1.01 | null control |
| np `spmv` | 0.99 | 1.01 | null control |

Open, and neither is the knob's doing -- both arms fail identically:

* `correlation` does not build. `symmetrize_col(j: _[i + 1:M])` lifts an `i`-dependent extent into
  the thread-block dim, and the launch wrapper's signature does not carry `__i`, so nvcc reports
  `identifier "__i" is undefined`. The extent rule above stops the taskloop classifier from creating
  this shape, but the wrapper-signature gap is still there for anything else that reaches it.
* `lenet` runs and disagrees with its reference.
