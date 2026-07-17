# Why BLAS was never multithreaded in the perf jobs — root cause and fixes

**Symptom.** GEMM-heavy kernels (gemm/mlp/k3mm) ran at identical times in every
thread configuration (e.g. `gemm` ~581 ms in ALL codegen_variants cells);
explicit A/B tests showed 2048^3 DGEMM **24x SLOWER at 72 threads than at 1**
(time growing ~linearly with thread count), while DaCe's own OpenMP kernels
scaled fine on the same nodes (heat_3d 2.26x at 72 threads).

**Root cause** (affinity trace, job 4231061): the job scripts export
`OMP_PROC_BIND=close OMP_PLACES=cores` — correct for DaCe's OpenMP kernels —
but libgomp **binds the master thread to its single place (core 0) the moment
any `-fopenmp`-linked kernel `.so` is dlopened** (no parallel region needed;
parent affinity drops 288 cpus -> `{0}` exactly at `sdfg.compile()`). Then:

1. Every subprocess spawned afterwards inherits the ONE-CORE mask. Its libgomp
   resolves `OMP_PLACES=cores` against that mask -> places `'{0}'` -> all 72
   OpenMP threads time-slice core 0. Measured: 423 ms (1T) -> 9,500-10,200 ms
   (72T) for 2048^3 DGEMM — time proportional to thread count.
2. Affinity-derived thread detection (pthreads OpenBLAS, numpy's bundled BLAS)
   sees ONE cpu and silently caps at a single thread — every BLAS-backed
   measurement to date ran single-threaded (the flat ~581 ms gemm cells).

**Exoneration matrix** (jobs 4230984/4230995/4230999/4231024): pure-C
`cblas_dgemm` in the batch step scales 59x (2381 GFLOP/s, threads=openmp
build); the same under `srun`: scales; python+numpy+ctypes dgemm: scales
(2166 GFLOP/s); libgomp triad control: scales (445 GB/s). ONLY children of a
process that had loaded a DaCe kernel collapsed — and `OMP_DISPLAY_ENV`
showed their `OMP_PLACES = '{0}'`.

**Fixes (all applied):**

1. `engine.py`: capture the job's full cpu mask at import
   (`_INITIAL_CPU_AFFINITY`, before any kernel `.so` load);
   `restore_cpu_affinity()` is called (a) in `run_isolated` before spawning a
   measurement child, (b) at `time_sdfg` entry, (c) defensively in
   `configure_dace_process`. Verified: the collapsing repro now reaches
   **2188 GFLOP/s at 72 threads** (job 4231078).
2. All job scripts switched from `openblas threads=pthreads` to a newly built
   **`threads=openmp`** (spack hash rsaxs76, gcc@14.2.0): BLAS threads run on
   the same libgomp as DaCe's regions, honor OMP_PROC_BIND/PLACES pinning, and
   `OMP_MAX_ACTIVE_LEVELS=1` (exported in every script) makes a BLAS call
   inside an already-parallel DaCe region serialize instead of forking a
   nested team. (The old pthreads build additionally self-capped to 1 thread
   under the poisoned mask, and its unpinned spin-wait pool collapsed even
   when forced to 72 via `openblas_set_num_threads`.)

**Implication for existing results:** every prior measurement of a
BLAS-backed kernel (gemm, k2mm/k3mm, mlp, lenet, atax/bicg/gemver via
OpenBLAS lanes) used SINGLE-THREADED BLAS in all lanes. Comparisons were
internally consistent (all lanes equally crippled) but absolute BLAS numbers
and any BLAS-vs-nonBLAS ratio must be re-measured.

**Diagnostics kept** (this directory): `blas_isolation.c` +
`slurm_blas_isolation.sh` (pure-C layer isolation), `blas_matrix_c.py` /
`blas_matrix_d.py` (child-poisoning discriminator cells),
`blas_affinity_trace.py` (per-stage affinity trace that caught the clobber),
`openblas_*` check scripts (canon-pipeline lift verification, source dump,
fresh-child sweeps).
