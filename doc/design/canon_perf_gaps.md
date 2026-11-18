# Canonical pipeline performance gaps

This file tracks where the canonical pipeline (`canonicalize`, then `offload_to_gpu` on GPU, then
`finalize_for_target`) is slower than the best baseline, what causes each gap, and whether canonicalization closes
it or leaves it to an optimizing agent. Canonicalization does general work only: parallelization, fusion and
transient removal driven by structural facts. Time tiling, tile sizes, schedules and other tuning are the agent's
job, so the gaps they would close stay open by design and are listed for the evaluation.

Status values: **fixed** (commit named), **proposed** (design change, not implemented), **by design** (canon leaves
it to the agent), **hypothesis** (from static reading, not yet confirmed against the generated code).

## Data

Measured on MI300A, fuzzed preset, float64, 2026-11-17/18. Baselines are the best of `cc`, `cc_autopar`, `numba`
and `pluto` on CPU, and of `triton` and `ppcg_hip` on GPU (`baselines-rev-1117a2-5af033a76`). Canon runs:
`c1118s` (scicomp40 CPU), `vs1118l2` (llr40 CPU), `g1117s` (scicomp40 GPU), all at fix-1117 before the fixes below.

| suite | kernels | geomean canon / best |
|---|---|---|
| scicomp40 CPU | 37 | 0.59 |
| llr40 CPU | 40 | 0.34 |
| scicomp40 GPU | 35 | 1.34 |

## GPU

| kernel | canon / best | canon ms | best ms | status |
|---|---|---|---|---|
| gem | 64.0x | 165.5 | 2.58 triton | fixed, 1f9298431, untested (no GPU run; mapreduce_test passed once on the login node) |
| cloudsc | 45.7x | 73.9 | 1.62 triton | not yet analysed |
| cp2k_grid_integrate | 38.5x | 2203.4 | 57.2 triton | by design, hint 7fd2f6650 + 09f005b08, untested |
| qe_vloc_psi_k_acc | 20.2x | 21.2 | 1.05 triton | proposed |
| fv3_dycore | 10.9x | 256.0 | 23.6 triton | copies fixed, 6587cb40c, untested |
| lulesh | 9.4x | 40.8 | 4.36 triton | fixed, ce0afd4a4, not yet re-measured |
| cp2k_density_matrix_trs4 | 8.6x | 24.3 | 2.82 triton | by design (owned by the scan work) |
| lavamd | 4.9x | 198.2 | 40.4 triton | proposed |
| minife | 2.6x | 645.9 | 251.5 triton | not yet analysed |

- **gem.** `MapReduceFusion` kept the reduction's identity write inside the block map because a write of the same
  output outside the map (the leftover points after the blocks) looked like a conflict, although it provably
  misses every tile. The blocked loop then ran as `npoints/1024` thread blocks with every lane adding atomically
  into one `phi` element. The fix skips a write outside the map whose subset does not intersect the tiles; the
  main kernel becomes one grid over every point with one block reduction and one atomic per block.
- **lulesh.** `TileKernelsCoalesced` read the selected `gpu_block_size`, which is in CUDA order (reversed map
  parameters), as if it were in map order. On the `numElem x 4 x 8` hourglass kernel the outer axis took the single
  thread of `z` and the tiled block shrank to 32 threads. Same commit: the coalesced axis of a domain-matched block
  now starts at half a warp of the backend (`gpu_warp_size() // 2`) instead of a configured 64.
- **cp2k_grid_integrate.** One thread per task (about 2.6M tasks) is right, but each thread holds about 72 KB of
  private scratch: the tensordot chain fills 1125 + 2025 + 3645 elements and the final gather into `hab` reads about
  100 of them. The full contraction is the canonical form of the numpy source. Narrowing it to the elements read is
  left to the agent; `AnnotateLoopKinds` now records a `specialization_hint` on a contraction whose transient output
  is read by fewer elements than it holds (both counts named, 7fd2f6650). The triton baseline also puts lanes over the 5^3 cube points, a width choice left to the agent.
- **qe_vloc_psi_k_acc.** The whole band map runs on the host: a task loop that encloses an FFT the kernel cannot
  call is pinned to the host with its subtree (`offload_to_accelerator.py`), so each band runs CPU FFTW on stack
  grids and the inputs are copied to the host and back. Proposal: a `BatchMapFFTs` pass in `offload_to_gpu`, beside
  `BatchMapProducts`, that splits the band map around its FFTs and emits one batched hipFFT over stacked grids.
  Expected 21 ms to 1-2 ms. Blocked on `MapFission` handling the scatter guard inside the band body; pending a
  decision. Until then it is an evaluation gap: an agent would stage all bands' grids at once and call the batched
  vendor FFT pair, as the triton baseline does with `cupy.fft`.
- **fv3_dycore.** Each time step issues about 204 device-to-device copies of `nk` elements, from `copy_corners_x/y`,
  which spell out 36 single-column copies per call; the frontend chains them as access nodes of `q` joined by copy
  edges. Fixed (6587cb40c): `FuseCopyChains`, run in `offload_to_gpu` before the device move, makes a chain whose
  copies move the same extents, no destination meeting another copy's source or destination, one map with a tasklet
  per copy: about 6 kernels per step instead of about 204 copies. The remaining 33 launches per step are a launch-only loop (a
  megakernel candidate).
- **cp2k_density_matrix_trs4.** Per iteration: 20 launches, 10 stream syncs and two whole-array copies to the host,
  which then runs the `cumsum(...)[-1]` reductions serially. The source keeps the left-to-right summation order on
  purpose (a reassociated `np.sum` fails validation, per the kernel's docstring), so canon keeps the scan, and
  its last element is computed by a serial host loop over `3 * bs^2 * nbr` elements (`canon_gpu.cpu.cpp:613`). Between the GEMMs, small elementwise kernels (`_convert_to_int64_`, `_Add_` on the
  3-element `c_pos` tables, `where`) are not fused. The host branches on convergence every iteration, so this is not
  a launch-only loop. What an agent would do: run the ordered reduction on the device in one pass (one thread per
  array, or a decoupled look-back scan), keep the iteration loop on the host, and fuse the elementwise chain.
- **lavamd.** One thread per box with the particle-pair loop serial in the thread and global atomics into `fv`.
  The index tables `pointers` and `neighbor_offsets` have data-dependent sizes, so a fold kernel, a copy to the host
  and a sync run before the main kernel can be allocated. The lift and the fold are `MoveArrayOutOfKernel`
  (`extent_bounds` / `compute_bound`, `move_array_out_of_kernel.py:480-560`): a per-thread array whose extent names a
  symbol read from data gets one global slot per thread, sized by a fold of that symbol's maximum. Proposed (not
  implemented, it needs runs to get right): recompute the index tables at their use. The consumer reads
  `neighbor_offsets[c, w]` through the flattened `bj[i1]` with `c = i1 // ppb`, `w = i1 % ppb`, and
  `neighbor_offsets[c, w] = box_offsets[pointers[c]] + within_box[w]`, where `pointers[c]` is `l` for `c == 0` and
  `neighbor_list[l, c - 1]` otherwise. Forwarding that expression (a select, then two gathers) into the consumer's
  index removes both tables, so `MoveArrayOutOfKernel` has nothing to lift: no allocation, no fold kernel, no copy
  to the host, no sync. The box body is then an init map plus one reduction map, which `BlockTileKernels` can stride
  over the particles. The diagnosis job (751988) was cancelled, so which `BlockTileKernels` check refuses today is
  not confirmed.

## CPU

No CPU code was generated for these kernels (the codegen job was cancelled), so every entry below comes from the
numpy sources, the pass code and the GPU code of the same canonical form (gem, trs4, cp2k_grid_integrate). Nothing
here is confirmed against generated CPU code or re-measured.

| kernel | canon / best | canon ms | best ms | status |
|---|---|---|---|---|
| cp2k_density_matrix_trs4 | 4.17x | 17.2 | 4.12 numba | by design |
| cp2k_grid_integrate | 1.98x | 322.4 | 162.9 numba | by design (see GPU) |
| jacobi_2d | 1.84x | 755.0 | 411.2 pluto | by design |
| warpx_field_gather | 1.84x | 229.7 | 125.1 numba | hypothesis, fusion; needs the CPU code |
| gem | 1.31x | 223.0 | 169.8 numba | fixed in `MarkSIMDMaps`, untested |
| ls3df_kleinman_bylander_nonlocal | 1.26x | 1107.0 | 875.5 cc_autopar | open; same OpenBLAS as the baseline |
| tsvc_2_s115 | 1.77x | 45.7 | 25.8 pluto | by design |
| tsvc_2_vag | 1.45x | 29.0 | 20.1 numba | hypothesis; needs the CPU code |
| wf_triangular | 1.40x | 55.0 | 39.2 pluto | by design |
| tsvc_2_s275 | 1.29x | 42.5 | 32.9 numba | accepted |
| tsvc_2_s233 | 1.23x | 26.2 | 21.3 pluto | by design |

- **Two OpenMP runtimes (measurement environment).** 11 of the 40 CPU runs map two `libgomp` copies in the grading
  process: the spack gcc-16 one, pulled in through OpenBLAS's RPATH, and `/usr/lib64/libgomp`. Affected:
  cp2k_grid_integrate, cp2k_density_matrix_trs4, ls3df_rayleigh_ritz_rotation, ls3df_scf, qe_cegterg, minife, nbody,
  qe_vexx_k, lulesh, seissol_tensor_contraction, quatrex_rgf. Two thread pools then compete for the cores; the CPU
  numbers of these kernels are suspect until a run with one runtime confirms them.
- **gem (fixed, untested).** The kernel is a points x atoms sum of `charge * exp(-kappa r) / (diel r)`; numba runs it
  as one scalar loop per point. Canon shares out blocks x points and walks atoms in an inner sequential map, and the
  3-element `d * d` sum is a map of its own inside it. `MarkSIMDMaps` marked only a map whose body holds no map, so
  the 3-trip map took the `simd` clause and the atom loop none; with reassociation off gcc then cannot vectorize the
  sum over atoms, and `exp` stays a scalar libm call (`-fno-math-errno` and the libmvec declarations are already in
  the flags). Fix: a nested map over a small literal domain (below `SMALL_CONSTANT_EXTENT`) whose every write lands in
  a scope-local transient is straight-line code of its parent (`is_straight_line_map`): it takes no clause of its
  own and the enclosing map takes it, so the atom loop gets `simd reduction(+:...)` and `exp` its vector form. A
  small map at the top level keeps its own clause. Test: `tests/codegen/simd_pragma_test.py`
  (`test_a_small_literal_inner_map_into_a_local_vectorizes_the_enclosing_loop`). Expected: from 223 ms to below
  numba's 170 ms. Not compiled or timed yet.
- **ls3df_kleinman_bylander_nonlocal (open).** Three GEMMs. The baseline uses the same library: the numpy-to-C
  emitter writes `cblas_dgemm` (`translators/numpyto_c/emit.py:1520`) and links `blas`, which resolves to the same
  spack OpenBLAS (`languages.py:2011`, `envs/toolset.yaml:90`). `cc` (1507.6 ms) runs one thread and `cc_autopar`
  (875.5 ms) the team (`flags.py:127`). So the gap is neither the vendor nor the 2-runtime issue; ls3df is not among the 11
  flagged runs. What remains is in the canonical form around the calls: whether `beta.T @ psi` keeps the transpose
  as a BLAS flag (`transposes_as_blas_flags`) or makes a copy of `beta` (843000 x 815, 5.5 GB), and whether `hpsi`
  is written in place or through a temporary and a copy. This needs the generated code.
- **warpx_field_gather, tsvc_2_vag (hypothesis).** The per-particle index and weight planes of warpx (`_tap3`'s
  `ix_b/iy_b/iz_b` int64 planes, the `sx_*` shape-factor arrays and `np.where` selections), and the vag gather, may
  stay materialized where numba runs one fused loop. No code exists to confirm either; no fix is made blind.
- **tsvc_2_s275 (accepted).** If-conversion turns every skipped column into an unconditional read, select and write,
  about twice the traffic on half the columns, in exchange for a vectorized loop. Accepted as the default.

Entries for the evaluation (canon leaves these gaps to the agent by design):

- **cp2k_density_matrix_trs4: 4.17x, 17.2 ms against numba's 4.12 ms.** Root cause: every block multiply is 8 to 10
  maps (GPU form: prod init, `row_of`/`row_cols`, the product with `c_pos`, the `sink` fill, the prefix memset, the
  scatter add, the prefix, the filter). Index setup on the loop-invariant `col_idx` is fused into the compute maps
  and redone every multiply, and `sink` is copied in and out each time. That is about 10 to 20 regions or barriers
  per TRS4 step against numba's 5. Why canon leaves it: loop-invariant code motion hoists whole states and top-level
  tasklets only (`loop_invariant_code_motion.py:239-321`, `:143-176`). Hoisting an invariant map out of a fused state
  would unfuse it, and the source rebuilds the pattern inside its multiply. An agent would compute the CSR
  pattern (`c_pos`, `valid`) once outside the TRS4 loop, write the product into `c_blocks` without `sink`, and fuse
  the block product, scatter and filter into one per-row loop, as numba does.
- **jacobi_2d: 1.84x, 755 ms against Pluto's 411 ms.** Root cause: one team around the time loop
  (`HoistParallelRegion`) with two `omp for` sweeps and their barriers per step; each sweep streams `A` and `B`
  once. `StripMineInterchange` refuses because the stencil reads distance 1 in both axes. Why canon leaves it: canon
  never tiles across time steps. An agent would time-tile (diamond or skewed tiles) as Pluto does, which needs a tile
  size.
- **tsvc_2_s115: 1.77x, 45.7 ms against Pluto's 25.8 ms.** Root cause: a triangular solve; the `i` sweep is
  parallel under the carried `j` loop, so the hoisted team pays one barrier per `j` trip: about 22.8k barriers at
  about 2 us is the measured 45.7 ms. `StripMineInterchange` refuses because `a[j]` is read by every tile and written
  by one. Why canon leaves it: removing the barriers needs a blocked solve, with a block size, or a doacross. An agent
  would block `j`: solve the diagonal block, then update the rest of `a` with one parallel matrix-vector product per
  block.
- **wf_triangular: 1.40x, 55.0 ms against Pluto's 39.2 ms.** Root cause: the wavefront skew already tiles the nest
  64 x 64 (`parallelization/wavefront/skew.py:73`), with a sequential tile diagonal and about 2N/64 barriers. The
  rest is the tile interior, a row recurrence that does not vectorize, and the tile size. Why canon leaves it: tile
  size and interior schedule are tuning. An agent would tune the tile and vectorize the interior across the tile's
  anti-diagonal.
- **tsvc_2_s233: 1.23x, 26.2 ms against Pluto's 21.3 ms.** Expected root cause (from the passes): the two `j` recurrences of each
  `i` become two nests: the `aa` column recurrence, strip-mined into `i` tiles, and the `bb` row recurrence, parallel over `j` with
  `i` inside. Each is one sweep with its own barrier, so the residual is cache reuse between the two sweeps over `cc`.
  Why canon leaves it: fusing the two nests back needs the tile shape Pluto picks. An agent would tile `(i, j)` and
  run both recurrences per tile.

## Tuning-style settings in the pipeline

Listed for review; none of them is removed yet. Kept by decision: `ChooseOmpSchedule`, `SpecializeSmallGemms`,
thread coarsening by 4 (`tile_kernels_coalesced.py`, `coarsen_kernel_threads.py`), `CoarsenScatterKernels`,
`PrivatizeScatterPerBlock` and the grid-stride hint. Removed: `BoundMaskedMaps` from `cpu_specialize` (8a50409ba).

| where | what it tunes |
|---|---|
| `cpu_specialization/machine.py:34-38` | fork/join and memcpy thresholds measured on one host, scaled by core count |
| `cpu_specialization/calibrate_thresholds.py` | host calibration of those thresholds; not called by the pipeline |
| `machine.py:42`, `decide_cpu_parallel_axes.py:378` | `ITERATIONS_PER_THREAD = 4` picks the collapse depth |
| `decide_cpu_parallel_axes.py:312` | a map below `max(64, cores)` iterations runs sequentially |
| `machine.py:46-47`, `strip_mine_interchange.py:1215` | strip-mine tile sized to fit L2, between `MIN_TILE` and `MAX_TILE` |
| `hoist_parallel_region.py:809`, `codegen/targets/cpu.py:4475` | run-time `num_threads(work / 1024)` cap on hoisted teams |
| `parallelization/wavefront/skew.py:72-73` | wavefront tile 64 on CPU (measured at N = 768, 4 threads) and on GPU |
| `gpu_specialization/coarsen_scatter_kernels.py:23-25` | `COVERAGE = 16`, `MAX_FACTOR = 64` for scatter coarsening |
| `chunk_stream_compaction.py:39,43` | fixed compaction block, 16384 on CPU and 1024 on GPU |
