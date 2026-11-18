# Transient removal audit

Static audit of the canonicalization passes that remove, shrink, privatize or forward a transient (2026-11-18,
fix-1117). Nothing here was compiled or run: the two fixes and their tests are untested. Line numbers refer to the
working tree after the fixes.

## Where the passes run

| pass | file | runs in |
|---|---|---|
| `ArrayElimination` (drives `RedundantArray`, `RedundantSecondArray`, slice/squeeze view removals) | `passes/array_elimination.py` | `StructuralCleanup` (every `rail`, ingest, finish), `dead_dataflow_units` (statement round, derive), `after_fusion_units`, `finish` (last unit) |
| `RedundantArray` / `RedundantSecondArray` alone | `dataflow/redundant_array.py` | `after_fusion_units`; `finalize.py:757` after `expand_gathered_dots` |
| `RedundantMapReadCopy` | `dataflow/redundant_array.py:2514` | `finish` |
| `DeadDataflowElimination` | `passes/dead_dataflow_elimination.py` | `PropagateAndPrune` (statement round), `dead_dataflow_units`, `after_fusion_units` |
| `PruneUnreferencedTransients` | `canonicalize/prune_unreferenced_transients.py` | last unit of `StructuralCleanup`, `finish` |
| `RemoveViews` | `passes/remove_views.py` | ingest, `StructuralCleanup` |
| `FoldScalarReadCopies` | `canonicalize/fold_scalar_read_copies.py` | ingest, `StructuralCleanup`, `finish` (`scopes=True`) |
| `ForwardOffsetCopies` | `canonicalize/forward_offset_copies.py` | `StructuralCleanup`, after `ArrayElimination` |
| `EliminateCopyBack` | `canonicalize/eliminate_copy_back.py` | normalize, statement round, `after_fusion_units` |
| `ForwardStoreToLoad` | `canonicalize/forward_store_to_load.py` | statement round |
| `ScalarToSymbolPromotion` | `passes/scalar_to_symbol.py` | statement round, `IvSubstitutionFissionFixpoint` |
| `PrivatizeScalars` / `PrivatizeScalarsAndArrays` (scalar and array fission) | `passes/scalar_fission.py` | statement round, derive (after each lift) |
| `DeadCarriedStoreElimination` | `canonicalize/dead_carried_store.py` | `dead_dataflow_units`, derive |
| `DeadFillElimination` | `passes/dead_fill_elimination.py` | derive, recompose, `finish` |
| `ConvertLengthOneArraysToScalars` | `passes/length_one_array_scalar_conversion.py` | normalize, `finish`, `finalize_transient_storage` |
| `FuseCopyChains` | `offloading/fuse_copy_chains.py` | `offload_to_gpu`, before the device move |
| `ShrinkMapLocalTransients` | `canonicalize/shrink_map_local_transients.py` | `finalize_transient_storage` |
| `SplitMapLocalTransients` | `canonicalize/split_map_local_transients.py` | `finalize_for_target`, before storage |
| `MinimizeTransients` (coalescing + `BufferSharing`) | `passes/minimize_transients.py` | `finalize_for_target` on CPU; `GPUCodegenPreprocessPipeline` on GPU after `MoveArrayOutOfKernel` |
| `MoveArrayOutOfKernel`, `BoundDataDependentExtents` | lead-owned | GPU preprocess; `finalize_for_target` |
| `PrivatizeScatterReduction`, `RangePrivatizeScatterReductions`, `PrivatizeScatterPerBlock` | specialization | `cpu_specialize`; GPU preprocess |

## Refusals

"Required" means the check guards correctness as written; "no" means a sound rule exists that the check does not
try. Paths are under `dace/transformation/`.

| pass | refuses when | file:line | required? | status |
|---|---|---|---|---|
| `RedundantArray`, `RedundantSecondArray` | the copy covers the removed array only up to the spelling of its extent (`int_floor` in the memlet, sympy `floor` in a shape annotation; `Lb**2` against `ipow`) | `dataflow/redundant_array.py:21` (`_subset_has_shape`, used at :603, :931, :1019, :1255, :1471, :1957, :2115, so the slice removals and the reshape test gain it too) | no | **fixed**: structural compare first, then `symbolic.equal` |
| `ForwardOffsetCopies` | a reader of the shifted copy sits in a loop or branch after the copy instead of a top-level state of the copy's region | `canonicalize/forward_offset_copies.py:119-127` | no | **fixed**: the enclosing block stands for the reader; it must not write the source anywhere |
| `ForwardOffsetCopies` | destination subset is not structurally `Range.from_array`, source and destination sizes differ structurally | `forward_offset_copies.py:112-114` | no (same spelling issue) | proposed: `covers_precise` / `symbolic.equal` |
| `ForwardOffsetCopies` | a gather (`t[i] = a[idx[i]]`) or strided copy | `forward_offset_copies.py:112` (`step != 1`), the copy must be an edge | yes for the offset form | by design: forwarding a gather trades one load of `t` for two loads, a recompute-or-store choice |
| `ForwardOffsetCopies`, `RedundantArray`, `MinimizeTransients` | source and copy in different storages (host/device twins) | `forward_offset_copies.py:105`, `redundant_array.py:723`, `:1343`, `minimize_transients.py:177` | yes | by design |
| `ArrayElimination` | the container is accessed in another state reachable from this one (copy in one state, readers in a later one) | `passes/array_elimination.py:259` | yes for an in-state fold | by design; the cross-state offset copy is `ForwardOffsetCopies`' job |
| `ArrayElimination` | a one-element copy of a WAR carrier (`t = s; s = f(...)`) | `array_elimination.py:107` | yes (the delay line) | by design |
| `RedundantArray`, `RedundantSecondArray` | the transient has any other access node or interstate use in the SDFG | `redundant_array.py:772`, `:1382` | yes for a whole-descriptor removal | by design |
| `RedundantArray`, `RedundantSecondArray` | strides differ and the other end is a view or a nested SDFG | `redundant_array.py:803-821`, `:1422-1440` | yes (the nest indexes through its own strides) | by design; `MinimizeTransients` placement has the same `twin` rule (`minimize_transients.py:199`) |
| `RedundantArray` | the copy's source access node has `setzero` | `redundant_array.py:534` | yes | by design |
| `RedundantSecondArray` | the copy is not the only write (`in_degree != 1`) | `redundant_array.py:1180` | yes | by design |
| `EliminateCopyBack` | the producer is not a map exit (a tasklet or library node writing `t`, then `A[S] = t`) | `canonicalize/eliminate_copy_back.py:127` | no | by design: `RedundantArray` takes that shape |
| `EliminateCopyBack` | strided copy, or the transient has a second access node | `eliminate_copy_back.py:133`, `:138` | yes | by design |
| `ShrinkMapLocalTransients` | the accesses name different subsets of the per-iteration box (`t[i, 0:K]` written, `t[i, k]` read in an inner map) | `canonicalize/shrink_map_local_transients.py:112` | no | proposed: drop the dimensions every access indexes by the same enclosing-map expression, keep the rest whole |
| `ShrinkMapLocalTransients` | accesses in two states; a WCR or `other_subset` access | `shrink_map_local_transients.py:71`, `:106` | yes (`SplitMapLocalTransients` renames per scope first) | by design |
| `MinimizeTransients` coalescing | the transient lives in more than one state, or is accumulated into (WCR) | `minimize_transients.py:179`, `:245` | yes (values must match while both live) | by design |
| `DeadDataflowElimination` | a non-transient write; a multi-element pointer connector of a tasklet; a nested SDFG or library node that also reads the container | `passes/dead_dataflow_elimination.py:404`, `:433`, `:443` | yes | by design |
| `DeadCarriedStoreElimination` | nested SDFG, conditional, WCR, non-affine access | `canonicalize/dead_carried_store.py:21-24` | yes (a false positive drops a needed store) | by design |

## Corpus drivers

- **ls3df `hpsi`** (`hpsi[:] = beta @ (dij @ overlap)`): the GEMM writes a transient copied whole into `hpsi`.
  `RedundantArray` takes it: one access of the transient, a library-node producer that reads neither `hpsi` nor a
  view of it (`redundant_array.py:614-633`), equal storage and C strides. So, read statically, `hpsi` is the
  GEMM's output; confirm on the generated code.
- **fv3** shifted operands (`q_i = q[lo:hi]`) are slices, so `RemoveViews` handles them, not a copy pass. A copy
  that does land before the time loop is now forwarded into the loop body.
- **cp2k_grid_integrate** scratch: a contraction output read in part. Narrowing it is a recompute choice (hinted,
  `canon_perf_gaps.md`), not a removal. **trs4** `sink`, copied in and out per multiply: not analysed here; it needs
  the canonical SDFG to see which check keeps the copies.
- **lavamd**: see below.

## For the lead-owned files

- `bound_data_dependent_extents.py:315` (`extent_access_bound`) and `:270` (`extent_upper_bound`) accept only a
  numeric coefficient for the data-read symbol. lavamd's `neighbor_offsets` extent `(1 + count) * particles_per_box`
  has the coefficient `particles_per_box`, a signature symbol, so it gets no capacity even once `count` is bounded.
  The sound rule: accept a coefficient whose free symbols are all argument extents, which are nonnegative under
  `SetSymbolNonnegativeAssumptions`.
- `bound_data_dependent_extents.py:301` (`access_bounds`) keeps a relation only when the symbol's coefficient is
  exactly 1. `nl[l, 0:2 * count]` gives `2 * count <= w`, which bounds `count` by `w // 2`; dropping it is sound but
  loses the bound.
