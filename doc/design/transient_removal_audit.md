# Transient removal audit

Static audit of the canonicalization passes that remove, shrink, privatize or forward a transient (2026-11-18,
fix-1117; second and third pass the same day). Nothing here was compiled or run: every fix and test is untested. Line numbers refer to the
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
| `ForwardOffsetCopies` | destination subset is not structurally `Range.from_array`, source and destination sizes differ structurally | `forward_offset_copies.py:112-120` | no (same spelling issue) | **fixed** (second pass): bounds and sizes compared by `same_extent` (structural, then `symbolic.equal(..., is_length=False)`), unit steps checked on both sides |
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
| `ShrinkMapLocalTransients` | the accesses name different subsets of the per-iteration box (`t[i, 0]` and `t[i, 1]`), or sit in a map nested in the body (`t[i, 0:K]` written by an inner map, `t[i, k]` read in another) | `shrink_map_local_transients.py:78` (one scope), `:112` (one subset) | no | **fixed** (second pass): `shrink_shared_dimensions` (`:414`, helpers `map_body_accesses` `:117`, `shared_dimensions` `:159`) shrinks each dimension every edge names by one structurally equal unit-step range through no symbol an inner map defines, keeps the rest whole and the packed layout; not for `GPU_Global`, offset descriptors, nested-SDFG endpoints, the body's exit |
| `ShrinkMapLocalTransients` | accesses in two states; an `other_subset` access | `shrink_map_local_transients.py:71`, `:106` | yes (`SplitMapLocalTransients` renames per scope first; a copy names the other side's index) | by design |
| `ShrinkMapLocalTransients` | a WCR access | `shrink_map_local_transients.py:106` | no: renaming the one box an iteration touches is a bijection, a read-modify-write included | the one-box path still refuses; the per-dimension path takes it |
| `MinimizeTransients` coalescing | the transient lives in more than one state, or is accumulated into (WCR) | `minimize_transients.py:179`, `:245` | yes (values must match while both live) | by design |
| `DeadDataflowElimination` | a non-transient write; a multi-element pointer connector of a tasklet; a nested SDFG or library node that also reads the container | `passes/dead_dataflow_elimination.py:404`, `:433`, `:443` | yes | by design |
| `DeadCarriedStoreElimination` | nested SDFG, conditional, WCR, non-affine access | `canonicalize/dead_carried_store.py:21-24` | yes (a false positive drops a needed store) | by design |

## Views

| pass | refuses when | file:line | required? | why |
|---|---|---|---|---|
| `RemoveViews` | `StructureView` / `ContainerView`, or a dtype-reinterpreting view | `passes/remove_views.py:107`, `:132` | yes | no element-for-element splice exists |
| `RemoveViews` | an ordering edge on the side the splice ignores, with no data edge to re-home it onto | `remove_views.py:143` | yes | dropping it drops a happens-before |
| `RemoveViews` | the view binds a nested SDFG connector | `remove_views.py:871` (`_binds_nested_sdfg` `:823`) | yes under No-View | the connector descriptor is the container it is bound to; splicing needs the nest's descriptor rewritten (`c186a341b`) |
| `RemoveViews` | a library node reads the view whole and has no `operand_dims` for the slice | `remove_views.py:794-810`, `:861` | by design | the view gives the node its operand shape until expansion |
| `RemoveViews` | a symbolic reshape whose accesses would divide by a run-time size (lavamd `neighbor_offsets.reshape(-1)`, extent `(1 + count) * particles_per_box`) | strategies 2/3, `2446a10f4` | by design | the view is free; a `//` and `%` per access is not |
| `RemoveViews` | an interstate subscript of the view the rewriter cannot express | `_interstate_feasible` | yes | the view would survive in the condition |
| `RedundantArray` (view twin) | strides differ and the other end is a view or a nested SDFG | `redundant_array.py:803-821` | yes | see the table above |

View chains (a view of a view) fold one link per fixpoint round; the viewed descriptor may itself be a `View`, nothing
refuses it. A view of a non-transient folds like any other: it names no storage of its own.

Spelling (`int_floor` against `floor`, `int_ceil` against `ceiling`, `ipow` against `**`, one name under two
assumption sets) is compared by value everywhere in these files (third pass) through one helper,
`same_extent` (`dataflow/redundant_array.py:22`: structural first, then `symbolic.equal(..., is_length=False)`;
scalars, shapes and `Range`s), which `ForwardOffsetCopies` reuses instead of its own `same_value`:

| site | file:line |
|---|---|
| `_subset_has_shape` | `redundant_array.py:37` |
| `RedundantArray` View->Access / Access->View shape, reshape test, view-to-view simple case | `redundant_array.py:743`, `:756`, `:937`, `:978` |
| `RedundantSecondArray` View->Access / Access->View shape | `redundant_array.py:1363`, `:1378` |
| `SqueezeViewRemove`, `UnsqueezeViewRemove` subsets | `redundant_array.py:1734`, `:1823` |
| `_is_identity_view` (shape, strides, offset, subset) | `redundant_array.py:2230` |
| `staged_dims` (`same_size`: the old `simplify` test, then by value), `RedundantMapReadCopy.staging` | `redundant_array.py:2467`, `:2579` |
| `RemoveViews`: `library_node_operands`, full-view rewrite, linearization full range, whole-view tasklet part (both ends) | `remove_views.py:805`, `:997`, `:1181`, `:1237`, `:1311` |
| `ForwardOffsetCopies` whole destination, equal sizes | `forward_offset_copies.py:113`, `:115` |
| `ShrinkMapLocalTransients` `uniform_subset`, `shared_dimensions`, `names_the_box` | `shrink_map_local_transients.py:113`, `:183`, `:308` |

Left structural: `updates_in_place` (`redundant_array.py:371`, element indices, not extents) and the stride tuples of
`SqueezeViewRemove` (a stride, not an extent).

## Counterexamples (third pass)

| fix | counterexample | broke it? | status |
|---|---|---|---|
| `ShrinkMapLocalTransients` shared dimensions | a View of `t[i, 0:K]` with its own stride `N`, `t` column-major: shrunk to `(1, K)`, the row is packed and the View reads past it | yes | **fixed**: `map_body_accesses` refuses a View endpoint (`:151`). The one-box path (`map_local_accesses`) has the same exposure for a view whose stride a shrink changes; not changed |
| same | a nested map defining the indexing symbol; a WCR from a nested map; a library node reading `t[i, 0:K]` (expands from the resized descriptor) | no | `defined` excludes the symbol; the buffer is per body iteration, so no WCR spans two |
| `ForwardOffsetCopies` loops and branches | a loop writing the source after the read (next trip reads the new value); `k` rebound in a branch arm before the read | no | `_writes` over the whole block; `_offset_rebound` walks every region nested in a later block |
| same | a copy inside a map (offset names a map parameter) | no | the transient must leave the map through its exit, a second write, refused |
| `BoundDataDependentExtents` access bounds | the bounding access runs in a loop over ANOTHER transient's extent (`pad[trip]`, `trip = 0`, `count > w`) | yes | **fixed**: `access_bounds` takes as nonzero only the sized transient's own extents (`:290`) |
| same | a loop or map of a literal trip count `0` | yes | **fixed**: `nonzero_trips` (`:102`) wants a positive literal |
| same | a symbol assigned twice in one arm of a conditional (`n = sizes[0]`, access, `n = sizes[1]`) | yes | **fixed**: one assignment per arm (`:185`) |
| same | an extent decreasing in the symbol; an argument extent that is zero (`w = 0` gives `count <= 0`, sound) | no | negative coefficients refused |

Still open in the older `set_capacities`: it shares one `nonempty` set across every sized transient, the
pattern fixed above for `access_bounds`.

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
- **lavamd**: the per-box scratch (`pointers`, `first_j`, `neighbor_offsets`) has a data-dependent extent; a
  lifted box loop leaves them map-body local, sized by `count`, which the lead-owned extent bound handles (below).
  `rv[bj]` and `qv[bj]` are gathers: kept by design.
- **row scratch under a map** (`t[i, :]` produced by one inner map, read by another; cloudsc per-column
  temporaries after a column map is lifted, minife row loops): previously kept at full extent, one `(N, K)` stack
  buffer per iteration; the per-dimension shrink gives `(1, K)`. Not confirmed on the canonical SDFGs.

## For the lead-owned files

- `bound_data_dependent_extents.py:315` (`extent_access_bound`) and `:270` (`extent_upper_bound`) accept only a
  numeric coefficient for the data-read symbol. lavamd's `neighbor_offsets` extent `(1 + count) * particles_per_box`
  has the coefficient `particles_per_box`, a signature symbol, so it gets no capacity even once `count` is bounded.
  The sound rule: accept a coefficient whose free symbols are all argument extents, which are nonnegative under
  `SetSymbolNonnegativeAssumptions`.
- `bound_data_dependent_extents.py:301` (`access_bounds`) keeps a relation only when the symbol's coefficient is
  exactly 1. `nl[l, 0:2 * count]` gives `2 * count <= w`, which bounds `count` by `w // 2`; dropping it is sound but
  loses the bound.
