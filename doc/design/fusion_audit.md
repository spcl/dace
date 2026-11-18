# Fusion audit (canonicalization and specialization)

Status 2026-11-18. Static audit: nothing here was compiled or run. Line numbers refer to branch `fix-1117`.

## Where fusion runs

| Pass | Runs in | Call site |
|---|---|---|
| `FuseConsecutiveLoops` | canon, `normalize_statements` | `canonicalize/pipeline.py:816` |
| `FuseChainedScalarReductions` | canon, `derive_parallelism` | `pipeline.py:954` |
| `FuseLoops` (`LoopFusion`), GPU after `ReorderStateForLoopFusion` | canon, `derive_parallelism` and every fusion round | `pipeline.py:997-998`, `:1021` |
| `FuseElementwiseIntoGemv` | canon, `recompose` | `pipeline.py:1089` |
| `FuseStates` (`StateFusionExtended`, `BlockFusion`) | canon structural cleanup, fusion rounds; finalize | `pipeline.py:310`, `:1019`; `finalize.py:879`, `:1027`, `:1099` |
| `FuseMaps` (`MapFusionVertical`, `MapFusionHorizontal`, `MapPartialReduceFusion`) | canon fusion rounds; finalize DOT gather; offload | `pipeline.py:1027`; `finalize.py:754`; `offloading/offload_to_accelerator.py:670` |
| `FuseMapsAtSharedAxes` | canon fusion rounds | `pipeline.py:1033` |
| `MapReduceFusion`, `MapWCRFusion` | canon fusion rounds | `pipeline.py:1036` |
| `MergeGuards`, `SelectBranchMaps`, `LiftTrivialIf` | canon outer fusion round | `pipeline.py:1048-1051` |
| `maximal_fusion` again, after `MoveMapInvariantIfUp` | CPU and GPU specialization (`hoist_invariant_guards`) | `interstate/move_map_invariant_if_up.py:950`; `cpu_specialization/pipeline.py:103`; `gpu_specialization/gpu_specialization_pipeline.py:36` |
| `RecomputeIntermediates` (`OTFMapFusion`) | CPU specialization | `cpu_specialization/pipeline.py:107` |
| `FuseScanChains` | CPU specialization | `cpu_specialization/pipeline.py:108` |
| `FuseCopyChains`, `recompute_fuse_for_gpu` (`OTFMapFusion`) | GPU offload | `finalize.py:850`, `:856` (`:783`) |

## Refusals

"Required" means a fusion past the check would compute wrong values. "Conservative" means it would be correct in some
of the refused cases.

| Pass | Refusal | Where | Required? | Status |
|---|---|---|---|---|
| MapFusionVertical | different schedules | `dataflow/map_fusion_helper.py:1242` | no, but schedules are unset in canon | by design |
| MapFusionVertical | ranges not structurally equal (no offset, no symbol relation) | `map_fusion_helper.py:92`, `:112`; prefilter `passes/fuse_maps.py:95` | conservative | by design: origins are normalized upstream (`NormalizeLoopAndMapOrigin`), steps by `align_maps_to_unit_step`. Symbols compare by name (`symbolic.py:390`), so a dtype or assumption difference no longer splits a range |
| MapFusionVertical | maps in different scopes | `map_fusion_helper.py:1256` | yes for this rewrite | by design |
| MapFusionVertical | dynamic map ranges bind different values | `map_fusion_helper.py:1266` | yes | by design |
| MapFusionVertical | conflicting accesses inside the bodies / through the scope nodes / a read moved past a write | `dataflow/map_fusion_vertical.py:454`, `:469`, `:480` | yes | by design |
| MapFusionVertical | intermediate is a View, or neither Array nor Scalar | `map_fusion_vertical.py:971`, `:976` | conservative (reshapes) | by design; `RemoveViews` runs before |
| MapFusionVertical | reduced intermediate of symbolic size | `map_fusion_vertical.py:1019`, `:1066` | no | canon sets `allow_intermediates_with_symbolic_size=True` |
| MapFusionVertical | WCR producer edge | `map_fusion_vertical.py:1040` | yes for a partial result | by design; `MapWCRFusion`, `MapPartialReduceFusion` take the reduction cases |
| MapFusionVertical | conditional (dynamic) producer, unless reader runs under an equal guard (`guarded_alike`, `:2899`) | `map_fusion_vertical.py:1045` | no when fused in place (`in_place_intermediates`, `:2343`: a skipped element keeps its value) | by design: a performance guard (commit `6003805d8`, warpx 37 -> 1603 nested SDFGs) |
| MapFusionVertical | two producers of the intermediate may overlap | `map_fusion_vertical.py:1108` | yes | by design |
| MapFusionVertical | consumer read not covered by exactly one producer of the same iteration: gathers `t[idx[i]]` of the intermediate, data-dependent subsets, reads of another iteration | `map_fusion_vertical.py:1178` | yes (in place: unless no other iteration writes it) | by design |
| MapFusionVertical | intermediate read again downstream (strict dataflow, reduced) | `map_fusion_vertical.py:1236` | no | canon runs `relaxed_dataflow=True` |
| MapFusionVertical | nested SDFG reads the intermediate through an InOut connector, or its strides cannot be rewritten | `map_fusion_vertical.py:2781`, `:2746` | conservative | by design |
| OTFMapFusion | producer with more than one output | `dataflow/otf_map_fusion.py:69` | conservative (multiple outputs) | by design |
| OTFMapFusion | intermediate written by a second node / producer not its only writer | `otf_map_fusion.py:71`, `:116` | yes (masked writes after the producer) | by design |
| OTFMapFusion | dynamic producer write | `otf_map_fusion.py:160` | yes | by design |
| OTFMapFusion | producer parameter not solved from the consumer read (whole-range nested SDFG writes) | `otf_map_fusion.py:240` | yes | by design |
| FuseMapsAtSharedAxes | only top-level maps and transient intermediates | `canonicalize/fuse_maps_at_shared_axes.py:91`, `:97` | conservative (nested scopes) | by design |
| FuseMapsAtSharedAxes | shared axis picked by range alone: with two producer axes over one range the first one was tried, and a refused split was never retried with the other | `fuse_maps_at_shared_axes.py:23` | no | **fixed** (axis read off the intermediate's index, below) |
| FuseMapsAtSharedAxes | consumer split only when the shared axes are its outermost | `fuse_maps_at_shared_axes.py:111` | no (loop order only) | **proposed** (warpx, below) |
| StateFusionExtended | side-effect nodes, scoped writes the other state touches, reused transients with several producers | `interstate/state_fusion_with_happens_before.py:408`, `:460-487`, `:498` | yes | by design |
| StateFusion / maps across control flow | states in different `ConditionalBlock` arms or on either side of one never share a state | structural | yes without duplication | by design: fusing across a guard means pushing the guard in (`MoveIfIntoLoop`, `MoveIfIntoMap`) or duplicating the other side |
| LoopFusion | edge between the loops carries assignments or a condition; different iteration spaces; independently DOALL loop | `interstate/loop_fusion.py:104-116`, `:206` | yes / conservative | by design |
| MoveMapInvariantIfUp (spec.) | unswitching `map {B; if c: A}` copies `B` into both arms | `interstate/move_map_invariant_if_up.py` docstring | n/a | note: code duplication in specialization, not canon |

## Change made

`FuseMapsAtSharedAxes` matched each consumer axis to the first producer axis with an equal range. When two producer
axes share that range (`T[i, k]` over `0:8, 0:8`, read as `T[3, j]`), it split at `i`, the vertical fusion refused
(the iteration `i = j` writes row `j`, not column `j`), and the merge-back left the pair unfused. Now the pass reads
the intermediate's index on both sides first: a dimension that the producer's writes index by one parameter alone and
the consumer's reads index by one parameter alone pairs those two parameters (`index_pairs`, `bare_indices`). The
range rule fills in the axes without such a pair. Legality stays with `MapFusionVertical.can_be_applied`, so a wrong
pairing can only cost a fusion. A pair of maps with an equal number of axes is skipped before the walk.
Test: `tests/passes/canonicalize/fuse_maps_at_shared_axes_test.py`
(`test_the_split_axis_is_the_one_the_consumer_indexes_when_two_share_its_range`). Not run.

## warpx_field_gather trace

Kernel: `hpcagent_bench/benchmarks/scientific_computing/n_body_methods/field_gather/warpx_field_gather_dace.py`
(generated from the numpy file). `geom = 3` and `galerkin_interpolation = 1` are module constants; `depos_order` is a
runtime symbol. Numba (`warpx_field_gather_numba.py`) runs one `prange` over particles with the order branch inside
the loop body, all shape factors in per-particle registers.

Evidence: no current CPU canonical build is on disk. The newest canonical graph is the GPU prerender of 2026-11-10
(`.hpcagentbench-cache/.cpf-prerender/cache/59/5919c9e4.../canonical.sdfgz`). It predates `3630e92d7` (guarded producer
fusion, 11-12), so it shows the structure but not the current fusion verdicts. The plain DaCe CPU build in
`.hpcagentbench-cache/dace_numeric/warpx_field_gather_rank0/` is not canonicalized.

What the graph holds, per axis (x, y, z) and per field type selector:

1. `compute_shape_factor_into` writes `sx_node[k, :]` (literal rows `k`), `idx[:]`, in five arms guarded by
   `depos_order == 0..4`, nested in the `ey_type[0] == NODE` elif chain. Inside one arm the per-particle maps fuse (the
   prerender shows one map per arm with `__map_fusion_*` scalars). Across arms nothing fuses: each arm writes the rows
   its order has, and the consumer reads `sx[0:depos_order + 1, i]`. That the five arms together cover those rows
   holds only per arm (`order == depos_order` there). Proving coverage needs path-sensitive facts, and fusing into the
   arms duplicates the gather five times, which is the cost trade canon does not make. **By design.** This leaves 12
   shape-factor planes `(o + 1, n)` and 12 index vectors `(n,)` per call materialized, written in the shape phase and
   read in the gather phase. Numba keeps all of them in registers.
2. `j_node[:] = idx` copies the arm result out of the type guard. That is a transient-forwarding case
   (`RedundantArray` / `EliminateCopyBack`), owned by the transient-removal audit, not fusion.
3. `_select` is `np.where(type[0] == NODE, node, cell)` over `(k, i)`, a select on a data scalar. The int casts
   (`.astype(np.int64)`) are tasklet casts. Neither blocks fusion. `_copy_sel` is dead in the kernel (never called).
4. `_tap3`: `ix/iy/iz` index planes, `gathered`, `weight` iterate `(tx, ty, tz, i)` with the particle axis innermost
   (numpy broadcasting). They fuse among themselves (the prerender holds `__map_fusion_*_ix_b` as `(1,1,1,1)`
   registers). The per-particle producers feeding them, `lox + j_ex` over `(i,)` and `sx_ex` over `(k, i)`, have
   fewer axes. `FuseMapsAtSharedAxes` would split the consumer at `i`, but refuses (`:111`) because `i` is not the
   consumer's outermost axis. On CPU, `RecomputeIntermediates` (OTF) can still recompute both inside the gather (one
   reader, single writer, point reads). It cannot do that for the shape-factor planes of item 1, which have several
   guarded writers (`otf_map_fusion.py:116`).
5. The `sz_*` rows are read at `sz[tz, i]`, where `ty` and `tz` both range over `0:depos_order + 1`. Matching by
   range alone paired `sz`'s row axis with `ty`. That is fixed above, but item 4's outermost rule still refuses
   these.

Conclusion: the 1.84x is dominated by item 1, by design. The cheaper structural blocker is item 4, and it is a loop
order choice.

## Proposals (not implemented)

- **Split a consumer at inner shared axes (`fuse_maps_at_shared_axes.py:111`).** Allow the split-consumer case when
  the shared axes are not the consumer's outermost: reorder the consumer to shared-axes-first, as the producer split
  already does unconditionally (`try_fuse`, `reorder(split, axes + rest)`). This makes `_tap*` one particle map with
  the taps inside, the numba shape. The cost: a consumer access that is indexed by the remaining axes and stays
  materialized gets an outer stride. That makes it a loop-order decision, so it needs a user ruling. The check was
  added by an agent (commit `aa3c92afb`, "so its loop order stays"), not by a recorded user decision. Expected warpx
  effect: removes `lox + j_*` and the `sx_*` selections as per-particle planes in the canonical form. On CPU, OTF
  already removes them, so the CPU gain is small. The GPU gain is larger.
- **Dynamic producer fused in place (`map_fusion_vertical.py:1045`).** Correct as is. Leave it, because the refusal
  guards the 1603-nested-SDFG regression.
