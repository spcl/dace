# Multi-Dim Vectorization — Code Audit (Keep / Drop / Refactor)

**Question**: After
[TILIFICATION_TRANSFORMATION_DESIGN.md](TILIFICATION_TRANSFORMATION_DESIGN.md)
freezes, what stays, what leaves, and what gets reworked?

**Scope**: 45 modules in `dace/transformation/passes/vectorization/`
(17,103 LoC) + 24 modules in `utils/`.

---

## Verdict in one sentence

**Keep everything that the new ``VectorizeCPUMultiDim`` orchestrator
already consumes; drop the 1D-tasklet-templated, lane-id-symbol-
expanding legacy machinery once the K=2 path reaches numerical parity
on the test corpus we care about.**

The keep / drop split runs cleanly along the *body shape* line: the
new design's body is a NestedSDFG of tile lib nodes; the legacy
design's body is a flat per-lane tasklet fan-out keyed on
`_laneid_<i>` symbols. The two cannot share more than utility code.

---

## Section A — Keep (post-freeze)

These are the multi-dim path. **Do not touch except for the design-
driven gaps (G1–G6).**

### A.1 Orchestrator + immediate pipeline

| Module                                            | LoC  | Why it stays                             |
|---------------------------------------------------|------|------------------------------------------|
| `vectorize_cpu_multi_dim.py`                      | ~700 | The orchestrator. Pipeline already named in the design. |
| `mark_tile_dims.py`                               | ~170 | Classifies inner maps + builds `TileDimSpec`. |
| `split_map_for_tile_remainder.py`                 | ~250 | K-boundary peel (§8.2 in the design).    |
| `stride_map_by_tile_widths.py`                    | ~100 | Sets the per-dim Map step to `widths[d]`.|
| `generate_tile_iteration_mask.py`                 | ~300 | Emits the `(K_0,…,K_{K-1})` mask (§7.4). |
| `tile_map_by_num_cores.py`                        | ~150 | Outer tiling for cores; orthogonal to the K-dim path. |

### A.2 Body rewriters (the descent + the inlined outer-state port)

| Module                                            | LoC   | Why it stays                                                |
|---------------------------------------------------|-------|-------------------------------------------------------------|
| `promote_nsdfg_body_to_tiles.py`                  | ~3000 | The body-NSDFG descent (the **K=1 path**). Stays as-is.     |
| `promote_inlined_map_to_tiles.py`                 | ~600  | The **K=2 outer-state port** (slices 1+2, slices 3+4 next). |
| `rewrite_array_scalar_to_tile_op.py`              | ~230  | Direct `AN↔tile` Array copies → `TileLoad` / `TileStore`.   |
| `emit_tile_ops.py`                                | ~1500 | Tasklet-body classifiers (binop / unop / merge), `_tile_region_subset`, `_mask_name_for_map`. Body-agnostic; both descent and outer-state port import from here. |
| `bypass_trivial_assign_tasklets.py`               | ~205  | `AN → [_out=_in] → AN` folding (slice extracted from descent). |
| `stage_global_array_through_scalars.py`           | ~500  | The §3 staging contract (tasklet ↔ global through per-subset scalars). |
| `widen_in_map_nsdfg_inputs.py`                    | ~400  | Restores collapsed dims so the inner subset matches `TileDimSpec`. |
| `insert_body_nsdfg_copies.py`                     | ~250  | Copy-in / copy-out states for body NSDFGs (§3 staging).     |
| `eliminate_dead_copies.py`                        | ~140  | Drops dead `AN → AN` copies inside body NSDFGs.             |

### A.3 Branch lowering (target-agnostic; both paths use it)

| Module                                            | LoC  | Why it stays                                |
|---------------------------------------------------|------|---------------------------------------------|
| `branch_normalization.py`                         | ~600 | Lifts if/elif/else into the canonical form. |
| `same_write_set_if_else_to_ite_cfg.py`            | ~500 | Per-clone unique-name minting for ITE-CFG.  |
| `lower_ite_to_fp_factor.py`                       | ~250 | Optional FP-factor lowering (knob-gated).   |
| `lower_interstate_conditional_assignments_to_tasklets.py` | ~200 | Folds conditional state-edge assigns into dataflow. |

### A.4 Cleanup / housekeeping (used by both)

| Module                                            | LoC  | Why it stays                            |
|---------------------------------------------------|------|-----------------------------------------|
| `remove_empty_states.py`                          | ~50  | Tidies the CFG after branch lowering.   |
| `remove_reduntant_assignments.py`                 | ~120 | DCE for assign-only states (vectorization-local, distinct from the live `RemoveRedundantAssignmentTasklets`). |
| `tasklet_preprocessing_passes.py`                 | ~500 | `RemoveFPTypeCasts`, `RemoveIntTypeCasts`, `PowerOperatorExpansion`, `RemoveMathCall`. The multi-dim orchestrator already calls these. |
| `resolve_other_subset_an_edges.py`                | ~150 | Reinserts `_out = _in` between AN-AN with `other_subset`. Pre-pass for both descent and outer-state. |
| `refuse_other_subset_in_nsdfg.py`                 | ~120 | Loud failure when `other_subset` survives into a body NSDFG the descent must read. |
| `nest_innermost_map_body.py`                      | ~200 | Wraps a flat-body inner Map into a NestedSDFG so the descent can target it. |
| `split_multi_slice_boundary_connectors.py`        | ~300 | Splits an AoR-style multi-slice boundary connector so each slice gets its own descriptor. |
| `fuse_overlapping_tile_loads.py`                  | ~250 | CSE on multiple `TileLoad`s reading overlapping windows of the same source. |

### A.5 Utils that the multi-dim path consumes

| Module                                  | LoC  | Why it stays                                |
|-----------------------------------------|------|---------------------------------------------|
| `utils/tile_access.py`                  | ~600 | Per-dim classifier (§4). Add MODULAR + G4 lattice join. |
| `utils/tile_access_compat.py`           | ~100 | Drives legacy API from the new per-dim classifier. **Drop once §A.7 is purged.** |
| `utils/tile_dims.py`                    | ~150 | `TileDimSpec`, `TileAccessKind`, `TileAccessClassification`. |
| `utils/promote_helpers.py`              | ~80  | Body-agnostic perfect-box classifier (slice 0).  |
| `utils/name_schemes.py`                 | ~500 | `TileLaneScheme`, `TileNameScheme`, `TileConnectors`. |
| `utils/post_descent_invariants.py`      | ~150 | `assert_post_descent_invariants` audit.     |
| `utils/iteration.py`, `subsets.py`, `code_rewrite.py`, `symbolic_polymorphism.py` | ~1000 | Pure helpers; both paths use them. |
| `utils/map_predicates.py`               | ~250 | `is_innermost_map`, scope predicates.       |
| `utils/nsdfg_reshape.py`                | ~600 | NSDFG connector reshape + copy emission.    |
| `utils/arrays.py`, `tasklets.py`        | ~350 | Generic helpers (still consumed).           |

### A.6 Library nodes (the freeze surface)

All 10 nodes under `dace/libraries/tileops/nodes/` stay. Interface
changes are confined to the design's G3 + §7 mask lock + §10.3
TileReduce lock; no node is dropped.

| Node          | LoC  | Action under freeze                                                  |
|---------------|------|----------------------------------------------------------------------|
| `tile_load.py`    | ~405 | Keep; already supports the §5.1 contract.                            |
| `tile_store.py`   | ~340 | Keep; analogous to `TileLoad`.                                       |
| `tile_gather.py`  | ~365 | **Extend** with `_idx_full` + `index_form` (G3).                     |
| `tile_scatter.py` | ~290 | **Extend** symmetrically with G3.                                    |
| `tile_binop.py`   | ~485 | Keep; operand-kind contract aligns.                                  |
| `tile_unop.py`    | ~370 | Keep.                                                                |
| `tile_merge.py`   | ~300 | Keep.                                                                |
| `tile_mask_gen.py`| ~180 | Keep; tighten `validate()` to enforce §7.1 + §10.2 mask shape.       |
| `tile_iota.py`    | ~175 | Keep; the per-dim index source for the diagonal / transpose path.    |
| `tile_reduce.py`  | ~365 | **Lock** to tile→scalar (§10.3); refuse any `axis` / `keepdims`.     |

---

## Section B — Drop (once K=2 reaches parity)

These are the **1D-tasklet-templated, lane-id-symbol legacy machinery**.
Their callers are `VectorizeCPU` / `VectorizeSVE` / `vectorize.py`
(the `Vectorize` base class). When those orchestrators retire, these
go with them.

### B.1 The `_laneid_<i>` fanout family

| Module                                            | LoC  | Why it goes                                                           |
|---------------------------------------------------|------|-----------------------------------------------------------------------|
| `detect_gather.py`                                | ~250 | 1D gather detector; replaced by `TileGather` emission from the classifier (G3). |
| `detect_scatter.py`                               | ~250 | Mirror; same fate.                                                    |
| `detect_strided_load.py`                          | ~200 | 1D strided-load detector; the per-dim `dim_strides` carries this in `TileLoad`. |
| `detect_strided_store.py`                         | ~200 | Mirror.                                                               |
| `detect_multi_dim_strided_load.py`                | ~250 | Was the first multi-dim extension of the strided detect; superseded by `TileLoad` `src_dims` + `dim_strides`. |
| `detect_multi_dim_strided_store.py`               | ~250 | Mirror.                                                               |
| `utils/lane_fanout.py`                            | ~500 | The per-lane fanout matcher feeding `detect_*`. Pure-1D.              |
| `utils/lane_expansion.py`                         | ~600 | Mints `_laneid_<i>` symbols + expands interstate assignments per lane. Pure-1D. |
| `utils/lane_access.py`                            | ~200 | 1D access classification (precursor to the per-dim classifier).       |
| `utils/multiplex.py`                              | ~250 | The `int_floor` multiplex helper — already subsumed by `REPLICATE` in `tile_access.py`. |

### B.2 The 1D orchestrators (drop with their families)

| Module                | LoC   | Why                                                                                                |
|-----------------------|-------|----------------------------------------------------------------------------------------------------|
| `vectorize.py`        | ~2270 | The 1D `Vectorize` base. Its body-rewrite path (mints `_laneid_<i>`, templates per tasklet) is what B.1 supports. |
| `vectorize_cpu.py`    | ~600  | 1D CPU orchestrator. Replaced by `VectorizeCPUMultiDim` at `widths=(W,)`.                          |
| `vectorize_sve.py`    | ~900  | 1D SVE orchestrator (runtime VL). Replaced by `VectorizeCPUMultiDim` with `full_mask` + SVE target.|
| `vectorize_gpu.py`    | ~600  | 1D GPU orchestrator. Replaced by `VectorizeCPUMultiDim` with cuTile target.                        |

### B.3 The 1D iteration mask + remainder helpers

| Module                            | LoC  | Why                                                                       |
|-----------------------------------|------|---------------------------------------------------------------------------|
| `generate_iteration_mask.py`      | ~400 | 1D `_iter_mask`; superseded by `generate_tile_iteration_mask.py`.         |
| `split_map_for_vector_remainder.py`| ~300 | 1D remainder split; superseded by `split_map_for_tile_remainder.py`.     |
| `remove_vector_maps.py`           | ~200 | 1D collapse of trivial vector maps; the K-dim path doesn't produce them.  |
| `fuse_overlapping_loads.py`       | ~400 | 1D overlapping-load fusion; superseded by `fuse_overlapping_tile_loads.py`.|
| `for_loop_to_masked_while.py`     | ~300 | 1D break-to-masked-while; not used by the K-dim path.                     |

### B.4 Cleanup that exists only to serve B.1–B.3

| Module                            | LoC  | Why                                                          |
|-----------------------------------|------|--------------------------------------------------------------|
| `clean_scalar_assign_to_map_exit.py` | ~225 | Tidies a 1D-codegen-induced shape; the K-dim staging design (§3) prevents the shape entirely. |

---

## Section C — Refactor (post-freeze, not a drop)

These need touchups in light of the design but stay.

| Module                                            | What changes                                                                                            |
|---------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `utils/tile_access.py`                            | **Add MODULAR** enum member + detector. **Rename** `BROADCAST → CONSTANT`, `STRUCTURED_1 → LINEAR` (mechanical). Wire the §4.2 loop-variant → GATHER join (G4). |
| `utils/tile_access_compat.py`                     | Update for the renamed enum members. Will become trivial once B.1 retires.                              |
| `split_map_for_tile_remainder.py`                 | Audit against §8.2 corner-absorbing peel (G5). May already be correct; needs a regression test.         |
| `generate_tile_iteration_mask.py`                 | Tighten the `bool_` / shape contract per §7.1 + §10.2.                                                  |
| `dace/libraries/tileops/nodes/tile_gather.py`     | Extend with `_idx_full` + `index_form` switch (G3). Add the mutual-exclusion validator.                 |
| `dace/libraries/tileops/nodes/tile_scatter.py`    | Same as above, symmetric.                                                                               |
| `dace/libraries/tileops/nodes/tile_mask_gen.py`   | Validate output array shape + dtype + storage per §10.2.                                                |
| `dace/libraries/tileops/nodes/tile_reduce.py`     | Refuse `axis` / `keepdims` (§10.3); document the future axis-keep path.                                  |
| `promote_inlined_map_to_tiles.py`                 | After freeze: extend `_operand_kind` with broadcast Scalar + NDTile walk-back (§G2-also).               |

---

## Section D — Test corpus impact

Rough counts (numbers approximate, from `grep`):

| Cohort                                                                | Tests | Action                                                       |
|-----------------------------------------------------------------------|-------|--------------------------------------------------------------|
| Touch legacy `VectorizeCPU` / `VectorizeSVE` only                     | ~14   | Stay until B.2 retires; then port the genuinely-1D ones to `VectorizeCPUMultiDim(widths=(W,))` and delete the 1D-only ones. |
| Touch both legacy and new                                             | ~3    | Same fate.                                                   |
| Multi-dim / tile-ops only                                             | ~12   | Stay.                                                        |
| Hand-built classifier / lattice / helper tests                        | ~28   | Stay (introduced this branch).                               |
| TSVC integration sweep (`tsvc_vectorization_test.py`)                 | many  | Stays as an end-to-end check; re-parametrise once `VectorizeCPUMultiDim` covers all the patterns the legacy CPU path does. |

---

## Section E — Drop order (mechanical)

Each step is a single commit, gated by the test sweep staying green.

1. **G2 rename** (`BROADCAST → CONSTANT`, `STRUCTURED_1 → LINEAR`,
   add `MODULAR`). Single commit, mechanical. Updates `tile_access.py`
   and the compat shim only. (Touches §A.5, §C.)
2. **G3 + G4** lib-node extensions + classifier join rule. Multiple
   focused commits. Lands the freeze contract.
3. **G1 full-subset NSDFG call convention**. New pass; existing
   widening utils stay.
4. **G5 audit** for `split_map_for_tile_remainder.py`. Test-only or
   tiny fix.
5. **Slice 3 + 4 of `promote_inlined_map_to_tiles.py`** — closes the
   K=2 outer-state path.
6. **K=2 numerical-parity gate**: run the TSVC sweep on
   `VectorizeCPUMultiDim` with both K=1 (descent) and K=2 (outer-state)
   in parallel; require both to match the legacy result.
7. **Drop B.2 1D orchestrators** (`vectorize_cpu.py`,
   `vectorize_sve.py`, `vectorize_gpu.py`, `vectorize.py`). Their
   tests get ported per Section D. Compat shim
   (`utils/tile_access_compat.py`) goes with them.
8. **Drop B.1 `_laneid_` family** (all `detect_*.py`, `lane_fanout.py`,
   `lane_expansion.py`, `lane_access.py`, `multiplex.py`).
9. **Drop B.3 1D mask / remainder helpers** (`generate_iteration_mask.py`,
   `split_map_for_vector_remainder.py`, `remove_vector_maps.py`,
   `fuse_overlapping_loads.py`, `for_loop_to_masked_while.py`).
10. **Drop B.4 cleanup** (`clean_scalar_assign_to_map_exit.py`).

Estimated net deletion after step 10: **~9,500 LoC** out of the 17,103
LoC in `dace/transformation/passes/vectorization/`, plus **~2,300 LoC**
of utils (lane family + multiplex), plus the test files that touched
the dropped orchestrators only. Roughly a **55-60% trim** with no
functionality loss for the kernels the new path covers.

---

## Section F — Don't drop yet (revisit later)

These are flagged as "drop" candidates but only after a real signal:

- **`vectorize.py`'s `Vectorize` base class** if any downstream
  consumer (`tests/transformations/`, the SVE inference plumbing)
  still imports the base class for inheritance. Check before deletion.
- **`for_loop_to_masked_while.py`** if a kernel in the velocity-
  tendencies corpus actually needs break-to-masked-while; the K-dim
  path's branch normalisation might not cover it.
- **`utils/multiplex.py`** if any cuTile fixture still consumes its
  `multiplex.h` runtime header before being ported to
  `REPLICATE`-driven `TileLoad`.

Run a grep gate before each drop. The audit above is the plan;
verifying each step's call sites is the execution.
