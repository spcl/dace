# Multi-Dim Vectorization -- Code Audit (Keep / Drop / Refactor)

**Question**: After
[TILIFICATION_TRANSFORMATION_DESIGN.md](TILIFICATION_TRANSFORMATION_DESIGN.md)
freezes, what stays, what leaves, and what gets reworked?

**Scope**: 45 modules in `dace/transformation/passes/vectorization/`
(~17,100 LoC) + 24 modules in `utils/` (~5,000 LoC).

## Verdict

Keep what the multi-dim path needs; drop the 1D `_laneid_<i>` family
and the now-superseded inlined-outer-state slice-1/2 work. The split
runs along the body shape line: the new design body is a NestedSDFG
of tile lib nodes (multi-dim K >= 2 keeps the NSDFG -- no inlining);
the legacy 1D body is a flat per-lane tasklet fan-out.

Three drop tiers:

- **Tier S (immediate, this branch)**: code superseded by the design
  polish (inlined-outer-state slice 1/2, helpers that won't be reused).
- **Tier B (after K=2 parity gate)**: the 1D `_laneid_<i>` family + 1D
  orchestrators.
- **Tier C (after kernel signal)**: items in section F that need a
  kernel to decide.

## Section A -- Keep

The multi-dim path. Touch only for the design gaps G1..G7.

### A.1 Orchestrator + immediate pipeline

| Module | LoC | Why it stays |
|---|---|---|
| `vectorize_cpu_multi_dim.py` | ~700 | Orchestrator. |
| `mark_tile_dims.py` | ~170 | `TileDimSpec` builder. |
| `split_map_for_tile_remainder.py` | ~250 | K-boundary peel (section 8.2). G5 audit. |
| `stride_map_by_tile_widths.py` | ~100 | Per-dim Map step setter. |
| `generate_tile_iteration_mask.py` | ~300 | ANY-OOB conjunction mask (section 7.4). |
| `tile_map_by_num_cores.py` | ~150 | Outer-tiling for cores. |

### A.2 Body lowering (descent is canonical for K >= 2)

| Module | LoC | Why it stays |
|---|---|---|
| `promote_nsdfg_body_to_tiles.py` | ~3000 | **Canonical body lowering for K=1 and K>=2.** |
| `emit_tile_ops.py` | ~1500 | Tasklet-body classifiers + `_tile_region_subset` + `_mask_name_for_map`. |
| `bypass_trivial_assign_tasklets.py` | ~205 | Trivial assign folding. |
| `stage_global_array_through_scalars.py` | ~500 | Staging (mandatory for K>=2). G7 extends. |
| `widen_in_map_nsdfg_inputs.py` | ~400 | Restores collapsed dims for the descent. |
| `insert_body_nsdfg_copies.py` | ~250 | Copy-in / copy-out boundary states. |
| `eliminate_dead_copies.py` | ~140 | Dead AN -> AN copies. |

### A.3 Branch lowering (target-agnostic)

| Module | LoC | Why it stays |
|---|---|---|
| `branch_normalization.py` | ~600 | Canonical if/elif/else lifting. |
| `same_write_set_if_else_to_ite_cfg.py` | ~500 | Per-clone unique naming. |
| `lower_ite_to_fp_factor.py` | ~250 | FP-factor lowering knob. |
| `lower_interstate_conditional_assignments_to_tasklets.py` | ~200 | Folds conditional assigns. |

### A.4 Housekeeping (still consumed)

| Module | LoC | Why it stays |
|---|---|---|
| `remove_empty_states.py` | ~50 | CFG tidy. |
| `remove_reduntant_assignments.py` | ~120 | Assign-only state DCE. |
| `tasklet_preprocessing_passes.py` | ~500 | Type casts, power expansion, math-call. |
| `resolve_other_subset_an_edges.py` | ~150 | other_subset reinsertion. |
| `refuse_other_subset_in_nsdfg.py` | ~120 | Loud failure precheck. |
| `nest_innermost_map_body.py` | ~200 | Map-body -> NSDFG nesting. |
| `split_multi_slice_boundary_connectors.py` | ~300 | AoR boundary split. |
| `fuse_overlapping_tile_loads.py` | ~250 | CSE on tile loads. |

### A.5 Utils

| Module | LoC | Why it stays |
|---|---|---|
| `utils/tile_access.py` | ~600 | Per-dim classifier (section 4). Add MODULAR + G4. |
| `utils/tile_dims.py` | ~150 | `TileDimSpec`, kinds. |
| `utils/promote_helpers.py` | ~80 | Body-agnostic box classifier. |
| `utils/name_schemes.py` | ~500 | `TileLaneScheme`, `TileNameScheme`, `TileConnectors`. |
| `utils/post_descent_invariants.py` | ~150 | Audit. |
| `utils/iteration.py`, `subsets.py`, `code_rewrite.py`, `symbolic_polymorphism.py` | ~1000 | Pure helpers. |
| `utils/map_predicates.py` | ~250 | Scope predicates. |
| `utils/nsdfg_reshape.py` | ~600 | NSDFG connector reshape + copy emission. |
| `utils/arrays.py`, `tasklets.py` | ~350 | Generic helpers. |

### A.6 Library nodes (freeze surface)

All 10 nodes under `dace/libraries/tileops/nodes/` stay. Per-gap
extensions only:

| Node | Action under freeze |
|---|---|
| `tile_load.py` (~405) | Keep. Layout-validator per section 2.3. |
| `tile_store.py` (~340) | Keep. Same layout validator. |
| `tile_gather.py` (~365) | Extend with `_idx_full` + `index_form` (G3). Mutual-exclusion validator. |
| `tile_scatter.py` (~290) | Same as gather. |
| `tile_binop.py` (~485) | Keep. |
| `tile_unop.py` (~370) | Keep. |
| `tile_merge.py` (~300) | Keep. |
| `tile_mask_gen.py` (~180) | Tighten output validator (section 10.2). |
| `tile_iota.py` (~175) | Keep. |
| `tile_reduce.py` (~365) | Refuse `axis` / `keepdims` (section 10.3). |

## Section S -- Drop now (superseded by the design polish)

The design polish (commit `959246a41`) supersedes the inlined-outer-
state path. The slice-1 / slice-2 work was built against the previous
"K=2 inlines the body" direction; the new design mandates the body
stays a NestedSDFG for K >= 2 and the descent is canonical.

| Module / test | LoC | Reason for drop |
|---|---|---|
| `promote_inlined_map_to_tiles.py` | 415 | Superseded -- multi-dim no longer inlines (section 2.4 invariant 4). |
| `rewrite_array_scalar_to_tile_op.py` | 231 | Was the Array<->scalar copy rewriter for the inlined path. Not needed under NestedSDFG body. |
| `tests/.../test_promote_inlined_map_to_tiles.py` | 122 | Tests for superseded code. |
| `tests/.../test_promote_inlined_map_to_tiles_slice2.py` | 136 | Same. |
| `tests/.../test_rewrite_array_scalar_to_tile_op.py` | 154 | Same. |
| `PROMOTE_INLINED_MAP_TO_TILES_PLAN.md` | 149 | Slice plan for the superseded path. |

**Net drop in tier S**: ~1,200 LoC across 6 files. No production
consumer; safe to delete.

`utils/promote_helpers.py` stays -- the descent's `_box_classification`
still delegates to it (slice 0 was clean, not part of the inlined
path).

## Section B -- Drop (after K=2 parity gate)

The 1D `_laneid_<i>` family + 1D orchestrators. Same as the previous
audit version; unchanged by the polish.

### B.1 The `_laneid_<i>` fanout family

| Module | LoC |
|---|---|
| `detect_gather.py` | ~250 |
| `detect_scatter.py` | ~250 |
| `detect_strided_load.py` | ~200 |
| `detect_strided_store.py` | ~200 |
| `detect_multi_dim_strided_load.py` | ~250 |
| `detect_multi_dim_strided_store.py` | ~250 |
| `utils/lane_fanout.py` | ~500 |
| `utils/lane_expansion.py` | ~600 |
| `utils/lane_access.py` | ~200 |
| `utils/multiplex.py` | ~250 |

### B.2 1D orchestrators

| Module | LoC |
|---|---|
| `vectorize.py` | ~2270 |
| `vectorize_cpu.py` | ~600 |
| `vectorize_sve.py` | ~900 |
| `vectorize_gpu.py` | ~600 |

### B.3 1D mask + remainder helpers

| Module | LoC |
|---|---|
| `generate_iteration_mask.py` | ~400 |
| `split_map_for_vector_remainder.py` | ~300 |
| `remove_vector_maps.py` | ~200 |
| `fuse_overlapping_loads.py` | ~400 |
| `for_loop_to_masked_while.py` | ~300 |

### B.4 1D cleanup

| Module | LoC |
|---|---|
| `clean_scalar_assign_to_map_exit.py` | ~225 |

## Section C -- Refactor (post-freeze, not drop)

### C.1 Module-level extensions per design gap

| Module | What changes |
|---|---|
| `utils/tile_access.py` | G2: add MODULAR + rename. G4: tile-dep symbol classifier. G6: AFFINE-stride tile-invariance gate. |
| `utils/tile_access_compat.py` | G2 rename followup. Drops with B.2. |
| `split_map_for_tile_remainder.py` | G5 audit. May need K=3 fix. |
| `generate_tile_iteration_mask.py` | Tighten descriptor contract per section 10.2. |
| `stage_global_array_through_scalars.py` | G7: full-tile fallback + multi-dim entry hook (sections 3.3 + 12). |
| Lib nodes | See A.6. |

### C.2 Chained-if cleanup (touched in the same commits as the gap work)

Audit hits:

- `bypass_trivial_assign_tasklets.py` -- `_dedup_identity_assigns` and
  `_bypass_transient_assigns` share the same gate cascade (assign
  tasklet, in/out edge count, both endpoints AN). Extract a small
  `_assign_triple(state, tasklet) -> Optional[Tuple[edge, edge]]`
  helper used by both; collapses the 6-line cascade to one call.
- `promote_nsdfg_body_to_tiles.py::_box_classification` -- already
  delegates to `classify_box_for_widths`. Verify no residual cascade.
- `promote_nsdfg_body_to_tiles.py::_promote_binops::_operand` -- 100+
  line nested if/elif over operand shape. Reorganise as a `match` /
  small dispatch table over the operand-classifier output.
- `emit_tile_ops.py::_classify_binop_tasklet_body` and
  `_classify_unop_tasklet_body` -- mixed regex + dispatch. Pull the
  regex table into a module constant; the body becomes a 5-line
  `for pattern in PATTERNS: ...`.
- `stage_global_array_through_scalars.py` -- multi-condition refusal
  guards (transient flag, AN type, WCR, multi-element). Collapse into
  a single `_eligible(producer, consumer, A, s1, s2) -> Reason`
  enum return.
- `tile_access.py::classify_tile_access` -- per-dim classification has
  a deep if cascade. Move per-kind detectors to a registry and run a
  `for detector in DETECTORS: if (kind := detector(...)) is not None`
  loop. Makes G2 (MODULAR add) one new entry in the registry.

Land these inline with the gap work, not as separate commits.

## Section D -- Test corpus impact

| Cohort | Tests | Action |
|---|---|---|
| Legacy `VectorizeCPU` / `VectorizeSVE` only | ~14 | Hold until B.2 retires. |
| Both legacy + new | ~3 | Same. |
| Multi-dim / tile-ops only | ~12 | Stay. |
| Inlined-outer-state (slice 1 / slice 2 / rewriter) | 3 files, ~412 LoC | **Drop in tier S.** |
| Classifier / lattice / helper tests | ~28 | Stay. |
| TSVC integration sweep | many | End-to-end check; re-parametrise after VectorizeCPUMultiDim coverage. |

## Section E -- Drop order (mechanical)

Each step is a single commit, gated by the test sweep staying green.

1. **Tier S drops** (~1,200 LoC + plan doc). This branch.
2. **G2** vocab rename + add MODULAR.
3. **G7** (a) multi-dim entry hook in `StageGlobalArrayThroughScalars`,
   (b) full-tile fallback.
4. **G1** `EnforceFullSubsetNSDFGBody` pass.
5. **G3 + G4** classifier + lib-node extensions.
6. **G5 + G6** remainder regression tests + AFFINE-stride gate.
7. Validators: layout (section 2.3), mask descriptor (section 10.2),
   index dtype (section 10.4), WCR boundary (section 3.5), TileReduce
   axis-refuse (section 10.3). One commit per lib node.
8. **K=2 parity gate**: run the corpus on `VectorizeCPUMultiDim`
   widths=(W,) vs widths=(W_0, W_1); require numerical parity.
9. **Tier B drops** in order: B.2 1D orchestrators -> B.1 `_laneid_<i>`
   family -> B.3 1D mask/remainder -> B.4 cleanup.

Estimated final trim:

- Tier S now: ~1,200 LoC.
- Tier B post-parity: ~9,500 LoC of passes + ~2,300 LoC of utils.
- Final: ~13,000 LoC removed, roughly 60% of the vectorization tree.

## Section G -- Knobs that became no-ops post-design-polish

- `VectorizeCPUMultiDim(insert_copies=...)` -- the multi-dim pipeline
  inserts every copy intrinsically (inside-body staging of section
  3.1: each non-transient access stages through a fresh scalar / tile
  transient, with the boundary AN -> staged AN being a direct
  AccessNode -> AccessNode edge per section 3.6). The knob has no
  effect for K >= 2; kept on the constructor for harness parity with
  the 1D path until tier B retires it.

## Section F -- Don't drop yet (revisit later)

- `vectorize.py` `Vectorize` base class -- check inheritance use
  before deletion.
- `for_loop_to_masked_while.py` -- needed only if a kernel in the
  velocity-tendencies corpus has break-out-of-loop semantics.
- `utils/multiplex.py` -- needed only if a cuTile fixture still uses
  its runtime header before being ported to REPLICATE.

Grep gate per drop. The audit above is the plan; verifying each call
site is the execution.
