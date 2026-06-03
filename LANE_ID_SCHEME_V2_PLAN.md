# Work plan v2 — unified `LaneIdScheme` + cleanup-pipeline parity + TileReduce coverage

**Branch**: `multi-dim-tileops`. **Precondition**: post-merge sweep
(`bxnte16wh`, runs after `origin/yakup/dev` was merged with `-X theirs`)
clears with ≤ today's failure count.

---

## Goals (in priority order)

1. **Cleanup-pipeline parity** between the legacy 1D `VectorizeCPU` and
   the K-dim `VectorizeCPUMultiDim`. The latter currently skips
   `InsertAssignTaskletsAtMapBoundary` and uses a different
   redundant-assignment cleanup; both deviations come back to bite the
   descent (gather composition gap; ``other_subset`` interpretation).
2. **Single unified `LaneIdScheme`** using **Option B**:
   `<base>_lane<d>id_<n>` — one chunk per fanned tile dim. Replaces the
   legacy K=1 `_laneid_<n>` AND the unused K≥2 `_tilelane_<i0>x<i1>…`.
   Per-dim chunks make partial-gather / partial-scatter detection a
   one-pass regex walk.
3. **TileReduce e2e coverage**. ``TileReduce`` (full + axis reduce, with
   and without mask) is emitted today by ``EmitTileOps._emit_tile_reduce``
   but has zero unit tests pinning numerical correctness against a numpy
   reference.

## Non-goals (this slice)

- Tile-node label naming (`binop_add`, `gather`, `store_w_broadcast_(0,0)`)
  — separate cosmetic improvement, tracked.
- The 12 pre-merge K=0/W=1 failures — closed by `0b3eceb33`.
- Codegen changes to stop collapsing `Register Array((1,))` to plain ``T``
  — closed by VLEN=1 runtime overloads + TileIota W=1 emit (`c8e45b655`).

---

## Step B — full cleanup-pipeline parity (1 commit)

Today's divergence (one file, four places):

| Pass | `VectorizeCPU` | `VectorizeCPUMultiDim` |
|---|---|---|
| `CleanAccessNodeToScalarSliceToTaskletPattern` | front of pipeline | missing |
| `RemoveRedundantAssignments` (live) vs `RemoveRedundantAssignmentTasklets` (alternate) | live | alternate |
| `_LoopToMapPass(permissive=…)` | inline in `passes` | runs from `apply_pass` instead |
| `InsertAssignTaskletsAtMapBoundary` | active | intentionally skipped |

### Edits (all in `dace/transformation/passes/vectorization/vectorize_cpu_multi_dim.py`)

| # | Action |
|---|---|
| B1 | Prepend `CleanAccessNodeToScalarSliceToTaskletPattern()` to the `passes` list. |
| B2 | Replace `RemoveRedundantAssignmentTasklets()` with `RemoveRedundantAssignments()`. If the `depends_on EliminateBranches` reapply issue (cited in the comment) resurfaces, fix it where it lives in the pipeline-reapply logic — do NOT pick a different pass to dodge. |
| B3 | Add `_LoopToMapPass(permissive=loop_to_map_permissive)` inline in the `passes` list at the legacy position (after `RemoveEmptyStates`). |
| B4 | Add `InsertAssignTaskletsAtMapBoundary()` after `SplitTasklets`, matching legacy's position. |
| B5 | Delete the inline comment "intentionally NOT run" justifying skipping `InsertAssignTaskletsAtMapBoundary`. |

### Verification gate B

- Run `tests/passes/vectorization/orchestrator/`, `unit/`, `analysis/`,
  `kernels/`, `cloudsc/`, `test_tsvc_hardening.py`, `test_k0_remainder.py`,
  `test_branches.py` under `-n 4`.
- The `vectorize_config` conftest fixture parametrises every cloudsc /
  tsvc-hardening test over `["tile_nodes", "legacy_cpu"]`, so D2-L
  (legacy arm) and D2-K (tile_nodes arm) run automatically.
- Expected: 5 K=2 gather composition xfails close on their own once
  `other_subset` gets normalised away (the descent's `_collapse_tile_gathers`
  already handles tasklet-input edges with gather subsets — which is what
  `InsertAssignTaskletsAtMapBoundary` produces).

---

## Step A — unified `LaneIdScheme` (2 commits)

### A.1 — helper rewrite + back-compat (1 commit)

In `dace/transformation/passes/vectorization/utils/name_schemes.py`:

| API | Signature |
|---|---|
| `LaneIdScheme.make(base, dim, lane) -> str` | `<base>_lane<dim>id_<lane>` |
| `LaneIdScheme.make_multi(base, [(dim, lane), …]) -> str` | concatenated chunks |
| `LaneIdScheme.parse_chunks(name) -> List[(dim, lane)]` | regex `_lane(\d+)id_(\d+)` peel-and-repeat |
| `LaneIdScheme.peel_dim(name, d) -> name` | remove the rightmost chunk for dim `d` |
| `LaneIdScheme.base_of(name) -> base` | strip all chunks |
| `LaneIdScheme.varies_with_dim(name, d) -> bool` | True iff `name` has a `_lane<d>id_*` chunk |
| `LaneIdScheme.is_lane_fanned(name) -> bool` | True iff at least one chunk matches |

**Backward compat**: `parse_chunks` ALSO recognises the legacy
`_laneid_<n>` form as a single chunk with `dim=0`. Released for one
release; the new emitter writes only the new form.

**Delete** `TileLaneScheme` — it was unused; one class now owns the
scheme.

`assert_no_laneid_in_tile_path` collapses to a single matcher:
`LaneIdScheme.is_lane_fanned(name)`.

### A.2 — emitter + descent classifier switch (1 commit)

| Caller | File | Change |
|---|---|---|
| K=1 fan-out | `utils/lane_fanout.py::fan_out_tile_gather_index_symbols` | emit `_lane0id_<n>` (was `_laneid_<n>`) |
| K≥2 fan-out | `utils/lane_fanout.py::fan_out_tile_gather_index_symbols_kd` | emit multi-chunk `_lane<d>id_<n>` per fanned dim |
| Fan-out recognisers | `utils/lane_fanout.py::_recognize_laneid_index_slice`, `_drop_collapsed_laneid_syms` | use `LaneIdScheme.parse_chunks` |
| Legacy 1D symbol-fan | `utils/lane_expansion.py` | use `LaneIdScheme.make(base, 0, lane)` for K=1 emission |
| Legacy 1D gather detect | `detect_gather.py::collapse_laneid_index_loads` knob path | use `LaneIdScheme.parse_chunks` |
| Legacy 1D scatter detect | `detect_scatter.py` | same |
| K-dim descent | `promote_nsdfg_body_to_tiles.py::_fanned_symbols`, `_gather_index_symbols`, `_collapse_tile_gathers` | partial-gather detection via `LaneIdScheme.varies_with_dim(sym, d)` per subset dim |
| K-dim classifier | `utils/tile_dims.py::classify_tile_access` | for dims whose subset symbols vary with a tile lane (per `varies_with_dim`), return partial-GATHER instead of UNRECOGNIZED |

### Verification gate A

- A.1 (helper-only) → run D2-K + D2-L. Behavioural no-op; legacy
  recognises both forms via the back-compat parser; gates verify
  nothing regressed.
- A.2 (emitter switch) → **load-bearing for legacy**. Both arms
  (`tile_nodes` and `legacy_cpu`) must stay green on cloudsc +
  tsvc-hardening + test_k0_remainder + test_branches.
- Drop the 5 K=2 gather composition `xfail(strict=True)` markers in
  `tests/passes/vectorization/unit/test_kdim_broadcasts.py` and
  `test_icon_zekinh_gather.py`.

---

## Step C — TileReduce e2e coverage (1 commit)

### New file: `tests/passes/vectorization/unit/test_tile_reduce.py`

One vectorization variant per shape (the `EmitTileOps._emit_tile_reduce`
path). Compile + run + `numpy.allclose` against numpy reduce.

| # | Kernel | K | widths | axis | mask | op |
|---|---|---|---|---|---|---|
| C1 | `acc = sum(tile)` over a 1-D body tile | 1 | (8,) | `None` | no | `+` |
| C2 | `acc = prod(tile)` over a 1-D body tile | 1 | (8,) | `None` | no | `*` |
| C3 | `acc = min(tile)` | 1 | (8,) | `None` | no | `min` |
| C4 | `acc = max(tile)` | 1 | (8,) | `None` | no | `max` |
| C5 | `acc = sum(tile)` over a 2-D body tile (full reduce) | 2 | (8,8) | `None` | no | `+` |
| C6 | axis-0 row-sum over 2-D body tile → (8,) tile | 2 | (8,8) | `0` | no | `+` |
| C7 | axis-1 col-sum over 2-D body tile → (8,) tile | 2 | (8,8) | `1` | no | `+` |
| C8 | Masked C1 — inactive lanes contribute identity | 1 | (8,) | `None` | yes | `+` |
| C9 | Masked C5 | 2 | (8,8) | `None` | yes | `+` |

### Shape of each test

```python
sdfg = build_kernel(...)           # dace.program with the reduction
sdfg.simplify()
# Apply ONE vectorization variant — the orchestrator goes through
# _emit_tile_reduce because the body has a WCR-write-to-scalar pattern.
VectorizeCPUMultiDim(widths=W, ..., expand_tile_nodes=False).apply_pass(sdfg, {})

# Compile + run twice with the same fixed-seed input.
a = numpy.random.default_rng(42).uniform(...)
expected = numpy.<op>.reduce(a, axis=...)        # numpy reference
sdfg(a=a, out=actual, ...)
assert numpy.allclose(actual, expected, rtol=1e-12)
```

### Verification gate C

- All 9 tests pass under D2-K. Legacy arm is not exercised here
  (TileReduce is K-dim only); the legacy 1D path has its own existing
  reduction lowering that is tested elsewhere.

---

## D — verification matrix + cleanup

| Gate | When | Runs |
|---|---|---|
| D1 | precondition | wait for `bxnte16wh` to clear with ≤ 12 failures (the K=0/W=1 pre-existing, since closed; any other failure aborts) |
| D2-K | after each commit | K-dim arm — `orchestrator/`, `unit/`, `analysis/`, `kernels/test_strided_gather_scatter`, `kernels/test_gather_scatter_knob`, `kernels/test_inter_lane_stride`, `kernels/test_basic_ops` |
| D2-L | after each commit | **legacy arm** — `cloudsc/test_cloudsc.py`, `test_tsvc_hardening.py`, `test_k0_remainder.py`, `kernels/test_branches.py` (the `vectorize_config` fixture runs both `tile_nodes` and `legacy_cpu` automatically) |
| D2-F | final pre-cleanup | full `tests/passes/vectorization/` sweep, must be 0F or strictly ≤ the prior failure count |
| D3 | after Step A.2 | drop the 5 K=2 gather composition `xfail(strict=True)` markers |
| D4 | after D2-F is green | clean `/home/primrose/Work/dace/.dacecache/`, `_dacegraphs/`, `.pytest_cache/` (and any nested `pytest_cache/`) |

---

## Sequencing (locked)

1. **Wait for `bxnte16wh`** (D1) — confirms the `-X theirs` merge is stable on both arms.
2. **Commit 1 — Step B.** Full cleanup-pipeline parity in
   `vectorize_cpu_multi_dim.py`. Run D2-K + D2-L. **Expected**: 5 K=2
   gather composition xfails XPASS automatically (the descent's
   `_collapse_tile_gathers` now sees the normalised tasklet-input gather
   edges); drop markers in the same commit.
3. **Commit 2 — Step A.1.** `LaneIdScheme` rewritten in-place for
   Option B (`_lane<d>id_<n>`) + legacy `_laneid_<n>` parser
   back-compat for one release. Delete `TileLaneScheme`. Audit
   collapsed to a single matcher. Run D2-K + D2-L.
4. **Commit 3 — Step A.2.** Switch every emitter + descent classifier
   to the new form. Load-bearing for legacy. Run D2-K + D2-L (heavy
   coverage on cloudsc + tsvc-hardening on BOTH arms).
5. **Commit 4 — Step C.** `test_tile_reduce.py` (9 tests). Run D2-K.
6. **D2-F** — full sweep. Confirm 0F.
7. **D4** — cleanup artefacts.

Estimated: 4 commits, ~6 sweeps (each ~10-12 min on `-n 4`).

---

## Open risks / decisions

1. **B2 reapply issue.** The K-dim pipeline picks `RemoveRedundantAssignmentTasklets` over `RemoveRedundantAssignments` to avoid a "depends_on EliminateBranches" pipeline reapply trip. If that issue reappears under the unified pass set, fix it where it lives (in the pipeline-reapply machinery) and DO NOT pick the alternate pass — would re-introduce the divergence we're closing.
2. **Legacy 1D may have hidden `_laneid_<n>` string-literal users** beyond `LaneIdScheme.parse`. Grep for the bare literal `_laneid_` before A.2 and migrate any such site to `LaneIdScheme.parse_chunks` / `is_lane_fanned`.
3. **Audit pass at session boundaries.** `assert_no_laneid_in_tile_path` is invoked at the end of `VectorizeCPUMultiDim.apply_pass`; with the unified scheme, the "lane-id should not leak" assertion still applies — just under the new regex.
4. **Step C uses a single vectorization variant.** That's deliberate per the principle that the lib node's contract is invariant across vectorization knobs. If we want to be paranoid, expand to all `remainder_strategy` values — but the cost grows quickly and the additional confidence is small.

---

## Tracked separately (NOT in this plan)

- Tile-node label naming scheme (`binop_add`, `unop_not`, `gather`, `store_w_broadcast_(0,0)`) — separate cosmetic improvement, not covered here.
- Gather-composition for K=2 multi-source patterns (zekin's 3-edge sum) — once Step B closes the simple gather xfails, the multi-source patterns will need their own follow-up if they don't fall out for free.
