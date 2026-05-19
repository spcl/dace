# Core transformation / utility bug-fixes (for the consolidated bugfixes PR)

Tracks every fix to a **core** DaCe transformation or utility (not
canonicalize-only glue) made while building the canonicalization pipeline.
Each entry: the bug, the file/locus, the fix, and the reproducer unit test.
The consolidated PR = these fixes, each with its reproducer test, separated
from the canonicalize-pipeline work.

**Reproducer constraint (binding):** every reproducer test must use only
APIs available on `main` — it must NOT import or depend on the
canonicalization pass or any not-yet-upstream pipeline code. Build the SDFG
with core APIs (SDFG/state construction, `LoopToMap`, `MapToForLoop`,
`propagate_memlets_*`, `add_nested_sdfg`, …) so each test is mergeable to
`main` on its own.

## Fixed

### 1. `LoopToMap` could not prove iteration-independence through a `NestedSDFG` body
- **File:** `dace/transformation/interstate/loop_to_map.py`
- **Commits:** `f38df48d3`, `49d0b11ba` (pushed `yakup/dev`)
- **Bug:** a loop-invariant-guarded loop body becomes a `NestedSDFG`; memlet
  propagation (correctly) widens its external write connector to the
  whole-array loop-union `b[0:N]`. `can_be_applied`'s write-pattern check
  only inspected that external memlet, so it could not prove each iteration
  writes a distinct location and refused. The `LoopToMap -> MapToForLoop ->
  LoopToMap` round-trip then never recovered the map.
- **Fix:** `_nested_writes_iter_indexed` — when the external check fails and
  the writer is a `NestedSDFG`, look at the inner per-iteration writes,
  remap through the node's `symbol_mapping` into the outer iteration symbol,
  and apply the same `a*i+b` check (recursive, conservative).
- **Reproducer tests:** `tests/transformations/interstate/loop_to_map_test.py`
  — `test_loop_to_map_with_loop_invariant_if` (capability, value-preserving),
  `test_loop_to_map_round_trip_through_nested_sdfg_recovers_map` (round-trip
  recovery).

### 2. `MoveIfIntoLoop._move` left stale `NestedSDFG` parent references
- **File:** `dace/transformation/passes/move_if_into_loop.py`
- **Commit:** `e74237dda` (pushed `yakup/dev`)
- **Bug:** `_move` deepcopy + re-adds blocks; a nested SDFG carried along
  kept stale `parent`/`parent_nsdfg_node`, failing validation
  (`Parent SDFG not properly set for nested SDFG node`) once a body with a
  NestedSDFG was routed through it.
- **Fix:** call `set_nested_sdfg_parent_references(sdfg)` after moves.
- **Reproducer test:** covered via `tests/transformations/move_if_into_loop_test.py`
  + `tests/canonicalize/canonicalize_perf_loop_nesting_test.py` (guarded
  imperfect nest now validates). TODO: add a focused unit test.

### 3. `unsqueeze_memlet` `preserve_minima` collapsed a translated access to the external origin
- **File:** `dace/transformation/helpers.py` (`unsqueeze_memlet`, `preserve_minima` branch)
- **Bug:** `preserve_minima` set each result dim to
  `(external_min, internal_end, step)` — it forced the lower bound to the
  **external array origin** while keeping the internal access' upper bound,
  dropping the internal access' own offset. A per-iteration point
  `a[i, :]` propagated out through a whole-array external memlet
  `a[0:N, :]` became the prefix-growing `a[0:i+1, :]` (uses the scope
  iterator in a bound, over-approximates a point to a triangular slab) →
  the enclosing map became non-parallelizable and the SDFG invalid
  (`'NestedSDFG' object has no attribute 'data'`, validation.py:471, via
  `loop_to_map.py`). `preserve_minima=True` is used **only** by
  `propagate_memlets_nested_sdfg` (in/out NSDFG connector propagation), so
  the blast radius is nested-SDFG memlet propagation.
- **Fix attempts (BOTH REVERTED — regress core consumers):**
  (a) restore the translated lower bound `external_min + internal_min`
  preserving extent; (b) skip point dims (`if (rb - re) == 0: continue`).
  Both fix the prefix-growth but **regress `map_fission`**
  (`test_single_data_multiple_connectors`, `test_dependent_symbol` →
  *Memlet subset out-of-bounds*, `test_mapfission_splits_semi_indirect`):
  `preserve_minima`'s "force begin to external origin" is *context-correct*
  for `map_fission`'s inner-local (0-based) NSDFG accesses, where the
  internal point genuinely must be re-anchored to the external offset.
- **Conclusion:** the bug is real but **not safely fixable inside
  `unsqueeze_memlet`** — `preserve_minima` is context-dependent
  (inner-local vs already-outer-coordinate accesses) and other core
  consumers rely on the current behaviour. The fix belongs at the *caller*
  (`propagate_memlets_nested_sdfg` deciding when the internal subset is
  already in outer coords) or structurally (don't form the
  NSDFG-bodied-guard shape on re-canonicalize). Core utility left
  UNTOUCHED (never-regress rule).
- **Reproducer (canonicalize-free):** drafted but **not landed** (it would
  assert the reverted behaviour). Re-add only with a non-regressing fix.
- **Status:** REVERTED. Real bug, wrong locus; reclassify under Open #4 /
  the structural fix. Single-run `canonicalize` is unaffected.

## Open (separate issue, root-caused)

### 4. Canonicalize structural non-idempotence on the guarded imperfect nest
- **Symptom:** with fix #3 applied, `canonicalize` on the guarded
  imperfect nest is valid at 1x and 2x but the ConditionalBlock count
  drifts `1x->(1 CB) 2x->(2 CB)` and 3x crashes (same
  `'NestedSDFG' has no attribute data`). The guard is **duplicated** on
  re-canonicalization rather than recognised as already-canonical.
- **Not** the `unsqueeze_memlet` bug (that is fixed); a pipeline-level
  guard-duplication / fixpoint issue. Single-run canonicalize is correct.
- **Status:** root-caused to guard duplication on re-entry; fix pending,
  tracked in the memory checkpoint. Lower priority (single-run is correct).

### 6. `MoveLoopInvariantIfUp` is broken (dead code) — redesign in progress
- **File:** `dace/transformation/interstate/move_loop_invariant_if_up.py`
- **Bugs:** `expressions()` returns `node_path_graph(cls.loop)` but no
  `loop` PatternNode exists (only `map_state`/`map_entry`/`if_block`) →
  `AttributeError` on any `apply_transformations*` (transformation is
  unusable). `can_be_applied` is unconditional `return True`. `apply`
  references `self.if_block`/`self.map_state`/`self.map_entry` though
  `expressions()` yields three *separate* single-node alternatives so only
  one binds → `KeyError`. Contains `print()` debug. Not used by the active
  canonicalize pipeline.
- **Redesign (in progress):** proper single pattern, real invariance
  `can_be_applied`, hoist a loop/map-invariant guard out one scope level,
  applied to a fixpoint so an innermost guard sifts all the way up; must
  support a **mix of loops and maps**; the chain of interstate-edge symbol
  assignments the guard depends on moves up with it; emptied states
  cleaned. Linear-CFG assumption (blocks may have parents).
- **Reproducer tests (planned, main-safe):** guard on symbolic expr (no
  loop var); on data (no loop var); on data + loop var (must NOT hoist);
  interstate symbol-assignment chain hoisted with the guard; innermost
  guard sifts to top via repeated application; mixed map/loop nest.
- **Status:** redesign + tests in progress.
