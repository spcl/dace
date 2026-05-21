# Core transformation / utility bug-fixes (for the consolidated bugfixes PR)

Tracks every fix to a **core** DaCe transformation or utility (not
canonicalize-only glue) made while building the canonicalization pipeline.
Each entry: the bug, the file/locus, the fix, and the reproducer unit test.
The consolidated PR = these fixes, each with its reproducer test, separated
from the canonicalize-pipeline work.

## Scope — what is actually a fix to `main` code

Only fixes to files/passes that **exist on `origin/main`** belong in the
consolidated main PR. As of now that set is:

- **Existing-transformation/pass fixes:** `loop_to_map.py` (#1),
  `map_fission.py` (#7), `trivial_tasklet_elimination.py` (#11),
  `transformation/passes/symbol_propagation.py` (#12).
- **Core-file fixes:** `sdfg/validation.py` (#9),
  `frontend/python/newast.py` (#10, already up as **PR #2375**).
- **Investigated, NOT a bug (no code change):** `helpers.py`
  `preserve_minima` (#3).

`MoveIfIntoLoop` (#2) and `MoveLoopInvariantIfUp` (#6) are **yakup/dev-only**
passes — they do **not** exist on `main` (verified: no `class MoveIfIntoLoop`
/ `class MoveLoopInvariantIfUp` anywhere on `origin/main`). Their entries below
are kept for history but are **excluded from the consolidated main PR**; they
are internal pipeline code, fixed in place on `yakup/dev`.

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
- **Scope: yakup/dev-only — NOT on `main`, excluded from the consolidated PR.**
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
- **Bug (as originally diagnosed):** `preserve_minima` set each result dim to
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
- **Fix attempts — ALL FIVE REVERTED (each regresses some core consumer):**
  (a) [utility, subset-symbols] restore translated lower bound
  `external_min + internal_min` preserving extent;
  (b) [utility, subset-pattern] skip point dims (`if (rb - re) == 0:
  continue`);
  (c) [caller, broad symbol discriminator] pass `preserve_minima=False`
  when the internal subset references an outer symbol the external edge
  doesn't (`outer_symbols.keys() - ext_syms`);
  (d) [caller, narrowed-symbol discriminator] restrict (c) to enclosing
  iteration variables only (MapEntry params + LoopRegion vars);
  (e) [caller, descriptor-shape signal] pass `preserve_minima=False` iff
  the NSDFG's inner array descriptor has the **same shape** as the outer
  (the inner access is already in outer coordinates), preserve otherwise.
  **(a)-(d)** all regress `map_fission`
  (`test_symbolic_strided_fission`, `test_mapfission_splits_semi_indirect`,
  `test_single_data_multiple_connectors`, `test_dependent_symbol`) -- the
  same syntactic pattern in `map_fission`'s inner-local NSDFG carries the
  opposite semantics. **(e)** passes the full `map_fission` gate (34/34,
  no regressions there) and the propagation suites (29P/0F), AND fixes
  the original 2x-`canonicalize()` invalid-SDFG crash (idem now `1x ->
  (1,2), 2x -> (2,2) valid`); but in the broader sample it regresses
  `tests/transformations/interstate/loop_to_map_test.py::test_nested_loops`
  (passes on clean HEAD, fails with the fix). So even the
  descriptor-shape signal -- which directly captures the semantic
  difference between the two contexts -- is not sufficient: a third
  context (loop_to_map's nested-loops NSDFG) shares descriptor-shape
  geometry with the canonicalize case but needs `preserve_minima=True`.
- **Conclusion (after 4 attempts):** the bug is real but **not safely
  fixable at this point in the codebase**. Both the utility (`unsqueeze_memlet`)
  and its caller (`propagate_memlets_nested_sdfg`) share a syntactic-pattern
  ambiguity: the same internal subset that for `map_fission` is an inner-local
  per-iteration shape needing re-anchoring is, for the canonicalize NSDFG,
  the per-iteration shape that must NOT be re-anchored. No discriminator
  built from the local information (free symbols, outer_symbols, enclosing
  iteration variables) separates these without deeper context. The fix has
  to be **structural** -- either prevent the canonicalize pipeline from
  forming the NSDFG-bodied-guard shape on a 2nd `canonicalize()` call, or
  refactor `preserve_minima`'s contract across all call sites with
  explicit context (e.g. an explicit `internal_in_outer_coords` argument
  the *caller* sets based on how it built the border memlet, not inferred).
  Core utility + propagation left UNTOUCHED (never-regress rule).
- **Reproducer (canonicalize-free):** drafted but **not landed** (it would
  assert the reverted behaviour). Re-add only with a non-regressing fix.
- **Status (revised):** **NOT A BUG — INTENTIONAL.** Inspection of the
  surviving sparse-dim propagation test (`test_nsdfg_memlet_propagation_with_one_sparse_dimension`,
  authored 2022-12-13 by Philipp Schaad in PR #1176 _"Fix memlet propagation
  out of nested SDFGs"_) shows the `[0:i+1, ...]` running-union form is the
  **codified expected behaviour** for the inner-out memlet of a per-iter WCR
  write, encoding "writes contributed up to iteration `i`". `i` *is* in
  scope on the NSDFG↔MapExit edge (it is inside the enclosing Map's scope —
  `symbols_defined_at(NSDFG) = {M, N, i, j}`), so the resulting SDFG is
  well-formed; the form is over-approximating, not invalid. The original
  validation crash that appeared on 2nd/3rd `canonicalize()` is an
  **orthogonal validation bug** in `dace/sdfg/validation.py:887`, tracked
  separately as fix #9 below. With #9 applied, the multi-pass `canonicalize`
  reaches a stable fixed point on the guarded imperfect nest
  (1x→(1,2), 2x=3x=4x=(2,2)) WITHOUT touching `preserve_minima`. No code
  change to `helpers.py` / `propagation.py`.

### 10. Python frontend shared a `Subset` object between two memlets
- **File:** `dace/frontend/python/newast.py` (`ProgramVisitor.make_slice`)
- **Commit:** `294bc51a5` (pushed `yakup/dev`); **upstream PR #2375** (cherry-pick
  off `main`).
- **Bug:** `make_slice` builds the slice-read memlet with
  `Memlet.simple(array, rng, ...)`. `Memlet.simple` stores a passed-in `Subset`
  **by reference**, and `rng` is frequently the cached `Range` the per-array
  `accesses` cache hands back on a repeated read of the same slice. Two sibling
  reads of e.g. `arr[i, k]` (two loop bodies under a map) produced two distinct
  edges sharing one subset object — violating the invariant that each memlet
  owns its subset. Any later in-place subset rewrite (loop-iterator renaming,
  symbol replacement, offsetting) on one edge silently corrupted the other; it
  surfaced as a value corruption when a guarded sibling loop's read kept the
  first loop's renamed iterator while its write used its own.
- **Fix:** deepcopy `rng` for the slice memlet's subset, mirroring the
  `other_subset` deepcopy two lines above. Pure object-identity fix — subset
  values and the generated SDFG are unchanged.
- **Reproducer test (main-safe):**
  `tests/python_frontend/slice_subset_aliasing_test.py` — asserts no two memlets
  in the parsed SDFG share a subset (or other_subset) object, plus an
  end-to-end value check.
- **Status:** fixed, pushed, PR #2375 open against `main`.

### 11. `TrivialTaskletElimination` dropped the read offset when the source is a `MapEntry`
- **File:** `dace/transformation/dataflow/trivial_tasklet_elimination.py` (`apply`, expr_index 1)
- **Commit:** `6d33f47d2` (pushed `yakup/dev`).
- **Bug:** when the eliminated copy tasklet's source is a `MapEntry`, the
  surviving edge leaves the map's `OUT_<read>` connector, so its memlet must
  describe the read data and its (possibly offset) subset. `apply` reused the
  *write* memlet for every expr_index, stranding the read offset (e.g.
  `a[i + 1]`) in `other_subset`; a later `MapToForLoop` re-lowering read only
  `.subset` (`[0]`) and dropped the offset, yielding an out-of-bounds /
  wrong-value SDFG. This surfaced as a canonicalize **idempotency** failure (the
  second `canonicalize` folded `a[i + k]` to `a[0]`).
- **Fix:** for the `MapEntry`-source case keep the read-side memlet (data +
  subset) on the surviving edge and carry the write subset in `other_subset`;
  the AccessNode→AccessNode and AccessNode→MapExit cases are unchanged.
- **Reproducer test (main-safe):**
  `tests/transformations/trivial_tasklet_elimination_test.py::test_trivial_tasklet_map_source_preserves_offset_subset`
  — `MapEntry --a[i+1]--> copy --> a_idx`, eliminate, assert the surviving edge
  has `memlet.data == 'a'` and keeps the offset subset. Without the fix the edge
  carries `data == 'a_idx'` (offset stranded in `other_subset`); it still
  validates and runs, so the assertion targets the connector/memlet-data
  invariant directly rather than relying on a re-lowering pass.
- **Status:** fixed, regression-verified (canonicalize 138P/4xf, trivial-tasklet
  4P), pushed.

### 12. `SymbolPropagation` mis-propagated symbols and never converged
- **File:** `dace/transformation/passes/symbol_propagation.py`
- **Commits:** `8afc4c6d7`, `e40abc3b8`, `7206d33a8` (pushed `yakup/dev`).
- **Bugs (forward fixpoint propagating single-valued symbols across interstate
  edges):**
  1. **Same-edge read-write race:** a propagated value was substituted into an
     out-edge assignment RHS without excluding that edge's own keys.
     Interstate assignments are simultaneous, so substituting ``anext -> a + b``
     into ``{b: a, a: anext}`` produced ``{b: a, a: a + b}`` (``a`` read and
     written on one edge) -- a validation race.
  2. **Cross-CFG assert crash:** ``_get_in_syms`` asserted a start/branch
     region's edge-accumulated table was empty; on some cross-CFG shapes it
     already carried symbols, crashing the pass.
  3. **Non-termination on cyclic value deps:** the inner substitution loop
     oscillated forever on swaps (``x: tx, tx: y, y: ty, ty: x``).
  4. **Cyclic over-substitution:** raw edge RHS strings were stored without
     resolving against the incoming table, so symbol-to-symbol chains formed
     cycles the final ``replace_dict`` could not resolve (a swap produced no
     swap; ``m = t`` with ``t = m + 2`` double-counted to ``B[m + 4]``).
  5. **Dishonest return value:** ``apply_pass`` always returned ``set()``. The
     pipeline treats any non-None return as "modified" (``Pipeline.apply_pass``),
     so a ``FixedPointPipeline`` like ``SimplifyPass`` could never converge on
     this pass.
- **Fixes:** per-edge self-collision guard (drop substitutions whose value
  free-symbols intersect that edge's keys); conservative ``_combine_syms`` for
  start/branch regions instead of the assert; an iteration cap
  (``len(in)+len(out)+2``) for termination; a ``_resolve`` helper that resolves
  each edge's RHSes against the pre-edge table (simultaneous semantics) and
  keeps a value live (``None``) when it references a same-edge key; and
  ``apply_pass`` now returns the set of symbols actually propagated, or ``None``
  when nothing changed.
- **Reproducer tests:** `tests/passes/symbol_propagation_test.py` (7) +
  `tests/passes/symbol_propagation_hard_test.py` (35) -- race, cyclic-swap,
  diamond-merge, indirection and inter-dependent-symbol cases; all main-safe
  (built with the SDFG/LoopRegion API, no canonicalize dependency).
- **Still deferred (not bugs, refinements):** parse-once memoization;
  IndexedBase-aware ``_resolve`` (the current ``"["`` string guard already drops
  array-reads, same outcome); cross-CFG if-else grouping refinement.
- **Status:** fixed, regression-verified (42 pass; canonicalize + const-prop +
  DCE + loop-to-map blast radius 218P/2xf), pushed.

## Open (separate issue, root-caused)

### 4. Canonicalize structural non-idempotence on the guarded imperfect nest
- **Symptom:** without fix #9, `canonicalize` on the guarded imperfect nest
  is valid at 1x and 2x but the ConditionalBlock count drifts
  `1x->(1 CB) 2x->(2 CB)` and 3x crashes
  (`'NestedSDFG' has no attribute data`).
- **Resolution:** the 3x crash is the validation bug in fix #9 below; with
  #9 applied the pipeline reaches a stable fixed point
  (`1x->(1,2), 2x->(2,2), 3x->(2,2), 4x->(2,2)`). The 1→2 CB drift on the
  *first* re-entry is a fixpoint-recognition issue: the pipeline produces a
  semantically-equivalent form (guard duplicated then collapsed in further
  passes) without re-introducing the loose end. Single-run canonicalize is
  correct and the multi-run output is stable from the 2nd call onward.
- **Status:** RESOLVED by #9 (validation-side); the 1→2 CB single-step drift
  on the very first re-canonicalize is documented but no longer blocks
  multi-pass usage.

### 6. `MoveLoopInvariantIfUp` is broken (dead code) — redesign in progress
- **Scope: yakup/dev-only — NOT on `main`, excluded from the consolidated PR.**
  (The broken version was yakup/dev's own earlier code, not main's.)
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
- **Status:** REDESIGNED + tested + committed. Rewritten from scratch as a
  clean `ppl.Pass` fixpoint (the inverse of `MoveIfIntoLoop`): proper
  `_match` invariance analysis (no loop var; no loop-written data/symbols
  except a hoistable invariant interstate-assignment chain), `_move`
  splices `for k:{prep*; if c: body}` -> `[assign chain]; if c:{for k: body}`,
  empty boundary states dropped, parent-refs repaired, applied to a
  fixpoint so an innermost guard sifts all the way up through nested loops.
  Reproducer suite `tests/transformations/interstate/move_loop_invariant_if_up_test.py`
  (5 tests, all green): invariant-symbolic-guard hoist, invariant-data-guard
  hoist, loop-var-dependent guard NOT hoisted (no-op), innermost sifts all
  the way up (fixpoint, nested loops), interstate-assignment-chain hoisted
  with the guard -- each value-preserving for guard taken/not-taken. The
  old broken API had no importers, so the rewrite is non-breaking.

### 7. `MapFission` unsoundly fissioned a NestedSDFG whose indirection symbol transitively depends on the map iterator
- **File:** `dace/transformation/dataflow/map_fission.py` (`can_be_applied`, expr_index 1 — map-with-NestedSDFG)
- **Bug:** the iterator-dependence rejection only checked ONE hop. The Fortran
  frontend lowers `a(idx(i))` to a NestedSDFG that loads `idx(i)` into an
  internal transient `__tmp` then carries `__sym = __tmp` on an interstate
  edge; the assignment RHS is `__tmp` (not the iterator-indexed input), so
  `assign_free & map_params` / `& inputs_dep_on_map` were empty and
  MapFission would proceed to fission a NestedSDFG whose indirection symbol
  transitively depends on the map iterator — an unsound split.
- **Fix:** seed a `tainted` set from iterator-dependent input connectors,
  propagate the taint forward through every NestedSDFG state's dataflow
  (monotone closure), and additionally `return False` if an interstate
  assignment names any tainted container. Strictly *additive refusal* —
  only adds `return False` paths, never removes one; cannot make a
  previously-correct fission wrong nor reduce any asserted fission count.
- **Verification:** `tests/transformations/map_fission_test.py` +
  `map_fission_indirect_test.py` + `map_fission_e2e_structure_test.py`
  = **37 passed, 0 regressions** with the change.
- **Reproducer test:** the Fortran-frontend
  `tests/mapfission_indirect_loopnests_test.py` (dace-fortran, UNVERIFIED —
  needs the d2/FaCe env) exercises the exact `a(idx(i))` lowering; a
  main-safe focused unit test is TODO.
- **Status:** fixed, regression-verified; commit pending.

### 9. `validate_state` crashed accessing `.data` on a non-AccessNode endpoint
- **File:** `dace/sdfg/validation.py` (`validate_state`, dimensionality-mismatch check)
- **Bug:** the src/dst-element-count check assumed both endpoints are
  `AccessNode` (it accessed `sdfg.arrays[src_node.data].veclen` and
  `sdfg.arrays[dst_node.data].veclen` unconditionally). On any edge with
  `other_subset is not None` whose endpoint is a `NestedSDFG`/`MapEntry`/
  `MapExit`, the `.data` attribute does not exist, raising
  `AttributeError: 'NestedSDFG' object has no attribute 'data'` and aborting
  validation. Triggered any time a multi-pass canonicalize produced a deeper
  NSDFG nest whose inner NSDFG↔MapExit edge carried both `src_subset` and
  `dst_subset` (e.g. a single-scalar reduction destination materialised on a
  per-iteration write). The View-exception branch had the same latent
  assumption.
- **Fix:** read each side's `veclen` *only* when its endpoint is an
  `AccessNode`; default to `1` otherwise (scope nodes route data through
  connectors and contribute `veclen = 1` at the edge boundary). Guard the
  two `View` exception clauses with the same `isinstance(..., AccessNode)`
  check so they no longer dereference `.data` on scope nodes.
- **Verification:** the `canonicalize` idempotence repro
  (`for i: for j: b[i,j]=a[i,j]*2.0; c[i]=a[i,0]+1.0` under a top-level
  guard) now reaches a stable fixed point at the 2nd call and stays there
  (`1x->(1,2), 2x=3x=4x=(2,2)`, no warnings). Full transformations +
  canonicalize + propagation + passes sweep: same 14 failures as the
  unchanged baseline (all pre-existing: 5 `offset_loop_and_maps`
  TODO-`raise` cases, 1 `perf_loop_nesting` refusal on "Loop symbols used in
  block", 1 `branch_elimination::test_s441` `np.random.rand(symbol)` test
  bug, plus 7 environmental cache/import errors). Zero net regressions.
- **Reproducer test (main-safe):** TODO — construct the minimal
  NSDFG→MapExit edge with `other_subset` set and assert `sdfg.validate()`
  succeeds (without the fix the assertion errors with `AttributeError`).
- **Status:** fixed, regression-verified; commit pending.

### 8. (BACKLOG, not a bug) MoveIfIntoMap underpowered for the top-level guard-over-map shape
- **Observation:** `MoveIfIntoMap.can_be_applied` only matches a guard
  already inside a NestedSDFG that sits in an outer map; it is a no-op on
  the raw frontend `if c: for i,k in dace.map: body`. The desirable
  canonical move `if c: map` -> `map: if c: body` is currently delivered
  only by the full pipeline / the existing-pass composition.
- **Delivered (Option A, low-risk, zero core surgery):** the capability is
  achieved by *reusing existing transformations* --
  `MapExpansion` + `MapToForLoop` + `InlineMultistateSDFG` +
  `MoveIfIntoLoop` + `LoopToMap` (LoopToMap re-applies, parallelism
  recovered). Verified + covered by symmetric loop/map tests
  (`tests/transformations/move_if_into_loop_and_map_symmetric_test.py`).
- **Deferred (Option B, deliberate future effort):** extend
  `MoveIfIntoMap` for the top-level/no-outer-map case + a
  MoveIfIntoMap/MoveIfIntoLoop fixpoint. Key finding lowering its cost:
  `_rewrite_inner_sdfg` is **already outer-map-agnostic** (only uses
  `enclosing_sdfg`+`inner_nsdfg`), so the work is mostly relaxing
  `can_be_applied` (accept no-outer-map with a soundness check: condition
  not dependent on map-written data) + validating `apply`'s symbol/array
  piping for array-valued conditions without an outer nsdfg. Needs the
  full move_if_into_map + canonicalize regression sweep. Not a bug; an
  ergonomics/architecture improvement.

## Design limitations / known canonicalize gaps

These are not bugs in a transformation; they are pipeline-shape gaps where
canonicalize is value-correct but does not yet reach the structural ideal, or
where an input is invalid by DaCe semantics. Tracked so the design effort is
explicit.

### L-A. `MoveLoopInvariantIfUp` does not hoist a guard out of a parallel MAP nest
- **Shape:** `for i in dace.map: for j in dace.map: if lim < N: ... else: ...`
  with the guard reading only outer-scope symbols (``lim``, ``N``).
- **Ideal:** hoist the guard above the whole ``i, j`` map nest -> one
  top-level ``ConditionalBlock``, each branch a clean collapsed map.
- **Gap:** MLIU sifts invariant guards out of LoopRegion nests, but the
  Python-frontend map-nest shape (guard inside the inner map-body NestedSDFG)
  is not matched/lifted. All-or-nothing upward (no partial one-level hoist) is
  a deliberate constraint, so the guard must clear every enclosing scope at
  once. Pinned by `canonicalize_branchy_polybench_test.py::test_loop_invariant_guard_over_inner_hoisted_to_top`
  (strict xfail). Value-correct today.

### L-B. Fully-parallel statement not fissioned to a standalone collapsed map
- **Shape:** one ``for i: for j:`` nest with a fully-parallel statement
  ``A[j,i] = A[j,i]*2`` beside a ``j``-carried ``B[i,j] = B[i,j-1] + B[i,j]``.
- **Ideal:** fission ``A`` into a standalone collapsed 2D Map ``[i, j]`` and
  leave ``B`` as ``map i: { loop j }``.
- **Gap:** today ``A`` and ``B`` keep sharing the outer ``i``-map
  (``map_param_counts == [1, 1]``) rather than ``A`` becoming a 2-parameter
  map. Needs fission at the outer-nest level (or a post-LoopToMap
  map-fission + map-collapse) so ``A``'s full iteration space parallelizes
  independently. Pinned by `canonicalize_mixed_parallelism_test.py::test_mixed_parallelism_A_becomes_collapsed_2d_map`
  (strict xfail). Value-correct today.

### L-C. Scalars are not map-local — loop-carried scalar reductions need a sequential outer loop
- **Finding:** a scalar transient declared inside a ``dace.map`` body is NOT
  thread-local; it is shared across the parallel map. A per-row reduction with
  a scalar accumulator (``for i in dace.map: s = 0; for j: s += ...; b[i] =
  s``) is therefore invalid (a data race / mis-accumulation, wrong even
  un-canonicalized). The same shape with an array-element accumulator
  (``b[i] += ...`` or an ``N``-vector ``s[i]``) under a parallel map is
  likewise mis-lowered by the frontend for the inner loop-carried case.
- **Valid form:** use a sequential ``range`` outer loop with the scalar
  accumulator; canonicalize then parallelizes the row-independent work into a
  map and keeps the loop-carried reduction sequential (``LoopToMap`` refuses it
  on the write-pattern check). This is the form the canonicalize reduction
  tests now use after correcting the invalid map-local-scalar kernels.

### L-D. `SymbolPropagation` refinements (see #12)
Parse-once memoization; IndexedBase-aware ``_resolve`` (replacing the ``"["``
string guard); cross-CFG if-else grouping refinement. None are correctness
bugs.

### L-E. TODO — re-roll (untile) a manually-unrolled lane chain
- **Shape:** a loop with step ``S != 1`` whose body is ``S`` manually-unrolled
  lanes -- the lane ``k`` statement is the lane-0 statement with every index
  shifted by ``+k``. TSVC ``s353`` is the direct example (step 4, "unrolled
  sparse saxpy"):

  .. code-block:: python

      for i in range(0, LEN - 3, 4):
          a[i]     += alpha * b[ip[i]]
          a[i + 1] += alpha * b[ip[i + 1]]
          a[i + 2] += alpha * b[ip[i + 2]]
          a[i + 3] += alpha * b[ip[i + 3]]

  Both the **indirect** form above (gather ``b[ip[i+k]]``) and the **dense**
  form (``a[i+k] += alpha * b[i+k]``) must be handled.
- **Transformation:** detect that the ``S`` lanes are the lane-0 body replicated
  at offsets ``0..S-1``; keep lane 0, drop the rest, and re-roll the loop to
  step 1 (``for i in range(0, LEN): a[i] += alpha * b[ip[i]]``). After this
  flattening, ``LoopToMap`` parallelizes the loop (``a[i]`` independent per
  ``i``; the indirect read is per-element).
- **Match conditions:** loop step ``S`` equals the lane count; every non-lane-0
  statement is structurally lane-0 with ``i -> i + k``; no cross-lane carried
  dependence (each lane writes its own ``a[i+k]``). Refuse otherwise.
- **Goal:** add to canonicalize so ``s353`` and the dense variant become
  parallel maps. (User-requested 2026-05-21.)
- **TSVC re-roll candidates** (`VectraArtifacts/tsvc_2/tsvc2_core.py`):
  ``s351`` (dense step-4 saxpy -- the dense case), ``s353`` (indirect step-4
  sparse saxpy -- the gather case), ``s352`` (step-5 unrolled dot reduction),
  ``s116`` (step-4), ``s31111`` (step-4 reduction). Pinned by
  ``tests/canonicalize/canonicalize_reroll_unrolled_test.py``.

### L-G. TODO — broaden loop-distribution (node splitting) coverage
- **Have:** ``canonicalize_mixed_parallelism_test.py`` (one fully-parallel + one
  carried statement in a nest) and the ``LoopFission`` pass.
- **TSVC node-splitting kernels to cover** (splitting the loop body into
  separate loops exposes parallel/vectorizable sub-loops):
  ``s211``, ``s212``, ``s221``, ``s222``, ``s231``, ``s232``, ``s233``,
  ``s243``, ``s244``, ``s126``, ``s1213`` (all in ``tsvc2_core.py``). Survey
  which already fission+parallelize under canonicalize and which need work;
  see also L-B. (User-requested 2026-05-21.)

### L-F. TODO — best-effort loop peeling to expose a parallel middle
- **Idea:** a best-effort pass that peels up to ``X`` leading and ``Y`` trailing
  iterations of a loop and checks whether the remaining middle iterations are
  then parallelizable / vectorizable (the boundary iterations carrying the
  dependence or the special-case access that blocks the whole loop).
- **Use:** several TSVC benchmarks need front/back peeling before the steady-
  state middle vectorizes. Try small ``X``, ``Y`` and keep the peeling only if
  the middle becomes a clean parallel map. (User-requested 2026-05-21.)
