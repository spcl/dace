# `CascadeInterstateEdgeAssignmentsUp` — design draft (NOT YET IMPLEMENTED)

> **Status**: design only. Implementation is gated on this design being
> reviewed. The three strict-xfail tests below pin the patterns the pass
> must address.

## Purpose

LICM-equivalent for **interstate-edge symbol assignments**. The existing
`LoopInvariantCodeMotion` hoists *tasklet* and *map-scope* invariant code
out of `LoopRegion`s; it does not touch interstate-edge assignments. The
Python frontend coins symbols like `kfdia_plus_1` for non-symbol bound
expressions (`get_target_name` heuristic in `newast.py`); after
canonicalize stages remix CFGs, those `kp1 = expr` interstate-edge
assignments often end up *inside* the loops whose bounds read `kp1`,
producing the body-assigns-loop-range-symbol shape that `LoopToMap`'s
refuse-check (`dace/transformation/interstate/loop_to_map.py`, "Loop
range references symbol(s) assigned by the loop body's interstate
edges" gate) declines.

A pass that **cascades each invariant interstate-edge assignment up** as
far as it's legal would (a) fix this class of refusal, (b) reach the
canonicalize idempotence the existing pipeline approximates with
`UniqueLoopIterators(post_value=False)`, and (c) eliminate per-iteration
re-assignment of values that never change.

## Pinned-by-test patterns

These three tests are currently `pytest.mark.xfail(strict=True, ...)`
with precise reasons linking to this design doc:

1. `tests/canonicalize/canonicalize_symbol_lifting_test.py::test_reduction_with_inner_accumulator_value_preserving`
   — per-row reduction with inner-loop accumulator. The inner `for j` loop
   carries a state-level assignment that touches a scalar transient `s`;
   after the rename `j` is renamed but the accumulator's surrounding
   interstate-edge structure needs to lift cleanly.

2. `tests/canonicalize/canonicalize_symbol_lifting_test.py::test_cloudsc_style_range_plus_one_value_preserving`
   — `range(kidia, kfdia + 1)`. The `kfdia_plus_1` promoted symbol's
   `kfdia_plus_1 = kfdia + 1` assignment must end up *above* the loop
   whose range reads it. Currently after canonicalize the per-iteration
   subscript `a[i]` folds to a constant index (every iteration reads the
   same element).

3. `tests/canonicalize/canonicalize_map_structure_test.py::test_stencil_reduction_mixed_value_preserving`
   — same family as (1): the inner-accumulator pattern. SDFG validates
   and compiles after the cleanup-gate fix
   (`unique_loop_iterators.py`, the `used_symbols(all_symbols=False)`
   switch), but per-row reduction values drift.

## The "all-or-nothing upward" principle

Stated by the user, binding:

> *"In general when we hoist iedge assignments or conditions, we should
> cascade them all the way up or not at all. Moving inside is fine as we
> can move as much as we want."*

A one-level partial hoist is forbidden: it can move an assignment out of
one enclosing loop while leaving it inside another enclosing loop where
the same scope-mismatch family of bugs reappears (the per-iteration shape
is just relocated). The pass must:

* Compute the **destination scope** of a hoist atomically before moving.
  The destination is the outermost CFG region that satisfies all legality
  predicates below.
* If the destination is the assignment's current scope (i.e., it cannot
  move legally), do nothing.
* Otherwise move the assignment exactly to that destination.

## Legality predicates (per-assignment `key = rhs`)

Walking the CFG outward from the assignment's current edge, the
destination scope `D` is the **outermost** region where ALL of these
hold:

### L1. RHS invariance at `D`

Every free symbol of `rhs` is defined at `D` (either a Map parameter,
LoopRegion `loop_variable`, an SDFG-level declared symbol bound by an
ancestor's `symbol_mapping`, or an interstate-edge assignment that
dominates `D`). If any RHS symbol is defined *inside* `D` (e.g. an
enclosing loop's variable), `D` is too high.

### L2. No write to RHS symbols between the assignment's current edge and `D`

If any path between `D`'s entry and the current edge writes to a free
symbol of `rhs`, the lift would change the value seen by readers below
that write. Reject.

### L3. No read of `key` between `D` and the current edge

If any block between `D`'s entry and the assignment's current edge reads
`key`, moving the assignment up would change the value those reads
observe (they would see the new value rather than whatever value `key`
held before the loop's first iteration). Reject.

### L4. No write to `key` between `D` and the current edge

A write to the same key on a different interstate edge (or by a tasklet
into the underlying symbol) would conflict with the relocated assignment.

### L5. The current edge is unconditionally executed by `D`

If the assignment lives on an edge whose condition depends on values
that `D` doesn't fix (e.g. inside a `ConditionalBlock`'s branch), the
hoist would make the assignment fire on paths that the original
conditional excluded. **Acceptable subset** (matching the user's
`MoveLoopInvariantIfUp`-family rule): if `key`'s value is never read on
any other path through `D`, the conditional execution is observationally
moot and the hoist is safe. Conservative: refuse when in doubt.

### L6. NSDFG-boundary symbol routing

If the path from `D` to the current edge crosses one or more
NestedSDFG boundaries, each crossing requires the symbol to appear in
the corresponding NSDFG node's `symbol_mapping` (passthrough). The
pass must add these passthroughs atomically *as part of the move*; if
an inner NSDFG already declares `key` in its own `symbols`, the
pass must also remove that declaration to avoid a phantom free symbol
(same family as the cleanup-gate bug fixed in `unique_loop_iterators.py`).

## Algorithm

```
def cascade_assignments_up(sdfg, fixed_point=True):
    changed = True
    while changed:
        changed = False
        for cfg in sdfg.all_control_flow_regions(recursive=True):
            for e in list(cfg.edges()):
                if not e.data.assignments:
                    continue
                # Snapshot of (key, rhs) pairs so we can mutate during iteration.
                for key, rhs in list(e.data.assignments.items()):
                    D = find_destination(cfg, e, key, rhs)
                    if D is cfg:
                        continue  # already at destination
                    move_assignment_to(D, key, rhs, source_edge=e)
                    add_nsdfg_passthroughs(path_from=cfg, path_to=D, sym=key)
                    drop_inner_declarations(sym=key)
                    changed = True
```

`find_destination` walks the `parent_graph` chain outward and stops at
the first ancestor where any of L1–L6 would fail. The legal destination
is one step *inside* that ancestor (the last legal scope).

## Tests the implementation must pass

In addition to flipping the three XFAILs above to passing, the unit
suite must cover (each as a small focused test):

* **Outer-only invariant single hoist** — `kp1 = K + 1` inside a single
  loop, K is outer; hoist to outer SDFG scope.
* **Two-loop shared hoist** — `kp1 = K + 1` inside two sibling loops;
  one hoist serves both (no duplicate per loop).
* **Mixed outer + loop-var (refuse)** — `tmp = K + i`; cannot hoist
  past the `i` loop.
* **Data-dependent (refuse)** — `tmp = arr[i]`; depends on per-iteration
  array read, cannot hoist.
* **Conditional-guarded assignment (L5)** — `if c: { kp1 = K + 1 }`;
  hoist only when `kp1` is unread on the not-taken path.
* **Cross-NSDFG hoist (L6)** — assignment inside an NSDFG body, RHS
  references an outer symbol; hoist past the NSDFG boundary and route
  through `symbol_mapping`.
* **Transitive chain** — `s1 = K + 1; s2 = 2 * s1`; both hoist together
  to the same outer scope, preserving order.
* **All-or-nothing** — assignment that can hoist one level but not two;
  must NOT hoist (the user's binding principle).
* **Idempotence** — running the pass twice changes nothing on the
  second call (declared by `should_reapply`).
* **Value preservation** — for each of the above, a numerical-
  equivalence check against a pure-numpy oracle.

## Open design questions (to settle before implementation)

* Should the pass run within the existing canonicalize fixpoint, or as a
  standalone post-canonicalize step? The interaction with
  `MoveIfIntoLoop`/`LoopToMap` ordering matters.
* For L5, do we strengthen the conditional-guard check via dataflow
  analysis, or stay strictly conservative (refuse anything inside a
  ConditionalBlock branch)?
* How does this interact with `UniqueLoopIterators(post_value=True)` for
  callers who explicitly want the Fortran-style postamble? Probably
  orthogonal but needs explicit handling.

## Non-goals

* Hoisting tasklets / dataflow — that is `LoopInvariantCodeMotion`'s
  job. This pass moves interstate-edge symbol assignments only.
* Moving conditionals out of loops — that is `MoveLoopInvariantIfUp`'s
  job (separate pass, complementary to `MoveIfIntoLoop`).
* Per-iteration LICM (i.e. moving inside a deeper iteration). The user's
  binding rule is upward-only.
