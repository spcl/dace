# StageGlobalArrayThroughScalars — Specification

## Purpose

In several cloudsc SDFGs the dataflow between two compute tasklets is routed
*through a GLOBAL (non-transient) array access node*:

```
Tasklet1 --[global[s1]]--> A(global) --[global[s2]]--> Tasklet2
```

`A` is an access node for a **non-transient** array. `Tasklet1` writes the
global array at subset ``s1`` and the *same* access node `A` then feeds
`Tasklet2`, which reads at subset ``s2``. Routing the value through a global
array node forces a global-memory round-trip and a false serialization that
blocks vectorization / tiling.

This pass **stages that hop through fresh transient scalars** so the
producer→consumer value flow is decoupled from the global store, while the real
store to the global array is always preserved.

This pattern is exactly the ``zsolqa`` / ``zqlhs`` / ``zsolqb`` reuse seen in
``_get_cloudsc_snippet_three`` / ``_get_cloudsc_snippet_four``: a global array
access node sits between two tasklets in a single state.

## Pinned API

```python
from dace.transformation.passes.vectorization.stage_global_array_through_scalars import (
    StageGlobalArrayThroughScalars,
)
```

- **Module**: ``dace/transformation/passes/vectorization/stage_global_array_through_scalars.py``
- **Class**: ``StageGlobalArrayThroughScalars`` — a
  ``dace.transformation.pass_pipeline.Pass`` subclass (standalone, decorated
  ``@transformation.explicit_cf_compatible``).
- ``apply_pass(self, sdfg, pipeline_results) -> Optional[int]`` — applies the
  rewrite in place. Returns the number of ``T1 -> A(global) -> T2`` occurrences
  rewritten, or ``None`` when nothing changed (Pass-pipeline no-op convention).
- ``modifies(self) -> ppl.Modifies`` — returns
  ``Modifies.AccessNodes | Modifies.Memlets | Modifies.Descriptors`` (adds
  transient scalar descriptors + access nodes, rewires memlets).
- ``should_reapply(self, modified) -> bool`` — ``False`` (single fixpoint pass).
- ``depends_on(self) -> set`` — ``{}`` (standalone).
- Recurses into nested SDFGs (the cloudsc pattern lives inside body NSDFGs).

## Definitions

- **A** — an ``AccessNode`` whose descriptor ``sdfg.arrays[A.data]`` is **not**
  transient (a global / argument array).
- **T1** — a ``Tasklet`` (or any non-AccessNode node) that has an out-edge into
  ``A`` writing the subset ``s1`` (``e1.data`` describes ``A.data[s1]``).
- **T2** — a ``Tasklet`` (or any non-AccessNode node) that has an in-edge from
  ``A`` reading the subset ``s2`` (``e2.data`` describes ``A.data[s2]``).
- ``s1`` / ``s2`` are the ``subsets.Range`` (or ``Indices`` normalized to a
  ``Range``) carried by the respective memlets, taken on the ``A`` side
  (i.e. the array subset of the global array, not the connector subset).
- A scalar is **fresh** when introduced by the pass via
  ``sdfg.add_scalar(find_new_name=True, transient=True,
  storage=StorageType.Register)`` with the global array's element dtype.

## Targeting

For each state, and each non-transient access node ``A`` that has at least one
incoming edge from a tasklet (``T1``) **and** at least one outgoing edge to a
tasklet (``T2``), enumerate every ``(T1, e1) x (T2, e2)`` producer/consumer
pair through ``A``. Each such pair is one *occurrence*. The pass handles every
occurrence independently (a single ``A`` may host several).

The producer side ``T1 -> A`` and the consumer side ``A -> T2`` are matched
per *edge*: ``s1`` is read from ``e1`` (``T1 -> A``); ``s2`` from ``e2``
(``A -> T2``).

## Disjointness test

Use ``dace.subsets.intersects(s1, s2)`` (the module-level function), which
returns:

- ``False`` — provably disjoint  → **Case A**.
- ``True`` — provably overlapping → **Case B**.
- ``None`` — indeterminate → treated as **NOT provably disjoint** → **Case B**
  (conservative: a possible RMW is preserved as an RMW).

Both subsets are first normalized: ``Indices`` → ``Range.from_indices``; the
subsets compared are the **array-side** subsets of ``A`` (``e.data.subset`` when
``e.data.data == A.data``; otherwise ``e.data.other_subset``).

## Refusal / skip conditions (occurrence left UNCHANGED)

An occurrence is **skipped** (left exactly as-is, contributing 0 to the count)
when any of the following hold:

1. ``A`` is **transient** (only global arrays are staged).
2. ``T1`` or ``T2`` is itself an ``AccessNode`` (the pass stages
   *tasklet→tasklet* hops, not array→array copies).
3. **Intervening-write invariant violated**: another write to the *same global
   element* exists in the state — i.e. some access node of ``A.data`` other than
   this occurrence's bridge node carries an incoming write whose subset is NOT
   provably disjoint from ``s1``, *and* that other write is not itself this
   occurrence's producer ``T1``. In that situation the value the transient would
   carry is not the sole authority for ``A.data[s1]`` and folding the store
   through a single transient could drop or reorder a write — **refuse**.

   A **linear accumulation chain** does NOT trip this guard: when each bridge
   access node of ``A.data`` has exactly one writing in-edge (from a tasklet)
   and one reading out-edge (to a tasklet) — the canonical
   ``t1 -> A@1 -> t2 -> A@2 -> t3 -> ...`` cloudsc ``zqlhs`` / ``zsolqa`` reuse —
   each hop is staged independently (the prior hop's bridge node was already
   rewritten / is the value source, so there is no *competing* unrelated write).
   The guard fires only for a write to the same element from a node that is
   neither the occurrence's producer nor a consumed-and-restaged prior hop (as
   in a dead overwritten store sitting beside the read).
4. The producer edge ``e1`` or consumer edge ``e2`` carries a ``wcr``
   (write-conflict resolution / reduction) — staging would drop the
   accumulation. **Refuse.**
5. ``e1`` or ``e2`` is a *whole-array* / multi-element memlet (volume > 1, i.e.
   not a single scalar element). The transient is a **scalar**; only
   single-element hops are stageable. Multi-element hops are left unchanged.

When *all* occurrences in an SDFG are skipped, ``apply_pass`` returns ``None``
and the SDFG is byte-for-byte unchanged.

## Case A — s1 and s2 PROVABLY DISJOINT

The ``T1 -> T2`` dependency through ``A`` is **false** (they touch different
elements). Decouple the read from the write while keeping the real store.

### Before

```
        e1: A.data[s1]                e2: A.data[s2]
  T1 ───────────────────►  A(global) ───────────────────►  T2
                            (one global access node)
```

### After

```
  T1 ──[A1[0]]──►  A1(transient scalar)                    # T1's output value
                       │
                       └──[A.data[s1]]──►  A(global)        # the real store of T1's value
                                            │
                                  (dep edge: A1 ─dep─► A is NOT needed;
                                   the store edge A1 -> A already orders it)
                       A(global) ──[A.data[s2]]──►  A2(transient scalar)   # the global read at s2
                                                      │
                                                      └──[A2[0]]──►  T2
```

Concretely the pass:

1. Adds two fresh transient scalars ``A1`` and ``A2`` (element dtype of
   ``A.data``, ``Register`` storage).
2. **Producer side**: redirects ``e1`` so ``T1``'s output writes ``A1``
   (memlet ``A1[0]``), then adds an AccessNode→AccessNode store edge
   ``A1 --[A.data[s1]]--> A`` (other_subset ``A1[0]``) — the *real* store of
   ``T1``'s value to ``global[s1]`` (the "writes to later" edge).
3. **Consumer side**: adds a load edge ``A --[A.data[s2]]--> A2``
   (other_subset ``A2[0]``), then redirects ``e2`` so ``T2`` reads ``A2``
   (memlet ``A2[0]``).

The dataflow ``T1 → A1 → A`` and ``A → A2 → T2`` makes ``A`` a pure pass-through
node; the false ``T1→T2`` serialization through a single shared ``A`` element is
gone because ``T2`` now reads its own staged copy ``A2`` of ``global[s2]``,
independent of ``T1``'s store to ``global[s1]``.

**Dep edge note**: because the store ``A1 -> A`` and the load ``A -> A2`` both
touch the same global node ``A`` in the same state, the existing state-level
ordering (A is written before it is read) already sequences the store ahead of
the load. No explicit empty/dependency edge is required for Case A. (An
implementation MAY add an explicit ``A1 -dep-> A`` empty memlet if it splits the
store/load across states, but in-state it is redundant.)

## Case B — s1 and s2 NOT provably disjoint (real RMW)

``T2`` may read what ``T1`` wrote; the dependency is real. The value must flow
``T1 -> T2`` *and* also reach the global array.

### Before

```
        e1: A.data[s1]                e2: A.data[s2]   (s1, s2 may overlap)
  T1 ───────────────────►  A(global) ───────────────────►  T2
```

### After

```
  T1 ──[A1[0]]──►  A1(transient scalar) ──[A1[0]]──►  T2     # value flows T1->T2 through transient (RMW kept)
                       │
                       └──[A1[0]]──►  AssignTasklet ──[A.data[s1]]──►  A(global)   # value ALSO stored to global
```

Concretely the pass:

1. Adds one fresh transient scalar ``A1`` (element dtype of ``A.data``).
2. Redirects ``e1`` so ``T1`` writes ``A1`` (memlet ``A1[0]``).
3. Redirects ``e2`` so ``T2`` reads ``A1`` (memlet ``A1[0]``) — the RMW value
   now flows producer→consumer directly through the transient.
4. Adds an ``AssignTasklet`` (``_out = _in``, single in / single out) reading
   ``A1[0]`` and writing ``A --[A.data[s1]]--> `` so the global array still
   receives the stored value at ``s1``.

**Invariant**: there must be **no other write** to ``A.data`` on this chain
(refusal condition 3). The pass verifies this before rewriting; if violated, the
occurrence is skipped.

The global node ``A`` keeps exactly one incoming edge (from ``AssignTasklet``)
and loses its outgoing edge to ``T2`` — it is now write-only on this chain,
which no longer serializes ``T2``'s read behind the global store.

## Connector / memlet details

- New scalars use connector-free AccessNode↔AccessNode edges where one side is
  the global store/load (``data=A.data`` subset on the global side,
  ``other_subset`` = the scalar's ``[0]``), matching the convention in
  ``RemoveRedundantAssignmentTasklets`` (``Memlet(data=..., subset=...,
  other_subset=...)``).
- Tasklet-side edges keep the tasklet's original connector name (the
  ``src_conn`` of ``e1`` / the ``dst_conn`` of ``e2``); only the
  ``AccessNode`` endpoint and the memlet change.
- ``A1`` / ``A2`` memlets are ``Memlet("A1[0]")`` / ``Memlet("A2[0]")``
  (single-element scalar subset).
- The ``AssignTasklet`` (Case B) has inputs ``{"_in"}`` / outputs ``{"_out"}``
  and code ``"_out = _in"``.
- All copied subsets are ``copy.deepcopy``-d to avoid shared-Subset aliasing.

## Numerical contract

The rewrite is value-preserving: in both cases the global array receives the
same value at ``s1`` that ``T1`` produced, and ``T2`` receives the same value it
would have read. Tests compile + run the original SDFG as reference and compare
(``numpy.allclose``) against a deep-copied, pass-applied SDFG.
