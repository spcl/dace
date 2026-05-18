# Canonicalization Pipeline — Living Design Document

Authoritative, **living** design doc for the SDFG canonicalization pipeline. It
lives next to the code (`dace/transformation/passes/canonicalize/`) and is
updated in lockstep with it.

## Purpose

Rewrite any SDFG into one deterministic canonical form so later passes (fusion,
vectorization, scheduling, equivalence checks) see a single shape per
computation instead of many incidental ones (tiled / fused / arbitrary iterator
names / permuted nests / conditionals buried inside maps).

Recipe shape (Trümper et al. canonicalization style): blow apart to minimal
units → order nests canonically → normalize iterators / induction variables /
invariants → recompose by maximal fusion → hoist conditionals out.

## Public API

`pipeline.py` provides, modeled on ``auto_optimize`` (a recipe function):

- `CanonicalizationPipeline(ppl.Pass)` — composable pass; applies the stages
  imperatively, once, in order. There is **no** `CanonicalizePass` and no
  `skip` parameter — every stage always runs.
- `canonicalize(sdfg, validate=True, validate_all=False) -> SDFG` — one-call
  recipe.

Importable as `from dace.transformation.passes import
CanonicalizationPipeline, canonicalize` (re-exported) or
`from dace.transformation.passes.canonicalize import …`.

It is **not** a :class:`~dace.transformation.pass_pipeline.Pipeline` subclass: a
single pipeline forbids duplicate pass types and `Pipeline` /
`PatternMatchAndApplyRepeated` are unhashable, so stages are applied directly.
Stages that need iteration iterate internally; the pipeline does not re-run.

## Organizational rule

A pass that exists only for this pipeline lives in this subpackage. Reusable
passes keep their normal homes and are imported/composed here so they stay
usable from outside.

`Untile` (new, reusable — inverse of `MapTiling`) will live in
`dace/transformation/dataflow/untile.py`, not here.

## Stage order (applied once, top to bottom)

| # | Stage | Concrete passes | Status |
|---|---|---|---|
| 0 | `pre_simplify` | `SimplifyPass()` | wired |
| 1 | Undo tiling | `Untile` | **TODO** (lives outside this dir) |
| 1b | `split_tasklets` | `SplitTasklets()` | wired |
| 1c | `prepare_fission` | `_cleanup(loop_to_reduce=False)` → `PatternMatchAndApplyRepeated([MoveIfIntoMap])` → `SimplifyPass()` (inline SDFGs + structure) — run before fission so conditionals/nested SDFGs don't block it | wired |
| 2 | `maximal_fission` | `PatternMatchAndApplyRepeated([MapExpansion, MapFission])` → `SimplifyPass()` → `_cleanup(loop_to_reduce=True)` | partial (loop-fission step **TODO**) |
| 3 | `reorder_offsets` | `NormalizeLoopsAndMaps` (every map range → `0:trip:1`) → `SimplifyPass()` | wired |
| 4 | `perfect_loop_nesting` | `MoveIfIntoMap` → `MapExpansion` (= "map uncollapse", `map[i,j]`→`map[i];map[j]`) → `PerfLoopNesting` → `MapCollapse` (re-collapse maximally) → `MinimizeStridePermutation` (permute to minimize strides) → `SimplifyPass()` → `_cleanup(loop_to_reduce=True)` | being generalized |
| 5 | `normalize` | `SSALoopIterators` → `SimplifyInductionVariables` → `LoopInvariantCodeMotion` → `LoopToReduce` → `SimplifyPass()` | wired |
| 6 | `loop_to_map` | `PatternMatchAndApplyRepeated([LoopToMap])` | wired |
| 7 | `maximal_fusion` | `SimplifyPass()` → `Pipeline([FullMapFusion])` → `PatternMatchAndApplyRepeated([TaskletFusion, TrivialTaskletElimination])` | wired |
| 8 | `hoist_if` | hoist invariant conditionals above maps | **no-op for now** |

### `perfect_loop_nesting` (Stage 4) — sub-pipeline & generalization

Sub-pipeline order (each step generalizes what the next can collapse):

1. `MoveIfIntoMap` — push conditionals guarding maps inside, so more maps
   become collapsible/adjacent (sub-pass B; must work strongly).
2. `MapExpansion` ("map uncollapse") — split every multi-dimensional map
   ``map[i, j]`` into nested single-dim ``map[i]; map[j]``. Already exists
   (`dace/transformation/dataflow/map_expansion.py`); do **not** reimplement.
3. `PerfLoopNesting` — duplicate a parent nest per independent inner map.
4. `MapCollapse` — re-collapse nested single-dim maps maximally.
5. `MinimizeStridePermutation` — permute the (now canonical) nests to
   minimize access strides (sub-pass A, committed `0324f332d`).

`PerfLoopNesting` **must work without a NestedSDFG** — operating within the
same state. Required general behavior (no NSDFG indirection needed)::

    map1
      map2
        map3
        map4
    ==>
    map1
      map2
        map3
    map1
      map2
        map4

i.e. duplicate the enclosing parent nest once per independent inner sibling
map, **respecting data dependencies** between the inner maps (dependent
siblings must not be split / must keep correct order). The NSDFG-wrapped form
remains supported; the same-state inlined form (the common ``simplify=True``
frontend shape ``parent-map body = [MapExit, MapEntry, MapEntry, …]``) is the
generalization being added. New edge cases (intervening producer chain;
dependent inner maps that must NOT split; unsound extra write) land as
pure-frontend unit tests.

### `MoveIfIntoMap` (sub-pass B) — correct generalization

The base pass only matched inner maps whose body is a single `NestedSDFG`;
the Python frontend emits inner `dace.map` bodies as plain `Tasklet`
subgraphs, so it never engaged. Generalization: `_normalize_inner_map_bodies`
nests any non-NSDFG inner-map body into a single `NestedSDFG`
(`helpers.nest_state_subgraph`) before the proven per-map guard injection
runs, and `can_be_applied` is relaxed to the normalizable sibling-maps shape
(`_inner_maps_shape_ok`). Pure-frontend tests: two sibling guarded maps
(applies, becomes collapsible), single guarded map, and a negative case where
the condition depends on an outer-map parameter (must be rejected — unsound
to hoist).

### `pre_simplify` (Stage 0)

Preparation: first `SSALoopIterators` makes every loop iterator name unique so
no later pattern match / fusion is blocked by incidental name reuse, then the
full `SimplifyPass`: inline nested SDFGs, raise control flow,
fuse *states*, scalar→symbol, constant propagation, dead-code/state
elimination, etc. This puts the SDFG into a stable explicit-control-flow form
so the later pattern-matching stages match on a clean, predictable graph.

`SimplifyPass` does **not** perform map fusion: yakup/dev `SIMPLIFY_PASSES`
fuses *states* (`FuseStates`) and inlines nested SDFGs (`InlineSDFGs`), but
contains no `MapFusion`/`FullMapFusion`. Map fusion happens only in Stage 7.
(`LoopToReduce` is not in yakup/dev `SIMPLIFY_PASSES`, so it is never lifted
early — no `skip` needed.)

Structure simplification between steps: `SimplifyPass` (`InlineSDFGs`,
`InlineControlFlowRegions`, `FuseStates`, `Dead{Dataflow,State}Elimination`,
`ArrayElimination`, `ConsolidateEdges`, `EmptyLoopElimination`,
`PruneEmptyConditionalBranches`, …) now runs at the end of every transforming
stage (`prepare_fission`, `maximal_fission`, `reorder_offsets`,
`perfect_loop_nesting`, `normalize`) and at the start of `maximal_fusion`, so
each stage hands the next a clean, flattened graph.

### `_cleanup` (copy/WCR normalization) and `LoopToReduce` re-run points

`_cleanup(loop_to_reduce)` is a reusable composition, not a numbered stage:

1. `PatternMatchAndApplyRepeated([WCRToAugAssign])` — decompose every
   write-conflict resolution into an explicit augmented-assignment subgraph
   (``a = a + b``), the canonical "minimal unit" form. Recomposed by
   `AugAssignToWCR` only as a backend concern, never here.
2. `InsertAssignTaskletsAtMapBoundary` — split map-boundary staging and
   ``other_subset`` ``AccessNode -> AccessNode`` copies into ``_out = _in``
   tasklets.
3. `InsertAssignTaskletsForUnitCopies` (new, `passes/`, general — companion to
   #2) — split a plain ``AccessNode -> AccessNode`` copy into
   ``AN -> (_out = _in) -> AN`` **only** when the moved region is provably a
   single element (``num_elements() == 1`` and every dimension extent ``== 1``,
   on the subset and on ``other_subset`` if present); WCR edges are skipped
   (left to #1). A symbolic extent that is not structurally ``1`` is treated
   as non-unit (conservative).
4. `LoopToReduce` — only when ``loop_to_reduce=True``.

Why #2/#3 here and not inside `NormalizeLoopsAndMaps`: running them in
`prepare_fission` guarantees every later stage sees ``other_subset``-free
copies, so subset-substituting passes never special-case copy memlets.
`NormalizeLoopsAndMaps._create_new_memlet` keeps its override only for the
genuine reason (dace-symbol-correct substitution; the base mis-parses a symbol
named ``S`` as ``sympy.S``), with a thin defensive ``other_subset`` branch for
standalone callers that skip the cleanup.

**`LoopToReduce` re-run points** (``loop_to_reduce=True``): it is sound only
once maximal fission has isolated each accumulator into its own stride-1 loop,
so it must **not** run in `pre_simplify`/`prepare_fission`. A *new* reducible
loop can appear exactly at three points, and it re-runs at each:
`maximal_fission` (loops first isolated — the first sound point),
`perfect_loop_nesting` (parent-nest duplication can expose new inner
accumulators), and `normalize` (canonical run, after IV canonicalization). It
need not re-run after `reorder_offsets` (pure range rewrite — no new
accumulator loops), after `loop_to_map` (would be a map already), or in
`maximal_fusion`.

### Future TODO — map-writes-full-array → reduced array → OpenMP reduction

Pattern: a map writes every element of an array, and that array is then
reduced. This should lower to an OpenMP-style reduction: each core
accumulates into its **own** padded slot (each partial padded to ≥ 64 bytes /
one cache line so distinct cores never share a line — avoids false sharing),
followed by a single small final reduction over the per-core partials. Not yet
a pass; capture as a planned canonicalize/codegen optimization.

### Ordering constraints (must hold)

- Split tasklets before fission (atomic ops → isolatable by `MapFission`);
  recomposed by `TaskletFusion` in Stage 7.
- Untile before reorder (`MapInterchange` rejects inner ranges depending on
  outer params).
- `SSALoopIterators` before fusion / loop→map (avoid `i`-reuse aliasing).
- `SimplifyInductionVariables` before LICM / `LoopToReduce` / loop→map.
- LICM after fission, before loop→map; never hoists a WCR/accumulator update.
- PLN after fission, before fusion.
- `LoopToReduce` strictly after maximal fission, before loop→map; re-run at
  the three points where a fresh reducible loop can appear (see the
  ``_cleanup`` / re-run-points section above).
- `WCRToAugAssign` (in `_cleanup`) decomposes WCR everywhere; the inverse
  `AugAssignToWCR` is a backend concern and never runs in this pipeline.
- `hoist_if` is terminal (last stage), currently a no-op.

## References

- Muchnick, *Advanced Compiler Design and Implementation* §13.2 (LICM),
  §14.1–14.2 (induction variables / strength reduction / LFTR).
- Polly SCoP detection (Grosser et al., PPL 2012) — fission separates a fused
  loop into maximally parallelizable SCoPs.
- LLVM `licm` / `indvars` test scenarios are ported as `test_llvm_*` DaCe tests.

## Testing

- Per-pass tests already exist under `tests/passes/` — extend, not replace.
- `tests/canonicalize/` (mirrors `tests/autooptimize/`): numerical-equivalence
  (deep-copy pre-canonicalization, compile both, `np.allclose`), a SCoP
  fission→parallelization test from the Polly paper, plus a ported
  Polybench/Pluto corpus.
- `yakup-env` needs `-fopenmp` in the CPU compiler args to compile generated
  SDFGs (otherwise `undefined symbol: omp_get_max_threads`).

## Change log

- *init*: subpackage scaffolded as `pipeline.py`; `CanonicalizationPipeline`
  + `canonicalize()` compose existing yakup/dev passes, applied once in order.
  Stage 1 (`Untile`) and the Stage-2 loop-fission step are TODOs (their code
  lives outside this subpackage). Stage 8 (`hoist_if`) is a no-op for now.
  Verified end-to-end on an `axpy` SDFG (numerically exact).
- *scop test*: added `tests/canonicalize/scop_fission_parallelization_test.py`
  (Polly PPL'12 Listing 1). Surfaced and fixed two real defects in reused
  passes (outside this dir): `OffsetLoopsAndMaps` (`int.subs` /
  ``_apply`` traversal; its suite 2→5 passing, 0 regressions) and `LoopToMap`
  (now rejects a loop with multiple distinct write subscripts to one
  container, e.g. ``A[5*i]``/``A[3*i]``, which overlap across iterations;
  reproducer at `tests/transformations/loop_to_map_overlapping_writes_test.py`,
  existing `loop_to_map_test.py` still fully green). SCoP test now passes both
  numerical-correctness and fission-enables-parallelism.
