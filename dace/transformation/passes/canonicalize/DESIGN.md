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
| 2 | `maximal_fission` | `PatternMatchAndApplyRepeated([MapExpansion, MapFission])` → `SimplifyPass()` | partial (loop-fission step **TODO**) |
| 3 | `reorder_offsets` | `OffsetLoopsAndMaps(0, 0, convert_leq_to_lt, normalize_loops)` | wired |
| 4 | `perfect_loop_nesting` | `MoveIfIntoMap` → `MapExpansion` (= "map uncollapse", `map[i,j]`→`map[i];map[j]`) → `PerfLoopNesting` → `MapCollapse` (re-collapse maximally) → `MinimizeStridePermutation` (permute to minimize strides) → `SimplifyPass()` | being generalized |
| 5 | `normalize` | `SSALoopIterators` → `SimplifyInductionVariables` → `LoopInvariantCodeMotion` → `LoopToReduce` | wired |
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

### Ordering constraints (must hold)

- Split tasklets before fission (atomic ops → isolatable by `MapFission`);
  recomposed by `TaskletFusion` in Stage 7.
- Untile before reorder (`MapInterchange` rejects inner ranges depending on
  outer params).
- `SSALoopIterators` before fusion / loop→map (avoid `i`-reuse aliasing).
- `SimplifyInductionVariables` before LICM / `LoopToReduce` / loop→map.
- LICM after fission, before loop→map; never hoists a WCR/accumulator update.
- PLN after fission, before fusion.
- `LoopToReduce` strictly after maximal fission; before loop→map.
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
