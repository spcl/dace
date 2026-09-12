# Canonicalization — Design

Rewrite any SDFG into one canonical form, so later passes (fusion, vectorization,
scheduling, equivalence checks) see one shape per computation instead of many incidental
ones — tiled / fused / arbitrary iterator names / permuted nests / conditionals buried
inside maps.

Recipe (Trümper et al.): blow apart to minimal units → order nests canonically →
normalize iterators, induction variables, invariants → recompose by maximal fusion →
hoist conditionals out.

Stage order lives in `pipeline.py`; read it there. This doc holds only what the code
cannot say.

## Not a `pass_pipeline.Pipeline`

Stages are applied imperatively, once, in order. A `Pipeline` forbids duplicate pass
types, and `Pipeline` / `PatternMatchAndApplyRepeated` are unhashable — both fatal here,
because the same pass must run at several points. Stages that need iteration iterate
internally; the pipeline never re-runs. Every stage always runs: no `skip`.

Passes existing only for this pipeline live here. Reusable passes keep their normal
homes and are composed here, so they stay usable from outside.

## Ordering constraints — the real invariants

- Split tasklets before fission: atomic ops must be isolatable by `MapFission`.
  `TaskletFusion` recomposes at the end.
- Untile before reorder: `MapInterchange` rejects inner ranges depending on outer params.
- `UniqueLoopIterators` before fusion and loop→map, else `i`-reuse aliases.
- `SimplifyInductionVariables` before LICM, `LoopToReduce`, loop→map.
- LICM after fission, before loop→map. Never hoists a WCR/accumulator update.
- `PerfLoopNesting` after fission, before fusion.
- `LoopToReduce` strictly after maximal fission — sound only once each accumulator sits
  in its own stride-1 loop. So never in `pre_simplify` / `prepare_fission`.
- `WCRToAugAssign` decomposes WCR into explicit `a = a + b` everywhere. The inverse
  `AugAssignToWCR` is a backend concern and never runs here.
- `SimplifyPass` does not fuse maps — it fuses *states* and inlines NSDFGs. Map fusion
  happens only in the final fusion stage.
- `hoist_if` is terminal.

## `LoopToReduce` re-run points

A fresh reducible loop can appear at exactly three places, so it re-runs at each:
`maximal_fission` (loops first isolated — the first sound point), `perfect_loop_nesting`
(parent-nest duplication exposes new inner accumulators), `normalize` (after IV
canonicalization). Not needed after `reorder_offsets` (pure range rewrite), after
`loop_to_map` (already a map), nor in `maximal_fusion`.

## `PerfLoopNesting` needs no NestedSDFG

Duplicate the enclosing parent nest once per independent inner sibling map, in the same
state, respecting data dependencies — dependent siblings must not be split:

    map1 / map2 / {map3, map4}   ==>   map1 / map2 / map3   +   map1 / map2 / map4

The NSDFG-wrapped form still works, but the same-state inlined form is what the frontend
actually emits at `simplify=True` (`parent-map body = [MapExit, MapEntry, MapEntry, …]`).

## `MoveIfIntoMap`

Matched only inner maps whose body is a single `NestedSDFG`, so it never engaged on
frontend output (plain tasklet subgraphs). Non-NSDFG inner-map bodies are nested into one
NSDFG before guard injection. Hoisting is unsound when the condition depends on an
outer-map parameter — rejected.

## Copy normalization (`_cleanup`)

Map-boundary staging copies, and `AccessNode → AccessNode` copies that are provably a
single element, are split into `_out = _in` tasklets. WCR edges are left to
`WCRToAugAssign`. A symbolic extent that is not structurally `1` counts as non-unit
(conservative).

Runs early (`prepare_fission`) so every later stage sees `other_subset`-free copies and
no subset-substituting pass has to special-case copy memlets.
`NormalizeLoopsAndMaps._create_new_memlet` keeps its override for one genuine reason:
dace-symbol-correct substitution — the base mis-parses a symbol named `S` as `sympy.S`.
