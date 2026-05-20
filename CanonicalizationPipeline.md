# SDFG Canonicalization Pipeline — Design Document

> Status: design / plan. Target branch: **`yakup/dev`**. Repo:
> `spcl/dace` at `/home/primrose/Work/dace`.

## Table of contents

1. [Motivation](#1-motivation)
2. [Goal](#2-goal)
3. [What already exists on `yakup/dev`](#3-what-already-exists-on-yakupdev)
4. [Prior art and implementation references](#4-prior-art-and-implementation-references)
5. [The pipeline](#5-the-pipeline)
6. [Ordering constraints](#6-ordering-constraints)
7. [New components](#7-new-components)
8. [Testing and verification](#8-testing-and-verification)
9. [Incremental landing order](#9-incremental-landing-order)
10. [Code guidelines](#10-code-guidelines)

---

## 1. Motivation

Downstream optimization in DaCe — vectorization, GPU codegen, the HLFIR/ICON
velocity work — keeps re-deriving the same loop/map shapes ad hoc because there
is **no canonical form**. The same computation reaches the optimizer tiled,
fused, with arbitrary iterator names, permuted nests, and conditionals buried
inside maps. Pattern transformations therefore fire inconsistently depending on
incidental structure.

## 2. Goal

A single deterministic **`CanonicalizePass`** (Trümper et al. canonicalization
style) that rewrites any SDFG into one canonical form:

> blow everything apart to minimal computation units → order loop nests
> canonically → normalize iterators / induction variables / invariants →
> recompose by maximal fusion → hoist conditionals out of maps.

A canonical SDFG lets every later pass (fusion, vectorization, scheduling,
equivalence checking) see exactly one shape per computation.

The pass is **standalone** (analogous to `SimplifyPass`); it does not change the
default behavior of `sdfg.simplify()` or auto-optimization.

## 3. What already exists on `yakup/dev`

Almost every stage pass is already implemented on `yakup/dev`, each with a test.
Verified by inspecting the branch directly.

| Pipeline role | File (`dace/transformation/…`) | Class | Test |
|---|---|---|---|
| Loop-invariant code motion | `passes/loop_invariant_code_motion.py` (825 ln) | `LoopInvariantCodeMotion` | `tests/passes/loop_invariant_code_motion_test.py` |
| Induction-variable canon | `passes/simplify_induction_variables.py` (290 ln) | `SimplifyInductionVariables` | `tests/passes/simplify_induction_variables_test.py` |
| Unique loop iterators | `passes/unique_loop_iterators.py` (87 ln) | `UniqueLoopIterators` | `tests/passes/unique_loop_iterators_test.py` |
| Min-offset reorder | `passes/offset_loop_and_maps.py` (371 ln) | `OffsetLoopsAndMaps` | `tests/passes/offset_loop_and_maps_test.py` |
| Perfect loop nesting | `dataflow/perf_loop_nesting.py` (238 ln) | `PerfLoopNesting` | `tests/transformations/perf_loop_nesting_test.py` |
| Hoist invariant `IF` up | `interstate/move_loop_invariant_if_up.py` (456 ln) | `MoveLoopInvariantIfUp` | — |
| Split tasklets | `passes/split_tasklets.py` (extended) | `SplitTasklets` | `tests/passes/split_tasklets_test.py` |
| Cleanup | `passes/remove_loops_with_empty_bodies.py`, `passes/remove_redundant_assignment_tasklets.py`, `passes/remove_redundant_copy_tasklets.py` | — | — |
| Loop analysis (extended) | `passes/analysis/loop_analysis.py` (+240) | — | — |

All carry `@explicit_cf_compatible` (modern `LoopRegion`/`ConditionalBlock` IR).

**LICM invariance definition** (from the pass's own module docstring) — a tasklet
is loop-invariant iff:

1. no free symbol is a loop induction variable or a symbol written on an
   interstate edge inside the loop;
2. every in-edge memlet subset is IV-free and its source `AccessNode`'s data
   container is not written anywhere inside the loop (conservative alias check);
3. it is unconditionally executed, has no side effects, no WCR output, and is
   not an integer div/mod by a possibly-zero invariant divisor.

It hoists from a `LoopRegion` body to a preheader state, and from a `Map` scope
to the enclosing `SDFGState`. **This is the pass to harden directly** — review
and extend it; do not reimplement.

### Still missing (the only genuinely new code)

1. **`Untile`** — a transformation that undoes tiling (no untile/detile file
   exists on `yakup/dev`).
2. **`CanonicalizePass`** — the pipeline composing the passes above (they are not
   yet composed into a canonical pipeline).

`MapFission`, `MapExpansion`, `MapInterchange`, `FullMapFusion`, `LoopToMap`,
`LoopToReduce`, `TaskletFusion`, `TrivialTaskletElimination`, and
`PatternMatchAndApplyRepeated` are all available and used as-is.

> Because `PerfLoopNesting` and `MoveLoopInvariantIfUp` already exist on
> `yakup/dev`, the earlier plan to `git merge` PRs #2350 / #2351 is **dropped**.

## 4. Prior art and implementation references

| Step | LLVM | Polly | Pluto |
|---|---|---|---|
| unique iterators / IV canon | `indvars`, `loop-simplify`, `lcssa`, `loop-rotation` | `polly-canonicalize` | implicit (poly domain) |
| loop-invariant code motion | `licm` | `licm` (canon set) | implicit in poly model |
| maximal fission | `loop-distribute` | isl SCC split | loop distribution |
| reorder / min offsets | `loop-interchange` | isl band reorder | cost model (min reuse dist) |
| perfect loop nesting | implicit | band sinking | perfectly-nested precond |
| loop→reduce | `loop-idiom` | reduction detect | reduction handling |
| loop→map | `LoopVectorize` / parallel | `polly-parallel` | `--parallel` |
| maximal fusion | `loop-fusion` | isl fusion | `--fuse` maxfuse |
| undo tiling | (n/a) | reschedule from poly | tiling is the output |

The pipeline is the SDFG analogue of `polly-canonicalize` followed by an isl
schedule (distribute → reorder → fuse). LLVM keeps these as discrete passes;
Pluto fuses them in one polyhedral cost model. We follow the discrete-pass
structure because DaCe's transformation framework is pattern-based. Undoing
tiling has no LLVM/Pluto analog (tiling is *their* result), so `Untile` is novel.

**Implementation reference ("red book"):** Muchnick, *Advanced Compiler Design
and Implementation* — §13.2 loop-invariant code motion (invariance lattice,
preheader insertion, speculation safety); §14.1 induction-variable detection;
§14.1.2 strength reduction; §14.2 linear-function test replacement / dead-IV
elimination. These give the canonical algorithms to validate the `yakup/dev`
LICM and `SimplifyInductionVariables` against.

> "Red book" is taken as Muchnick — the standard algorithmic reference for these
> two passes. If the red Dragon Book (Aho/Sethi/Ullman, 1986) was meant instead,
> note that its loop-optimization chapter is shallower for LICM/IV; flag to
> switch references.

## 5. The pipeline

`CanonicalizePass` is a `FixedPointPipeline` over Stages 1–7. Stage 0 runs once
at entry; Stage 8 runs once at exit (it deliberately de-structures, so it must
stay out of the fixed point). An iteration cap is exposed as a `Property` to
prevent oscillation. Pattern transformations are wrapped as passes via
`PatternMatchAndApplyRepeated` (the mechanism `FullMapFusion` uses,
`passes/full_map_fusion.py:214`).

| # | Stage | Concrete pass (existing on `yakup/dev` unless **NEW**) | Fixed point? |
|---|---|---|---|
| 0 | Pre-normalize | `SimplifyPass(skip={'LoopToReduce'})` | yes |
| 1 | **Undo tiling** | `PatternMatchAndApplyRepeated([Untile()])` — **NEW** | yes |
| 1b | Split tasklets | `SplitTasklets()` | yes |
| 2 | Maximal fission | FP: `MapExpansion` → `MapFission` → loop-fission CFG step → interleaved `SimplifyPass(skip LoopToReduce, InlineSDFGs)` | yes |
| 3 | Reorder (min offsets) | `OffsetLoopsAndMaps()` | yes |
| 4 | Perfect loop nesting | `PatternMatchAndApplyRepeated([PerfLoopNesting()])` + `SimplifyPass` | yes |
| 5 | Normalize | FP, in order: `UniqueLoopIterators` → `SimplifyInductionVariables` → `LoopInvariantCodeMotion` → `LoopToReduce` | yes |
| 6 | Loop→map | `PatternMatchAndApplyRepeated([LoopToMap()])` | yes |
| 7 | Maximal fusion | `SimplifyPass` → `FullMapFusion(vertical, horizontal)` (+ optional `SubgraphFusion`) → `TaskletFusion` + `TrivialTaskletElimination` | yes |
| 8 | Hoist `IF` above maps | `PatternMatchAndApplyRepeated([MoveLoopInvariantIfUp()])` (terminal) | yes (idempotent, hoist-only) |

Cleanup passes (`remove_loops_with_empty_bodies`,
`remove_redundant_assignment_tasklets`, `remove_redundant_copy_tasklets`) run
inside the Stage-2 / Stage-7 interleaved simplifies.

## 6. Ordering constraints

These are correctness requirements, not preferences.

- **Split tasklets after Untile, before fission.** `SplitTasklets` decomposes
  compound tasklets into single-op SSA tasklets so `MapFission` can isolate each
  atomic computation into its own minimal map. It correctly refuses tasklets
  with more than one output or mixed-precision inputs. Recomposed by
  `TaskletFusion` + `TrivialTaskletElimination` in Stage 7 so the canonical
  output is not needlessly fragmented.
- **Untile before reorder.** `MapInterchange.can_be_applied` rejects an inner
  range that depends on outer parameters; a tiled `ii = i:i+T` depends on `i`.
- **Simplify between fission and fusion is mandatory.** `MapFission` leaves split
  transients; `FullMapFusion`'s `FindSingleUseData` needs clean descriptors or
  fusion under-fires.
- **`UniqueLoopIterators` before fusion and loop→map.** Independently fissioned
  nests reusing `i` block fusion or cause aliasing on fuse.
- **`SimplifyInductionVariables` before LICM, `LoopToReduce`, loop→map.** It
  folds affine-derived IVs into closed form over the primary IV. Before LICM (a
  non-canonical derived IV looks variant and blocks hoisting); before
  `LoopToReduce` / `LoopToMap` (both need `loop_analysis` to see a single affine
  IV; `LoopToMap.can_be_applied` rejects non-affine / sympy bounds). LLVM
  analogue: `indvars`.
- **LICM after Untile + fission + IV-canon, before loop→map; re-run in the fixed
  point.** Hoist only on minimal post-fission bodies (invariance is precise and
  never crosses a soon-removed tile boundary); never hoist a WCR / accumulator
  update. Stays inside the Stage-5 fixed point so invariants newly exposed by
  Stage-7 fusion (via the outer fixed point) are hoisted too. Verify the
  Map-scope hoist path enforces the same WCR/side-effect guards as the
  LoopRegion path.
- **Perfect loop nesting after fission, before fusion.** PLN needs the
  parent-map body to be one `NestedSDFG` with ≥2 children — the shape that
  fission + inline produces. Post-fusion it fights the fusion just performed.
- **`LoopToReduce` strictly after maximal fission.** It must never fire in Stage
  0 or in any Stage-2 interleaved `SimplifyPass` (skipped there). It runs only in
  Stage 5, after each accumulator loop is fissioned into its own stride-1
  `LoopRegion` (its `_extract` requires that). Running it after Stage 6 is too
  late (the loop is already a map). It stays before loop→map.
- **Stage 8 is strictly terminal and hoist-only.** `MoveLoopInvariantIfUp` only
  raises conditionals; keep it out of the Stage-1–7 fixed point so it cannot
  ping-pong with any inward-moving step.
- New code must return `True` from `annotates_memlets()` and re-propagate
  memlets (cf. `MapTiling` `tiling.py:38`, `MapInterchange`
  `map_interchange.py:133`).

## 7. New components

Only two pieces of genuinely new code.

| Component | Path | Class |
|---|---|---|
| Undo tiling | `dace/transformation/dataflow/untile.py` | `Untile(SingleStateTransformation)` |
| Pipeline | `dace/transformation/passes/canonicalize.py` | `CanonicalizePass(FixedPointPipeline)` |

Register `Untile` in `dace/transformation/dataflow/__init__.py`. Reference
`passes/full_map_fusion.py` for the pass-wrapping pattern and `pass_pipeline.py`
for `FixedPointPipeline` / `depends_on` / `modifies` semantics.

### `Untile`

Matches a two-level map nest: an outer strided map `p = lo:hi:T` whose sole
child is an inner map `q = p : min(p+T, hi)` — the `i, ii` shape `MapTiling`
emits (`dace/transformation/dataflow/tiling.py`).

- `can_be_applied`: the inner range is exactly the outer parameter's tile
  window, the inner map is the only child scope, and body subscripts use `q`
  (not `p` independently) so the merge is sound.
- `apply`: replace both maps with a single map `r = lo:hi:1`, substitute `q → r`
  (and `p → r` where `p` appears as the tile base) in body memlets and tasklets,
  then re-propagate.
- Handles the `divides_evenly=False` `min(p+T, hi)` remainder form. Generalizes
  to N tiled dimensions by repeated application (fixed point). It round-trips
  with `MapTiling` — that round-trip is its primary test.

> The min-offset reorder is already implemented as `OffsetLoopsAndMaps`. If its
> heuristic needs tuning, tune it in that file; do not add a parallel pass.
> Before wiring it into Stage 3, confirm its ordering key is a pure function of
> the access pattern (idempotent and fixed-point-safe).

## 8. Testing and verification

Per-pass tests already exist on `yakup/dev` — **extend, do not replace**.

### Ported scenarios (LLVM → DaCe SDFG)

Re-express each as a small `@dace.program` kernel; assert the pass produces the
expected hoist/fold and numerical equivalence against the un-canonicalized SDFG.
Keep one DaCe test per scenario, named after the LLVM file for traceability
(`.ll` is not portable verbatim — the scenario is).

**From `llvm/test/Transforms/LICM/` → `loop_invariant_code_motion_test.py`:**

- hoist invariant load / invariant binop / invariant address computation;
- hoist a side-effect-free "call" (DaCe: invariant nested SDFG / pure tasklet);
- **must NOT hoist:** aliased store blocks load hoist (container written in
  loop); conditionally executed op (speculation-unsafe); WCR / reduction update;
  integer div/mod by a possibly-zero invariant divisor;
- multiple-exit loop; nested loop (hoist to the correct outer preheader);
- Map-scope hoist (LICM's second mode: `Map` → enclosing state).

**From `llvm/test/Transforms/IndVarSimplify/` →
`simplify_induction_variables_test.py`:**

- canonicalize primary IV (non-zero start / non-unit / negative stride);
- fold derived/secondary IV `k = c*i + d` to closed form;
- redundant-IV elimination; exit-value replacement;
- IV-comparison normalization feeding `LoopToMap` bound detection.

### Pipeline-level

- **Numerical-equivalence harness** (`tests/passes/canonicalize_test.py`): deep
  copy the SDFG before canonicalization, compile both, run on identical random
  inputs, assert `np.allclose`. Reuse the compile+run+compare pattern from the
  existing pass tests.
- **Ported corpus** (parametrized): curated Polybench from `samples/polybench/`
  (gemm, 2mm, 3mm, jacobi-2d, seidel-2d, heat-3d, lu, fdtd-2d, plus a
  `MapTiling`-tiled matmul for `Untile`) plus Pluto `fusion1-11` /
  `intratileopt*` / tiled kernels — each targeting a specific stage.
- **Idempotence:** running `canonicalize` twice no-ops the second time
  (validates fixed-point termination and deterministic offset/IV keys).
- **Per-stage isolation** tests so regressions localize.
- **Non-regression gate:** all existing `yakup/dev` pass and transformation
  suites stay green.
- Run sweeps with `pytest -n 8`.

## 9. Incremental landing order

1. **Harden `LoopInvariantCodeMotion`** (user priority): read the 825-line pass
   against Muchnick §13.2; extend its test with the ported LLVM `LICM`
   scenarios; fix gaps. Standalone, no pipeline needed.
2. Audit/extend `SimplifyInductionVariables` against Muchnick §14.1–14.2 and the
   ported `IndVarSimplify` scenarios.
3. Sanity-audit `UniqueLoopIterators`, `OffsetLoopsAndMaps`, `PerfLoopNesting`,
   `MoveLoopInvariantIfUp` (run existing tests; confirm `annotates_memlets`,
   idempotence, `explicit_cf_compatible`).
4. `Untile` + `tests/transformations/untile_test.py` (inverse of `MapTiling`).
5. Loop-fission CFG step for Stage 2 (map fission already exists).
6. Assemble `CanonicalizePass` (Stages 0–8, wire `depends_on` / `modifies`,
   iteration cap) + `tests/passes/canonicalize_test.py`.
7. *(Optional)* `SDFG.canonicalize()` convenience entry point.

## 10. Code guidelines

See `CODE_GUIDELINES.md` (same directory). Summary: follow
`CONTRIBUTING.md`; Sphinx docstrings required; type-hint parameters and
non-`None` returns only; `ALL_CAPS` module constants; no banner comments; reuse
over new code (only `Untile` and `CanonicalizePass` are new — every other pass is
an existing `yakup/dev` pass used as-is or hardened in place, never forked); no
`Co-Authored-By` Claude trailer in commits.
