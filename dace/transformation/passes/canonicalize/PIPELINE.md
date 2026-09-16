# Canonicalization pipeline: phased order

Two recipes. Same knobs, same pass classes.

- `legacy`: `_build_stages` in `pipeline.py`. 181 units, 50 labels. Default.
- `phased`: `build_phased_stages` in `phased_pipeline.py`. 138 units, 4 labels, one per paper phase.

Pick one: `canonicalize(sdfg, order='phased')`, `CanonicalizationPipeline(order='phased')`, or
`DACE_optimizer_canonicalize_order=phased`.

Both keep: structured control flow required, canonical = parallel, no tiling, units never validate.

## Phases = paper phases

Paper: Fig. `fig:phases` (Normalize Statements, Lift Semantics, Derive Parallelism; Specialize is a
separate call) and appendix "The Schedule in Detail" (recomposition closes the list).

| Label | Paper sentence it implements | Steps, in order |
|---|---|---|
| `normalize_statements` | "It normalizes statements and expressions, substitutes induction variables, distributes statements where dependences permit, and makes loop nests perfect where possible." / "normalization untiles and rerolls such loops." / "normalization itself lowers the parallel loops of the input, so that every loop is derived." | raise CF, one spelling, lower maps to loops; reroll; untile; unroll tiny loops; statement fixpoint; perfect nesting; stride permutation; `SimplifyPass` once |
| `lift_semantics` | "Semantic lifting recognizes operators before parallelism is derived." | transpose, einsum, reduce, scan, argmax, conditional reduce, stream compaction, symmetrize, copy/fill, loop-carried reduction |
| `derive_parallelism` | "Parallelism is derived in three levels ... with wavefront skewing where every loop of a nest carries a dependence." / "Turn accumulators into reduction Parallel Do-All Loops, derive again, and emit guarded loops." | Loop2Map; rescue (rotate, peel, break anti-dependence) of the refused loops; Loop2Map retry; guarded loops; reduction loops; scatter guard; wavefront skew |
| `recompose` | "Recompose: coalesce states, fuse loops, interchange, hoist guards, and run the final simplification and dead-code passes to a fixed point." | WCR re-normalize; fixpoint of (map permutation, fixpoint of (MapCollapse x FuseMaps x FuseStates), condition fusion); hoist guards; final reclaim, symbols, assumption guard |

### Statement fixpoint (inside `normalize_statements`)

Round, capped at 3, another round only if invariant motion (`LoopInvariantCodeMotion`,
`MoveLoopInvariantIfUp`) changed the graph:

1. IV substitution: scalar-to-symbol, simplify, hoist IV updates, substitute, recover index subsets.
2. Split: forward store to load, `SplitStatements`.
3. Privatize: scalars, arrays, constant-index slots, scratch buffers.
4. Invariant motion: LICM, then invariant-if hoisting.
5. WCR normalization.
6. Dead dataflow: dead dataflow, dead arrays, dead carried stores.

Why this order: LICM hoists whole statements, so split first (`a[i] = x*y + b[i]`). Privatize before
LICM, so a split temporary rewritten every iteration is not hoisted. LICM feeds split back (unswitched
ifs, invariant factors from IV substitution), hence the fixpoint. Paper: "Induction variables are
substituted before invariant motion"; "Invariant motion itself runs after distribution".

Measured: this round and the older IV-first round (IVS, LICM, dead code, WCR, privatize, split; cap 2)
give identical parallel structure on all 321 kernels. This one also matches legacy on tsvc `s244`
(one map, not two).

## StructuralCleanup

Idempotent block, not a phase. Runs where the next step reads state structure:

- after lowering: MapToForLoop mints pre/post states and nested bodies.
- after the statement fixpoint: split mints states.
- end of `lift_semantics`: lifts splice states, Loop2Map reads bodies.
- after the reduction Loop2Map: body inlining.
- head and inner fixpoint of `recompose`: two maps fuse only when they share a state.
- end of `recompose`: last tidy.

## Passes outside their phase, with the kernel that needs it

Corpus: tsvc, tsvc_2_5, polybench, npbench, llr-focus40 = 321 kernels, canonicalize only.

| Pass, position | Without it |
|---|---|
| `LoopToSymm`, `LoopToRankKUpdate` in `normalize_statements`, before lowering | polybench `symm`, `syrk`, `syr2k` lose Symm/Syrk/Syr2k (their matchers need the npbench slice skeleton; split, lowering and nesting remove it) |
| `LiftInv` in `normalize_statements` | library-node rewrite `Solve(A, eye)` to `Inv`, needs the eye map before lowering |
| `PrivatizeScatterReduction`, `NormalizeWCR` before lowering | npbench `azimint_naive` computes WRONG values when `NormalizeWCR` moves after lowering |
| dead dataflow again in the rescue step | tsvc_2 `s252`: rotation leaves a dead store, loop stays sequential |
| copy/fill lift right after the Loop2Map retry | tsvc_2_5 `scan_strided_sym`, `fission_dep_sym_offset`: later cleanup fuses a second writer into the fill map's state, the matcher refuses |
| WCR re-normalization at the head of `recompose` | tsvc `s221`: reduction Loop2Map mints a map-exit WCR, fusion refuses, two maps stay |
| `MoveMapInvariantIfUp` after fusion | tsvc_2_5 `config_select_branch` loses its two-map specialization (needs maps) |
| copy/fill and `LiftEinsum` after fusion | tsvc `va` loses Memcpy (loop matcher needs an `out = in` tasklet that `EliminateTrivialTasklets` removed); npbench `nbody` loses an einsum |

## Pass-to-phase table

| Pass | Phase, step |
|---|---|
| ControlFlowRaising, RequireStructuredControlFlow, RemoveViews, SupplyNumThreads, CollapseNoOpCast, RewriteModuloToPyMod, NormalizeNegativeStride, UniqueLoopIterators, ContinueToCondition, SimplifyPass, ConvertLengthOneArraysToScalars | normalize_statements, ingest |
| RevertNonReductionWCR, MapToForLoop, PruneConnectors, InlineSDFGs, scalar-slice folds, EliminateTrivialTasklets | normalize_statements, lower |
| RerollUnrolledLoops | normalize_statements, reroll |
| UntileLoops, FuseConsecutiveLoops, ShortLoopUnroll, SymbolSSA | normalize_statements, untile |
| ScalarToSymbolPromotion, HoistInductionVariableUpdates, InductionVariableSubstitution, PropagateIndexSubsets, RemoveUnusedSymbols, PropagateAndPrune | normalize_statements, fixpoint 1 |
| ForwardStoreToLoad, SplitStatements | normalize_statements, fixpoint 2 |
| PrivatizeScalars, PrivatizeArrays, PromoteConstantIndexAccess, BufferExpansion | normalize_statements, fixpoint 3 |
| LoopInvariantCodeMotion, MoveLoopInvariantIfUp | normalize_statements, fixpoint 4 |
| RevertNonReductionWCR, NormalizeWCR | normalize_statements, fixpoint 5 |
| DeadDataflowElimination, ArrayElimination, DeadCarriedStoreElimination | normalize_statements, fixpoint 6 |
| MaterializeLoopExitSymbols | normalize_statements, after fixpoint |
| DistributeProducerConsumerLoop, MoveIfIntoLoop, CascadeInterstateEdgeAssignmentsUp, PerfectLoopNesting, TrivialLoopElimination | normalize_statements, perfect nesting |
| LoopStridePermutation, NormalizeLoopAndMapOrigin (knob) | normalize_statements, permutation |
| LoopToSymm, LiftInv, LoopToRankKUpdate, PrivatizeScatterReduction, NormalizeWCR | before lowering (see above) |
| LoopToSymmetrize, AssignmentAndCopyKernelToMemsetAndMemcpy, LoopToTranspose, LoopToEinsum, LoopToReduce, LiftPreprocess, LoopToScan, ArgMaxLift, LoopToConditionalReduce, LoopToStreamCompaction, LiftLoopCarriedReduction | lift_semantics |
| PropagateMemlets, ParallelizeLoops (LoopToMap) | derive_parallelism, Loop2Map and retry |
| LoopCarriedRotationSubstitution, BestEffortLoopPeeling, BreakAntiDependence | derive_parallelism, rescue |
| ParallelizeUnderConstraint, ScatterToGuardedMaps | derive_parallelism, guarded |
| FuseChainedScalarReductions, PinCarriedTopLevelLoops, AccumulatorCopyChainToWCR, RetargetWCRAccumulator | derive_parallelism, reductions |
| PerfLoopNesting (gpu), ReconstructWavefrontNest (knob), ReorderStateForLoopFusion (gpu), FuseLoops, WavefrontSkew | derive_parallelism, wavefront |
| NormalizeWCRSource, EmptyStateElimination, TrivialMapElimination, EmptyLoopElimination, MoveIfIntoMap | recompose, head |
| ReverseMapTraversal, MinimizeStridePermutation, MoveLoopIntoMapGated, ConditionFusion, LiftTrivialIf, NormalizeMapBody | recompose, outer fixpoint |
| MapCollapse, DistributeTaskletIntoMap, FuseMaps, MapReduceFusion, MapWCRFusion, StructuralCleanup | recompose, inner fixpoint |
| MoveMapInvariantIfUp | recompose, guard hoist |
| InsertAssignTaskletsAtMapBoundary, LiftEinsum | lift_semantics, after fusion |
| RelaxIntegerPowers, RedundantArray, RematerializeDerivedTemporaries, PruneEmptyConditionalBranches, SymbolDedup, OptionalArrayInference, PruneUnreferencedTransients, InlineControlFlowRegions, AssumeSymbolConstraints, NormalizeFloorDivision, AnnotateLoopKinds | recompose, final |

Not in either recipe: `PartitionGuardedLoop`, `SinkStateIntoLoop`, `PrivatizeReductionAccumulator`
(no caller). `HoistLoopRangeCalls`, `ShrinkMapLocalTransients` run in `finalize_for_target`.
