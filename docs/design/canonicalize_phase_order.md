# Canonicalization phase order: design and handoff

Status 2026-09-16: PARKED. Branch `new-canon`, worktree `~/.cache/wt-dace-new-canon`. Not merged, not pushed.
Resume trigger: "continue working on improved ordering for canonicalization".

## Goal

Replace `_build_stages` in `dace/transformation/passes/canonicalize/pipeline.py` (181 units, 50 labels,
same passes re-run everywhere) with one ordered list of phases:

- phases in the paper's order (mpr-paper `ics25template/canonicalization.tex`, Fig. `fig:phases`,
  appendix `schedule_detail.tex`): Normalize Statements, Lift Semantics, Derive Parallelism, then
  recomposition (fusion). Specialize is a separate call.
- no phase mixing: a pass lives in exactly one phase.
- no pass runs twice. A fixpoint sub-phase that iterates a set of passes is allowed.
- the between-phase structural cleanup may run anywhere (it is not a phase).
- same parallelism as legacy on every kernel (0 worse), numerics correct.

## User-decided order and placements

1. NORMALIZE: reroll -> untile -> IV substitution -> LICM/hoisting -> WCR normalization -> dead dataflow
   -> privatization -> split statements -> perfect loop nesting (when valid) -> min-stride permutation;
   then `SimplifyPass` once (check it does not undo normalization).
   - Later refinement: statement fixpoint IV subst -> split -> privatize -> LICM (statement, then
     invariant if) -> DDE, until no change, cap 3. Split before LICM (LICM hoists whole statements);
     privatize before LICM (do not hoist a per-iteration temporary).
   - Privatization (ScalarFission/ArrayFission, privatize_reduction/scatter) is normalization.
   - WCR normalization (normalize_wcr, revert_nonreduction_wcr, normalize_reduction) is normalization.
   - LICM/hoisting (licm, lift_inv, hoist_guards, hoist_iv_updates, hoist_loop_range_calls) is normalization.
   - Dead dataflow (DDE, ArrayElimination, dead carried store) is normalization.
   - `LiftInv` (library-node rewrite Solve(A, eye) -> Inv) may stay at the front.
2. LIFT: scan, einsum, reduction, symm, syrk, syr2k, transpose, copy/fill, argmax, stream compaction,
   conditional reduce. symm/syrk/syr2k/histogram matchers must match the NORMALIZED form
   (not the npbench slice skeleton), from both slice and scalar-loop spellings.
3. DERIVE: Loop2Map -> rescue (peel, rotate, break anti-dependence) only on loops that failed ->
   Loop2Map retry -> wavefront. reduction_to_wcr_map belongs with Loop2Map.
4. FUSE: (MapFusion x MapCollapse x FuseStates) fixpoint, inside an outer fixpoint.

## Implementation state

Switch: `canonicalize(sdfg, order=...)`, `CanonicalizationPipeline(order=...)`, or config
`optimizer.canonicalize_order` (env `DACE_optimizer_canonicalize_order`). Default `legacy`.

| order | builder | state |
|---|---|---|
| `legacy` | `pipeline._build_stages` | unchanged default |
| `phased` | `phased_pipeline.build_phased_stages` | paper phases, parity reached, passes still repeat |
| `single` | `single_occurrence_pipeline.build_single_occurrence_stages` | WIP, each pass once, 17 kernels worse |

`phased_pipeline.PhaseFixpoint(name, factory, max_rounds, drivers=())`: runs fresh units per round until no
unit reports a change (or, with `drivers`, until none of the named classes changed). `PIPELINE.md` in the
package documents the `phased` recipe, the paper sentence per phase and the pass-to-phase table.

Commits on `new-canon` (base `origin/extended` fdfe4d0cb):

- 946408e7e LoopToMap: declare body access nodes joined only by ordering edges (was KeyError on tsvc_2 s252).
  Reproducer `tests/transformations/interstate/loop_to_map_test.py::test_body_node_joined_only_by_ordering_edge`.
- d714dfc97 `order` knob + `phased_pipeline.py`.
- 895ef367f privatize stages report None on no change (PropagateAndPrune half reverted in eacdebefc).
- 4966b171d paper phase names + `PIPELINE.md`.
- eacdebefc restore PropagateAndPrune round report (pinned by `canonicalize_pipeline_stages_test`).
- this commit: `single_occurrence_pipeline.py` (WIP) + this doc.

## Measured results

Corpus = 321 kernels: llr-focus40 (40), tsvc, tsvc_2_5, polybench, npbench (from `tests/corpus`),
canonicalize only, `mp.cpu_params(4)`. Metric per kernel: residual sequential loops
(`loops - inmap - guarded`), maps, lifted (`reduce + scan + libnode`), states.
"worse" = more residual loops, or equal residual and fewer lifts.

| recipe | same | worse | better | other | errors | time (legacy 131s) |
|---|---|---|---|---|---|---|
| `phased` (4966b171d) | 317 | 0 | 1 | 3 | 0 | 108s |
| `single` p1 | 282 | 25 | 5 | 9 | 0 | 92s |
| `single` p2 | 293 | 17 | 4 | 7 | 0 | 112s |

`phased` better: tsvc_2_5 scan_conditional (+1 lift). Other = state counts only (s255, s2251, ext_peel_multi_back).

Numerics, CI driver (`hpcagent_bench.cli run-framework -f dace_cpu_canonicalize -b all@llr-focus40 -p S`):
legacy 40/40 ok+validated, phased 40/40 ok+validated. tsvc/poly/np numerics under phased: NOT RUN
(azimint_hist, azimint_naive spot-checked correct).

Statement fixpoint variants in `phased` (IVS->split->priv->LICM->WCR->DDE cap 3, LICM-driven vs
IVS->LICM->DDE->WCR->priv->split cap 2): identical structure on 321 kernels; the first also matches legacy
on tsvc s244 (1 map, not 2). `SimplifyPass` after normalize: no structure change on 321 kernels; changes
the graph on npbench vadv (FuseStates) and tsvc s31111 (ArrayElimination); 112s with, 140s without.

Tests, `tests/canonicalize` + `tests/passes/canonicalize` + `tests/canonicalize_memlet_tree_test.py`:
- pristine extended, legacy: 1955 passed, 15 failed, 5 xfailed (stopped by --maxfail). The 15: cloudsc staged
  timeout, 10 `canonicalize_perf_corpus` speedup asserts, 2 `libnode_gpu_scope_selection`, 3 tiled-cuda
  `loop_to_symmetrize` config.
- new-canon (before eacdebefc), legacy, those 15 deselected, -n5, interrupted at 68%: 1528 passed,
  5 failed, 5 xfailed. 4 more perf-corpus speedup asserts (shared box) + `test_propagate_and_prune_...
  reports_every_round` (fixed by eacdebefc, file now 11/11 passed).
- phased order test run: NOT RUN.

sw4 `rhs4sg` (`~/.cache/stale-worktree-backup/perf-canon-20260916/scratch-perf/sw4_rhs4sg.sdfg`), apply_pass
calls under a profiler: legacy 1064, phased 1149 (pre-fixpoint-rework snapshot). Clean wall time NOT DONE:
the box was loaded (load 3-9) by other agents.

### Reproduce

Scripts are in `~/.cache/newcanon` (absolute paths inside, scratch only):

```bash
N=~/.cache/newcanon
$N/snap.sh <tag>                        # copy the worktree to $N/snap-<tag>
$N/sweepp.sh <tag>                      # phased sweep of snap-<tag>, 321 kernels -> $N/parity_<tag>.txt
SRC=$N/snap-<tag> $N/sweep.sh <tag> 0/1 legacy,phased llr40 tsvc poly np tsvc25   # both orders
python3 $N/parity.py $N/sweep_v5.jsonl $N/sweep_<tag>.jsonl   # sweep_v5.jsonl = legacy reference rows
$N/recheck.sh out.jsonl phased tsvc:s176_d_single ...          # a few kernels, current worktree
PYTHONPATH=<src> python $N/trace.py tsvc s176_d_single phased --deep [--probe]   # counts after every unit
PYTHONPATH=<src> python $N/dumpat.py tsvc s176_d_single phased derive_parallelism  # graph before a label
PYTHONPATH=<src> python $N/repeats.py StructuralCleanup   # passes at more than one position
$N/run_tests.sh <order> <n> <src> [pytest args]           # canonicalize test dirs under an order
$N/bench/run.sh <order>                                   # CI driver on llr-focus40
PYTHONPATH=<src> python $N/sw4_time.py <order> time|count
```

Env for every run: `OMP_NUM_THREADS=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp PMIX_MCA_gds=hash
UCX_VFS_ENABLE=n HWLOC_COMPONENTS=-gl MPI4PY_RC_INITIALIZE=0 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=`, python
`~/.pyenv/versions/py12` (has xdist and hpcagent_bench).

## `single` recipe (WIP) shape

- normalize_statements: CF raising, views, LiftInv, spelling; fixpoint(cap 3) { PrivatizeScatterReduction,
  NormalizeWCR, SimplifyPass, RevertNonReductionWCR, MapToForLoop, fold slices, NegStride, len-1 scalars,
  trivial tasklets, rail, reroll, untile, unroll, IVS, split, privatize, LICM + if hoist, DDE, distribute,
  PLN, permute }.
- lift_semantics: fixpoint(cap 2) { all lifts }.
- derive_parallelism: fixpoint(cap 3) { L2M, privatize, rail, rotate, peel, break anti-dep (forward reads),
  guarded, reductions to WCR, scatter, wavefront }.
- recompose: WCR source norm, fusion fixpoints, map-invariant if hoist, LiftEinsum, seal.
- rail (between phases): PruneAndInlineNestedSDFGs, CascadeInterstateEdgeAssignmentsUp, UniqueLoopIterators,
  StructuralCleanup.

Fixed during the `single` work (cause, evidence kernel):

- SimplifyPass must see the frontend map form before lowering, and again after scalar promotion:
  llr40 argmax_with_index, compact_threshold_pack, tsvc_2_s3110, tsvc_2_s316 (no single Simplify member
  suffices, `simp_ablate.py`), tsvc_2_s318. Fix: WCR norm + Simplify + lowering inside the statement fixpoint.
- LoopToScan's stride specialization clones loops with duplicate iterator names, Loop2Map refuses them:
  llr40 versioned_distance_update. Fix: UniqueLoopIterators in the rail.

## Open (numbered)

1. `single`: tsvc s176, s1113, s2251, s281 lose maps. Cause (bisect, `s176_bisect.py`):
   AccumulatorCopyChainToWCR must run BEFORE the post-Loop2Map inline; the WIP round inlines first.
   Next step: reorder the derive round to [L2M, rescue, guarded, reductions, rail] and resweep.
2. `single`: tsvc_2 s252/s255 and tsvc s252/s255 stay sequential: rotation leaves a dead store and derive
   has no DDE. Make `try_substitute_rotation` (`induction_variable_substitution.py`) remove it.
3. `single`: tsvc s254 needs the scalar-slice folds again right before lifting.
4. `single`: tsvc va, polybench durbin lose Memcpy: the loop copy matcher needs an `out = in` tasklet that
   EliminateTrivialTasklets removed. Accept a direct AccessNode->AccessNode edge in
   `AssignmentAndCopyKernelToMemsetAndMemcpy._detect_loop_transfer`.
5. `single`: tsvc_2_5 scan_strided_sym, fission_dep_sym_offset: LoopToScan emits a zero-fill map; it is
   lifted only by copy/fill after Loop2Map and before cleanup fuses a second writer into its state.
6. `single`: npbench nbody: LiftEinsum only matches maps joined by fusion.
7. symm, syrk, syr2k: matchers (`rank_k_match.py` 552, `loop_to_rank_k_update.py` 179, `loop_to_symm.py` 876
   lines) need the npbench slice skeleton, so they run before lowering in both `phased` and `single`.
   Normalized-form rewrite NOT STARTED. Scalar-loop spellings do not lift in legacy either
   (`~/.cache/newcanon/spell/spellings.py`).
8. NormalizeWCR after lowering makes npbench azimint_naive numerically WRONG: MapToForLoop lowers an
   in-nsdfg WCR into a write-only connector. Needs a reproducer and a MapToForLoop refusal; then
   NormalizeWCR can move after lowering.
9. `phased` test run, numerics on tsvc/poly/np, clean sw4 timing: not run.
10. PartitionGuardedLoop, SinkStateIntoLoop, PrivatizeReductionAccumulator: in neither recipe (no caller).
11. Never change the graph on 321 kernels in legacy (removal candidates, not proof): LiftInv, CollapseNoOpCast,
    ContinueToCondition, second LoopToScan, every CascadeInterstateEdgeAssignmentsUp, FuseConsecutiveLoops,
    LoopToTranspose, LoopToEinsum (4.4s), BufferExpansion, MaterializeLoopExitSymbols, MoveLoopIntoMapGated.

## Design questions for the user (options)

A. Rail contents. (a) StructuralCleanup only; (b) + PruneAndInline + CascadeIedgesUp + UniqueLoopIterators
   (paper step 9 pairs forward substitution with iterator uniquing). `single` uses (b).
B. SimplifyPass. (a) once after normalize (user order; argmax and 4 more lose lifts);
   (b) inside the statement fixpoint on the pre-lowering map form (`single`).
C. LiftEinsum on fused maps (nbody). (a) cross-phase placement after recompose; (b) drop, lose one lift;
   (c) teach LoopToEinsum the split-apart contraction.
D. symm/syrk/syr2k. (a) keep before lowering; (b) rewrite matchers for the normalized nest (size above).
E. MoveMapInvariantIfUp needs maps, so it cannot join MoveLoopInvariantIfUp in normalize:
   (a) keep after fusion (tsvc_2_5 config_select_branch needs it); (b) a loop-level unswitch that fusion does
   not undo.
