-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-!
Axiom audit: every trusted theorem of the development is listed here so that
`scripts/check_axioms.sh` can verify none depends on `sorryAx`.
-/

-- Theorems are appended here as they are proven, e.g.
-- #print axioms Dace.Equiv.corr_sound

-- M3: soundness of the SDFG interpreter with respect to the relational semantics.
#print axioms Dace.sound
#print axioms Dace.execSDFG_sound
#print axioms Dace.SDFG.run_sound
#print axioms Dace.SDFG.run_ok_sound

-- R1.1: algebra of logs and of their parallel merge.
#print axioms Dace.Log.seq_assoc
#print axioms Dace.Log.empty_seq
#print axioms Dace.Log.WF.seq
#print axioms Dace.Log.conflicts_eq_false_iff
#print axioms Dace.Log.conflicts_comm
#print axioms Dace.Log.conflicts_mono
#print axioms Dace.Log.conflicts_unionAll
#print axioms Dace.Log.pairwiseFree_iff_pairwise
#print axioms Dace.Log.pairwiseFree_perm
#print axioms Dace.Log.pairwiseFree_of_sub
#print axioms Dace.Log.apply_seq
#print axioms Dace.Log.unionAll_cons_eq_seq
#print axioms Dace.merge_eq_apply
#print axioms Dace.merge_isOk_iff
#print axioms Dace.merge_singleton
#print axioms Dace.merge_cons_seq
#print axioms Dace.merge_eq_foldl
#print axioms Dace.merge_perm
#print axioms Dace.Log.dropNames_seq
#print axioms Dace.Log.dropNames_unionAll
#print axioms Dace.Log.dropNames_mono
#print axioms Dace.Log.Le.seq
#print axioms Dace.Log.Le.unionAll
#print axioms Dace.Log.Le.dropNames
#print axioms Dace.Log.Le.apply_eq
#print axioms Dace.Log.Le.seq_comm
#print axioms Dace.merge_le
#print axioms Dace.merge_le_conflict
#print axioms Dace.Log.apply_push

-- M4: soundness of the schedule tree interpreter with respect to the relational semantics.
#print axioms Dace.STree.sound
#print axioms Dace.STree.execList_sound
#print axioms Dace.STree.run_sound
#print axioms Dace.STree.run_ok_sound

-- R0.4: specification of the (total) scope, component, memlet-tree and topological-order functions.
#print axioms Dace.mem_closure
#print axioms Dace.closure_nodup
#print axioms Dace.State.scopeWF?_sound
#print axioms Dace.State.mem_scopeNodes
#print axioms Dace.State.mem_scopeNodes_iff_parent
#print axioms Dace.State.scopeNodes_nodup
#print axioms Dace.State.mem_component
#print axioms Dace.State.components_cover
#print axioms Dace.State.components_flatten_nodup
#print axioms Dace.State.mem_components_iff
#print axioms Dace.State.components_edge
#print axioms Dace.State.components_eq_of_mem
#print axioms Dace.State.components_flatten_perm
#print axioms Dace.State.mem_treeEdges
#print axioms Dace.State.treeEdges_nodup
#print axioms Dace.State.topoOrder_spec
#print axioms Dace.State.isTopologicalOrder_perm

-- R0.5: happens-before races (reachability, trace logs).
#print axioms Dace.State.reaches_iff
#print axioms Dace.State.reaches_irrefl
#print axioms Dace.SeqExec.log_eq
#print axioms Dace.SeqExec.log_eq_seqAll
#print axioms Dace.SeqExec.log_eq_seqAll_empty

-- R2.1–R2.3: facts, polynomial normal form, certificate checking.
#print axioms Dace.Constraints.satisfies_empty
#print axioms Dace.toPoly?_sound
#print axioms Dace.toPoly?_spec
#print axioms Dace.atomEnv_axioms
#print axioms Dace.axioms_atomOk
#print axioms Dace.Cert.check_sound
#print axioms Dace.Query.withAxioms_sound
#print axioms Dace.Query.decide_sound
#print axioms Dace.Constraints.toHyps_sound

-- R2.4: points of subsets and verified subset procedures.
#print axioms Dace.Expr.eval_congr
#print axioms Dace.Expr.eval_subst
#print axioms Dace.Dim.mem_elems
#print axioms Dace.Dim.elems_sep
#print axioms Dace.Range.mem_points
#print axioms Dace.Subset.mem_points
#print axioms Dace.exists_firstDiff
#print axioms Dace.SymSubset.dominates_inj
#print axioms Dace.Expr.nonneg?_sound
#print axioms Dace.Expr.pos?_sound
#print axioms Dace.Subset.disjoint?_sound
#print axioms Dace.Subset.subsetOf?_sound
#print axioms Dace.iterObligations?_sound
#print axioms Dace.iterDisjoint?_sound

-- R1.2: observables, refinement and equivalence.
#print axioms Dace.Refines.refl_of_supported
#print axioms Dace.Refines.refl_iff_supported
#print axioms Dace.Refines.trans
#print axioms Dace.Refines.conservative
#print axioms Dace.Equiv.symm
#print axioms Dace.Equiv.trans
#print axioms Dace.Equiv.refl_iff_supported
#print axioms Dace.Refines_iff_sim
#print axioms Dace.Sim.trans

-- R1.3: logs are write-only (log-shift lemmas for both relation families).
#print axioms Dace.NodeExec.shift_iff
#print axioms Dace.SeqExec.shift_iff
#print axioms Dace.OrderedExec.shift_iff
#print axioms Dace.IterExec.shift_iff
#print axioms Dace.ConsumeTasks.shift_iff
#print axioms Dace.StateExec.shift_iff
#print axioms Dace.BlockExec.shift_iff
#print axioms Dace.RegionAt.shift_iff
#print axioms Dace.RegionExec.shift_iff
#print axioms Dace.LoopExec.shift_iff
#print axioms Dace.LoopGuard.shift_iff
#print axioms Dace.LoopIter.shift_iff
#print axioms Dace.LoopTail.shift_iff
#print axioms Dace.CondExec.shift_iff
#print axioms Dace.BlockExec.log_iff
#print axioms Dace.NodeExec.log_iff
#print axioms Dace.BlockExec.rel
#print axioms Dace.NodeExec.rel
#print axioms Dace.SeqExec.rel
#print axioms Dace.OrderedExec.rel
#print axioms Dace.STree.NodeExec.shift_iff
#print axioms Dace.STree.ListExec.shift_iff
#print axioms Dace.STree.IterExec.shift_iff
#print axioms Dace.STree.ConsumeTasks.shift_iff
#print axioms Dace.STree.LoopExec.shift_iff
#print axioms Dace.STree.LoopGuard.shift_iff
#print axioms Dace.STree.LoopIter.shift_iff
#print axioms Dace.STree.LoopTail.shift_iff
#print axioms Dace.STree.NodeExec.log_iff
#print axioms Dace.STree.NodeExec.rel
#print axioms Dace.STree.ListExec.rel

-- R2.6: static footprints of dataflow executions.
#print axioms Dace.NodeExec.fp_step
#print axioms Dace.SeqExec.fp_step
#print axioms Dace.SeqExec.fp_trace
#print axioms Dace.OrderedExec.fp_step
#print axioms Dace.OrderedExec.fp_log
#print axioms Dace.NodeExec.fp_log
#print axioms Dace.IterExec.fp_logs
#print axioms Dace.CompsExec.fp_logs
#print axioms Dace.IterExec.footprint
#print axioms Dace.Access.hit_static

-- R1.4: block congruence (context congruence, paths, block replacement).
#print axioms Dace.BlockExec.congr_iff
#print axioms Dace.SDFGExec.congr_iff
#print axioms Dace.NodeExec.congr
#print axioms Dace.Ctx.agree_setBlockAt
#print axioms Dace.Refines.of_simE
#print axioms Dace.Supported.of_simE
#print axioms Dace.RegionAt.simE
#print axioms Dace.LoopExec.simE
#print axioms Dace.CondExec.simE
#print axioms Dace.Congr.region_setBlockAt
#print axioms Dace.Region.setBlockAt_setBlockAt
#print axioms Dace.Region.countMentions_setBlockAt
#print axioms Dace.Congr.exec_simE
#print axioms Dace.Congr.block
#print axioms Dace.Congr.blockEquiv
#print axioms Dace.Congr.emptyState
#print axioms Dace.Congr.splitEmpty
#print axioms Dace.Congr.mergeEmpty

-- R1.6: frame property, St-coherence and order independence on `SeqPure`.
#print axioms Dace.Ctx.eval_pure
#print axioms Dace.Ctx.pointsOf_pure
#print axioms Dace.FootA.bind
#print axioms Dace.Foot.foldlM
#print axioms Dace.seqPure?_sound
#print axioms Dace.execSimple_foot
#print axioms Dace.NodeExec.frame
#print axioms Dace.SeqExec.frame
#print axioms Dace.OrderedExec.frame
#print axioms Dace.IterExec.frame
#print axioms Dace.NodeExec.frame_ok
#print axioms Dace.OrderedExec.frame_ok
#print axioms Dace.NodeExec.coherent
#print axioms Dace.OrderedExec.coherent
#print axioms Dace.SeqExec.coherent
#print axioms Dace.SeqExec.swap
#print axioms Dace.SeqExec.bubble
#print axioms Dace.SeqExec.reorder
#print axioms Dace.State.topo_pairwise
#print axioms Dace.State.components_convex
#print axioms Dace.OrderedExec.order_indep
#print axioms Dace.OrderedExec.order_indep_exec
#print axioms Dace.OrderedExec.order_indep_component
#print axioms Dace.CompsExec.sequential
#print axioms Dace.CompsExec.merge_perm

-- R2.7: static race freedom
#print axioms Dace.raceFree?_sound
#print axioms Dace.NodeExec.clean
#print axioms Dace.RegionExec.cgood
#print axioms Dace.execSimple_clean
#print axioms Dace.raceFreeWith_sound

-- R1.5: map-body congruence
#print axioms Dace.Congr.scopeBody
#print axioms Dace.Congr.scopeBodyEquiv
#print axioms Dace.Congr.mapBody
#print axioms Dace.sameOutside?_spec
#print axioms Dace.StateExec.simB
#print axioms Dace.State.isTopologicalOrder_splice
#print axioms Dace.NodeLocal.execSimple

-- R1.6: determinism
#print axioms Dace.NodeExec.det
#print axioms Dace.OrderedExec.det
#print axioms Dace.CompsExec.det
#print axioms Dace.StateExec.det
#print axioms Dace.StateExec.no_err
#print axioms Dace.scopesOk?_sound
