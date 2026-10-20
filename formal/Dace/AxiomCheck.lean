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
