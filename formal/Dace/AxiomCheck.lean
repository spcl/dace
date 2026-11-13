-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-!
Axiom audit: `scripts/check_axioms.sh` verifies that none of the theorems below
depends on `sorryAx`.  The list mirrors `tests/Headlines.lean` (the headline
theorems and the specifications kept for pending milestones); keep the two in
sync.
-/

-- Headlines
#print axioms Dace.sound
#print axioms Dace.execSDFG_sound
#print axioms Dace.SDFG.run_sound
#print axioms Dace.SDFG.run_ok_sound
#print axioms Dace.STree.sound
#print axioms Dace.STree.execList_sound
#print axioms Dace.STree.run_sound
#print axioms Dace.STree.run_ok_sound
#print axioms Dace.raceFree?_sound
#print axioms Dace.raceFreeWith_sound
#print axioms Dace.StateExec.det
#print axioms Dace.StateExec.no_err
#print axioms Dace.scopesOk?_sound
#print axioms Dace.seqPure?_sound
#print axioms Dace.Congr.block
#print axioms Dace.Congr.blockEquiv
#print axioms Dace.Congr.emptyState
#print axioms Dace.Congr.splitEmpty
#print axioms Dace.Congr.mergeEmpty
#print axioms Dace.Congr.scopeBody
#print axioms Dace.Congr.scopeBodyEquiv
#print axioms Dace.Congr.mapBody
#print axioms Dace.sameOutside?_spec
#print axioms Dace.Congr.relabelState
#print axioms Dace.Congr.replaceState
#print axioms Dace.Congr.regionEquiv
#print axioms Dace.Congr.relabelBlocks
#print axioms Dace.Congr.insertEmpty
#print axioms Dace.Congr.removeEmpty
#print axioms Dace.SDFG.beq_sound
#print axioms Dace.Refines.refl_of_supported
#print axioms Dace.Refines.refl_iff_supported
#print axioms Dace.Refines.trans
#print axioms Dace.Refines.conservative
#print axioms Dace.Equiv.symm
#print axioms Dace.Equiv.trans
#print axioms Dace.Equiv.refl_iff_supported
#print axioms Dace.Refines_iff_matched
#print axioms Dace.Subset.disjoint?_sound
#print axioms Dace.Subset.subsetOf?_sound
#print axioms Dace.iterObligations?_sound
#print axioms Dace.iterDisjoint?_sound
#print axioms Dace.Cert.check_sound
#print axioms Dace.Query.withAxioms_sound
#print axioms Dace.Query.decide_sound
#print axioms Dace.Constraints.toHyps_sound
#print axioms Dace.toPoly?_sound
#print axioms Dace.Expr.nonneg?_sound
#print axioms Dace.Expr.pos?_sound
#print axioms Dace.SDFGExec.sim
#print axioms Dace.STree.ListExec.ren
#print axioms Dace.STree.ListExec.unlabel
#print axioms Dace.SDFG.Exec.det
#print axioms Dace.SDFG.Exec.no_err
#print axioms Dace.Corr.ProgCorr.refines
#print axioms Dace.Corr.corrCheck_sound
#print axioms Dace.Corr.corr?_exec
#print axioms Dace.Corr.corr?_refines
#print axioms Dace.Congr.rename
#print axioms Dace.Congr.swapNames
#print axioms Dace.checkBy_sound

-- Specifications kept for pending milestones
#print axioms Dace.OrderedExec.coherent'
#print axioms Dace.merge_perm
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
#print axioms Dace.State.components_convex
#print axioms Dace.State.mem_treeEdges
#print axioms Dace.State.treeEdges_nodup
#print axioms Dace.State.topoOrder_spec
#print axioms Dace.State.isTopologicalOrder_perm
#print axioms Dace.State.topo_pairwise
#print axioms Dace.State.reaches_iff
#print axioms Dace.State.reaches_irrefl
#print axioms Dace.Log.seq_assoc
#print axioms Dace.Log.conflicts_comm
#print axioms Dace.Log.pairwiseFree_perm
#print axioms Dace.Log.apply_seq
#print axioms Dace.Log.dropNames_seq
#print axioms Dace.Log.dropNames_unionAll
#print axioms Dace.Log.dropNames_mono
#print axioms Dace.Log.Le.seq_comm
#print axioms Dace.merge_eq_apply
#print axioms Dace.merge_cons_seq
#print axioms Dace.merge_eq_foldl
#print axioms Dace.merge_le
#print axioms Dace.merge_le_conflict
#print axioms Dace.CompsExec.merge_perm
#print axioms Dace.CompsExec.sequential
#print axioms Dace.OrderedExec.order_indep
#print axioms Dace.OrderedExec.order_indep_exec
#print axioms Dace.OrderedExec.order_indep_component
#print axioms Dace.OrderedExec.frame
#print axioms Dace.IterExec.frame
#print axioms Dace.SeqExec.reorder
#print axioms Dace.NodeExec.fp_step
#print axioms Dace.SeqExec.fp_trace
#print axioms Dace.CompsExec.fp_logs
#print axioms Dace.IterExec.footprint
#print axioms Dace.Access.hit_static
