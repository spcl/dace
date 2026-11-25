-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-!
# Headline theorems

The statements the development exists for.  `lake env lean tests/Headlines.lean`
prints them; the output is compared with `tests/Headlines.expected`, so a
refactoring cannot silently weaken or drop one.  `scripts/audit.py` takes
the names checked here as the roots of its reachability analysis: a
declaration that neither an executable nor one of these theorems uses is
reported as unreachable.

The second part lists specification theorems that no headline uses yet but
that pending milestones rely on (state split/merge, map permutation atoms,
memlet propagation); they are roots too.
-/

-- Headlines
#check @Dace.sound
#check @Dace.execSDFG_sound
#check @Dace.SDFG.run_sound
#check @Dace.SDFG.run_ok_sound
#check @Dace.STree.sound
#check @Dace.STree.execList_sound
#check @Dace.STree.run_sound
#check @Dace.STree.run_ok_sound
#check @Dace.raceFree?_sound
#check @Dace.raceFreeWith_sound
#check @Dace.StateExec.det
#check @Dace.StateExec.no_err
#check @Dace.scopesOk?_sound
#check @Dace.seqPure?_sound
#check @Dace.Congr.block
#check @Dace.Congr.blockEquiv
#check @Dace.Congr.emptyState
#check @Dace.Congr.splitEmpty
#check @Dace.Congr.mergeEmpty
#check @Dace.Congr.scopeBody
#check @Dace.Congr.scopeBodyEquiv
#check @Dace.Congr.mapBody
#check @Dace.sameOutside?_spec
#check @Dace.Congr.relabelState
#check @Dace.Congr.replaceState
#check @Dace.Congr.regionEquiv
#check @Dace.Congr.relabelBlocks
#check @Dace.Congr.insertEmpty
#check @Dace.Congr.removeEmpty
#check @Dace.SDFG.beq_sound
#check @Dace.Refines.refl_of_supported
#check @Dace.Refines.refl_iff_supported
#check @Dace.Refines.trans
#check @Dace.Refines.conservative
#check @Dace.Equiv.symm
#check @Dace.Equiv.trans
#check @Dace.Equiv.refl_iff_supported
#check @Dace.Refines_iff_matched
#check @Dace.Subset.disjoint?_sound
#check @Dace.Subset.subsetOf?_sound
#check @Dace.iterObligations?_sound
#check @Dace.iterDisjoint?_sound
#check @Dace.Cert.check_sound
#check @Dace.Query.withAxioms_sound
#check @Dace.Query.decide_sound
#check @Dace.Constraints.toHyps_sound
#check @Dace.toPoly?_sound
#check @Dace.Expr.nonneg?_sound
#check @Dace.Expr.pos?_sound
#check @Dace.SDFGExec.sim
#check @Dace.STree.ListExec.ren
#check @Dace.STree.ListExec.unlabel
#check @Dace.SDFG.Exec.det
#check @Dace.SDFG.Exec.no_err
#check @Dace.Corr.ProgCorr.refines
#check @Dace.Corr.corrCheck_sound
#check @Dace.Corr.corr?_exec
#check @Dace.Corr.corr?_refines
#check @Dace.Congr.rename
#check @Dace.Congr.swapNames
#check @Dace.checkBy_sound

-- Specifications kept for pending milestones
#check @Dace.OrderedExec.coherent'
#check @Dace.merge_perm
#check @Dace.State.scopeWF?_sound
#check @Dace.State.mem_scopeNodes
#check @Dace.State.mem_scopeNodes_iff_parent
#check @Dace.State.scopeNodes_nodup
#check @Dace.State.mem_component
#check @Dace.State.components_cover
#check @Dace.State.components_flatten_nodup
#check @Dace.State.mem_components_iff
#check @Dace.State.components_edge
#check @Dace.State.components_eq_of_mem
#check @Dace.State.components_flatten_perm
#check @Dace.State.components_convex
#check @Dace.State.mem_treeEdges
#check @Dace.State.treeEdges_nodup
#check @Dace.State.topoOrder_spec
#check @Dace.State.isTopologicalOrder_perm
#check @Dace.State.topo_pairwise
#check @Dace.State.reaches_iff
#check @Dace.State.reaches_irrefl
#check @Dace.Log.seq_assoc
#check @Dace.Log.conflicts_comm
#check @Dace.Log.pairwiseFree_perm
#check @Dace.Log.apply_seq
#check @Dace.Log.dropNames_seq
#check @Dace.Log.dropNames_unionAll
#check @Dace.Log.dropNames_mono
#check @Dace.Log.Le.seq_comm
#check @Dace.merge_eq_apply
#check @Dace.merge_cons_seq
#check @Dace.merge_eq_foldl
#check @Dace.merge_le
#check @Dace.merge_le_conflict
#check @Dace.CompsExec.merge_perm
#check @Dace.CompsExec.sequential
#check @Dace.OrderedExec.order_indep
#check @Dace.OrderedExec.order_indep_exec
#check @Dace.OrderedExec.order_indep_component
#check @Dace.OrderedExec.frame
#check @Dace.IterExec.frame
#check @Dace.SeqExec.reorder
#check @Dace.NodeExec.fp_step
#check @Dace.SeqExec.fp_trace
#check @Dace.CompsExec.fp_logs
#check @Dace.IterExec.footprint
#check @Dace.Access.hit_static
