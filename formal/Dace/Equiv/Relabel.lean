-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Relabel
import Dace.Equiv.Congr

/-!
# Rule: relabeling the nodes of a state

The first rule of the SDFG–SDFG checker (`Dace.Equiv.SdfgCorr`): replacing a
state of an SDFG, at any depth of the control-flow hierarchy, by the same
state with its nodes renumbered (`State.relabel φ` for a permutation `φ` of
node ids that fixes the nested SDFG nodes) gives an equivalent SDFG.

The state-level argument is `StateExec.relabel` / `StateExec.of_relabel`
(`Dace.SDFG.Relabel`): runs correspond with equal outcomes, except that a
race witness names the relabeled nodes (both are undefined behavior).
`Congr.blockEquiv` lifts it to the program.
-/

set_option autoImplicit false

namespace Dace

variable {φ : Equiv.Perm NodeId}

theorem DNodes.externalNames_relabel (pfx : String) :
    ∀ ns : List DNode, (∀ n ∈ ns, n.isNested = true → φ n.id = n.id) →
      DNodes.externalNames pfx (ns.map (DNode.relabel φ)) = DNodes.externalNames pfx ns
  | [], _ => rfl
  | n :: ns, h => by
    obtain ⟨id, label, ic, oc, se, kind⟩ := n
    simp only [List.map_cons, DNode.relabel, DNodes.externalNames]
    rw [DNodes.externalNames_relabel pfx ns fun n hn => h n (List.mem_cons_of_mem _ hn)]
    congr 1
    cases kind with
    | nested g m ni =>
      have := h _ List.mem_cons_self (by simp [DNode.isNested])
      simp only at this
      rw [this]
    | _ => rfl

theorem sameTransientMentions?_relabel (g : SDFG) {s : State} (hfix : s.FixesNested φ) :
    sameTransientMentions? g (.state (s.relabel φ)) (.state s) = true := by
  unfold sameTransientMentions?
  simp only [Bool.and_eq_true, List.all_eq_true, beq_iff_eq]
  refine ⟨fun x _ => ?_, ?_⟩
  · simp [Block.countMentions, State.relabel_nodes, List.filter_map, Function.comp_def]
  · obtain ⟨label, nodes, edges, nosync⟩ := s
    simp only [State.relabel, Block.externalNames]
    exact DNodes.externalNames_relabel "" nodes hfix

/-- Outcomes simulated with `Eq` are below each other in the sense of `MatchedE`. -/
theorem Res.Below.of_resSim {r₂ r : Res (Ctl × CSt)} (h : ResSim Eq r₂ r) : Res.Below CtlRel r r₂ := by
  rcases h with ⟨e, rfl, hu⟩ | ⟨e, rfl, rfl⟩ | ⟨a, b, rfl, rfl, rfl⟩
  · exact .inl ⟨e, rfl, hu⟩
  · exact .inr rfl
  · exact .inr (CtlRel.refl _)

/-- **Relabeling rule**: renumbering the nodes of the state at path `p`,
index `i` by a permutation that fixes its nested SDFG nodes gives an
equivalent SDFG. -/
theorem Congr.relabelState {g : SDFG} {p : List PathStep} {i : BlockId} {s : State}
    (hat : g.top.blockAt? p i = some (.state s)) (hfix : s.FixesNested φ) (hsup : Supported g.prog) :
    Equiv (g.setBlockAt p i (.state (s.relabel φ))).prog g.prog := by
  refine Congr.blockEquiv hat (sameTransientMentions?_relabel g hfix) (fun I _ cst r' h => ?_)
    (fun I _ cst r h => ?_) hsup
  · cases h with
    | state h =>
      obtain ⟨r, hr, hs⟩ := StateExec.of_relabel hfix h
      exact ⟨r, .state hr, Res.Below.of_resSim hs⟩
  · cases h with
    | state h =>
      obtain ⟨r₂, hr, hs⟩ := StateExec.relabel hfix h
      exact ⟨r₂, .state hr, Res.Below.of_resSim hs⟩

end Dace
