-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.Relabel
import Dace.SDFG.BEq

/-!
# The SDFG–SDFG checker (M5c)

`Equiv` is IR-generic, so an SDFG–SDFG equivalence checker needs only proof
rules that rewrite one SDFG into another.  The checker is split into an
untrusted **search** and a verified **replay**:

* the search (`findSteps`) walks both SDFGs in parallel and proposes a list
  of rule applications (`Step`), e.g. "relabel the nodes of the state at
  this path by these id pairs";
* the replay (`replay`) applies the rules to the first SDFG, each guarded by
  its decidable side condition (`Step.apply?`), and the result is compared
  with the second SDFG by the sound structural equality `SDFG.beq`.

Soundness (`check?_sound`): if the check succeeds and the first SDFG is
`Supported`, the two SDFGs are equivalent.  Each rule carries a theorem
(`Step.apply?_sound`): equivalence of the rewritten SDFG, with equal facts
and interface, so that rule applications chain (`Equiv.trans_of_eq`).

## Rules

* `relabel p i pairs` (`Congr.relabelState`): renumber the nodes of the
  state at path `p`, index `i` by the permutation `permOfPairs pairs`
  (side condition: nested SDFG nodes keep their ids).

The permutation is built from the id pairs by swaps (`permOfPairs`); its
correctness is never assumed: the replayed SDFG is compared with the target.
-/

set_option autoImplicit false

namespace Dace

/-! ## Interfaces after replacing a block -/

@[simp] theorem Block.outlivingNames_state (pfx : String) (s : State) :
    Block.outlivingNames pfx (.state s) = DNodes.outlivingNames pfx s.nodes := by
  cases s; rfl

theorem Blocks.outlivingNames_set (pfx : String) :
    ∀ {bs : List Block} {j : Nat} {x x' : Block}, bs[j]? = some x →
      Block.outlivingNames pfx x' = Block.outlivingNames pfx x →
        Blocks.outlivingNames pfx (bs.set j x') = Blocks.outlivingNames pfx bs
  | [], _, _, _, h, _ => by simp at h
  | y :: bs, 0, x, x', h, hx => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    simp [Blocks.outlivingNames, hx]
  | y :: bs, j + 1, x, x', h, hx => by
    simp only [List.getElem?_cons_succ] at h
    simp [Blocks.outlivingNames, Blocks.outlivingNames_set pfx h hx]

theorem Branches.outlivingNames_set (pfx : String) :
    ∀ {brs : List (Option Expr × Region)} {k : Nat} {e : Option Expr} {c c' : Region}, brs[k]? = some (e, c) →
      Region.outlivingNames pfx c' = Region.outlivingNames pfx c →
        Branches.outlivingNames pfx (brs.set k (e, c')) = Branches.outlivingNames pfx brs
  | [], _, _, _, _, h, _ => by simp at h
  | y :: brs, 0, e, c, c', h, hx => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    simp [Branches.outlivingNames, hx]
  | y :: brs, k + 1, e, c, c', h, hx => by
    obtain ⟨c₀, r₀⟩ := y
    simp only [List.getElem?_cons_succ] at h
    simp [Branches.outlivingNames, Branches.outlivingNames_set pfx h hx]

theorem Region.outlivingNames_setBlocks (pfx : String) {r : Region} {j : BlockId} {x x' : Block}
    (hj : r.block? j = some x) (hx : Block.outlivingNames pfx x' = Block.outlivingNames pfx x) :
    Region.outlivingNames pfx { r with blocks := r.blocks.set j x' } = Region.outlivingNames pfx r := by
  obtain ⟨_, _, blocks, _, edges⟩ := r
  simp only [Region.outlivingNames]
  rw [Blocks.outlivingNames_set pfx hj hx]

theorem Region.outlivingNames_setChild (pfx : String) {r : Region} {st : PathStep} {c c' : Region}
    (hc : r.child? st = some c) (h : Region.outlivingNames pfx c' = Region.outlivingNames pfx c) :
    Region.outlivingNames pfx (r.setChild c' st) = Region.outlivingNames pfx r := by
  cases st with
  | sub j =>
    simp only [Region.child?] at hc
    split at hc
    · rename_i c₀ hj
      cases hc
      simp only [Region.setChild, hj]
      exact Region.outlivingNames_setBlocks pfx hj (by simp [Block.outlivingNames, h])
    · cases hc
  | body j =>
    simp only [Region.child?] at hc
    split at hc
    · rename_i l hj
      cases hc
      simp only [Region.setChild, hj]
      refine Region.outlivingNames_setBlocks pfx hj ?_
      obtain ⟨_, _, _, _, _, _, _, _⟩ := l
      simp only [Block.outlivingNames] at h ⊢
      rw [h]
    · cases hc
  | branch j k =>
    simp only [Region.child?] at hc
    split at hc
    · rename_i lb brs hj
      obtain ⟨⟨e, c₀⟩, hk, rfl⟩ := Option.map_eq_some_iff.1 hc
      simp only [Region.setChild, hj, hk]
      exact Region.outlivingNames_setBlocks pfx hj (by simp [Block.outlivingNames, Branches.outlivingNames_set pfx hk h])
    · cases hc

theorem Region.outlivingNames_setBlockAt (pfx : String) {i : BlockId} {b b' : Block}
    (hx : Block.outlivingNames pfx b' = Block.outlivingNames pfx b) :
    ∀ (p : List PathStep) (r : Region), r.blockAt? p i = some b →
      Region.outlivingNames pfx (r.setBlockAt p i b') = Region.outlivingNames pfx r
  | [], r, h => Region.outlivingNames_setBlocks pfx h hx
  | st :: p, r, h => by
    simp only [Region.blockAt?] at h
    split at h
    · rename_i c hc
      simp only [Region.setBlockAt, hc]
      exact Region.outlivingNames_setChild pfx hc (Region.outlivingNames_setBlockAt pfx hx p c h)
    · cases h


theorem SDFG.interface_setBlockAt {g : SDFG} {p : List PathStep} {i : BlockId} {b b' : Block}
    (hat : g.top.blockAt? p i = some b) (hx : Block.outlivingNames "" b' = Block.outlivingNames "" b) :
    (g.setBlockAt p i b').interface = g.interface := by
  simp only [SDFG.interface, SDFG.setBlockAt]
  rw [Region.outlivingNames_setBlockAt "" hx p g.top hat]

theorem DNodes.outlivingNames_relabel {φ : Equiv.Perm NodeId} (pfx : String) :
    ∀ ns : List DNode, (∀ n ∈ ns, n.isNested = true → φ n.id = n.id) →
      DNodes.outlivingNames pfx (ns.map (DNode.relabel φ)) = DNodes.outlivingNames pfx ns
  | [], _ => rfl
  | n :: ns, h => by
    obtain ⟨id, label, ic, oc, se, kind⟩ := n
    simp only [List.map_cons, DNode.relabel, DNodes.outlivingNames]
    rw [DNodes.outlivingNames_relabel pfx ns fun n hn => h n (List.mem_cons_of_mem _ hn)]
    congr 1
    cases kind with
    | nested g m ni =>
      have := h _ List.mem_cons_self (by simp [DNode.isNested])
      simp only at this
      rw [this]
    | _ => rfl

/-! ## Rules -/

/-- A permutation sending each `a` to `b` for the pairs `(a, b)` (when the
`a`s and the `b`s are distinct), built from swaps. -/
def permOfPairs : List (NodeId × NodeId) → Equiv.Perm NodeId
  | [] => 1
  | (a, b) :: rest => Equiv.swap (permOfPairs rest a) b * permOfPairs rest

/-- Do the nested SDFG nodes of `s` keep their ids under `φ`? -/
def State.fixesNested? (φ : Equiv.Perm NodeId) (s : State) : Bool :=
  s.nodes.all fun n => !n.isNested || φ n.id == n.id

theorem State.fixesNested?_sound {φ : Equiv.Perm NodeId} {s : State} (h : s.fixesNested? φ = true) :
    s.FixesNested φ := by
  intro n hn hnest
  have := List.all_eq_true.1 h n hn
  simpa [hnest] using this

/-- One rule application. -/
inductive Step where
  /-- Relabel the nodes of the state at path `path`, index `idx`. -/
  | relabel (path : List PathStep) (idx : BlockId) (pairs : List (NodeId × NodeId))
  deriving Repr, Inhabited

/-- Apply a rule, if its side condition holds. -/
def Step.apply? (g : SDFG) : Step → Option SDFG
  | .relabel p i pairs =>
    match g.top.blockAt? p i with
    | some (.state s) =>
      if s.fixesNested? (permOfPairs pairs) then some (g.setBlockAt p i (.state (s.relabel (permOfPairs pairs))))
      else none
    | _ => none

/-- What every rule guarantees: an equivalent SDFG with the same facts and
interface. -/
structure Rewrites (g' g : SDFG) : Prop where
  equiv : Equiv g'.prog g.prog
  facts : g'.facts = g.facts
  interface : g'.interface = g.interface

theorem Rewrites.trans {g'' g' g : SDFG} (h₂ : Rewrites g'' g') (h₁ : Rewrites g' g) : Rewrites g'' g :=
  ⟨Equiv.trans_of_eq h₂.equiv h₁.equiv h₂.facts h₁.facts.symm h₂.interface h₁.interface.symm,
    h₂.facts.trans h₁.facts, h₂.interface.trans h₁.interface⟩

theorem Rewrites.refl {g : SDFG} (hs : Supported g.prog) : Rewrites g g :=
  ⟨Equiv.refl_iff_supported.2 hs, rfl, rfl⟩

theorem Rewrites.supported {g' g : SDFG} (h : Rewrites g' g) : Supported g'.prog :=
  h.equiv.supported_left h.facts h.interface

/-- **Every rule application is sound.** -/
theorem Step.apply?_sound {g g' : SDFG} {st : Step} (h : st.apply? g = some g') (hs : Supported g.prog) :
    Rewrites g' g := by
  cases st with
  | relabel p i pairs =>
    simp only [Step.apply?] at h
    split at h
    · rename_i s hat
      split at h
      · rename_i hfix
        cases h
        have hfix := State.fixesNested?_sound hfix
        refine ⟨Congr.relabelState hat hfix hs, rfl, SDFG.interface_setBlockAt hat ?_⟩
        obtain ⟨label, nodes, edges, nosync⟩ := s
        simp only [State.relabel, Block.outlivingNames]
        exact DNodes.outlivingNames_relabel "" nodes hfix
      · cases h
    · cases h

/-- Apply a list of rules in order. -/
def replay (g : SDFG) : List Step → Option SDFG
  | [] => some g
  | st :: sts => (st.apply? g).bind fun g' => replay g' sts

theorem replay_sound : ∀ {g g' : SDFG} {sts : List Step}, replay g sts = some g' → Supported g.prog → Rewrites g' g
  | g, g', [], h, hs => by
    cases h
    exact Rewrites.refl hs
  | g, g', st :: sts, h, hs => by
    simp only [replay, Option.bind_eq_some_iff] at h
    obtain ⟨g₁, h₁, h₂⟩ := h
    have r₁ := Step.apply?_sound h₁ hs
    exact (replay_sound h₂ r₁.supported).trans r₁

/-- The check: replaying the rules on `a` gives `b`. -/
def checkBy (a b : SDFG) (sts : List Step) : Bool :=
  match replay a sts with
  | some g => SDFG.beq g b
  | none => false

/-- **Soundness of the SDFG–SDFG check**: `b` is equivalent to a supported
`a`. -/
theorem checkBy_sound {a b : SDFG} {sts : List Step} (h : checkBy a b sts = true) (hs : Supported a.prog) :
    Equiv b.prog a.prog := by
  unfold checkBy at h
  split at h
  · rename_i g hg
    rw [← SDFG.beq_sound h]
    exact (replay_sound hg hs).equiv
  · cases h

/-! ## Search (untrusted) -/

mutual

/-- Rule applications that turn the region `ra` (at path `p`) into `rb`. -/
partial def findRegion (p : List PathStep) (ra rb : Region) : Except String (List Step) := do
  if ra.blocks.length ≠ rb.blocks.length then
    throw s!"regions with {ra.blocks.length} and {rb.blocks.length} blocks"
  let mut out := []
  for (j, ba, bb) in (List.range ra.blocks.length).zip (ra.blocks.zip rb.blocks) do
    out := out ++ (← findBlock p j ba bb)
  return out

/-- Rule applications that turn block `j` of the region at path `p` into `bb`. -/
partial def findBlock (p : List PathStep) (j : BlockId) : Block → Block → Except String (List Step)
  | .state sa, .state sb =>
    if State.beq sa sb then pure []
    else if sa.nodes.length = sb.nodes.length then
      pure [.relabel p j ((sa.nodes.map (·.id)).zip (sb.nodes.map (·.id)))]
    else throw s!"state '{sa.label}' has {sa.nodes.length} nodes, '{sb.label}' {sb.nodes.length}"
  | .region ra, .region rb => findRegion (p ++ [.sub j]) ra rb
  | .loop la, .loop lb => findRegion (p ++ [.body j]) la.body lb.body
  | .cond _ ba, .cond _ bb => do
    if ba.length ≠ bb.length then throw "conditionals with different branch counts"
    let mut out := []
    for (k, (_, ra), (_, rb)) in (List.range ba.length).zip (ba.zip bb) do
      out := out ++ (← findRegion (p ++ [.branch j k]) ra rb)
    return out
  | _, _ => pure []

end

/-- The SDFG–SDFG check: `ok steps` when replaying the found rules turns `a`
into `b` (then `checkBy_sound` applies), or a reason. -/
def sdfgCheck (a b : SDFG) : Except String (List Step) := do
  let sts ← findRegion [] a.top b.top
  if checkBy a b sts then pure sts
  else throw "the SDFGs differ beyond the rules (node relabeling)"

end Dace
