-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.Relabel
import Dace.Equiv.StateLocal
import Dace.Equiv.RegionRules
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
  (no side condition).
* `replaceState p i s'` (`Congr.replaceState`): replace the state at path
  `p`, index `i` by `s'` when the two states run alike (`stateSame?`: the
  same nodes and adjacent edges seen the same way, the same components in
  the same order, each up to a permutation of its nodes).  Covers
  reordering the node list of a state.
* `relabelBlocks p pairs` (`Congr.relabelBlocks`): renumber the blocks of
  the region at path `p` by `permOfPairs pairs` (side condition: the
  permutation maps the region's block ids onto themselves, start and edges
  in range, mention counts kept).

The permutation is built from the id pairs by swaps (`permOfPairs`); its
correctness is never assumed: the replayed SDFG is compared with the target.
-/

set_option autoImplicit false

namespace Dace

/-! ## Interfaces after replacing a block -/

/-- The interface depends only on the descriptors. -/
theorem SDFG.interface_setBlockAt (g : SDFG) (p : List PathStep) (i : BlockId) (b' : Block) :
    (g.setBlockAt p i b').interface = g.interface := rfl

/-! ## Rules -/

/-- A permutation sending each `a` to `b` for the pairs `(a, b)` (when the
`a`s and the `b`s are distinct), built from swaps. -/
def permOfPairs : List (NodeId × NodeId) → Equiv.Perm NodeId
  | [] => 1
  | (a, b) :: rest => Equiv.swap (permOfPairs rest a) b * permOfPairs rest

/-- One rule application. -/
inductive Step where
  /-- Relabel the nodes of the state at path `path`, index `idx`. -/
  | relabel (path : List PathStep) (idx : BlockId) (pairs : List (NodeId × NodeId))
  /-- Replace the state at path `path`, index `idx` by a state that runs alike. -/
  | replaceState (path : List PathStep) (idx : BlockId) (s' : State)
  /-- Renumber the blocks of the region at path `path`. -/
  | relabelBlocks (path : List PathStep) (pairs : List (BlockId × BlockId))
  deriving Inhabited

/-- Apply a rule, if its side condition holds. -/
def Step.apply? (g : SDFG) : Step → Option SDFG
  | .relabel p i pairs =>
    match g.top.blockAt? p i with
    | some (.state s) => some (g.setBlockAt p i (.state (s.relabel (permOfPairs pairs))))
    | _ => none
  | .replaceState p i s' =>
    match g.top.blockAt? p i with
    | some (.state s) => if stateSame? g s' s then some (g.setBlockAt p i (.state s')) else none
    | _ => none
  | .relabelBlocks p pairs =>
    match g.top.regionAt? p with
    | some c =>
      if c.blocksPermOk (permOfPairs pairs) && sameRegionMentions? g (c.relabelBlocks (permOfPairs pairs)) c then
        some (g.setRegionAt p (c.relabelBlocks (permOfPairs pairs)))
      else none
    | none => none

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
      cases h
      exact ⟨Congr.relabelState hat hs, rfl, rfl⟩
    · cases h
  | replaceState p i s' =>
    simp only [Step.apply?] at h
    split at h
    · rename_i s hat
      split at h
      · rename_i hchk
        cases h
        exact ⟨Congr.replaceState hat hchk hs, rfl, rfl⟩
      · cases h
    · cases h
  | relabelBlocks p pairs =>
    simp only [Step.apply?] at h
    split at h
    · rename_i c hat
      split at h
      · rename_i hchk
        simp only [Bool.and_eq_true] at hchk
        cases h
        exact ⟨Congr.relabelBlocks hat hchk.1 hchk.2 hs, rfl, rfl⟩
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

/-- A block's label (empty for break, continue and return). -/
def Block.key : Block → String
  | .state s => "s:" ++ s.label
  | .region r => "r:" ++ r.label
  | .loop l => "l:" ++ l.label
  | .cond lb _ => "c:" ++ lb
  | .breakBlock => "break"
  | .continueBlock => "continue"
  | .returnBlock => "return"

mutual

/-- Rule applications that turn the region `ra` (at path `p`) into `rb`. -/
partial def findRegion (p : List PathStep) (ra rb : Region) : Except String (List Step) := do
  if ra.blocks.length ≠ rb.blocks.length then
    throw s!"regions with {ra.blocks.length} and {rb.blocks.length} blocks"
  -- blocks in another order: match them by label
  let la := ra.blocks.map Block.key
  let lb := rb.blocks.map Block.key
  if la ≠ lb && la.isPerm lb && la.Nodup then
    let pairs := (List.range la.length).map fun i => (i, lb.idxOf la[i]!)
    return [.relabelBlocks p pairs] ++ (← findRegion p (ra.relabelBlocks (permOfPairs pairs)) rb)
  let mut out := []
  for (j, ba, bb) in (List.range ra.blocks.length).zip (ra.blocks.zip rb.blocks) do
    out := out ++ (← findBlock p j ba bb)
  return out

/-- Rule applications that turn block `j` of the region at path `p` into `bb`. -/
partial def findBlock (p : List PathStep) (j : BlockId) : Block → Block → Except String (List Step)
  | .state sa, .state sb =>
    if State.beq sa sb then pure []
    else
      let pairs := (sa.nodes.map (·.id)).zip (sb.nodes.map (·.id))
      if sa.nodes.length = sb.nodes.length && State.beq (sa.relabel (permOfPairs pairs)) sb then
        pure [.relabel p j pairs]
      else if (sa.nodes.map (·.id)).isPerm (sb.nodes.map (·.id)) then pure [.replaceState p j sb]
      else throw s!"state '{sa.label}' and '{sb.label}' have different nodes"
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
  else throw "the SDFGs differ beyond the rules (node relabeling, state replacement, block renumbering)"

end Dace
