-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Sim

/-!
# Deciding agreement on the names of a region

`Region.namesOk? okD okE nested rg` checks the names of every state and
control-flow expression of `rg` (recursively, not into nested SDFGs): the
container names of the states pass `okD`, the expression names `okE`, and,
unless `nested`, no state contains a nested SDFG.  When `okD` implies
`Ctx.Strong`, `okE` implies `Ctx.Good₀` and `nested` implies `ScopeSim`, a
passing region satisfies `Region.Names` (`Region.namesOk?_sound`), the
hypothesis of the simulation theorems of `Dace.SDFG.Sim`.
-/

set_option autoImplicit false
set_option linter.unusedSectionVars false

namespace Dace

section Check

variable (okD okE : Name → Bool) (nested : Bool)

/-- The names of a state pass the checks. -/
def stateNamesOk? (s : State) : Bool :=
  s.dataNames.all okD && s.exprNames.all okE && (nested || !s.nodes.any DNode.isNested)

def optNamesOk? (e : Option Expr) : Bool := ((e.map Expr.names).getD []).all okE

def assignNamesOk? (a : Option (Sym × Expr)) : Bool := ((a.map fun (_, e) => e.names).getD []).all okE

mutual

/-- The names of a region pass the checks. -/
def Region.namesOk? : Region → Bool
  | ⟨_, _, blocks, _, edges⟩ => Blocks.namesOk? blocks && edges.all fun e => e.names.all okE

def Blocks.namesOk? : List Block → Bool
  | [] => true
  | b :: bs => Block.namesOk? b && Blocks.namesOk? bs

def Block.namesOk? : Block → Bool
  | .state s => stateNamesOk? okD okE nested s
  | .region r => Region.namesOk? r
  | .loop ⟨_, _, init, cond, update, _, _, body⟩ =>
    assignNamesOk? okE init && cond.names.all okE && assignNamesOk? okE update && Region.namesOk? body
  | .cond _ brs => Branches.namesOk? brs
  | .breakBlock | .continueBlock | .returnBlock => true

def Branches.namesOk? : List (Option Expr × Region) → Bool
  | [] => true
  | (c, r) :: rest => optNamesOk? okE c && Region.namesOk? r && Branches.namesOk? rest

end

end Check

section Sound

variable {π : Equiv.Perm Name} {P : Name → Prop} {ctx₂ ctx₁ : Ctx} {okD okE : Name → Bool} {nested : Bool}
  (hD : ∀ x, okD x = true → Ctx.Strong π P ctx₂ ctx₁ x) (hE : ∀ x, okE x = true → Ctx.Good₀ π P ctx₂ ctx₁ x)
  (hN : nested = true → ScopeSim π P ctx₂ ctx₁)
include hD hE hN

theorem stateNamesOk?_sound {s : State} (h : stateNamesOk? okD okE nested s = true) : SNames π P ctx₂ ctx₁ s := by
  simp only [stateNamesOk?, Bool.and_eq_true, List.all_eq_true, Bool.or_eq_true, Bool.not_eq_true',
    List.any_eq_false] at h
  obtain ⟨⟨hd, he⟩, hn⟩ := h
  refine ⟨fun x hx => hD x (hd x hx), fun x hx => hE x (he x hx), fun ⟨n, hn', inner, m, ni, hk⟩ => ?_⟩
  rcases hn with hn | hn
  · exact hN hn
  · exact absurd (hn n hn') (by simp [DNode.isNested, hk])

omit hD hN in
theorem optNamesOk?_sound {e : Option Expr} (h : optNamesOk? okE e = true) : OptNames π P ctx₂ ctx₁ e :=
  fun x hx => hE x (List.all_eq_true.1 h x hx)

omit hD hN in
theorem assignNamesOk?_sound {a : Option (Sym × Expr)} (h : assignNamesOk? okE a = true) :
    AssignNames π P ctx₂ ctx₁ a :=
  fun x hx => hE x (List.all_eq_true.1 h x hx)

mutual

theorem Region.namesOk?_sound : ∀ (rg : Region), Region.namesOk? okD okE nested rg = true →
    Region.Names π P ctx₂ ctx₁ rg
  | ⟨_, _, blocks, _, edges⟩, h => by
    simp only [Region.namesOk?, Bool.and_eq_true, List.all_eq_true] at h
    exact ⟨Blocks.namesOk?_sound blocks h.1, fun e he x hx => hE x (h.2 e he x hx)⟩

theorem Blocks.namesOk?_sound : ∀ (bs : List Block), Blocks.namesOk? okD okE nested bs = true →
    Blocks.Names π P ctx₂ ctx₁ bs
  | [], _ => trivial
  | b :: bs, h => by
    simp only [Blocks.namesOk?, Bool.and_eq_true] at h
    exact ⟨Block.namesOk?_sound b h.1, Blocks.namesOk?_sound bs h.2⟩

theorem Block.namesOk?_sound : ∀ (b : Block), Block.namesOk? okD okE nested b = true →
    Block.Names π P ctx₂ ctx₁ b
  | .state s, h => stateNamesOk?_sound hD hE hN h
  | .region r, h => Region.namesOk?_sound r h
  | .loop ⟨_, _, init, cond, update, _, _, body⟩, h => by
    simp only [Block.namesOk?, Bool.and_eq_true] at h
    obtain ⟨⟨⟨hi, hc⟩, hu⟩, hb⟩ := h
    exact ⟨assignNamesOk?_sound hE hi, fun x hx => hE x (List.all_eq_true.1 hc x hx), assignNamesOk?_sound hE hu,
      Region.namesOk?_sound body hb⟩
  | .cond _ brs, h => Branches.namesOk?_sound brs h
  | .breakBlock, _ => trivial
  | .continueBlock, _ => trivial
  | .returnBlock, _ => trivial

theorem Branches.namesOk?_sound : ∀ (brs : List (Option Expr × Region)), Branches.namesOk? okD okE nested brs = true →
    Branches.Names π P ctx₂ ctx₁ brs
  | [], _ => trivial
  | (c, r) :: rest, h => by
    simp only [Branches.namesOk?, Bool.and_eq_true] at h
    exact ⟨optNamesOk?_sound hE h.1.1, Region.namesOk?_sound r h.1.2, Branches.namesOk?_sound rest h.2⟩

end

end Sound

end Dace
