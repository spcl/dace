-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.Det

/-!
# Determinism of whole programs

Control flow is deterministic: an interstate transition is taken only when
exactly one out-edge is true (several true edges are undefined behavior,
`Decisions.md#D2`), loop guards and conditional branches are evaluated in a
fixed order, and assignments, conditions and edge tests read only the
valuation and the memory, never the log.  Together with the determinism of
states (`StateExec.det`, `Dace.Equiv.Det`) this lifts to regions, loops,
conditional blocks and SDFG invocations:

* `CSt.Eqv`: equal valuation and memory, logs equal up to commuting
  reorderings (`Log.Eqv`);
* `BlockExec.det`, `RegionAt.det`, `RegionExec.det`, `LoopExec.det`,
  `LoopGuard.det`, `LoopIter.det`, `LoopTail.det`, `CondExec.det`: if one
  run from a control state is normal, every run from an equivalent control
  state is normal with the same control signal and an equivalent result;
* `SDFGExec.det` and, at program level, `SDFG.Exec.det`: a program whose
  states all pass `stateDet?` (the sequential fragment `seqPure?` with
  convex map scopes, `scopesOk?`), under a lawful WCR semantics, has at most
  one outcome once it has a normal one; in particular no erroneous (e.g.
  racy) outcome (`SDFG.Exec.no_err`).

The proofs recurse on the derivation of the other run and invert the normal
one; the only non-structural facts are `StateExec.det'` (states from
equivalent control states) and the log-independence of `trueEdges`,
`applyAssigns` and `evalBoolR`.
-/

set_option autoImplicit false

namespace Dace

/-! ## Equivalent control states -/

/-- Control states with equal valuation and memory and logs equal up to
commuting reorderings. -/
def CSt.Eqv (a b : CSt) : Prop := a.ρ = b.ρ ∧ a.mem = b.mem ∧ Log.Eqv a.log b.log

theorem CSt.Eqv.refl (a : CSt) : CSt.Eqv a a := ⟨rfl, rfl, Log.Eqv.refl _⟩

/-- The conclusion of the determinism theorems: `r` is normal with the same
control signal as `x` and an equivalent control state. -/
def OkEqv (r : Res (Ctl × CSt)) (x : Ctl × CSt) : Prop :=
  ∃ x', r = .ok x' ∧ x'.1 = x.1 ∧ CSt.Eqv x'.2 x.2

/-- Results related pointwise: normal results by `R`, errors equal. -/
inductive ResRel {α β : Type} (R : α → β → Prop) : Res α → Res β → Prop where
  | ok {a : α} {b : β} (h : R a b) : ResRel R (.ok a) (.ok b)
  | err {e : ErrReason} : ResRel R (.err e) (.err e)

section Eqv

variable {ctx : Ctx} {a b : CSt} (he : CSt.Eqv a b)
include he

theorem CSt.Eqv.trueEdges (rg : Region) (bid : BlockId) :
    Dace.trueEdges ctx rg a bid = Dace.trueEdges ctx rg b bid := by
  unfold Dace.trueEdges; rw [he.1, he.2.1]

theorem CSt.Eqv.evalBoolR (c : Expr) : ctx.evalBoolR a.ρ a.mem c = ctx.evalBoolR b.ρ b.mem c := by
  rw [he.1, he.2.1]

theorem CSt.Eqv.applyAssigns (as : List (Sym × Expr)) :
    ResRel CSt.Eqv (Dace.applyAssigns ctx a as) (Dace.applyAssigns ctx b as) := by
  obtain ⟨hρ, hm, hl⟩ := he
  unfold Dace.applyAssigns
  rw [hρ, hm]
  generalize (List.mapM _ as : Res (List (Sym × Val))) = V
  cases V with
  | ok vs => exact .ok ⟨rfl, rfl, hl⟩
  | err e => exact .err

theorem CSt.Eqv.applyAssign? (o : Option (Sym × Expr)) :
    ResRel CSt.Eqv (Dace.applyAssign? ctx a o) (Dace.applyAssign? ctx b o) := by
  cases o with
  | none => exact .ok he
  | some x => exact he.applyAssigns [x]

end Eqv

/-! ## States from equivalent control states -/

section StateDet

variable {ctx : Ctx} {s : State} (h : SeqPure ctx s) (hl : ctx.I.wcr.Lawful) (hsc : ScopesOk s)
include h hl hsc

/-- `StateExec.det` from equivalent control states. -/
theorem StateExec.det' {a b : CSt} {r : Res (Ctl × CSt)} {x : Ctl × CSt}
    (hy : StateExec ctx a s r) (hx : StateExec ctx b s (.ok x)) (he : CSt.Eqv a b) : OkEqv r x := by
  have hcs : ∀ c ∈ s.components, c.Nodup ∧ s.Convex c :=
    fun c hc => ⟨State.components_nodup hc, State.components_convex hc⟩
  suffices H : ∀ rx, StateExec ctx b s rx → rx = .ok x → OkEqv r x from H _ hx rfl
  intro rx hx hrx
  cases hx with
  | compsErr => cases hrx
  | run hc =>
    rename_i logs
    cases hy with
    | compsErr hc' =>
      rw [he.1, he.2.1] at hc'
      obtain ⟨_, hr, -⟩ := CompsExec.det h hl hsc hcs hc' hc
      cases hr
    | run hc' =>
      rw [he.1, he.2.1] at hc'
      obtain ⟨logs', hr, hlogs⟩ := CompsExec.det h hl hsc hcs hc' hc
      cases hr
      unfold stateFinish at hrx ⊢
      rw [he.2.1]
      cases hm : merge ctx.I.wcr b.mem logs with
      | error e => rw [hm] at hrx; cases hrx
      | ok m' =>
        rw [hm] at hrx
        simp only [Res.ok_inj] at hrx
        subst hrx
        have hpf := (merge_isOk_iff _ _ _).1 ⟨m', hm⟩
        obtain ⟨-, hm'⟩ := merge_le ctx.I.wcr hl b.mem (hlogs.imp fun _ _ h => h.1) hpf
        rw [hm', hm]
        exact ⟨_, rfl, rfl, he.1, rfl, he.2.2.seq (Log.Eqv.unionAll hlogs)⟩

end StateDet

/-! ## The states of a region -/

mutual

/-- Every state of a region (recursively) passes `p`. -/
def Region.statesOk (p : State → Bool) : Region → Bool
  | ⟨_, _, blocks, _, _⟩ => Blocks.statesOk p blocks

def Blocks.statesOk (p : State → Bool) : List Block → Bool
  | [] => true
  | b :: bs => Block.statesOk p b && Blocks.statesOk p bs

def Block.statesOk (p : State → Bool) : Block → Bool
  | .state s => p s
  | .region r => Region.statesOk p r
  | .loop ⟨_, _, _, _, _, _, _, body⟩ => Region.statesOk p body
  | .cond _ brs => Branches.statesOk p brs
  | .breakBlock | .continueBlock | .returnBlock => true

def Branches.statesOk (p : State → Bool) : List (Option Expr × Region) → Bool
  | [] => true
  | (_, r) :: rest => Region.statesOk p r && Branches.statesOk p rest

end

/-- The states the determinism theorems cover: the sequential fragment with
convex map scopes. -/
def stateDet? (ctx : Ctx) (s : State) : Bool := seqPure? ctx s && scopesOk? s

section StatesOk

variable {p : State → Bool}

theorem Region.statesOk_eq (rg : Region) : Region.statesOk p rg = Blocks.statesOk p rg.blocks := by
  cases rg; rfl

theorem Block.statesOk_loop (l : LoopRegion) : Block.statesOk p (.loop l) = Region.statesOk p l.body := by
  cases l; rfl

theorem Blocks.statesOk_mem : ∀ {bs : List Block} {b : Block}, Blocks.statesOk p bs = true → b ∈ bs →
    Block.statesOk p b = true
  | [], _, _, hb => absurd hb List.not_mem_nil
  | _ :: _, _, h, hb => by
    simp only [Blocks.statesOk, Bool.and_eq_true] at h
    rcases List.mem_cons.1 hb with rfl | hb
    · exact h.1
    · exact Blocks.statesOk_mem h.2 hb

theorem Region.statesOk_block {rg : Region} {bid : BlockId} {b : Block} (hok : Region.statesOk p rg = true)
    (hb : rg.block? bid = some b) : Block.statesOk p b = true := by
  rw [Region.statesOk_eq] at hok
  exact Blocks.statesOk_mem hok (List.mem_of_getElem? hb)

end StatesOk

/-! ## Determinism of control flow -/

/-- Unpack `OkEqv` of a normal result. -/
theorem OkEqv.ok_inv {c₁ c₂ : Ctl} {a₁ b₁ : CSt}
    (hdet : OkEqv (.ok (c₁, a₁)) (c₂, b₁)) : c₁ = c₂ ∧ CSt.Eqv a₁ b₁ := by
  obtain ⟨_, hr, h1, h2⟩ := hdet
  cases hr
  exact ⟨h1, h2⟩

section Control

variable {ctx : Ctx} (hl : ctx.I.wcr.Lawful) {p : State → Bool}
  (hp : ∀ s, p s = true → SeqPure ctx s ∧ ScopesOk s)
include hl hp

set_option linter.unusedSectionVars false

mutual

theorem BlockExec.det {a b : CSt} {blk : Block} {r : Res (Ctl × CSt)} {x : Ctl × CSt} :
    BlockExec ctx a blk r → BlockExec ctx b blk (.ok x) → Block.statesOk p blk = true → CSt.Eqv a b →
      OkEqv r x
  | .state hy, hx, hok, he => by
    cases hx with
    | state hx =>
      obtain ⟨h₁, h₂⟩ := hp _ hok
      exact StateExec.det' h₁ hl h₂ hy hx he
  | .region hy, hx, hok, he => by
    cases hx with
    | region hx => exact RegionExec.det hy hx hok he
  | .loop hy, hx, hok, he => by
    cases hx with
    | loop hx => exact LoopExec.det hy hx hok he
  | .cond hy, hx, hok, he => by
    cases hx with
    | cond hx => exact CondExec.det hy hx hok he
  | .brk, hx, _, he => by cases hx; exact ⟨_, rfl, rfl, he⟩
  | .cont, hx, _, he => by cases hx; exact ⟨_, rfl, rfl, he⟩
  | .ret, hx, _, he => by cases hx; exact ⟨_, rfl, rfl, he⟩

theorem RegionAt.det {rg : Region} {a b : CSt} {bid : BlockId} {r : Res (Ctl × CSt)} {x : Ctl × CSt} :
    RegionAt ctx rg a bid r → RegionAt ctx rg b bid (.ok x) → Region.statesOk p rg = true → CSt.Eqv a b →
      OkEqv r x
  | .missing hb, hx, _, _ => by
    exfalso
    cases hx with
    | exit hb' _ _ | stop hb' _ _ | step hb' _ _ _ _ => rw [hb] at hb'; cases hb'
  | .blockErr hb hy, hx, hok, he => by
    exfalso
    cases hx with
    | exit hb' hx _ | stop hb' hx _ | step hb' hx _ _ _ =>
      rw [hb] at hb'; cases hb'
      obtain ⟨_, hr, -⟩ := BlockExec.det hy hx (Region.statesOk_block hok hb) he
      cases hr
  | .exit hb hy hc, hx, hok, he => by
    cases hx with
    | exit hb' hx _ =>
      rw [hb] at hb'; cases hb'
      exact BlockExec.det hy hx (Region.statesOk_block hok hb) he
    | stop hb' hx _ | step hb' hx _ _ _ =>
      rw [hb] at hb'; cases hb'
      exact absurd (OkEqv.ok_inv (BlockExec.det hy hx (Region.statesOk_block hok hb) he)).1 hc
  | .edgesErr hb hy hte, hx, hok, he => by
    exfalso
    cases hx with
    | exit hb' hx hc =>
      rw [hb] at hb'; cases hb'
      exact hc (OkEqv.ok_inv (BlockExec.det hy hx (Region.statesOk_block hok hb) he)).1.symm
    | stop hb' hx hte' | step hb' hx hte' _ _ =>
      rw [hb] at hb'; cases hb'
      have he₁ := (OkEqv.ok_inv (BlockExec.det hy hx (Region.statesOk_block hok hb) he)).2
      rw [he₁.trueEdges, hte'] at hte; cases hte
  | .stop hb hy hte, hx, hok, he => by
    cases hx with
    | exit hb' hx hc =>
      rw [hb] at hb'; cases hb'
      exact absurd (OkEqv.ok_inv (BlockExec.det hy hx (Region.statesOk_block hok hb) he)).1.symm hc
    | stop hb' hx _ =>
      rw [hb] at hb'; cases hb'
      exact BlockExec.det hy hx (Region.statesOk_block hok hb) he
    | step hb' hx hte' _ _ =>
      rw [hb] at hb'; cases hb'
      have he₁ := (OkEqv.ok_inv (BlockExec.det hy hx (Region.statesOk_block hok hb) he)).2
      rw [he₁.trueEdges, hte'] at hte; cases hte
  | .ambiguous hb hy hte, hx, hok, he => by
    exfalso
    cases hx with
    | exit hb' hx hc =>
      rw [hb] at hb'; cases hb'
      exact hc (OkEqv.ok_inv (BlockExec.det hy hx (Region.statesOk_block hok hb) he)).1.symm
    | stop hb' hx hte' | step hb' hx hte' _ _ =>
      rw [hb] at hb'; cases hb'
      have he₁ := (OkEqv.ok_inv (BlockExec.det hy hx (Region.statesOk_block hok hb) he)).2
      rw [he₁.trueEdges, hte'] at hte; cases hte
  | .assignErr (ie := ie) hb hy hte ha, hx, hok, he => by
    exfalso
    cases hx with
    | exit hb' hx hc =>
      rw [hb] at hb'; cases hb'
      exact hc (OkEqv.ok_inv (BlockExec.det hy hx (Region.statesOk_block hok hb) he)).1.symm
    | stop hb' hx hte' =>
      rw [hb] at hb'; cases hb'
      have he₁ := (OkEqv.ok_inv (BlockExec.det hy hx (Region.statesOk_block hok hb) he)).2
      rw [he₁.trueEdges, hte'] at hte; cases hte
    | step hb' hx hte' ha' _ =>
      rw [hb] at hb'; cases hb'
      have he₁ := (OkEqv.ok_inv (BlockExec.det hy hx (Region.statesOk_block hok hb) he)).2
      rw [he₁.trueEdges, hte'] at hte; cases hte
      have hrel := he₁.applyAssigns (ctx := ctx) ie.assigns
      rw [ha, ha'] at hrel; cases hrel
  | .step (ie := ie) hb hy hte ha hn, hx, hok, he => by
    cases hx with
    | exit hb' hx hc =>
      rw [hb] at hb'; cases hb'
      exact absurd (OkEqv.ok_inv (BlockExec.det hy hx (Region.statesOk_block hok hb) he)).1.symm hc
    | stop hb' hx hte' =>
      rw [hb] at hb'; cases hb'
      have he₁ := (OkEqv.ok_inv (BlockExec.det hy hx (Region.statesOk_block hok hb) he)).2
      rw [he₁.trueEdges, hte'] at hte; cases hte
    | step hb' hx hte' ha' hn' =>
      rw [hb] at hb'; cases hb'
      have he₁ := (OkEqv.ok_inv (BlockExec.det hy hx (Region.statesOk_block hok hb) he)).2
      rw [he₁.trueEdges, hte'] at hte; cases hte
      have hrel := he₁.applyAssigns (ctx := ctx) ie.assigns
      rw [ha, ha'] at hrel
      cases hrel with
      | ok he₂ => exact RegionAt.det hn hn' hok he₂

theorem RegionExec.det {rg : Region} {a b : CSt} {r : Res (Ctl × CSt)} {x : Ctl × CSt} :
    RegionExec ctx a rg r → RegionExec ctx b rg (.ok x) → Region.statesOk p rg = true → CSt.Eqv a b →
      OkEqv r x
  | .start hy, hx, hok, he => by
    cases hx with
    | start hx => exact RegionAt.det hy hx hok he

theorem LoopExec.det {l : LoopRegion} {a b : CSt} {r : Res (Ctl × CSt)} {x : Ctl × CSt} :
    LoopExec ctx a l r → LoopExec ctx b l (.ok x) → Block.statesOk p (.loop l) = true → CSt.Eqv a b →
      OkEqv r x
  | .initErr hi, hx, _, he => by
    exfalso
    cases hx with
    | guardFirst hi' _ _ | bodyFirst hi' _ _ =>
      have hrel := he.applyAssign? (ctx := ctx) l.init
      rw [hi, hi'] at hrel; cases hrel
  | .guardFirst hi hv hg, hx, hok, he => by
    cases hx with
    | guardFirst hi' _ hg' =>
      have hrel := he.applyAssign? (ctx := ctx) l.init
      rw [hi, hi'] at hrel
      cases hrel with
      | ok he₁ => exact LoopGuard.det hg hg' hok he₁
    | bodyFirst _ hv' _ => rw [hv] at hv'; cases hv'
  | .bodyFirst hi hv hb, hx, hok, he => by
    cases hx with
    | guardFirst _ hv' _ => rw [hv] at hv'; cases hv'
    | bodyFirst hi' _ hb' =>
      have hrel := he.applyAssign? (ctx := ctx) l.init
      rw [hi, hi'] at hrel
      cases hrel with
      | ok he₁ => exact LoopIter.det hb hb' hok he₁

theorem LoopGuard.det {l : LoopRegion} {a b : CSt} {r : Res (Ctl × CSt)} {x : Ctl × CSt} :
    LoopGuard ctx a l r → LoopGuard ctx b l (.ok x) → Block.statesOk p (.loop l) = true → CSt.Eqv a b →
      OkEqv r x
  | .condErr hc, hx, _, he => by
    exfalso
    cases hx with
    | exit hc' | enter hc' _ => rw [he.evalBoolR, hc'] at hc; cases hc
  | .exit hc, hx, _, he => by
    cases hx with
    | exit _ => exact ⟨_, rfl, rfl, he⟩
    | enter hc' _ => rw [he.evalBoolR, hc'] at hc; cases hc
  | .enter hc hb, hx, hok, he => by
    cases hx with
    | exit hc' => rw [he.evalBoolR, hc'] at hc; cases hc
    | enter _ hb' => exact LoopIter.det hb hb' hok he

theorem LoopIter.det {l : LoopRegion} {a b : CSt} {r : Res (Ctl × CSt)} {x : Ctl × CSt} :
    LoopIter ctx a l r → LoopIter ctx b l (.ok x) → Block.statesOk p (.loop l) = true → CSt.Eqv a b →
      OkEqv r x
  | .bodyErr hb, hx, hok, he => by
    exfalso
    have hok' : Region.statesOk p l.body = true := by rw [← Block.statesOk_loop]; exact hok
    cases hx with
    | ret hb' | brk hb' | tail hb' _ _ =>
      obtain ⟨_, hr, -⟩ := RegionExec.det hb hb' hok' he
      cases hr
  | .ret hb, hx, hok, he => by
    have hok' : Region.statesOk p l.body = true := by rw [← Block.statesOk_loop]; exact hok
    cases hx with
    | ret hb' => exact RegionExec.det hb hb' hok' he
    | brk hb' => cases (OkEqv.ok_inv (RegionExec.det hb hb' hok' he)).1
    | tail hb' hc _ =>
      rw [← (OkEqv.ok_inv (RegionExec.det hb hb' hok' he)).1] at hc; cases hc
  | .brk hb, hx, hok, he => by
    have hok' : Region.statesOk p l.body = true := by rw [← Block.statesOk_loop]; exact hok
    cases hx with
    | ret hb' => cases (OkEqv.ok_inv (RegionExec.det hb hb' hok' he)).1
    | brk hb' => exact ⟨_, rfl, rfl, (OkEqv.ok_inv (RegionExec.det hb hb' hok' he)).2⟩
    | tail hb' hc _ =>
      rw [← (OkEqv.ok_inv (RegionExec.det hb hb' hok' he)).1] at hc; cases hc
  | .tail hb hc ht, hx, hok, he => by
    have hok' : Region.statesOk p l.body = true := by rw [← Block.statesOk_loop]; exact hok
    cases hx with
    | ret hb' => rw [(OkEqv.ok_inv (RegionExec.det hb hb' hok' he)).1] at hc; cases hc
    | brk hb' => rw [(OkEqv.ok_inv (RegionExec.det hb hb' hok' he)).1] at hc; cases hc
    | tail hb' _ ht' => exact LoopTail.det ht ht' hok (OkEqv.ok_inv (RegionExec.det hb hb' hok' he)).2

theorem LoopTail.det {l : LoopRegion} {a b : CSt} {r : Res (Ctl × CSt)} {x : Ctl × CSt} :
    LoopTail ctx a l r → LoopTail ctx b l (.ok x) → Block.statesOk p (.loop l) = true → CSt.Eqv a b →
      OkEqv r x
  | .updateErr hv hu, hx, _, he => by
    exfalso
    cases hx with
    | updateGuard _ hu' _ =>
      have hrel := he.applyAssign? (ctx := ctx) l.update
      rw [hu, hu'] at hrel; cases hrel
    | condFirstExit hv' _ | condFirstNext hv' _ _ _ => rw [hv] at hv'; cases hv'
  | .updateGuard hv hu hg, hx, hok, he => by
    cases hx with
    | updateGuard _ hu' hg' =>
      have hrel := he.applyAssign? (ctx := ctx) l.update
      rw [hu, hu'] at hrel
      cases hrel with
      | ok he₁ => exact LoopGuard.det hg hg' hok he₁
    | condFirstExit hv' _ | condFirstNext hv' _ _ _ => rw [hv] at hv'; cases hv'
  | .condFirstErr hv hc, hx, _, he => by
    exfalso
    cases hx with
    | updateGuard hv' _ _ => rw [hv] at hv'; cases hv'
    | condFirstExit _ hc' | condFirstNext _ hc' _ _ => rw [he.evalBoolR, hc'] at hc; cases hc
  | .condFirstExit hv hc, hx, _, he => by
    cases hx with
    | updateGuard hv' _ _ => rw [hv] at hv'; cases hv'
    | condFirstExit _ _ => exact ⟨_, rfl, rfl, he⟩
    | condFirstNext _ hc' _ _ => rw [he.evalBoolR, hc'] at hc; cases hc
  | .condFirstUpdateErr hv hc hu, hx, _, he => by
    exfalso
    cases hx with
    | updateGuard hv' _ _ => rw [hv] at hv'; cases hv'
    | condFirstExit _ hc' => rw [he.evalBoolR, hc'] at hc; cases hc
    | condFirstNext _ _ hu' _ =>
      have hrel := he.applyAssign? (ctx := ctx) l.update
      rw [hu, hu'] at hrel; cases hrel
  | .condFirstNext hv hc hu hb, hx, hok, he => by
    cases hx with
    | updateGuard hv' _ _ => rw [hv] at hv'; cases hv'
    | condFirstExit _ hc' => rw [he.evalBoolR, hc'] at hc; cases hc
    | condFirstNext _ _ hu' hb' =>
      have hrel := he.applyAssign? (ctx := ctx) l.update
      rw [hu, hu'] at hrel
      cases hrel with
      | ok he₁ => exact LoopIter.det hb hb' hok he₁

theorem CondExec.det {brs : List (Option Expr × Region)} {a b : CSt} {r : Res (Ctl × CSt)} {x : Ctl × CSt} :
    CondExec ctx a brs r → CondExec ctx b brs (.ok x) → Branches.statesOk p brs = true → CSt.Eqv a b →
      OkEqv r x
  | .nil, hx, _, he => by cases hx; exact ⟨_, rfl, rfl, he⟩
  | .els hy, hx, hok, he => by
    simp only [Branches.statesOk, Bool.and_eq_true] at hok
    cases hx with
    | els hx => exact RegionExec.det hy hx hok.1 he
  | .condErr hc, hx, _, he => by
    exfalso
    cases hx with
    | taken hc' _ | skipped hc' _ => rw [he.evalBoolR, hc'] at hc; cases hc
  | .taken hc hy, hx, hok, he => by
    simp only [Branches.statesOk, Bool.and_eq_true] at hok
    cases hx with
    | taken _ hx => exact RegionExec.det hy hx hok.1 he
    | skipped hc' _ => rw [he.evalBoolR, hc'] at hc; cases hc
  | .skipped hc hy, hx, hok, he => by
    simp only [Branches.statesOk, Bool.and_eq_true] at hok
    cases hx with
    | taken hc' _ => rw [he.evalBoolR, hc'] at hc; cases hc
    | skipped _ hx => exact CondExec.det hy hx hok.2 he

end

/-- **Determinism of an SDFG invocation**: if one run is normal, every run
is normal with the same valuation and memory and an equivalent log. -/
theorem SDFGExec.det {ρ : Valuation} {m : Mem} {g : SDFG} {r : Res (Valuation × Mem × Log)}
    {ρ' : Valuation} {m' : Mem} {l : Log} (hok : Region.statesOk p g.top = true)
    (hy : SDFGExec ctx ρ m g r) (hx : SDFGExec ctx ρ m g (.ok (ρ', m', l))) :
    ∃ l', r = .ok (ρ', m', l') ∧ Log.Eqv l' l := by
  suffices H : ∀ rx, SDFGExec ctx ρ m g rx → rx = .ok (ρ', m', l) → ∃ l', r = .ok (ρ', m', l') ∧ Log.Eqv l' l
    from H _ hx rfl
  intro rx hx₀ hrx
  cases hx₀ with
  | err => cases hrx
  | escape => cases hrx
  | run hreg hc =>
    simp only [Res.ok_inj, Prod.mk.injEq] at hrx
    obtain ⟨rfl, rfl, rfl⟩ := hrx
    cases hy with
    | err hy =>
      obtain ⟨_, hr, -⟩ := RegionExec.det hl hp hy hreg hok (CSt.Eqv.refl _)
      cases hr
    | escape hy hc' =>
      have h₁ := (OkEqv.ok_inv (RegionExec.det hl hp hy hreg hok (CSt.Eqv.refl _))).1
      subst h₁
      simp only [Bool.or_eq_true, beq_iff_eq] at hc hc'
      rcases hc' with h | h <;> rcases hc with h' | h' <;> rw [h] at h' <;> cases h'
    | run hy _ =>
      obtain ⟨-, he⟩ := OkEqv.ok_inv (RegionExec.det hl hp hy hreg hok (CSt.Eqv.refl _))
      exact ⟨_, by rw [he.1, he.2.1], he.2.2⟩

end Control

/-! ## Program level -/

/-- The determinism check of an SDFG: every state is in the sequential
fragment with convex map scopes.  It does not depend on the interpretation
(`SDFG.Exec.det`). -/
def SDFG.det? (g : SDFG) : Bool :=
  Region.statesOk (stateDet? { I := Interp.uninterpreted, g }) g.top

theorem stateDet?_sound {I : Interp} {g : SDFG} {s : State}
    (h : stateDet? { I := Interp.uninterpreted, g } s = true) :
    SeqPure { I, g } s ∧ ScopesOk s := by
  simp only [stateDet?, Bool.and_eq_true] at h
  have hs := seqPure?_sound h.1
  exact ⟨⟨⟨⟨hs.plain, hs.shapes, hs.memlets, hs.ranges, hs.reduces, hs.codeOut⟩, hs.kinds⟩, hs.noPriv⟩, scopesOk?_sound h.2⟩

/-- **Determinism of programs**: for a lawful WCR semantics, a program
passing `SDFG.det?` with a normal outcome has no other outcome. -/
theorem SDFG.Exec.det {I : Interp} {g : SDFG} {ρ₀ : Valuation} {m₀ : Mem} {r : Res Mem} {m : Mem}
    (hl : I.wcr.Lawful) (hok : g.det? = true) (hy : g.Exec I ρ₀ m₀ r) (hx : g.Exec I ρ₀ m₀ (.ok m)) :
    r = .ok m := by
  obtain ⟨res, hy, rfl⟩ := hy
  obtain ⟨resx, hx, hmx⟩ := hx
  cases resx with
  | err => cases hmx
  | ok y =>
    obtain ⟨ρ', m', l⟩ := y
    simp only [Res.map_ok, Res.ok_inj] at hmx
    subst hmx
    obtain ⟨_, rfl, -⟩ := SDFGExec.det (ctx := { I, g }) hl (fun _ h => stateDet?_sound h) hok hy hx
    rfl

/-- A program passing `SDFG.det?` with a normal outcome has no erroneous
(e.g. racy) outcome. -/
theorem SDFG.Exec.no_err {I : Interp} {g : SDFG} {ρ₀ : Valuation} {m₀ : Mem} {m : Mem} {e : ErrReason}
    (hl : I.wcr.Lawful) (hok : g.det? = true) (hx : g.Exec I ρ₀ m₀ (.ok m)) : ¬ g.Exec I ρ₀ m₀ (.err e) :=
  fun hy => by cases SDFG.Exec.det hl hok hy hx

end Dace
