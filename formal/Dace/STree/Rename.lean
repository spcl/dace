-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Sim
import Dace.STree.Semantics

/-!
# Renaming containers in a schedule tree

A nested SDFG may name a container differently from its caller (a connector
`_in` bound to the outer `A`), like `symbol_mapping` does for symbols; the
schedule tree inlines the nested SDFG under the outer names.  To match the
inlined tree against the nested SDFG, the tree is renamed to the inner names
by a permutation `σ` of names (outer name ↦ inner name) and run in a context
whose lookups agree through `σ` (`Ctx.Ren`): the descriptor of `σ x` is that
of `x`, and so is its memory name.  Memory names are unchanged, so the
renamed run ends in the same state; only error messages, which name
containers, may differ.

* `SNode.ren`, `Memlet.ren`, …: rename the container names of memlets, copy
  targets, views and reference sets.  Expressions are not renamed: renamed
  names must not occur in them (`ExprFixed`), which also keeps symbols and
  containers apart (a bare name may be a symbol, `Decisions.md#D33`).
* `Res.get?`: an outcome without its error.  The leaf functions renamed
  agree on `get?` (`Ctx.Ren.*`, `TCtx.Ren.*`, `TCtx.Ren.stepLeaf`).
* `STree.NodeExec.ren`, `STree.ListExec.ren`, …: a run of a tree list is
  matched by a run of the renamed list, with the same final state, for the
  nodes `SNode.renOk` accepts (decidable: no consume scopes, expressions over
  fixed names, map scopes with agreeing private transients).
-/

set_option autoImplicit false

namespace Dace

/-! ## Outcomes up to errors -/

namespace Res

/-- The result of an outcome, forgetting the error. -/
def get? {α : Type} : Res α → Option α
  | ok a => some a
  | err _ => none

variable {α β : Type}

@[simp] theorem get?_ok (a : α) : (ok a : Res α).get? = some a := rfl
@[simp] theorem get?_err (e : ErrReason) : (err e : Res α).get? = none := rfl
@[simp] theorem get?_pure (a : α) : (pure a : Res α).get? = some a := rfl

@[simp] theorem get?_bind (r : Res α) (f : α → Res β) : (r >>= f).get? = r.get?.bind fun a => (f a).get? := by
  cases r <;> rfl

@[simp] theorem get?_bind' (r : Res α) (f : α → Res β) : (r.bind f).get? = r.get?.bind fun a => (f a).get? := by
  cases r <;> rfl

@[simp] theorem get?_map (f : α → β) (r : Res α) : (r.map f).get? = r.get?.map f := by
  cases r <;> rfl

@[simp] theorem get?_ofOption (o : Option α) (e : ErrReason) : (ofOption o e).get? = o := by
  cases o <;> rfl

theorem get?_eq_none {r : Res α} : r.get? = none ↔ ∃ e, r = err e := by
  cases r <;> simp

theorem get?_eq_some {r : Res α} {a : α} : r.get? = some a ↔ r = ok a := by
  cases r <;> simp

/-- Monadic folds agree on results when every step does. -/
theorem get?_foldlM {τ : Type} {f₂ f₁ : α → τ → Res α} (hf : ∀ a x, (f₂ a x).get? = (f₁ a x).get?) :
    ∀ (xs : List τ) (a : α), (xs.foldlM f₂ a).get? = (xs.foldlM f₁ a).get?
  | [], _ => rfl
  | x :: xs, a => by
    simp only [List.foldlM_cons, get?_bind, hf]
    cases (f₁ a x).get? with
    | none => rfl
    | some b => exact get?_foldlM hf xs b

theorem get?_foldlM_mem {τ : Type} {f₂ f₁ : α → τ → Res α} :
    ∀ (xs : List τ) (a : α), (∀ a, ∀ x ∈ xs, (f₂ a x).get? = (f₁ a x).get?) →
      (xs.foldlM f₂ a).get? = (xs.foldlM f₁ a).get?
  | [], _, _ => rfl
  | x :: xs, a, hf => by
    simp only [List.foldlM_cons, get?_bind, hf a x List.mem_cons_self]
    cases (f₁ a x).get? with
    | none => rfl
    | some b => exact get?_foldlM_mem xs b fun a y hy => hf a y (List.mem_cons_of_mem _ hy)

theorem get?_mapM {τ : Type} {f₂ f₁ : τ → Res α} :
    ∀ (xs : List τ), (∀ x ∈ xs, (f₂ x).get? = (f₁ x).get?) → (xs.mapM f₂).get? = (xs.mapM f₁).get?
  | [], _ => rfl
  | x :: xs, hf => by
    simp only [List.mapM_cons, get?_bind, hf x List.mem_cons_self,
      get?_mapM xs fun y hy => hf y (List.mem_cons_of_mem _ hy)]

end Res

/-! ## Renaming syntax -/

section Syntax

variable (σ : Equiv.Perm Name)

/-- Rename the container of a memlet. -/
def Memlet.ren (m : Memlet) : Memlet := { m with data := σ m.data }

def MemletMap.ren : MemletMap → MemletMap
  | .dict items => .dict (items.map fun p => (p.1, p.2.ren σ))
  | .set items => .set (items.map (Memlet.ren σ))

mutual

/-- Rename the containers of a tree node (memlets, copy targets, views and
reference sets); expressions, symbols and labels stay. -/
def SNode.ren : SNode → SNode
  | .namedRegion l cs => .namedRegion l (SNodes.ren cs)
  | .gblock cs => .gblock (SNodes.ren cs)
  | .stateLabel n => .stateLabel n
  | .goto t => .goto t
  | .assign n v => .assign n v
  | .loopScope l cs => .loopScope l (SNodes.ren cs)
  | .forScope l cs => .forScope l (SNodes.ren cs)
  | .whileScope l cs => .whileScope l (SNodes.ren cs)
  | .doWhileScope l cs => .doWhileScope l (SNodes.ren cs)
  | .ifScope c cs => .ifScope c (SNodes.ren cs)
  | .stateIfScope c cs => .stateIfScope c (SNodes.ren cs)
  | .breakNode => .breakNode
  | .continueNode => .continueNode
  | .elifScope c cs => .elifScope c (SNodes.ren cs)
  | .elseScope cs => .elseScope (SNodes.ren cs)
  | .mapScope mp cs => .mapScope mp (SNodes.ren cs)
  | .consumeScope c cs => .consumeScope c (SNodes.ren cs)
  | .taskletNode t ins outs => .taskletNode t (ins.ren σ) (outs.ren σ)
  | .libraryCall l props ins outs => .libraryCall l props (ins.ren σ) (outs.ren σ)
  | .copyNode target m => .copyNode (σ target) (m.ren σ)
  | .dynScopeCopy target m => .dynScopeCopy target (m.ren σ)
  | .viewNode target source m sd vd => .viewNode (σ target) (σ source) (m.ren σ) sd vd
  | .refSetNode target m src d => .refSetNode (σ target) (m.ren σ) src d

def SNodes.ren : List SNode → List SNode
  | [] => []
  | n :: ns => SNode.ren n :: SNodes.ren ns

end

/-- Rename the stream of the pending `IN_stream` input. -/
def Pending.ren (p : Pending) : Pending := { p with stream := p.stream.map fun q => (σ q.1, q.2) }

def SibCtx.ren (sc : SibCtx) : SibCtx := { sc with pending := sc.pending.ren σ }

/-- Rename the sibling context of a node outcome. -/
def NodeOut.ren (o : NodeOut) : NodeOut := (o.1.ren σ, o.2.1, o.2.2)

/-- The context whose lookups at `σ x` are those of `ctx` at `x`. -/
def Ctx.ren (ctx : Ctx) : Ctx :=
  { ctx with g := { ctx.g with arrays := ctx.g.arrays.map fun p => (σ p.1, p.2) },
             cname := fun n => ctx.cname (σ.symm n) }

end Syntax

theorem SNodes.ren_eq (σ : Equiv.Perm Name) : ∀ cs : List SNode, SNodes.ren σ cs = cs.map (SNode.ren σ)
  | [] => rfl
  | c :: cs => by simp [SNodes.ren, SNodes.ren_eq σ cs]

theorem SNodes.ren_append (σ : Equiv.Perm Name) (a b : List SNode) :
    SNodes.ren σ (a ++ b) = SNodes.ren σ a ++ SNodes.ren σ b := by
  simp [SNodes.ren_eq]

@[simp] theorem Memlet.ren_data (σ : Equiv.Perm Name) (m : Memlet) : (m.ren σ).data = σ m.data := rfl
@[simp] theorem Memlet.ren_dataSubset (σ : Equiv.Perm Name) (m : Memlet) : (m.ren σ).dataSubset = m.dataSubset := rfl
@[simp] theorem Memlet.ren_otherSubset (σ : Equiv.Perm Name) (m : Memlet) :
    (m.ren σ).otherSubset = m.otherSubset := rfl
@[simp] theorem Memlet.ren_wcr (σ : Equiv.Perm Name) (m : Memlet) : (m.ren σ).wcr = m.wcr := rfl
@[simp] theorem Memlet.ren_allowOob (σ : Equiv.Perm Name) (m : Memlet) : (m.ren σ).allowOob = m.allowOob := rfl
@[simp] theorem Memlet.ren_dynamic (σ : Equiv.Perm Name) (m : Memlet) : (m.ren σ).dynamic = m.dynamic := rfl
@[simp] theorem Memlet.ren_hasWcr (σ : Equiv.Perm Name) (m : Memlet) : (m.ren σ).hasWcr = m.hasWcr := rfl

theorem MemletMap.ren_pairs (σ : Equiv.Perm Name) (mm : MemletMap) :
    (mm.ren σ).pairs = mm.pairs.map fun p => (p.1, p.2.ren σ) := by
  cases mm <;> simp [MemletMap.ren, MemletMap.pairs]

theorem MemletMap.ren_memlets (σ : Equiv.Perm Name) (mm : MemletMap) :
    (mm.ren σ).memlets = mm.memlets.map (Memlet.ren σ) := by
  cases mm <;> simp [MemletMap.ren, MemletMap.memlets, Function.comp_def]

theorem MemletMap.ren_on (σ : Equiv.Perm Name) (mm : MemletMap) (c : Conn) :
    (mm.ren σ).on c = (mm.on c).map (Memlet.ren σ) := by
  cases mm with
  | dict items =>
    simp only [MemletMap.ren, MemletMap.on, List.filter_map, List.map_map]
    rfl
  | set items =>
    simp only [MemletMap.ren, MemletMap.on]
    split <;> simp

theorem MemletMap.snd_mem_memlets {mm : MemletMap} {p : Conn × Memlet} (hp : p ∈ mm.pairs) : p.2 ∈ mm.memlets := by
  cases mm with
  | dict items => exact List.mem_map_of_mem hp
  | set items =>
    simp only [MemletMap.pairs, List.mem_map] at hp
    obtain ⟨m, hm, rfl⟩ := hp
    exact hm

theorem MemletMap.mem_memlets_of_on {mm : MemletMap} {c : Conn} {m : Memlet} (hm : m ∈ mm.on c) : m ∈ mm.memlets := by
  cases mm with
  | dict items =>
    simp only [MemletMap.on, List.mem_map, List.mem_filter] at hm
    obtain ⟨p, ⟨hp, -⟩, rfl⟩ := hm
    exact List.mem_map_of_mem hp
  | set items =>
    simp only [MemletMap.on] at hm
    split at hm
    · exact hm
    · cases hm

theorem MemletMap.ren_hasDynamicWcr (σ : Equiv.Perm Name) (mm : MemletMap) :
    (mm.ren σ).hasDynamicWcr = mm.hasDynamicWcr := by
  simp [MemletMap.hasDynamicWcr, MemletMap.ren_memlets, Function.comp_def]

/-! ## Contexts agreeing through a renaming -/

/-- `ctx'` agrees with `ctx` through `σ`: the same interpretation, and the
descriptor and memory name of `σ x` in `ctx'` are those of `x` in `ctx`. -/
structure Ctx.Ren (σ : Equiv.Perm Name) (ctx' ctx : Ctx) : Prop where
  I : ctx'.I = ctx.I
  desc : ∀ x, ctx'.desc? (σ x) = ctx.desc? x
  cname : ∀ x, ctx'.cname (σ x) = ctx.cname x

theorem SDFG.array?_map_ren (σ : Equiv.Perm Name) (g : SDFG) (x : Name) :
    ({ g with arrays := g.arrays.map fun p => (σ p.1, p.2) } : SDFG).array? (σ x) = g.array? x := by
  simp only [SDFG.array?, List.find?_map, Function.comp_def, EmbeddingLike.apply_eq_iff_eq]
  cases List.find? (fun p => decide (p.1 = x)) g.arrays <;> rfl

theorem Ctx.ren_ren (σ : Equiv.Perm Name) (ctx : Ctx) : Ctx.Ren σ (ctx.ren σ) ctx where
  I := rfl
  desc x := SDFG.array?_map_ren σ ctx.g x
  cname x := by simp [Ctx.ren]

namespace Ctx.Ren

variable {σ : Equiv.Perm Name} {ctx' ctx : Ctx} (h : Ctx.Ren σ ctx' ctx)
include h

theorem desc_fixed {x : Name} (hx : σ x = x) : ctx'.desc? x = ctx.desc? x := by
  have := h.desc x; rwa [hx] at this

theorem cname_fixed {x : Name} (hx : σ x = x) : ctx'.cname x = ctx.cname x := by
  have := h.cname x; rwa [hx] at this

theorem isStream (n : Name) : ctx'.isStream (σ n) = ctx.isStream n := by
  simp [Ctx.isStream, h.desc]

theorem isView (n : Name) : ctx'.isView (σ n) = ctx.isView n := by
  simp [Ctx.isView, h.desc]

theorem isReference (n : Name) : ctx'.isReference (σ n) = ctx.isReference n := by
  simp [Ctx.isReference, h.desc]

theorem elemOffset (ρ : Valuation) (d : DataDesc) (idx : Idx) : ctx'.elemOffset ρ d idx = ctx.elemOffset ρ d idx := by
  simp [Ctx.elemOffset, DataDesc.linearOffset, h.I]

theorem addr (ρ : Valuation) (m : Mem) (n : Name) (idx : Idx) :
    (ctx'.addr ρ m (σ n) idx).get? = (ctx.addr ρ m n idx).get? := by
  unfold Ctx.addr
  simp only [h.desc, h.cname, h.elemOffset]
  cases ctx.desc? n with
  | none => rfl
  | some d =>
    simp only
    split <;> [rfl; (cases ctx.elemOffset ρ d idx <;> rfl)]

theorem checkBounds (ρ : Valuation) (n : Name) (idx : Idx) (oob : Bool) :
    (ctx'.checkBounds ρ (σ n) idx oob).get? = (ctx.checkBounds ρ n idx oob).get? := by
  unfold Ctx.checkBounds
  rw [h.desc, h.I]
  cases ctx.desc? n with
  | none => rfl
  | some d =>
    simp only
    split
    · rfl
    · simp only [Res.get?_bind]
      congr 1
      funext shape
      split <;> rfl

theorem loc (ρ : Valuation) (m : Mem) (n : Name) (idx : Idx) (oob : Bool) :
    (ctx'.loc ρ m (σ n) idx oob).get? = (ctx.loc ρ m n idx oob).get? := by
  unfold Ctx.loc
  simp only [Res.get?_bind, h.addr, h.checkBounds]

end Ctx.Ren

/-! ## Expressions over names the renaming fixes -/

theorem List.mapM_congr_res {α β : Type} {f g : α → Res β} :
    ∀ (l : List α), (∀ a ∈ l, f a = g a) → l.mapM f = l.mapM g
  | [], _ => rfl
  | a :: l, h => by
    simp only [List.mapM_cons, h a List.mem_cons_self,
      List.mapM_congr_res l fun b hb => h b (List.mem_cons_of_mem _ hb)]

/-- Every name of the list is fixed by `σ`. -/
def NamesFixed (σ : Equiv.Perm Name) (xs : List Name) : Prop := ∀ x ∈ xs, σ x = x

/-- The full ranges of all descriptors only read names fixed by `σ`. -/
def Ctx.DescFixed (σ : Equiv.Perm Name) (ctx : Ctx) : Prop :=
  ∀ n d, ctx.desc? n = some d → NamesFixed σ d.fullRange.names

theorem OptLe.antisymm {α : Type} {a b : Option α} (h₁ : OptLe a b) (h₂ : OptLe b a) : a = b := by
  rcases h₁ with rfl | rfl
  · rcases h₂ with rfl | rfl <;> rfl
  · rfl

namespace Ctx.Ren

variable {σ : Equiv.Perm Name} {ctx' ctx : Ctx} (h : Ctx.Ren σ ctx' ctx)
include h

theorem readData (ρ : Valuation) (m : Mem) {d : Name} (hd : σ d = d) (is : List Int) :
    (ctx'.exprInterp ρ m).readData d is = (ctx.exprInterp ρ m).readData d is := by
  have hl := h.loc ρ m d is false
  rw [hd] at hl
  simp only [Ctx.exprInterp]
  cases h₁ : ctx'.loc ρ m d is <;> cases h₂ : ctx.loc ρ m d is <;> simp_all

theorem val (ρ : Valuation) (m : Mem) {x : Name} (hx : σ x = x) : ctx'.val ρ m x = ctx.val ρ m x := by
  have hl := h.loc ρ m x [] false
  rw [hx] at hl
  simp only [Ctx.val, h.desc_fixed hx]
  cases ρ x with
  | some _ => rfl
  | none =>
    simp only
    cases ctx.desc? x with
    | none => rfl
    | some d =>
      simp only
      split
      · cases h₁ : ctx'.loc ρ m x [] <;> cases h₂ : ctx.loc ρ m x [] <;> simp_all
      · rfl

theorem happ (ρ : Valuation) (m : Mem) : (ctx'.exprInterp ρ m).app = (ctx.exprInterp ρ m).app := by
  simp [Ctx.exprInterp, h.I]

theorem hattr (ρ : Valuation) (m : Mem) : (ctx'.exprInterp ρ m).readAttr = (ctx.exprInterp ρ m).readAttr := by
  simp [Ctx.exprInterp, h.I]

theorem le (ρ : Valuation) (m : Mem) :
    (∀ d is, σ d = d → OptLe ((ctx'.exprInterp ρ m).readData d is) ((ctx.exprInterp ρ m).readData d is)) ∧
    (∀ x, σ x = x → OptLe (ctx'.val ρ m x) (ctx.val ρ m x)) :=
  ⟨fun _ is hd => .inr (h.readData ρ m hd is), fun _ hx => .inr (h.val ρ m hx)⟩

theorem ge (ρ : Valuation) (m : Mem) :
    (∀ d is, σ d = d → OptLe ((ctx.exprInterp ρ m).readData d is) ((ctx'.exprInterp ρ m).readData d is)) ∧
    (∀ x, σ x = x → OptLe (ctx.val ρ m x) (ctx'.val ρ m x)) :=
  ⟨fun _ is hd => .inr (h.readData ρ m hd is).symm, fun _ hx => .inr (h.val ρ m hx).symm⟩

theorem eval (ρ : Valuation) (m : Mem) (e : Expr) (he : NamesFixed σ e.names) :
    ctx'.eval ρ m e = ctx.eval ρ m e :=
  OptLe.antisymm
    (Expr.eval_mono' (h.happ ρ m) (h.hattr ρ m) (h.le ρ m).1 (h.le ρ m).2 e he)
    (Expr.eval_mono' (h.happ ρ m).symm (h.hattr ρ m).symm (h.ge ρ m).1 (h.ge ρ m).2 e he)

theorem evalInt (ρ : Valuation) (m : Mem) (e : Expr) (he : NamesFixed σ e.names) :
    ctx'.evalInt ρ m e = ctx.evalInt ρ m e := by
  have := h.eval ρ m e he
  simp only [Ctx.eval] at this
  simp only [Ctx.evalInt, Expr.evalInt, this]

theorem evalBool (ρ : Valuation) (m : Mem) (e : Expr) (he : NamesFixed σ e.names) :
    ctx'.evalBool ρ m e = ctx.evalBool ρ m e := by
  have := h.eval ρ m e he
  simp only [Ctx.eval] at this
  simp only [Ctx.evalBool, Expr.evalBool, this]

theorem evalR (ρ : Valuation) (m : Mem) (e : Expr) (he : NamesFixed σ e.names) :
    ctx'.evalR ρ m e = ctx.evalR ρ m e := by
  simp only [Ctx.evalR, h.eval ρ m e he]

theorem evalIntR (ρ : Valuation) (m : Mem) (e : Expr) (he : NamesFixed σ e.names) :
    ctx'.evalIntR ρ m e = ctx.evalIntR ρ m e := by
  simp only [Ctx.evalIntR, h.evalInt ρ m e he]

theorem evalBoolR (ρ : Valuation) (m : Mem) (e : Expr) (he : NamesFixed σ e.names) :
    ctx'.evalBoolR ρ m e = ctx.evalBoolR ρ m e := by
  simp only [Ctx.evalBoolR, h.evalBool ρ m e he]

theorem rangePoints' (ρ : Valuation) (m : Mem) (r : Range) (hr : NamesFixed σ r.names) :
    r.points (ctx'.exprInterp ρ m) (ctx'.val ρ m) = r.points (ctx.exprInterp ρ m) (ctx.val ρ m) :=
  OptLe.antisymm
    (Range.points_mono (h.happ ρ m) (h.hattr ρ m) (h.le ρ m).1 (h.le ρ m).2 r hr)
    (Range.points_mono (h.happ ρ m).symm (h.hattr ρ m).symm (h.ge ρ m).1 (h.ge ρ m).2 r hr)

theorem subsetPoints (ρ : Valuation) (m : Mem) (S : Subset) (hS : NamesFixed σ S.names) :
    S.points (ctx'.exprInterp ρ m) (ctx'.val ρ m) = S.points (ctx.exprInterp ρ m) (ctx.val ρ m) :=
  OptLe.antisymm
    (Subset.points_mono (h.happ ρ m) (h.hattr ρ m) (h.le ρ m).1 (h.le ρ m).2 S hS)
    (Subset.points_mono (h.happ ρ m).symm (h.hattr ρ m).symm (h.ge ρ m).1 (h.ge ρ m).2 S hS)

theorem rangePoints (ρ : Valuation) (m : Mem) (r : Range) (hr : NamesFixed σ r.names) :
    ctx'.rangePoints ρ m r = ctx.rangePoints ρ m r := by
  simp only [Ctx.rangePoints, h.rangePoints' ρ m r hr]

theorem pointsOf (hD : ctx.DescFixed σ) (ρ : Valuation) (m : Mem) (n : Name) {sub : Option Subset}
    (hsub : NamesFixed σ (optSubsetNames sub)) :
    (ctx'.pointsOf ρ m (σ n) sub).get? = (ctx.pointsOf ρ m n sub).get? := by
  unfold Ctx.pointsOf
  rw [h.desc]
  cases hd : ctx.desc? n with
  | none => rfl
  | some d =>
    simp only [Res.get?_ofOption]
    cases sub with
    | none => exact h.subsetPoints ρ m _ (hD n d hd)
    | some S => exact h.subsetPoints ρ m S hsub

theorem applyAssigns (cst : CSt) (as : List (Sym × Expr)) (has : ∀ a ∈ as, NamesFixed σ a.2.names) :
    Dace.applyAssigns ctx' cst as = Dace.applyAssigns ctx cst as := by
  unfold Dace.applyAssigns
  have : as.mapM (fun (x, e) => do let v ← ctx'.evalR cst.ρ cst.mem e; pure (x, v)) =
      as.mapM (fun (x, e) => do let v ← ctx.evalR cst.ρ cst.mem e; pure (x, v)) := by
    exact List.mapM_congr_res _ fun a ha => by
      obtain ⟨x, e⟩ := a
      simp only [h.evalR cst.ρ cst.mem e (has _ ha)]
  rw [this]

theorem applyAssign? (cst : CSt) (a : Option (Sym × Expr)) (ha : ∀ x ∈ a, NamesFixed σ x.2.names) :
    Dace.applyAssign? ctx' cst a = Dace.applyAssign? ctx cst a := by
  cases a with
  | none => rfl
  | some a => exact h.applyAssigns cst [a] (by simpa using ha)

end Ctx.Ren

/-! ## Memory accesses -/

namespace Ctx.Ren

variable {σ : Equiv.Perm Name} {ctx' ctx : Ctx} (h : Ctx.Ren σ ctx' ctx) (ρ : Valuation)
include h

theorem read (st : St) (n : Name) (idx : Idx) (oob : Bool) :
    (st.read ctx' ρ (σ n) idx oob).get? = (st.read ctx ρ n idx oob).get? := by
  simp only [St.read, Res.get?_bind, h.loc]
  congr 1; funext l; split <;> rfl

theorem readMany (st : St) (n : Name) (pts : List Idx) (oob : Bool) :
    (st.readMany ctx' ρ (σ n) pts oob).get? = (st.readMany ctx ρ n pts oob).get? := by
  unfold St.readMany
  refine Res.get?_foldlM (fun acc p => ?_) pts _
  simp only [Res.get?_bind, h.read]

theorem write (st : St) (n : Name) (idx : Idx) (oob : Bool) (v : Val) :
    (st.write ctx' ρ (σ n) idx oob v).get? = (st.write ctx ρ n idx oob v).get? := by
  simp only [St.write, Res.get?_bind, h.loc]

theorem wcr (st : St) (n : Name) (idx : Idx) (oob : Bool) (op : WcrOp) (v : Val) :
    (st.wcr ctx' ρ (σ n) idx oob op v).get? = (st.wcr ctx ρ n idx oob op v).get? := by
  simp only [St.wcr, Res.get?_bind, h.loc, h.I]

theorem store (st : St) (n : Name) (idx : Idx) (oob : Bool) (w : Option WcrOp) (v : Val) :
    (st.store ctx' ρ (σ n) idx oob w v).get? = (st.store ctx ρ n idx oob w v).get? := by
  cases w <;> simp only [St.store, h.write, h.wcr]

theorem queue (st : St) (n : Name) (idx : Idx) :
    (st.queue ctx' ρ (σ n) idx).get? = (st.queue ctx ρ n idx).get? := by
  simp only [St.queue, Res.get?_bind, h.loc]

theorem push (st : St) (n : Name) (idx : Idx) (vs : List Val) :
    (st.push ctx' ρ (σ n) idx vs).get? = (st.push ctx ρ n idx vs).get? := by
  simp only [St.push, Res.get?_bind, h.loc]

theorem popN (st : St) (n : Name) (idx : Idx) (k : Nat) :
    (st.popN ctx' ρ (σ n) idx k).get? = (st.popN ctx ρ n idx k).get? := by
  simp only [St.popN, Res.get?_bind, h.loc]

end Ctx.Ren

/-- Reading a location as an optional value only depends on its result. -/
theorem Res.matchLoc (m : Mem) : ∀ (r₂ r₁ : Res Loc), r₂.get? = r₁.get? →
    (match r₂ with
      | .ok l => some (m.read l)
      | .err _ => none) =
    (match r₁ with
      | .ok l => some (m.read l)
      | .err _ => none)
  | .ok _, .ok _, hr => by simp only [Res.get?, Option.some.injEq] at hr; subst hr; rfl
  | .ok _, .err _, hr => by simp only [Res.get?, reduceCtorEq] at hr
  | .err _, .ok _, hr => by simp only [Res.get?, reduceCtorEq] at hr
  | .err _, .err _, _ => rfl

/-! ## Leaf functions -/

namespace Ctx.Ren

variable {σ : Equiv.Perm Name} {ctx' ctx : Ctx} (h : Ctx.Ren σ ctx' ctx) (hD : ctx.DescFixed σ) (ρ : Valuation)
include h hD

theorem streamPoint (m : Mem) (d : Name) {sub : Option Subset} (hsub : NamesFixed σ (optSubsetNames sub)) :
    (Dace.streamPoint ctx' ρ m (σ d) sub).get? = (Dace.streamPoint ctx ρ m d sub).get? := by
  simp only [Dace.streamPoint, Res.get?_bind, Res.get?_ofOption, h.pointsOf hD ρ m d hsub]

theorem lowerCorner (m : Mem) (n : Name) {sub : Option Subset} (hsub : NamesFixed σ (optSubsetNames sub)) :
    (ctx'.lowerCorner ρ m (σ n) sub).get? = (ctx.lowerCorner ρ m n sub).get? := by
  simp only [Ctx.lowerCorner, Res.get?_bind, h.pointsOf hD ρ m n hsub]
  congr 1; funext _
  cases sub with
  | none =>
    simp only [h.desc]
    cases ctx.desc? n <;> rfl
  | some S =>
    simp only
    have hS : NamesFixed σ S.names := hsub
    cases S with
    | range r =>
      simp only
      refine Res.get?_mapM _ fun dim hdim => ?_
      rw [h.evalInt ρ m dim.lo fun x hx => hS x (List.mem_flatMap.2 ⟨dim, hdim, by
        simp only [Dim.names, List.mem_append]; exact .inl (.inl (.inl hx))⟩)]
    | union rs =>
      simp only
      cases hr : rs.head? with
      | none => rfl
      | some r =>
        simp only
        refine Res.get?_mapM _ fun dim hdim => ?_
        have hrm : r ∈ rs := List.mem_of_head? hr
        rw [h.evalInt ρ m dim.lo fun x hx => hS x (List.mem_flatMap.2 ⟨r, hrm, List.mem_flatMap.2 ⟨dim, hdim, by
          simp only [Dim.names, List.mem_append]; exact .inl (.inl (.inl hx))⟩⟩)]

theorem readScalarData (d : Name) {sub : Option Subset} (hsub : NamesFixed σ (optSubsetNames sub)) (oob : Bool)
    (st : St) : (Dace.readScalarData ctx' ρ (σ d) sub oob st).get? = (Dace.readScalarData ctx ρ d sub oob st).get? := by
  simp only [Dace.readScalarData, Res.get?_bind, h.pointsOf hD ρ st.mem d hsub]
  congr 1; funext pts
  split
  · rw [h.isStream]
    split
    · simp only [Res.get?_bind, h.popN]
    · simp only [Res.get?_bind, h.read]
  · rfl

theorem pushTo (st : St) (d : Name) {sub : Option Subset} (hsub : NamesFixed σ (optSubsetNames sub)) (vs : List Val) :
    (Dace.pushTo ctx' ρ st (σ d) sub vs).get? = (Dace.pushTo ctx ρ st d sub vs).get? := by
  simp only [Dace.pushTo, h.isStream]
  split
  · simp only [Res.get?_bind, h.streamPoint hD ρ st.mem d hsub, h.push]
  · rfl

theorem popFrom (st : St) (d : Name) {sub : Option Subset} (hsub : NamesFixed σ (optSubsetNames sub)) (k : Nat) :
    (Dace.popFrom ctx' ρ st (σ d) sub k).get? = (Dace.popFrom ctx ρ st d sub k).get? := by
  simp only [Dace.popFrom, h.isStream]
  split
  · simp only [Res.get?_bind, h.streamPoint hD ρ st.mem d hsub, h.popN]
  · rfl

theorem storeScalar (st : St) (d : Name) {sub : Option Subset} (hsub : NamesFixed σ (optSubsetNames sub))
    (w : Option WcrOp) (oob : Bool) (v : Val) :
    (Dace.storeScalar ctx' ρ st (σ d) sub w oob v).get? = (Dace.storeScalar ctx ρ st d sub w oob v).get? := by
  simp only [Dace.storeScalar, h.isStream]
  split
  · simp only [Res.get?_bind, h.streamPoint hD ρ st.mem d hsub, h.push]
  · simp only [Res.get?_bind, h.pointsOf hD ρ st.mem d hsub]
    congr 1; funext pts
    split
    · exact h.store ρ _ _ _ _ _ _
    · rfl
    · rfl

theorem storeArrayWrite (st : St) (d : Name) {sub : Option Subset} (hsub : NamesFixed σ (optSubsetNames sub))
    (w : Option WcrOp) (oob : Bool) (rel : Idx) (v : Val) :
    (Dace.storeArrayWrite ctx' ρ st (σ d) sub w oob rel v).get? =
      (Dace.storeArrayWrite ctx ρ st d sub w oob rel v).get? := by
  simp only [Dace.storeArrayWrite, h.isStream]
  split
  · rfl
  · simp only [Res.get?_bind, h.lowerCorner hD ρ st.mem d hsub]
    congr 1; funext lo
    split
    · simp only [Res.get?_bind, h.pointsOf hD ρ st.mem d hsub]
      congr 1; funext pts
      split
      · exact h.store ρ _ _ _ _ _ _
      · rfl
    · rfl

end Ctx.Ren

namespace Ctx.Ren

variable {σ : Equiv.Perm Name} {ctx' ctx : Ctx} (h : Ctx.Ren σ ctx' ctx) (hD : ctx.DescFixed σ) (ρ : Valuation)
include h hD

theorem copyData (st : St) (sd : Name) {ssub : Option Subset} (hs : NamesFixed σ (optSubsetNames ssub)) (dd : Name)
    {dsub : Option Subset} (hds : NamesFixed σ (optSubsetNames dsub)) (w : Option WcrOp) (oob : Bool) :
    (Dace.copyData ctx' ρ st (σ sd) ssub (σ dd) dsub w oob).get? =
      (Dace.copyData ctx ρ st sd ssub dd dsub w oob).get? := by
  unfold Dace.copyData
  rw [h.isStream, h.isStream]
  split
  · rfl
  · simp only [Res.get?_bind, h.pointsOf hD ρ st.mem dd hds, h.streamPoint hD ρ st.mem sd hs]
    congr 1; funext dpts; congr 1; funext sp
    simp only [h.popN]
    congr 1; funext p
    obtain ⟨vs, st'⟩ := p
    refine Res.get?_foldlM (fun st qv => ?_) _ _
    exact h.store ρ st dd qv.1 oob w qv.2
  · simp only [Res.get?_bind, h.pointsOf hD ρ st.mem sd hs, h.streamPoint hD ρ st.mem dd hds]
    congr 1; funext spts; congr 1; funext dp
    simp only [h.readMany]
    congr 1; funext p
    obtain ⟨vs, st'⟩ := p
    exact h.push ρ _ _ _ _
  · simp only [Res.get?_bind, h.pointsOf hD ρ st.mem sd hs, h.pointsOf hD ρ st.mem dd hds]
    congr 1; funext spts; congr 1; funext dpts
    refine Res.get?_foldlM (fun st pq => ?_) _ _
    simp only [Res.get?_bind, h.read]
    congr 1; funext x
    obtain ⟨v, st'⟩ := x
    exact h.store ρ _ _ _ _ _ _

theorem bindInData (d : Name) {sub : Option Subset} (hsub : NamesFixed σ (optSubsetNames sub)) (oob : Bool) (st : St) :
    (Dace.bindInData ctx' ρ (σ d) sub oob st).get? = (Dace.bindInData ctx ρ d sub oob st).get? := by
  unfold Dace.bindInData
  rw [h.isStream]
  split
  · simp only [Res.get?_bind, h.streamPoint hD ρ st.mem d hsub, h.queue]
  · simp only [Res.get?_bind, h.pointsOf hD ρ st.mem d hsub]
    congr 1; funext pts
    split
    · simp only [Res.get?_bind, h.read]
    · simp only [Res.get?_bind, h.lowerCorner hD ρ st.mem d hsub, h.readMany]
      congr 1; funext lo; congr 1; funext p
      obtain ⟨vs, st'⟩ := p
      simp only [Res.get?_pure, Option.some.injEq, Prod.mk.injEq, and_true]
      congr 1; funext rel
      split
      · split
        · exact Res.matchLoc st'.mem _ _ (h.loc ρ st'.mem d _ oob)
        · rfl
      · rfl

omit hD in
theorem reinterprets (d base : Name) : ctx'.reinterprets (σ d) (σ base) = ctx.reinterprets d base := by
  simp only [Ctx.reinterprets, h.desc]

theorem aliasTo (st : St) (d base : Name) {sub : Option Subset} (hsub : NamesFixed σ (optSubsetNames sub)) :
    (Dace.aliasTo ctx' ρ st (σ d) (σ base) sub).get? = (Dace.aliasTo ctx ρ st d base sub).get? := by
  simp only [Dace.aliasTo, h.reinterprets]
  split
  · rfl
  · simp only [Res.get?_bind, h.lowerCorner hD ρ st.mem base hsub, h.addr, h.cname]

theorem reduceData (st : St) (props : ReduceProps) (hid : ∀ e ∈ props.identity, NamesFixed σ e.names)
    (sd : Name) {ssub : Option Subset} (hs : NamesFixed σ (optSubsetNames ssub)) (dd : Name)
    {dsub : Option Subset} (hds : NamesFixed σ (optSubsetNames dsub)) (w : Option WcrOp) (soob doob : Bool) :
    (Dace.reduceData ctx' ρ st props (σ sd) ssub (σ dd) dsub w soob doob).get? =
      (Dace.reduceData ctx ρ st props sd ssub dd dsub w soob doob).get? := by
  simp only [Dace.reduceData, h.isStream]
  split
  · rfl
  · simp only [Res.get?_bind, h.pointsOf hD ρ st.mem sd hs, h.pointsOf hD ρ st.mem dd hds]
    congr 1; funext spts; congr 1; funext dpts
    split
    · rfl
    · simp only [Res.get?_bind, h.readMany]
      congr 1; funext p
      refine Res.get?_foldlM (fun st gq => ?_) _ _
      cases hi : props.identity with
      | some ide =>
        simp only [Res.get?_bind, h.evalR ρ st.mem ide (hid ide hi), h.I]
        congr 1; funext i
        exact h.store ρ _ _ _ _ _ _
      | none =>
        simp only
        split
        · rfl
        · simp only [h.I]; exact h.wcr ρ _ _ _ _ _ _

end Ctx.Ren

/-! ## Tree leaves -/

/-- The names read by the expressions of a loop's metadata. -/
def LoopMeta.exprNames (l : LoopMeta) : List Name :=
  l.cond.expr.names ++ ((l.init.map (·.rhs.names)).getD []) ++ ((l.update.map (·.rhs.names)).getD [])

theorem LoopMeta.fixed_cond {σ : Equiv.Perm Name} {l : LoopMeta} (h : NamesFixed σ l.exprNames) :
    NamesFixed σ l.cond.expr.names := fun x hx => h x (by simp [LoopMeta.exprNames, hx])

theorem LoopMeta.fixed_init {σ : Equiv.Perm Name} {l : LoopMeta} (h : NamesFixed σ l.exprNames) :
    ∀ a ∈ l.initAssign, NamesFixed σ a.2.names := fun a ha x hx => h x (by
  cases hi : l.init with
  | none => simp [LoopMeta.initAssign, hi] at ha
  | some i =>
    simp only [LoopMeta.initAssign, hi, Option.map_some, Option.mem_def, Option.some.injEq] at ha
    subst ha
    simp [LoopMeta.exprNames, hi, hx])

theorem LoopMeta.fixed_update {σ : Equiv.Perm Name} {l : LoopMeta} (h : NamesFixed σ l.exprNames) :
    ∀ a ∈ l.updateAssign, NamesFixed σ a.2.names := fun a ha x hx => h x (by
  cases hu : l.update with
  | none => simp [LoopMeta.updateAssign, hu] at ha
  | some u =>
    simp only [LoopMeta.updateAssign, hu, Option.map_some, Option.mem_def, Option.some.injEq] at ha
    subst ha
    simp [LoopMeta.exprNames, hu, hx])

/-- The names read by the expressions of a node itself (not its children). -/
def SNode.exprNames : SNode → List Name
  | .assign _ v => v.expr.names
  | .loopScope l _ | .forScope l _ | .whileScope l _ | .doWhileScope l _ => l.exprNames
  | .ifScope c _ | .stateIfScope c _ | .elifScope c _ => c.expr.names
  | .mapScope mp _ => mp.range.names
  | .consumeScope c _ => c.numPes.names ++ (c.condition.map Expr.names).getD []
  | .taskletNode _ ins outs => (ins.memlets ++ outs.memlets).flatMap Memlet.exprNames
  | .libraryCall l _ ins outs =>
    (ins.memlets ++ outs.memlets).flatMap Memlet.exprNames ++ ((l.reduce.bind (·.identity)).map Expr.names).getD []
  | .copyNode _ m | .dynScopeCopy _ m | .viewNode _ _ m _ _ | .refSetNode _ m _ _ => m.exprNames
  | _ => []

theorem Memlet.fixed_dataSubset {σ : Equiv.Perm Name} {m : Memlet} (h : NamesFixed σ m.exprNames) :
    NamesFixed σ (optSubsetNames m.dataSubset) := fun x hx => h x (by
  simp only [Memlet.dataSubset] at hx
  split at hx <;> simp [Memlet.exprNames, hx])

theorem Memlet.fixed_otherSubset {σ : Equiv.Perm Name} {m : Memlet} (h : NamesFixed σ m.exprNames) :
    NamesFixed σ (optSubsetNames m.otherSubset) := fun x hx => h x (by
  simp only [Memlet.otherSubset] at hx
  split at hx <;> simp [Memlet.exprNames, hx])

/-- Tree contexts agreeing through `σ`, outside consume scopes. -/
structure TCtx.Ren (σ : Equiv.Perm Name) (tc' tc : TCtx) : Prop where
  ctx : Ctx.Ren σ tc'.ctx tc.ctx
  desc : tc.ctx.DescFixed σ
  cons' : tc'.consumed = none
  cons : tc.consumed = none

namespace TCtx.Ren

variable {σ : Equiv.Perm Name} {tc' tc : TCtx} (h : TCtx.Ren σ tc' tc) (ρ : Valuation)
include h

theorem inData (m : Memlet) : tc'.inData (m.ren σ) = (σ (tc.inData m).1, (tc.inData m).2) := by
  simp [TCtx.inData, TCtx.isChunk, h.cons, h.cons']

theorem bindMemletIn (m : Memlet) (hm : NamesFixed σ m.exprNames) (st : St) :
    (Dace.bindMemletIn tc' ρ (m.ren σ) st).get? = (Dace.bindMemletIn tc ρ m st).get? := by
  simp only [Dace.bindMemletIn, h.inData]
  simp only [TCtx.inData, TCtx.isChunk, h.cons]
  exact h.ctx.bindInData h.desc ρ m.data (Memlet.fixed_dataSubset hm) m.allowOob st

theorem bindInputsT (ins : MemletMap) (hins : ∀ m ∈ ins.memlets, NamesFixed σ m.exprNames) (st : St) :
    (Dace.bindInputsT tc' ρ (ins.ren σ) st).get? = (Dace.bindInputsT tc ρ ins st).get? := by
  unfold Dace.bindInputsT
  rw [MemletMap.ren_pairs, List.foldlM_map]
  simp only [Res.get?_bind]
  congr 1
  refine Res.get?_foldlM_mem _ _ fun a p hp => ?_
  obtain ⟨acc, st₀⟩ := a
  simp only [Res.get?_bind, h.bindMemletIn ρ p.2 (hins p.2 (MemletMap.snd_mem_memlets hp))]

theorem applyOutT (ins outs : MemletMap) (hins : ∀ m ∈ ins.memlets, NamesFixed σ m.exprNames)
    (houts : ∀ m ∈ outs.memlets, NamesFixed σ m.exprNames) (st : St) (out : TaskletOut) :
    (Dace.applyOutT tc' ρ (ins.ren σ) (outs.ren σ) st out).get? = (Dace.applyOutT tc ρ ins outs st out).get? := by
  have hin : ∀ m, tc.inData m = (m.data, m.dataSubset) := fun m => by simp [TCtx.inData, TCtx.isChunk, h.cons]
  have hin' : ∀ m, tc'.inData m = (m.data, m.dataSubset) := fun m => by simp [TCtx.inData, TCtx.isChunk, h.cons']
  have hon : ∀ c, ∀ m ∈ outs.on c, NamesFixed σ m.exprNames := fun c m hm => houts m (MemletMap.mem_memlets_of_on hm)
  unfold Dace.applyOutT
  simp only [Res.get?_bind, MemletMap.ren_on, List.foldlM_map, hin, hin']
  congr 1
  · refine Res.get?_foldlM (fun st ck => ?_) _ _
    have hm : ∀ m ∈ ins.on ck.1, NamesFixed σ m.exprNames := fun m hm => hins m (MemletMap.mem_memlets_of_on hm)
    revert hm
    cases ins.on ck.1 with
    | nil => intro; rfl
    | cons m₀ ms =>
      intro hm
      simp only [List.map_cons]
      rw [← List.map_cons, List.foldlM_map]
      refine Res.get?_foldlM_mem _ _ fun st m hmm => ?_
      exact h.ctx.popFrom h.desc ρ st m.data (Memlet.fixed_dataSubset (hm m hmm)) ck.2
  · funext st₁
    congr 1
    · refine Res.get?_foldlM (fun st cv => ?_) _ _
      refine Res.get?_foldlM_mem _ _ fun st m hmm => ?_
      exact h.ctx.storeScalar h.desc ρ st m.data (Memlet.fixed_dataSubset (hon _ m hmm)) _ _ _
    · funext st₂
      congr 1
      · refine Res.get?_foldlM (fun st cv => ?_) _ _
        refine Res.get?_foldlM_mem _ _ fun st m hmm => ?_
        exact h.ctx.storeArrayWrite h.desc ρ st m.data (Memlet.fixed_dataSubset (hon _ m hmm)) _ _ _ _
      · funext st₃
        refine Res.get?_foldlM (fun st cv => ?_) _ _
        refine Res.get?_foldlM_mem _ _ fun st m hmm => ?_
        exact h.ctx.pushTo h.desc ρ st m.data (Memlet.fixed_dataSubset (hon _ m hmm)) _

theorem execCodeT (label : String) (ins outs : MemletMap) (hins : ∀ m ∈ ins.memlets, NamesFixed σ m.exprNames)
    (houts : ∀ m ∈ outs.memlets, NamesFixed σ m.exprNames) (st : St) (run : (Conn → Option ConnIn) → Option TaskletOut)
    (err : ErrReason) :
    (Dace.execCodeT tc' ρ label (ins.ren σ) (outs.ren σ) st run err).get? =
      (Dace.execCodeT tc ρ label ins outs st run err).get? := by
  unfold Dace.execCodeT
  rw [MemletMap.ren_hasDynamicWcr]
  split
  · rfl
  · simp only [Res.get?_bind, h.bindInputsT ρ ins hins]
    congr 1; funext p
    obtain ⟨b, st'⟩ := p
    simp only
    split
    · rfl
    · exact h.applyOutT ρ ins outs hins houts st' _

theorem execLibraryT (l : LibNode) (ins outs : MemletMap) (hins : ∀ m ∈ ins.memlets, NamesFixed σ m.exprNames)
    (houts : ∀ m ∈ outs.memlets, NamesFixed σ m.exprNames)
    (hid : ∀ e ∈ l.reduce.bind (·.identity), NamesFixed σ e.names) (st : St) :
    (Dace.execLibraryT tc' ρ l (ins.ren σ) (outs.ren σ) st).get? = (Dace.execLibraryT tc ρ l ins outs st).get? := by
  unfold Dace.execLibraryT
  cases hr : l.reduce with
  | none =>
    simp only
    rw [h.ctx.I]
    exact h.execCodeT ρ _ ins outs hins houts st _ _
  | some props =>
    simp only [Dace.execReduceT, MemletMap.ren_memlets]
    cases hi : ins.memlets with
    | nil => cases outs.memlets <;> rfl
    | cons im irest =>
      cases irest with
      | cons _ _ => cases outs.memlets <;> rfl
      | nil =>
        cases ho : outs.memlets with
        | nil => rfl
        | cons om orest =>
          cases orest with
          | cons _ _ => rfl
          | nil =>
            simp only [List.map_cons, List.map_nil, h.inData]
            have him := hins im (by rw [hi]; exact List.mem_cons_self)
            have hom := houts om (by rw [ho]; exact List.mem_cons_self)
            have hin : tc.inData im = (im.data, im.dataSubset) := by simp [TCtx.inData, TCtx.isChunk, h.cons]
            rw [hin]
            exact h.ctx.reduceData h.desc ρ st props (fun e he => hid e (by simp [hr, he])) im.data
              (Memlet.fixed_dataSubset him) om.data (Memlet.fixed_dataSubset hom) _ _ _

theorem execCopyT (target : Name) (m : Memlet) (hm : NamesFixed σ m.exprNames) (st : St) :
    (Dace.execCopyT tc' ρ (σ target) (m.ren σ) st).get? = (Dace.execCopyT tc ρ target m st).get? := by
  have hin : tc.inData m = (m.data, m.dataSubset) := by simp [TCtx.inData, TCtx.isChunk, h.cons]
  simp only [Dace.execCopyT, h.inData, hin, Memlet.ren_otherSubset, Memlet.ren_wcr, Memlet.ren_allowOob]
  exact h.ctx.copyData h.desc ρ st m.data (Memlet.fixed_dataSubset hm) target (Memlet.fixed_otherSubset hm) _ _

theorem execViewT (target source : Name) (m : Memlet) (hm : NamesFixed σ m.exprNames) (st : St) :
    (Dace.execViewT tc' ρ (σ target) (σ source) (m.ren σ) st).get? = (Dace.execViewT tc ρ target source m st).get? := by
  simp only [Dace.execViewT, Memlet.ren_data, EmbeddingLike.apply_eq_iff_eq, Memlet.ren_dataSubset]
  split
  · exact h.ctx.aliasTo h.desc ρ st target source (Memlet.fixed_dataSubset hm)
  · rfl

theorem execRefSetT (target : Name) (m : Memlet) (hm : NamesFixed σ m.exprNames) (src : RefSetSource) (st : St) :
    (Dace.execRefSetT tc' ρ (σ target) (m.ren σ) src st).get? = (Dace.execRefSetT tc ρ target m src st).get? := by
  cases src with
  | data _ => exact h.ctx.aliasTo h.desc ρ st target m.data (Memlet.fixed_dataSubset hm)
  | codeNode _ _ => rfl

theorem execDynScopeCopy (p : Pending) (target : Sym) (m : Memlet) (hm : NamesFixed σ m.exprNames) (st : St) :
    (Dace.execDynScopeCopy tc' ρ (p.ren σ) target (m.ren σ) st).get? =
      (Dace.execDynScopeCopy tc ρ p target m st).get?.map fun q => (q.1.ren σ, q.2) := by
  have hin : tc.inData m = (m.data, m.dataSubset) := by simp [TCtx.inData, TCtx.isChunk, h.cons]
  unfold Dace.execDynScopeCopy
  split
  · simp only [Memlet.ren_data, h.ctx.isStream]
    split
    · simp only [Res.get?_bind, h.ctx.streamPoint h.desc ρ st.mem m.data (Memlet.fixed_dataSubset hm),
        Memlet.ren_dataSubset]
      cases (Dace.streamPoint tc.ctx ρ st.mem m.data m.dataSubset).get? <;> rfl
    · rfl
  · simp only [Res.get?_bind, h.inData, hin, Memlet.ren_allowOob,
      h.ctx.readScalarData h.desc ρ m.data (Memlet.fixed_dataSubset hm)]
    cases (Dace.readScalarData tc.ctx ρ m.data m.dataSubset m.allowOob st).get? with
    | none => rfl
    | some q =>
      obtain ⟨v?, st'⟩ := q
      cases v? <;> rfl

theorem mapSetupT (sc : SibCtx) (mp : MapNode) (hmp : NamesFixed σ mp.range.names) (cst : CSt) :
    Dace.mapSetupT tc' (sc.ren σ) mp cst = Dace.mapSetupT tc sc mp cst := by
  simp only [Dace.mapSetupT, SibCtx.ren, Pending.ren, Option.isSome_map]
  split
  · rfl
  · simp only [h.ctx.rangePoints _ _ _ hmp]

theorem scopeFinishT (pre : CSt) (logs : List Log) :
    Dace.scopeFinishT tc' pre logs = Dace.scopeFinishT tc pre logs := by
  simp only [Dace.scopeFinishT, Dace.scopeFinish, h.ctx.I]

end TCtx.Ren

theorem fixed_memlets {σ : Equiv.Perm Name} {ins outs : MemletMap}
    (h : NamesFixed σ ((ins.memlets ++ outs.memlets).flatMap Memlet.exprNames)) :
    (∀ m ∈ ins.memlets, NamesFixed σ m.exprNames) ∧ (∀ m ∈ outs.memlets, NamesFixed σ m.exprNames) :=
  ⟨fun m hm x hx => h x (List.mem_flatMap.2 ⟨m, List.mem_append_left _ hm, hx⟩),
   fun m hm x hx => h x (List.mem_flatMap.2 ⟨m, List.mem_append_right _ hm, hx⟩)⟩

@[simp] theorem SibCtx.ren_empty (σ : Equiv.Perm Name) : ({} : SibCtx).ren σ = {} := rfl

theorem leafOut_get? (σ : Equiv.Perm Name) {r₂ r₁ : Res CSt} (hr : r₂.get? = r₁.get?) :
    (leafOut r₂).get? = (leafOut r₁).get?.map (NodeOut.ren σ) := by
  simp only [leafOut, Res.get?_map, hr, Option.map_map]
  rfl

namespace TCtx.Ren

variable {σ : Equiv.Perm Name} {tc' tc : TCtx} (h : TCtx.Ren σ tc' tc)
include h

theorem stepLeaf (sc : SibCtx) (n : SNode) (hn : NamesFixed σ n.exprNames) (cst : CSt) :
    (Dace.stepLeaf tc' (sc.ren σ) (n.ren σ) cst).map Res.get? =
      (Dace.stepLeaf tc sc n cst).map fun r => r.get?.map (NodeOut.ren σ) := by
  cases n with
  | assign x v =>
    simp only [SNode.ren, Dace.stepLeaf, Option.map_some]
    rw [h.ctx.applyAssigns cst [(x, v.expr)] (by simpa [SNode.exprNames] using hn)]
    exact congrArg some (leafOut_get? σ rfl)
  | taskletNode t ins outs =>
    obtain ⟨hi, ho⟩ := fixed_memlets (by simpa [SNode.exprNames] using hn)
    simp only [SNode.ren, Dace.stepLeaf, Option.map_some, h.ctx.I]
    refine congrArg some (leafOut_get? σ ?_)
    simp only [CSt.lift, Res.get?_map, h.execCodeT cst.ρ _ ins outs hi ho]
  | libraryCall l props ins outs =>
    have hn' : NamesFixed σ ((ins.memlets ++ outs.memlets).flatMap Memlet.exprNames) ∧
        ∀ e ∈ l.reduce.bind (·.identity), NamesFixed σ e.names := by
      simp only [SNode.exprNames] at hn
      refine ⟨fun x hx => hn x (List.mem_append_left _ hx), fun e he x hx => hn x (List.mem_append_right _ ?_)⟩
      cases hr : l.reduce with
      | none => simp [hr] at he
      | some r =>
        simp only [hr, Option.bind_some, Option.mem_def] at he ⊢
        simp [he, hx]
    obtain ⟨hi, ho⟩ := fixed_memlets hn'.1
    simp only [SNode.ren, Dace.stepLeaf, Option.map_some]
    refine congrArg some (leafOut_get? σ ?_)
    simp only [CSt.lift, Res.get?_map, h.execLibraryT cst.ρ l ins outs hi ho hn'.2]
  | copyNode target m =>
    simp only [SNode.ren, Dace.stepLeaf, Option.map_some]
    refine congrArg some (leafOut_get? σ ?_)
    simp only [CSt.lift, Res.get?_map, h.execCopyT cst.ρ target m (by simpa [SNode.exprNames] using hn)]
  | dynScopeCopy target m =>
    simp only [SNode.ren, Dace.stepLeaf, Option.map_some, Option.some.injEq, Res.get?_map, SibCtx.ren,
      h.execDynScopeCopy cst.ρ sc.pending target m (by simpa [SNode.exprNames] using hn), Option.map_map]
    rfl
  | viewNode target source m sd vd =>
    simp only [SNode.ren, Dace.stepLeaf, Option.map_some]
    refine congrArg some (leafOut_get? σ ?_)
    simp only [CSt.lift, Res.get?_map, h.execViewT cst.ρ target source m (by simpa [SNode.exprNames] using hn)]
  | refSetNode target m src d =>
    simp only [SNode.ren, Dace.stepLeaf, Option.map_some]
    refine congrArg some (leafOut_get? σ ?_)
    simp only [CSt.lift, Res.get?_map, h.execRefSetT cst.ρ target m (by simpa [SNode.exprNames] using hn)]
  | goto t => cases t <;> rfl
  | _ => rfl

end TCtx.Ren

/-! ## The fragment -/

/-- Are all names of the list fixed by `σ`? -/
def fixedB (σ : Equiv.Perm Name) (xs : List Name) : Bool := xs.all fun x => σ x == x

theorem fixedB_iff {σ : Equiv.Perm Name} {xs : List Name} : fixedB σ xs = true ↔ NamesFixed σ xs := by
  simp [fixedB, NamesFixed]

mutual

/-- The nodes the renaming theorem covers: no consume scopes, expressions
reading only names fixed by `σ`, and map scopes whose private transients
agree between the renamed and the original context. -/
def SNode.renOk (σ : Equiv.Perm Name) (tc' tc : TCtx) : SNode → Bool
  | .namedRegion _ cs => SNodes.renOk σ tc' tc cs
  | .gblock cs => SNodes.renOk σ tc' tc cs
  | .loopScope l cs => fixedB σ l.exprNames && SNodes.renOk σ tc' tc cs
  | .forScope l cs => fixedB σ l.exprNames && SNodes.renOk σ tc' tc cs
  | .whileScope l cs => fixedB σ l.exprNames && SNodes.renOk σ tc' tc cs
  | .doWhileScope l cs => fixedB σ l.exprNames && SNodes.renOk σ tc' tc cs
  | .ifScope c cs => fixedB σ c.expr.names && SNodes.renOk σ tc' tc cs
  | .stateIfScope c cs => fixedB σ c.expr.names && SNodes.renOk σ tc' tc cs
  | .elifScope c cs => fixedB σ c.expr.names && SNodes.renOk σ tc' tc cs
  | .elseScope cs => SNodes.renOk σ tc' tc cs
  | .mapScope mp cs =>
    fixedB σ mp.range.names && (tc'.privateTransients (SNodes.ren σ cs) == tc.privateTransients cs) &&
      SNodes.renOk σ tc' tc cs
  | .consumeScope _ _ => false
  | .stateLabel _ => true
  | .goto _ => true
  | .breakNode => true
  | .continueNode => true
  | .assign _ v => fixedB σ v.expr.names
  | n@(.taskletNode ..) | n@(.libraryCall ..) | n@(.copyNode ..) | n@(.dynScopeCopy ..) | n@(.viewNode ..)
  | n@(.refSetNode ..) => fixedB σ n.exprNames

def SNodes.renOk (σ : Equiv.Perm Name) (tc' tc : TCtx) : List SNode → Bool
  | [] => true
  | c :: cs => SNode.renOk σ tc' tc c && SNodes.renOk σ tc' tc cs

end

section Fragment

variable {σ : Equiv.Perm Name} {tc' tc : TCtx}

theorem SNode.renOk_fixed {n : SNode} (h : SNode.renOk σ tc' tc n = true) : NamesFixed σ n.exprNames := by
  cases n <;> simp_all [SNode.renOk, SNode.exprNames, fixedB_iff, NamesFixed]

theorem SNodes.renOk_cons {c : SNode} {cs : List SNode} :
    SNodes.renOk σ tc' tc (c :: cs) = true ↔ SNode.renOk σ tc' tc c = true ∧ SNodes.renOk σ tc' tc cs = true := by
  simp [SNodes.renOk]

theorem SNodes.renOk_afterLabel {l : String} :
    ∀ {xs ys : List SNode}, afterLabel l xs = some ys → SNodes.renOk σ tc' tc xs = true → SNodes.renOk σ tc' tc ys = true
  | [], _, h, _ => by cases h
  | x :: xs, ys, h, hok => by
    rw [SNodes.renOk_cons] at hok
    cases x with
    | stateLabel name =>
      simp only [afterLabel] at h
      split at h
      · cases h; exact hok.2
      · exact SNodes.renOk_afterLabel h hok.2
    | _ => simp only [afterLabel] at h; exact SNodes.renOk_afterLabel h hok.2

theorem SNodes.renOk_resumeAt {gb : Bool} {full rest suffix : List SNode} {l : String}
    (h : resumeAt gb full rest l = some suffix) (hf : SNodes.renOk σ tc' tc full = true)
    (hr : SNodes.renOk σ tc' tc rest = true) : SNodes.renOk σ tc' tc suffix = true := by
  unfold resumeAt at h
  split at h
  · cases h; exact SNodes.renOk_afterLabel (by assumption) hr
  · split at h
    · exact SNodes.renOk_afterLabel h hf
    · cases h

end Fragment

/-! ## Renaming and tree structure -/

section Structure

variable (σ : Equiv.Perm Name)

theorem SNode.ren_consumesPending (n : SNode) : (n.ren σ).consumesPending = n.consumesPending := by
  cases n <;> rfl

theorem SNode.ren_isDynScopeCopy (n : SNode) : (n.ren σ).isDynScopeCopy = n.isDynScopeCopy := by
  cases n <;> rfl

theorem SNode.ren_loop? (n : SNode) : (n.ren σ).loop? = n.loop?.map fun p => (p.1, SNodes.ren σ p.2) := by
  cases n <;> rfl

@[simp] theorem SibCtx.ren_pending_isEmpty (sc : SibCtx) : (sc.ren σ).pending.isEmpty = sc.pending.isEmpty := by
  simp [SibCtx.ren, Pending.ren, Pending.isEmpty]

@[simp] theorem SibCtx.ren_chain (sc : SibCtx) : (sc.ren σ).chain = sc.chain := rfl

theorem afterLabel_ren (l : String) :
    ∀ xs : List SNode, afterLabel l (SNodes.ren σ xs) = (afterLabel l xs).map (SNodes.ren σ)
  | [] => rfl
  | x :: xs => by
    cases x <;> simp only [SNodes.ren, SNode.ren, afterLabel, afterLabel_ren l xs] <;> split <;> rfl

theorem resumeAt_ren (gb : Bool) (full rest : List SNode) (l : String) :
    resumeAt gb (SNodes.ren σ full) (SNodes.ren σ rest) l = (resumeAt gb full rest l).map (SNodes.ren σ) := by
  unfold resumeAt
  rw [afterLabel_ren, afterLabel_ren]
  cases afterLabel l rest with
  | some _ => rfl
  | none => cases gb <;> rfl

theorem scopeOut_get? {r₂ r₁ : Res (TCtl × CSt)} (h : r₂.get? = r₁.get?) :
    (scopeOut r₂).get? = (scopeOut r₁).get?.map (NodeOut.ren σ) := by
  simp only [scopeOut, Res.get?_map, h, Option.map_map]; rfl

theorem chainOut_get? {r₂ r₁ : Res (TCtl × CSt)} (h : r₂.get? = r₁.get?) :
    (chainOut r₂).get? = (chainOut r₁).get?.map (NodeOut.ren σ) := by
  simp only [chainOut, Res.get?_map, h, Option.map_map]; rfl

theorem dataflowOut_get? (r : Res CSt) : (dataflowOut r).get? = (dataflowOut r).get?.map (NodeOut.ren σ) := by
  cases r <;> rfl

theorem iterLog_get? (priv : List Name) {r₂ r₁ : Res (TCtl × CSt)} (h : r₂.get? = r₁.get?) :
    (iterLog priv r₂).get? = (iterLog priv r₁).get? := by
  cases r₂ <;> cases r₁ <;> simp only [Res.get?, Option.some.injEq, reduceCtorEq] at h
  · subst h; rfl
  · rfl

end Structure

/-! ## The renaming theorem -/

theorem Res.exists_err_of_get? {α : Type} {r : Res α} (hr : r.get? = none) : ∃ e, r = .err e :=
  Res.get?_eq_none.1 hr

namespace STree

section Rel

variable {σ : Equiv.Perm Name} {tc' tc : TCtx} (h : TCtx.Ren σ tc' tc)
include h

mutual

/-- **Renaming a tree run**: a run of a node is matched by a run of the
renamed node in a context agreeing through `σ`, with the same result up to
the error and the renamed sibling context. -/
theorem NodeExec.ren {sc : SibCtx} {n : SNode} {cst : CSt} {r : Res NodeOut} :
    STree.NodeExec tc sc n cst r → SNode.renOk σ tc' tc n = true →
      ∃ r', STree.NodeExec tc' (sc.ren σ) (n.ren σ) cst r' ∧ r'.get? = r.get?.map (NodeOut.ren σ)
  | .pendingErr hn hp, _ =>
    ⟨_, .pendingErr (by rw [SNode.ren_consumesPending]; exact hn) (by simpa using hp), rfl⟩
  | .leaf hp hs, hok => by
    have hl := h.stepLeaf sc n (SNode.renOk_fixed hok) cst
    rw [hs, Option.map_some] at hl
    obtain ⟨r', hr', heq⟩ := Option.map_eq_some_iff.1 hl
    exact ⟨r', .leaf (by simpa [SNode.ren_isDynScopeCopy] using hp) hr', heq⟩
  | .namedRegion hp hl, hok => by
    obtain ⟨r', hl', hr⟩ := ListExec.ren hl hok hok
    exact ⟨_, .namedRegion (by simpa using hp) hl', scopeOut_get? σ hr⟩
  | .gblock hp hl, hok => by
    obtain ⟨r', hl', hr⟩ := ListExec.ren hl hok hok
    exact ⟨_, .gblock (by simpa using hp) hl', scopeOut_get? σ hr⟩
  | .ifCondErr hp hc, hok => by
    simp only [SNode.renOk, Bool.and_eq_true, fixedB_iff] at hok
    exact ⟨_, .ifCondErr (by simpa using hp) (by rw [h.ctx.evalBoolR _ _ _ hok.1]; exact hc), rfl⟩
  | .ifSkip hp hc, hok => by
    simp only [SNode.renOk, Bool.and_eq_true, fixedB_iff] at hok
    exact ⟨_, .ifSkip (by simpa using hp) (by rw [h.ctx.evalBoolR _ _ _ hok.1]; exact hc), rfl⟩
  | .ifTaken hp hc hl, hok => by
    simp only [SNode.renOk, Bool.and_eq_true, fixedB_iff] at hok
    obtain ⟨r', hl', hr⟩ := ListExec.ren hl hok.2 hok.2
    exact ⟨_, .ifTaken (by simpa using hp) (by rw [h.ctx.evalBoolR _ _ _ hok.1]; exact hc) hl', chainOut_get? σ hr⟩
  | .stateIfCondErr hp hc, hok => by
    simp only [SNode.renOk, Bool.and_eq_true, fixedB_iff] at hok
    exact ⟨_, .stateIfCondErr (by simpa using hp) (by rw [h.ctx.evalBoolR _ _ _ hok.1]; exact hc), rfl⟩
  | .stateIfSkip hp hc, hok => by
    simp only [SNode.renOk, Bool.and_eq_true, fixedB_iff] at hok
    exact ⟨_, .stateIfSkip (by simpa using hp) (by rw [h.ctx.evalBoolR _ _ _ hok.1]; exact hc), rfl⟩
  | .stateIfTaken hp hc hl, hok => by
    simp only [SNode.renOk, Bool.and_eq_true, fixedB_iff] at hok
    obtain ⟨r', hl', hr⟩ := ListExec.ren hl hok.2 hok.2
    exact ⟨_, .stateIfTaken (by simpa using hp) (by rw [h.ctx.evalBoolR _ _ _ hok.1]; exact hc) hl',
      scopeOut_get? σ hr⟩
  | .elifNoChain hp hc, _ => ⟨_, .elifNoChain (by simpa using hp) hc, rfl⟩
  | .elifAfterTaken hp hc, _ => ⟨_, .elifAfterTaken (by simpa using hp) hc, rfl⟩
  | .elifCondErr hp hc he, hok => by
    simp only [SNode.renOk, Bool.and_eq_true, fixedB_iff] at hok
    exact ⟨_, .elifCondErr (by simpa using hp) hc (by rw [h.ctx.evalBoolR _ _ _ hok.1]; exact he), rfl⟩
  | .elifSkip hp hc he, hok => by
    simp only [SNode.renOk, Bool.and_eq_true, fixedB_iff] at hok
    exact ⟨_, .elifSkip (by simpa using hp) hc (by rw [h.ctx.evalBoolR _ _ _ hok.1]; exact he), rfl⟩
  | .elifTaken hp hc he hl, hok => by
    simp only [SNode.renOk, Bool.and_eq_true, fixedB_iff] at hok
    obtain ⟨r', hl', hr⟩ := ListExec.ren hl hok.2 hok.2
    exact ⟨_, .elifTaken (by simpa using hp) hc (by rw [h.ctx.evalBoolR _ _ _ hok.1]; exact he) hl',
      chainOut_get? σ hr⟩
  | .elseNoChain hp hc, _ => ⟨_, .elseNoChain (by simpa using hp) hc, rfl⟩
  | .elseAfterTaken hp hc, _ => ⟨_, .elseAfterTaken (by simpa using hp) hc, rfl⟩
  | .elseTaken hp hc hl, hok => by
    obtain ⟨r', hl', hr⟩ := ListExec.ren hl hok hok
    exact ⟨_, .elseTaken (by simpa using hp) hc hl', scopeOut_get? σ hr⟩
  | @STree.NodeExec.loop _ _ n l cs _ _ hp hl hL, hok => by
    have hok' : fixedB σ l.exprNames = true ∧ SNodes.renOk σ tc' tc cs = true := by
      cases n <;> simp only [SNode.loop?, reduceCtorEq, Option.some.injEq, Prod.mk.injEq] at hl <;>
        (obtain ⟨rfl, rfl⟩ := hl; simpa [SNode.renOk] using hok)
    obtain ⟨r', hL', hr⟩ := LoopExec.ren hL (fixedB_iff.1 hok'.1) hok'.2
    exact ⟨_, .loop (by simpa using hp) (by rw [SNode.ren_loop?, hl]; rfl) hL', scopeOut_get? σ hr⟩
  | .mapSetupErr hs, hok => by
    simp only [SNode.renOk, Bool.and_eq_true, fixedB_iff, beq_iff_eq] at hok
    exact ⟨_, .mapSetupErr (by rw [h.mapSetupT _ _ hok.1.1]; exact hs), rfl⟩
  | .mapIterErr hs hi, hok => by
    simp only [SNode.renOk, Bool.and_eq_true, fixedB_iff, beq_iff_eq] at hok
    obtain ⟨r', hi', hr⟩ := IterExec.ren hi hok.2
    obtain ⟨e', rfl⟩ := Res.exists_err_of_get? (by rw [hr]; rfl)
    exact ⟨_, .mapIterErr (by rw [h.mapSetupT _ _ hok.1.1]; exact hs) (by rw [hok.1.2]; exact hi'), rfl⟩
  | .mapScope hs hi, hok => by
    simp only [SNode.renOk, Bool.and_eq_true, fixedB_iff, beq_iff_eq] at hok
    obtain ⟨r', hi', hr⟩ := IterExec.ren hi hok.2
    rw [Res.get?_ok, Res.get?_eq_some] at hr
    subst hr
    refine ⟨_, .mapScope (by rw [h.mapSetupT _ _ hok.1.1]; exact hs) (by rw [hok.1.2]; exact hi'), ?_⟩
    rw [h.scopeFinishT]
    exact dataflowOut_get? σ _
  | .consumeSetupErr _, hok => by simp [SNode.renOk] at hok
  | .consumeTasksErr _ _, hok => by simp [SNode.renOk] at hok
  | .consumeScope _ _, hok => by simp [SNode.renOk] at hok

theorem ListExec.ren {full : List SNode} {gb : Bool} {sc : SibCtx} {cs : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} :
    STree.ListExec tc full gb sc cs cst r → SNodes.renOk σ tc' tc full = true → SNodes.renOk σ tc' tc cs = true →
      ∃ r', STree.ListExec tc' (SNodes.ren σ full) gb (sc.ren σ) (SNodes.ren σ cs) cst r' ∧ r'.get? = r.get?
  | .nil hp, _, _ => ⟨_, .nil (by simpa using hp), rfl⟩
  | .nilPending hp, _, _ => ⟨_, .nilPending (by simpa using hp), rfl⟩
  | .headErr hn, _, hcs => by
    obtain ⟨r', hn', hr⟩ := NodeExec.ren hn (SNodes.renOk_cons.1 hcs).1
    obtain ⟨e', rfl⟩ := Res.exists_err_of_get? (by rw [hr]; rfl)
    exact ⟨_, .headErr hn', rfl⟩
  | .step hn hl, hf, hcs => by
    obtain ⟨r', hn', hr⟩ := NodeExec.ren hn (SNodes.renOk_cons.1 hcs).1
    rw [Res.get?_ok, Option.map_some, Res.get?_eq_some] at hr
    subst hr
    obtain ⟨r'', hl', hr'⟩ := ListExec.ren hl hf (SNodes.renOk_cons.1 hcs).2
    exact ⟨_, .step hn' hl', hr'⟩
  | .jump hn hc, _, hcs => by
    obtain ⟨r', hn', hr⟩ := NodeExec.ren hn (SNodes.renOk_cons.1 hcs).1
    rw [Res.get?_ok, Option.map_some, Res.get?_eq_some] at hr
    subst hr
    exact ⟨_, .jump hn' hc, rfl⟩
  | .gotoResume hn hl hr, hf, hcs => by
    obtain ⟨r', hn', hr'⟩ := NodeExec.ren hn (SNodes.renOk_cons.1 hcs).1
    rw [Res.get?_ok, Option.map_some, Res.get?_eq_some] at hr'
    subst hr'
    have hsuf := SNodes.renOk_resumeAt hl hf (SNodes.renOk_cons.1 hcs).2
    obtain ⟨r'', hl', hr''⟩ := ListExec.ren hr hf hsuf
    exact ⟨_, .gotoResume hn' (by rw [resumeAt_ren, hl]; rfl) hl', hr''⟩
  | .gotoEscape hn hl, _, hcs => by
    obtain ⟨r', hn', hr'⟩ := NodeExec.ren hn (SNodes.renOk_cons.1 hcs).1
    rw [Res.get?_ok, Option.map_some, Res.get?_eq_some] at hr'
    subst hr'
    exact ⟨_, .gotoEscape hn' (by rw [resumeAt_ren, hl]; rfl), rfl⟩

theorem IterExec.ren {params : List Sym} {body : List SNode} {priv : List Name} {pre : CSt} {pts : List Idx}
    {r : Res (List Log)} :
    STree.IterExec tc params body priv pre pts r → SNodes.renOk σ tc' tc body = true →
      ∃ r', STree.IterExec tc' params (SNodes.ren σ body) priv pre pts r' ∧ r'.get? = r.get?
  | .nil, _ => ⟨_, .nil, rfl⟩
  | .headErr hb he, hok => by
    obtain ⟨r', hb', hr⟩ := ListExec.ren hb hok hok
    have := iterLog_get? priv hr
    rw [he] at this
    obtain ⟨e', he'⟩ := Res.exists_err_of_get? this
    exact ⟨_, .headErr hb' he', rfl⟩
  | .tailErr hb hl hi, hok => by
    obtain ⟨r', hb', hr⟩ := ListExec.ren hb hok hok
    have := iterLog_get? priv hr
    rw [hl, Res.get?_ok, Res.get?_eq_some] at this
    obtain ⟨r'', hi', hr''⟩ := IterExec.ren hi hok
    obtain ⟨e', rfl⟩ := Res.exists_err_of_get? (by rw [hr'']; rfl)
    exact ⟨_, .tailErr hb' this hi', rfl⟩
  | .cons hb hl hi, hok => by
    obtain ⟨r', hb', hr⟩ := ListExec.ren hb hok hok
    have := iterLog_get? priv hr
    rw [hl, Res.get?_ok, Res.get?_eq_some] at this
    obtain ⟨r'', hi', hr''⟩ := IterExec.ren hi hok
    rw [Res.get?_ok, Res.get?_eq_some] at hr''
    subst hr''
    exact ⟨_, .cons hb' this hi', rfl⟩

theorem LoopExec.ren {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopExec tc l body cst r → NamesFixed σ l.exprNames → SNodes.renOk σ tc' tc body = true →
      ∃ r', STree.LoopExec tc' l (SNodes.ren σ body) cst r' ∧ r'.get? = r.get?
  | .initErr hi, hl, _ => ⟨_, .initErr (by rw [h.ctx.applyAssign? _ _ (LoopMeta.fixed_init hl)]; exact hi), rfl⟩
  | .guardFirst hi hv hg, hl, hok => by
    obtain ⟨r', hg', hr⟩ := LoopGuard.ren hg hl hok
    exact ⟨_, .guardFirst (by rw [h.ctx.applyAssign? _ _ (LoopMeta.fixed_init hl)]; exact hi) hv hg', hr⟩
  | .bodyFirst hi hv hb, hl, hok => by
    obtain ⟨r', hb', hr⟩ := LoopIter.ren hb hl hok
    exact ⟨_, .bodyFirst (by rw [h.ctx.applyAssign? _ _ (LoopMeta.fixed_init hl)]; exact hi) hv hb', hr⟩

theorem LoopGuard.ren {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopGuard tc l body cst r → NamesFixed σ l.exprNames → SNodes.renOk σ tc' tc body = true →
      ∃ r', STree.LoopGuard tc' l (SNodes.ren σ body) cst r' ∧ r'.get? = r.get?
  | .condErr hc, hl, _ => ⟨_, .condErr (by rw [h.ctx.evalBoolR _ _ _ (LoopMeta.fixed_cond hl)]; exact hc), rfl⟩
  | .exit hc, hl, _ => ⟨_, .exit (by rw [h.ctx.evalBoolR _ _ _ (LoopMeta.fixed_cond hl)]; exact hc), rfl⟩
  | .enter hc hb, hl, hok => by
    obtain ⟨r', hb', hr⟩ := LoopIter.ren hb hl hok
    exact ⟨_, .enter (by rw [h.ctx.evalBoolR _ _ _ (LoopMeta.fixed_cond hl)]; exact hc) hb', hr⟩

theorem LoopIter.ren {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopIter tc l body cst r → NamesFixed σ l.exprNames → SNodes.renOk σ tc' tc body = true →
      ∃ r', STree.LoopIter tc' l (SNodes.ren σ body) cst r' ∧ r'.get? = r.get?
  | .bodyErr hb, _, hok => by
    obtain ⟨r', hb', hr⟩ := ListExec.ren hb hok hok
    obtain ⟨e', rfl⟩ := Res.exists_err_of_get? (by rw [hr]; rfl)
    exact ⟨_, .bodyErr hb', rfl⟩
  | .ret hb, _, hok => by
    obtain ⟨r', hb', hr⟩ := ListExec.ren hb hok hok
    rw [Res.get?_ok, Res.get?_eq_some] at hr
    subst hr
    exact ⟨_, .ret hb', rfl⟩
  | .brk hb, _, hok => by
    obtain ⟨r', hb', hr⟩ := ListExec.ren hb hok hok
    rw [Res.get?_ok, Res.get?_eq_some] at hr
    subst hr
    exact ⟨_, .brk hb', rfl⟩
  | .gotoOut hb, _, hok => by
    obtain ⟨r', hb', hr⟩ := ListExec.ren hb hok hok
    rw [Res.get?_ok, Res.get?_eq_some] at hr
    subst hr
    exact ⟨_, .gotoOut hb', rfl⟩
  | .tail hb hc ht, hl, hok => by
    obtain ⟨r', hb', hr⟩ := ListExec.ren hb hok hok
    rw [Res.get?_ok, Res.get?_eq_some] at hr
    subst hr
    obtain ⟨r'', ht', hr''⟩ := LoopTail.ren ht hl hok
    exact ⟨_, .tail hb' hc ht', hr''⟩

theorem LoopTail.ren {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopTail tc l body cst r → NamesFixed σ l.exprNames → SNodes.renOk σ tc' tc body = true →
      ∃ r', STree.LoopTail tc' l (SNodes.ren σ body) cst r' ∧ r'.get? = r.get?
  | .updateErr hv hu, hl, _ =>
    ⟨_, .updateErr hv (by rw [h.ctx.applyAssign? _ _ (LoopMeta.fixed_update hl)]; exact hu), rfl⟩
  | .updateGuard hv hu hg, hl, hok => by
    obtain ⟨r', hg', hr⟩ := LoopGuard.ren hg hl hok
    exact ⟨_, .updateGuard hv (by rw [h.ctx.applyAssign? _ _ (LoopMeta.fixed_update hl)]; exact hu) hg', hr⟩
  | .condFirstErr hv hc, hl, _ =>
    ⟨_, .condFirstErr hv (by rw [h.ctx.evalBoolR _ _ _ (LoopMeta.fixed_cond hl)]; exact hc), rfl⟩
  | .condFirstExit hv hc, hl, _ =>
    ⟨_, .condFirstExit hv (by rw [h.ctx.evalBoolR _ _ _ (LoopMeta.fixed_cond hl)]; exact hc), rfl⟩
  | .condFirstUpdateErr hv hc hu, hl, _ =>
    ⟨_, .condFirstUpdateErr hv (by rw [h.ctx.evalBoolR _ _ _ (LoopMeta.fixed_cond hl)]; exact hc)
      (by rw [h.ctx.applyAssign? _ _ (LoopMeta.fixed_update hl)]; exact hu), rfl⟩
  | .condFirstNext hv hc hu hb, hl, hok => by
    obtain ⟨r', hb', hr⟩ := LoopIter.ren hb hl hok
    exact ⟨_, .condFirstNext hv (by rw [h.ctx.evalBoolR _ _ _ (LoopMeta.fixed_cond hl)]; exact hc)
      (by rw [h.ctx.applyAssign? _ _ (LoopMeta.fixed_update hl)]; exact hu) hb', hr⟩

end

end Rel

end STree

end Dace
