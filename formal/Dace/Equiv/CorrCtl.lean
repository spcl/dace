-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.Corr
import Dace.Equiv.RaceFree

/-!
# SDFG and schedule tree: structured control flow (M5b, steps 1–3)

## Interstate assignments

`sdfg_to_tree` turns the assignments of an interstate edge into one
`AssignNode` each, in dictionary order; the tree runs them one after the
other, while an SDFG edge assigns simultaneously (right-hand sides in the
old valuation, `Decisions.md#D49`).  The two agree when no right-hand side
depends on a symbol assigned before it on the same edge
(`SeqAssigns`, decided by `seqAssigns?`): the symbol is not free in the
later right-hand sides, and it is not a symbol of any descriptor, since a
data read `A[i]` in a right-hand side evaluates the shape and strides of
`A` (`Ctx.eval_congr`).  Then `assigns_seq`: running the assignments one by
one gives the result of `applyAssigns`, or the same error.
-/

set_option autoImplicit false

namespace Dace

/-! ## Sequential versus simultaneous assignment -/

/-- The assignments of an edge may run one by one: names pairwise distinct,
and no assigned symbol is read by a later right-hand side or by a
descriptor. -/
def SeqAssigns (ctx : Ctx) : List (Sym × Expr) → Prop
  | [] => True
  | (x, _) :: rest => x ∉ rest.map Prod.fst ∧ (∀ p ∈ rest, x ∉ p.2.freeSyms) ∧ x ∉ ctx.g.descSyms ∧
      SeqAssigns ctx rest

def seqAssigns? (ctx : Ctx) : List (Sym × Expr) → Bool
  | [] => true
  | (x, _) :: rest => !(rest.map Prod.fst).contains x && rest.all (fun p => !p.2.freeSyms.contains x) &&
      !ctx.g.descSyms.contains x && seqAssigns? ctx rest

theorem SeqAssigns.of_g {c₁ c₂ : Ctx} (hg : c₁.g = c₂.g) :
    ∀ {as : List (Sym × Expr)}, SeqAssigns c₁ as → SeqAssigns c₂ as
  | [], _ => trivial
  | (_, _) :: _, ⟨h₁, h₂, h₃, h₄⟩ => ⟨h₁, h₂, hg ▸ h₃, SeqAssigns.of_g hg h₄⟩

theorem seqAssigns?_sound {ctx : Ctx} : ∀ {as : List (Sym × Expr)}, seqAssigns? ctx as = true → SeqAssigns ctx as
  | [], _ => trivial
  | (x, _) :: rest, h => by
    simp only [seqAssigns?, Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, List.all_eq_true,
      List.contains_eq_mem, decide_eq_false_iff_not] at h
    exact ⟨h.1.1.1, fun p hp => h.1.1.2 p hp, h.1.2, seqAssigns?_sound h.2⟩

/-- Running the assignments one by one (the tree's `AssignNode`s). -/
def assignSeq (ctx : Ctx) : CSt → List (Sym × Expr) → Res CSt
  | cst, [] => .ok cst
  | cst, a :: rest => applyAssigns ctx cst [a] >>= fun c => assignSeq ctx c rest

theorem setAll_cons_eq_set (ρ : Valuation) (x : Sym) (v : Val) (vs : List (Sym × Val))
    (hx : x ∉ vs.map Prod.fst) :
    Valuation.setAll ρ ((x, v) :: vs) = Valuation.setAll (Valuation.setAll ρ [(x, v)]) vs := by
  funext y
  simp only [Valuation.setAll, List.find?_cons]
  by_cases hy : x = y
  · subst hy
    have : vs.find? (·.1 = x) = none := by
      rw [List.find?_eq_none]
      intro p hp hpx
      exact hx (List.mem_map.2 ⟨p, hp, by simpa using hpx⟩)
    simp [this]
  · simp [hy]

theorem applyAssigns_def (ctx : Ctx) (c : CSt) (as : List (Sym × Expr)) :
    applyAssigns ctx c as =
      ((as.mapM fun (p : Sym × Expr) => ctx.evalR c.ρ c.mem p.2 >>= fun w => Res.ok (p.1, w)) >>=
        fun vs => Res.ok { c with ρ := Valuation.setAll c.ρ vs }) := rfl

/-- **Sequential assignment equals simultaneous assignment** under
`SeqAssigns`. -/
theorem assigns_seq (ctx : Ctx) : ∀ (as : List (Sym × Expr)) (cst : CSt), SeqAssigns ctx as →
    assignSeq ctx cst as = applyAssigns ctx cst as
  | [], cst, _ => by simp [assignSeq, applyAssigns]; rfl
  | (x, e) :: rest, cst, ⟨hxr, hxf, hxd, hrest⟩ => by
    simp only [assignSeq]
    rw [applyAssigns_def ctx cst [(x, e)], applyAssigns_def ctx cst ((x, e) :: rest)]
    cases hv : ctx.evalR cst.ρ cst.mem e with
    | err err =>
      simp [hv]
    | ok v =>
      simp only [List.mapM_cons, List.mapM_nil, hv, Res.ok_bind, Res.pure_eq]
      rw [assigns_seq ctx rest _ hrest, applyAssigns_def]
      dsimp only
      have hagree : ∀ y, y ≠ x → Valuation.setAll cst.ρ [(x, v)] y = cst.ρ y := by
        intro y hy
        simp [Valuation.setAll, Ne.symm hy]
      have hd : ∀ y ∈ ctx.g.descSyms, Valuation.setAll cst.ρ [(x, v)] y = cst.ρ y :=
        fun y hy => hagree y fun h => hxd (h ▸ hy)
      have hmap : (rest.mapM fun (p : Sym × Expr) =>
            ctx.evalR (Valuation.setAll cst.ρ [(x, v)]) cst.mem p.2 >>= fun w => Res.ok (p.1, w)) =
          (rest.mapM fun (p : Sym × Expr) => ctx.evalR cst.ρ cst.mem p.2 >>= fun w => Res.ok (p.1, w)) := by
        refine mapM_congr_mem (m := Res) _ fun p hp => ?_
        rw [ctx.evalR_congr hd cst.mem p.2 fun y hy => hagree y fun h => hxf p hp (h ▸ hy)]
      rw [hmap]
      cases hV : (rest.mapM fun (p : Sym × Expr) => ctx.evalR cst.ρ cst.mem p.2 >>= fun w => Res.ok (p.1, w)) with
      | err e' => rfl
      | ok vs =>
        have hk := mapM_keys (key := Prod.fst) (fun a b hab => by
          obtain ⟨w, -, hw⟩ := Res.bind_eq_ok.1 hab
          cases hw; rfl) hV
        show Res.ok _ = Res.ok _
        rw [setAll_cons_eq_set cst.ρ x v vs (by rw [hk]; exact hxr)]

/-! ## Permuted assignments

The SDFG importer reads the assignments of an edge from a JSON object, whose
keys Lean keeps sorted; the tree lists them in dictionary order.  With
pairwise distinct names the order of a simultaneous assignment does not
matter: a permutation gives the same valuation, and an undefined right-hand
side is an `undefinedExpr` error (undefined behavior) in either order. -/

theorem find_nodup {l : List (Sym × Val)} (hn : (l.map Prod.fst).Nodup) {y : Sym} {p : Sym × Val} :
    l.find? (fun q => decide (q.1 = y)) = some p ↔ p ∈ l ∧ p.1 = y := by
  constructor
  · intro h; exact ⟨List.mem_of_find?_eq_some h, by simpa using List.find?_some h⟩
  · rintro ⟨hp, rfl⟩
    cases hf : l.find? (fun q => decide (q.1 = p.1)) with
    | none =>
      rw [List.find?_eq_none] at hf
      exact absurd (by simp) (hf p hp)
    | some q =>
      have hq := List.mem_of_find?_eq_some hf
      have hq1 : q.1 = p.1 := by simpa using List.find?_some hf
      rw [List.inj_on_of_nodup_map hn hq hp hq1]

theorem setAll_perm (ρ : Valuation) {l l' : List (Sym × Val)} (hn : (l.map Prod.fst).Nodup) (hp : l'.Perm l) :
    Valuation.setAll ρ l' = Valuation.setAll ρ l := by
  have hn' : (l'.map Prod.fst).Nodup := ((hp.map Prod.fst).nodup_iff).2 hn
  funext y
  simp only [Valuation.setAll]
  cases h : l.find? (fun q => decide (q.1 = y)) with
  | none =>
    have h' : l'.find? (fun q => decide (q.1 = y)) = none := by
      rw [List.find?_eq_none] at h ⊢
      exact fun x hx => h x (hp.subset hx)
    rw [h']
  | some p =>
    obtain ⟨hp₁, hp₂⟩ := (find_nodup hn).1 h
    rw [(find_nodup hn').2 ⟨hp.symm.subset hp₁, hp₂⟩]

/-- The evaluation of one assignment. -/
def assignEval (ctx : Ctx) (cst : CSt) (a : Sym × Expr) : Res (Sym × Val) :=
  ctx.evalR cst.ρ cst.mem a.2 >>= fun w => Res.ok (a.1, w)

theorem assignEval_err {ctx : Ctx} {cst : CSt} {a : Sym × Expr} {e : ErrReason} (h : assignEval ctx cst a = .err e) :
    e.isUB = true := by
  unfold assignEval at h
  cases hv : ctx.evalR cst.ρ cst.mem a.2 with
  | ok v => rw [hv] at h; cases h
  | err e' =>
    rw [hv] at h; cases h
    unfold Ctx.evalR Res.ofOption at hv
    split at hv <;> cases hv
    rfl

theorem mapM_assignEval_err {ctx : Ctx} {cst : CSt} :
    ∀ {as : List (Sym × Expr)} {e : ErrReason}, as.mapM (assignEval ctx cst) = .err e → e.isUB = true
  | [], _, h => by simp at h
  | a :: as, e, h => by
    rw [List.mapM_cons] at h
    cases ha : assignEval ctx cst a with
    | err e' => rw [ha] at h; cases h; exact assignEval_err ha
    | ok b =>
      rw [ha] at h
      cases hr : as.mapM (assignEval ctx cst) with
      | err e' => rw [hr] at h; cases h; exact mapM_assignEval_err hr
      | ok bs => rw [hr] at h; cases h

theorem mapM_assignEval_perm {ctx : Ctx} {cst : CSt} {as' as : List (Sym × Expr)} (hp : as'.Perm as) :
    ∀ {vs' : List (Sym × Val)}, as'.mapM (assignEval ctx cst) = .ok vs' →
      ∃ vs, as.mapM (assignEval ctx cst) = .ok vs ∧ vs'.Perm vs := by
  induction hp with
  | nil => intro vs' h; exact ⟨vs', h, .refl _⟩
  | @cons x l₁ l₂ _ ih =>
    intro vs' h
    rw [List.mapM_cons] at h ⊢
    cases hx : assignEval ctx cst x with
    | err e => rw [hx] at h; cases h
    | ok b =>
      rw [hx] at h
      cases hr : List.mapM (assignEval ctx cst) l₁ with
      | err e => rw [hr] at h; cases h
      | ok bs =>
        rw [hr] at h; cases h
        obtain ⟨vs, hvs, hpv⟩ := ih hr
        exact ⟨b :: vs, by simp [hvs], hpv.cons b⟩
  | swap x y l =>
    intro vs' h
    simp only [List.mapM_cons] at h ⊢
    cases hx : assignEval ctx cst x with
    | err e => rw [hx] at h; cases hy : assignEval ctx cst y <;> rw [hy] at h <;> cases h
    | ok bx =>
      cases hy : assignEval ctx cst y with
      | err e => rw [hx, hy] at h; cases h
      | ok y' =>
        rw [hx, hy] at h
        cases hr : List.mapM (assignEval ctx cst) l with
        | err e => rw [hr] at h; cases h
        | ok bs =>
          rw [hr] at h; cases h
          exact ⟨bx :: y' :: bs, by simp, .swap _ _ _⟩
  | trans _ _ ih₁ ih₂ =>
    intro vs' h
    obtain ⟨vs₁, h₁, hp₁⟩ := ih₁ h
    obtain ⟨vs₂, h₂, hp₂⟩ := ih₂ h₁
    exact ⟨vs₂, h₂, hp₁.trans hp₂⟩

theorem applyAssigns_eq_mapM (ctx : Ctx) (cst : CSt) (as : List (Sym × Expr)) :
    applyAssigns ctx cst as = (as.mapM (assignEval ctx cst) >>= fun vs => Res.ok { cst with ρ := Valuation.setAll cst.ρ vs }) :=
  rfl

/-- **Permuted simultaneous assignments** with distinct names: the same
normal result; an error on one side is undefined behavior on the other. -/
theorem applyAssigns_perm {ctx : Ctx} {cst : CSt} {as' as : List (Sym × Expr)} (hp : as'.Perm as)
    (hn : (as.map Prod.fst).Nodup) :
    (∀ c, applyAssigns ctx cst as' = .ok c → applyAssigns ctx cst as = .ok c) ∧
    (∀ e, applyAssigns ctx cst as' = .err e → ∃ e', e'.isUB = true ∧ applyAssigns ctx cst as = .err e') := by
  rw [applyAssigns_eq_mapM, applyAssigns_eq_mapM]
  constructor
  · intro c h
    cases hr' : as'.mapM (assignEval ctx cst) with
    | err e => rw [hr'] at h; cases h
    | ok vs' =>
      rw [hr'] at h; cases h
      obtain ⟨vs, hvs, hpv⟩ := mapM_assignEval_perm hp hr'
      rw [hvs]
      have hk := mapM_keys (key := Prod.fst) (fun a b hab => by
        obtain ⟨w, -, hw⟩ := Res.bind_eq_ok.1 hab
        cases hw; rfl) hvs
      show Res.ok _ = Res.ok _
      rw [setAll_perm cst.ρ (by rw [hk]; exact hn) hpv]
  · intro e h
    cases hr : as.mapM (assignEval ctx cst) with
    | err e' => exact ⟨e', mapM_assignEval_err hr, rfl⟩
    | ok vs =>
      exfalso
      cases hr' : as'.mapM (assignEval ctx cst) with
      | err e'' =>
        obtain ⟨vs', hvs', -⟩ := mapM_assignEval_perm hp.symm hr
        rw [hvs'] at hr'; cases hr'
      | ok vs' => rw [hr'] at h; cases h

theorem SeqAssigns.nodup {ctx : Ctx} : ∀ {as : List (Sym × Expr)}, SeqAssigns ctx as → (as.map Prod.fst).Nodup
  | [], _ => .nil
  | (_, _) :: _, ⟨h₁, _, _, h₄⟩ => List.nodup_cons.2 ⟨h₁, SeqAssigns.nodup h₄⟩

/-! ## Control signals and outcomes -/

/-- The tree signal of an SDFG control signal. -/
def Ctl.toT : Ctl → TCtl
  | .normal => .normal
  | .brk => .brk
  | .cont => .cont
  | .ret => .ret

theorem Ctl.toT_inj {a b : Ctl} (h : a.toT = b.toT) : a = b := by
  cases a <;> cases b <;> first | rfl | cases h

@[simp] theorem Ctl.toT_ne_goto (a : Ctl) (l : String) : a.toT ≠ .goto l := by cases a <;> simp [Ctl.toT]

theorem Ctl.toT_normal {a : Ctl} (h : a.toT = .normal) : a = .normal := Ctl.toT_inj (b := .normal) h

theorem Ctl.toT_continuesLoop (a : Ctl) : a.toT.continuesLoop = a.continuesLoop := by
  cases a <;> rfl

theorem Ctl.toT_isJump {a : Ctl} (h : a ≠ .normal) : a.toT.isJump = true := by
  cases a <;> first | exact absurd rfl h | rfl

namespace Corr

open STree

/-- The tree outcome `r` is matched by the SDFG outcomes `X`: `X` may be
undefined, or has an error when the tree fails, or the same control-flow state with the
corresponding signal. -/
def Out (X : Res (Ctl × CSt) → Prop) (r : Res (TCtl × CSt)) : Prop :=
  (∃ e, e.isUB = true ∧ X (.err e)) ∨ (∃ e e', r = .err e ∧ X (.err e')) ∨
    ∃ ctl c, r = .ok (ctl.toT, c) ∧ X (.ok (ctl, c))

theorem Out.mono {X Y : Res (Ctl × CSt) → Prop} (hXY : ∀ r, X r → Y r) {r : Res (TCtl × CSt)} :
    Out X r → Out Y r
  | .inl ⟨e, hu, h⟩ => .inl ⟨e, hu, hXY _ h⟩
  | .inr (.inl ⟨e, e', hr, h⟩) => .inr (.inl ⟨e, e', hr, hXY _ h⟩)
  | .inr (.inr ⟨ctl, c, hr, h⟩) => .inr (.inr ⟨ctl, c, hr, hXY _ h⟩)

/-- The same for the outcome of one tree node (with its sibling context,
whose pending inputs are empty). -/
def NOut (X : Res (Ctl × CSt) → Prop) (res : Res NodeOut) : Prop :=
  (∃ e, e.isUB = true ∧ X (.err e)) ∨ (∃ e e', res = .err e ∧ X (.err e')) ∨
    ∃ sc ctl c, sc.pending = {} ∧ res = .ok (sc, ctl.toT, c) ∧ X (.ok (ctl, c))

theorem Out.scopeOut {X : Res (Ctl × CSt) → Prop} {r : Res (TCtl × CSt)} (h : Out X r) : NOut X (scopeOut r) := by
  rcases h with ⟨e, hu, h⟩ | ⟨e, e', rfl, h⟩ | ⟨ctl, c, rfl, h⟩
  · exact .inl ⟨e, hu, h⟩
  · exact .inr (.inl ⟨e, e', rfl, h⟩)
  · exact .inr (.inr ⟨{}, ctl, c, rfl, rfl, h⟩)

/-- The outcome of a block whose tree segment is followed by `rest`: as
`Out`, except that a normal end continues with `P` (the run of `rest`). -/
def BOut (X : Res (Ctl × CSt) → Prop) (P : CSt → Prop) (r : Res (TCtl × CSt)) : Prop :=
  (∃ e, e.isUB = true ∧ X (.err e)) ∨ (∃ e e', r = .err e ∧ X (.err e')) ∨
    (∃ ctl c, ctl ≠ .normal ∧ r = .ok (ctl.toT, c) ∧ X (.ok (ctl, c))) ∨
    ∃ c, P c ∧ X (.ok (.normal, c))

theorem BOut.mono {X Y : Res (Ctl × CSt) → Prop} (hXY : ∀ r, X r → Y r) {P : CSt → Prop}
    {r : Res (TCtl × CSt)} : BOut X P r → BOut Y P r
  | .inl ⟨e, hu, h⟩ => .inl ⟨e, hu, hXY _ h⟩
  | .inr (.inl ⟨e, e', hr, h⟩) => .inr (.inl ⟨e, e', hr, hXY _ h⟩)
  | .inr (.inr (.inl ⟨ctl, c, hn, hr, h⟩)) => .inr (.inr (.inl ⟨ctl, c, hn, hr, hXY _ h⟩))
  | .inr (.inr (.inr ⟨c, hp, h⟩)) => .inr (.inr (.inr ⟨c, hp, hXY _ h⟩))

/-! ## Sibling contexts -/

/-- `elif`/`else` continue an open chain. -/
def SNode.isChainCont : SNode → Bool
  | .elifScope .. | .elseScope .. => true
  | _ => false

/-- The list does not start with a chain continuation. -/
def HeadOk (l : List SNode) : Prop := ∀ n ∈ l.head?, SNode.isChainCont n = false

def headOk? (l : List SNode) : Bool := l.head?.all fun n => !SNode.isChainCont n

theorem headOk?_sound {l : List SNode} (h : headOk? l = true) : HeadOk l := by
  intro n hn
  cases l with
  | nil => cases hn
  | cons a l =>
    simp only [List.head?_cons, Option.mem_def, Option.some.injEq] at hn
    subst hn
    simpa [headOk?] using h

theorem HeadOk.nil : HeadOk [] := fun _ h => by cases h

/-- A node not continuing a chain runs alike after a closed or an open
chain. -/
theorem nodeChainFree {tc : TCtx} {c : Option Bool} {n : SNode} {cst : CSt} {res : Res NodeOut}
    (h : STree.NodeExec tc ⟨{}, c⟩ n cst res) (hn : SNode.isChainCont n = false) : STree.NodeExec tc {} n cst res := by
  cases h with
  | pendingErr _ hp => simp [Pending.isEmpty] at hp
  | leaf hp h => exact .leaf hp h
  | namedRegion hp h => exact .namedRegion rfl h
  | gblock hp h => exact .gblock rfl h
  | ifCondErr hp hc => exact .ifCondErr rfl hc
  | ifSkip hp hc => exact .ifSkip rfl hc
  | ifTaken hp hc h => exact .ifTaken rfl hc h
  | stateIfCondErr hp hc => exact .stateIfCondErr rfl hc
  | stateIfSkip hp hc => exact .stateIfSkip rfl hc
  | stateIfTaken hp hc h => exact .stateIfTaken rfl hc h
  | loop hp hl h => exact .loop rfl hl h
  | mapSetupErr hs => exact .mapSetupErr hs
  | mapIterErr hs hi => exact .mapIterErr hs hi
  | mapScope hs hi => exact .mapScope hs hi
  | consumeSetupErr hs => exact .consumeSetupErr hs
  | consumeTasksErr hs ht => exact .consumeTasksErr hs ht
  | consumeScope hs ht => exact .consumeScope hs ht
  | _ => simp [SNode.isChainCont] at hn

theorem listChainFree {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {rest : List SNode}
    {cst : CSt} {r : Res (TCtl × CSt)} (h : STree.ListExec tc full gb sc rest cst r) (hp : sc.pending = {})
    (hh : HeadOk rest) : STree.ListExec tc full gb {} rest cst r := by
  obtain ⟨p, c⟩ := sc
  simp only at hp
  subst hp
  cases h with
  | nil _ => exact .nil rfl
  | nilPending hp => simp [Pending.isEmpty] at hp
  | headErr h => exact .headErr (nodeChainFree h (hh _ rfl))
  | step h hr => exact .step (nodeChainFree h (hh _ rfl)) hr
  | jump h hc => exact .jump (nodeChainFree h (hh _ rfl)) hc
  | gotoResume h hl hr => exact .gotoResume (nodeChainFree h (hh _ rfl)) hl hr
  | gotoEscape h hl => exact .gotoEscape (nodeChainFree h (hh _ rfl)) hl

/-! ## Inversion of tree runs -/

/-- A list run: the head node's outcome and how the list proceeds. -/
theorem list_cases {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {n : SNode} {rest : List SNode}
    {cst : CSt} {r : Res (TCtl × CSt)} (h : STree.ListExec tc full gb sc (n :: rest) cst r) :
    ∃ res, STree.NodeExec tc sc n cst res ∧
      ((∃ e, res = .err e ∧ r = .err e) ∨
       (∃ sc' c', res = .ok (sc', .normal, c') ∧ STree.ListExec tc full gb sc' rest c' r) ∨
       (∃ sc' ctl c', res = .ok (sc', ctl, c') ∧ ctl.isJump = true ∧ r = .ok (ctl, c')) ∨
       (∃ sc' l c', res = .ok (sc', .goto l, c'))) := by
  cases h with
  | headErr h => exact ⟨_, h, .inl ⟨_, rfl, rfl⟩⟩
  | step h hr => exact ⟨_, h, .inr (.inl ⟨_, _, rfl, hr⟩)⟩
  | jump h hc => exact ⟨_, h, .inr (.inr (.inl ⟨_, _, _, rfl, hc, rfl⟩))⟩
  | gotoResume h _ _ => exact ⟨_, h, .inr (.inr (.inr ⟨_, _, _, rfl⟩))⟩
  | gotoEscape h _ => exact ⟨_, h, .inr (.inr (.inr ⟨_, _, _, rfl⟩))⟩

/-- **A node matched by `X`, followed by `rest`**: the list run is matched
by `X` followed by the run of `rest`. -/
theorem node_list {tc : TCtx} {full : List SNode} {gb : Bool} {sc₀ : SibCtx} {n : SNode} {rest : List SNode}
    {cst : CSt} {r : Res (TCtl × CSt)} {X : Res (Ctl × CSt) → Prop} (hh : HeadOk rest)
    (hn : ∀ res, STree.NodeExec tc sc₀ n cst res → NOut X res) (hx : STree.ListExec tc full gb sc₀ (n :: rest) cst r) :
    BOut X (fun c => STree.ListExec tc full gb {} rest c r) r := by
  obtain ⟨res, hres, hk⟩ := list_cases hx
  rcases hn res hres with ⟨e, hu, hX⟩ | ⟨e, e', rfl, hX⟩ | ⟨sc, ctl, c, hsc, rfl, hX⟩
  · exact .inl ⟨e, hu, hX⟩
  · rcases hk with ⟨e', he', rfl⟩ | ⟨_, _, he', _⟩ | ⟨_, _, _, he', _⟩ | ⟨_, _, _, he'⟩ <;> cases he'
    exact .inr (.inl ⟨e, e', rfl, hX⟩)
  · rcases hk with ⟨_, he', _⟩ | ⟨sc', c', he', hr⟩ | ⟨sc', ctl', c', he', hj, rfl⟩ | ⟨_, l, _, he'⟩
    · cases he'
    · simp only [Res.ok.injEq, Prod.mk.injEq] at he'
      obtain ⟨rfl, hctl, rfl⟩ := he'
      cases Ctl.toT_normal hctl
      exact .inr (.inr (.inr ⟨c, listChainFree hr hsc hh, hX⟩))
    · simp only [Res.ok.injEq, Prod.mk.injEq] at he'
      obtain ⟨rfl, rfl, rfl⟩ := he'
      refine .inr (.inr (.inl ⟨ctl, c, fun hn' => ?_, rfl, hX⟩))
      subst hn'; cases hj
    · simp only [Res.ok.injEq, Prod.mk.injEq] at he'
      exact absurd he'.2.1 (Ctl.toT_ne_goto _ _)

theorem loop_inv {tc : TCtx} {n : SNode} {lm : LoopMeta} {body : List SNode} {cst : CSt} {res : Res NodeOut}
    (hn : n.loop? = some (lm, body)) (h : STree.NodeExec tc {} n cst res) :
    ∃ r', STree.LoopExec tc lm body cst r' ∧ res = scopeOut r' := by
  cases h with
  | loop _ hl h => rw [hn] at hl; cases hl; exact ⟨_, h, rfl⟩
  | pendingErr _ hp => simp [Pending.isEmpty] at hp
  | leaf _ h' => cases n <;> simp_all [stepLeaf, SNode.loop?]
  | _ => simp [SNode.loop?] at hn

theorem leafStep_inv {tc : TCtx} {n : SNode} {cst : CSt} {res r : Res NodeOut}
    (hs : stepLeaf tc {} n cst = some r) (h : STree.NodeExec tc {} n cst res) : res = r := by
  cases h with
  | pendingErr _ hp => simp [Pending.isEmpty] at hp
  | leaf _ h' => rw [hs] at h'; exact (Option.some.inj h').symm
  | loop _ hl _ => cases n <;> simp_all [stepLeaf, SNode.loop?]
  | _ => simp [stepLeaf] at hs

theorem if_inv {tc : TCtx} {pc : PyExpr} {body : List SNode} {cst : CSt} {res : Res NodeOut}
    (h : STree.NodeExec tc {} (.ifScope pc body) cst res) :
    (∃ e, tc.ctx.evalBoolR cst.ρ cst.mem pc.expr = .err e ∧ res = .err e) ∨
    (tc.ctx.evalBoolR cst.ρ cst.mem pc.expr = .ok false ∧ res = .ok ({ chain := some false }, .normal, cst)) ∨
    ∃ r', tc.ctx.evalBoolR cst.ρ cst.mem pc.expr = .ok true ∧ STree.ListExec tc body false {} body cst r' ∧
      res = chainOut r' := by
  generalize hn : SNode.ifScope pc body = n at h
  cases h with
  | pendingErr _ hp => simp [Pending.isEmpty] at hp
  | leaf _ h' => subst hn; simp [stepLeaf] at h'
  | loop _ hl _ => subst hn; simp [SNode.loop?] at hl
  | ifCondErr _ hc => cases hn; exact .inl ⟨_, hc, rfl⟩
  | ifSkip _ hc => cases hn; exact .inr (.inl ⟨hc, rfl⟩)
  | ifTaken _ hc hb => cases hn; exact .inr (.inr ⟨_, hc, hb, rfl⟩)
  | _ => cases hn

theorem elif_inv {tc : TCtx} {pc : PyExpr} {body : List SNode} {cst : CSt} {res : Res NodeOut}
    (h : STree.NodeExec tc { chain := some false } (.elifScope pc body) cst res) :
    (∃ e, tc.ctx.evalBoolR cst.ρ cst.mem pc.expr = .err e ∧ res = .err e) ∨
    (tc.ctx.evalBoolR cst.ρ cst.mem pc.expr = .ok false ∧ res = .ok ({ chain := some false }, .normal, cst)) ∨
    ∃ r', tc.ctx.evalBoolR cst.ρ cst.mem pc.expr = .ok true ∧ STree.ListExec tc body false {} body cst r' ∧
      res = chainOut r' := by
  generalize hn : SNode.elifScope pc body = n at h
  cases h with
  | pendingErr _ hp => simp [Pending.isEmpty] at hp
  | leaf _ h' => subst hn; simp [stepLeaf] at h'
  | loop _ hl _ => subst hn; simp [SNode.loop?] at hl
  | elifNoChain _ hc => cases hc
  | elifAfterTaken _ hc => cases hc
  | elifCondErr _ _ hc => cases hn; exact .inl ⟨_, hc, rfl⟩
  | elifSkip _ _ hc => cases hn; exact .inr (.inl ⟨hc, rfl⟩)
  | elifTaken _ _ hc hb => cases hn; exact .inr (.inr ⟨_, hc, hb, rfl⟩)
  | _ => cases hn

theorem else_inv {tc : TCtx} {body : List SNode} {cst : CSt} {res : Res NodeOut}
    (h : STree.NodeExec tc { chain := some false } (.elseScope body) cst res) :
    ∃ r', STree.ListExec tc body false {} body cst r' ∧ res = scopeOut r' := by
  generalize hn : SNode.elseScope body = n at h
  cases h with
  | pendingErr _ hp => simp [Pending.isEmpty] at hp
  | leaf _ h' => subst hn; simp [stepLeaf] at h'
  | loop _ hl _ => subst hn; simp [SNode.loop?] at hl
  | elseNoChain _ hc => cases hc
  | elseAfterTaken _ hc => cases hc
  | elseTaken _ _ hb => cases hn; exact ⟨_, hb, rfl⟩
  | _ => cases hn

theorem elif_inv_taken {tc : TCtx} {pc : PyExpr} {body : List SNode} {cst : CSt} {res : Res NodeOut}
    (h : STree.NodeExec tc { chain := some true } (.elifScope pc body) cst res) :
    res = .ok ({ chain := some true }, .normal, cst) := by
  generalize hn : SNode.elifScope pc body = n at h
  cases h with
  | pendingErr _ hp => simp [Pending.isEmpty] at hp
  | leaf _ h' => subst hn; simp [stepLeaf] at h'
  | loop _ hl _ => subst hn; simp [SNode.loop?] at hl
  | elifNoChain _ hc => cases hc
  | elifAfterTaken _ _ => cases hn; rfl
  | elifCondErr _ hc _ => cases hc
  | elifSkip _ hc _ => cases hc
  | elifTaken _ hc _ _ => cases hc
  | _ => cases hn

theorem else_inv_taken {tc : TCtx} {body : List SNode} {cst : CSt} {res : Res NodeOut}
    (h : STree.NodeExec tc { chain := some true } (.elseScope body) cst res) : res = .ok ({}, .normal, cst) := by
  generalize hn : SNode.elseScope body = n at h
  cases h with
  | pendingErr _ hp => simp [Pending.isEmpty] at hp
  | leaf _ h' => subst hn; simp [stepLeaf] at h'
  | loop _ hl _ => subst hn; simp [SNode.loop?] at hl
  | elseNoChain _ hc => cases hc
  | elseAfterTaken _ _ => cases hn; rfl
  | elseTaken _ hc _ => cases hc
  | _ => cases hn

/-! ## Loops -/

/-- A loop region and the `LoopMeta` of a loop scope agree. -/
structure LoopCorr (l : LoopRegion) (lm : LoopMeta) : Prop where
  init : lm.initAssign = l.init
  cond : lm.cond.expr = l.cond
  update : lm.updateAssign = l.update
  inverted : lm.inverted = l.inverted
  /-- Only this combination is read (`LoopTail`). -/
  ubc : (lm.inverted && !lm.updateBeforeCond) = (l.inverted && !l.updateBeforeCond)

def loopCorr? (l : LoopRegion) (lm : LoopMeta) : Bool :=
  lm.initAssign = l.init && lm.cond.expr = l.cond && lm.updateAssign = l.update && lm.inverted = l.inverted &&
    (lm.inverted && !lm.updateBeforeCond) = (l.inverted && !l.updateBeforeCond)

theorem loopCorr?_sound {l : LoopRegion} {lm : LoopMeta} (h : loopCorr? l lm = true) : LoopCorr l lm := by
  simp only [loopCorr?, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1.1.1, h.1.1.1.2, h.1.1.2, h.1.2, h.2⟩

section Loop

variable {ctx : Ctx} {tc : TCtx} (he : tc.ctx.eraseL = ctx.eraseL)

include he in
theorem evalBoolR_tc (ρ : Valuation) (m : Mem) (e : Expr) : tc.ctx.evalBoolR ρ m e = ctx.evalBoolR ρ m e :=
  Ctx.leaf_eq he (fun c => c.evalBoolR ρ m e) (fun _ => rfl)

include he in
theorem applyAssign?_tc (cst : CSt) (o : Option (Sym × Expr)) : applyAssign? tc.ctx cst o = applyAssign? ctx cst o :=
  Ctx.leaf_eq he (fun c => applyAssign? c cst o) (fun _ => rfl)

include he in
theorem applyAssigns_tc (cst : CSt) (as : List (Sym × Expr)) : applyAssigns tc.ctx cst as = applyAssigns ctx cst as :=
  Ctx.leaf_eq he (fun c => applyAssigns c cst as) (fun _ => rfl)

variable {l : LoopRegion} {lm : LoopMeta} {body : List SNode} (hm : LoopCorr l lm)
  (hbody : ∀ cst r, STree.ListExec tc body false {} body cst r → Out (RegionExec ctx cst l.body) r)
include he hm hbody

set_option linter.unusedSectionVars false

mutual

theorem LoopExecT.matched {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopExec tc lm body cst r → Out (LoopExec ctx cst l) r
  | .initErr hi => .inr (.inl ⟨_, _, rfl, .initErr (by rw [← hm.init, ← applyAssign?_tc he]; exact hi)⟩)
  | .guardFirst hi hv hg => (LoopGuardT.matched hg).mono fun _ h =>
      .guardFirst (by rw [← hm.init, ← applyAssign?_tc he]; exact hi) (hm.inverted ▸ hv) h
  | .bodyFirst hi hv hb => (LoopIterT.matched hb).mono fun _ h =>
      .bodyFirst (by rw [← hm.init, ← applyAssign?_tc he]; exact hi) (hm.inverted ▸ hv) h

theorem LoopGuardT.matched {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopGuard tc lm body cst r → Out (LoopGuard ctx cst l) r
  | .condErr hc => .inr (.inl ⟨_, _, rfl, .condErr (by rw [← hm.cond, ← evalBoolR_tc he]; exact hc)⟩)
  | .exit hc => .inr (.inr ⟨.normal, _, rfl, .exit (by rw [← hm.cond, ← evalBoolR_tc he]; exact hc)⟩)
  | .enter hc hb => (LoopIterT.matched hb).mono fun _ h =>
      .enter (by rw [← hm.cond, ← evalBoolR_tc he]; exact hc) h

theorem LoopIterT.matched {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopIter tc lm body cst r → Out (LoopIter ctx cst l) r
  | .bodyErr hb => by
    rcases hbody _ _ hb with ⟨e, hu, h⟩ | ⟨e, e', hr, h⟩ | ⟨ctl, c, hr, _⟩
    · exact .inl ⟨e, hu, .bodyErr h⟩
    · cases hr; exact .inr (.inl ⟨_, _, rfl, .bodyErr h⟩)
    · cases hr
  | .ret hb => by
    rcases hbody _ _ hb with ⟨e, hu, h⟩ | ⟨e, e', hr, _⟩ | ⟨ctl, c, hr, h⟩
    · exact .inl ⟨e, hu, .bodyErr h⟩
    · cases hr
    · simp only [Res.ok.injEq, Prod.mk.injEq] at hr
      obtain ⟨hc, rfl⟩ := hr
      cases Ctl.toT_inj (a := .ret) hc
      exact .inr (.inr ⟨.ret, _, rfl, .ret h⟩)
  | .brk hb => by
    rcases hbody _ _ hb with ⟨e, hu, h⟩ | ⟨e, e', hr, _⟩ | ⟨ctl, c, hr, h⟩
    · exact .inl ⟨e, hu, .bodyErr h⟩
    · cases hr
    · simp only [Res.ok.injEq, Prod.mk.injEq] at hr
      obtain ⟨hc, rfl⟩ := hr
      cases Ctl.toT_inj (a := .brk) hc
      exact .inr (.inr ⟨.normal, _, rfl, .brk h⟩)
  | .gotoOut hb => by
    rcases hbody _ _ hb with ⟨e, hu, h⟩ | ⟨e, e', hr, _⟩ | ⟨ctl, c, hr, h⟩
    · exact .inl ⟨e, hu, .bodyErr h⟩
    · cases hr
    · simp only [Res.ok.injEq, Prod.mk.injEq] at hr
      exact absurd hr.1.symm (Ctl.toT_ne_goto _ _)
  | .tail hb hc ht => by
    rcases hbody _ _ hb with ⟨e, hu, h⟩ | ⟨e, e', hr, _⟩ | ⟨ctl, c, hr, h⟩
    · exact .inl ⟨e, hu, .bodyErr h⟩
    · cases hr
    · simp only [Res.ok.injEq, Prod.mk.injEq] at hr
      obtain ⟨rfl, rfl⟩ := hr
      rw [Ctl.toT_continuesLoop] at hc
      exact (LoopTailT.matched ht).mono fun _ h' => .tail h hc h'

theorem LoopTailT.matched {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopTail tc lm body cst r → Out (LoopTail ctx cst l) r
  | .updateErr hv hu => .inr (.inl ⟨_, _, rfl, .updateErr (by rw [← hm.ubc]; exact hv)
      (by rw [← hm.update, ← applyAssign?_tc he]; exact hu)⟩)
  | .updateGuard hv hu hg => (LoopGuardT.matched hg).mono fun _ h =>
      .updateGuard (by rw [← hm.ubc]; exact hv) (by rw [← hm.update, ← applyAssign?_tc he]; exact hu) h
  | .condFirstErr hv hc => .inr (.inl ⟨_, _, rfl, .condFirstErr (by rw [← hm.ubc]; exact hv)
      (by rw [← hm.cond, ← evalBoolR_tc he]; exact hc)⟩)
  | .condFirstExit hv hc => .inr (.inr ⟨.normal, _, rfl, .condFirstExit (by rw [← hm.ubc]; exact hv)
      (by rw [← hm.cond, ← evalBoolR_tc he]; exact hc)⟩)
  | .condFirstUpdateErr hv hc hu => .inr (.inl ⟨_, _, rfl, .condFirstUpdateErr
      (by rw [← hm.ubc]; exact hv) (by rw [← hm.cond, ← evalBoolR_tc he]; exact hc)
      (by rw [← hm.update, ← applyAssign?_tc he]; exact hu)⟩)
  | .condFirstNext hv hc hu hb => (LoopIterT.matched hb).mono fun _ h =>
      .condFirstNext (by rw [← hm.ubc]; exact hv) (by rw [← hm.cond, ← evalBoolR_tc he]; exact hc)
        (by rw [← hm.update, ← applyAssign?_tc he]; exact hu) h

end

end Loop

/-! ## Edges and assignments -/

theorem evalBoolR_true (ctx : Ctx) (ρ : Valuation) (m : Mem) : ctx.evalBoolR ρ m (.lit (.bool true)) = .ok true := by
  simp [Ctx.evalBoolR, Ctx.evalBool, Expr.evalBool, Expr.eval, Val.toBool?, Res.ofOption]

theorem trueEdges_uncond {ctx : Ctx} {rg : Region} {cst : CSt} {bid : BlockId} {e : IEdge}
    (he : rg.outEdges bid = [e]) (hc : e.cond = .lit (.bool true)) : trueEdges ctx rg cst bid = .ok [e] := by
  simp [trueEdges, he, List.filterMapM, List.filterMapM.loop, hc, evalBoolR_true]

theorem trueEdges_nil {ctx : Ctx} {rg : Region} {cst : CSt} {bid : BlockId} (he : rg.outEdges bid = []) :
    trueEdges ctx rg cst bid = .ok [] := by
  simp only [trueEdges, he]
  rfl

theorem evalBoolR_int_one (ctx : Ctx) (ρ : Valuation) (m : Mem) : ctx.evalBoolR ρ m (.lit (.int 1)) = .ok true := by
  simp [Ctx.evalBoolR, Ctx.evalBool, Expr.evalBool, Expr.eval, Val.toBool?, Res.ofOption]

theorem trueEdges_isUncond {ctx : Ctx} {rg : Region} {cst : CSt} {bid : BlockId} {e : IEdge}
    (he : rg.outEdges bid = [e]) (hc : e.isUnconditional = true) : trueEdges ctx rg cst bid = .ok [e] := by
  simp only [IEdge.isUnconditional, Bool.or_eq_true, decide_eq_true_eq] at hc
  rcases hc with hc | hc
  · exact trueEdges_uncond he hc
  · simp [trueEdges, he, List.filterMapM, List.filterMapM.loop, hc, evalBoolR_int_one]

/-- The tree's `AssignNode`s of an edge's assignments, in order. -/
def AssignsCorr (as : List (Sym × Expr)) (ns : List SNode) : Prop :=
  List.Forall₂ (fun a n => ∃ pe : PyExpr, n = .assign a.1 pe ∧ pe.expr = a.2) as ns

/-- Take `k` leading `AssignNode`s of a tree list: their assignments and the
rest of the list. -/
def takeAssignNodes : Nat → List SNode → Option (List (Sym × Expr) × List SNode)
  | 0, cs => some ([], cs)
  | k + 1, .assign x pe :: cs => (takeAssignNodes k cs).map fun p => ((x, pe.expr) :: p.1, p.2)
  | _ + 1, _ => none

theorem takeAssignNodes_sound : ∀ {k : Nat} {cs : List SNode} {as : List (Sym × Expr)} {rest : List SNode},
    takeAssignNodes k cs = some (as, rest) → ∃ ns, cs = ns ++ rest ∧ AssignsCorr as ns
  | 0, cs, as, rest, h => by
    simp only [takeAssignNodes, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨[], rfl, .nil⟩
  | k + 1, c :: cs, as, rest, h => by
    cases c with
    | assign x pe =>
      simp only [takeAssignNodes, Option.map_eq_some_iff] at h
      obtain ⟨⟨as', rest'⟩, h', he⟩ := h
      simp only [Prod.mk.injEq] at he
      obtain ⟨rfl, rfl⟩ := he
      obtain ⟨ns, rfl, hns⟩ := takeAssignNodes_sound h'
      exact ⟨.assign x pe :: ns, rfl, .cons ⟨pe, rfl, rfl⟩ hns⟩
    | _ => simp [takeAssignNodes] at h
  | _ + 1, [], _, _, h => by simp [takeAssignNodes] at h

section Assigns

variable {ctx : Ctx} {tc : TCtx} (he : tc.ctx.eraseL = ctx.eraseL)
include he

/-- A run of the `AssignNode`s of an edge, followed by `rest`. -/
theorem assigns_run {full : List SNode} {gb : Bool} {rest : List SNode} {r : Res (TCtl × CSt)} :
    ∀ {as : List (Sym × Expr)} {ns : List SNode} {cst : CSt}, AssignsCorr as ns →
      STree.ListExec tc full gb {} (ns ++ rest) cst r →
      (∃ e, r = .err e ∧ assignSeq ctx cst as = .err e) ∨
      ∃ c, assignSeq ctx cst as = .ok c ∧ STree.ListExec tc full gb {} rest c r
  | [], [], cst, .nil, hx => .inr ⟨cst, rfl, hx⟩
  | (x, e) :: as, n :: ns, cst, .cons ⟨pe, hn, hpe⟩ hns, hx => by
    subst hn
    simp only at hpe
    have hstep : stepLeaf tc {} (.assign x pe) cst = some (leafOut (applyAssigns ctx cst [(x, e)])) := by
      rw [← hpe, ← applyAssigns_tc he]; rfl
    rcases leaf_inv hstep hx with ⟨e', hR, rfl⟩ | ⟨c', hR, hx'⟩
    · exact .inl ⟨e', rfl, by simp only [assignSeq, hR]; rfl⟩
    · rcases assigns_run hns hx' with ⟨e', rfl, h'⟩ | ⟨c, h', hx''⟩
      · exact .inl ⟨e', rfl, by simp only [assignSeq, hR, Res.ok_bind]; exact h'⟩
      · exact .inr ⟨c, by simp only [assignSeq, hR, Res.ok_bind]; exact h', hx''⟩

end Assigns

/-! ## The correspondence -/

mutual

/-- The tree list `cs` corresponds to the region `rg` run from block `bid`:
each block's segment (`BlockCorr`), then the `AssignNode`s of its single
unconditional out-edge (its assignments in some order that may run one by
one, `SeqAssigns`), then the segment of the edge's target. -/
inductive RegCorr (ctx : Ctx) (tc : TCtx) : Region → BlockId → List SNode → Prop where
  | last {rg : Region} {bid : BlockId} {b : Block} {cs : List SNode}
      (hb : rg.block? bid = some b) (hc : BlockCorr ctx tc b cs) (he : rg.outEdges bid = []) :
      RegCorr ctx tc rg bid cs
  | step {rg : Region} {bid : BlockId} {b : Block} {cs ns rest : List SNode} {e : IEdge} {as : List (Sym × Expr)}
      (hb : rg.block? bid = some b) (hc : BlockCorr ctx tc b cs) (he : rg.outEdges bid = [e])
      (hu : e.isUnconditional = true) (hp : as.Perm e.assigns) (ha : AssignsCorr as ns) (hs : SeqAssigns ctx as)
      (hh : HeadOk (ns ++ rest)) (h : RegCorr ctx tc rg e.dst rest) :
      RegCorr ctx tc rg bid (cs ++ (ns ++ rest))

/-- The tree segment of one block. -/
inductive BlockCorr (ctx : Ctx) (tc : TCtx) : Block → List SNode → Prop where
  | state {s : State} {cs : List SNode} (h : StateCorr ctx tc s cs) : BlockCorr ctx tc (.state s) cs
  | loop {l : LoopRegion} {n : SNode} {lm : LoopMeta} {body : List SNode} (hn : n.loop? = some (lm, body))
      (hm : LoopCorr l lm) (hb : RegCorr ctx tc l.body l.body.start body) : BlockCorr ctx tc (.loop l) [n]
  | cond {label : String} {c : Expr} {rg : Region} {brs : List (Option Expr × Region)} {pc : PyExpr}
      {body tl : List SNode} (hc : pc.expr = c) (hb : RegCorr ctx tc rg rg.start body) (ht : ElifCorr ctx tc brs tl) :
      BlockCorr ctx tc (.cond label ((some c, rg) :: brs)) (.ifScope pc body :: tl)
  | brk : BlockCorr ctx tc .breakBlock [.breakNode]
  | cont : BlockCorr ctx tc .continueBlock [.continueNode]
  | ret : BlockCorr ctx tc .returnBlock [.goto none]
  /-- A plain or function-call region: `sdfg_to_tree` inlines its blocks. -/
  | flat {rg : Region} {cs : List SNode} (h : RegCorr ctx tc rg rg.start cs) : BlockCorr ctx tc (.region rg) cs
  /-- A named region: a `NamedRegionScope`. -/
  | named {rg : Region} {label : String} {cs : List SNode} (h : RegCorr ctx tc rg rg.start cs) :
      BlockCorr ctx tc (.region rg) [.namedRegion label cs]

/-- The `elif`/`else` scopes after the `if` of a conditional block. -/
inductive ElifCorr (ctx : Ctx) (tc : TCtx) : List (Option Expr × Region) → List SNode → Prop where
  | nil : ElifCorr ctx tc [] []
  | elif {c : Expr} {rg : Region} {brs : List (Option Expr × Region)} {pc : PyExpr} {body tl : List SNode}
      (hc : pc.expr = c) (hb : RegCorr ctx tc rg rg.start body) (ht : ElifCorr ctx tc brs tl) :
      ElifCorr ctx tc ((some c, rg) :: brs) (.elifScope pc body :: tl)
  | els {rg : Region} {body : List SNode} (hb : RegCorr ctx tc rg rg.start body) :
      ElifCorr ctx tc [(none, rg)] [.elseScope body]

end

/-- After a taken branch the remaining `elif`/`else` scopes are skipped. -/
theorem ElifCorr.skip {ctx : Ctx} {tc : TCtx} {full : List SNode} {gb : Bool} {rest : List SNode}
    {r : Res (TCtl × CSt)} :
    ∀ {brs : List (Option Expr × Region)} {tl : List SNode} {cst : CSt}, ElifCorr ctx tc brs tl →
      STree.ListExec tc full gb { chain := some true } (tl ++ rest) cst r →
      ∃ sc : SibCtx, sc.pending = {} ∧ STree.ListExec tc full gb sc rest cst r
  | _, _, _, .nil, hx => ⟨_, rfl, hx⟩
  | _, _, _, .elif _ _ ht, hx => by
    obtain ⟨res, hres, hk⟩ := list_cases hx
    rw [elif_inv_taken hres] at hk
    rcases hk with ⟨_, he', _⟩ | ⟨sc', c', he', hr⟩ | ⟨_, _, _, he', hj, _⟩ | ⟨_, _, _, he'⟩
    · cases he'
    · simp only [Res.ok.injEq, Prod.mk.injEq] at he'
      obtain ⟨rfl, -, rfl⟩ := he'
      exact ElifCorr.skip ht hr
    · simp only [Res.ok.injEq, Prod.mk.injEq] at he'
      obtain ⟨-, rfl, -⟩ := he'
      cases hj
    · simp only [Res.ok.injEq, Prod.mk.injEq] at he'
      cases he'.2.1
  | _, _, _, .els _, hx => by
    obtain ⟨res, hres, hk⟩ := list_cases hx
    rw [else_inv_taken hres] at hk
    rcases hk with ⟨_, he', _⟩ | ⟨sc', c', he', hr⟩ | ⟨_, _, _, he', hj, _⟩ | ⟨_, _, _, he'⟩
    · cases he'
    · simp only [Res.ok.injEq, Prod.mk.injEq] at he'
      obtain ⟨rfl, -, rfl⟩ := he'
      exact ⟨_, rfl, hr⟩
    · simp only [Res.ok.injEq, Prod.mk.injEq] at he'
      obtain ⟨-, rfl, -⟩ := he'
      cases hj
    · simp only [Res.ok.injEq, Prod.mk.injEq] at he'
      cases he'.2.1

/-! ## Matching -/

theorem HeadOk.append {l rest : List SNode} (hl : HeadOk l) (hr : HeadOk rest) : HeadOk (l ++ rest) := by
  cases l with
  | nil => exact hr
  | cons a l => intro n hn; exact hl n (by simpa using hn)

theorem HeadOk.of_append {l rest : List SNode} (h : HeadOk (l ++ rest)) : HeadOk l := by
  cases l with
  | nil => exact HeadOk.nil
  | cons a l => intro n hn; exact h n (by simpa using hn)

/-- A block outcome followed by the empty list is a list outcome. -/
theorem BOut.toOut {X : Res (Ctl × CSt) → Prop} {tc : TCtx} {full : List SNode} {gb : Bool} {r : Res (TCtl × CSt)}
    (h : BOut X (fun c => STree.ListExec tc full gb {} [] c r) r) : Out X r := by
  rcases h with ⟨e, hu, h⟩ | ⟨e, e', hr, h⟩ | ⟨ctl, c, -, hr, h⟩ | ⟨c, hl, h⟩
  · exact .inl ⟨e, hu, h⟩
  · exact .inr (.inl ⟨e, e', hr, h⟩)
  · exact .inr (.inr ⟨ctl, c, hr, h⟩)
  · rw [STree.ListExec_nil rfl hl]
    exact .inr (.inr ⟨.normal, c, rfl, h⟩)

theorem ListExec.append_nil {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {cs : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} (h : STree.ListExec tc full gb sc cs cst r) : STree.ListExec tc full gb sc (cs ++ []) cst r := by
  rw [List.append_nil]; exact h

theorem namedRegion_inv {tc : TCtx} {label : String} {body : List SNode} {cst : CSt} {res : Res NodeOut}
    (h : STree.NodeExec tc {} (.namedRegion label body) cst res) :
    ∃ r', STree.ListExec tc body false {} body cst r' ∧ res = scopeOut r' := by
  generalize hn : SNode.namedRegion label body = n at h
  cases h with
  | pendingErr _ hp => simp [Pending.isEmpty] at hp
  | leaf _ h' => subst hn; simp [stepLeaf] at h'
  | loop _ hl _ => subst hn; simp [SNode.loop?] at hl
  | namedRegion _ hb => cases hn; exact ⟨_, hb, rfl⟩
  | _ => cases hn

section Matched

variable {ctx : Ctx} {tc : TCtx} (he : tc.ctx.eraseL = ctx.eraseL) (hcons : tc.consumed = none)
  (hw : ∀ op v, ctx.I.wcr op .undef v = .undef)

include he in
/-- One `if`/`elif` scope of a chain, from either sibling context. -/
theorem chain_step {full : List SNode} {gb : Bool} {sc : SibCtx} {n : SNode} {c : Expr} {rg : Region}
    {brs : List (Option Expr × Region)} {body tl rest : List SNode} {cst : CSt} {r : Res (TCtl × CSt)}
    (hinv : ∀ res, STree.NodeExec tc sc n cst res →
      (∃ e, tc.ctx.evalBoolR cst.ρ cst.mem c = .err e ∧ res = .err e) ∨
      (tc.ctx.evalBoolR cst.ρ cst.mem c = .ok false ∧ res = .ok ({ chain := some false }, .normal, cst)) ∨
      ∃ r', tc.ctx.evalBoolR cst.ρ cst.mem c = .ok true ∧ STree.ListExec tc body false {} body cst r' ∧
        res = chainOut r')
    (hbody : ∀ r', STree.ListExec tc body false {} body cst r' → Out (RegionAt ctx rg cst rg.start) r')
    (htail : STree.ListExec tc full gb { chain := some false } (tl ++ rest) cst r →
      BOut (CondExec ctx cst brs) (fun c => STree.ListExec tc full gb {} rest c r) r)
    (hskip : ∀ c, STree.ListExec tc full gb { chain := some true } (tl ++ rest) c r →
      ∃ sc : SibCtx, sc.pending = {} ∧ STree.ListExec tc full gb sc rest c r)
    (hh : HeadOk rest) (hx : STree.ListExec tc full gb sc (n :: (tl ++ rest)) cst r) :
    BOut (CondExec ctx cst ((some c, rg) :: brs)) (fun c => STree.ListExec tc full gb {} rest c r) r := by
  obtain ⟨res, hres, hk⟩ := list_cases hx
  rcases hinv res hres with ⟨e, hc, rfl⟩ | ⟨hc, rfl⟩ | ⟨r', hc, hb, rfl⟩
  · rw [evalBoolR_tc he] at hc
    rcases hk with ⟨e', he', rfl⟩ | ⟨_, _, he', _⟩ | ⟨_, _, _, he', _⟩ | ⟨_, _, _, he'⟩ <;> cases he'
    exact .inr (.inl ⟨e, _, rfl, .condErr hc⟩)
  · rw [evalBoolR_tc he] at hc
    rcases hk with ⟨_, he', _⟩ | ⟨sc', c', he', hr⟩ | ⟨_, ctl, _, he', hj, _⟩ | ⟨_, _, _, he'⟩
    · cases he'
    · simp only [Res.ok.injEq, Prod.mk.injEq] at he'
      obtain ⟨rfl, -, rfl⟩ := he'
      exact (htail hr).mono fun _ h => .skipped hc h
    · simp only [Res.ok.injEq, Prod.mk.injEq] at he'
      obtain ⟨-, rfl, -⟩ := he'
      cases hj
    · simp only [Res.ok.injEq, Prod.mk.injEq] at he'
      cases he'.2.1
  · rw [evalBoolR_tc he] at hc
    rcases hbody r' hb with ⟨e, hu, h⟩ | ⟨e, e', rfl, h⟩ | ⟨ctl, c', rfl, h⟩
    · exact .inl ⟨e, hu, .taken hc (.start h)⟩
    · rcases hk with ⟨e', he', rfl⟩ | ⟨_, _, he', _⟩ | ⟨_, _, _, he', _⟩ | ⟨_, _, _, he'⟩ <;> cases he'
      exact .inr (.inl ⟨e, _, rfl, .taken hc (.start h)⟩)
    · rcases hk with ⟨_, he', _⟩ | ⟨sc', c'', he', hr⟩ | ⟨_, ctl', _, he', hj, rfl⟩ | ⟨_, _, _, he'⟩
      · cases he'
      · simp only [chainOut, Res.map, Res.ok.injEq, Prod.mk.injEq] at he'
        obtain ⟨rfl, hn, rfl⟩ := he'
        cases Ctl.toT_normal hn
        obtain ⟨sc'', hp, hr'⟩ := hskip _ hr
        exact .inr (.inr (.inr ⟨_, listChainFree hr' hp hh, .taken hc (.start h)⟩))
      · simp only [chainOut, Res.map, Res.ok.injEq, Prod.mk.injEq] at he'
        obtain ⟨-, rfl, rfl⟩ := he'
        refine .inr (.inr (.inl ⟨ctl, c', fun hn => ?_, rfl, .taken hc (.start h)⟩))
        subst hn; cases hj
      · simp only [chainOut, Res.map, Res.ok.injEq, Prod.mk.injEq] at he'
        exact absurd he'.2.1 (Ctl.toT_ne_goto _ _)

include he hcons hw

set_option linter.unusedSectionVars false

mutual

theorem RegCorr.matched {rg : Region} {bid : BlockId} {cs : List SNode} :
    RegCorr ctx tc rg bid cs → ∀ {full : List SNode} {gb : Bool} {rest : List SNode} {cst : CSt}
      {r : Res (TCtl × CSt)}, HeadOk rest → STree.ListExec tc full gb {} (cs ++ rest) cst r →
      BOut (RegionAt ctx rg cst bid) (fun c => STree.ListExec tc full gb {} rest c r) r
  | .last hb hc hout, full, gb, rest, cst, r, hh, hx => by
    rcases BlockCorr.matched hc hh hx with ⟨e, hu, h⟩ | ⟨e, e', hr, h⟩ | ⟨ctl, c, hn, hr, h⟩ | ⟨c, hl, h⟩
    · exact .inl ⟨e, hu, .blockErr hb h⟩
    · exact .inr (.inl ⟨e, _, hr, .blockErr hb h⟩)
    · exact .inr (.inr (.inl ⟨ctl, c, hn, hr, .exit hb h hn⟩))
    · exact .inr (.inr (.inr ⟨c, hl, .stop hb h (trueEdges_nil hout)⟩))
  | .step hb hc hout hu hp ha hs hh h, full, gb, rest, cst, r, hhr, hx => by
    have hn : (_ : List (Sym × Expr)).map Prod.fst |>.Nodup := ((hp.map Prod.fst).nodup_iff).1 hs.nodup
    rw [List.append_assoc] at hx
    rcases BlockCorr.matched hc (hh.append hhr) hx with
      ⟨e, hu', h₁⟩ | ⟨e, e', hr, h₁⟩ | ⟨ctl, c, hn', hr, h₁⟩ | ⟨c, hl, h₁⟩
    · exact .inl ⟨e, hu', .blockErr hb h₁⟩
    · exact .inr (.inl ⟨e, _, hr, .blockErr hb h₁⟩)
    · exact .inr (.inr (.inl ⟨ctl, c, hn', hr, .exit hb h₁ hn'⟩))
    · rw [List.append_assoc] at hl
      rcases assigns_run he ha hl with ⟨e, -, ha'⟩ | ⟨c₂, ha', hl'⟩
      · rw [assigns_seq ctx _ _ hs] at ha'
        obtain ⟨e', hu', ha''⟩ := (applyAssigns_perm hp hn).2 _ ha'
        exact .inl ⟨e', hu', .assignErr hb h₁ (trueEdges_isUncond hout hu) ha''⟩
      · rw [assigns_seq ctx _ _ hs] at ha'
        have ha'' := (applyAssigns_perm hp hn).1 _ ha'
        exact (RegCorr.matched h hhr hl').mono fun _ h' => .step hb h₁ (trueEdges_isUncond hout hu) ha'' h'

theorem BlockCorr.matched {b : Block} {cs : List SNode} :
    BlockCorr ctx tc b cs → ∀ {full : List SNode} {gb : Bool} {rest : List SNode} {cst : CSt}
      {r : Res (TCtl × CSt)}, HeadOk rest → STree.ListExec tc full gb {} (cs ++ rest) cst r →
      BOut (BlockExec ctx cst b) (fun c => STree.ListExec tc full gb {} rest c r) r
  | .state hs, full, gb, rest, cst, r, _, hx => by
    rcases hs.matched he hcons hw hx with ⟨e, hu, h⟩ | ⟨e, e', hr, h⟩ | ⟨c, hl, h⟩
    · exact .inl ⟨e, hu, .state h⟩
    · exact .inr (.inl ⟨e, _, hr, .state h⟩)
    · exact .inr (.inr (.inr ⟨c, hl, .state h⟩))
  | .loop (l := l) (body := body) hn hm hb, full, gb, rest, cst, r, hh, hx => by
    have hbody : ∀ c r', STree.ListExec tc body false {} body c r' → Out (RegionExec ctx c l.body) r' :=
      fun c r' hx' => (RegCorr.matched hb HeadOk.nil (ListExec.append_nil hx')).toOut.mono fun _ h => .start h
    refine node_list hh (fun res hres => ?_) hx
    obtain ⟨r', hl, rfl⟩ := loop_inv hn hres
    exact ((LoopExecT.matched he hm hbody hl).mono fun _ h => .loop h).scopeOut
  | .cond hc hb ht, full, gb, rest, cst, r, hh, hx => by
    subst hc
    refine (chain_step he (fun res hres => if_inv hres) (fun r' hx' => (RegCorr.matched hb HeadOk.nil (ListExec.append_nil hx')).toOut)
      (fun hx' => ElifCorr.matched ht hh hx') (fun c hx' => ElifCorr.skip ht hx') hh hx).mono
      fun _ h => .cond h
  | .brk, full, gb, rest, cst, r, hh, hx => by
    refine node_list hh (fun res hres => ?_) hx
    rw [leafStep_inv rfl hres]
    exact .inr (.inr ⟨{}, .brk, cst, rfl, rfl, .brk⟩)
  | .cont, full, gb, rest, cst, r, hh, hx => by
    refine node_list hh (fun res hres => ?_) hx
    rw [leafStep_inv rfl hres]
    exact .inr (.inr ⟨{}, .cont, cst, rfl, rfl, .cont⟩)
  | .ret, full, gb, rest, cst, r, hh, hx => by
    refine node_list hh (fun res hres => ?_) hx
    rw [leafStep_inv rfl hres]
    exact .inr (.inr ⟨{}, .ret, cst, rfl, rfl, .ret⟩)
  | .flat h, full, gb, rest, cst, r, hh, hx =>
    (RegCorr.matched h hh hx).mono fun _ h' => .region (.start h')
  | .named h, full, gb, rest, cst, r, hh, hx => by
    refine node_list hh (fun res hres => ?_) hx
    obtain ⟨r', hb', rfl⟩ := namedRegion_inv hres
    exact ((RegCorr.matched h HeadOk.nil (ListExec.append_nil hb')).toOut.mono fun _ h' => .region (.start h')).scopeOut

theorem ElifCorr.matched {brs : List (Option Expr × Region)} {cs : List SNode} :
    ElifCorr ctx tc brs cs → ∀ {full : List SNode} {gb : Bool} {rest : List SNode} {cst : CSt}
      {r : Res (TCtl × CSt)}, HeadOk rest → STree.ListExec tc full gb { chain := some false } (cs ++ rest) cst r →
      BOut (CondExec ctx cst brs) (fun c => STree.ListExec tc full gb {} rest c r) r
  | .nil, full, gb, rest, cst, r, hh, hx => .inr (.inr (.inr ⟨cst, listChainFree hx rfl hh, .nil⟩))
  | .elif hc hb ht, full, gb, rest, cst, r, hh, hx => by
    subst hc
    exact chain_step he (fun res hres => elif_inv hres) (fun r' hx' => (RegCorr.matched hb HeadOk.nil (ListExec.append_nil hx')).toOut)
      (fun hx' => ElifCorr.matched ht hh hx') (fun c hx' => ElifCorr.skip ht hx') hh hx
  | .els hb, full, gb, rest, cst, r, hh, hx => by
    refine node_list hh (fun res hres => ?_) hx
    obtain ⟨r', hb', rfl⟩ := else_inv hres
    exact ((RegCorr.matched hb HeadOk.nil (ListExec.append_nil hb')).toOut.mono fun _ h => .els (.start h)).scopeOut

end

end Matched

/-! ## Programs -/

/-- The contexts in which an SDFG and a schedule tree run. -/
def sctx (I : Interp) (g : SDFG) : Ctx := { I, g }

/-- The SDFG `g` as the tree `t` sees it (`Decisions.md#D34`, `#D68`): the
tree's descriptor repository, and per-call memory names of nested SDFGs
renamed by `π⁻¹` to the tree's names for them (`Ctx.scope`). -/
def flatCtx (I : Interp) (g : SDFG) (t : STree) (π : Equiv.Perm Name) : Ctx :=
  { I, g := { g with arrays := t.containers }, scope := π.symm }

/-- The tree's root list corresponds to the SDFG's top region, run in the
flattened context `flatCtx I g t π`, which renaming by `π` relates to the
SDFG's own context. -/
structure ProgCorrWith (I : Interp) (g : SDFG) (t : STree) (π : Equiv.Perm Name) : Prop where
  consts : t.toSDFG.withConstants = g.withConstants
  /-- Both programs start from the same memory (`Decisions.md#D68`). -/
  start : t.keepAtStart = g.keepAtStart
  reg : RegCorr (flatCtx I g t π) (TCtx.ofTree I t) g.top g.top.start t.children
  /-- The renaming fixes every name the SDFG's own states mention. -/
  names : Region.Names π (fun _ => True) (sctx I g) (flatCtx I g t π) g.top
  /-- Renamed names are uninitialized at the start. -/
  fresh : ∀ x, π x ≠ x → g.keepAtStart x = false
  /-- The renaming fixes the observable containers. -/
  iface : ∀ x ∈ g.interface, π x = x

/-- The tree's root list corresponds to the SDFG's top region. -/
def ProgCorr (I : Interp) (g : SDFG) (t : STree) : Prop := ∃ π, ProgCorrWith I g t π

/-- Renaming fresh names of a start memory without aliases is an exact
simulation. -/
theorem startMem_sim {g : SDFG} {π : Equiv.Perm Name} (hf : ∀ x, π x ≠ x → g.keepAtStart x = false) {m₀ : Mem}
    (hna : ∀ n, m₀.aliases n = none) :
    Mem.Sim π (fun _ => True) (fun _ => False) (g.startMem m₀) (g.startMem m₀) := by
  have hkeep : ∀ l : Loc, π l.name ≠ l.name → g.keepAtStart l.name = false ∧ g.keepAtStart (π l.name) = false :=
    fun l hx => ⟨hf _ hx, hf _ fun h => hx (π.injective h)⟩
  have hfix : ∀ l : Loc, π l.name = l.name → l.ren π = l := by
    intro l h; cases l; simp only [Loc.ren, Loc.mk.injEq, and_true]; exact h
  refine ⟨fun n _ => ?_, fun n b _ hb => ?_, fun l _ => .inr ?_, fun l _ => ?_⟩
  · simp [SDFG.startMem, Mem.restrict, hna]
  · simp [SDFG.startMem, Mem.restrict, hna] at hb
  · by_cases hx : π l.name = l.name
    · rw [hfix l hx]
    · simp only [SDFG.startMem, Mem.restrict, Loc.ren_name, (hkeep l hx).1, (hkeep l hx).2]; rfl
  · by_cases hx : π l.name = l.name
    · rw [hfix l hx]
    · simp only [SDFG.startMem, Mem.restrict, Loc.ren_name, (hkeep l hx).1, (hkeep l hx).2]; rfl

/-- An exact renaming fixing the interface is observationally equal there. -/
theorem obsEq_of_sim {iface : List Name} {π : Equiv.Perm Name} (hfix : ∀ x ∈ iface, π x = x) {m₂ m₁ : Mem}
    (h : Mem.Sim π (fun _ => True) (fun _ => False) m₂ m₁) : obsEq iface m₁ m₂ := by
  intro l hl
  have hl' : l.ren π = l := by
    cases l; simp only [Loc.ren, Loc.mk.injEq, and_true]; exact hfix _ hl
  rcases h.cells l trivial with ⟨hF, -⟩ | hc
  · exact hF.elim
  · exact ⟨by rw [← hc, hl'], by rw [← h.queues l trivial, hl']⟩

section Exec

variable {I : Interp} {g : SDFG} {t : STree} {π : Equiv.Perm Name} {ρ₀ : Valuation} {m₀ : Mem}

/-- A run of the flattened SDFG is a run of the SDFG up to the renaming. -/
theorem ProgCorrWith.unflat (hc : ProgCorrWith I g t π) (hw : ∀ op v, I.wcr op .undef v = .undef)
    (hna : ∀ n, m₀.aliases n = none) {r₁ : Res (Valuation × Mem × Log)}
    (hx : SDFGExec (flatCtx I g t π) ρ₀ (g.startMem m₀) g r₁) :
    (∃ e, e.isUB = true ∧ SDFG.Exec I g ρ₀ m₀ (.err e)) ∨
      (∃ e, r₁.map (fun x => x.2.1) = .err e ∧ SDFG.Exec I g ρ₀ m₀ (.err e)) ∨
      ∃ m, r₁.map (fun x => x.2.1) = .ok m ∧ ∃ m', SDFG.Exec I g ρ₀ m₀ (.ok m') ∧ obsEq g.interface m m' := by
  have hcs : CtxSim π (fun _ => True) (sctx I g) (flatCtx I g t π) := ⟨rfl, rfl⟩
  obtain ⟨r₂, hx₂, hr⟩ := SDFGExec.sim hcs hw hc.names hx _ (startMem_sim hc.fresh hna)
  cases r₁ with
  | err e =>
    obtain ⟨e', rfl, hu⟩ := ResSim.err_inv hr
    rcases hu with hu | rfl
    · exact .inl ⟨e', hu, _, hx₂, rfl⟩
    · exact .inr (.inl ⟨e', rfl, _, hx₂, rfl⟩)
  | ok x₁ =>
    rcases ResSim.ok_cases hr with ⟨e, rfl, hu⟩ | ⟨x₂, rfl, -, hm, -, -⟩
    · exact .inl ⟨e, hu, _, hx₂, rfl⟩
    · exact .inr (.inr ⟨_, rfl, x₂.2.1, ⟨_, hx₂, rfl⟩, obsEq_of_sim hc.iface hm⟩)

/-- A matched error is in particular some error. -/
theorem relax_err {α : Type} {A C : Prop} {r : Res α} {X : ErrReason → Prop}
    (h : A ∨ (∃ e, r = .err e ∧ X e) ∨ C) : A ∨ (∃ e e', r = .err e ∧ X e') ∨ C :=
  h.imp_right (Or.imp_left fun ⟨e, hr, hx⟩ => ⟨e, e, hr, hx⟩)

/-- **Programs**: every outcome of the tree is an outcome of the SDFG (an
error an error, possibly another one: error messages name containers, which
nested SDFGs may rename; a memory observationally equal), or the SDFG may
end in undefined behavior. -/

theorem ProgCorrWith.exec (hc : ProgCorrWith I g t π) (hw : ∀ op v, I.wcr op .undef v = .undef)
    (hna : ∀ n, m₀.aliases n = none) {r : Res Mem} (ht : STree.Exec I t ρ₀ m₀ r) :
    (∃ e, e.isUB = true ∧ SDFG.Exec I g ρ₀ m₀ (.err e)) ∨ (∃ e e', r = .err e ∧ SDFG.Exec I g ρ₀ m₀ (.err e')) ∨
      ∃ m, r = .ok m ∧ ∃ m', SDFG.Exec I g ρ₀ m₀ (.ok m') ∧ obsEq g.interface m m' := by
  obtain ⟨res, hx, rfl⟩ := ht
  have he : (TCtx.ofTree I t).ctx.eraseL = (flatCtx I g t π).eraseL := by
    simp [Ctx.eraseL, TCtx.ofTree, STree.toSDFG, flatCtx]
  have hstart : rootStart t ρ₀ m₀ = { ρ := g.withConstants ρ₀, mem := g.startMem m₀ } := by
    simp [rootStart, hc.consts, hc.start, SDFG.startMem]
  rw [hstart] at hx
  rcases (hc.reg.matched he rfl hw HeadOk.nil (ListExec.append_nil hx)).toOut with
    ⟨e, hu, hy⟩ | ⟨e, e', rfl, hy⟩ | ⟨ctl, c, rfl, hy⟩
  · rcases hc.unflat hw hna (.err (.start hy)) with h | ⟨e', he', h⟩ | ⟨m, hm, -⟩
    · exact .inl h
    · cases he'; exact .inl ⟨_, hu, h⟩
    · cases hm
  · rcases hc.unflat hw hna (.err (.start hy)) with h | ⟨e₁, he₁, h⟩ | ⟨m, hm, -⟩
    · exact .inl h
    · exact .inr (.inl ⟨e, e₁, rfl, h⟩)
    · cases hm
  · cases ctl with
    | normal => exact relax_err (hc.unflat hw hna (.run (.start hy) rfl))
    | ret => exact relax_err (hc.unflat hw hna (.run (.start hy) rfl))
    | brk => exact relax_err (hc.unflat hw hna (.escape (.start hy) rfl))
    | cont => exact relax_err (hc.unflat hw hna (.escape (.start hy) rfl))

end Exec

/-- **Refinement**: if the SDFG is supported, the tree refines it. -/
theorem ProgCorr.refines {g : SDFG} {t : STree} (hc : ∀ I, ProgCorr I g t) (hs : Supported g.prog) :
    Refines t.prog g.prog := by
  intro I ρ₀ m₀ hpre
  by_cases hmu : mayUB g.prog I ρ₀ m₀
  · exact .inl hmu
  · refine .inr fun r hr => ?_
    obtain ⟨π, hcπ⟩ := hc I
    rcases hcπ.exec hpre.1.undef_absorb hpre.2.2 hr with ⟨e, hu, hy⟩ | ⟨e, e', rfl, hy⟩ | ⟨m, rfl, m', hy, ho⟩
    · exact absurd ⟨e, hu, hy⟩ hmu
    · rcases hs I ρ₀ m₀ hpre with h | h
      · exact absurd h hmu
      · exact absurd hy (h e')
    · exact ⟨m, rfl, m', hy, ho⟩

end Corr

end Dace
