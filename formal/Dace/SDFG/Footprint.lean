-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Footprinted

/-!
# Static footprints of dataflow (SDFG)

A sound static over-approximation of the locations a **successful**
dataflow execution logs (`Log.reads`, `Log.writes`, `Log.wcrs`), for the
fragment in which logs carry no alias bindings and no queue effects.

* An `Access` is a leaf memlet as seen by the semantics: a container, its
  subset on the memlet (`none` = full range, `Decisions.md#D9`), the kind of
  access (read, plain write, WCR contribution with its operator), and the
  map scopes (`binders`, outermost first) between the access and the
  footprinted region, whose parameters are additional free variables.
  Element writes through pointer output connectors are confined to the
  memlet subset by the semantics (a write outside it is the undefined
  behavior `outsideSubset`, `Decisions.md#D67`), so a code node's data
  output is the access of its subset.
* `nodeFp ctx s F n` computes the accesses of node `n` (a map entry: of its
  whole scope, by fuel-bounded recursion over the scope tree), or `none`
  outside the fragment: consume scopes, nested SDFGs, view access nodes,
  reference `set` edges, stream sources or destinations of copies and code
  nodes, dynamic map inputs, or exhausted fuel.  A `some` result *is* the
  fragment check.
* `Access.Hit ctx m₀ ρ a l`: the location `l` is the element at linear offset
  `off` of `a`'s container, resolved through the aliases of `m₀`, where
  `off = ctx.elemOffset` of a point of `a`'s subset (as `Ctx.pointsOf`
  enumerates it under some memory), under `ρ` extended by values of the binders'
  parameters taken from their ranges (`Binders.Ext`).

Main results (`St.Step` = aliases unchanged and the log extended by covered
entries only):

* `NodeExec.fp_step`, `SeqExec.fp_step`, `OrderedExec.fp_step`,
  `IterExec.fp_logs`: the mutual induction over the dataflow relations;
* `SeqExec.fp_trace`: every node delta of a sequential trace (the deltas
  `State.hbFree` compares, `Decisions.md#D61`) is covered by that node's own
  footprint;
* `CompsExec.fp_logs`: the log of every state component is covered by the
  footprint of its top-level nodes.

Reads that are **not** logged (`Decisions.md#D33`, R1/R2 design F5: data in
dataflow expressions such as symbolic subset bounds and reduction
identities, stream peeks, alias lookups) are outside the scope of a
footprint; the fragment of `Dace.Equiv.Footprint` excludes data in subset
and range expressions (`Expr.readData`, container names as symbols).
-/

set_option autoImplicit false

namespace Dace

/-! ## Accesses -/

/-- Kind of a logged access. -/
inductive AccKind where
  | read
  | write
  | wcr (op : WcrOp)
  deriving DecidableEq, Repr, Inhabited

/-- The kind of a store through a memlet with WCR operator `w`. -/
def AccKind.ofStore : Option WcrOp → AccKind
  | none => .write
  | some op => .wcr op

/-- A static access of a leaf memlet (see the module documentation). -/
structure Access where
  data : Name
  sub : Option Subset
  kind : AccKind
  /-- Map scopes between the access and the footprinted region, outermost
  first: `(params, range)`. -/
  binders : List (List Sym × Range) := []
  deriving DecidableEq, Repr, Inhabited

/-- The access inside one more map scope. -/
def Access.bind (params : List Sym) (r : Range) (a : Access) : Access :=
  { a with binders := (params, r) :: a.binders }

/-- `ρ'` extends `ρ` by values of the binders' parameters taken from their
ranges (each range evaluated, as `Ctx.rangePoints` does, under the
valuation bound so far and some memory). -/
def Binders.Ext (ctx : Ctx) : Valuation → List (List Sym × Range) → Valuation → Prop
  | ρ, [], ρ' => ρ' = ρ
  | ρ, (ps, r) :: bs, ρ' =>
    ∃ m pts, ctx.rangePoints ρ m r = .ok pts ∧ ∃ i ∈ pts, Binders.Ext ctx (ρ.bindIdx ps i) bs ρ'

/-- `off` is an element offset of the access under the (fully bound)
valuation `ρ`. -/
def Access.HitsOff (ctx : Ctx) (ρ : Valuation) (a : Access) (off : Int) : Prop :=
  ∃ d, ctx.desc? a.data = some d ∧
    ∃ m pts, ctx.pointsOf ρ m a.data a.sub = .ok pts ∧ ∃ p ∈ pts, ctx.elemOffset ρ d p = some off

/-- The location `l` is hit by the access under `ρ` (aliases resolved in `m₀`). -/
def Access.Hit (ctx : Ctx) (m₀ : Mem) (ρ : Valuation) (a : Access) (l : Loc) : Prop :=
  ∃ ρ', Binders.Ext ctx ρ a.binders ρ' ∧ ∃ off, a.HitsOff ctx ρ' off ∧
    l = m₀.resolve (Loc.elem (ctx.cname a.data) off)

/-! ## Location predicates and covered logs -/

/-- Predicates on the locations of reads, plain writes and WCR
contributions (by operator). -/
structure LocPred where
  read : Loc → Prop
  write : Loc → Prop
  wcr : WcrOp → Loc → Prop

namespace LocPred

/-- The predicate of an access kind. -/
def ofKind (P : LocPred) : AccKind → Loc → Prop
  | .read => P.read
  | .write => P.write
  | .wcr op => P.wcr op

/-- Pointwise implication. -/
structure Le (P Q : LocPred) : Prop where
  read : ∀ l, P.read l → Q.read l
  write : ∀ l, P.write l → Q.write l
  wcr : ∀ op l, P.wcr op l → Q.wcr op l

theorem Le.refl (P : LocPred) : Le P P := ⟨fun _ h => h, fun _ h => h, fun _ _ h => h⟩

theorem Le.trans {P Q R : LocPred} (h₁ : Le P Q) (h₂ : Le Q R) : Le P R :=
  ⟨fun l h => h₂.read l (h₁.read l h), fun l h => h₂.write l (h₁.write l h),
    fun op l h => h₂.wcr op l (h₁.wcr op l h)⟩

/-- The locations hit by accesses of a list, by kind. -/
def ofAccesses (ctx : Ctx) (m₀ : Mem) (ρ : Valuation) (A : List Access) : LocPred where
  read l := ∃ a ∈ A, a.kind = .read ∧ a.Hit ctx m₀ ρ l
  write l := ∃ a ∈ A, a.kind = .write ∧ a.Hit ctx m₀ ρ l
  wcr op l := ∃ a ∈ A, a.kind = .wcr op ∧ a.Hit ctx m₀ ρ l

theorem ofAccesses_ofKind {ctx : Ctx} {m₀ : Mem} {ρ : Valuation} {A : List Access} {a : Access} (ha : a ∈ A)
    {l : Loc} (h : a.Hit ctx m₀ ρ l) : (ofAccesses ctx m₀ ρ A).ofKind a.kind l := by
  cases hk : a.kind with
  | read => exact ⟨a, ha, hk, h⟩
  | write => exact ⟨a, ha, hk, h⟩
  | wcr op => exact ⟨a, ha, hk, h⟩

theorem ofAccesses_mono {ctx : Ctx} {m₀ : Mem} {ρ : Valuation} {A B : List Access} (h : ∀ a ∈ A, a ∈ B) :
    Le (ofAccesses ctx m₀ ρ A) (ofAccesses ctx m₀ ρ B) :=
  ⟨fun _ ⟨a, ha, hk, hh⟩ => ⟨a, h a ha, hk, hh⟩, fun _ ⟨a, ha, hk, hh⟩ => ⟨a, h a ha, hk, hh⟩,
    fun _ _ ⟨a, ha, hk, hh⟩ => ⟨a, h a ha, hk, hh⟩⟩

end LocPred

/-- Every entry of the log is in the footprint `P`, and the log binds no
alias and touches no queue. -/
structure Log.Within (P : LocPred) (g : Log) : Prop where
  reads : ∀ l ∈ g.reads, P.read l
  writes : ∀ x ∈ g.writes, P.write x.1
  wcrs : ∀ x ∈ g.wcrs, P.wcr x.2.1 x.1
  aliasSets : g.aliasSets = []
  queues : g.queues = []

namespace Log.Within

variable {P Q : LocPred}

theorem empty : Log.empty.Within P := ⟨by simp [Log.empty], by simp [Log.empty], by simp [Log.empty], rfl, rfl⟩

theorem mono (hPQ : LocPred.Le P Q) {g : Log} (h : g.Within P) : g.Within Q :=
  ⟨fun l hl => hPQ.read l (h.reads l hl), fun x hx => hPQ.write _ (h.writes x hx),
    fun x hx => hPQ.wcr _ _ (h.wcrs x hx), h.aliasSets, h.queues⟩

theorem of_sub {g' g : Log} (hs : Log.Sub g' g) (h : g.Within P) : g'.Within P where
  reads l hl := h.reads l (hs.reads l hl)
  writes x hx := h.writes x (hs.writes x hx)
  wcrs x hx := h.wcrs x (hs.wcrs x hx)
  aliasSets := List.eq_nil_iff_forall_not_mem.2 fun x hx => by
    have := hs.aliasSets x hx
    rw [h.aliasSets] at this
    exact List.not_mem_nil this
  queues := List.eq_nil_iff_forall_not_mem.2 fun x hx => by
    have := hs.queues x hx
    rw [h.queues] at this
    exact List.not_mem_nil this

theorem unionAll {gs : List Log} (h : ∀ g ∈ gs, g.Within P) : (Log.unionAll gs).Within P where
  reads l hl := let ⟨g, hg, hl⟩ := List.mem_flatMap.1 hl; (h g hg).reads l hl
  writes x hx := let ⟨g, hg, hx⟩ := List.mem_flatMap.1 hx; (h g hg).writes x hx
  wcrs x hx := let ⟨g, hg, hx⟩ := List.mem_flatMap.1 hx; (h g hg).wcrs x hx
  aliasSets := by
    simp only [Log.unionAll, List.flatMap_eq_nil_iff]
    exact fun g hg => (h g hg).aliasSets
  queues := by
    simp only [Log.unionAll, List.flatMap_eq_nil_iff]
    exact fun g hg => (h g hg).queues

theorem seq {a b : Log} (ha : a.Within P) (hb : b.Within P) : (a.seq b).Within P :=
  of_sub (Log.seq_sub_unionAll a b) (unionAll (by
    intro g hg
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
    rcases hg with rfl | rfl
    · exact ha
    · exact hb))

theorem dropNames {g : Log} (h : g.Within P) (priv : List Name) : (g.dropNames priv).Within P :=
  of_sub (Log.dropNames_sub g priv) h

end Log.Within

/-! ## Steps of a sequential execution -/

/-- One part of a sequential execution, from `st` to `st'`: if the aliases
are those of `m₀`, they stay so, and a log within `P` stays within `P`. -/
def St.Step (m₀ : Mem) (P : LocPred) (st st' : St) : Prop :=
  st.mem.aliases = m₀.aliases → st'.mem.aliases = m₀.aliases ∧ (st.log.Within P → st'.log.Within P)

namespace St.Step

variable {m₀ : Mem} {P Q : LocPred}

theorem refl (st : St) : St.Step m₀ P st st := fun h => ⟨h, id⟩

theorem trans {a b c : St} (h₁ : St.Step m₀ P a b) (h₂ : St.Step m₀ P b c) : St.Step m₀ P a c := fun h =>
  let ⟨hb, wb⟩ := h₁ h
  let ⟨hc, wc⟩ := h₂ hb
  ⟨hc, wc ∘ wb⟩

/-- A step that only changes cells and values and logs `g`. -/
theorem of_logged {st st' : St} {g : Log} (hal : st'.mem.aliases = st.mem.aliases) (hlog : st'.log = st.log.seq g)
    (hg : st.mem.aliases = m₀.aliases → g.Within P) : St.Step m₀ P st st' := fun h =>
  ⟨hal.trans h, fun hw => hlog ▸ hw.seq (hg h)⟩

/-- A step that changes neither the aliases nor the log. -/
theorem of_eq {st st' : St} (hal : st'.mem.aliases = st.mem.aliases) (hlog : st'.log = st.log) :
    St.Step m₀ P st st' := fun h => ⟨hal.trans h, fun hw => hlog ▸ hw⟩

end St.Step

/-! ## `Res` helpers -/

namespace Res

theorem bind_eq_ok {α β : Type} {x : Res α} {f : α → Res β} {b : β} :
    (x >>= f) = .ok b ↔ ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x <;> simp [Bind.bind, Res.bind]

theorem ofOption_eq_ok {α : Type} {o : Option α} {e : ErrReason} {a : α} :
    Res.ofOption o e = .ok a ↔ o = some a := by
  cases o <;> simp [Res.ofOption]

theorem pure_eq_ok {α : Type} {a b : α} : (pure a : Res α) = .ok b ↔ a = b := by
  simp [Pure.pure]

end Res

/-- Steps compose through `foldlM` in `Res`. -/
theorem List.foldlM_step {α β : Type} {m₀ : Mem} {P : LocPred} (proj : β → St) (f : β → α → Res β) :
    ∀ (xs : List α) (b b' : β),
      (∀ x ∈ xs, ∀ c c', f c x = .ok c' → St.Step m₀ P (proj c) (proj c')) →
      xs.foldlM f b = .ok b' → St.Step m₀ P (proj b) (proj b')
  | [], b, b', _, h => by
    simp only [List.foldlM_nil, Res.pure_eq_ok] at h
    subst h
    exact St.Step.refl _
  | x :: xs, b, b', hf, h => by
    simp only [List.foldlM_cons] at h
    obtain ⟨c, hc, h⟩ := Res.bind_eq_ok.1 h
    exact (hf x (List.mem_cons_self ..) b c hc).trans
      (List.foldlM_step proj f xs c b' (fun y hy => hf y (List.mem_cons_of_mem _ hy)) h)

/-! ## Element accesses -/

/-- Every location the semantics computes for element `idx` of `n` (through a
memlet with `allow_oob` flag `oob`) satisfies `Q`, after resolution in `m₀`. -/
def Ctx.ElemOk (ctx : Ctx) (ρ : Valuation) (m₀ : Mem) (Q : Loc → Prop) (n : Name) (idx : Idx) (oob : Bool) : Prop :=
  ∀ d off, ctx.desc? n = some d → ctx.elemOffset ρ d idx = some off → ctx.checkBounds ρ n idx oob = .ok () →
    Q (m₀.resolve (Loc.elem (ctx.cname n) off))

/-- A successful location computation: an element of the container,
bounds-checked. -/
theorem Ctx.loc_ok {ctx : Ctx} {ρ : Valuation} {m : Mem} {n : Name} {idx : Idx} {oob : Bool} {l : Loc}
    (h : ctx.loc ρ m n idx oob = .ok l) :
    ∃ d off, ctx.desc? n = some d ∧ ctx.elemOffset ρ d idx = some off ∧ ctx.checkBounds ρ n idx oob = .ok () ∧
      l = Loc.elem (ctx.cname n) off := by
  unfold Ctx.loc at h
  obtain ⟨l', ha, h⟩ := Res.bind_eq_ok.1 h
  obtain ⟨u, hb, h⟩ := Res.bind_eq_ok.1 h
  rw [Res.pure_eq_ok] at h
  subst h
  cases u
  unfold Ctx.addr at ha
  split at ha
  · cases ha
  · rename_i d hd
    split at ha
    · cases ha
    · split at ha
      · rename_i off ho
        cases ha
        exact ⟨d, off, hd, ho, hb, rfl⟩
      · cases ha

namespace St

variable {ctx : Ctx} {ρ : Valuation} {m₀ : Mem} {P : LocPred}

theorem read_step {st st' : St} {n : Name} {idx : Idx} {oob : Bool} {v : Val}
    (h : st.read ctx ρ n idx oob = .ok (v, st')) (hq : ctx.ElemOk ρ m₀ P.read n idx oob) : St.Step m₀ P st st' := by
  unfold St.read at h
  obtain ⟨l, hl, h⟩ := Res.bind_eq_ok.1 h
  dsimp only at h
  split at h
  · cases h
  rw [Res.pure_eq_ok] at h
  cases h
  obtain ⟨d, off, hd, ho, hb, rfl⟩ := Ctx.loc_ok hl
  refine St.Step.of_logged rfl rfl fun hal => ⟨?_, by simp, by simp, rfl, rfl⟩
  intro l' hl'
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hl'
  subst hl'
  rw [Mem.resolve_congr hal]
  exact hq d off hd ho hb

theorem readMany_step {st st' : St} {n : Name} {pts : List Idx} {oob : Bool} {vs : List Val}
    (h : st.readMany ctx ρ n pts oob = .ok (vs, st')) (hq : ∀ p ∈ pts, ctx.ElemOk ρ m₀ P.read n p oob) :
    St.Step m₀ P st st' := by
  unfold St.readMany at h
  refine List.foldlM_step (β := List Val × St) Prod.snd _ pts ([], st) (vs, st') ?_ h
  intro p hp ⟨acc, c⟩ ⟨acc', c'⟩ hc
  obtain ⟨⟨v, c₁⟩, h₁, h₂⟩ := Res.bind_eq_ok.1 hc
  rw [Res.pure_eq_ok] at h₂
  cases h₂
  exact read_step h₁ (hq p hp)

theorem store_step {st st' : St} {n : Name} {idx : Idx} {oob : Bool} {w : Option WcrOp} {v : Val}
    (h : st.store ctx ρ n idx oob w v = .ok st') (hq : ctx.ElemOk ρ m₀ (P.ofKind (.ofStore w)) n idx oob) :
    St.Step m₀ P st st' := by
  cases w with
  | none =>
    unfold St.store St.write at h
    obtain ⟨l, hl, h⟩ := Res.bind_eq_ok.1 h
    rw [Res.pure_eq_ok] at h
    subst h
    obtain ⟨d, off, hd, ho, hb, rfl⟩ := Ctx.loc_ok hl
    refine St.Step.of_logged rfl rfl fun hal => ⟨by simp, ?_, by simp, rfl, rfl⟩
    intro x hx
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    subst hx
    rw [Mem.resolve_congr hal]
    exact hq d off hd ho hb
  | some op =>
    unfold St.store St.wcr at h
    obtain ⟨l, hl, h⟩ := Res.bind_eq_ok.1 h
    rw [Res.pure_eq_ok] at h
    subst h
    obtain ⟨d, off, hd, ho, hb, rfl⟩ := Ctx.loc_ok hl
    refine St.Step.of_logged rfl rfl fun hal => ⟨by simp, by simp, ?_, rfl, rfl⟩
    intro x hx
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    subst hx
    rw [Mem.resolve_congr hal]
    exact hq d off hd ho hb

end St

/-- Every point of the subset (enumerated under any memory) is covered. -/
def Ctx.SubOk (ctx : Ctx) (ρ : Valuation) (m₀ : Mem) (Q : Loc → Prop) (n : Name) (sub : Option Subset) (oob : Bool) :
    Prop :=
  ∀ m pts, ctx.pointsOf ρ m n sub = .ok pts → ∀ p ∈ pts, ctx.ElemOk ρ m₀ Q n p oob

section Hit

variable {ctx : Ctx} {ρ : Valuation} {m₀ : Mem} {A : List Access}

/-- A binder-free access covers the points of its subset. -/
theorem Access.subOk {a : Access} (ha : a ∈ A) (hb : a.binders = []) (oob : Bool) :
    ctx.SubOk ρ m₀ ((LocPred.ofAccesses ctx m₀ ρ A).ofKind a.kind) a.data a.sub oob := by
  intro m pts hp p hpm d off hd ho _
  exact LocPred.ofAccesses_ofKind ha ⟨ρ, by rw [hb]; rfl, off, ⟨d, hd, m, pts, hp, p, hpm, ho⟩, rfl⟩

end Hit

/-! ## Static footprints -/

/-- Concatenate the results of `f`, `none` if any is `none`. -/
def optFlatMap {α β : Type} (f : α → Option (List β)) : List α → Option (List β)
  | [] => some []
  | x :: xs =>
    match f x, optFlatMap f xs with
    | some a, some b => some (a ++ b)
    | _, _ => none

theorem optFlatMap_mem {α β : Type} {f : α → Option (List β)} :
    ∀ {xs : List α} {B : List β}, optFlatMap f xs = some B → ∀ x ∈ xs, ∃ A, f x = some A ∧ ∀ b ∈ A, b ∈ B
  | [], _, _, x, hx => absurd hx List.not_mem_nil
  | y :: ys, B, h, x, hx => by
    simp only [optFlatMap] at h
    split at h
    · rename_i a b ha hb
      cases h
      rcases List.mem_cons.1 hx with rfl | hx
      · exact ⟨a, ha, fun c hc => List.mem_append_left _ hc⟩
      · obtain ⟨A, hA, hs⟩ := optFlatMap_mem hb x hx
        exact ⟨A, hA, fun c hc => List.mem_append_right _ (hs c hc)⟩
    · cases h

theorem mem_optFlatMap {α β : Type} {f : α → Option (List β)} :
    ∀ {xs : List α} {B : List β}, optFlatMap f xs = some B → ∀ b ∈ B, ∃ x ∈ xs, ∃ A, f x = some A ∧ b ∈ A
  | [], B, h, b, hb => by
    simp only [optFlatMap, Option.some.injEq] at h
    subst h
    exact absurd hb List.not_mem_nil
  | y :: ys, B, h, b, hb => by
    simp only [optFlatMap] at h
    split at h
    · rename_i a c ha hc
      cases h
      rcases List.mem_append.1 hb with hb | hb
      · exact ⟨y, List.mem_cons_self .., a, ha, hb⟩
      · obtain ⟨x, hx, A, hA, hbA⟩ := mem_optFlatMap hc b hb
        exact ⟨x, List.mem_cons_of_mem _ hx, A, hA, hbA⟩
    · cases h

/-- Accesses of the copy along the in-edge `e` of an access or exit node
(`execCopiesInto`): a read of the source subset and a store (plain or WCR)
to the destination subset.  `none` for reference `set` edges and copies
from or to streams. -/
def copyFp (ctx : Ctx) (s : State) (e : DEdge) : Option (List Access) :=
  match e.data with
  | .dep => some []
  | .memlet m =>
    if ctx.isViewDef s e || s.isRouting e then some []
    else
      match refSetTarget? ctx s e with
      | some _ => none
      | none =>
        match s.srcEndpoint e with
        | .data sd ssub =>
          match s.dstEndpoint e with
          | .data dd dsub =>
            if ctx.isStream sd || ctx.isStream dd then none
            else some [{ data := sd, sub := ssub, kind := .read }, { data := dd, sub := dsub, kind := .ofStore m.wcr }]
          | _ => some []
        | _ => some []

/-- Accesses of the copies into a node. -/
def copiesFp (ctx : Ctx) (s : State) (n : DNode) : Option (List Access) :=
  optFlatMap (copyFp ctx s) (s.inEdges n.id)

/-- Accesses of an input connector of a code node: a read of the source
subset; `none` for a stream source. -/
def codeInFp (ctx : Ctx) (s : State) (e : DEdge) : Option (List Access) :=
  match e.data with
  | .dep => some []
  | .memlet _ =>
    match s.srcEndpoint e with
    | .data d sub => if ctx.isStream d then none else some [{ data := d, sub, kind := .read }]
    | _ => some []

/-- Accesses of an output edge of a code node: a store to the destination
subset (scalar outputs and element writes through a pointer connector,
`Decisions.md#D67`); `none` for a stream destination. -/
def codeOutFp (ctx : Ctx) (s : State) (e : DEdge) : Option (List Access) :=
  match e.data with
  | .dep => some []
  | .memlet m =>
    match s.dstEndpoint e with
    | .data d sub =>
      if ctx.isStream d then none
      else some [{ data := d, sub, kind := .ofStore m.wcr }]
    | _ => some []

/-- Accesses of a code node (tasklet, library node without a direct model). -/
def codeFp (ctx : Ctx) (s : State) (n : DNode) : Option (List Access) :=
  match optFlatMap (codeInFp ctx s) (s.inEdges n.id), optFlatMap (codeOutFp ctx s) (s.outEdges n.id) with
  | some a, some b => some (a ++ b)
  | _, _ => none

/-- Accesses of a `Reduce` node (`execReduce`): a read of the input subset,
plain/WCR stores and WCR contributions to the output subset. -/
def reduceFp (s : State) (n : DNode) (props : ReduceProps) : Option (List Access) :=
  let ins := (s.inEdges n.id).filter fun e => e.data.memlet?.isSome
  let outs := (s.outEdges n.id).filter fun e => e.data.memlet?.isSome
  match ins, outs with
  | [ie], [oe] =>
    match s.srcEndpoint ie, s.dstEndpoint oe, oe.data.memlet? with
    | .data sd ssub, .data dd dsub, some om =>
      some [{ data := sd, sub := ssub, kind := .read }, { data := dd, sub := dsub, kind := .ofStore om.wcr },
            { data := dd, sub := dsub, kind := .wcr (om.wcr.getD props.wcr) }]
    | _, _, _ => some []
  | _, _ => some []

/-- Accesses of a node executed by `execSimple`; `none` for view access
nodes (bound by the view prologue) and for nodes that are not simple. -/
def simpleFp (ctx : Ctx) (s : State) (n : DNode) : Option (List Access) :=
  match n.kind with
  | .access d => if ctx.isView d then none else copiesFp ctx s n
  | .mapExit | .consumeExit => copiesFp ctx s n
  | .tasklet _ => codeFp ctx s n
  | .library l =>
    match l.reduce with
    | some props => reduceFp s n props
    | none => codeFp ctx s n
  | .mapEntry _ | .consumeEntry _ | .nested _ _ _ => none

/-- Static footprint of node `n` with recursion depth `fuel`: its own
accesses, or for a map entry the accesses of its direct children (the
scope body, including the exit node), each inside one more binder.  `none`
outside the fragment (see the module documentation) or when the fuel runs
out. -/
def nodeFp (ctx : Ctx) (s : State) : Nat → DNode → Option (List Access)
  | 0, _ => none
  | fuel + 1, n =>
    match n.kind with
    | .mapEntry mp =>
      if (s.inEdges n.id).any (s.isDynamicInput ·) then none
      else
        (optFlatMap (fun id => match s.node? id with
            | some c => nodeFp ctx s fuel c
            | none => some []) (s.directChildren (some n.id))).map (·.map (Access.bind mp.params mp.range))
    | _ => simpleFp ctx s n

/-- Static footprint of the nodes `ids` (unknown ids contribute nothing,
their execution fails). -/
def scopeFp (ctx : Ctx) (s : State) (fuel : Nat) (ids : List NodeId) : Option (List Access) :=
  optFlatMap (fun id => match s.node? id with
    | some c => nodeFp ctx s fuel c
    | none => some []) ids

theorem nodeFp_mapEntry {ctx : Ctx} {s : State} {fuel : Nat} {n : DNode} {mp : MapNode} (hk : n.kind = .mapEntry mp)
    {A : List Access} (h : nodeFp ctx s fuel n = some A) :
    ∃ f B, fuel = f + 1 ∧ (s.inEdges n.id).any (s.isDynamicInput ·) = false ∧
      scopeFp ctx s f (s.directChildren (some n.id)) = some B ∧ A = B.map (Access.bind mp.params mp.range) := by
  cases fuel with
  | zero => simp [nodeFp] at h
  | succ f =>
    simp only [nodeFp, hk] at h
    split at h
    · cases h
    · rename_i hd
      obtain ⟨B, hB, rfl⟩ := Option.map_eq_some_iff.1 h
      exact ⟨f, B, rfl, by simpa using hd, hB, rfl⟩

theorem nodeFp_simple {ctx : Ctx} {s : State} {fuel : Nat} {n : DNode} (hk : n.isSimple = true)
    {A : List Access} (h : nodeFp ctx s fuel n = some A) : simpleFp ctx s n = some A := by
  cases fuel with
  | zero => simp [nodeFp] at h
  | succ f =>
    simp only [nodeFp] at h
    split at h
    · rename_i mp hm
      simp [DNode.isSimple, hm] at hk
    · exact h

/-! ## Leaf functions -/

section Leaf

variable {ctx : Ctx} {ρ : Valuation} {m₀ : Mem} {P : LocPred}

theorem St.Step.of_pure {st st' : St} (h : (pure st : Res St) = .ok st') : St.Step m₀ P st st' := by
  rw [Res.pure_eq_ok] at h
  subst h
  exact St.Step.refl _

theorem copyData_step {st st' : St} {sd dd : Name} {ssub dsub : Option Subset} {w : Option WcrOp} {oob : Bool}
    (h : copyData ctx ρ st sd ssub dd dsub w oob = .ok st')
    (hs : ctx.isStream sd = false) (hd : ctx.isStream dd = false)
    (hr : ctx.SubOk ρ m₀ P.read sd ssub oob) (hw : ctx.SubOk ρ m₀ (P.ofKind (.ofStore w)) dd dsub oob) :
    St.Step m₀ P st st' := by
  unfold copyData at h
  rw [hs, hd] at h
  obtain ⟨spts, hsp, h⟩ := Res.bind_eq_ok.1 h
  obtain ⟨dpts, hdp, h⟩ := Res.bind_eq_ok.1 h
  refine List.foldlM_step id _ (spts.zip dpts) st st' ?_ h
  intro ⟨p, q⟩ hpq c c' hc
  obtain ⟨⟨v, c₁⟩, h₁, h₂⟩ := Res.bind_eq_ok.1 hc
  exact (St.read_step h₁ (hr _ _ hsp p (List.of_mem_zip hpq).1)).trans
    (St.store_step h₂ (hw _ _ hdp q (List.of_mem_zip hpq).2))

theorem execCopiesInto_step {s : State} {n : DNode} {st st' : St} (h : execCopiesInto ctx ρ s n st = .ok st')
    {A : List Access} (hfp : copiesFp ctx s n = some A) :
    St.Step m₀ (LocPred.ofAccesses ctx m₀ ρ A) st st' := by
  unfold execCopiesInto at h
  refine List.foldlM_step id _ _ st st' ?_ h
  intro e he c c' hc
  obtain ⟨Ae, hAe, hsub⟩ := optFlatMap_mem hfp e he
  unfold copyFp at hAe
  split at hc
  · exact St.Step.of_pure hc
  · rename_i m hm
    rw [hm] at hAe
    simp only at hAe
    split at hc
    · exact St.Step.of_pure hc
    · rename_i hvr
      rw [ite_eq_right hvr] at hAe
      split at hc
      · rename_i d hd
        rw [hd] at hAe
        cases hAe
      · rename_i hd
        rw [hd] at hAe
        simp only at hAe
        cases hsrc : s.srcEndpoint e with
        | data sd ssub =>
          simp only [hsrc] at hc hAe
          unfold execCopy at hc
          simp only [hm, hsrc] at hc
          cases hdst : s.dstEndpoint e with
          | data dd dsub =>
            simp only [hdst] at hc hAe
            split at hAe
            · cases hAe
            · rename_i hns
              simp only [Bool.or_eq_true, not_or, Bool.not_eq_true] at hns
              cases hAe
              exact copyData_step hc hns.1 hns.2
                (Access.subOk (a := { data := sd, sub := ssub, kind := .read }) (hsub _ (by simp)) rfl _)
                (Access.subOk (a := { data := dd, sub := dsub, kind := .ofStore m.wcr }) (hsub _ (by simp)) rfl _)
          | code k => simp only [hdst] at hc; cases hc
          | invalid => simp only [hdst] at hc; cases hc
        | code k =>
          simp only [hsrc] at hc
          exact St.Step.of_pure hc
        | invalid =>
          simp only [hsrc] at hc
          exact St.Step.of_pure hc

/-- The input edges of a code node are covered (and none reads a stream). -/
def CodeInOk (ctx : Ctx) (ρ : Valuation) (m₀ : Mem) (P : LocPred) (s : State) (n : DNode) : Prop :=
  ∀ e ∈ s.inEdges n.id, ∀ m, e.data = .memlet m → ∀ d sub, s.srcEndpoint e = .data d sub →
    ctx.isStream d = false ∧ ctx.SubOk ρ m₀ P.read d sub m.allowOob

/-- The output edges of a code node are covered (and none writes a stream). -/
def CodeOutOk (ctx : Ctx) (ρ : Valuation) (m₀ : Mem) (P : LocPred) (s : State) (n : DNode) : Prop :=
  ∀ e ∈ s.outEdges n.id, ∀ m, e.data = .memlet m → ∀ d sub, s.dstEndpoint e = .data d sub →
    ctx.isStream d = false ∧ ctx.SubOk ρ m₀ (P.ofKind (.ofStore m.wcr)) d sub m.allowOob

theorem DEdge.allowOob_of {e : DEdge} {m : Memlet} (h : e.data = .memlet m) : e.allowOob = m.allowOob := by
  simp [DEdge.allowOob, h, EdgeData.memlet?]

theorem bindInData_step {d : Name} {sub : Option Subset} {oob : Bool} {st st' : St} {ci : ConnIn}
    (h : bindInData ctx ρ d sub oob st = .ok (ci, st')) (hr : ctx.SubOk ρ m₀ P.read d sub oob) :
    St.Step m₀ P st st' := by
  unfold bindInData at h
  split at h
  · obtain ⟨p, _, h⟩ := Res.bind_eq_ok.1 h
    obtain ⟨q, _, h⟩ := Res.bind_eq_ok.1 h
    rw [Res.pure_eq_ok] at h
    cases h
    exact St.Step.refl _
  · obtain ⟨pts, hp, h⟩ := Res.bind_eq_ok.1 h
    split at h
    · rename_i p
      obtain ⟨⟨v, c⟩, h₁, h⟩ := Res.bind_eq_ok.1 h
      rw [Res.pure_eq_ok] at h
      cases h
      exact St.read_step h₁ (hr _ _ hp p (by simp))
    · obtain ⟨lo, _, h⟩ := Res.bind_eq_ok.1 h
      obtain ⟨⟨vs, c⟩, h₁, h⟩ := Res.bind_eq_ok.1 h
      rw [Res.pure_eq_ok] at h
      cases h
      exact St.readMany_step h₁ (hr _ _ hp)

theorem bindInputs_step {s : State} {n : DNode} {st st' : St} {ins : Conn → Option ConnIn}
    (h : bindInputs ctx ρ s n st = .ok (ins, st')) (hin : CodeInOk ctx ρ m₀ P s n) : St.Step m₀ P st st' := by
  unfold bindInputs at h
  obtain ⟨⟨bs, c⟩, hf, h⟩ := Res.bind_eq_ok.1 h
  rw [Res.pure_eq_ok] at h
  cases h
  refine List.foldlM_step (β := List (Conn × ConnIn) × St) Prod.snd _ _ ([], st) (bs, st') ?_ hf
  intro e he ⟨acc, c⟩ ⟨acc', c'⟩ hc
  dsimp only at hc
  split at hc
  · rw [Res.pure_eq_ok] at hc
    cases hc
    exact St.Step.refl _
  · rename_i m hm
    obtain ⟨⟨ci, c₁⟩, h₁, h₂⟩ := Res.bind_eq_ok.1 hc
    have hstep : St.Step m₀ P c c₁ := by
      unfold bindIn at h₁
      split at h₁
      · rw [Res.ok_inj] at h₁
        cases h₁
        exact St.Step.refl _
      · rename_i d sub hsrc
        obtain ⟨⟨cin, c₂⟩, h₃, h₄⟩ := Res.bind_eq_ok.1 h₁
        rw [Res.pure_eq_ok] at h₄
        cases h₄
        rw [DEdge.allowOob_of hm] at h₃
        exact bindInData_step h₃ (hin e he m hm d sub hsrc).2
      · cases h₁
    refine hstep.trans ?_
    split at h₂ <;> (rw [Res.pure_eq_ok] at h₂; cases h₂; exact St.Step.refl _)

theorem mem_outEdges_of_outEdgesOn {s : State} {n : DNode} {c : Conn} {e : DEdge} (h : e ∈ s.outEdgesOn n c) :
    e ∈ s.outEdges n.id ∧ ∃ m, e.data = .memlet m := by
  unfold State.outEdgesOn at h
  obtain ⟨he, hp⟩ := List.mem_filter.1 h
  refine ⟨he, ?_⟩
  simp only [Bool.and_eq_true] at hp
  cases hd : e.data with
  | memlet m => exact ⟨m, rfl⟩
  | dep => rw [hd] at hp; simp at hp

theorem applyScalar_step {s : State} {n : DNode} {st st' : St} {c : Conn} {v : Val}
    (h : applyScalar ctx ρ s n st c v = .ok st') (hout : CodeOutOk ctx ρ m₀ P s n) : St.Step m₀ P st st' := by
  unfold applyScalar at h
  refine List.foldlM_step id _ _ st st' ?_ h
  intro e he c₁ c₂ hc
  obtain ⟨heo, -⟩ := mem_outEdges_of_outEdgesOn he
  split at hc
  · exact St.Step.of_pure hc
  · rename_i m hm
    split at hc
    · rw [Res.pure_eq_ok] at hc
      subst hc
      exact St.Step.of_eq rfl rfl
    · rename_i d sub hdst
      obtain ⟨hns, hsub⟩ := hout e heo m hm d sub hdst
      unfold storeScalar at hc
      rw [hns] at hc
      simp only [Bool.false_eq_true, ↓reduceIte] at hc
      obtain ⟨pts, hp, hc⟩ := Res.bind_eq_ok.1 hc
      split at hc
      · rename_i p
        exact St.store_step hc (hsub _ _ hp p (by simp))
      · cases hc
      · cases hc
    · cases hc

theorem applyArrayWrite_step {s : State} {n : DNode} {st st' : St} {c : Conn} {rel : Idx} {v : Val}
    (h : applyArrayWrite ctx ρ s n st c rel v = .ok st') (hout : CodeOutOk ctx ρ m₀ P s n) :
    St.Step m₀ P st st' := by
  unfold applyArrayWrite at h
  refine List.foldlM_step id _ _ st st' ?_ h
  intro e he c₁ c₂ hc
  obtain ⟨heo, -⟩ := mem_outEdges_of_outEdgesOn he
  split at hc
  · exact St.Step.of_pure hc
  · rename_i m hm
    split at hc
    · rename_i d sub hdst
      obtain ⟨hns, hsub⟩ := hout e heo m hm d sub hdst
      unfold storeArrayWrite at hc
      rw [hns] at hc
      simp only [Bool.false_eq_true, ↓reduceIte] at hc
      obtain ⟨lo, _, hc⟩ := Res.bind_eq_ok.1 hc
      split at hc
      · obtain ⟨pts, hp, hc⟩ := Res.bind_eq_ok.1 hc
        split at hc
        · rename_i hcont
          exact St.store_step hc (hsub _ _ hp _ (by simpa using hcont))
        · cases hc
      · cases hc
    · cases hc

theorem applyPush_step {s : State} {n : DNode} {st st' : St} {c : Conn} {vs : List Val}
    (h : applyPush ctx ρ s n st c vs = .ok st') (hout : CodeOutOk ctx ρ m₀ P s n) : St.Step m₀ P st st' := by
  unfold applyPush at h
  refine List.foldlM_step id _ _ st st' ?_ h
  intro e he c₁ c₂ hc
  obtain ⟨heo, m, hm⟩ := mem_outEdges_of_outEdgesOn he
  split at hc
  · rename_i d sub hdst
    have hns := (hout e heo m hm d sub hdst).1
    unfold pushTo at hc
    rw [hns] at hc
    cases hc
  · cases hc

theorem applyPop_step {s : State} {n : DNode} {st st' : St} {c : Conn} {k : Nat}
    (h : applyPop ctx ρ s n st c k = .ok st') (hin : CodeInOk ctx ρ m₀ P s n) : St.Step m₀ P st st' := by
  unfold applyPop at h
  split at h
  · cases h
  · rename_i e he
    have hmem : e ∈ s.inEdges n.id := List.mem_of_find?_eq_some he
    have hp := List.find?_some he
    simp only [Bool.and_eq_true] at hp
    obtain ⟨m, hm⟩ : ∃ m, e.data = .memlet m := by
      cases hd : e.data with
      | memlet m => exact ⟨m, rfl⟩
      | dep => rw [hd] at hp; simp at hp
    split at h
    · rename_i d sub hsrc
      have hns := (hin e hmem m hm d sub hsrc).1
      unfold popFrom at h
      rw [hns] at h
      cases h
    · cases h

theorem applyOut_step {s : State} {n : DNode} {st st' : St} {out : TaskletOut}
    (h : applyOut ctx ρ s n st out = .ok st') (hin : CodeInOk ctx ρ m₀ P s n) (hout : CodeOutOk ctx ρ m₀ P s n) :
    St.Step m₀ P st st' := by
  unfold applyOut at h
  obtain ⟨c₁, h₁, h⟩ := Res.bind_eq_ok.1 h
  obtain ⟨c₂, h₂, h⟩ := Res.bind_eq_ok.1 h
  obtain ⟨c₃, h₃, h⟩ := Res.bind_eq_ok.1 h
  refine (List.foldlM_step id _ _ st c₁ ?_ h₁).trans ((List.foldlM_step id _ _ c₁ c₂ ?_ h₂).trans
    ((List.foldlM_step id _ _ c₂ c₃ ?_ h₃).trans (List.foldlM_step id _ _ c₃ st' ?_ h)))
  · exact fun x _ a b hx => applyPop_step hx hin
  · exact fun x _ a b hx => applyScalar_step hx hout
  · exact fun x _ a b hx => applyArrayWrite_step hx hout
  · exact fun x _ a b hx => applyPush_step hx hout

theorem execCode_step {s : State} {n : DNode} {st st' : St} {run : (Conn → Option ConnIn) → Option TaskletOut}
    {err : ErrReason} (h : execCode ctx ρ s n st run err = .ok st') (hin : CodeInOk ctx ρ m₀ P s n)
    (hout : CodeOutOk ctx ρ m₀ P s n) : St.Step m₀ P st st' := by
  unfold execCode at h
  split at h
  · cases h
  · obtain ⟨⟨ins, c⟩, h₁, h⟩ := Res.bind_eq_ok.1 h
    refine (bindInputs_step h₁ hin).trans ?_
    dsimp only at h
    cases hr : run ins with
    | none => rw [hr] at h; cases h
    | some out =>
      rw [hr] at h
      exact applyOut_step h hin hout

theorem reduceData_step {st st' : St} {props : ReduceProps} {sd dd : Name} {ssub dsub : Option Subset}
    {outWcr : Option WcrOp} {soob doob : Bool}
    (h : reduceData ctx ρ st props sd ssub dd dsub outWcr soob doob = .ok st')
    (hr : ctx.SubOk ρ m₀ P.read sd ssub soob) (hw : ctx.SubOk ρ m₀ (P.ofKind (.ofStore outWcr)) dd dsub doob)
    (hc : ctx.SubOk ρ m₀ (P.wcr (outWcr.getD props.wcr)) dd dsub doob) : St.Step m₀ P st st' := by
  unfold reduceData at h
  split at h
  · cases h
  · obtain ⟨spts, hsp, h⟩ := Res.bind_eq_ok.1 h
    obtain ⟨dpts, hdp, h⟩ := Res.bind_eq_ok.1 h
    dsimp only at h
    split at h
    · cases h
    · obtain ⟨⟨vals, c₁⟩, h₁, h⟩ := Res.bind_eq_ok.1 h
      refine (St.readMany_step h₁ (hr _ _ hsp)).trans ?_
      dsimp only at h
      refine List.foldlM_step id _ _ c₁ st' ?_ h
      intro ⟨g, q⟩ hgq a b hab
      have hq : q ∈ dpts := (List.of_mem_zip hgq).2
      dsimp only at hab
      split at hab
      · obtain ⟨i, _, hab⟩ := Res.bind_eq_ok.1 hab
        exact St.store_step hab (hw _ _ hdp q hq)
      · split at hab
        · exact St.Step.of_pure hab
        · exact St.store_step (w := some _) hab (hc _ _ hdp q hq)

/-- The input edges of a code node are covered by its static footprint. -/
theorem codeInOk_of {s : State} {n : DNode} {A : List Access} {B : List Access}
    (h : optFlatMap (codeInFp ctx s) (s.inEdges n.id) = some B) (hB : ∀ a ∈ B, a ∈ A) :
    CodeInOk ctx ρ m₀ (LocPred.ofAccesses ctx m₀ ρ A) s n := by
  intro e he m hm d sub hsrc
  obtain ⟨Ae, hAe, hsub⟩ := optFlatMap_mem h e he
  unfold codeInFp at hAe
  simp only [hm, hsrc] at hAe
  split at hAe
  · cases hAe
  · rename_i hns
    cases hAe
    refine ⟨by simpa using hns, ?_⟩
    exact Access.subOk (a := { data := d, sub, kind := .read }) (hB _ (hsub _ (by simp))) rfl _

/-- The output edges of a code node are covered by its static footprint. -/
theorem codeOutOk_of {s : State} {n : DNode} {A : List Access} {B : List Access}
    (h : optFlatMap (codeOutFp ctx s) (s.outEdges n.id) = some B) (hB : ∀ a ∈ B, a ∈ A) :
    CodeOutOk ctx ρ m₀ (LocPred.ofAccesses ctx m₀ ρ A) s n := by
  intro e he m hm d sub hdst
  obtain ⟨Ae, hAe, hsub⟩ := optFlatMap_mem h e he
  unfold codeOutFp at hAe
  simp only [hm, hdst] at hAe
  split at hAe
  · cases hAe
  · rename_i hns
    cases hAe
    exact ⟨by simpa using hns,
      Access.subOk (a := { data := d, sub, kind := .ofStore m.wcr }) (hB _ (hsub _ (by simp))) rfl _⟩

theorem execCode_step_of_fp {s : State} {n : DNode} {st st' : St}
    {run : (Conn → Option ConnIn) → Option TaskletOut} {err : ErrReason} (h : execCode ctx ρ s n st run err = .ok st')
    {A : List Access} (hfp : codeFp ctx s n = some A) : St.Step m₀ (LocPred.ofAccesses ctx m₀ ρ A) st st' := by
  unfold codeFp at hfp
  split at hfp
  · rename_i a b ha hb
    cases hfp
    exact execCode_step h (codeInOk_of ha fun x hx => List.mem_append_left _ hx)
      (codeOutOk_of hb fun x hx => List.mem_append_right _ hx)
  · cases hfp

theorem execReduce_step {s : State} {n : DNode} {props : ReduceProps} {st st' : St}
    (h : execReduce ctx ρ s n props st = .ok st') {A : List Access} (hfp : reduceFp s n props = some A) :
    St.Step m₀ (LocPred.ofAccesses ctx m₀ ρ A) st st' := by
  unfold execReduce at h
  unfold reduceFp at hfp
  dsimp only at h hfp
  split at h
  · rename_i ie oe hins houts
    have h₁ : (s.inEdges n.id).filter (fun e => e.data.memlet?.isSome) = [ie] := by
      rw [← hins]
      exact List.filter_congr fun x _ => by cases x.data <;> rfl
    have h₂ : (s.outEdges n.id).filter (fun e => e.data.memlet?.isSome) = [oe] := by
      rw [← houts]
      exact List.filter_congr fun x _ => by cases x.data <;> rfl
    rw [h₁, h₂] at hfp
    dsimp only at hfp
    split at h
    · rename_i sd ssub dd dsub om hsrc hdst hom
      rw [hsrc, hdst, hom] at hfp
      dsimp only at hfp
      cases hfp
      exact reduceData_step h
        (Access.subOk (a := { data := sd, sub := ssub, kind := .read }) (by simp) rfl _)
        (Access.subOk (a := { data := dd, sub := dsub, kind := .ofStore om.wcr }) (by simp) rfl _)
        (Access.subOk (a := { data := dd, sub := dsub, kind := .wcr (om.wcr.getD props.wcr) }) (by simp) rfl _)
    · cases h
  · cases h

theorem execSimple_step {s : State} {n : DNode} {st st' : St} (h : execSimple ctx ρ s n st = .ok st')
    {A : List Access} (hfp : simpleFp ctx s n = some A) : St.Step m₀ (LocPred.ofAccesses ctx m₀ ρ A) st st' := by
  unfold execSimple at h
  unfold simpleFp at hfp
  cases hk : n.kind with
  | access d =>
    rw [hk] at h hfp
    dsimp only at h hfp
    split at hfp
    · cases hfp
    · exact execCopiesInto_step h hfp
  | mapExit =>
    rw [hk] at h hfp
    exact execCopiesInto_step h hfp
  | consumeExit =>
    rw [hk] at h hfp
    exact execCopiesInto_step h hfp
  | tasklet t =>
    rw [hk] at h hfp
    exact execCode_step_of_fp h hfp
  | library l =>
    rw [hk] at h hfp
    dsimp only at h hfp
    cases hr : l.reduce with
    | some props =>
      rw [hr] at h hfp
      exact execReduce_step h hfp
    | none =>
      rw [hr] at h hfp
      exact execCode_step_of_fp h hfp
  | mapEntry mp => rw [hk] at h; cases h
  | consumeEntry c => rw [hk] at h; cases h
  | nested g mapping ni => rw [hk] at h; cases h

end Leaf

/-! ## Scopes -/

/-- The nodes `ids` are in the fragment at fuel `F` and their footprints are
contained in `A`. -/
def FpIncl (ctx : Ctx) (s : State) (F : Nat) (ids : List NodeId) (A : List Access) : Prop :=
  ∀ id ∈ ids, ∀ n, s.node? id = some n → ∃ An, nodeFp ctx s F n = some An ∧ ∀ a ∈ An, a ∈ A

theorem FpIncl.of_sub {ctx : Ctx} {s : State} {F : Nat} {ids ids' : List NodeId} {A : List Access}
    (h : FpIncl ctx s F ids A) (hs : ∀ id ∈ ids', id ∈ ids) : FpIncl ctx s F ids' A :=
  fun id hid => h id (hs id hid)

theorem scopeFp_incl {ctx : Ctx} {s : State} {F : Nat} {ids : List NodeId} {A : List Access}
    (h : scopeFp ctx s F ids = some A) : FpIncl ctx s F ids A := by
  intro id hid n hn
  obtain ⟨An, hAn, hs⟩ := optFlatMap_mem h id hid
  simp only [hn] at hAn
  exact ⟨An, hAn, hs⟩

theorem nodeFp_consumeEntry {ctx : Ctx} {s : State} {F : Nat} {n : DNode} {c : ConsumeNode}
    (hk : n.kind = .consumeEntry c) : nodeFp ctx s F n = none := by
  cases F with
  | zero => rfl
  | succ f => simp [nodeFp, simpleFp, hk]

theorem nodeFp_nested {ctx : Ctx} {s : State} {F : Nat} {n : DNode} {g : SDFG} {mapping : List (Sym × Expr)}
    {ni : Bool} (hk : n.kind = .nested g mapping ni) : nodeFp ctx s F n = none := by
  cases F with
  | zero => rfl
  | succ f => simp [nodeFp, simpleFp, hk]

section Setup

variable {ctx : Ctx} {ρ : Valuation} {m₀ : Mem} {P : LocPred}

/-- Without view access nodes among `ids`, the view prologue does nothing. -/
theorem bindViews_step {s : State} {ids : List NodeId} {st st' : St} {F : Nat} {A : List Access}
    (h : bindViews ctx ρ s ids st = .ok st') (hinc : FpIncl ctx s F ids A) : St.Step m₀ P st st' := by
  unfold bindViews at h
  refine List.foldlM_step id _ ids st st' ?_ h
  intro x hx c c' hc
  split at hc
  · rename_i n hn
    split at hc
    · rename_i d hk
      split at hc
      · rename_i hv
        exfalso
        obtain ⟨An, hAn, -⟩ := hinc x hx n hn
        have := nodeFp_simple (by simp [DNode.isSimple, hk]) hAn
        simp [simpleFp, hk, hv] at this
      · exact St.Step.of_pure hc
    · exact St.Step.of_pure hc
  · exact St.Step.of_pure hc

/-- Without dynamic inputs, the map set-up only enumerates the range. -/
theorem mapSetup_eq {s : State} {n : DNode} {mp : MapNode} {st st₁ : St} {ρ' : Valuation} {pts : List Idx}
    (hdyn : (s.inEdges n.id).any (s.isDynamicInput ·) = false)
    (h : mapSetup ctx ρ s n mp st = .ok (ρ', st₁, pts)) :
    ρ' = ρ ∧ st₁ = st ∧ ctx.rangePoints ρ st.mem mp.range = .ok pts := by
  have hb : bindDynamicInputs ctx ρ s n st = .ok (ρ, st) := by
    unfold bindDynamicInputs
    have : ∀ es : List DEdge, (∀ e ∈ es, s.isDynamicInput e = false) →
        es.foldlM (fun (acc : Valuation × St) e =>
          if s.isDynamicInput e then
            match e.data, e.dstConn with
            | .memlet _, some c => do
              let (v?, st) ← readEdgeScalar ctx acc.1 s e acc.2
              match v? with
              | some v => pure (Valuation.set acc.1 c v, st)
              | none => pure (acc.1, st)
            | _, _ => pure (acc.1, acc.2)
          else pure (acc.1, acc.2)) (ρ, st) = .ok (ρ, st) := by
      intro es hes
      induction es with
      | nil => rfl
      | cons e es ih =>
        simp only [List.foldlM_cons, hes e (List.mem_cons_self ..), Bool.false_eq_true, ↓reduceIte]
        exact ih fun x hx => hes x (List.mem_cons_of_mem _ hx)
    refine this _ fun e he => ?_
    simpa using List.any_eq_false.1 hdyn e he
  unfold mapSetup at h
  rw [hb] at h
  obtain ⟨x, hx, h⟩ := Res.bind_eq_ok.1 h
  cases hx
  dsimp only at h
  obtain ⟨pts', hp, h⟩ := Res.bind_eq_ok.1 h
  rw [Res.pure_eq_ok] at h
  cases h
  exact ⟨rfl, rfl, hp⟩

theorem merge_aliases {w : WcrSem} {m m' : Mem} {gs : List Log} (h : merge w m gs = .ok m')
    (ha : (Log.unionAll gs).aliasSets = []) : m'.aliases = m.aliases := by
  unfold merge at h
  split at h
  · cases h
    rw [Merge.applyQueues_eq, Merge.applyAliases_eq, ha, Merge.applyWcrs_eq, Merge.applyWrites_eq]
    rfl
  · split at h <;> cases h

/-- The merge of covered logs. -/
theorem scopeFinish_step {pre st' : St} {logs : List Log} (h : scopeFinish ctx pre logs = .ok st')
    (hw : ∀ g ∈ logs, g.Within P) : St.Step m₀ P pre st' := by
  unfold scopeFinish at h
  split at h
  · rename_i m' hm
    cases h
    have hu := Log.Within.unionAll hw
    exact St.Step.of_logged (merge_aliases hm hu.aliasSets) rfl fun _ => hu
  · cases h

/-- A log covered inside an iteration `i` of a map is covered outside the
map by the accesses with one more binder. -/
theorem Log.Within.bind {g : Log} {B : List Access} {ps : List Sym} {r : Range} {i : Idx}
    (hw : g.Within (LocPred.ofAccesses ctx m₀ (ρ.bindIdx ps i) B))
    (hi : ∃ m pts, ctx.rangePoints ρ m r = .ok pts ∧ i ∈ pts) :
    g.Within (LocPred.ofAccesses ctx m₀ ρ (B.map (Access.bind ps r))) := by
  obtain ⟨m, pts, hp, hip⟩ := hi
  have lift : ∀ a ∈ B, ∀ l, a.Hit ctx m₀ (ρ.bindIdx ps i) l →
      ∃ a' ∈ B.map (Access.bind ps r), a'.kind = a.kind ∧ a'.Hit ctx m₀ ρ l := by
    intro a ha l ⟨ρ', hext, off, hoff, hl⟩
    exact ⟨a.bind ps r, List.mem_map_of_mem ha, rfl, ρ', ⟨m, pts, hp, i, hip, hext⟩, off, hoff, hl⟩
  refine hw.mono ⟨?_, ?_, ?_⟩
  · intro l ⟨a, ha, hk, hh⟩
    obtain ⟨a', ha', hk', hh'⟩ := lift a ha l hh
    exact ⟨a', ha', hk'.trans hk, hh'⟩
  · intro l ⟨a, ha, hk, hh⟩
    obtain ⟨a', ha', hk', hh'⟩ := lift a ha l hh
    exact ⟨a', ha', hk'.trans hk, hh'⟩
  · intro op l ⟨a, ha, hk, hh⟩
    obtain ⟨a', ha', hk', hh'⟩ := lift a ha l hh
    exact ⟨a', ha', hk'.trans hk, hh'⟩

end Setup

/-! ## The dataflow relations -/

section Relations

variable {ctx : Ctx} {m₀ : Mem}

mutual

/-- **Footprint of a node**: a successful execution of a node in the
fragment changes no alias and logs only locations of its static footprint
(`nodeFp`). -/
theorem NodeExec.fp_step {ρ : Valuation} {s : State} {n : DNode} {st st' : St} {r : Res St} {F : Nat}
    {A : List Access} :
    NodeExec ctx ρ s n st r → r = .ok st' → nodeFp ctx s F n = some A →
      St.Step m₀ (LocPred.ofAccesses ctx m₀ ρ A) st st'
  | .simple hs, hr, hfp => execSimple_step hr (nodeFp_simple hs hfp)
  | .mapSetupErr _ _, hr, _ => by cases hr
  | .mapIterErr _ _ _, hr, _ => by cases hr
  | .mapScope hk hs hi, hr, hfp => by
    obtain ⟨f, B, hF, hdyn, hB, hA⟩ := nodeFp_mapEntry hk hfp
    have hlogs := IterExec.fp_logs hi rfl (scopeFp_incl hB)
    obtain ⟨hρ, hst, hpts⟩ := mapSetup_eq hdyn hs
    subst hρ hst hA
    intro hal
    exact scopeFinish_step hr (fun g hg =>
      let ⟨i, hi, hw⟩ := hlogs hal g hg
      hw.bind ⟨_, _, hpts, hi⟩) hal
  | .consumeSetupErr _ _, hr, _ => by cases hr
  | .consumeTasksErr _ _ _, hr, _ => by cases hr
  | .consumeScope hk _ _, _, hfp => by rw [nodeFp_consumeEntry hk] at hfp; cases hfp
  | .nestedSetupErr _ _, hr, _ => by cases hr
  | .nestedInnerErr _ _ _, hr, _ => by cases hr
  | .nested hk _ _, _, hfp => by rw [nodeFp_nested hk] at hfp; cases hfp

/-- **Footprint of a sequential execution**: the accumulated log stays
within the footprints of the nodes, and every node's delta in the trace is
within that node's own footprint. -/
theorem SeqExec.fp_step {ρ : Valuation} {s : State} {ids : List NodeId} {st st₂ : St}
    {tr : List (NodeId × Log)} {r : Res (St × List (NodeId × Log))} {F : Nat} {A : List Access} :
    SeqExec ctx ρ s ids st r → r = .ok (st₂, tr) → FpIncl ctx s F ids A →
      St.Step m₀ (LocPred.ofAccesses ctx m₀ ρ A) st st₂ ∧
      (st.mem.aliases = m₀.aliases → ∀ x ∈ tr, ∃ n An, s.node? x.1 = some n ∧ nodeFp ctx s F n = some An ∧
        x.2.Within (LocPred.ofAccesses ctx m₀ ρ An))
  | .nil, hr, _ => by
    cases hr
    exact ⟨St.Step.refl _, fun _ x hx => absurd hx List.not_mem_nil⟩
  | .missing _, hr, _ => by cases hr
  | .headErr _ _, hr, _ => by cases hr
  | .tailErr _ _ _, hr, _ => by cases hr
  | .cons (id := id) (n := n) (st' := st₁) hn hnode hrest, hr, hinc => by
    obtain ⟨An, hAn, hsub⟩ := hinc id (List.mem_cons_self ..) n hn
    have hnode' : St.Step m₀ (LocPred.ofAccesses ctx m₀ ρ An) st.forNode st₁ := NodeExec.fp_step hnode rfl hAn
    have hrest' := SeqExec.fp_step hrest rfl (hinc.of_sub fun x hx => List.mem_cons_of_mem _ hx)
    cases hr
    have h1 : St.Step m₀ (LocPred.ofAccesses ctx m₀ ρ A) st (st.afterNode st₁) := fun hal =>
      let ⟨hal₁, hw⟩ := hnode' hal
      ⟨hal₁, fun hwst => hwst.seq ((hw Log.Within.empty).mono (LocPred.ofAccesses_mono hsub))⟩
    refine ⟨h1.trans hrest'.1, fun hal x hx => ?_⟩
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨n, An, hn, hAn, (hnode' hal).2 Log.Within.empty⟩
    · exact hrest'.2 (h1 hal).1 x hx

/-- **Footprint of a scope body or component**. -/
theorem OrderedExec.fp_step {ρ : Valuation} {s : State} {all direct : List NodeId} {st st' : St} {r : Res St}
    {F : Nat} {A : List Access} :
    OrderedExec ctx ρ s all direct st r → r = .ok st' → FpIncl ctx s F direct A →
      St.Step m₀ (LocPred.ofAccesses ctx m₀ ρ A) st st'
  | .prologueErr _, hr, _ => by cases hr
  | .seqErr _ _ _, hr, _ => by cases hr
  | .hbRace _ _ _ _, hr, _ => by cases hr
  | .run _ hp hs _, hr, hinc => by
    have h₂ := (SeqExec.fp_step hs rfl
      (hinc.of_sub fun x hx => by simpa using (List.mem_filter.1 hx).2)).1
    cases hr
    exact (bindViews_step hp hinc).trans h₂

/-- **Footprint of the iterations of a map** (`IterDen.footprint`): the log
of every iteration (private transients dropped) is within the footprint of
the scope body under the valuation of some iteration point. -/
theorem IterExec.fp_logs {ρ : Valuation} {s : State} {entry : NodeId} {params : List Sym} {priv : List Name}
    {pre : St} {pts : List Idx} {r : Res (List Log)} {F : Nat} {A : List Access} {logs : List Log} :
    IterExec ctx ρ s entry params priv pre pts r → r = .ok logs →
      FpIncl ctx s F (s.directChildren (some entry)) A → pre.mem.aliases = m₀.aliases →
      ∀ g ∈ logs, ∃ i ∈ pts, g.Within (LocPred.ofAccesses ctx m₀ (ρ.bindIdx params i) A)
  | .nil, hr, _, _ => by
    cases hr
    intro g hg
    exact absurd hg List.not_mem_nil
  | .headErr _, hr, _, _ => by cases hr
  | .tailErr _ _, hr, _, _ => by cases hr
  | .cons (i := i) hb hrest, hr, hinc, hal => by
    have h₁ := OrderedExec.fp_step hb rfl hinc (by simpa [iterStart] using hal)
    have h₂ := IterExec.fp_logs hrest rfl hinc hal
    cases hr
    intro g hg
    rcases List.mem_cons.1 hg with rfl | hg
    · exact ⟨i, List.mem_cons_self .., (h₁.2 Log.Within.empty).dropNames priv⟩
    · obtain ⟨j, hj, hw⟩ := h₂ g hg
      exact ⟨j, List.mem_cons_of_mem _ hj, hw⟩

end

/-- **Footprint of the node deltas of a sequential trace** (the deltas the
happens-before check `State.hbFree` compares, `Decisions.md#D61`): each is
within the footprint of its own node. -/
theorem SeqExec.fp_trace {ρ : Valuation} {s : State} {ids : List NodeId} {st st₂ : St}
    {tr : List (NodeId × Log)} {F : Nat} {A : List Access}
    (h : SeqExec ctx ρ s ids st (.ok (st₂, tr))) (hinc : FpIncl ctx s F ids A) (hal : st.mem.aliases = m₀.aliases) :
    ∀ x ∈ tr, ∃ n An, s.node? x.1 = some n ∧ nodeFp ctx s F n = some An ∧
      x.2.Within (LocPred.ofAccesses ctx m₀ ρ An) :=
  (SeqExec.fp_step h rfl hinc).2 hal

/-- A successful scope body or component started from an empty log logs
only within its footprint. -/
theorem OrderedExec.fp_log {ρ : Valuation} {s : State} {all direct : List NodeId} {st st' : St} {F : Nat}
    {A : List Access} (h : OrderedExec ctx ρ s all direct st (.ok st')) (hinc : FpIncl ctx s F direct A)
    (hal : st.mem.aliases = m₀.aliases) (hlog : st.log = Log.empty) :
    st'.mem.aliases = m₀.aliases ∧ st'.log.Within (LocPred.ofAccesses ctx m₀ ρ A) :=
  let ⟨h₁, h₂⟩ := OrderedExec.fp_step h rfl hinc hal
  ⟨h₁, h₂ (hlog ▸ Log.Within.empty)⟩

/-- A successful node started from an empty log (as every node of a
sequential execution is, `St.forNode`) logs only within its footprint. -/
theorem NodeExec.fp_log {ρ : Valuation} {s : State} {n : DNode} {st st' : St} {F : Nat} {A : List Access}
    (h : NodeExec ctx ρ s n st (.ok st')) (hfp : nodeFp ctx s F n = some A) (hal : st.mem.aliases = m₀.aliases)
    (hlog : st.log = Log.empty) :
    st'.mem.aliases = m₀.aliases ∧ st'.log.Within (LocPred.ofAccesses ctx m₀ ρ A) :=
  let ⟨h₁, h₂⟩ := NodeExec.fp_step h rfl hfp hal
  ⟨h₁, h₂ (hlog ▸ Log.Within.empty)⟩

end Relations

/-- **Footprint of the components of a state**: the log of every component
is within the footprint of its top-level nodes (aliases resolved in the
state's initial memory `m`). -/
theorem CompsExec.fp_logs {ctx : Ctx} {ρ : Valuation} {s : State} {m : Mem} {F : Nat} :
    ∀ {comps : List (List NodeId)} {r : Res (List Log)} {logs : List Log}, CompsExec ctx ρ s m comps r →
      r = .ok logs →
      List.Forall₂ (fun comp g => ∀ A, scopeFp ctx s F (s.topLevelOf comp) = some A →
        g.Within (LocPred.ofAccesses ctx m ρ A)) comps logs
  | _, _, _, .nil, hr => by cases hr; exact .nil
  | _, _, _, .headErr _, hr => by cases hr
  | _, _, _, .tailErr _ _, hr => by cases hr
  | _, _, _, .cons hc hrest, hr => by
    have h₂ := CompsExec.fp_logs (F := F) hrest rfl
    cases hr
    exact .cons (fun A hA => (OrderedExec.fp_log hc (scopeFp_incl hA) rfl rfl).2) h₂

end Dace
