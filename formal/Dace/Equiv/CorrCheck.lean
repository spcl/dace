-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.CorrCtl
import Dace.Equiv.NamesCheck
import Dace.STree.Retarget

/-!
# The verified SDFG / schedule-tree pair checker (M5a, M5b steps 1–3)

`corrCheck g t` decides the correspondence `ProgCorr` of `Dace.Equiv.Corr`
for the M5a v1 fragment and is proven sound (`corrCheck_sound`), so that
`corr? g t = true` gives, for every interpretation, that every outcome of
the tree is an outcome of the SDFG or the SDFG may be undefined
(`corr?_exec`), hence `Refines t.prog g.prog` when the SDFG is supported
(`corr?_refines`).

The checker walks the tree's root list state by state.  In a state (or map
scope) it places the SDFG nodes greedily: the first node whose predecessors
are placed and whose image (`matchOne`: copies, a tasklet or library node
with equal keys, a map scope whose body matches recursively) is a prefix of
the remaining tree list.  The order found must be a topological order of the
component (or scope); the greedy choice only affects completeness.

The control flow is walked by `regCheck`/`blockCheck`/`elifCheck` (fuel
bounded by the size of the region): a region is a chain of blocks joined
by single unconditional edges, each edge's assignments a permutation of the
following `AssignNode`s that may run one by one (`SeqAssigns`); a block is a
state, a loop region (equal loop properties, body checked recursively), a
conditional block (`if`/`elif`/`else` chain), a break/continue/return, or
a nested region (a `NamedRegionScope` for a named region; inlined for plain
and function-call regions, `regCheck` then returns the rest of the list).

Fragment (`UNKNOWN` with a reason outside it): no branching or conditional
interstate edges (GBlocks, unstructured regions); every state is
`SeqPlain` (no streams, views, references, consume scopes or code→code
values; private transients allowed when both sides compute the same private
sets) with at most one component, its nodes simple nodes and map scopes;
no constants; a code
node's input memlets in the order of its in-edges (`sdfg_to_tree` lists
them in `state.in_edges` order, so the tree must be built from the SDFG as
stored).
-/

set_option autoImplicit false

namespace Dace

namespace Corr

/-! ## Matching -/

/-- The leading copy nodes of a tree list, exactly `k` of them. -/
def takeCopies : Nat → List SNode → Option (List (Name × Memlet) × List SNode)
  | 0, cs => some ([], cs)
  | k + 1, .copyNode tgt m :: cs => (takeCopies k cs).map fun p => ((tgt, m) :: p.1, p.2)
  | _ + 1, _ => none

theorem takeCopies_sound : ∀ {k : Nat} {cs : List SNode} {ps : List (Name × Memlet)} {rest : List SNode},
    takeCopies k cs = some (ps, rest) → cs = ps.map (fun p => SNode.copyNode p.1 p.2) ++ rest
  | 0, cs, ps, rest, h => by
    simp only [takeCopies, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    rfl
  | k + 1, cs, ps, rest, h => by
    cases cs with
    | nil => simp [takeCopies] at h
    | cons c cs =>
      cases c with
      | copyNode tgt m =>
        simp only [takeCopies, Option.map_eq_some_iff] at h
        obtain ⟨⟨ps', rest'⟩, h', he⟩ := h
        simp only [Prod.mk.injEq] at he
        obtain ⟨rfl, rfl⟩ := he
        rw [takeCopies_sound h']
        rfl
      | _ => simp [takeCopies] at h

/-- The leading `DynScopeCopyNode`s of a tree list, exactly `k` of them:
their input keys and the rest. -/
def takeDyn (tc : TCtx) : Nat → List SNode → Option (List InKey × List SNode)
  | 0, cs => some ([], cs)
  | k + 1, .dynScopeCopy t m :: cs => (takeDyn tc k cs).map fun p => (tc.inKey t m :: p.1, p.2)
  | _ + 1, _ => none

theorem takeDyn_sound {tc : TCtx} : ∀ {k : Nat} {cs : List SNode} {ks : List InKey} {rest : List SNode},
    takeDyn tc k cs = some (ks, rest) → ∃ ds, cs = ds ++ rest ∧ DynNodes tc ks ds
  | 0, cs, ks, rest, h => by
    simp only [takeDyn, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨[], rfl, .nil⟩
  | k + 1, c :: cs, ks, rest, h => by
    cases c with
    | dynScopeCopy t m =>
      simp only [takeDyn, Option.map_eq_some_iff] at h
      obtain ⟨⟨ks', rest'⟩, h', he⟩ := h
      simp only [Prod.mk.injEq] at he
      obtain ⟨rfl, rfl⟩ := he
      obtain ⟨ds, rfl, hds⟩ := takeDyn_sound h'
      have hkc : (tc.inKey t m).conn = t := rfl
      exact ⟨.dynScopeCopy t m :: ds, rfl, .cons ⟨m, by rw [hkc], rfl⟩ hds⟩
    | _ => simp [takeDyn] at h
  | _ + 1, [], _, _, h => by simp [takeDyn] at h

/-! ## Nested SDFGs -/

/-- Swap each pair of names (pairs on disjoint names make an involution). -/
def swapsPerm : List (Name × Name) → Equiv.Perm Name
  | [] => 1
  | (a, b) :: ps => Equiv.swap a b * swapsPerm ps

/-- The names of the pairs. -/
def pairNames (ps : List (Name × Name)) : List Name := ps.flatMap fun p => [p.1, p.2]

theorem swapsPerm_of_not_mem : ∀ (ps : List (Name × Name)) (x : Name), x ∉ pairNames ps → swapsPerm ps x = x
  | [], _, _ => rfl
  | (a, b) :: ps, x, h => by
    simp only [pairNames, List.flatMap_cons, List.mem_append, List.mem_cons, List.not_mem_nil, or_false,
      not_or] at h
    obtain ⟨⟨ha, hb⟩, hr⟩ := h
    simp only [swapsPerm, Equiv.Perm.mul_apply]
    rw [swapsPerm_of_not_mem ps x hr, Equiv.swap_apply_of_ne_of_ne ha hb]

/-- The renaming of a nested call's tree body (`InnerCorr`): each outer
container swapped with the connector it is bound to (`nestedBindings`), like
`symbol_mapping` does for symbols.  The name checks (`innerNames?`) verify
that it relates every inner name to its outer name. -/
def connPerm (conns : List (Name × Name)) : Equiv.Perm Name :=
  swapsPerm ((conns.filter fun p => p.1 != p.2).map fun p => (p.2, p.1))

/-- Decision procedure for `Ctx.DescFixed`. -/
def descFixed? (σ : Equiv.Perm Name) (ctx : Ctx) : Bool := ctx.g.arrays.all fun p => fixedB σ p.2.fullRange.names

theorem descFixed?_sound {σ : Equiv.Perm Name} {ctx : Ctx} (h : descFixed? σ ctx = true) : ctx.DescFixed σ := by
  intro n d hd
  simp only [Ctx.desc?, SDFG.array?, Option.map_eq_some_iff] at hd
  obtain ⟨p, hp, rfl⟩ := hd
  exact fixedB_iff.1 (List.all_eq_true.1 h p (List.mem_of_find?_eq_some hp))


/-- `l₁` and `l₂` have the same members. -/
def sameMembers? (l₁ l₂ : List Name) : Bool := l₁.all l₂.contains && l₂.all l₁.contains

theorem sameMembers?_sound {l₁ l₂ : List Name} (h : sameMembers? l₁ l₂ = true) : ∀ x, x ∈ l₁ ↔ x ∈ l₂ := by
  simp only [sameMembers?, Bool.and_eq_true, List.all_eq_true] at h
  intro x
  exact ⟨fun hx => by simpa using h.1 x hx, fun hx => by simpa using h.2 x hx⟩

/-- Decision procedure for `NestedMapOk`. -/
def nestedMapOk? (ctx : Ctx) (inner : SDFG) (mapping : List (Sym × Expr)) : Bool :=
  mapping.all (fun p => p.2 == .sym p.1 && !ctx.isScalarName p.1) && inner.constants.isEmpty

theorem nestedMapOk?_sound {ctx : Ctx} {inner : SDFG} {mapping : List (Sym × Expr)}
    (h : nestedMapOk? ctx inner mapping = true) : NestedMapOk ctx inner mapping := by
  simp only [nestedMapOk?, Bool.and_eq_true, List.all_eq_true, beq_iff_eq, Bool.not_eq_true',
    List.isEmpty_iff] at h
  refine ⟨fun p hp => h.1 p hp, fun ρ => ?_⟩
  funext x
  simp [SDFG.withConstants, h.2]

/-- A name agrees between `c₂` and `c₁` up to `π` (`Ctx.Good₀`). -/
def nameOkE (π : Equiv.Perm Name) (c₂ c₁ : Ctx) (x : Name) : Bool :=
  c₂.desc? x == c₁.desc? x && ((c₁.desc? x).isNone || c₂.cname x == π (c₁.cname x))

/-- A container name agrees between `c₂` and `c₁` up to `π` (`Ctx.Strong`). -/
def nameOkD (π : Equiv.Perm Name) (c₂ c₁ : Ctx) (x : Name) : Bool :=
  nameOkE π c₂ c₁ x && c₂.cname x == π (c₁.cname x) &&
    ((c₁.desc? x).map fun d => d.fullRange.names.all (nameOkE π c₂ c₁)).getD true

theorem nameOkE_sound {π : Equiv.Perm Name} {c₂ c₁ : Ctx} {x : Name} (h : nameOkE π c₂ c₁ x = true) :
    Ctx.Good₀ π (fun _ => True) c₂ c₁ x := by
  simp only [nameOkE, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, Option.isNone_iff_eq_none] at h
  refine ⟨h.1, fun hn => ⟨?_, trivial⟩⟩
  rcases h.2 with h' | h'
  · exact absurd h' hn
  · exact h'

theorem nameOkD_sound {π : Equiv.Perm Name} {c₂ c₁ : Ctx} {x : Name} (h : nameOkD π c₂ c₁ x = true) :
    Ctx.Strong π (fun _ => True) c₂ c₁ x := by
  simp only [nameOkD, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨hE, hc⟩, hs⟩ := h
  refine ⟨(nameOkE_sound hE).1, hc, trivial, fun d hd y hy => nameOkE_sound ?_⟩
  rw [hd] at hs
  simp only [Option.map_some, Option.getD_some, List.all_eq_true] at hs
  exact hs y hy

/-- The names of the top region of a nested SDFG agree between the call's
context and the tree's (`InnerNames`); the inner states contain no nested
SDFGs. -/
def innerNames? (ctx : Ctx) (tc : TCtx) (inner : SDFG) (node : NodeId) (conns : List (Name × Name)) : Bool :=
  Region.namesOk? (nameOkD 1 (ctx.enter inner node conns) (tc.innerCtx inner))
    (nameOkE 1 (ctx.enter inner node conns) (tc.innerCtx inner)) false inner.top

theorem innerNames?_sound {ctx : Ctx} {tc : TCtx} {inner : SDFG} {node : NodeId} {conns : List (Name × Name)}
    (h : innerNames? ctx tc inner node conns = true) : InnerNames ctx tc inner node conns := by
  intro I
  exact Region.namesOk?_sound (fun x hx => nameOkD_sound hx) (fun x hx => nameOkE_sound hx) (fun h => by cases h) _ h

/-- Decision procedure for `IdleNodes`. -/
def idle? (ctx : Ctx) (s : State) (xs : List NodeId) : Bool :=
  xs.all fun x => match s.node? x with
    | some xn => DNode.isCopyTarget xn && s.copyShape? ctx xn && (s.copiesInto ctx xn).isEmpty
    | none => false

theorem idle?_sound {ctx : Ctx} {s : State} {xs : List NodeId} (h : idle? ctx s xs = true) : IdleNodes ctx s xs := by
  intro x hx
  have := List.all_eq_true.1 h x hx
  cases hn : s.node? x with
  | none => rw [hn] at this; cases this
  | some xn =>
    rw [hn] at this
    simp only [Bool.and_eq_true, List.isEmpty_iff] at this
    exact ⟨xn, rfl, this.1.1, State.copyShape?_sound this.1.2, this.2⟩

/-- Decision procedure for `State.DataOuts`. -/
def _root_.Dace.State.dataOuts? (s : State) (n : DNode) : Bool :=
  (s.outEdges n.id).all fun e => match e.data with
    | .memlet _ => (s.dstEndpoint e).isData
    | .dep => true

theorem _root_.Dace.State.dataOuts?_sound {s : State} {n : DNode} (h : s.dataOuts? n = true) : s.DataOuts n := by
  intro e he ⟨m, hm⟩
  have := List.all_eq_true.1 h e he
  rw [hm] at this
  exact this

/-- Recognize a map scope whose body is one nested SDFG (`SegCorr.nestedMap`;
`nest` checks the inlined inner region): the order of the scope. -/
def nestedChild (ctx : Ctx) (tc : TCtx) (nest : Equiv.Perm Name → SDFG → List SNode → Bool) (s : State) (n : DNode)
    (body : List SNode) : Option (List NodeId) :=
  match s.topoOrder (s.scopeNodes n.id) with
  | none => none
  | some full =>
  match full.filter (s.directChildren (some n.id)).contains with
  | nid :: xs =>
    match s.node? nid with
    | some nd =>
      match nd.kind with
      | .nested inner mapping _ =>
        match nestedBindings ctx s nd with
        | .ok conns =>
            if idle? ctx s xs = true ∧
                nestedMapOk? ctx inner mapping = true ∧ (nestedConns nd).all (fun c => (nestedEndpoint s nd c).isData) = true ∧
                s.dataOuts? nd = true ∧
                sameMembers? (tc.privateTransients body)
                  (ctx.privateTransients s n.id ++ inner.callTransients.map (ctx.enter inner nd.id conns).cname) = true ∧
                innerNames? ctx (tc.renT (connPerm conns)) inner nd.id conns = true ∧
                nest (connPerm conns) inner body = true ∧
                s.isTopologicalOrder (s.scopeNodes n.id) full = true
            then some full else none
        | .err _ => none
      | _ => none
    | none => none
  | _ => none

theorem nestedChild_sound {ctx : Ctx} {tc : TCtx} {nest : Equiv.Perm Name → SDFG → List SNode → Bool} {s : State}
    (hnest : ∀ σ inner body, nest σ inner body = true → InnerCorr tc σ inner body)
    {n : DNode} {body : List SNode} {full : List NodeId} (h : nestedChild ctx tc nest s n body = some full) :
    ∀ {id : NodeId} {o : List NodeId} {cs : List SNode} {mp : MapNode} {keys : List InKey} {ds : List SNode},
      s.node? id = some n → n.kind = .mapEntry mp → s.dynKeys? n = some keys → DynOk ctx keys → DynNodes tc keys ds →
      SegCorr ctx tc s o cs → SegCorr ctx tc s (id :: o) (ds ++ .mapScope mp body :: cs) := by
  intro id o cs mp keys ds hn hk hkeys hok hds ht
  unfold nestedChild at h
  split at h
  · cases h
  rename_i full' _
  split at h
  · rename_i nid xs hdir
    split at h
    · rename_i nd hnd
      split at h
      · rename_i inner mapping ni hkn
        split at h
        · rename_i conns hb
          split at h
          · rename_i hc
            obtain ⟨hidle, hm, hcode, hout, hpriv, hnames, hn', hto⟩ := hc
            cases h
            exact .nestedMap hn hk hkeys hok hds hto hdir (idle?_sound hidle) hnd hkn hb (nestedMapOk?_sound hm)
              (List.all_eq_true.1 hcode) (State.dataOuts?_sound hout) (sameMembers?_sound hpriv)
              (innerNames?_sound hnames) (hnest _ _ _ hn') ht
          · cases h
        · cases h
      · cases h
    · cases h
  · cases h

/-- The longest prefix of `cs` satisfying `p`, with the rest of `cs`. -/
def longestPrefix (p : List SNode → Bool) (cs : List SNode) : Option (List SNode × List SNode) :=
  ((List.range (cs.length + 1)).reverse.find? fun k => p (cs.take k)).map fun k => (cs.take k, cs.drop k)

theorem longestPrefix_sound {p : List SNode → Bool} {cs body rest : List SNode}
    (h : longestPrefix p cs = some (body, rest)) : p body = true ∧ cs = body ++ rest := by
  simp only [longestPrefix, Option.map_eq_some_iff, Prod.mk.injEq] at h
  obtain ⟨k, hk, rfl, rfl⟩ := h
  exact ⟨by simpa using List.find?_some hk, (List.take_append_drop k cs).symm⟩

/-- Decision procedure for `NestedShape`. -/
def nestedShape? (body : List SNode) : Bool :=
  body.all SNode.isFlatT ||
    match exitLabel? body with
    | some (L, pre) => !SNodes.hasLabel L pre && SNodes.jumpsTo (some L) pre && SNodes.keepsSyms pre
    | none => false

theorem nestedShape?_sound {body : List SNode} (h : nestedShape? body = true) : NestedShape body := by
  unfold nestedShape? at h
  rcases Bool.or_eq_true_iff.1 h with h | h
  · exact .inl h
  · split at h
    · rename_i L pre hl
      simp only [Bool.and_eq_true, Bool.not_eq_true'] at h
      exact .inr ⟨L, pre, exitLabel?_sound hl, h.1.1, h.1.2, h.2⟩
    · cases h

/-- Recognize a nested SDFG node outside map scopes (`SegCorr.nested`; `nest`
checks the inlined inner region): the inlined body at the head of `cs` and
the rest. -/
def nestedCall (ctx : Ctx) (tc : TCtx) (nest : Equiv.Perm Name → SDFG → List SNode → Bool) (s : State) (n : DNode)
    (cs : List SNode) : Option (List SNode × List SNode) :=
  match n.kind with
  | .nested inner mapping _ =>
    match nestedBindings ctx s n with
    | .ok conns =>
      if nestedMapOk? ctx inner mapping = true ∧
          (nestedConns n).all (fun c => (nestedEndpoint s n c).isData) = true ∧
          s.dataOuts? n = true ∧ inner.callTransients = [] ∧
          innerNames? ctx (tc.renT (connPerm conns)) inner n.id conns = true
      then longestPrefix (fun body => nestedShape? body && nest (connPerm conns) inner body) cs
      else none
    | .err _ => none
  | _ => none

theorem nestedCall_sound {ctx : Ctx} {tc : TCtx} {nest : Equiv.Perm Name → SDFG → List SNode → Bool} {s : State}
    (hnest : ∀ σ inner body, nest σ inner body = true → InnerCorr tc σ inner body)
    {n : DNode} {cs body rest : List SNode} (h : nestedCall ctx tc nest s n cs = some (body, rest)) :
    cs = body ++ rest ∧
      ∀ id o tail, s.node? id = some n → SegCorr ctx tc s o tail → SegCorr ctx tc s (id :: o) (body ++ tail) := by
  unfold nestedCall at h
  split at h
  · rename_i inner mapping ni hkn
    split at h
    · rename_i conns hb
      split at h
      · rename_i hc
        obtain ⟨hm, hcode, hout, hctr, hnames⟩ := hc
        obtain ⟨hp, hcs⟩ := longestPrefix_sound h
        simp only [Bool.and_eq_true] at hp
        refine ⟨hcs, fun id o tail hn ht => ?_⟩
        exact .nested hn hkn hb (nestedMapOk?_sound hm) (List.all_eq_true.1 hcode) (State.dataOuts?_sound hout)
          hctr (nestedShape?_sound hp.1) (innerNames?_sound hnames) (hnest _ _ _ hp.2) ht
      · cases h
    · cases h
  · cases h

/-- A matched scope: the order of its direct nodes, the full (nested) order
and the rest of the tree list. -/
abbrev MatchRes := List NodeId × List NodeId × List SNode

/-- `SubCorr`, decided. -/
def subCorr? (ctx : Ctx) (base : Name) (tsub ssub : Option Subset) : Bool :=
  if tsub = ssub then true
  else
    match tsub, ctx.desc? base with
    | none, some dd => decide (ssub = some (.range dd.fullRange))
    | _, _ => false

theorem subCorr?_sound {ctx : Ctx} {base : Name} {tsub ssub : Option Subset} (h : subCorr? ctx base tsub ssub = true) :
    SubCorr ctx base tsub ssub := by
  unfold subCorr? at h
  by_cases heq : tsub = ssub
  · exact .inl heq
  · rw [if_neg heq] at h
    cases tsub with
    | some _ => simp at h
    | none =>
      cases hd : ctx.desc? base with
      | none => rw [hd] at h; cases h
      | some dd => rw [hd] at h; exact .inr ⟨rfl, dd, hd, of_decide_eq_true h⟩

/-- `KeyCorr` pointwise, decided. -/
def keysCorr? (ctx : Ctx) : List CopyKey → List CopyKey → Bool
  | [], [] => true
  | kT :: ksT, kS :: ksS =>
    (kT.dst = kS.dst ∧ kT.src = kS.src ∧ kT.wcr = kS.wcr ∧ kT.oob = kS.oob ∧
      subCorr? ctx kS.src kT.ssub kS.ssub = true ∧ subCorr? ctx kS.dst kT.dsub kS.dsub = true) &&
      keysCorr? ctx ksT ksS
  | _, _ => false

theorem keysCorr?_sound {ctx : Ctx} : ∀ {ksT ksS : List CopyKey}, keysCorr? ctx ksT ksS = true →
    List.Forall₂ (KeyCorr ctx) ksT ksS
  | [], [], _ => .nil
  | _ :: _, _ :: _, h => by
    simp only [keysCorr?, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨h1, h2, h3, h4, h5, h6⟩, hr⟩ := h
    exact .cons ⟨h1, h2, h3, h4, subCorr?_sound h5, subCorr?_sound h6⟩ (keysCorr?_sound hr)
  | [], _ :: _, h => by simp [keysCorr?] at h
  | _ :: _, [], h => by simp [keysCorr?] at h

/-- Match a reference set from a container (`SegCorr.refSet`). -/
def matchRefSet (ctx : Ctx) (s : State) (n : DNode) (cs : List SNode) : Option (List NodeId × List SNode) :=
  match s.refSetIn? ctx n, cs with
  | some (d, base, sub), .refSetNode d' m (.data _) _ :: rest =>
    if d' = d ∧ m.data = base ∧ subCorr? ctx base m.dataSubset sub = true then some ([], rest) else none
  | _, .refSetNode d' _ (.codeNode _ _) _ :: rest =>
    if s.refSetCode? ctx n = some d' then some ([], rest) else none
  | _, _ => none

theorem matchRefSet_sound {ctx : Ctx} {tc : TCtx} {s : State} {n : DNode} {cs : List SNode}
    {fullN : List NodeId} {rest : List SNode} (hk : DNode.isCopyTarget n = true)
    (h : matchRefSet ctx s n cs = some (fullN, rest)) :
    ∃ img, cs = img ++ rest ∧
      ∀ id o tail, s.node? id = some n → SegCorr ctx tc s o tail → SegCorr ctx tc s (id :: o) (img ++ tail) := by
  unfold matchRefSet at h
  split at h
  · rename_i d base sub d' m sd x rest' hr
    split at h
    · rename_i hc
      obtain ⟨rfl, hmd, hsc⟩ := hc
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨-, rfl⟩ := h
      exact ⟨[_], rfl, fun id o tail hn ht => .refSet hn hk hr hmd (subCorr?_sound hsc) ht⟩
    · cases h
  · split at h
    · rename_i hr
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨-, rfl⟩ := h
      exact ⟨[_], rfl, fun id o tail hn ht => .refSetCode hn hk hr ht⟩
    · cases h
  · cases h

section Check

variable (ctx : Ctx) (tc : TCtx) (nest : Equiv.Perm Name → SDFG → List SNode → Bool) (s : State)

/-- Match the image of node `n` at the head of `cs` (`rec` matches a scope
body): the full order nested in `n` and the rest of the list. -/
def matchOne (rec : TCtx → (Equiv.Perm Name → SDFG → List SNode → Bool) → List NodeId → List SNode → Option MatchRes)
    (n : DNode) (cs : List SNode) :
    Option (List NodeId × List SNode) :=
  match n.kind with
  | .access _ | .mapExit | .consumeExit =>
    if s.copyShape? ctx n = true then
      match takeCopies (s.copiesInto ctx n).length cs with
      | some (ps, rest) =>
        if keysCorr? ctx (ps.map (fun p => tc.copyKey p.1 p.2)) (s.copiesInto ctx n) = true then some ([], rest)
        else none
      | none => none
    else matchRefSet ctx s n cs
  | .tasklet t =>
    match cs with
    | .taskletNode t' ins outs :: rest =>
      if t' = t ∧ s.codeShape? n = true ∧ s.codeIns n = ins.pairs.map (fun p => tc.inKey p.1 p.2) ∧
          s.codeOuts n = outs.pairs.map (fun p => Memlet.outKey p.1 p.2) ∧ s.hasDynamicWcrOut n = false ∧
          outs.hasDynamicWcr = false then some ([], rest) else none
    | _ => none
  | .library l =>
    match cs with
    | .libraryCall l' _ ins outs :: rest =>
      if (l'.classpath = l.classpath ∧ l'.reduce = l.reduce) ∧ s.codeShape? n = true ∧ s.codeIns n = ins.pairs.map (fun p => tc.inKey p.1 p.2) ∧
          s.codeOuts n = outs.pairs.map (fun p => Memlet.outKey p.1 p.2) ∧ s.hasDynamicWcrOut n = false ∧
          outs.hasDynamicWcr = false then some ([], rest) else none
    | _ => none
  | .mapEntry mp =>
    match s.dynKeys? n with
    | none => none
    | some keys =>
      if dynOk? ctx keys = true then
        match takeDyn tc keys.length cs with
        | some (dks, cs') =>
          if dks = keys then
            match cs' with
            | .mapScope mp' body :: rest =>
              if mp' = mp then
                match nestedChild ctx tc nest s n body with
                | some fullN => some (fullN, rest)
                | none =>
                  if tc.privateTransients body = ctx.privateTransients s n.id then
                    match rec tc nest (s.directChildren (some n.id)) body with
                    | some (oM, fullM, []) =>
                      if s.isTopologicalOrder (s.scopeNodes n.id) fullM = true ∧
                          fullM.filter (s.directChildren (some n.id)).contains = oM then some (fullM, rest) else none
                    | _ => none
                  else none
              else none
            | _ => none
          else none
        | none => none
      else none
  | .consumeEntry c =>
    match s.dynKeys? n with
    | some [] =>
      match (s.inEdges n.id).find? (·.dstConn = some "IN_stream") with
      | some e =>
        match cs with
        | .dynScopeCopy t mIn :: .consumeScope c' body :: rest =>
          if t = "IN_stream" ∧ c' = c ∧ s.srcEndpoint e = .data mIn.data mIn.dataSubset ∧
              s.consumedStream? n.id = some mIn.data ∧ tc.privateTransients body = ctx.privateTransients s n.id then
            match rec (tc.consuming mIn.data) (fun _ _ _ => false) (s.directChildren (some n.id)) body with
            | some (oM, fullM, []) =>
              if s.isTopologicalOrder (s.scopeNodes n.id) fullM = true ∧
                  fullM.filter (s.directChildren (some n.id)).contains = oM then some (fullM, rest) else none
            | _ => none
          else none
        | _ => none
      | none => none
    | _ => none
  | .nested _ _ _ =>
    match nestedCall ctx tc nest s n cs with
    | some (_, rest) => some ([], rest)
    | none => none

/-- The unplaced nodes and the nodes nested in them. -/
def pendingAll (U : List NodeId) : List NodeId := U ++ U.flatMap s.scopeNodes

/-- All predecessors of `u` are placed. -/
def ready (U : List NodeId) (u : NodeId) : Bool := (s.inEdges u).all fun e => !(pendingAll s U).contains e.src

end Check

/-- Place the nodes `U` of a scope along the tree list `cs` (see the module
documentation); consume bodies are matched in the consuming context, without
nested SDFGs. -/
def matchSeg (ctx : Ctx) (tc : TCtx) (nest : Equiv.Perm Name → SDFG → List SNode → Bool) (s : State) :
    Nat → List NodeId → List SNode → Option MatchRes
  | 0, _, _ => none
  | fuel + 1, U, cs =>
    if U.isEmpty then some ([], [], cs)
    else
      match U.findSome? (fun u => if ready s U u then
          (s.node? u).bind fun n =>
            (matchOne ctx tc nest s (fun tc' nest' => matchSeg ctx tc' nest' s fuel) n cs).map fun p => (u, p)
          else none) with
      | some (u, fullN, rest₁) =>
        (matchSeg ctx tc nest s fuel (U.erase u) rest₁).map fun r => (u :: r.1, u :: fullN ++ r.2.1, r.2.2)
      | none => none

/-! ## Soundness of the matching -/

section Sound

variable {ctx : Ctx} {tc : TCtx} {nest : Equiv.Perm Name → SDFG → List SNode → Bool} {s : State}
  (hnest : ∀ σ inner body, nest σ inner body = true → InnerCorr tc σ inner body)
include hnest

theorem matchOne_sound
    {rec : TCtx → (Equiv.Perm Name → SDFG → List SNode → Bool) → List NodeId → List SNode → Option MatchRes}
    (hrec : ∀ tc' nest', (∀ σ inner body, nest' σ inner body = true → InnerCorr tc' σ inner body) →
      ∀ U cs o full rest, rec tc' nest' U cs = some (o, full, rest) → ∃ pre, cs = pre ++ rest ∧ SegCorr ctx tc' s o pre)
    {n : DNode} {cs : List SNode} {fullN : List NodeId} {rest : List SNode}
    (hm : matchOne ctx tc nest s rec n cs = some (fullN, rest)) :
    ∃ img, cs = img ++ rest ∧
      ∀ id o tail, s.node? id = some n → SegCorr ctx tc s o tail → SegCorr ctx tc s (id :: o) (img ++ tail) := by
  unfold matchOne at hm
  split at hm
  · rename_i d hk
    split at hm
    · rename_i hsh
      split at hm
      · rename_i ps rest' htk
        split at hm
        · rename_i hks
          simp only [Option.some.injEq, Prod.mk.injEq] at hm
          obtain ⟨-, rfl⟩ := hm
          refine ⟨ps.map (fun p => SNode.copyNode p.1 p.2), takeCopies_sound htk, fun id o tail hn ht => ?_⟩
          exact .copies hn (by simp [DNode.isCopyTarget, hk]) (State.copyShape?_sound hsh) (keysCorr?_sound hks) ht
        · cases hm
      · cases hm
    · exact matchRefSet_sound (by simp [DNode.isCopyTarget, hk]) hm
  · rename_i hk
    split at hm
    · rename_i hsh
      split at hm
      · rename_i ps rest' htk
        split at hm
        · rename_i hks
          simp only [Option.some.injEq, Prod.mk.injEq] at hm
          obtain ⟨-, rfl⟩ := hm
          refine ⟨ps.map (fun p => SNode.copyNode p.1 p.2), takeCopies_sound htk, fun id o tail hn ht => ?_⟩
          exact .copies hn (by simp [DNode.isCopyTarget, hk]) (State.copyShape?_sound hsh) (keysCorr?_sound hks) ht
        · cases hm
      · cases hm
    · exact matchRefSet_sound (by simp [DNode.isCopyTarget, hk]) hm
  · rename_i hk
    split at hm
    · rename_i hsh
      split at hm
      · rename_i ps rest' htk
        split at hm
        · rename_i hks
          simp only [Option.some.injEq, Prod.mk.injEq] at hm
          obtain ⟨-, rfl⟩ := hm
          refine ⟨ps.map (fun p => SNode.copyNode p.1 p.2), takeCopies_sound htk, fun id o tail hn ht => ?_⟩
          exact .copies hn (by simp [DNode.isCopyTarget, hk]) (State.copyShape?_sound hsh) (keysCorr?_sound hks) ht
        · cases hm
      · cases hm
    · exact matchRefSet_sound (by simp [DNode.isCopyTarget, hk]) hm
  · rename_i t hk
    split at hm
    · rename_i t' ins outs rest'
      split at hm
      · rename_i hc
        obtain ⟨rfl, hsh, hki, hko, hd, hdT⟩ := hc
        simp only [Option.some.injEq, Prod.mk.injEq] at hm
        obtain ⟨-, rfl⟩ := hm
        exact ⟨[_], rfl, fun id o tail hn ht => .tasklet hn hk (State.codeShape?_sound hsh) hki hko hd hdT ht⟩
      · cases hm
    · cases hm
  · rename_i l hk
    split at hm
    · rename_i l' props ins outs rest'
      split at hm
      · rename_i hc
        obtain ⟨hl, hsh, hki, hko, hd, hdT⟩ := hc
        simp only [Option.some.injEq, Prod.mk.injEq] at hm
        obtain ⟨-, rfl⟩ := hm
        exact ⟨[_], rfl, fun id o tail hn ht => .library hn hk hl (State.codeShape?_sound hsh) hki hko hd hdT ht⟩
      · cases hm
    · cases hm
  · rename_i mp hk
    split at hm
    · cases hm
    · rename_i keys hkeys
      split at hm
      · rename_i hok
        split at hm
        · rename_i dks cs' htk
          split at hm
          · rename_i hdks
            subst hdks
            split at hm
            · rename_i mp' body rest'
              split at hm
              · rename_i hmp
                subst hmp
                split at hm
                · rename_i fullN hnc
                  simp only [Option.some.injEq, Prod.mk.injEq] at hm
                  obtain ⟨-, rfl⟩ := hm
                  obtain ⟨ds, rfl, hds⟩ := takeDyn_sound htk
                  refine ⟨ds ++ [.mapScope mp' body], by simp, fun id o tail hn ht => ?_⟩
                  have := nestedChild_sound hnest hnc hn hk hkeys (dynOk?_sound hok) hds ht
                  simpa using this
                · split at hm
                  · rename_i hpriv
                    split at hm
                    · rename_i oM fullM hb
                      split at hm
                      · rename_i hto
                        simp only [Option.some.injEq, Prod.mk.injEq] at hm
                        obtain ⟨-, rfl⟩ := hm
                        obtain ⟨pre, hpre, hseg⟩ := hrec tc nest hnest _ _ _ _ _ hb
                        rw [List.append_nil] at hpre
                        subst hpre
                        obtain ⟨ds, rfl, hds⟩ := takeDyn_sound htk
                        refine ⟨ds ++ [.mapScope mp' body], by simp, fun id o tail hn ht => ?_⟩
                        have := SegCorr.map hn hk hkeys (dynOk?_sound hok) hds hto.1 hto.2 hpriv hseg ht
                        simpa using this
                      · cases hm
                    · cases hm
                  · cases hm
              · cases hm
            · cases hm
          · cases hm
        · cases hm
      · cases hm
  · rename_i c hk
    split at hm
    · rename_i hkeys
      split at hm
      · rename_i e hin
        split at hm
        · rename_i t mIn c' body rest'
          split at hm
          · rename_i hc
            obtain ⟨rfl, rfl, hsrc, hd, hpriv⟩ := hc
            split at hm
            · rename_i oM fullM hb
              split at hm
              · rename_i hto
                simp only [Option.some.injEq, Prod.mk.injEq] at hm
                obtain ⟨-, rfl⟩ := hm
                obtain ⟨pre, hpre, hseg⟩ := hrec _ _ (fun _ _ _ h => by cases h) _ _ _ _ _ hb
                rw [List.append_nil] at hpre
                subst hpre
                exact ⟨[.dynScopeCopy "IN_stream" mIn, .consumeScope c' body], rfl,
                  fun id o tail hn ht => .consume hn hk hkeys hin hsrc hd hto.1 hto.2 hpriv hseg ht⟩
              · cases hm
            · cases hm
          · cases hm
        · cases hm
      · cases hm
    · cases hm
  · split at hm
    · rename_i body rest' hnc
      simp only [Option.some.injEq, Prod.mk.injEq] at hm
      obtain ⟨-, rfl⟩ := hm
      obtain ⟨hcs, hseg⟩ := nestedCall_sound hnest hnc
      exact ⟨body, hcs, hseg⟩
    · cases hm

end Sound

theorem matchSeg_sound {ctx : Ctx} {s : State} : ∀ (fuel : Nat) {tc : TCtx}
    {nest : Equiv.Perm Name → SDFG → List SNode → Bool},
    (∀ σ inner body, nest σ inner body = true → InnerCorr tc σ inner body) →
    ∀ (U : List NodeId) (cs : List SNode) (o full : List NodeId) (rest : List SNode),
    matchSeg ctx tc nest s fuel U cs = some (o, full, rest) → ∃ pre, cs = pre ++ rest ∧ SegCorr ctx tc s o pre
  | 0, _, _, _, _, _, _, _, _, h => by simp [matchSeg] at h
  | fuel + 1, tc, nest, hnest, U, cs, o, full, rest, h => by
    unfold matchSeg at h
    split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -, rfl⟩ := h
      exact ⟨[], rfl, .nil⟩
    · split at h
      · rename_i u fullN rest₁ hf
        obtain ⟨u', -, hu'⟩ := List.exists_of_findSome?_eq_some hf
        split at hu'
        · obtain ⟨n, hn, hm⟩ := Option.bind_eq_some_iff.1 hu'
          obtain ⟨⟨fN, r₁⟩, hm', he⟩ := Option.map_eq_some_iff.1 hm
          simp only [Prod.mk.injEq] at he
          obtain ⟨rfl, rfl, rfl⟩ := he
          obtain ⟨img, hcs, hcons⟩ := matchOne_sound hnest (fun _ _ hn' => matchSeg_sound fuel hn') hm'
          obtain ⟨⟨o', full', rest'⟩, hr, he⟩ := Option.map_eq_some_iff.1 h
          simp only [Prod.mk.injEq] at he
          obtain ⟨rfl, -, rfl⟩ := he
          obtain ⟨pre, hpre, hseg⟩ := matchSeg_sound fuel hnest _ _ _ _ _ hr
          refine ⟨img ++ pre, by rw [hcs, hpre, List.append_assoc], hcons _ _ _ hn hseg⟩
        · cases hu'
      · cases h

/-! ## States, the chain and the program -/

/-- Take the view nodes of the keys `ks` from the head of `cs` (`ViewsCorr`). -/
def takeViews (ctx : Ctx) : List ViewKey → List SNode → Option (List SNode)
  | [], cs => some cs
  | k :: ks, .viewNode d base m sd vd :: cs =>
    if d = k.d ∧ base = k.base ∧ m.data = k.base ∧ subCorr? ctx k.base m.dataSubset k.sub = true then
      takeViews ctx ks cs
    else none
  | _ :: _, _ => none

theorem takeViews_sound {ctx : Ctx} : ∀ {ks : List ViewKey} {cs rest : List SNode}, takeViews ctx ks cs = some rest →
    ∃ vs, cs = vs ++ rest ∧ ViewsCorr ctx ks vs
  | [], cs, rest, h => by simp only [takeViews, Option.some.injEq] at h; subst h; exact ⟨[], rfl, .nil⟩
  | k :: ks, c :: cs, rest, h => by
    cases c with
    | viewNode d base m sd vd =>
      simp only [takeViews] at h
      split at h
      · rename_i hc
        obtain ⟨rfl, rfl, hmd, hms⟩ := hc
        obtain ⟨vs, rfl, hvs⟩ := takeViews_sound h
        exact ⟨_ :: vs, rfl, .cons ⟨m, sd, vd, rfl, hmd, subCorr?_sound hms⟩ hvs⟩
      · cases h
    | _ => simp [takeViews] at h
  | _ :: _, [], rest, h => by simp [takeViews] at h

/-- Check one state against the head of the tree list; the rest of the list. -/
def stateCheck (ctx : Ctx) (tc : TCtx) (nest : Equiv.Perm Name → SDFG → List SNode → Bool) (s : State) (cs : List SNode) :
    Except String (List SNode) :=
  if s.scopeViewsFree? ctx = false then .error s!"state '{s.label}': views inside scopes are not matched yet"
  else if !(ctx.stateTransients s).all (fun t => !ctx.isStream t) then
    .error s!"state '{s.label}': allocates a transient stream"
  else
    match s.components with
    | [] => .ok cs
    | [c] =>
      match s.viewKeys ctx (s.viewOrder ctx (s.topLevelOf c)) with
      | none => .error s!"state '{s.label}': a view edge cannot be resolved"
      | some ks =>
        match takeViews ctx ks cs with
        | none => .error s!"state '{s.label}': the view nodes differ"
        | some cs' =>
          match matchSeg ctx tc nest s (2 * s.nodes.length + 2) (s.topLevelOf c) cs' with
          | some (o, full, rest) =>
            if s.isTopologicalOrder c full = true ∧ full.filter (s.topLevelOf c).contains = o then .ok rest
            else .error s!"state '{s.label}': the matched order is not topological"
          | none => .error s!"state '{s.label}': no tree segment matches"
    | _ => .error s!"state '{s.label}': several components"

theorem stateCheck_sound {ctx : Ctx} {tc : TCtx} {nest : Equiv.Perm Name → SDFG → List SNode → Bool} {s : State} {cs rest : List SNode}
    (hnest : ∀ σ inner body, nest σ inner body = true → InnerCorr tc σ inner body)
    (h : stateCheck ctx tc nest s cs = .ok rest) : ∃ pre, cs = pre ++ rest ∧ StateCorr ctx tc s pre := by
  unfold stateCheck at h
  split at h
  · cases h
  · rename_i hp
    have hpure : s.scopeViewsFree? ctx = true := by simpa using hp
    split at h
    · cases h
    rename_i hns
    have hns' : ∀ t ∈ ctx.stateTransients s, ctx.isStream t = false := by
      simpa using hns
    split at h
    · rename_i hc
      cases h
      exact ⟨[], rfl, hpure, hns', .inl ⟨hc, rfl⟩⟩
    · rename_i c hc
      split at h
      · cases h
      · rename_i ks hks
        split at h
        · cases h
        · rename_i cs' hv
          obtain ⟨vs, rfl, hvs⟩ := takeViews_sound hv
          split at h
          · rename_i o full rest' hm
            split at h
            · rename_i hto
              cases h
              obtain ⟨pre, hpre, hseg⟩ := matchSeg_sound _ hnest _ _ _ _ _ hm
              subst hpre
              exact ⟨vs ++ pre, by simp, hpure, hns', .inr ⟨c, o, full, ks, vs, pre, hc, hks, hvs, rfl, hseg, hto.1, hto.2⟩⟩
            · cases h
          · cases h
    · cases h

mutual

/-- The size of a region (a bound on the checker's recursion). -/
def Region.size : Region → Nat
  | ⟨_, _, blocks, _, _⟩ => Blocks.size blocks + 1

def Blocks.size : List Block → Nat
  | [] => 0
  | b :: bs => Block.size b + Blocks.size bs

def Block.size : Block → Nat
  | .region r => Region.size r + 1
  | .loop ⟨_, _, _, _, _, _, _, body⟩ => Region.size body + 1
  | .cond _ brs => Branches.size brs + 1
  | _ => 1

def Branches.size : List (Option Expr × Region) → Nat
  | [] => 0
  | (_, r) :: rest => Region.size r + Branches.size rest

end

/-- Decision procedure for `GEnd`. -/
def gEnd? (rest : List SNode) : Bool :=
  match rest with
  | [] => true
  | [.stateLabel _] => true
  | .goto (some l) :: tl => match afterLabel l tl with
    | some [] => true
    | _ => false
  | _ => false

theorem gEnd?_sound {rest : List SNode} (h : gEnd? rest = true) : GEnd rest := by
  unfold gEnd? at h
  split at h
  · exact .inl rfl
  · exact .inr (.inl ⟨_, rfl⟩)
  · rename_i l tl
    split at h
    · rename_i hl; exact .inr (.inr ⟨l, tl, rfl, hl⟩)
    · cases h
  · cases h

section Check

variable (ctx : Ctx) (tc : TCtx) (nest : Equiv.Perm Name → SDFG → List SNode → Bool)

mutual

/-- Check the region `rg` from block `bid` against the head of the tree list;
the rest of the list. -/
def regCheck : Nat → Region → BlockId → List SNode → Except String (List SNode)
  | 0, _, _, _ => .error "control flow too deep or cyclic"
  | f + 1, rg, bid, cs =>
    match cs with
    | .gblock (.stateLabel L :: p) :: rest =>
      if bid = rg.start then
        match gposCheck f rg (.stateLabel L :: p) rg.start p with
        | .ok () => .ok rest
        | .error m => .error m
      else .error s!"general block at block {bid}, not at the start of the region"
    | _ =>
    match rg.block? bid with
    | none => .error s!"block {bid} out of range"
    | some b =>
      match blockCheck f b cs with
      | .error m => .error m
      | .ok rest =>
        match rg.outEdges bid with
        | [] => .ok rest
        | [e] =>
          if e.isUnconditional = true ∧ headOk? rest = true then
            match takeAssignNodes e.assigns.length rest with
            | some (as, rest') =>
              if as.isPerm e.assigns = true ∧ seqAssigns? ctx as = true then regCheck f rg e.dst rest'
              else .error s!"interstate edge from block {bid}: assignments differ or cannot run one by one"
            | none => .error s!"interstate edge from block {bid}: assignments do not match"
          else
            -- a conditional edge: an `IfScope` holding the assignments and the rest of the chain
            match rest with
            | .ifScope pc body :: rest' =>
              if pc.expr = e.cond then
                match takeAssignNodes e.assigns.length body with
                | some (as, body') =>
                  if as.isPerm e.assigns = true ∧ seqAssigns? ctx as = true then
                    match regCheck f rg e.dst body' with
                    | .ok [] => .ok rest'
                    | .ok _ => .error s!"conditional edge from block {bid}: tree nodes after the chain"
                    | .error m => .error m
                  else .error s!"conditional edge from block {bid}: assignments differ or cannot run one by one"
                | none => .error s!"conditional edge from block {bid}: assignments do not match"
              else .error s!"conditional edge from block {bid}: the if condition differs"
            | _ => .error s!"interstate edge from block {bid}: conditional, no if scope in the tree"
        | _ => .error s!"block {bid}: several out-edges (GBlocks are M5b step 5)"

/-- Check one block against the head of the tree list; the rest of the list. -/
def blockCheck : Nat → Block → List SNode → Except String (List SNode)
  | 0, _, _ => .error "control flow too deep"
  | f + 1, b, cs =>
    match b with
    | .state s => stateCheck ctx tc nest s cs
    | .loop l =>
      match cs with
      | n :: rest =>
        match n.loop? with
        | some (lm, body) =>
          if loopCorr? l lm = true then
            match regCheck f l.body l.body.start body with
            | .ok [] => .ok rest
            | .ok _ => .error s!"loop '{l.label}': tree nodes after the body"
            | .error m => .error m
          else .error s!"loop '{l.label}': loop properties differ"
        | none => .error s!"loop '{l.label}': no loop scope in the tree"
      | [] => .error s!"loop '{l.label}': no loop scope in the tree"
    | .cond _ brs =>
      match brs with
      | (some c, rg) :: brs' =>
        match cs with
        | .ifScope pc body :: tl =>
          if pc.expr = c then
            match regCheck f rg rg.start body with
            | .ok [] => elifCheck f brs' tl
            | .ok _ => .error "conditional block: tree nodes after a branch"
            | .error m => .error m
          else .error "conditional block: the if condition differs"
        | _ => .error "conditional block: no if scope in the tree"
      | _ => .error "conditional block: no leading if branch"
    | .breakBlock =>
      match cs with
      | .breakNode :: rest => .ok rest
      | _ => .error "break: no break node in the tree"
    | .continueBlock =>
      match cs with
      | .continueNode :: rest => .ok rest
      | _ => .error "continue: no continue node in the tree"
    | .returnBlock =>
      match cs with
      | .goto none :: rest => .ok rest
      | _ => .error "return: no exit goto in the tree"
    | .region rg =>
      match rg.kind with
      | .plain | .functionCall _ => regCheck f rg rg.start cs
      | .named label =>
        match cs with
        | .namedRegion label' body :: rest =>
          if label' = label then
            match regCheck f rg rg.start body with
            | .ok [] => .ok rest
            | .ok _ => .error s!"named region '{label}': tree nodes after the body"
            | .error m => .error m
          else .error s!"named region '{label}': the label differs"
        | _ => .error s!"named region '{label}': no named region scope in the tree"
      | .unstructured => regCheck f rg rg.start cs

/-- Check the `elif`/`else` branches of a conditional block. -/
def elifCheck : Nat → List (Option Expr × Region) → List SNode → Except String (List SNode)
  | 0, _, _ => .error "control flow too deep"
  | f + 1, brs, cs =>
    match brs with
    | [] => .ok cs
    | (some c, rg) :: brs' =>
      match cs with
      | .elifScope pc body :: tl =>
        if pc.expr = c then
          match regCheck f rg rg.start body with
          | .ok [] => elifCheck f brs' tl
          | .ok _ => .error "conditional block: tree nodes after a branch"
          | .error m => .error m
        else .error "conditional block: an elif condition differs"
      | _ => .error "conditional block: no elif scope in the tree"
    | (none, rg) :: more =>
      match more with
      | [] =>
        match cs with
        | .elseScope body :: tl =>
          match regCheck f rg rg.start body with
          | .ok [] => .ok tl
          | .ok _ => .error "conditional block: tree nodes after the else branch"
          | .error m => .error m
        | _ => .error "conditional block: no else scope in the tree"
      | _ => .error "conditional block: else is not the last branch"

/-- Check the position `p` of block `bid` in the general block `full`
(`GPos`). -/
def gposCheck : Nat → Region → List SNode → BlockId → List SNode → Except String Unit
  | 0, _, _, _, _ => .error "general block too deep (or a backward goto)"
  | f + 1, rg, full, bid, p =>
    match rg.block? bid with
    | none => .error s!"block {bid} out of range"
    | some b =>
      match blockCheck f b p with
      | .error m => .error m
      | .ok rest₁ =>
        if headOk? rest₁ = true then
          match itemsCheck f rg full (rg.orderedOut bid) rest₁ with
          | .ok rest =>
            if (rg.outEdges bid).all (fun e => !e.isUnconditional) = true → gEnd? rest = true then .ok ()
            else .error s!"general block: block {bid} does not end the block when no edge is taken"
          | .error m => .error m
        else .error s!"general block: block {bid} is followed by elif/else"

/-- Check the out-edge items of a general block (`GItems`); the rest of the
list. -/
def itemsCheck : Nat → Region → List SNode → List IEdge → List SNode → Except String (List SNode)
  | 0, _, _, _, _ => .error "general block too deep"
  | _ + 1, _, _, [], cs => .ok cs
  | f + 1, rg, full, e :: es, cs =>
    if e.isUnconditional = true then
      match takeAssignNodes e.assigns.length cs with
      | some (as, .goto (some L) :: tl) =>
        if as.isPerm e.assigns = true ∧ seqAssigns? ctx as = true then
          match afterLabel L tl with
          | some p =>
            match gposCheck f rg full e.dst p with
            | .ok () => itemsCheck f rg full es tl
            | .error m => .error m
          | none => .error s!"general block: goto '{L}' does not jump forward"
        else .error "general block: edge assignments differ or cannot run one by one"
      | _ => .error "general block: no goto for an unconditional edge"
    else
      match cs with
      | .stateIfScope pc body :: tl =>
        if pc.expr = e.cond then
          match takeAssignNodes e.assigns.length body with
          | some (as, [.goto (some L)]) =>
            if as.isPerm e.assigns = true ∧ seqAssigns? ctx as = true then
              match afterLabel L tl with
              | some p =>
                match gposCheck f rg full e.dst p with
                | .ok () => itemsCheck f rg full es tl
                | .error m => .error m
              | none => .error s!"general block: goto '{L}' does not jump forward"
            else .error "general block: edge assignments differ or cannot run one by one"
          | _ => .error "general block: a stateif body is not assignments and a goto"
        else .error "general block: a stateif condition differs"
      | _ => .error "general block: no stateif for a conditional edge"

end

end Check

section CheckSound

variable {ctx : Ctx} {tc : TCtx} {nest : Equiv.Perm Name → SDFG → List SNode → Bool}
  (hnest : ∀ σ inner body, nest σ inner body = true → InnerCorr tc σ inner body)
include hnest

mutual

theorem regCheck_sound : ∀ (f : Nat) (rg : Region) (bid : BlockId) (cs rest : List SNode),
    regCheck ctx tc nest f rg bid cs = .ok rest → ∃ pre, cs = pre ++ rest ∧ RegCorr ctx tc rg bid pre
  | 0, _, _, _, _, h => by simp [regCheck] at h
  | f + 1, rg, bid, cs, rest, h => by
    unfold regCheck at h
    split at h
    · rename_i L p rest'
      split at h
      · rename_i hbid
        split at h
        · rename_i hg
          cases h
          subst hbid
          exact ⟨[.gblock (.stateLabel L :: p)], rfl, .gblock (gposCheck_sound f _ _ _ _ hg)⟩
        · cases h
      · cases h
    · split at h
      · cases h
      · rename_i b hb
        split at h
        · cases h
        · rename_i restB hbc
          obtain ⟨pre, rfl, hpre⟩ := blockCheck_sound f b cs restB hbc
          split at h
          · rename_i hout
            cases h
            exact ⟨pre, rfl, .last hb hpre hout⟩
          · rename_i e hout
            split at h
            · rename_i hc
              obtain ⟨hu, hh⟩ := hc
              split at h
              · rename_i as rest' ht
                split at h
                · rename_i hc'
                  obtain ⟨hp, hs⟩ := hc'
                  obtain ⟨ns, rfl, hns⟩ := takeAssignNodes_sound ht
                  obtain ⟨pre₂, rfl, hpre₂⟩ := regCheck_sound f rg e.dst rest' rest h
                  refine ⟨pre ++ (ns ++ pre₂), by simp only [List.append_assoc], ?_⟩
                  exact .step hb hpre hout hu (List.isPerm_iff.1 hp) hns (seqAssigns?_sound hs)
                    (HeadOk.of_append (l := ns ++ pre₂) (rest := rest) (by
                      have := headOk?_sound hh; simpa only [List.append_assoc] using this)) hpre₂
                · cases h
              · cases h
            · split at h
              · rename_i pc body rest' _
                split at h
                · rename_i hpc
                  split at h
                  · rename_i as body' ht
                    split at h
                    · rename_i hc'
                      obtain ⟨hp, hs⟩ := hc'
                      obtain ⟨ns, rfl, hns⟩ := takeAssignNodes_sound ht
                      split at h
                      · rename_i hr
                        cases h
                        obtain ⟨pre₂, hpre₂, hc₂⟩ := regCheck_sound f rg e.dst _ _ hr
                        rw [List.append_nil] at hpre₂
                        subst hpre₂
                        refine ⟨pre ++ [.ifScope _ (ns ++ body')], by rw [List.append_assoc, List.singleton_append], ?_⟩
                        exact .condStep hb hpre hout hpc (List.isPerm_iff.1 hp) hns (seqAssigns?_sound hs) hc₂
                      · cases h
                      · cases h
                    · cases h
                  · cases h
                · cases h
              · cases h
          · cases h

theorem blockCheck_sound : ∀ (f : Nat) (b : Block) (cs rest : List SNode),
    blockCheck ctx tc nest f b cs = .ok rest → ∃ pre, cs = pre ++ rest ∧ BlockCorr ctx tc b pre
  | 0, _, _, _, h => by simp [blockCheck] at h
  | f + 1, b, cs, rest, h => by
    simp only [blockCheck] at h
    split at h
    · obtain ⟨pre, hpre, hsc⟩ := stateCheck_sound hnest h
      exact ⟨pre, hpre, .state hsc⟩
    · split at h
      · rename_i n rest'
        split at h
        · rename_i lm body hn
          split at h
          · rename_i hl
            split at h
            · rename_i hr
              cases h
              obtain ⟨pre, hpre, hc⟩ := regCheck_sound f _ _ _ _ hr
              rw [List.append_nil] at hpre
              subst hpre
              exact ⟨[n], rfl, .loop hn (loopCorr?_sound hl) hc⟩
            · cases h
            · cases h
          · cases h
        · cases h
      · cases h
    · split at h
      · split at h
        · rename_i c rg brs' pc body tl
          split at h
          · rename_i hc
            split at h
            · rename_i hr
              obtain ⟨pre, rfl, hpre⟩ := elifCheck_sound f _ _ rest h
              obtain ⟨preB, hpreB, hcB⟩ := regCheck_sound f _ _ _ _ hr
              rw [List.append_nil] at hpreB
              subst hpreB
              exact ⟨_, rfl, .cond hc hcB hpre⟩
            · cases h
            · cases h
          · cases h
        · cases h
      · cases h
    · split at h
      · cases h; exact ⟨_, rfl, .brk⟩
      · cases h
    · split at h
      · cases h; exact ⟨_, rfl, .cont⟩
      · cases h
    · split at h
      · cases h; exact ⟨_, rfl, .ret⟩
      · cases h
    · split at h
      · obtain ⟨pre, hpre, hc⟩ := regCheck_sound f _ _ _ _ h
        exact ⟨pre, hpre, .flat hc⟩
      · obtain ⟨pre, hpre, hc⟩ := regCheck_sound f _ _ _ _ h
        exact ⟨pre, hpre, .flat hc⟩
      · split at h
        · rename_i label' body rest'
          split at h
          · rename_i hl
            subst hl
            split at h
            · rename_i hr
              cases h
              obtain ⟨pre, hpre, hc⟩ := regCheck_sound f _ _ _ _ hr
              rw [List.append_nil] at hpre
              subst hpre
              exact ⟨_, rfl, .named hc⟩
            · cases h
            · cases h
          · cases h
        · cases h
      · obtain ⟨pre, hpre, hc⟩ := regCheck_sound f _ _ _ _ h
        exact ⟨pre, hpre, .flat hc⟩

theorem elifCheck_sound : ∀ (f : Nat) (brs : List (Option Expr × Region)) (cs rest : List SNode),
    elifCheck ctx tc nest f brs cs = .ok rest → ∃ pre, cs = pre ++ rest ∧ ElifCorr ctx tc brs pre
  | 0, _, _, _, h => by simp [elifCheck] at h
  | f + 1, brs, cs, rest, h => by
    simp only [elifCheck] at h
    split at h
    · cases h; exact ⟨[], rfl, .nil⟩
    · split at h
      · rename_i c rg brs' pc body tl
        split at h
        · rename_i hc
          split at h
          · rename_i hr
            obtain ⟨pre, rfl, hpre⟩ := elifCheck_sound f _ _ rest h
            obtain ⟨preB, hpreB, hcB⟩ := regCheck_sound f _ _ _ _ hr
            rw [List.append_nil] at hpreB
            subst hpreB
            exact ⟨_, rfl, .elif hc hcB hpre⟩
          · cases h
          · cases h
        · cases h
      · cases h
    · split at h
      · split at h
        · rename_i rg body tl
          split at h
          · rename_i hr
            cases h
            obtain ⟨preB, hpreB, hcB⟩ := regCheck_sound f _ _ _ _ hr
            rw [List.append_nil] at hpreB
            subst hpreB
            exact ⟨_, rfl, .els hcB⟩
          · cases h
          · cases h
        · cases h
      · cases h

theorem gposCheck_sound : ∀ (f : Nat) (rg : Region) (full : List SNode) (bid : BlockId) (p : List SNode),
    gposCheck ctx tc nest f rg full bid p = .ok () → GPos ctx tc rg full bid p
  | 0, _, _, _, _, h => by simp [gposCheck] at h
  | f + 1, rg, full, bid, p, h => by
    unfold gposCheck at h
    split at h
    · cases h
    · rename_i b hb
      split at h
      · cases h
      · rename_i rest₁ hbc
        obtain ⟨pre, rfl, hpre⟩ := blockCheck_sound f b p rest₁ hbc
        split at h
        · rename_i hh
          split at h
          · rename_i rest hi
            obtain ⟨items, rfl, hitems⟩ := itemsCheck_sound f rg full _ _ _ hi
            split at h
            · rename_i hend
              exact .mk hb hpre (headOk?_sound hh) hitems fun hall => gEnd?_sound (hend hall)
            · cases h
          · cases h
        · cases h

theorem itemsCheck_sound : ∀ (f : Nat) (rg : Region) (full : List SNode) (es : List IEdge) (cs rest : List SNode),
    itemsCheck ctx tc nest f rg full es cs = .ok rest → ∃ items, cs = items ++ rest ∧ GItems ctx tc rg full es items rest
  | 0, _, _, _, _, _, h => by simp [itemsCheck] at h
  | _ + 1, _, _, [], cs, rest, h => by
    simp only [itemsCheck] at h
    cases h
    exact ⟨[], rfl, .nil⟩
  | f + 1, rg, full, e :: es, cs, rest, h => by
    unfold itemsCheck at h
    split at h
    · rename_i hu
      split at h
      · rename_i as L tl ht
        split at h
        · rename_i hc
          obtain ⟨hp, hs⟩ := hc
          split at h
          · rename_i p hl
            split at h
            · rename_i hg
              obtain ⟨items, rfl, hitems⟩ := itemsCheck_sound f rg full es tl rest h
              obtain ⟨ns, rfl, hns⟩ := takeAssignNodes_sound ht
              exact ⟨ns ++ .goto (some L) :: items, by simp,
                .uncond hu (List.isPerm_iff.1 hp) hns (seqAssigns?_sound hs) hl (gposCheck_sound f _ _ _ _ hg) hitems⟩
            · cases h
          · cases h
        · cases h
      · cases h
    · rename_i hu
      split at h
      · rename_i pc body tl
        split at h
        · rename_i hpc
          split at h
          · rename_i as L ht
            split at h
            · rename_i hc
              obtain ⟨hp, hs⟩ := hc
              split at h
              · rename_i p hl
                split at h
                · rename_i hg
                  obtain ⟨items, rfl, hitems⟩ := itemsCheck_sound f rg full es tl rest h
                  obtain ⟨ns, rfl, hns⟩ := takeAssignNodes_sound ht
                  exact ⟨.stateIfScope pc (ns ++ [.goto (some L)]) :: items, rfl,
                    .cond (by simpa using hu) hpc (List.isPerm_iff.1 hp) hns (seqAssigns?_sound hs) hl
                      (gposCheck_sound f _ _ _ _ hg) hitems⟩
                · cases h
              · cases h
            · cases h
          · cases h
        · cases h
      · cases h

end

end CheckSound

/-! ## Independence of the interpretation -/

section WithI

variable {ctx : Ctx} {tc : TCtx} {s : State} (I₂ : Interp)

theorem CopyShape.withI {n : DNode} (h : s.CopyShape ctx n) : s.CopyShape { ctx with I := I₂ } n :=
  ⟨h.noRef, h.dst⟩

theorem SeqPlain.withI (h : SeqPlain ctx s) : SeqPlain { ctx with I := I₂ } s :=
  ⟨h.plain, h.shapes, h.memlets, h.ranges, h.reduces, h.codeOut⟩

theorem SegCorr.withI {o : List NodeId} {cs : List SNode} (h : SegCorr ctx tc s o cs) :
    SegCorr { ctx with I := I₂ } (tc.withI I₂) s o cs := by
  induction h with
  | nil => exact .nil
  | copies hn hk hsh hks _ ih => exact .copies hn hk (CopyShape.withI I₂ hsh) hks ih
  | refSet hn hk hr hm hsub _ ih => exact .refSet hn hk hr hm hsub ih
  | refSetCode hn hk hr _ ih => exact .refSetCode hn hk hr ih
  | tasklet hn hk hsh hki hko hd hdT _ ih => exact .tasklet hn hk hsh hki hko hd hdT ih
  | library hn hk hl hsh hki hko hd hdT _ ih => exact .library hn hk hl hsh hki hko hd hdT ih
  | map hn hk hkeys hok hds hto hdir hpriv _ _ ihb ih => exact .map hn hk hkeys hok hds hto hdir hpriv ihb ih
  | consume hn hk hkeys hin hsrc hd hto hdir hpriv _ _ ihb ih =>
    exact .consume hn hk hkeys hin hsrc hd hto hdir hpriv ihb ih
  | nestedMap hn hk hkeys hok hds hto hdir hxs hnd hkn hb hmap hcode hout hpriv hnames hinner _ ih =>
    exact .nestedMap hn hk hkeys hok hds hto hdir
      (fun x hx => let ⟨xn, h₁, h₂, h₃, h₄⟩ := hxs x hx; ⟨xn, h₁, h₂, CopyShape.withI I₂ h₃, h₄⟩)
      hnd hkn hb ⟨hmap.ident, hmap.consts⟩ hcode hout hpriv hnames hinner ih
  | nested hnd hkn hb hmap hcode hout hctr hshape hnames hinner _ ih =>
    exact .nested hnd hkn hb ⟨hmap.ident, hmap.consts⟩ hcode hout hctr hshape hnames hinner ih

theorem StateCorr.withI {cs : List SNode} (h : StateCorr ctx tc s cs) :
    StateCorr { ctx with I := I₂ } (tc.withI I₂) s cs := by
  refine ⟨h.scopeViews, h.noStreams, ?_⟩
  rcases h.comps with h₁ | ⟨c, o, full, ks, vs, seg, hc, hks, hvs, hcs, hseg, hto, hdir⟩
  · exact .inl h₁
  · exact .inr ⟨c, o, full, ks, vs, seg, hc, by rw [State.viewKeys_withI]; exact hks, hvs, hcs,
      SegCorr.withI I₂ hseg, hto, hdir⟩

mutual

theorem RegCorr.withI {rg : Region} {bid : BlockId} {cs : List SNode} :
    RegCorr ctx tc rg bid cs → RegCorr { ctx with I := I₂ } (tc.withI I₂) rg bid cs
  | .last hb hc he => .last hb (BlockCorr.withI hc) he
  | .step hb hc he hu hp ha hs hh h =>
    .step hb (BlockCorr.withI hc) he hu hp ha (SeqAssigns.of_g (c₁ := ctx) (c₂ := { ctx with I := I₂ }) rfl hs) hh
      (RegCorr.withI h)
  | .condStep hb hc he hpc hp ha hs h =>
    .condStep hb (BlockCorr.withI hc) he hpc hp ha (SeqAssigns.of_g (c₁ := ctx) (c₂ := { ctx with I := I₂ }) rfl hs)
      (RegCorr.withI h)
  | .gblock h => .gblock (GPos.withI h)

theorem BlockCorr.withI {b : Block} {cs : List SNode} :
    BlockCorr ctx tc b cs → BlockCorr { ctx with I := I₂ } (tc.withI I₂) b cs
  | .state h => .state (StateCorr.withI I₂ h)
  | .loop hn hm hb => .loop hn hm (RegCorr.withI hb)
  | .cond hc hb ht => .cond hc (RegCorr.withI hb) (ElifCorr.withI ht)
  | .brk => .brk
  | .cont => .cont
  | .ret => .ret
  | .flat h => .flat (RegCorr.withI h)
  | .named h => .named (RegCorr.withI h)

theorem GPos.withI {rg : Region} {full : List SNode} {bid : BlockId} {p : List SNode} :
    GPos ctx tc rg full bid p → GPos { ctx with I := I₂ } (tc.withI I₂) rg full bid p
  | .mk hb hc hh hi hend => .mk hb (BlockCorr.withI hc) hh (GItems.withI hi) hend

theorem GItems.withI {rg : Region} {full : List SNode} {es : List IEdge} {items rest : List SNode} :
    GItems ctx tc rg full es items rest → GItems { ctx with I := I₂ } (tc.withI I₂) rg full es items rest
  | .nil => .nil
  | .cond hu hpc hp ha hs hl hpos h =>
    .cond hu hpc hp ha (SeqAssigns.of_g (c₁ := ctx) (c₂ := { ctx with I := I₂ }) rfl hs) hl (GPos.withI hpos)
      (GItems.withI h)
  | .uncond hu hp ha hs hl hpos h =>
    .uncond hu hp ha (SeqAssigns.of_g (c₁ := ctx) (c₂ := { ctx with I := I₂ }) rfl hs) hl (GPos.withI hpos)
      (GItems.withI h)

theorem ElifCorr.withI {brs : List (Option Expr × Region)} {cs : List SNode} :
    ElifCorr ctx tc brs cs → ElifCorr { ctx with I := I₂ } (tc.withI I₂) brs cs
  | .nil => .nil
  | .elif hc hb ht => .elif hc (RegCorr.withI hb) (ElifCorr.withI ht)
  | .els hb => .els (RegCorr.withI hb)

end

end WithI

/-! ## Nested SDFGs: the inlined inner region and the renaming

A map scope whose body is one nested SDFG (followed by exit nodes without
copies) is matched against a tree map scope whose body is the inlined inner
top region (`nestedChild`, `SegCorr.nestedMap`): identity symbol mapping, no
inner constants, data connectors only, inner states without nested SDFGs
(`innerNames?`), the inner top region a chain of states (`innerCheck`), the
tree's private transients of the map the SDFG's plus the per-call transients.
The program runs in the flattened context `flatCtx … (corrPerm g)`, which
renames the per-call memory names of nested SDFGs to the tree's names; the
checker verifies that the renaming fixes every name of the SDFG's own states
(`outerNames?`), the interface, and that renamed names are undefined at the
start (`ProgCorrWith`). -/

theorem StateExec.normal {ctx : Ctx} {cst : CSt} {s : State} {ctl : Ctl} {c : CSt}
    (h : StateExec ctx cst s (.ok (ctl, c))) : ctl = .normal := by
  generalize hr : (Res.ok (ctl, c) : Res (Ctl × CSt)) = r at h
  cases h with
  | compsErr => cases hr
  | run _ =>
    unfold stateFinish at hr
    split at hr
    · simp only [Res.ok.injEq, Prod.mk.injEq] at hr
      exact hr.1
    · cases hr

mutual

/-- Can a `break`/`continue` leave the region (one outside loops)? -/
def _root_.Dace.Region.escapes : Region → Bool
  | ⟨_, _, blocks, _, _⟩ => Blocks.escapes blocks

def _root_.Dace.Blocks.escapes : List Block → Bool
  | [] => false
  | b :: bs => Block.escapes b || Blocks.escapes bs

def _root_.Dace.Block.escapes : Block → Bool
  | .breakBlock | .continueBlock => true
  | .region r => Region.escapes r
  | .cond _ brs => Branches.escapes brs
  | _ => false

def _root_.Dace.Branches.escapes : List (Option Expr × Region) → Bool
  | [] => false
  | (_, r) :: rest => Region.escapes r || Branches.escapes rest

end

theorem _root_.Dace.Region.escapes_eq (rg : Region) : rg.escapes = Blocks.escapes rg.blocks := by cases rg; rfl

theorem _root_.Dace.Blocks.escapes_mem {b : Block} : ∀ {bs : List Block}, Blocks.escapes bs = false → b ∈ bs → b.escapes = false
  | [], _, hb => by cases hb
  | b' :: bs, h, hb => by
    simp only [Blocks.escapes, Bool.or_eq_false_iff] at h
    rcases List.mem_cons.1 hb with rfl | hb
    · exact h.1
    · exact Blocks.escapes_mem h.2 hb

mutual

/-- A loop ends normally or returns. -/
theorem LoopExec.ctl {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopExec ctx cst l r → ∀ ctl c, r = .ok (ctl, c) → ctl = .normal ∨ ctl = .ret
  | .initErr _, _, _, hr => by cases hr
  | .guardFirst _ _ hg, ctl, c, hr => LoopGuard.ctl hg ctl c hr
  | .bodyFirst _ _ hb, ctl, c, hr => LoopIter.ctl hb ctl c hr

theorem LoopGuard.ctl {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopGuard ctx cst l r → ∀ ctl c, r = .ok (ctl, c) → ctl = .normal ∨ ctl = .ret
  | .condErr _, _, _, hr => by cases hr
  | .exit _, _, _, hr => by cases hr; exact .inl rfl
  | .enter _ hb, ctl, c, hr => LoopIter.ctl hb ctl c hr

theorem LoopIter.ctl {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopIter ctx cst l r → ∀ ctl c, r = .ok (ctl, c) → ctl = .normal ∨ ctl = .ret
  | .bodyErr _, _, _, hr => by cases hr
  | .ret _, _, _, hr => by cases hr; exact .inr rfl
  | .brk _, _, _, hr => by cases hr; exact .inl rfl
  | .tail _ _ ht, ctl, c, hr => LoopTail.ctl ht ctl c hr

theorem LoopTail.ctl {ctx : Ctx} {cst : CSt} {l : LoopRegion} {r : Res (Ctl × CSt)} :
    LoopTail ctx cst l r → ∀ ctl c, r = .ok (ctl, c) → ctl = .normal ∨ ctl = .ret
  | .updateErr .., _, _, hr => by cases hr
  | .updateGuard _ _ hg, ctl, c, hr => LoopGuard.ctl hg ctl c hr
  | .condFirstErr .., _, _, hr => by cases hr
  | .condFirstExit .., _, _, hr => by cases hr; exact .inl rfl
  | .condFirstUpdateErr .., _, _, hr => by cases hr
  | .condFirstNext _ _ _ hb, ctl, c, hr => LoopIter.ctl hb ctl c hr

end

mutual

/-- A region without escaping `break`/`continue` ends normally or returns. -/
theorem RegionAt.noEscape {ctx : Ctx} {rg : Region} (hs : rg.escapes = false) :
    ∀ {cst : CSt} {bid : BlockId} {r : Res (Ctl × CSt)}, RegionAt ctx rg cst bid r →
      ∀ ctl c, r = .ok (ctl, c) → ctl = .normal ∨ ctl = .ret
  | _, _, _, .exit (b := b) hb hbx _, ctl, c, hr => by
    have hmem : b ∈ rg.blocks := List.mem_of_getElem? hb
    rw [Region.escapes_eq] at hs
    exact BlockExec.noEscape (Blocks.escapes_mem hs hmem) hbx ctl c hr
  | _, _, _, .stop .., _, _, hr => by cases hr; exact .inl rfl
  | _, _, _, .step _ _ _ _ hn, ctl, c, hr => RegionAt.noEscape hs hn ctl c hr
  | _, _, _, .missing _, _, _, hr => by cases hr
  | _, _, _, .blockErr .., _, _, hr => by cases hr
  | _, _, _, .edgesErr .., _, _, hr => by cases hr
  | _, _, _, .ambiguous .., _, _, hr => by cases hr
  | _, _, _, .assignErr .., _, _, hr => by cases hr

theorem BlockExec.noEscape {ctx : Ctx} {cst : CSt} {b : Block} (hs : b.escapes = false) {r : Res (Ctl × CSt)} :
    BlockExec ctx cst b r → ∀ ctl c, r = .ok (ctl, c) → ctl = .normal ∨ ctl = .ret
  | .state h, ctl, c, hr => by subst hr; exact .inl (StateExec.normal h)
  | .region h, ctl, c, hr => RegionExec.noEscape hs h ctl c hr
  | .loop h, ctl, c, hr => LoopExec.ctl h ctl c hr
  | .cond h, ctl, c, hr => CondExec.noEscape hs h ctl c hr
  | .brk, _, _, _ => by simp [Block.escapes] at hs
  | .cont, _, _, _ => by simp [Block.escapes] at hs
  | .ret, _, _, hr => by cases hr; exact .inr rfl

theorem RegionExec.noEscape {ctx : Ctx} {cst : CSt} {rg : Region} (hs : rg.escapes = false) {r : Res (Ctl × CSt)} :
    RegionExec ctx cst rg r → ∀ ctl c, r = .ok (ctl, c) → ctl = .normal ∨ ctl = .ret
  | .start h, ctl, c, hr => RegionAt.noEscape hs h ctl c hr

theorem CondExec.noEscape {ctx : Ctx} {cst : CSt} {brs : List (Option Expr × Region)}
    (hs : Branches.escapes brs = false) {r : Res (Ctl × CSt)} :
    CondExec ctx cst brs r → ∀ ctl c, r = .ok (ctl, c) → ctl = .normal ∨ ctl = .ret
  | .nil, _, _, hr => by cases hr; exact .inl rfl
  | .els (rg := rg) h, ctl, c, hr => by
    simp only [Branches.escapes, Bool.or_eq_false_iff] at hs
    exact RegionExec.noEscape hs.1 h ctl c hr
  | .condErr _, _, _, hr => by cases hr
  | .taken _ h, ctl, c, hr => by
    simp only [Branches.escapes, Bool.or_eq_false_iff] at hs
    exact RegionExec.noEscape hs.1 h ctl c hr
  | .skipped _ h, ctl, c, hr => by
    simp only [Branches.escapes, Bool.or_eq_false_iff] at hs
    exact CondExec.noEscape hs.2 h ctl c hr

end

/-- A tree outcome matched up to the error is matched. -/
theorem NestedOut.of_get? {X : Res (Ctl × CSt) → Prop} {r' r : Res (TCtl × CSt)} (hr : r'.get? = r.get?)
    (h : NestedOut X r') : NestedOut X r := by
  rcases h with h | ⟨e, e', rfl, hx⟩ | ⟨ctl, c, hctl, rfl, hx⟩
  · exact .inl h
  · obtain ⟨e₀, rfl⟩ := Res.exists_err_of_get? (by rw [← hr]; rfl)
    exact .inr (.inl ⟨e₀, e', rfl, hx⟩)
  · rw [Res.get?_ok, eq_comm, Res.get?_eq_some] at hr
    exact .inr (.inr ⟨ctl, c, hctl, hr, hx⟩)

/-- The tree list `regCheck` matches against the top region of `inner`. -/
def innerRegOk (tc : TCtx) (σ : Equiv.Perm Name) (inner : SDFG) (body : List SNode) : Bool :=
  match regCheck ((tc.renT σ).innerCtx inner) (tc.renT σ) (fun _ _ _ => false) (3 * Region.size inner.top + 3)
      inner.top inner.top.start body with
  | .ok [] => true
  | _ => false

theorem innerRegOk_sound {tc : TCtx} {σ : Equiv.Perm Name} {inner : SDFG} {body : List SNode}
    (h : innerRegOk tc σ inner body = true) (I : Interp) :
    RegCorr (((tc.withI I).renT σ).innerCtx inner) ((tc.withI I).renT σ) inner.top inner.top.start body := by
  unfold innerRegOk at h
  split at h
  · rename_i hr
    obtain ⟨pre, hpre, hc⟩ := regCheck_sound (fun _ _ _ h => by cases h) _ _ _ _ _ hr
    rw [List.append_nil] at hpre
    rw [hpre]; exact RegCorr.withI I hc
  · cases h

/-- Check the inlined top region of a nested SDFG (`InnerCorr`): the tree
body renamed by `σ` (`SNode.ren`) in the renamed tree context (`TCtx.renT`)
is matched against the region (the renaming theorem `STree.ListExec.ren`
carries the tree's runs over).  No `break`/`continue` leaves the region.  A
body ending in the label its returns were rebound to is matched with those
gotos turned back into returns (`SNode.retarget`, `STree.ListExec.unlabel`);
otherwise the body has no gotos. -/
def innerCheck (tc : TCtx) (σ : Equiv.Perm Name) (inner : SDFG) (body : List SNode) : Bool :=
  !inner.top.escapes && descFixed? σ tc.ctx &&
    SNodes.renOk σ (tc.renT σ).privateTransients tc.privateTransients body &&
    match exitLabel? (SNodes.ren σ body) with
    | some (L, pre) =>
      !SNodes.hasLabel L pre && SNodes.jumpsTo (some L) pre && innerRegOk tc σ inner (SNodes.retarget L pre)
    | none => SNodes.jumpsTo none (SNodes.ren σ body) && innerRegOk tc σ inner (SNodes.ren σ body)

theorem innerCheck_sound {tc : TCtx} {σ : Equiv.Perm Name} {inner : SDFG} {body : List SNode}
    (hcons : tc.consumed = none) (h : innerCheck tc σ inner body = true) : InnerCorr tc σ inner body := by
  simp only [innerCheck, Bool.and_eq_true, Bool.not_eq_true'] at h
  obtain ⟨⟨⟨hs, hd⟩, hok⟩, h⟩ := h
  intro I hw cst r hx
  have hR : TCtx.Ren σ ((tc.withI I).renT σ) (tc.withI I) := ⟨Ctx.ren_ren σ _, descFixed?_sound hd, rfl, hcons⟩
  obtain ⟨r₁, hx₁, hr₁⟩ := STree.ListExec.ren hR hx hok hok
  refine NestedOut.of_get? hr₁ ?_
  have hX : ∀ {r' : Res (TCtl × CSt)} {b : List SNode}, innerRegOk tc σ inner b = true →
      STree.ListExec ((tc.withI I).renT σ) b false {} b cst r' →
      Out (RegionAt (((tc.withI I).renT σ).innerCtx inner) inner.top cst inner.top.start) r' := fun hb hxb =>
    (RegCorr.matched (ctx := ((tc.withI I).renT σ).innerCtx inner) (tc := (tc.withI I).renT σ) rfl rfl hw
      (innerRegOk_sound hb I) HeadOk.nil (ListExec.append_nil hxb)).toOut
  split at h
  · rename_i L pre hlab
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at h
    obtain ⟨⟨hnl, hj⟩, hreg⟩ := h
    rw [exitLabel?_sound hlab] at hx₁
    obtain ⟨r₂, hx₂, hrel⟩ := STree.ListExec.unlabel L (full' := SNodes.retarget L pre) hnl hj hx₁
    rcases hX hreg hx₂ with ⟨e, hu, h⟩ | ⟨e, e', rfl, h⟩ | ⟨ctl, c, rfl, h⟩
    · exact .inl ⟨e, hu, h⟩
    · rcases hrel with ⟨e₁, rfl, -⟩ | ⟨c, -, hc⟩ | ⟨_, _, _, _, -, hc⟩
      · exact .inr (.inl ⟨e₁, e', rfl, h⟩)
      · rcases hc with hc | hc <;> cases hc
      · cases hc
    · have hctl := RegionAt.noEscape hs h ctl c rfl
      rcases hrel with ⟨e₁, -, he⟩ | ⟨c₁, rfl, hc⟩ | ⟨ctl', c₁, hj', hne, rfl, hc⟩
      · cases he
      · rcases hc with hc | hc <;>
          (simp only [Res.ok.injEq, Prod.mk.injEq] at hc; obtain ⟨-, rfl⟩ := hc;
           exact .inr (.inr ⟨ctl, c, hctl, rfl, h⟩))
      · simp only [Res.ok.injEq, Prod.mk.injEq] at hc
        obtain ⟨hc, rfl⟩ := hc
        rcases hctl with rfl | rfl
        · rw [← hc] at hj'; cases hj'
        · exact absurd hc.symm hne
  · simp only [Bool.and_eq_true] at h
    obtain ⟨hj, hreg⟩ := h
    rcases hX hreg hx₁ with ⟨e, hu, h⟩ | ⟨e, e', he, h⟩ | ⟨ctl, c, rfl, h⟩
    · exact .inl ⟨e, hu, h⟩
    · exact .inr (.inl ⟨e, e', he, h⟩)
    · have hctl := RegionAt.noEscape hs h ctl c rfl
      have hok' := STree.ListExec.okFor none hx₁ hj hj _ _ rfl
      rcases hctl with rfl | rfl
      · exact .inr (.inr ⟨.normal, c, .inl rfl, rfl, h⟩)
      · cases hok'

/-- The per-call transients of the nested SDFGs of `g`'s states, each with
its memory name in `g` (`Ctx.enter` at top level): the tree's flattened name
and the SDFG's. -/
def nestedPairs (g : SDFG) : List (Name × Name) :=
  g.top.allStates.flatMap fun s => s.nodes.flatMap fun n =>
    match n.kind with
    | .nested inner _ _ => inner.callTransients.map fun t => (t, inner.name ++ "#" ++ toString n.id ++ "/" ++ t)
    | _ => []

/-- The renaming between `g` and its flattened tree. -/
def corrPerm (g : SDFG) : Equiv.Perm Name := swapsPerm (nestedPairs g)

/-- The SDFG and the tree start from the same memory: compared on every name
either defines (both are false elsewhere). -/
def keepEq? (g : SDFG) (t : STree) : Bool :=
  (t.containers.map Prod.fst ++ g.arrays.map Prod.fst ++ SDFG.externalNames "" g).all fun n =>
    t.keepAtStart n == g.keepAtStart n

theorem keepEq?_sound {g : SDFG} {t : STree} (h : keepEq? g t = true) : t.keepAtStart = g.keepAtStart := by
  funext n
  by_cases hn : n ∈ t.containers.map Prod.fst ++ g.arrays.map Prod.fst ++ SDFG.externalNames "" g
  · simpa using List.all_eq_true.1 h n hn
  · simp only [List.mem_append, List.mem_map, not_or, not_exists, not_and] at hn
    obtain ⟨⟨ht, hg⟩, he⟩ := hn
    have h1 : t.keepAtStart n = false := by
      simp only [STree.keepAtStart, List.any_eq_false, Bool.and_eq_true, beq_iff_eq, not_and]
      intro x hx hxn
      exact absurd hxn (ht x hx)
    have h2 : g.keepAtStart n = false := by
      simp only [SDFG.keepAtStart, Bool.or_eq_false_iff, List.any_eq_false, Bool.and_eq_true, beq_iff_eq, not_and]
      refine ⟨fun x hx hxn => absurd hxn (hg x hx), ?_⟩
      simpa using he
    rw [h1, h2]

/-- The names of the SDFG's own states agree with the flattened context up to
the renaming. -/
def outerNames? (g : SDFG) (t : STree) (π : Equiv.Perm Name) : Bool :=
  Region.namesOk? (nameOkD π (sctx Interp.uninterpreted g) (flatCtx Interp.uninterpreted g t π))
    (nameOkE π (sctx Interp.uninterpreted g) (flatCtx Interp.uninterpreted g t π)) true g.top

theorem outerNames?_sound {g : SDFG} {t : STree} {π : Equiv.Perm Name} (h : outerNames? g t π = true) (I : Interp) :
    Region.Names π (fun _ => True) (sctx I g) (flatCtx I g t π) g.top :=
  Region.namesOk?_sound (fun x hx => nameOkD_sound hx) (fun x hx => nameOkE_sound hx)
    (fun _ x => ⟨by simp [sctx, flatCtx], trivial⟩) _ h

/-- The checker. -/
def corrCheck (g : SDFG) (t : STree) : Except String Unit :=
  if keepEq? g t = false then .error "the start memories differ"
  else if (pairNames (nestedPairs g)).any g.keepAtStart then .error "a per-call name of a nested SDFG is defined at start"
  else if !g.interface.all (fun x => corrPerm g x == x) then .error "the renaming moves an observable container"
  else if outerNames? g t (corrPerm g) = false then .error "the descriptor repositories differ"
  else if t.constants.isEmpty && g.constants.isEmpty then
    match regCheck (flatCtx Interp.uninterpreted g t (corrPerm g)) (TCtx.ofTree Interp.uninterpreted t)
        (innerCheck (TCtx.ofTree Interp.uninterpreted t)) (3 * Region.size g.top + 3) g.top g.top.start t.children with
    | .ok [] => .ok ()
    | .ok _ => .error "tree nodes after the end of the program"
    | .error m => .error m
  else .error "constants are not compared yet"

/-- `corrCheck` as a Boolean. -/
def corr? (g : SDFG) (t : STree) : Bool :=
  match corrCheck g t with
  | .ok () => true
  | .error _ => false


/-! ## Main theorems -/

/-- When every `External` name of the SDFG is one of its own containers
defined at start, the tree with the same repository keeps the same names. -/
theorem keepAtStart_eq {g : SDFG} {t : STree} (ha : t.containers = g.arrays)
    (he : (SDFG.externalNames "" g).all (fun n => g.arrays.any fun (a, d) => a == n && d.definedAtStart) = true) :
    t.keepAtStart = g.keepAtStart := by
  funext n
  simp only [STree.keepAtStart, SDFG.keepAtStart, ha]
  cases hx : (SDFG.externalNames "" g).contains n with
  | false => simp
  | true =>
    have := List.all_eq_true.1 he n (List.elem_iff.1 hx)
    simp [this]

/-- **Soundness of the checker.** -/
theorem corrCheck_sound {g : SDFG} {t : STree} (h : corrCheck g t = .ok ()) (I : Interp) : ProgCorr I g t := by
  unfold corrCheck at h
  split at h
  · cases h
  rename_i hk
  split at h
  · cases h
  rename_i hf
  split at h
  · cases h
  rename_i hi
  split at h
  · cases h
  rename_i hn
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, List.isEmpty_iff] at hc
    refine ⟨corrPerm g, ?_, keepEq?_sound (by simpa using hk), ?_, outerNames?_sound (by simpa using hn) I,
      fun x hx => ?_, fun x hx => ?_⟩
    · funext ρ x
      simp [SDFG.withConstants, STree.toSDFG, hc.1, hc.2]
    · split at h
      · rename_i hr
        obtain ⟨pre, hpre, hc⟩ := regCheck_sound (fun _ _ _ hb => innerCheck_sound rfl hb) _ _ _ _ _ hr
        rw [List.append_nil] at hpre
        subst hpre
        exact RegCorr.withI I hc
      · cases h
      · cases h
    · have hm : x ∈ pairNames (nestedPairs g) := by
        by_contra hm
        exact hx (swapsPerm_of_not_mem _ _ hm)
      simp only [List.any_eq_true, not_exists, not_and, Bool.not_eq_true] at hf
      exact hf x hm
    · simp only [Bool.not_eq_true', Bool.not_eq_false, List.all_eq_true, beq_iff_eq] at hi
      exact hi x hx
  · cases h

theorem corr?_sound {g : SDFG} {t : STree} (h : corr? g t = true) (I : Interp) : ProgCorr I g t := by
  unfold corr? at h
  split at h
  · rename_i hc
    exact corrCheck_sound hc I
  · cases h

/-- **Certified pairs (M5a)**: every outcome of the schedule tree is an
outcome of the SDFG, or the SDFG may end in undefined behavior. -/
theorem corr?_exec {g : SDFG} {t : STree} (h : corr? g t = true) {I : Interp} (hl : I.Lawful) {ρ₀ : Valuation}
    {m₀ : Mem} (hna : ∀ n, m₀.aliases n = none) {r : Res Mem} (ht : STree.Exec I t ρ₀ m₀ r) :
    (∃ e, e.isUB = true ∧ SDFG.Exec I g ρ₀ m₀ (.err e)) ∨ (∃ e e', r = .err e ∧ SDFG.Exec I g ρ₀ m₀ (.err e')) ∨
      ∃ m, r = .ok m ∧ ∃ m', SDFG.Exec I g ρ₀ m₀ (.ok m') ∧ obsEq g.interface m m' :=
  let ⟨_, hc⟩ := corr?_sound h I
  hc.exec hl.undef_absorb hna ht

/-- **Certified pairs (M5a)**: the schedule tree refines a supported SDFG. -/
theorem corr?_refines {g : SDFG} {t : STree} (h : corr? g t = true) (hs : Supported g.prog) :
    Refines t.prog g.prog :=
  ProgCorr.refines (corr?_sound h) hs

end Corr

end Dace
