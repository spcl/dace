-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Equiv.CorrCtl

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
`SeqPure` (no streams, views, references, consume scopes, nested SDFGs,
private transients or code→code values) with at most one component; no
dynamic map inputs; equal descriptor repositories; no constants; a code
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

/-- A matched scope: the order of its direct nodes, the full (nested) order
and the rest of the tree list. -/
abbrev MatchRes := List NodeId × List NodeId × List SNode

section Check

variable (ctx : Ctx) (tc : TCtx) (s : State)

/-- Match the image of node `n` at the head of `cs` (`rec` matches a scope
body): the full order nested in `n` and the rest of the list. -/
def matchOne (rec : List NodeId → List SNode → Option MatchRes) (n : DNode) (cs : List SNode) :
    Option (List NodeId × List SNode) :=
  match n.kind with
  | .access _ | .mapExit =>
    if s.copyShape? ctx n = true then
      match takeCopies (s.copiesInto ctx n).length cs with
      | some (ps, rest) => if ps.map (fun p => p.2.copyKey p.1) = s.copiesInto ctx n then some ([], rest) else none
      | none => none
    else none
  | .tasklet t =>
    match cs with
    | .taskletNode t' ins outs :: rest =>
      if t' = t ∧ s.codeShape? n = true ∧ s.codeIns n = ins.pairs.map (fun p => Memlet.inKey p.1 p.2) ∧
          s.codeOuts n = outs.pairs.map (fun p => Memlet.outKey p.1 p.2) ∧ s.hasDynamicWcrOut n = false ∧
          outs.hasDynamicWcr = false then some ([], rest) else none
    | _ => none
  | .library l =>
    match cs with
    | .libraryCall l' _ ins outs :: rest =>
      if (l'.classpath = l.classpath ∧ l'.reduce = l.reduce) ∧ s.codeShape? n = true ∧ s.codeIns n = ins.pairs.map (fun p => Memlet.inKey p.1 p.2) ∧
          s.codeOuts n = outs.pairs.map (fun p => Memlet.outKey p.1 p.2) ∧ s.hasDynamicWcrOut n = false ∧
          outs.hasDynamicWcr = false then some ([], rest) else none
    | _ => none
  | .mapEntry mp =>
    match cs with
    | .mapScope mp' body :: rest =>
      if mp' = mp ∧ (s.inEdges n.id).all (fun e => !s.isDynamicInput e) = true ∧
          tc.privateTransients body = ctx.privateTransients s n.id then
        match rec (s.directChildren (some n.id)) body with
        | some (oM, fullM, []) =>
          if s.isTopologicalOrder (s.scopeNodes n.id) fullM = true ∧
              fullM.filter (s.directChildren (some n.id)).contains = oM then some (fullM, rest) else none
        | _ => none
      else none
    | _ => none
  | _ => none

/-- The unplaced nodes and the nodes nested in them. -/
def pendingAll (U : List NodeId) : List NodeId := U ++ U.flatMap s.scopeNodes

/-- All predecessors of `u` are placed. -/
def ready (U : List NodeId) (u : NodeId) : Bool := (s.inEdges u).all fun e => !(pendingAll s U).contains e.src

/-- Place the nodes `U` of a scope along the tree list `cs` (see the module
documentation). -/
def matchSeg : Nat → List NodeId → List SNode → Option MatchRes
  | 0, _, _ => none
  | fuel + 1, U, cs =>
    if U.isEmpty then some ([], [], cs)
    else
      match U.findSome? (fun u => if ready s U u then
          (s.node? u).bind fun n => (matchOne ctx tc s (matchSeg fuel) n cs).map fun p => (u, p) else none) with
      | some (u, fullN, rest₁) =>
        (matchSeg fuel (U.erase u) rest₁).map fun r => (u :: r.1, u :: fullN ++ r.2.1, r.2.2)
      | none => none

end Check

/-! ## Soundness of the matching -/

section Sound

variable {ctx : Ctx} {tc : TCtx} {s : State}

theorem matchOne_sound {rec : List NodeId → List SNode → Option MatchRes}
    (hrec : ∀ U cs o full rest, rec U cs = some (o, full, rest) → ∃ pre, cs = pre ++ rest ∧ SegCorr ctx tc s o pre)
    {n : DNode} {cs : List SNode} {fullN : List NodeId} {rest : List SNode}
    (hm : matchOne ctx tc s rec n cs = some (fullN, rest)) :
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
          exact .copies hn (by simp [DNode.isCopyTarget, hk]) (State.copyShape?_sound hsh) hks.symm ht
        · cases hm
      · cases hm
    · cases hm
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
          exact .copies hn (by simp [DNode.isCopyTarget, hk]) (State.copyShape?_sound hsh) hks.symm ht
        · cases hm
      · cases hm
    · cases hm
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
    · rename_i mp' body rest'
      split at hm
      · rename_i hc
        obtain ⟨rfl, hdyn, hpriv⟩ := hc
        split at hm
        · rename_i oM fullM hb
          split at hm
          · rename_i hto
            simp only [Option.some.injEq, Prod.mk.injEq] at hm
            obtain ⟨-, rfl⟩ := hm
            obtain ⟨pre, hpre, hseg⟩ := hrec _ _ _ _ _ hb
            rw [List.append_nil] at hpre
            subst hpre
            refine ⟨[_], rfl, fun id o tail hn ht => .map hn hk ?_ hto.1 hto.2 hpriv hseg ht⟩
            intro e he
            have := List.all_eq_true.1 hdyn e he
            simpa using this
          · cases hm
        · cases hm
      · cases hm
    · cases hm
  · cases hm

theorem matchSeg_sound : ∀ (fuel : Nat) (U : List NodeId) (cs : List SNode) (o full : List NodeId) (rest : List SNode),
    matchSeg ctx tc s fuel U cs = some (o, full, rest) → ∃ pre, cs = pre ++ rest ∧ SegCorr ctx tc s o pre
  | 0, _, _, _, _, _, h => by simp [matchSeg] at h
  | fuel + 1, U, cs, o, full, rest, h => by
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
          obtain ⟨img, hcs, hcons⟩ := matchOne_sound (matchSeg_sound fuel) hm'
          obtain ⟨⟨o', full', rest'⟩, hr, he⟩ := Option.map_eq_some_iff.1 h
          simp only [Prod.mk.injEq] at he
          obtain ⟨rfl, -, rfl⟩ := he
          obtain ⟨pre, hpre, hseg⟩ := matchSeg_sound fuel _ _ _ _ _ hr
          refine ⟨img ++ pre, by rw [hcs, hpre, List.append_assoc], hcons _ _ _ hn hseg⟩
        · cases hu'
      · cases h

end Sound

/-! ## States, the chain and the program -/

/-- Check one state against the head of the tree list; the rest of the list. -/
def stateCheck (ctx : Ctx) (tc : TCtx) (s : State) (cs : List SNode) : Except String (List SNode) :=
  if seqPure? ctx s = false then .error s!"state '{s.label}': outside the fragment (seqPure?)"
  else
    match s.components with
    | [] => .ok cs
    | [c] =>
      match matchSeg ctx tc s (2 * s.nodes.length + 2) (s.topLevelOf c) cs with
      | some (o, full, rest) =>
        if s.isTopologicalOrder c full = true ∧ full.filter (s.topLevelOf c).contains = o then .ok rest
        else .error s!"state '{s.label}': the matched order is not topological"
      | none => .error s!"state '{s.label}': no tree segment matches"
    | _ => .error s!"state '{s.label}': several components"

theorem stateCheck_sound {ctx : Ctx} {tc : TCtx} {s : State} {cs rest : List SNode}
    (h : stateCheck ctx tc s cs = .ok rest) : ∃ pre, cs = pre ++ rest ∧ StateCorr ctx tc s pre := by
  unfold stateCheck at h
  split at h
  · cases h
  · rename_i hp
    have hpure : SeqPure ctx s := seqPure?_sound (by simpa using hp)
    split at h
    · rename_i hc
      cases h
      exact ⟨[], rfl, hpure, .inl ⟨hc, rfl⟩⟩
    · rename_i c hc
      split at h
      · rename_i o full rest' hm
        split at h
        · rename_i hto
          cases h
          obtain ⟨pre, hpre, hseg⟩ := matchSeg_sound _ _ _ _ _ _ hm
          exact ⟨pre, hpre, hpure, .inr ⟨c, o, full, hc, hseg, hto.1, hto.2⟩⟩
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

section Check

variable (ctx : Ctx) (tc : TCtx)

mutual

/-- Check the region `rg` from block `bid` against the head of the tree list;
the rest of the list. -/
def regCheck : Nat → Region → BlockId → List SNode → Except String (List SNode)
  | 0, _, _, _ => .error "control flow too deep or cyclic"
  | f + 1, rg, bid, cs =>
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
          else .error s!"interstate edge from block {bid}: conditional"
        | _ => .error s!"block {bid}: several out-edges (GBlocks are M5b step 5)"

/-- Check one block against the head of the tree list; the rest of the list. -/
def blockCheck : Nat → Block → List SNode → Except String (List SNode)
  | 0, _, _ => .error "control flow too deep"
  | f + 1, b, cs =>
    match b with
    | .state s => stateCheck ctx tc s cs
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
      | .unstructured => .error "unstructured regions (GBlocks) are not supported yet"

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

end

end Check

section CheckSound

variable {ctx : Ctx} {tc : TCtx}

mutual

theorem regCheck_sound : ∀ (f : Nat) (rg : Region) (bid : BlockId) (cs rest : List SNode),
    regCheck ctx tc f rg bid cs = .ok rest → ∃ pre, cs = pre ++ rest ∧ RegCorr ctx tc rg bid pre
  | 0, _, _, _, _, h => by simp [regCheck] at h
  | f + 1, rg, bid, cs, rest, h => by
    unfold regCheck at h
    split at h
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
          · cases h
        · cases h

theorem blockCheck_sound : ∀ (f : Nat) (b : Block) (cs rest : List SNode),
    blockCheck ctx tc f b cs = .ok rest → ∃ pre, cs = pre ++ rest ∧ BlockCorr ctx tc b pre
  | 0, _, _, _, h => by simp [blockCheck] at h
  | f + 1, b, cs, rest, h => by
    simp only [blockCheck] at h
    split at h
    · obtain ⟨pre, hpre, hsc⟩ := stateCheck_sound h
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
      · cases h

theorem elifCheck_sound : ∀ (f : Nat) (brs : List (Option Expr × Region)) (cs rest : List SNode),
    elifCheck ctx tc f brs cs = .ok rest → ∃ pre, cs = pre ++ rest ∧ ElifCorr ctx tc brs pre
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

end

end CheckSound

/-- The checker. -/
def corrCheck (g : SDFG) (t : STree) : Except String Unit :=
  if t.containers ≠ g.arrays then .error "the descriptor repositories differ"
  else if t.constants.isEmpty && g.constants.isEmpty then
    match regCheck (checkCtx g) (TCtx.ofTree Interp.uninterpreted t) (3 * Region.size g.top + 3) g.top g.top.start
        t.children with
    | .ok [] => .ok ()
    | .ok _ => .error "tree nodes after the end of the program"
    | .error m => .error m
  else .error "constants are not compared yet"

/-- `corrCheck` as a Boolean. -/
def corr? (g : SDFG) (t : STree) : Bool :=
  match corrCheck g t with
  | .ok () => true
  | .error _ => false

/-! ## Independence of the interpretation -/

section WithI

variable {g : SDFG} {t : STree} {s : State} (I₁ I₂ : Interp)

theorem CopyShape.withI {n : DNode} (h : s.CopyShape (sctx I₁ g) n) : s.CopyShape (sctx I₂ g) n :=
  ⟨h.noRef, h.dst⟩

theorem SeqPure.withI (h : SeqPure (sctx I₁ g) s) : SeqPure (sctx I₂ g) s :=
  ⟨h.plain, h.shapes, h.memlets, h.kinds, h.maps, h.reduces, h.codeOut⟩

theorem SegCorr.withI {o : List NodeId} {cs : List SNode} (h : SegCorr (sctx I₁ g) (TCtx.ofTree I₁ t) s o cs) :
    SegCorr (sctx I₂ g) (TCtx.ofTree I₂ t) s o cs := by
  induction h with
  | nil => exact .nil
  | copies hn hk hsh hks _ ih => exact .copies hn hk (CopyShape.withI I₁ I₂ hsh) hks ih
  | tasklet hn hk hsh hki hko hd hdT _ ih => exact .tasklet hn hk hsh hki hko hd hdT ih
  | library hn hk hl hsh hki hko hd hdT _ ih => exact .library hn hk hl hsh hki hko hd hdT ih
  | map hn hk hdyn hto hdir hpriv _ _ ihb ih => exact .map hn hk hdyn hto hdir hpriv ihb ih

theorem StateCorr.withI {cs : List SNode} (h : StateCorr (sctx I₁ g) (TCtx.ofTree I₁ t) s cs) :
    StateCorr (sctx I₂ g) (TCtx.ofTree I₂ t) s cs := by
  refine ⟨SeqPure.withI I₁ I₂ h.pure, ?_⟩
  rcases h.comps with h₁ | ⟨c, o, full, hc, hseg, hto, hdir⟩
  · exact .inl h₁
  · exact .inr ⟨c, o, full, hc, SegCorr.withI I₁ I₂ hseg, hto, hdir⟩

mutual

theorem RegCorr.withI {rg : Region} {bid : BlockId} {cs : List SNode} :
    RegCorr (sctx I₁ g) (TCtx.ofTree I₁ t) rg bid cs → RegCorr (sctx I₂ g) (TCtx.ofTree I₂ t) rg bid cs
  | .last hb hc he => .last hb (BlockCorr.withI hc) he
  | .step hb hc he hu hp ha hs hh h =>
    .step hb (BlockCorr.withI hc) he hu hp ha (SeqAssigns.of_g (c₁ := sctx I₁ g) (c₂ := sctx I₂ g) rfl hs) hh
      (RegCorr.withI h)

theorem BlockCorr.withI {b : Block} {cs : List SNode} :
    BlockCorr (sctx I₁ g) (TCtx.ofTree I₁ t) b cs → BlockCorr (sctx I₂ g) (TCtx.ofTree I₂ t) b cs
  | .state h => .state (StateCorr.withI I₁ I₂ h)
  | .loop hn hm hb => .loop hn hm (RegCorr.withI hb)
  | .cond hc hb ht => .cond hc (RegCorr.withI hb) (ElifCorr.withI ht)
  | .brk => .brk
  | .cont => .cont
  | .ret => .ret
  | .flat h => .flat (RegCorr.withI h)
  | .named h => .named (RegCorr.withI h)

theorem ElifCorr.withI {brs : List (Option Expr × Region)} {cs : List SNode} :
    ElifCorr (sctx I₁ g) (TCtx.ofTree I₁ t) brs cs → ElifCorr (sctx I₂ g) (TCtx.ofTree I₂ t) brs cs
  | .nil => .nil
  | .elif hc hb ht => .elif hc (RegCorr.withI hb) (ElifCorr.withI ht)
  | .els hb => .els (RegCorr.withI hb)

end

end WithI

/-! ## Main theorems -/

/-- **Soundness of the checker.** -/
theorem corrCheck_sound {g : SDFG} {t : STree} (h : corrCheck g t = .ok ()) (I : Interp) : ProgCorr I g t := by
  unfold corrCheck at h
  split at h
  · cases h
  · rename_i ha
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, List.isEmpty_iff] at hc
      refine ⟨by simpa using ha, ?_, ?_⟩
      · funext ρ x
        simp [SDFG.withConstants, STree.toSDFG, hc.1, hc.2]
      · split at h
        · rename_i hr
          obtain ⟨pre, hpre, hc⟩ := regCheck_sound _ _ _ _ _ hr
          rw [List.append_nil] at hpre
          subst hpre
          exact RegCorr.withI Interp.uninterpreted I hc
        · cases h
        · cases h
    · cases h

theorem corr?_sound {g : SDFG} {t : STree} (h : corr? g t = true) (I : Interp) : ProgCorr I g t := by
  unfold corr? at h
  split at h
  · rename_i hc
    exact corrCheck_sound hc I
  · cases h

/-- **Certified pairs (M5a)**: every outcome of the schedule tree is an
outcome of the SDFG, or the SDFG may end in undefined behavior. -/
theorem corr?_exec {g : SDFG} {t : STree} (h : corr? g t = true) {I : Interp} {ρ₀ : Valuation} {m₀ : Mem}
    {r : Res Mem} (ht : STree.Exec I t ρ₀ m₀ r) :
    (∃ e, e.isUB = true ∧ SDFG.Exec I g ρ₀ m₀ (.err e)) ∨ SDFG.Exec I g ρ₀ m₀ r :=
  (corr?_sound h I).exec ht

/-- **Certified pairs (M5a)**: the schedule tree refines a supported SDFG. -/
theorem corr?_refines {g : SDFG} {t : STree} (h : corr? g t = true) (hs : Supported g.prog) :
    Refines t.prog g.prog :=
  ProgCorr.refines (corr?_sound h) hs

end Corr

end Dace
