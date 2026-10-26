-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.STree.Semantics

/-!
# Exit gotos of inlined nested SDFGs

`sdfg_to_tree` inlines a nested SDFG and rebinds its returns (exit gotos,
`goto none`) to a label after the inlined body (`_bind_exits_to_label`):
`goto L … label L`.  To match the body against the nested SDFG, whose
returns end the call, the gotos are turned back into returns.

* `SNode.retarget L`: `goto L` becomes `goto none`.  A run of a node without
  label `L` is a run of the retargeted node, with `goto L` outcomes turned
  into `ret` (`STree.NodeExec.retarget`, `STree.ListExec.retarget`, …).
* `SNode.jumpsTo J`: every goto of the node targets `J`, and there are no
  exit gotos; then runs of it never return and every escaping goto targets
  `J` (`STree.NodeExec.okFor`, …).
* `SNode.keepsSyms`: no symbol assignments and no loops; runs keep the
  valuation (`STree.NodeExec.keeps_ρ`, …).
-/

set_option autoImplicit false

namespace Dace

/-! ## Syntax -/

section Syntax

variable (L : String)

mutual

/-- Turn `goto L` into an exit goto. -/
def SNode.retarget : SNode → SNode
  | .namedRegion l cs => .namedRegion l (SNodes.retarget cs)
  | .gblock cs => .gblock (SNodes.retarget cs)
  | .stateLabel n => .stateLabel n
  | .goto t => .goto (if t = some L then none else t)
  | .assign n v => .assign n v
  | .loopScope l cs => .loopScope l (SNodes.retarget cs)
  | .forScope l cs => .forScope l (SNodes.retarget cs)
  | .whileScope l cs => .whileScope l (SNodes.retarget cs)
  | .doWhileScope l cs => .doWhileScope l (SNodes.retarget cs)
  | .ifScope c cs => .ifScope c (SNodes.retarget cs)
  | .stateIfScope c cs => .stateIfScope c (SNodes.retarget cs)
  | .breakNode => .breakNode
  | .continueNode => .continueNode
  | .elifScope c cs => .elifScope c (SNodes.retarget cs)
  | .elseScope cs => .elseScope (SNodes.retarget cs)
  | .mapScope mp cs => .mapScope mp (SNodes.retarget cs)
  | .consumeScope c cs => .consumeScope c (SNodes.retarget cs)
  | .taskletNode t ins outs => .taskletNode t ins outs
  | .libraryCall l props ins outs => .libraryCall l props ins outs
  | .copyNode target m => .copyNode target m
  | .dynScopeCopy target m => .dynScopeCopy target m
  | .viewNode target source m sd vd => .viewNode target source m sd vd
  | .refSetNode target m src d => .refSetNode target m src d

def SNodes.retarget : List SNode → List SNode
  | [] => []
  | n :: ns => SNode.retarget n :: SNodes.retarget ns

end

mutual

/-- Does label `L` occur in the node, at any depth? -/
def SNode.hasLabel : SNode → Bool
  | .namedRegion _ cs | .gblock cs | .loopScope _ cs | .forScope _ cs | .whileScope _ cs | .doWhileScope _ cs
  | .ifScope _ cs | .stateIfScope _ cs | .elifScope _ cs | .elseScope cs | .mapScope _ cs
  | .consumeScope _ cs => SNodes.hasLabel cs
  | .stateLabel n => n == L
  | _ => false

def SNodes.hasLabel : List SNode → Bool
  | [] => false
  | n :: ns => SNode.hasLabel n || SNodes.hasLabel ns

end

end Syntax

mutual

/-- Every goto of the node, at any depth, targets `J` (`some L`); exit gotos
are not allowed.  With `J = none` the node has no gotos at all. -/
def SNode.jumpsTo (J : Option String) : SNode → Bool
  | .namedRegion _ cs | .gblock cs | .loopScope _ cs | .forScope _ cs | .whileScope _ cs | .doWhileScope _ cs
  | .ifScope _ cs | .stateIfScope _ cs | .elifScope _ cs | .elseScope cs | .mapScope _ cs
  | .consumeScope _ cs => SNodes.jumpsTo J cs
  | .goto t => t.isSome && t == J
  | _ => true

def SNodes.jumpsTo (J : Option String) : List SNode → Bool
  | [] => true
  | n :: ns => SNode.jumpsTo J n && SNodes.jumpsTo J ns

end

/-- The control signals a node with `jumpsTo J` may end in. -/
def TCtl.okFor (J : Option String) : TCtl → Bool
  | .ret => false
  | .goto l => some l == J
  | _ => true

section Syntax

variable (L : String)

/-- `goto L` as a return. -/
def TCtl.retarget : TCtl → TCtl
  | .goto l => if l = L then .ret else .goto l
  | c => c

def NodeOut.retarget (o : NodeOut) : NodeOut := (o.1, o.2.1.retarget L, o.2.2)

/-- A list outcome with `goto L` as a return. -/
def retargetR (r : Res (TCtl × CSt)) : Res (TCtl × CSt) := r.map fun p => (p.1.retarget L, p.2)

end Syntax

mutual

/-- No symbol assignments and no loops, at any depth. -/
def SNode.keepsSyms : SNode → Bool
  | .namedRegion _ cs | .gblock cs | .ifScope _ cs | .stateIfScope _ cs | .elifScope _ cs | .elseScope cs
  | .mapScope _ cs | .consumeScope _ cs => SNodes.keepsSyms cs
  | .assign .. | .loopScope .. | .forScope .. | .whileScope .. | .doWhileScope .. => false
  | _ => true

def SNodes.keepsSyms : List SNode → Bool
  | [] => true
  | n :: ns => SNode.keepsSyms n && SNodes.keepsSyms ns

end

/-! ## Structure -/

section Structure

variable (L : String)

theorem SNodes.retarget_eq : ∀ cs : List SNode, SNodes.retarget L cs = cs.map (SNode.retarget L)
  | [] => rfl
  | c :: cs => by simp [SNodes.retarget, SNodes.retarget_eq cs]

theorem SNodes.retarget_append (a b : List SNode) :
    SNodes.retarget L (a ++ b) = SNodes.retarget L a ++ SNodes.retarget L b := by
  simp [SNodes.retarget_eq]

theorem SNodes.hasLabel_append (a b : List SNode) :
    SNodes.hasLabel L (a ++ b) = (SNodes.hasLabel L a || SNodes.hasLabel L b) := by
  induction a with
  | nil => rfl
  | cons x a ih => simp [SNodes.hasLabel, ih, Bool.or_assoc]

theorem SNodes.jumpsTo_append (J : Option String) (a b : List SNode) :
    SNodes.jumpsTo J (a ++ b) = (SNodes.jumpsTo J a && SNodes.jumpsTo J b) := by
  induction a with
  | nil => rfl
  | cons x a ih => simp [SNodes.jumpsTo, ih, Bool.and_assoc]

theorem SNodes.keepsSyms_append (a b : List SNode) :
    SNodes.keepsSyms (a ++ b) = (SNodes.keepsSyms a && SNodes.keepsSyms b) := by
  induction a with
  | nil => rfl
  | cons x a ih => simp [SNodes.keepsSyms, ih, Bool.and_assoc]

theorem SNode.retarget_consumesPending (n : SNode) : (n.retarget L).consumesPending = n.consumesPending := by
  cases n <;> rfl

theorem SNode.retarget_isDynScopeCopy (n : SNode) : (n.retarget L).isDynScopeCopy = n.isDynScopeCopy := by
  cases n <;> rfl

theorem SNode.retarget_loop? (n : SNode) :
    (n.retarget L).loop? = n.loop?.map fun p => (p.1, SNodes.retarget L p.2) := by
  cases n <;> rfl

theorem SNode.retarget_mentions (t : Name) (n : SNode) : (n.retarget L).mentions t = n.mentions t := by
  cases n <;> rfl

mutual

theorem SNode.retarget_countData (t : Name) : ∀ n : SNode, (n.retarget L).countData t = n.countData t
  | .namedRegion _ cs | .gblock cs | .loopScope _ cs | .forScope _ cs | .whileScope _ cs | .doWhileScope _ cs
  | .ifScope _ cs | .stateIfScope _ cs | .elifScope _ cs | .elseScope cs | .mapScope _ cs
  | .consumeScope _ cs => by
    simp [SNode.retarget, SNode.countData, SNodes.retarget_countData t cs, SNode.mentions] <;> rfl
  | .stateLabel _ | .goto _ | .assign .. | .breakNode | .continueNode | .taskletNode .. | .libraryCall ..
  | .copyNode .. | .dynScopeCopy .. | .viewNode .. | .refSetNode .. => by
    simp only [SNode.retarget, SNode.countData, SNode.mentions] <;> rfl

theorem SNodes.retarget_countData (t : Name) : ∀ cs : List SNode, SNodes.countData t (SNodes.retarget L cs) =
    SNodes.countData t cs
  | [] => rfl
  | c :: cs => by
    simp only [SNodes.retarget, SNodes.countData, SNode.retarget_countData t c, SNodes.retarget_countData t cs]

end

theorem TCtx.privateTransients_retarget (tc : TCtx) (cs : List SNode) :
    tc.privateTransients (SNodes.retarget L cs) = tc.privateTransients cs := by
  simp only [TCtx.privateTransients, SNodes.retarget_countData]

theorem afterLabel_retarget (l : String) :
    ∀ xs : List SNode, afterLabel l (SNodes.retarget L xs) = (afterLabel l xs).map (SNodes.retarget L)
  | [] => rfl
  | x :: xs => by
    cases x <;> simp only [SNodes.retarget, SNode.retarget, afterLabel, afterLabel_retarget l xs] <;> split <;> rfl

theorem resumeAt_retarget (gb : Bool) (full rest : List SNode) (l : String) :
    resumeAt gb (SNodes.retarget L full) (SNodes.retarget L rest) l =
      (resumeAt gb full rest l).map (SNodes.retarget L) := by
  unfold resumeAt
  rw [afterLabel_retarget, afterLabel_retarget]
  cases afterLabel l rest with
  | some _ => rfl
  | none => cases gb <;> rfl

theorem afterLabel_none : ∀ {xs : List SNode}, SNodes.hasLabel L xs = false → afterLabel L xs = none
  | [], _ => rfl
  | x :: xs, h => by
    simp only [SNodes.hasLabel, Bool.or_eq_false_iff] at h
    cases x <;> simp only [afterLabel, afterLabel_none h.2]
    simp only [SNode.hasLabel, beq_eq_false_iff_ne] at h
    simp [h.1]

/-- A list resumed after one of its labels is a suffix of it. -/
theorem afterLabel_suffix {l : String} : ∀ {xs suffix : List SNode}, afterLabel l xs = some suffix → suffix <:+ xs
  | [], _, h => by cases h
  | x :: xs, suffix, h => by
    cases x <;> simp only [afterLabel] at h
    all_goals first
      | exact (afterLabel_suffix h).trans (List.suffix_cons _ _)
      | (split at h
         · cases h; exact List.suffix_cons _ _
         · exact (afterLabel_suffix h).trans (List.suffix_cons _ _))

theorem resumeAt_suffix {gb : Bool} {full rest suffix : List SNode} {l : String}
    (h : resumeAt gb full rest l = some suffix) : suffix <:+ rest ∨ suffix <:+ full := by
  unfold resumeAt at h
  cases hr : afterLabel l rest with
  | some s => rw [hr] at h; cases h; exact .inl (afterLabel_suffix hr)
  | none =>
    rw [hr] at h
    cases gb
    · cases h
    · exact .inr (afterLabel_suffix h)

theorem SNodes.hasLabel_suffix {a b : List SNode} (h : a <:+ b) (hb : SNodes.hasLabel L b = false) :
    SNodes.hasLabel L a = false := by
  obtain ⟨p, rfl⟩ := h
  rw [SNodes.hasLabel_append] at hb
  simp only [Bool.or_eq_false_iff] at hb
  exact hb.2

theorem SNodes.jumpsTo_suffix {J : Option String} {a b : List SNode} (h : a <:+ b) (hb : SNodes.jumpsTo J b = true) :
    SNodes.jumpsTo J a = true := by
  obtain ⟨p, rfl⟩ := h
  rw [SNodes.jumpsTo_append] at hb
  simp only [Bool.and_eq_true] at hb
  exact hb.2

theorem SNodes.keepsSyms_suffix {a b : List SNode} (h : a <:+ b) (hb : SNodes.keepsSyms b = true) :
    SNodes.keepsSyms a = true := by
  obtain ⟨p, rfl⟩ := h
  rw [SNodes.keepsSyms_append] at hb
  simp only [Bool.and_eq_true] at hb
  exact hb.2

theorem leafOut_retarget (r : Res CSt) : (leafOut r).map (NodeOut.retarget L) = leafOut r := by
  cases r <;> rfl

theorem stepLeaf_retarget (tc : TCtx) (sc : SibCtx) (n : SNode) (cst : CSt) :
    stepLeaf tc sc (n.retarget L) cst = (stepLeaf tc sc n cst).map (·.map (NodeOut.retarget L)) := by
  cases n with
  | goto t =>
    cases t with
    | none => rfl
    | some l =>
      by_cases hl : l = L
      · subst hl; simp [SNode.retarget, stepLeaf, NodeOut.retarget, TCtl.retarget]
      · simp [SNode.retarget, stepLeaf, NodeOut.retarget, TCtl.retarget, hl]
  | dynScopeCopy target m =>
    simp only [SNode.retarget, stepLeaf, Option.map_some]
    cases execDynScopeCopy tc cst.ρ sc.pending target m cst.st <;> rfl
  | _ => simp only [SNode.retarget, stepLeaf, Option.map_some, Option.map_none, leafOut_retarget] <;> rfl

theorem scopeOut_retarget (r : Res (TCtl × CSt)) : scopeOut (retargetR L r) = (scopeOut r).map (NodeOut.retarget L) := by
  cases r <;> rfl

theorem chainOut_retarget (r : Res (TCtl × CSt)) : chainOut (retargetR L r) = (chainOut r).map (NodeOut.retarget L) := by
  cases r <;> rfl

theorem dataflowOut_retarget (r : Res CSt) : (dataflowOut r).map (NodeOut.retarget L) = dataflowOut r := by
  cases r <;> rfl

theorem iterLog_retarget (priv : List Name) (r : Res (TCtl × CSt)) : iterLog priv (retargetR L r) = iterLog priv r := by
  cases r with
  | err e => rfl
  | ok p =>
    obtain ⟨ctl, c⟩ := p
    cases ctl with
    | goto l => by_cases hl : l = L <;> simp [retargetR, TCtl.retarget, iterLog, hl]
    | _ => rfl

theorem taskOutcome_retarget (priv : List Name) (l : Loc) (r : Res (TCtl × CSt)) :
    taskOutcome priv l (retargetR L r) = taskOutcome priv l r := by
  cases r with
  | err e => rfl
  | ok p =>
    obtain ⟨ctl, c⟩ := p
    cases ctl with
    | goto l' => by_cases hl : l' = L <;> simp [retargetR, TCtl.retarget, taskOutcome, hl]
    | _ => rfl

theorem TCtl.retarget_of_isJump {ctl : TCtl} (h : ctl.isJump = true) : ctl.retarget L = ctl := by
  cases ctl <;> simp_all [TCtl.isJump, TCtl.retarget]

theorem TCtl.retarget_continuesLoop (ctl : TCtl) : (ctl.retarget L).continuesLoop = ctl.continuesLoop := by
  cases ctl with
  | goto l => by_cases hl : l = L <;> simp [TCtl.retarget, TCtl.continuesLoop, hl]
  | _ => rfl

end Structure

/-! ## Retargeting runs -/

namespace STree

section Rel

variable (L : String)

mutual

/-- **Retargeting a tree run**: a run of a node without label `L` is a run of
the retargeted node, with `goto L` outcomes turned into returns. -/
theorem NodeExec.retarget {tc : TCtx} {sc : SibCtx} {n : SNode} {cst : CSt} {r : Res NodeOut} :
    STree.NodeExec tc sc n cst r → n.hasLabel L = false →
      STree.NodeExec tc sc (n.retarget L) cst (r.map (NodeOut.retarget L))
  | .pendingErr hn hp, _ => .pendingErr (by rw [SNode.retarget_consumesPending]; exact hn) hp
  | .leaf hp hs, _ =>
    .leaf (by rw [SNode.retarget_isDynScopeCopy]; exact hp) (by rw [stepLeaf_retarget, hs]; rfl)
  | .namedRegion hp hl, hok => by
    rw [← scopeOut_retarget]
    exact .namedRegion hp (ListExec.retarget hl hok hok)
  | .gblock hp hl, hok => by
    rw [← scopeOut_retarget]
    exact .gblock hp (ListExec.retarget hl hok hok)
  | .ifCondErr hp hc, _ => .ifCondErr hp hc
  | .ifSkip hp hc, _ => .ifSkip hp hc
  | .ifTaken hp hc hl, hok => by
    rw [← chainOut_retarget]
    exact .ifTaken hp hc (ListExec.retarget hl hok hok)
  | .stateIfCondErr hp hc, _ => .stateIfCondErr hp hc
  | .stateIfSkip hp hc, _ => .stateIfSkip hp hc
  | .stateIfTaken hp hc hl, hok => by
    rw [← scopeOut_retarget]
    exact .stateIfTaken hp hc (ListExec.retarget hl hok hok)
  | .elifNoChain hp hc, _ => .elifNoChain hp hc
  | .elifAfterTaken hp hc, _ => .elifAfterTaken hp hc
  | .elifCondErr hp hc he, _ => .elifCondErr hp hc he
  | .elifSkip hp hc he, _ => .elifSkip hp hc he
  | .elifTaken hp hc he hl, hok => by
    rw [← chainOut_retarget]
    exact .elifTaken hp hc he (ListExec.retarget hl hok hok)
  | .elseNoChain hp hc, _ => .elseNoChain hp hc
  | .elseAfterTaken hp hc, _ => .elseAfterTaken hp hc
  | .elseTaken hp hc hl, hok => by
    rw [← scopeOut_retarget]
    exact .elseTaken hp hc (ListExec.retarget hl hok hok)
  | @STree.NodeExec.loop _ _ n l cs _ _ hp hl hL, hok => by
    have hok' : SNodes.hasLabel L cs = false := by
      cases n <;> simp only [SNode.loop?, reduceCtorEq, Option.some.injEq, Prod.mk.injEq] at hl <;>
        (obtain ⟨rfl, rfl⟩ := hl; simpa [SNode.hasLabel] using hok)
    rw [← scopeOut_retarget]
    exact .loop hp (by rw [SNode.retarget_loop?, hl]; rfl) (LoopExec.retarget hL hok')
  | .mapSetupErr hs, _ => .mapSetupErr hs
  | .mapIterErr hs hi, hok => by
    refine .mapIterErr hs ?_
    rw [TCtx.privateTransients_retarget]
    exact IterExec.retarget hi hok
  | .mapScope hs hi, hok => by
    rw [dataflowOut_retarget]
    refine .mapScope hs ?_
    rw [TCtx.privateTransients_retarget]
    exact IterExec.retarget hi hok
  | .consumeSetupErr hs, _ => .consumeSetupErr hs
  | .consumeTasksErr hs ht, hok => by
    refine .consumeTasksErr hs ?_
    rw [TCtx.privateTransients_retarget]
    exact ConsumeTasks.retarget ht hok
  | .consumeScope hs ht, hok => by
    rw [dataflowOut_retarget]
    refine .consumeScope hs ?_
    rw [TCtx.privateTransients_retarget]
    exact ConsumeTasks.retarget ht hok

theorem ListExec.retarget {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {cs : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} :
    STree.ListExec tc full gb sc cs cst r → SNodes.hasLabel L full = false → SNodes.hasLabel L cs = false →
      STree.ListExec tc (SNodes.retarget L full) gb sc (SNodes.retarget L cs) cst (retargetR L r)
  | .nil hp, _, _ => .nil hp
  | .nilPending hp, _, _ => .nilPending hp
  | .headErr hn, _, hcs => by
    simp only [SNodes.hasLabel, Bool.or_eq_false_iff] at hcs
    exact .headErr (NodeExec.retarget hn hcs.1)
  | .step hn hl, hf, hcs => by
    simp only [SNodes.hasLabel, Bool.or_eq_false_iff] at hcs
    exact .step (NodeExec.retarget hn hcs.1) (ListExec.retarget hl hf hcs.2)
  | .jump (ctl := ctl) hn hc, _, hcs => by
    simp only [SNodes.hasLabel, Bool.or_eq_false_iff] at hcs
    have := NodeExec.retarget hn hcs.1
    simp only [Res.map_ok, NodeOut.retarget, TCtl.retarget_of_isJump L hc] at this
    simp only [retargetR, Res.map_ok, TCtl.retarget_of_isJump L hc]
    exact .jump this hc
  | .gotoResume (l := l) (rest := rest) (suffix := suffix) hn hl hr, hf, hcs => by
    simp only [SNodes.hasLabel, Bool.or_eq_false_iff] at hcs
    have hn' := NodeExec.retarget hn hcs.1
    by_cases hlL : l = L
    · subst hlL
      unfold resumeAt at hl
      rw [afterLabel_none l hcs.2] at hl
      cases gb
      · cases hl
      · rw [afterLabel_none l hf] at hl; cases hl
    · simp only [Res.map_ok, NodeOut.retarget, TCtl.retarget, hlL, if_false] at hn'
      have hsuf : SNodes.hasLabel L suffix = false := by
        rcases resumeAt_suffix hl with h | h
        · exact SNodes.hasLabel_suffix L h hcs.2
        · exact SNodes.hasLabel_suffix L h hf
      exact .gotoResume hn' (by rw [resumeAt_retarget, hl]; rfl) (ListExec.retarget hr hf hsuf)
  | .gotoEscape (l := l) (cst' := cst') hn hl, hf, hcs => by
    simp only [SNodes.hasLabel, Bool.or_eq_false_iff] at hcs
    have hn' := NodeExec.retarget hn hcs.1
    by_cases hlL : l = L
    · subst hlL
      simp only [Res.map_ok, NodeOut.retarget, TCtl.retarget, if_true] at hn'
      simp only [retargetR, Res.map_ok, TCtl.retarget, if_true]
      exact .jump hn' rfl
    · simp only [Res.map_ok, NodeOut.retarget, TCtl.retarget, hlL, if_false] at hn'
      simp only [retargetR, Res.map_ok, TCtl.retarget, hlL, if_false]
      exact .gotoEscape hn' (by rw [resumeAt_retarget, hl]; rfl)

theorem IterExec.retarget {tc : TCtx} {params : List Sym} {body : List SNode} {priv : List Name} {pre : CSt} {pts : List Idx}
    {r : Res (List Log)} :
    STree.IterExec tc params body priv pre pts r → SNodes.hasLabel L body = false →
      STree.IterExec tc params (SNodes.retarget L body) priv pre pts r
  | .nil, _ => .nil
  | .headErr hb he, hok => .headErr (ListExec.retarget hb hok hok) (by rw [iterLog_retarget]; exact he)
  | .tailErr hb hl hi, hok =>
    .tailErr (ListExec.retarget hb hok hok) (by rw [iterLog_retarget]; exact hl) (IterExec.retarget hi hok)
  | .cons hb hl hi, hok =>
    .cons (ListExec.retarget hb hok hok) (by rw [iterLog_retarget]; exact hl) (IterExec.retarget hi hok)

theorem ConsumeTasks.retarget {tc : TCtx} {ρ : Valuation} {c : ConsumeNode} {d : Name} {body : List SNode} {l : Loc} {np : Int}
    {priv : List Name} {pre : CSt} {q : List Val} {r : Res (List Log × List Val × List Val)} :
    STree.ConsumeTasks tc ρ c d body l np priv pre q r → SNodes.hasLabel L body = false →
      STree.ConsumeTasks tc ρ c d (SNodes.retarget L body) l np priv pre q r
  | .condErr hc, _ => .condErr hc
  | .stop hc, _ => .stop hc
  | .bodyErr hc hq hp hb he, hok =>
    .bodyErr hc hq hp (ListExec.retarget hb hok hok) (by rw [taskOutcome_retarget]; exact he)
  | .restErr hc hq hp hb ho hr, hok =>
    .restErr hc hq hp (ListExec.retarget hb hok hok) (by rw [taskOutcome_retarget]; exact ho)
      (ConsumeTasks.retarget hr hok)
  | .step hc hq hp hb ho hr, hok =>
    .step hc hq hp (ListExec.retarget hb hok hok) (by rw [taskOutcome_retarget]; exact ho)
      (ConsumeTasks.retarget hr hok)

theorem LoopExec.retarget {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopExec tc l body cst r → SNodes.hasLabel L body = false →
      STree.LoopExec tc l (SNodes.retarget L body) cst (retargetR L r)
  | .initErr hi, _ => .initErr hi
  | .guardFirst hi hv hg, hok => .guardFirst hi hv (LoopGuard.retarget hg hok)
  | .bodyFirst hi hv hb, hok => .bodyFirst hi hv (LoopIter.retarget hb hok)

theorem LoopGuard.retarget {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopGuard tc l body cst r → SNodes.hasLabel L body = false →
      STree.LoopGuard tc l (SNodes.retarget L body) cst (retargetR L r)
  | .condErr hc, _ => .condErr hc
  | .exit hc, _ => .exit hc
  | .enter hc hb, hok => .enter hc (LoopIter.retarget hb hok)

theorem LoopIter.retarget {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopIter tc l body cst r → SNodes.hasLabel L body = false →
      STree.LoopIter tc l (SNodes.retarget L body) cst (retargetR L r)
  | .bodyErr hb, hok => .bodyErr (ListExec.retarget hb hok hok)
  | .ret hb, hok => .ret (ListExec.retarget hb hok hok)
  | .brk hb, hok => .brk (ListExec.retarget hb hok hok)
  | .gotoOut (lbl := lbl) hb, hok => by
    have hb' := ListExec.retarget hb hok hok
    by_cases hl : lbl = L
    · subst hl
      simp only [retargetR, Res.map_ok, TCtl.retarget, if_true] at hb' ⊢
      exact .ret hb'
    · simp only [retargetR, Res.map_ok, TCtl.retarget, hl, if_false] at hb' ⊢
      exact .gotoOut hb'
  | .tail (ctl := ctl) hb hc ht, hok => by
    have hb' := ListExec.retarget hb hok hok
    simp only [retargetR, Res.map_ok] at hb'
    exact .tail hb' (by rw [TCtl.retarget_continuesLoop]; exact hc) (LoopTail.retarget ht hok)

theorem LoopTail.retarget {tc : TCtx} {l : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopTail tc l body cst r → SNodes.hasLabel L body = false →
      STree.LoopTail tc l (SNodes.retarget L body) cst (retargetR L r)
  | .updateErr hv hu, _ => .updateErr hv hu
  | .updateGuard hv hu hg, hok => .updateGuard hv hu (LoopGuard.retarget hg hok)
  | .condFirstErr hv hc, _ => .condFirstErr hv hc
  | .condFirstExit hv hc, _ => .condFirstExit hv hc
  | .condFirstUpdateErr hv hc hu, _ => .condFirstUpdateErr hv hc hu
  | .condFirstNext hv hc hu hb, hok => .condFirstNext hv hc hu (LoopIter.retarget hb hok)

end

end Rel

/-! ## Escaping gotos -/

section Gotos

variable (L : String)

theorem leafOut_ne_goto {R : Res CSt} {sc' : SibCtx} {l : String} {c : CSt} :
    leafOut R ≠ .ok (sc', .goto l, c) := by
  cases R <;> simp [leafOut]

theorem stepLeaf_goto {tc : TCtx} {sc : SibCtx} {n : SNode} {cst : CSt} {r : Res NodeOut} {sc' : SibCtx} {l : String}
    {c : CSt} (hs : stepLeaf tc sc n cst = some r) (hr : r = .ok (sc', .goto l, c)) : n = .goto (some l) := by
  subst hr
  cases n with
  | goto t =>
    cases t with
    | none => simp [stepLeaf] at hs
    | some l' => simp only [stepLeaf, Option.some.injEq, Res.ok.injEq, Prod.mk.injEq, TCtl.goto.injEq] at hs; rw [hs.2.1]
  | dynScopeCopy target m =>
    simp only [stepLeaf, Option.some.injEq] at hs
    cases h : execDynScopeCopy tc cst.ρ sc.pending target m cst.st <;> rw [h] at hs <;> simp [Res.map] at hs
  | stateLabel _ | breakNode | continueNode => simp [stepLeaf] at hs
  | assign _ _ | taskletNode _ _ _ | libraryCall _ _ _ _ | copyNode _ _ | viewNode _ _ _ _ _ | refSetNode _ _ _ _ =>
    simp only [stepLeaf, Option.some.injEq] at hs
    exact absurd hs leafOut_ne_goto
  | _ => simp [stepLeaf] at hs

theorem scopeOut_goto {r : Res (TCtl × CSt)} {sc' : SibCtx} {l : String} {c : CSt}
    (h : scopeOut r = .ok (sc', .goto l, c)) : r = .ok (.goto l, c) := by
  rcases r with ⟨ctl, c'⟩ | e
  · simp only [scopeOut, Res.map_ok, Res.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl, rfl⟩ := h; rfl
  · cases h

theorem chainOut_goto {r : Res (TCtl × CSt)} {sc' : SibCtx} {l : String} {c : CSt}
    (h : chainOut r = .ok (sc', .goto l, c)) : r = .ok (.goto l, c) := by
  rcases r with ⟨ctl, c'⟩ | e
  · simp only [chainOut, Res.map_ok, Res.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl, rfl⟩ := h; rfl
  · cases h

theorem dataflowOut_goto {r : Res CSt} {sc' : SibCtx} {l : String} {c : CSt}
    (h : dataflowOut r = .ok (sc', .goto l, c)) : False := by
  rcases r with c' | e
  · simp [dataflowOut] at h
  · cases h

theorem leafOut_ctl {R : Res CSt} {o : NodeOut} (h : leafOut R = .ok o) : o.2.1 = .normal := by
  cases R with
  | err e => cases h
  | ok a => simp only [leafOut, Res.map_ok, Res.ok.injEq] at h; subst h; rfl

theorem stepLeaf_okFor {tc : TCtx} {sc : SibCtx} {n : SNode} {cst : CSt} {r : Res NodeOut} {o : NodeOut}
    {J : Option String} (hj : n.jumpsTo J = true) (hs : stepLeaf tc sc n cst = some r) (hr : r = .ok o) :
    o.2.1.okFor J = true := by
  subst hr
  cases n with
  | goto t =>
    cases t with
    | none => simp [SNode.jumpsTo] at hj
    | some l =>
      simp only [stepLeaf, Option.some.injEq] at hs
      cases hs
      simpa [SNode.jumpsTo, TCtl.okFor] using hj
  | dynScopeCopy target m =>
    simp only [stepLeaf, Option.some.injEq] at hs
    cases h : execDynScopeCopy tc cst.ρ sc.pending target m cst.st with
    | err e => rw [h] at hs; cases hs
    | ok p => rw [h] at hs; simp only [Res.map_ok, Res.ok.injEq] at hs; subst hs; rfl
  | stateLabel _ | breakNode | continueNode =>
    simp only [stepLeaf, Option.some.injEq, Res.ok.injEq] at hs; subst hs; rfl
  | assign _ _ | taskletNode _ _ _ | libraryCall _ _ _ _ | copyNode _ _ | viewNode _ _ _ _ _ | refSetNode _ _ _ _ =>
    simp only [stepLeaf, Option.some.injEq] at hs
    rw [leafOut_ctl hs]; rfl
  | _ => simp [stepLeaf] at hs

theorem scopeOut_ctl {r : Res (TCtl × CSt)} {o : NodeOut} (h : scopeOut r = .ok o) : r = .ok (o.2.1, o.2.2) := by
  rcases r with ⟨ctl, c'⟩ | e
  · simp only [scopeOut, Res.map_ok, Res.ok.injEq] at h; subst h; rfl
  · cases h

theorem chainOut_ctl {r : Res (TCtl × CSt)} {o : NodeOut} (h : chainOut r = .ok o) : r = .ok (o.2.1, o.2.2) := by
  rcases r with ⟨ctl, c'⟩ | e
  · simp only [chainOut, Res.map_ok, Res.ok.injEq] at h; subst h; rfl
  · cases h

theorem dataflowOut_ctl {r : Res CSt} {o : NodeOut} (h : dataflowOut r = .ok o) : o.2.1 = .normal := by
  rcases r with c' | e
  · simp only [dataflowOut, Res.map_ok, Res.ok.injEq] at h; subst h; rfl
  · cases h

variable (J : Option String)

mutual

/-- A run of a node with `jumpsTo J` ends in a signal `okFor J`: never a
return, and a goto only to `J`. -/
theorem NodeExec.okFor {tc : TCtx} {sc : SibCtx} {n : SNode} {cst : CSt} {r : Res NodeOut} :
    STree.NodeExec tc sc n cst r → n.jumpsTo J = true → ∀ o, r = .ok o → o.2.1.okFor J = true
  | .pendingErr .., _, _, hr => by cases hr
  | .leaf _ hs, hj, o, hr => stepLeaf_okFor hj hs hr
  | .namedRegion _ hl, hj, o, hr => ListExec.okFor hl hj hj _ _ (scopeOut_ctl hr)
  | .gblock _ hl, hj, o, hr => ListExec.okFor hl hj hj _ _ (scopeOut_ctl hr)
  | .ifCondErr .., _, _, hr => by cases hr
  | .ifSkip .., _, _, hr => by cases hr; rfl
  | .ifTaken _ _ hl, hj, o, hr => ListExec.okFor hl hj hj _ _ (chainOut_ctl hr)
  | .stateIfCondErr .., _, _, hr => by cases hr
  | .stateIfSkip .., _, _, hr => by cases hr; rfl
  | .stateIfTaken _ _ hl, hj, o, hr => ListExec.okFor hl hj hj _ _ (scopeOut_ctl hr)
  | .elifNoChain .., _, _, hr => by cases hr
  | .elifAfterTaken .., _, _, hr => by cases hr; rfl
  | .elifCondErr .., _, _, hr => by cases hr
  | .elifSkip .., _, _, hr => by cases hr; rfl
  | .elifTaken _ _ _ hl, hj, o, hr => ListExec.okFor hl hj hj _ _ (chainOut_ctl hr)
  | .elseNoChain .., _, _, hr => by cases hr
  | .elseAfterTaken .., _, _, hr => by cases hr; rfl
  | .elseTaken _ _ hl, hj, o, hr => ListExec.okFor hl hj hj _ _ (scopeOut_ctl hr)
  | @STree.NodeExec.loop _ _ n lm cs _ _ _ hl hL, hj, o, hr => by
    have hj' : SNodes.jumpsTo J cs = true := by
      cases n <;> simp only [SNode.loop?, reduceCtorEq, Option.some.injEq, Prod.mk.injEq] at hl <;>
        (obtain ⟨rfl, rfl⟩ := hl; simpa [SNode.jumpsTo] using hj)
    exact LoopExec.okFor hL hj' _ _ (scopeOut_ctl hr)
  | .mapSetupErr .., _, _, hr => by cases hr
  | .mapIterErr .., _, _, hr => by cases hr
  | .mapScope .., _, o, hr => by rw [dataflowOut_ctl hr]; rfl
  | .consumeSetupErr .., _, _, hr => by cases hr
  | .consumeTasksErr .., _, _, hr => by cases hr
  | .consumeScope .., _, o, hr => by rw [dataflowOut_ctl hr]; rfl

theorem ListExec.okFor {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {cs : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} :
    STree.ListExec tc full gb sc cs cst r → SNodes.jumpsTo J full = true → SNodes.jumpsTo J cs = true →
      ∀ ctl c, r = .ok (ctl, c) → ctl.okFor J = true
  | .nil _, _, _, _, _, hr => by cases hr; rfl
  | .nilPending _, _, _, _, _, hr => by cases hr
  | .headErr _, _, _, _, _, hr => by cases hr
  | .step _ hl, hf, hcs, ctl, c, hr => by
    simp only [SNodes.jumpsTo, Bool.and_eq_true] at hcs
    exact ListExec.okFor hl hf hcs.2 ctl c hr
  | .jump hn _, _, hcs, ctl, c, hr => by
    simp only [SNodes.jumpsTo, Bool.and_eq_true] at hcs
    simp only [Res.ok.injEq, Prod.mk.injEq] at hr
    obtain ⟨rfl, rfl⟩ := hr
    exact NodeExec.okFor hn hcs.1 _ rfl
  | .gotoResume (suffix := suffix) _ hl hr', hf, hcs, ctl, c, hr => by
    simp only [SNodes.jumpsTo, Bool.and_eq_true] at hcs
    have hsuf : SNodes.jumpsTo J suffix = true := by
      rcases resumeAt_suffix hl with h | h
      · exact SNodes.jumpsTo_suffix h hcs.2
      · exact SNodes.jumpsTo_suffix h hf
    exact ListExec.okFor hr' hf hsuf ctl c hr
  | .gotoEscape hn _, _, hcs, ctl, c, hr => by
    simp only [SNodes.jumpsTo, Bool.and_eq_true] at hcs
    simp only [Res.ok.injEq, Prod.mk.injEq] at hr
    obtain ⟨rfl, rfl⟩ := hr
    exact NodeExec.okFor hn hcs.1 _ rfl

theorem LoopExec.okFor {tc : TCtx} {lm : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopExec tc lm body cst r → SNodes.jumpsTo J body = true → ∀ ctl c, r = .ok (ctl, c) → ctl.okFor J = true
  | .initErr _, _, _, _, hr => by cases hr
  | .guardFirst _ _ hg, hj, ctl, c, hr => LoopGuard.okFor hg hj ctl c hr
  | .bodyFirst _ _ hb, hj, ctl, c, hr => LoopIter.okFor hb hj ctl c hr

theorem LoopGuard.okFor {tc : TCtx} {lm : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopGuard tc lm body cst r → SNodes.jumpsTo J body = true → ∀ ctl c, r = .ok (ctl, c) → ctl.okFor J = true
  | .condErr _, _, _, _, hr => by cases hr
  | .exit _, _, _, _, hr => by cases hr; rfl
  | .enter _ hb, hj, ctl, c, hr => LoopIter.okFor hb hj ctl c hr

theorem LoopIter.okFor {tc : TCtx} {lm : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopIter tc lm body cst r → SNodes.jumpsTo J body = true → ∀ ctl c, r = .ok (ctl, c) → ctl.okFor J = true
  | .bodyErr _, _, _, _, hr => by cases hr
  | .ret hb, hj, ctl, c, hr => by
    simp only [Res.ok.injEq, Prod.mk.injEq] at hr
    obtain ⟨rfl, rfl⟩ := hr
    exact ListExec.okFor hb hj hj _ _ rfl
  | .brk _, _, _, _, hr => by cases hr; rfl
  | .gotoOut hb, hj, ctl, c, hr => ListExec.okFor hb hj hj ctl c hr
  | .tail _ _ ht, hj, ctl, c, hr => LoopTail.okFor ht hj ctl c hr

theorem LoopTail.okFor {tc : TCtx} {lm : LoopMeta} {body : List SNode} {cst : CSt} {r : Res (TCtl × CSt)} :
    STree.LoopTail tc lm body cst r → SNodes.jumpsTo J body = true → ∀ ctl c, r = .ok (ctl, c) → ctl.okFor J = true
  | .updateErr .., _, _, _, hr => by cases hr
  | .updateGuard _ _ hg, hj, ctl, c, hr => LoopGuard.okFor hg hj ctl c hr
  | .condFirstErr .., _, _, _, hr => by cases hr
  | .condFirstExit .., _, _, _, hr => by cases hr; rfl
  | .condFirstUpdateErr .., _, _, _, hr => by cases hr
  | .condFirstNext _ _ _ hb, hj, ctl, c, hr => LoopIter.okFor hb hj ctl c hr

end

/-- A goto escaping a run of a node whose gotos all target `L` is a `goto L`. -/
theorem NodeExec.goto_target (L : String) {tc : TCtx} {sc : SibCtx} {n : SNode} {cst : CSt} {r : Res NodeOut}
    (h : STree.NodeExec tc sc n cst r) (hj : n.jumpsTo (some L) = true) :
    ∀ sc' l c, r = .ok (sc', .goto l, c) → l = L := by
  intro sc' l c hr
  have := NodeExec.okFor _ h hj _ hr
  simpa [TCtl.okFor] using this

end Gotos

/-! ## Runs that keep the valuation -/

section Keeps

theorem leafOut_lift_ρ {cst : CSt} {R : Res St} {o : NodeOut} (h : leafOut (cst.lift R) = .ok o) :
    o.2.2.ρ = cst.ρ := by
  cases R with
  | err e => cases h
  | ok a => simp only [leafOut, CSt.lift, Res.map_ok, Res.ok.injEq] at h; subst h; rfl

theorem stepLeaf_ρ {tc : TCtx} {sc : SibCtx} {n : SNode} {cst : CSt} {r : Res NodeOut} {o : NodeOut}
    (hn : n.keepsSyms = true) (hs : stepLeaf tc sc n cst = some r) (hr : r = .ok o) : o.2.2.ρ = cst.ρ := by
  subst hr
  cases n with
  | assign _ _ => simp [SNode.keepsSyms] at hn
  | dynScopeCopy target m =>
    simp only [stepLeaf, Option.some.injEq] at hs
    cases h : execDynScopeCopy tc cst.ρ sc.pending target m cst.st with
    | err e => rw [h] at hs; cases hs
    | ok p => rw [h] at hs; simp only [Res.map_ok, Res.ok.injEq] at hs; subst hs; rfl
  | goto t => cases t <;> (simp only [stepLeaf, Option.some.injEq, Res.ok.injEq] at hs; subst hs; rfl)
  | stateLabel _ | breakNode | continueNode =>
    simp only [stepLeaf, Option.some.injEq, Res.ok.injEq] at hs; subst hs; rfl
  | taskletNode _ _ _ | libraryCall _ _ _ _ | copyNode _ _ | viewNode _ _ _ _ _ | refSetNode _ _ _ _ =>
    simp only [stepLeaf, Option.some.injEq] at hs
    exact leafOut_lift_ρ hs
  | _ => simp [stepLeaf] at hs

theorem scopeOut_ρ {r : Res (TCtl × CSt)} {o : NodeOut} (h : scopeOut r = .ok o) :
    ∃ ctl, r = .ok (ctl, o.2.2) := by
  rcases r with ⟨ctl, c⟩ | e
  · simp only [scopeOut, Res.map_ok, Res.ok.injEq] at h; subst h; exact ⟨ctl, rfl⟩
  · cases h

theorem chainOut_ρ {r : Res (TCtl × CSt)} {o : NodeOut} (h : chainOut r = .ok o) :
    ∃ ctl, r = .ok (ctl, o.2.2) := by
  rcases r with ⟨ctl, c⟩ | e
  · simp only [chainOut, Res.map_ok, Res.ok.injEq] at h; subst h; exact ⟨ctl, rfl⟩
  · cases h

theorem dataflowOut_withSt_ρ {pre : CSt} {r : Res St} {o : NodeOut} (h : dataflowOut (r.map pre.withSt) = .ok o) :
    o.2.2.ρ = pre.ρ := by
  rcases r with a | e
  · simp only [dataflowOut, Res.map_ok, Res.ok.injEq] at h; subst h; rfl
  · cases h

mutual

/-- A run of a node without symbol assignments and loops keeps the valuation. -/
theorem NodeExec.keeps_ρ {tc : TCtx} {sc : SibCtx} {n : SNode} {cst : CSt} {r : Res NodeOut} :
    STree.NodeExec tc sc n cst r → n.keepsSyms = true → ∀ o, r = .ok o → o.2.2.ρ = cst.ρ
  | .pendingErr .., _, _, hr => by cases hr
  | .leaf _ hs, hk, o, hr => stepLeaf_ρ hk hs hr
  | .namedRegion _ hl, hk, o, hr => by
    obtain ⟨ctl, hr'⟩ := scopeOut_ρ hr; exact ListExec.keeps_ρ hl hk hk ctl _ hr'
  | .gblock _ hl, hk, o, hr => by
    obtain ⟨ctl, hr'⟩ := scopeOut_ρ hr; exact ListExec.keeps_ρ hl hk hk ctl _ hr'
  | .ifCondErr .., _, _, hr => by cases hr
  | .ifSkip .., _, _, hr => by cases hr; rfl
  | .ifTaken _ _ hl, hk, o, hr => by
    obtain ⟨ctl, hr'⟩ := chainOut_ρ hr; exact ListExec.keeps_ρ hl hk hk ctl _ hr'
  | .stateIfCondErr .., _, _, hr => by cases hr
  | .stateIfSkip .., _, _, hr => by cases hr; rfl
  | .stateIfTaken _ _ hl, hk, o, hr => by
    obtain ⟨ctl, hr'⟩ := scopeOut_ρ hr; exact ListExec.keeps_ρ hl hk hk ctl _ hr'
  | .elifNoChain .., _, _, hr => by cases hr
  | .elifAfterTaken .., _, _, hr => by cases hr; rfl
  | .elifCondErr .., _, _, hr => by cases hr
  | .elifSkip .., _, _, hr => by cases hr; rfl
  | .elifTaken _ _ _ hl, hk, o, hr => by
    obtain ⟨ctl, hr'⟩ := chainOut_ρ hr; exact ListExec.keeps_ρ hl hk hk ctl _ hr'
  | .elseNoChain .., _, _, hr => by cases hr
  | .elseAfterTaken .., _, _, hr => by cases hr; rfl
  | .elseTaken _ _ hl, hk, o, hr => by
    obtain ⟨ctl, hr'⟩ := scopeOut_ρ hr; exact ListExec.keeps_ρ hl hk hk ctl _ hr'
  | @STree.NodeExec.loop _ _ n _ _ _ _ _ hl _, hk, _, _ => by
    cases n <;> simp [SNode.loop?, SNode.keepsSyms] at hl hk
  | .mapSetupErr .., _, _, hr => by cases hr
  | .mapIterErr .., _, _, hr => by cases hr
  | .mapScope _ _, _, o, hr => dataflowOut_withSt_ρ hr
  | .consumeSetupErr .., _, _, hr => by cases hr
  | .consumeTasksErr .., _, _, hr => by cases hr
  | .consumeScope _ _, _, o, hr => dataflowOut_withSt_ρ hr

theorem ListExec.keeps_ρ {tc : TCtx} {full : List SNode} {gb : Bool} {sc : SibCtx} {cs : List SNode} {cst : CSt}
    {r : Res (TCtl × CSt)} :
    STree.ListExec tc full gb sc cs cst r → SNodes.keepsSyms full = true → SNodes.keepsSyms cs = true →
      ∀ ctl c, r = .ok (ctl, c) → c.ρ = cst.ρ
  | .nil _, _, _, _, _, hr => by cases hr; rfl
  | .nilPending _, _, _, _, _, hr => by cases hr
  | .headErr _, _, _, _, _, hr => by cases hr
  | .step hn hl, hf, hcs, ctl, c, hr => by
    simp only [SNodes.keepsSyms, Bool.and_eq_true] at hcs
    rw [ListExec.keeps_ρ hl hf hcs.2 ctl c hr]
    exact NodeExec.keeps_ρ hn hcs.1 _ rfl
  | .jump hn _, _, hcs, _, _, hr => by
    simp only [SNodes.keepsSyms, Bool.and_eq_true] at hcs
    simp only [Res.ok.injEq, Prod.mk.injEq] at hr
    obtain ⟨rfl, rfl⟩ := hr
    exact NodeExec.keeps_ρ hn hcs.1 _ rfl
  | .gotoResume (suffix := suffix) hn hl hr', hf, hcs, ctl, c, hr => by
    simp only [SNodes.keepsSyms, Bool.and_eq_true] at hcs
    have hsuf : SNodes.keepsSyms suffix = true := by
      rcases resumeAt_suffix hl with h | h
      · exact SNodes.keepsSyms_suffix h hcs.2
      · exact SNodes.keepsSyms_suffix h hf
    rw [ListExec.keeps_ρ hr' hf hsuf ctl c hr]
    exact NodeExec.keeps_ρ hn hcs.1 _ rfl
  | .gotoEscape hn _, _, hcs, _, _, hr => by
    simp only [SNodes.keepsSyms, Bool.and_eq_true] at hcs
    simp only [Res.ok.injEq, Prod.mk.injEq] at hr
    obtain ⟨rfl, rfl⟩ := hr
    exact NodeExec.keeps_ρ hn hcs.1 _ rfl

end

end Keeps

/-! ## Splitting a list after a label -/

section Split

variable (L : String)

theorem afterLabel_append_label {rest : List SNode} :
    ∀ {pre : List SNode}, SNodes.hasLabel L pre = false → afterLabel L (pre ++ .stateLabel L :: rest) = some rest
  | [], _ => by simp [afterLabel]
  | x :: pre, h => by
    simp only [SNodes.hasLabel, Bool.or_eq_false_iff] at h
    cases x <;> simp only [List.cons_append, afterLabel, afterLabel_append_label h.2]
    simp only [SNode.hasLabel, beq_eq_false_iff_ne] at h
    simp [h.1]

theorem resumeAt_of_afterLabel {gb : Bool} {full xs suffix : List SNode} {l : String}
    (h : afterLabel l xs = some suffix) : resumeAt gb full xs l = some suffix := by
  simp [resumeAt, h]

/-- A label never fails, jumps or changes the state. -/
theorem NodeExec.label_inv {tc : TCtx} {sc : SibCtx} {cst : CSt} {r : Res NodeOut}
    (h : STree.NodeExec tc sc (.stateLabel L) cst r) : ∀ o, r = .ok o → o = ({}, .normal, cst) := by
  intro o hr
  generalize hn : SNode.stateLabel L = n at h
  cases h with
  | pendingErr => cases hr
  | leaf _ hs =>
    subst hn hr
    simp only [stepLeaf, Option.some.injEq, Res.ok.injEq] at hs
    exact hs.symm
  | loop _ hl _ => subst hn; simp [SNode.loop?] at hl
  | _ => cases hn

/-- **Splitting at a trailing label**: a run of `pre ++ label L :: rest`,
where `pre` has no label `L` and all its gotos target `L`, either fails or
jumps out as the run of `pre ++ [label L]` alone does, or the run of
`pre ++ [label L]` ends normally and `rest` runs from there. -/
theorem ListExec.split_label {tc : TCtx} {full : List SNode} {gb : Bool} {rest : List SNode} :
    ∀ {pre : List SNode} {sc : SibCtx} {cst : CSt} {r : Res (TCtl × CSt)},
      SNodes.hasLabel L pre = false → SNodes.jumpsTo (some L) pre = true →
      STree.ListExec tc full gb sc (pre ++ .stateLabel L :: rest) cst r →
      (∃ e, r = .err e ∧ ∀ full', STree.ListExec tc full' false sc (pre ++ [.stateLabel L]) cst (.err e)) ∨
      (∃ ctl c, ctl.isJump = true ∧ r = .ok (ctl, c) ∧
        ∀ full', STree.ListExec tc full' false sc (pre ++ [.stateLabel L]) cst (.ok (ctl, c))) ∨
      ∃ c, (∀ full', STree.ListExec tc full' false sc (pre ++ [.stateLabel L]) cst (.ok (.normal, c))) ∧
        STree.ListExec tc full gb {} rest c r
  | [], sc, cst, r, _, _, hx => by
    rw [List.nil_append] at hx
    cases hx with
    | headErr hn => exact .inl ⟨_, rfl, fun _ => .headErr hn⟩
    | step hn hl =>
      have hlab := NodeExec.label_inv L hn _ rfl
      simp only [Prod.mk.injEq] at hlab
      obtain ⟨rfl, -, rfl⟩ := hlab
      exact .inr (.inr ⟨_, fun _ => .step hn (.nil rfl), hl⟩)
    | jump hn hc => exact .inr (.inl ⟨_, _, hc, rfl, fun _ => .jump hn hc⟩)
    | gotoResume hn _ _ => cases NodeExec.label_inv L hn _ rfl
    | gotoEscape hn _ => cases NodeExec.label_inv L hn _ rfl
  | n :: pre, sc, cst, r, hl, hj, hx => by
    simp only [SNodes.hasLabel, Bool.or_eq_false_iff] at hl
    simp only [SNodes.jumpsTo, Bool.and_eq_true] at hj
    rw [List.cons_append] at hx
    cases hx with
    | headErr hn => exact .inl ⟨_, rfl, fun _ => .headErr hn⟩
    | step hn hr =>
      rcases ListExec.split_label hl.2 hj.2 hr with ⟨e, rfl, h⟩ | ⟨ctl, c, hc, rfl, h⟩ | ⟨c, h, hrest⟩
      · exact .inl ⟨e, rfl, fun full' => .step hn (h full')⟩
      · exact .inr (.inl ⟨ctl, c, hc, rfl, fun full' => .step hn (h full')⟩)
      · exact .inr (.inr ⟨c, fun full' => .step hn (h full'), hrest⟩)
    | jump hn hc => exact .inr (.inl ⟨_, _, hc, rfl, fun _ => .jump hn hc⟩)
    | gotoResume hn hres hr =>
      have hL := NodeExec.goto_target L hn hj.1 _ _ _ rfl
      subst hL
      rw [resumeAt_of_afterLabel (afterLabel_append_label _ hl.2)] at hres
      cases hres
      exact .inr (.inr ⟨_, fun full' => .gotoResume hn
        (resumeAt_of_afterLabel (afterLabel_append_label (rest := []) _ hl.2)) (.nil rfl), hr⟩)
    | gotoEscape hn hres =>
      have hL := NodeExec.goto_target L hn hj.1 _ _ _ rfl
      subst hL
      rw [resumeAt_of_afterLabel (afterLabel_append_label _ hl.2)] at hres
      cases hres

/-- A label fails only on pending dynamic inputs. -/
theorem NodeExec.label_err {tc : TCtx} {sc : SibCtx} {cst : CSt} {e : ErrReason}
    (h : STree.NodeExec tc sc (.stateLabel L) cst (.err e)) : e = pendingError ∧ sc.pending.isEmpty = false := by
  generalize hn : SNode.stateLabel L = n at h
  generalize hr : (Res.err e : Res NodeOut) = r at h
  cases h with
  | pendingErr _ hp => cases hr; exact ⟨rfl, hp⟩
  | leaf _ hs => subst hn; simp only [stepLeaf, Option.some.injEq] at hs; rw [← hs] at hr; cases hr
  | loop _ hl _ => subst hn; simp [SNode.loop?] at hl
  | _ => cases hn

/-- A label runs normally only without pending dynamic inputs. -/
theorem NodeExec.label_ok {tc : TCtx} {sc : SibCtx} {cst : CSt} {o : NodeOut}
    (h : STree.NodeExec tc sc (.stateLabel L) cst (.ok o)) : sc.pending.isEmpty = true := by
  generalize hn : SNode.stateLabel L = n at h
  generalize hr : (Res.ok o : Res NodeOut) = r at h
  cases h with
  | pendingErr _ hp => cases hr
  | leaf hp _ => subst hn; simpa [SNode.isDynScopeCopy] using hp
  | loop _ hl _ => subst hn; simp [SNode.loop?] at hl
  | _ => cases hn

/-- The outcome `r` of `pre ++ [label L]` and `r'` of `pre` retargeted: a
normal end may be a return of the retargeted list. -/
def UnlabelRel (r r' : Res (TCtl × CSt)) : Prop :=
  (∃ e, r = .err e ∧ r' = .err e) ∨ (∃ c, r = .ok (.normal, c) ∧ (r' = .ok (.normal, c) ∨ r' = .ok (.ret, c))) ∨
  (∃ ctl c, ctl.isJump = true ∧ ctl ≠ .ret ∧ r = .ok (ctl, c) ∧ r' = .ok (ctl, c))

/-- **Removing a trailing label**: a run of `pre ++ [label L]`, where `pre`
has no label `L` and all its gotos target `L`, is matched by a run of `pre`
with `goto L` retargeted to returns. -/
theorem ListExec.unlabel {tc : TCtx} {full full' : List SNode} :
    ∀ {pre : List SNode} {sc : SibCtx} {cst : CSt} {r : Res (TCtl × CSt)},
      SNodes.hasLabel L pre = false → SNodes.jumpsTo (some L) pre = true →
      STree.ListExec tc full false sc (pre ++ [.stateLabel L]) cst r →
      ∃ r', STree.ListExec tc full' false sc (SNodes.retarget L pre) cst r' ∧ UnlabelRel r r'
  | [], sc, cst, r, _, _, hx => by
    rw [List.nil_append] at hx
    cases hx with
    | headErr hn =>
      obtain ⟨rfl, hp⟩ := NodeExec.label_err L hn
      exact ⟨_, .nilPending hp, .inl ⟨_, rfl, rfl⟩⟩
    | step hn hl =>
      have hp := NodeExec.label_ok L hn
      have hlab := NodeExec.label_inv L hn _ rfl
      simp only [Prod.mk.injEq] at hlab
      obtain ⟨rfl, -, rfl⟩ := hlab
      rw [STree.ListExec_nil rfl hl]
      exact ⟨_, .nil hp, .inr (.inl ⟨_, rfl, .inl rfl⟩)⟩
    | jump hn hc =>
      have hlab := NodeExec.label_inv L hn _ rfl
      simp only [Prod.mk.injEq] at hlab
      obtain ⟨-, rfl, -⟩ := hlab
      cases hc
    | gotoResume hn _ _ => cases NodeExec.label_inv L hn _ rfl
    | gotoEscape hn _ => cases NodeExec.label_inv L hn _ rfl
  | n :: pre, sc, cst, r, hl, hj, hx => by
    simp only [SNodes.hasLabel, Bool.or_eq_false_iff] at hl
    simp only [SNodes.jumpsTo, Bool.and_eq_true] at hj
    rw [List.cons_append] at hx
    cases hx with
    | headErr hn => exact ⟨_, .headErr (NodeExec.retarget L hn hl.1), .inl ⟨_, rfl, rfl⟩⟩
    | step hn hr =>
      obtain ⟨r', hr', hrel⟩ := ListExec.unlabel hl.2 hj.2 hr
      exact ⟨r', .step (NodeExec.retarget L hn hl.1) hr', hrel⟩
    | jump hn hc =>
      have hok := NodeExec.okFor _ hn hj.1 _ rfl
      have hn' := NodeExec.retarget L hn hl.1
      simp only [Res.map_ok, NodeOut.retarget, TCtl.retarget_of_isJump L hc] at hn'
      refine ⟨_, .jump hn' hc, .inr (.inr ⟨_, _, hc, fun h => ?_, rfl, rfl⟩)⟩
      subst h; cases hok
    | gotoResume hn hres hr =>
      have hL := NodeExec.goto_target L hn hj.1 _ _ _ rfl
      subst hL
      rw [resumeAt_of_afterLabel (afterLabel_append_label (rest := []) _ hl.2)] at hres
      cases hres
      rw [STree.ListExec_nil rfl hr]
      have hn' := NodeExec.retarget _ hn hl.1
      simp only [Res.map_ok, NodeOut.retarget, TCtl.retarget, if_true] at hn'
      exact ⟨_, .jump hn' rfl, .inr (.inl ⟨_, rfl, .inr rfl⟩)⟩
    | gotoEscape hn hres =>
      have hL := NodeExec.goto_target L hn hj.1 _ _ _ rfl
      subst hL
      rw [resumeAt_of_afterLabel (afterLabel_append_label (rest := []) _ hl.2)] at hres
      cases hres

end Split

end STree

end Dace
