-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Wcr

/-!
# Access logs and their parallel merge

A **log** records what one sequential execution (a map iteration, a consume
task, a state component, a nested-SDFG invocation) did to memory it did not
own: the locations it read from the initial memory, its plain writes, its
WCR contributions, alias (reference) bindings, and its effect on stream
queues.  Logs of parallel executions are **merged** onto the initial memory:

* two logs conflict (`error race`) if one plainly writes a location the other
  reads, writes or updates, if both update a location with different
  operators, if both bind the same alias, or if both have exclusive (popping)
  access to the same queue;
* otherwise plain writes are applied, WCR contributions are folded onto the
  initial value, aliases are bound, and queue effects are applied.

`merge` is defined on a list of logs but, for conflict-free lists, its result
is invariant under permutation of the list (`merge_perm`, assuming the WCR
semantics is lawful); this is what makes map iterations unordered.
Pushes to a shared queue by several parallel logs are appended in list order
(`Decisions.md#D25`).
-/

namespace Dace

/-- Effect of one execution on a queue. -/
inductive QueueEffect where
  /-- Only pushed these values (mergeable with other pushers). -/
  | pushOnly (vs : List Val)
  /-- Popped (and possibly pushed): requires exclusive access; `final` is the
      resulting queue content. -/
  | exclusive (final : List Val)
  deriving Inhabited, Repr

/-- A log.  All locations are stored **resolved** (aliases followed). -/
structure Log where
  reads : List Loc := []
  writes : List (Loc × Val) := []
  wcrs : List (Loc × WcrOp × Val) := []
  aliasSets : List (Name × AliasBinding) := []
  queues : List (Loc × QueueEffect) := []
  deriving Inhabited

namespace Log

def empty : Log := {}

/-- Sequential composition of two logs of the *same* execution.  Reads of
`b` that `a` wrote are not reads of the initial memory; writes and WCR
contributions of `a` that `b` overwrites are dead.  The resulting log
satisfies: every location has at most one plain write, and a plain write to a
location precedes all WCR contributions to it (so `merge` may apply all
writes before all contributions). -/
def seq (a b : Log) : Log :=
  let aWritten := a.writes.map Prod.fst ++ a.wcrs.map (·.1)
  let bWritten := b.writes.map Prod.fst
  { reads := a.reads ++ b.reads.filter (fun l => !aWritten.contains l)
    writes := a.writes.filter (fun (l, _) => !bWritten.contains l) ++ b.writes
    wcrs := a.wcrs.filter (fun (l, _, _) => !bWritten.contains l) ++ b.wcrs
    aliasSets := a.aliasSets ++ b.aliasSets
    queues := a.queues ++ b.queues }

def writeLocs (g : Log) : List Loc := g.writes.map Prod.fst
def wcrLocs (g : Log) : List Loc := g.wcrs.map (·.1)
def exclusiveQueues (g : Log) : List Loc :=
  g.queues.filterMap fun (l, e) => match e with | .exclusive _ => some l | .pushOnly _ => none
def pushedQueues (g : Log) : List Loc := g.queues.map Prod.fst

/-- Conflict between two logs of *parallel* executions. -/
def conflicts (a b : Log) : Bool :=
  let aw := a.writeLocs
  let bw := b.writeLocs
  let awc := a.wcrLocs
  let bwc := b.wcrLocs
  -- write/write, write/read, write/wcr in either direction
  aw.any (fun l => bw.contains l || b.reads.contains l || bwc.contains l) ||
  bw.any (fun l => a.reads.contains l || awc.contains l) ||
  -- wcr/wcr with different operators on the same location
  a.wcrs.any (fun (l, op, _) => b.wcrs.any fun (l', op', _) => l = l' && op ≠ op') ||
  -- a location updated by one and read by the other
  awc.any (fun l => b.reads.contains l) || bwc.any (fun l => a.reads.contains l) ||
  -- alias bindings
  a.aliasSets.any (fun (n, _) => b.aliasSets.any fun (n', _) => n = n') ||
  -- exclusive queue access
  a.exclusiveQueues.any (fun l => b.pushedQueues.contains l) ||
  b.exclusiveQueues.any (fun l => a.pushedQueues.contains l)

/-- No two distinct logs of the list conflict. -/
def pairwiseFree : List Log → Bool
  | [] => true
  | g :: gs => gs.all (fun h => !conflicts g h) && pairwiseFree gs

/-- Union of parallel logs as seen by the enclosing scope. -/
def unionAll (gs : List Log) : Log :=
  { reads := gs.flatMap (·.reads)
    writes := gs.flatMap (·.writes)
    wcrs := gs.flatMap (·.wcrs)
    aliasSets := gs.flatMap (·.aliasSets)
    queues := gs.flatMap (·.queues) }

/-- Drop every entry touching the given (private) containers. -/
def dropNames (g : Log) (priv : List Name) : Log :=
  { reads := g.reads.filter (fun l => !priv.contains l.name)
    writes := g.writes.filter (fun (l, _) => !priv.contains l.name)
    wcrs := g.wcrs.filter (fun (l, _, _) => !priv.contains l.name)
    aliasSets := g.aliasSets.filter (fun (n, _) => !priv.contains n)
    queues := g.queues.filter (fun (l, _) => !priv.contains l.name) }

end Log

/-! ## Merge -/

inductive MergeError where
  | race (l : Loc)
  | aliasRace (n : Name)
  | queueRace (l : Loc)
  deriving Repr, Inhabited

namespace Merge

/-- Apply plain writes (in list order; conflict-free lists have distinct
locations, so the order is irrelevant). -/
def applyWrites (m : Mem) (ws : List (Loc × Val)) : Mem :=
  ws.foldl (fun acc (l, v) => { acc with cells := Function.update acc.cells l v }) m

/-- Fold WCR contributions onto the memory. -/
def applyWcrs (w : WcrSem) (m : Mem) (cs : List (Loc × WcrOp × Val)) : Mem :=
  cs.foldl (fun acc (l, op, v) => { acc with cells := Function.update acc.cells l (w op (acc.cells l) v) }) m

def applyAliases (m : Mem) (as : List (Name × AliasBinding)) : Mem :=
  as.foldl (fun acc (n, b) => { acc with aliases := Function.update acc.aliases n (some b) }) m

def applyQueues (m : Mem) (qs : List (Loc × QueueEffect)) : Mem :=
  qs.foldl (fun acc (l, e) =>
    match e with
    | .pushOnly vs => { acc with queues := Function.update acc.queues l (acc.queues l ++ vs) }
    | .exclusive final => { acc with queues := Function.update acc.queues l final }) m

/-- Find the first conflicting pair, if any. -/
def findConflict : List Log → Option (Log × Log)
  | [] => none
  | g :: gs =>
    match gs.find? (fun h => Log.conflicts g h) with
    | some h => some (g, h)
    | none => findConflict gs

end Merge

/-- Merge the logs of parallel executions onto the initial memory `m₀`. -/
def merge (w : WcrSem) (m₀ : Mem) (gs : List Log) : Except MergeError Mem :=
  if Log.pairwiseFree gs then
    let u := Log.unionAll gs
    .ok (Merge.applyQueues (Merge.applyAliases (Merge.applyWcrs w (Merge.applyWrites m₀ u.writes) u.wcrs) u.aliasSets) u.queues)
  else
    match Merge.findConflict gs with
    | some (g, h) =>
      -- report a witness location
      match g.writeLocs.find? (fun l => h.writeLocs.contains l || h.reads.contains l || h.wcrLocs.contains l) with
      | some l => .error (.race l)
      | none =>
        match h.writeLocs.find? (fun l => g.reads.contains l || g.wcrLocs.contains l) with
        | some l => .error (.race l)
        | none =>
          match g.aliasSets.find? (fun (n, _) => h.aliasSets.any fun (n', _) => n = n') with
          | some (n, _) => .error (.aliasRace n)
          | none =>
            match (g.exclusiveQueues ++ h.exclusiveQueues).head? with
            | some l => .error (.queueRace l)
            | none => .error (.race default)
    | none => .error (.race default)

/-! ## Lemmas -/

namespace Merge

theorem applyWrites_nil (m : Mem) : applyWrites m [] = m := rfl

/-- Plain writes to distinct locations commute. -/
theorem applyWrites_perm (m : Mem) {ws ws' : List (Loc × Val)}
    (hp : ws.Perm ws') (hnd : (ws.map Prod.fst).Nodup) : applyWrites m ws = applyWrites m ws' := by
  unfold applyWrites
  refine hp.foldl_eq' ?_ m
  intro x hx y hy z
  by_cases hxy : x.1 = y.1
  · have := List.inj_on_of_nodup_map hnd hx hy hxy
    subst this
    rfl
  · simp only
    rw [Function.update_comm hxy]

/-- WCR contributions commute when the operator semantics is lawful and
contributions to the same location use the same operator. -/
theorem applyWcrs_perm (w : WcrSem) (hl : w.Lawful) (m : Mem) {cs cs' : List (Loc × WcrOp × Val)}
    (hp : cs.Perm cs') (hsame : ∀ x ∈ cs, ∀ y ∈ cs, x.1 = y.1 → x.2.1 = y.2.1) :
    applyWcrs w m cs = applyWcrs w m cs' := by
  unfold applyWcrs
  refine hp.foldl_eq' ?_ m
  intro x hx y hy z
  obtain ⟨lx, opx, vx⟩ := x
  obtain ⟨ly, opy, vy⟩ := y
  by_cases hxy : lx = ly
  · subst hxy
    have hop : opx = opy := hsame _ hx _ hy rfl
    subst hop
    simp only [Function.update_self, Function.update_idem]
    rw [hl.comm_assoc]
  · simp only [Function.update_of_ne hxy, Function.update_of_ne (Ne.symm hxy)]
    rw [Function.update_comm hxy]

/-- Alias bindings of distinct names commute. -/
theorem applyAliases_perm (m : Mem) {as as' : List (Name × AliasBinding)}
    (hp : as.Perm as') (hnd : (as.map Prod.fst).Nodup) : applyAliases m as = applyAliases m as' := by
  unfold applyAliases
  refine hp.foldl_eq' ?_ m
  intro x hx y hy z
  by_cases hxy : x.1 = y.1
  · have := List.inj_on_of_nodup_map hnd hx hy hxy
    subst this
    rfl
  · simp only
    rw [Function.update_comm hxy]

/-- Queue effects on distinct queues commute. -/
theorem applyQueues_perm (m : Mem) {qs qs' : List (Loc × QueueEffect)}
    (hp : qs.Perm qs') (hnd : (qs.map Prod.fst).Nodup) : applyQueues m qs = applyQueues m qs' := by
  unfold applyQueues
  refine hp.foldl_eq' ?_ m
  intro x hx y hy z
  by_cases hxy : x.1 = y.1
  · have := List.inj_on_of_nodup_map hnd hx hy hxy
    subst this
    rfl
  · obtain ⟨lx, ex⟩ := x
    obtain ⟨ly, ey⟩ := y
    simp only at hxy
    cases ex <;> cases ey <;>
      simp only [Function.update_of_ne hxy, Function.update_of_ne (Ne.symm hxy), Function.update_comm hxy]

end Merge

end Dace
