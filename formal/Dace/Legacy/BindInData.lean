-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.SDFG.Footprinted

/-!
# Bridge: array connectors hold the values read

`bindInData` binds an array connector to a function from relative indices to
values.  The function used to look the element up in the memory again
(`Ctx.loc`, `Mem.read`), although `St.readMany` had just read and returned
the same values.  `Legacy.bindInDataNew` builds the function from the values
read; `bindInData_eq_new` shows that nothing changes.  This file is removed
once `bindInData` is defined that way.
-/

set_option autoImplicit false

namespace Dace

/-- The values `vs` read at the points `pts`, addressed relative to the
lower corner `lo`. -/
def arrayOfRead (lo : Idx) (pts : List Idx) (vs : List Val) : Idx → Option Val := fun rel =>
  if rel.length = lo.length then List.lookup (List.zipWith (· + ·) lo rel) (pts.zip vs) else none

namespace Legacy

def bindInDataNew (ctx : Ctx) (ρ : Valuation) (d : Name) (sub : Option Subset) (oob : Bool) (st : St) :
    Res (ConnIn × St) :=
  if ctx.isStream d then do
    let p ← streamPoint ctx ρ st.mem d sub
    let q ← st.queue ctx ρ d p
    pure (.stream q, st)
  else do
    let pts ← ctx.pointsOf ρ st.mem d sub
    match pts with
    | [p] => do
      let (v, st) ← st.read ctx ρ d p oob
      pure (.scalar v, st)
    | _ => do
      let lo ← ctx.lowerCorner ρ st.mem d sub
      let (vs, st) ← st.readMany ctx ρ d pts oob
      pure (.array (arrayOfRead lo pts vs), st)

theorem lookup_read {ctx : Ctx} {ρ : Valuation} {m : Mem} {d : Name} {oob : Bool} {pts : List Idx} {ls : List Loc}
    (hf : List.Forall₂ (fun p l => ctx.loc ρ m d p oob = .ok l) pts ls) (abs : Idx) :
    (if pts.contains abs then
        match ctx.loc ρ m d abs oob with
        | .ok l => some (m.read l)
        | .err _ => none
      else none) = List.lookup abs (pts.zip (ls.map m.read)) := by
  induction hf with
  | nil => rfl
  | @cons p l ps ls hl _ ih =>
    simp only [List.contains_cons, List.map_cons, List.zip_cons_cons, List.lookup_cons]
    cases hp : abs == p
    · simpa using ih
    · have : abs = p := by simpa using hp
      subst this
      simp [hl]

theorem bindArray_eq (ctx : Ctx) (ρ : Valuation) (d : Name) (sub : Option Subset) (oob : Bool) (pts : List Idx)
    (st : St) :
    (do
      let lo ← ctx.lowerCorner ρ st.mem d sub
      let (_, st) ← st.readMany ctx ρ d pts oob
      let m := st.mem
      let get : Idx → Option Val := fun rel =>
        if rel.length = lo.length then
          let abs := List.zipWith (· + ·) lo rel
          if pts.contains abs then
            match ctx.loc ρ m d abs oob with
            | .ok l => some (m.read l)
            | .err _ => none
          else none
        else none
      pure (ConnIn.array get, st) : Res (ConnIn × St)) =
    (do
      let lo ← ctx.lowerCorner ρ st.mem d sub
      let (vs, st) ← st.readMany ctx ρ d pts oob
      pure (ConnIn.array (arrayOfRead lo pts vs), st)) := by
  cases ctx.lowerCorner ρ st.mem d sub with
  | err e => rfl
  | ok lo =>
    simp only [Res.ok_bind]
    cases hrm : st.readMany ctx ρ d pts oob with
    | err e => rfl
    | ok r =>
      obtain ⟨vs, st₁⟩ := r
      unfold St.readMany at hrm
      obtain ⟨ls, hf, hvs, rfl, -⟩ := St.readMany_ok ctx ρ pts [] st vs st₁ hrm
      simp only [List.nil_append] at hvs
      subst hvs
      simp only [Res.ok_bind, Res.pure_eq]
      congr 3
      funext rel
      unfold arrayOfRead
      split
      · exact lookup_read hf _
      · rfl

theorem bindInData_eq_new (ctx : Ctx) (ρ : Valuation) (d : Name) (sub : Option Subset) (oob : Bool) (st : St) :
    bindInData ctx ρ d sub oob st = bindInDataNew ctx ρ d sub oob st := by
  unfold bindInData bindInDataNew
  split
  · rfl
  · cases ctx.pointsOf ρ st.mem d sub with
    | err e => rfl
    | ok pts =>
      simp only [Res.ok_bind]
      rcases pts with _ | ⟨p, _ | ⟨q, rest⟩⟩
      · exact bindArray_eq ctx ρ d sub oob [] st
      · rfl
      · exact bindArray_eq ctx ρ d sub oob (p :: q :: rest) st

end Legacy

end Dace
