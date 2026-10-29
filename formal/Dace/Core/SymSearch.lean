-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.SymCert

/-!
# Certificate search (untrusted)

`Query.search` looks for a `Cert` that `Cert.check` accepts; nothing here is
trusted.  `Query.decide` runs the search and checks the result, so it is
sound by `Cert.check_sound` (`Query.decide_sound`).

The search is Fourier–Motzkin elimination over ℚ on the hypotheses together
with the negated goal, treating every non-constant monomial as a variable.
Each derived row remembers the non-negative combination of hypotheses it came
from; a row `c ≥ 0` with a constant `c < 0` yields a `Cert.psatz` after
clearing denominators.  If the linear pass fails, all pairwise products of
`≥`-hypotheses (including the negated goal) are added as extra rows.
Disjunctions are split (`Cert.split`) up to a bounded depth.  Elimination
gives up when the row count exceeds `maxRows`.

Limits: no integer reasoning beyond the tightening of the negated goal (no
cutting planes, no divisibility), products of degree two only, no products
of equalities with monomials.  `Query.toLean` emits the query as a closed
statement over ℤ for the kernel (`omega`/`nlinarith`) as a fallback.
-/

namespace Dace

set_option autoImplicit false

/-! ## Display -/

def Poly.show (p : Poly) : String :=
  match p with
  | [] => "0"
  | _ =>
    let terms := p.map fun (m, c) =>
      if m.isEmpty then toString c
      else if c = 1 then m.key
      else if c = -1 then "-" ++ m.key
      else s!"{c}*{m.key}"
    " + ".intercalate terms

def Ineq.show (h : Ineq) : String :=
  match h.kind with
  | .ge => h.poly.show ++ " >= 0"
  | .eq => h.poly.show ++ " == 0"

def Query.show (q : Query) : String :=
  let hs := q.hyps.map Ineq.show
  let ds := q.disj.map fun (l, r) =>
    "(" ++ " and ".intercalate (l.map Ineq.show) ++ ") or (" ++ " and ".intercalate (r.map Ineq.show) ++ ")"
  ", ".intercalate (hs ++ ds) ++ " |- " ++ q.goal.show ++ " >= 0"

namespace Search

/-! ## Fourier–Motzkin with provenance -/

/-- Origin of a row summand. -/
inductive Src where
  | lin (i : Nat)
  | prod (i j : Nat)
  | eq (i : Nat)
  deriving DecidableEq, Repr, Inhabited

/-- `Σ lin + c ≥ 0`, obtained as `Σ src`. -/
structure Row where
  lin : List (Nat × ℚ)
  c : ℚ
  src : List (Src × ℚ)
  deriving Inhabited

/-- Sum of sparse sorted vectors, dropping zeros. -/
def addLin : List (Nat × ℚ) → List (Nat × ℚ) → List (Nat × ℚ)
  | [], b => b
  | a, [] => a
  | (i, x) :: a, (j, y) :: b =>
    if i < j then (i, x) :: addLin a ((j, y) :: b)
    else if j < i then (j, y) :: addLin ((i, x) :: a) b
    else
      let s := x + y
      if s = 0 then addLin a b else (i, s) :: addLin a b

def addSrc (a b : List (Src × ℚ)) : List (Src × ℚ) :=
  b.foldl (fun acc (s, y) =>
    match acc.find? (·.1 = s) with
    | some _ => acc.map fun (t, x) => if t = s then (t, x + y) else (t, x)
    | none => acc ++ [(s, y)]) a

def Row.scale (k : ℚ) (r : Row) : Row :=
  { lin := r.lin.map fun (i, x) => (i, k * x), c := k * r.c, src := r.src.map fun (s, x) => (s, k * x) }

def Row.add (r s : Row) : Row :=
  { lin := addLin r.lin s.lin, c := r.c + s.c, src := addSrc r.src s.src }

/-- Scale so that the first coefficient has absolute value one. -/
def Row.normalize (r : Row) : Row :=
  match r.lin with
  | (_, x) :: _ => r.scale (1 / |x|)
  | [] => r

def Row.coef (r : Row) (v : Nat) : ℚ := ((r.lin.find? (·.1 = v)).map (·.2)).getD 0

def Row.isContra (r : Row) : Bool := r.lin.isEmpty && decide (r.c < 0)

def Row.isTrivial (r : Row) : Bool := r.lin.isEmpty && decide (0 ≤ r.c)

/-- Keep one row per linear part, the tightest one. -/
def dedupe (rows : List Row) : List Row :=
  rows.foldl (fun acc r =>
    match acc.find? (·.lin == r.lin) with
    | some s => if r.c < s.c then acc.map (fun t => if t.lin == r.lin then r else t) else acc
    | none => acc ++ [r]) []

def maxRows : Nat := 3000

/-- Eliminate variables until a contradiction appears. -/
def fm : Nat → List Row → Option Row
  | 0, rows => rows.find? Row.isContra
  | fuel + 1, rows =>
    match rows.find? Row.isContra with
    | some r => some r
    | none =>
      let rows := rows.filter fun r => !r.isTrivial
      let vars := (rows.flatMap fun r => r.lin.map (·.1)).eraseDups
      let score (v : Nat) : Nat :=
        let p := (rows.filter fun r => r.coef v > 0).length
        let n := (rows.filter fun r => r.coef v < 0).length
        p * n
      match vars with
      | [] => none
      | v0 :: vs =>
        let v := vs.foldl (fun best w => if score w < score best then w else best) v0
        let pos := rows.filter fun r => r.coef v > 0
        let neg := rows.filter fun r => r.coef v < 0
        let zero := rows.filter fun r => r.coef v = 0
        if pos.length * neg.length + zero.length > maxRows then none
        else
          let new := pos.flatMap fun p => neg.map fun n =>
            (Row.add (p.scale (-(n.coef v))) (n.scale (p.coef v))).normalize
          fm fuel (dedupe (zero ++ new))

/-- Index monomials as variables. -/
def monoIndex (tbl : List Mono) (m : Mono) : List Mono × Nat :=
  match tbl.idxOf? m with
  | some i => (tbl, i)
  | none => (tbl ++ [m], tbl.length)

/-- A polynomial as `(lin, c)` over the monomial table. -/
def toLin (tbl : List Mono) (p : Poly) : List Mono × List (Nat × ℚ) × ℚ :=
  p.foldl (fun (tbl, lin, c) (m, k) =>
    if m.isEmpty then (tbl, lin, c + k)
    else
      let (tbl, i) := monoIndex tbl m
      (tbl, addLin lin [(i, (k : ℚ))], c)) (tbl, [], 0)

/-- The rows of the extended hypotheses (and optionally their products). -/
def rowsOf (ext : List Ineq) (products : Bool) : List Row := Id.run do
  let mut tbl : List Mono := []
  let mut rows : List Row := []
  let ge := (ext.zipIdx.filter fun (h, _) => h.kind = .ge)
  for (h, i) in ext.zipIdx do
    let (t, lin, c) := toLin tbl h.poly
    tbl := t
    match h.kind with
    | .ge => rows := rows ++ [{ lin, c, src := [(.lin i, 1)] }]
    | .eq =>
      rows := rows ++ [{ lin, c, src := [(.eq i, 1)] },
        { lin := lin.map (fun (v, x) => (v, -x)), c := -c, src := [(.eq i, -1)] }]
  if products then
    for (h, i) in ge do
      for (h', j) in ge do
        if i ≤ j then
          let (t, lin, c) := toLin tbl (Poly.mul h.poly h'.poly)
          tbl := t
          rows := rows ++ [{ lin, c, src := [(.prod i j, 1)] }]
  return rows.map Row.normalize

/-- Clear denominators and build the certificate. -/
def certOf (r : Row) : Cert :=
  let l : Nat := r.src.foldl (fun acc (_, x) => Nat.lcm acc x.den) 1
  let terms := r.src.filterMap fun (s, x) =>
    let n : Int := (x * l).num
    if n = 0 then none
    else some (match s with
      | .lin i => PTerm.lin i n.toNat
      | .prod i j => PTerm.prod i j n.toNat
      | .eq i => PTerm.eq i (Poly.const n))
  .psatz terms

/-- One Positivstellensatz attempt. -/
def psatz (q : Query) (products : Bool) : Option Cert := do
  let rows := rowsOf q.ext products
  let fuel := rows.length + 1 + (rows.flatMap fun r => r.lin.map (·.1)).eraseDups.length
  let r ← fm fuel rows
  let c := certOf r
  if c.check q then some c else none

/-- Search with case splits on disjunctions (each disjunction is tried as the
first split), up to `depth` nested splits. -/
def searchAux (products : Bool) : Nat → Query → Option Cert
  | 0, q => psatz q products
  | depth + 1, q =>
    match psatz q products with
    | some c => some c
    | none =>
      (List.range q.disj.length).findSome? fun k =>
        match q.disj[k]? with
        | some d => do
          let l ← searchAux products depth (q.branch k d.1)
          let r ← searchAux products depth (q.branch k d.2)
          pure (.split k l r)
        | none => none

end Search

/-- Untrusted certificate search: linear Fourier–Motzkin first, then with
pairwise products of hypotheses; at most `depth` nested case splits. -/
def Query.search (q : Query) (depth : Nat := 2) : Option Cert :=
  let d := min depth q.disj.length
  (Search.searchAux false d q).orElse fun _ => Search.searchAux true d q

/-- Decide validity by search and check (sound, incomplete). -/
def Query.decide (q : Query) (depth : Nat := 2) : Bool :=
  match q.search depth with
  | some c => c.check q
  | none => false

theorem Query.decide_sound {q : Query} {depth : Nat} (h : q.decide depth = true) : q.Valid := by
  unfold Query.decide at h
  split at h
  · exact Cert.check_sound _ q h
  · simp at h

/-! ## Kernel fallback: emission as a Lean statement -/

/-- Emit the query as a closed theorem over ℤ (atoms become universally
quantified variables `x0, x1, …`, listed in the docstring) proven by
`omega` or `nlinarith` after splitting the disjunctions. -/
def Query.toLean (q : Query) (name : String) : String := Id.run do
  let atoms := q.atoms
  let var (a : Atom) : String := s!"x{(atoms.idxOf? a).getD 0}"
  let mono (m : Mono) : String := " * ".intercalate (m.map var)
  let poly (p : Poly) : String :=
    match p with
    | [] => "(0 : ℤ)"
    | _ =>
      let ts := p.map fun (m, c) =>
        if m.isEmpty then s!"({c} : ℤ)"
        else if c = 1 then mono m
        else s!"({c} : ℤ) * {mono m}"
      " + ".intercalate ts
  let ineq (h : Ineq) : String :=
    match h.kind with
    | .ge => s!"0 ≤ {poly h.poly}"
    | .eq => s!"{poly h.poly} = 0"
  let conj (hs : List Ineq) : String :=
    match hs with
    | [] => "True"
    | _ => " ∧ ".intercalate (hs.map fun h => s!"({ineq h})")
  let doc := ", ".intercalate ((atoms.zipIdx).map fun (a, i) => s!"x{i} = {a.key}")
  let vars := if atoms.isEmpty then "" else
    " (" ++ " ".intercalate ((List.range atoms.length).map fun i => s!"x{i}") ++ " : ℤ)"
  let hyps := (q.hyps.zipIdx).map fun (h, i) => s!" (h{i} : {ineq h})"
  let disj := (q.disj.zipIdx).map fun ((l, r), i) => s!" (d{i} : ({conj l}) ∨ ({conj r}))"
  let pat (pre : String) (hs : List Ineq) : String :=
    "⟨" ++ ", ".intercalate ((List.range hs.length).map fun j => s!"{pre}{j}") ++ "⟩"
  let splits := (q.disj.zipIdx).map fun ((l, r), i) =>
    s!"rcases d{i} with {pat s!"d{i}l" l} | {pat s!"d{i}r" r} <;> "
  let body := String.join splits ++ "first | omega | nlinarith"
  return s!"/-- {doc} -/\ntheorem {name}{vars}{String.join hyps}{String.join disj} :\n    {ineq (Ineq.ge q.goal)} := by\n  {body}\n"

/-- A self-contained obligation file (checked with `lake env lean`). -/
def Query.toLeanFile (obligations : List (String × Query)) : String :=
  "import Dace.Prelude\n\nset_option linter.unusedVariables false\nset_option linter.unusedTactic false\n" ++
    "set_option linter.unreachableTactic false\n\n" ++
    "\n".intercalate (obligations.map fun (n, q) => q.toLean n)

end Dace
