-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! Checks for the polynomial normal form (`Core/SymArith`), certificate
checking and search (`Core/SymCert`, `Core/SymSearch`).  Run with
`lake env lean tests/SymArithTest.lean`; every `#guard` must pass. -/

open Dace

/-- Polynomial of an expression string (empty on failure). -/
def P (s : String) : Option Poly := (Expr.parse? s).bind Expr.toPoly?

def Pshow (s : String) : String := ((P s).map Poly.show).getD "<not integer-closed>"

/-! ## Normal form -/

#guard Pshow "i + 1" == Pshow "1 + i"
#guard Pshow "(N + 1)*(M - 1)" == Pshow "N*M - N + M - 1"
#guard Pshow "2*N - N - N" == "0"
#guard Pshow "Min(i + T, N)" == Pshow "Min(T + i, N)"
#guard Pshow "(i // 2)*2 + i % 2" == Pshow "i % 2 + 2*(i // 2)"
#guard Pshow "-(x - y)" == Pshow "y - x"
#guard Pshow "x**3" == Pshow "x*x*x"
#guard Pshow "floor(i) + ceiling(j)" == Pshow "i + j"
#guard Pshow "SymExpr(N, 2*N)" == Pshow "N"
-- not integer-closed: true division, floats, booleans, opaque calls, subscripts
#guard (P "N / 2").isNone
#guard (P "0.5*N").isNone
#guard (P "True + 1").isNone
#guard (P "sqrt(N)").isNone
#guard (P "A[i]").isNone
#guard (P "x ** N").isNone

#eval Pshow "(N + 1)*(M - 1)"
#eval Pshow "Min(i + T, N) - 2*Abs(j)"

-- evaluation agrees with the polynomial at `atomEnv` (instances of `toPoly?_sound`)
def ρ : Valuation := fun s =>
  match s with
  | "i" => some (.int 7) | "j" => some (.int (-3)) | "N" => some (.int 10) | "T" => some (.int 4)
  | _ => none

def agrees (s : String) : Bool :=
  match Expr.parse? s with
  | none => false
  | some e =>
    match e.toPoly?, e.evalInt ExprInterp.pure ρ with
    | some p, some v => v == p.eval (atomEnv ExprInterp.pure ρ)
    | _, _ => false

#guard ["i*N + j", "Min(i + T, N) - 1", "i // 2 + i % 3 - int_ceil(i, T)", "Abs(j) * Max(i, N)",
  "(i - j)**2", "-j // T", "Mod(j, T)", "int_floor(i*N + j, T)"].all agrees

/-! ## Queries -/

/-- Inequalities of a comparison string `a <= b`, `a < b`, `a >= b`, `a > b`,
`a == b`. -/
def rel (s : String) : List Ineq :=
  match Expr.parse? s with
  | some (.bin op a b) =>
    match a.toPoly?, b.toPoly? with
    | some x, some y =>
      match op with
      | .le => [Ineq.le x y]
      | .lt => [Ineq.lt x y]
      | .ge => [Ineq.le y x]
      | .gt => [Ineq.lt y x]
      | .eq => [Ineq.eqOf x y]
      | _ => []
    | _, _ => []
  | _ => []

/-- Query `hyps ⊢ goal` with the atom axioms added. -/
def Q (hyps : List String) (goal : String) : Query :=
  match rel goal with
  | [h] => ({ hyps := hyps.flatMap rel, goal := h.poly } : Query).withAxioms
  | _ => { goal := Poly.const (-1) }

def found (q : Query) : String :=
  match q.search with
  | some (.psatz ts) => s!"psatz ({ts.length} terms)"
  | some (.split ..) => "split"
  | none => "none"

-- linear
def q1 := Q ["0 <= i", "i <= N - 1"] "i < N"
-- offsets `i*N + j` of distinct rows of an `M`-column tile with `M ≤ N` are distinct
def q2 := Q ["0 <= j", "j <= M - 1", "0 <= j2", "j2 <= M - 1", "M <= N", "i + 1 <= i2"]
  "i*N + j < i2*N + j2"
-- clipped tile `t*T : Min(t*T + T, N) - 1`
def q3 := Q ["t*T <= i", "i <= Min(t*T + T, N) - 1"] "i <= N - 1"
def q4 := Q ["t*T <= i", "i <= Min(t*T + T, N) - 1"] "i < t*T + T"
-- consecutive clipped tiles are disjoint
def q5 := Q ["0 <= t", "1 <= T", "i <= Min(t*T + T, N) - 1", "(t + 1)*T <= i2"] "i < i2"
-- floor division / modulo by literals
def q6 := Q ["0 <= i"] "0 <= i // 2"
def q7 := Q [] "2*(i // 2) <= i"
def q8 := Q [] "i - 2*(i // 2) <= 1"
def q9 := Q [] "0 <= i % 4"
def q10 := Q [] "i % 4 <= 3"
def q11 := Q ["0 <= i", "i <= 2*K - 1"] "i // 2 <= K - 1"
-- symbolic divisor (case split on its sign, products)
def q12 := Q ["0 <= x", "1 <= N"] "0 <= x // N"
-- Min / Max / Abs need case splits
def q13 := Q ["c <= a", "c <= b"] "c <= Min(a, b)"
def q14 := Q [] "a <= Max(a, b)"
def q15 := Q [] "0 <= Abs(i - j)"
def q16 := Q ["a <= c", "b <= c"] "Max(a, b) <= c"
-- products of bounds: `0 ≤ i < N, 0 ≤ j < M ⇒ i*M + j < N*M`
def q17 := Q ["0 <= i", "i <= N - 1", "0 <= j", "j <= M - 1"] "i*M + j <= N*M - 1"
-- invalid queries: the search must fail
def bad1 := Q ["0 <= i", "i <= N"] "i <= N - 1"
def bad2 := Q ["0 <= i"] "i // 2 <= i - 1"
def bad3 := Q ["0 <= j", "j <= M", "M <= N", "i + 1 <= i2"] "i*N + j < i2*N"

def valid : List (String × Query) :=
  [("q1", q1), ("q2", q2), ("q3", q3), ("q4", q4), ("q5", q5), ("q6", q6), ("q7", q7), ("q8", q8),
   ("q9", q9), ("q10", q10), ("q11", q11), ("q12", q12), ("q13", q13), ("q14", q14), ("q15", q15),
   ("q16", q16), ("q17", q17)]

#eval valid.map fun (n, q) => (n, found q)
#eval [bad1, bad2, bad3].map found

#guard valid.all fun (_, q) => q.decide
#guard !([bad1, bad2, bad3].any (·.decide))

-- a certificate the checker must reject (wrong sign of a multiplier)
#guard !(Cert.check q1 (.psatz [.lin 0 1]))

-- `Query.decide` is sound (`Query.decide_sound`); a found certificate can be re-checked
#guard (q2.search.map (Cert.check q2)) == some true

/-! ## Kernel fallback -/

-- `Query.toLeanFile` emits a file checkable by `lake env lean` (proved by `omega`/`nlinarith`)
#eval IO.println (Query.toLeanFile [("q13_obligation", q13), ("q2_obligation", q2), ("q12_obligation", q12)])
