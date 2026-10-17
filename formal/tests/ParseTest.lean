-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! Round-trip and evaluation checks for `Dace.Core.Parse`.  Run with
`lake env lean tests/ParseTest.lean`; every `#guard` must pass. -/

open Dace

/-- Parse, print, re-parse: the printed form must parse to the same tree. -/
def roundTrips (s : String) : Bool :=
  match Expr.parse? s with
  | none => false
  | some e => Expr.parse? e.toPy == some e

def samples : List String := [
  "N", "i + 1", "2*N - 1", "Min(N, M)", "Max(1, N, M)", "int_floor(N, 4)", "int_ceil(N, 4)",
  "N // 4", "N % 3", "A[i, j]", "(i < N) and (j >= 0)", "0.5*N", "1e-3*x", "x**2", "-i",
  "not (a == b)", "IfExpr(c, 1, 0)", "s.member", "a > b > c", "Abs(i - j)", "x ** -1",
  "-x**2", "(a + b)*c", "a + b*c", "a - (b - c)", "a / b / c", "True", "False and x",
  "i | j & k ^ 2", "i << 2", "Mod(i, 3)", "sqrt(x)", "Not(Eq(a, b))", "a != b", "2.0",
  "f(a, g(b, c))", "N*M*K", "(N + 1)*(M - 1)", "__sym", "x_1 + x_2"
]

#eval samples.filter (fun s => !roundTrips s)
#guard samples.all roundTrips

-- Printing of a few trees
#eval (Expr.parse? "a - (b - c)").map Expr.toPy
#eval (Expr.parse? "-x**2").map Expr.toPy
#eval (Expr.parse? "a > b > c").map Expr.toPy
#eval (Expr.parse? "1e-3*x").map Expr.toPy

-- Evaluation
def ρ : Valuation := fun s => if s = "N" then some (.int 10) else if s = "i" then some (.int 3) else none
def ev (s : String) : Option Val := (Expr.parse? s) >>= fun e => e.eval ExprInterp.pure ρ

#eval ev "2*N - 1"
#guard ev "2*N - 1" == some (.int 19)
#guard ev "int_floor(N, 4)" == some (.int 2)
#guard ev "int_ceil(N, 4)" == some (.int 3)
#guard ev "(-7) // 2" == some (.int (-4))
#guard ev "(-7) % 3" == some (.int 2)
#guard ev "Min(N, i)" == some (.int 3)
#guard ev "i < N and N > 5" == some (.bool true)
#guard ev "IfExpr(i > N, 1, 0)" == some (.int 0)
#guard ev "a > b > c" == none  -- unbound symbols
#guard ev "2 ** 10" == some (.int 1024)
#guard ev "N / 4" == some (.float 2.5)
#guard ev "i << 2" == some (.int 12)

-- Ranges
def r : Range := ⟨[{ lo := 0, hi := Expr.sym "N" - 1, step := 3 }, { lo := 1, hi := 2 }]⟩
#eval r.points ExprInterp.pure ρ
#guard r.points ExprInterp.pure ρ == some [[0,1],[0,2],[3,1],[3,2],[6,1],[6,2],[9,1],[9,2]]
#guard r.size ExprInterp.pure ρ == some 8
#guard (Range.ofIndices [Expr.sym "i"]).points ExprInterp.pure ρ == some [[3]]
#guard (Range.ofDims [{ lo := 0, hi := Expr.sym "N" - 1 }, Dim.index 5]).squeeze.ndim == 1
#guard r.squeeze.ndim == 2
#guard ((Range.ofIndices [1, 2]).unsqueeze [0, 2]).dims.length == 4
