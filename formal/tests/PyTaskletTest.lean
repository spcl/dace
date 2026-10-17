-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-! Unit tests for the executable Python-tasklet subset (`Dace.Core.PyTasklet`).
Run with `lake env lean tests/PyTaskletTest.lean`; every `#guard` must pass. -/

open Dace

/-- A Python tasklet with the given code and connector tables. -/
def mkT (code : String) (ins outs : List (Conn × Option Dtype)) : Tasklet :=
  { code := { code := code, language := .python }, inConns := ins, outConns := outs }

/-- Build an input binding from a list of scalar connector values. -/
def scalarIns (m : List (Conn × Val)) : Conn → Option ConnIn :=
  fun c => (m.find? (·.1 = c)).map (fun p => .scalar p.2)

def runT (t : Tasklet) (ins : Conn → Option ConnIn) : Option TaskletOut :=
  PyTasklet.run PyTasklet.exprInterp t (fun _ => none) ins

/-- Scalar outputs of a run. -/
def runScalars (t : Tasklet) (ins : Conn → Option ConnIn) : Option (List (Conn × Val)) :=
  (runT t ins).map (·.scalars)

def fl : Dtype := .float 64
def i64 : Dtype := .int 64 true

/-! ## Parsing -/

#guard (PyTasklet.parseStmts "c = (a * b)").isSome
#guard (PyTasklet.parseStmts "__out = (__in1 * 2.0)").isSome
#guard (PyTasklet.parseStmts "out = (0.2 * ((((c + n) + s) + w) + e))").isSome
#guard (PyTasklet.parseStmts "lookup = __ind_xval[index_A_col_0]").isSome
#guard (PyTasklet.parseStmts "# a comment\nx = 1  # trailing\npass").isSome
#guard (PyTasklet.parseStmts "x = a if c else b").isSome
#guard (PyTasklet.parseStmts "x += 1; y -= 2").isSome
-- Compound statements are unsupported.
#guard (PyTasklet.parseStmts "if x > 0:\n  y = 1").isNone
#guard (PyTasklet.parseStmts "for i in range(3):\n  pass").isNone
#guard (PyTasklet.parseStmts "return x").isNone

/-! ## Running -/

-- c = a * b  (float multiply)
#guard runScalars (mkT "c = (a * b)" [("a", some fl), ("b", some fl)] [("c", some fl)])
  (scalarIns [("a", .float 2.0), ("b", .float 3.0)]) == some [("c", .float 6.0)]

-- o = a  (int copy, coerced to the output dtype)
#guard runScalars (mkT "o = a" [("a", some i64)] [("o", some i64)])
  (scalarIns [("a", .int 5)]) == some [("o", .int 5)]

-- out = 2.0 * x
#guard runScalars (mkT "out = (2.0 * x)" [("x", some fl)] [("out", some fl)])
  (scalarIns [("x", .float 3.0)]) == some [("out", .float 6.0)]

-- stencil-style body runs
#guard (runScalars (mkT "out = (0.2 * ((((c + n) + s) + w) + e))"
  [("c", some fl), ("n", some fl), ("s", some fl), ("w", some fl), ("e", some fl)]
  [("out", some fl)])
  (scalarIns [("c", .float 1.0), ("n", .float 1.0), ("s", .float 1.0),
              ("w", .float 1.0), ("e", .float 1.0)])).isSome

-- augmented assignment, last-write + read-before
#guard runScalars (mkT "out = 1\nout += 2" [] [("out", some i64)])
  (fun _ => none) == some [("out", .int 3)]

-- float output coerced from an int expression
#guard runScalars (mkT "out = (a + 1)" [("a", some i64)] [("out", some fl)])
  (scalarIns [("a", .int 2)]) == some [("out", .float 3.0)]

-- int output truncates a float toward zero
#guard runScalars (mkT "out = x" [("x", some fl)] [("out", some i64)])
  (scalarIns [("x", .float 3.9)]) == some [("out", .int 3)]
#guard runScalars (mkT "out = x" [("x", some fl)] [("out", some i64)])
  (scalarIns [("x", .float (-3.9))]) == some [("out", .int (-3))]

-- ternary
#guard runScalars (mkT "out = (a if c else b)"
  [("a", some i64), ("b", some i64), ("c", some .bool)] [("out", some i64)])
  (scalarIns [("a", .int 1), ("b", .int 2), ("c", .bool true)]) == some [("out", .int 1)]

-- math + casts
#guard runScalars (mkT "out = math.sqrt(x)" [("x", some fl)] [("out", some fl)])
  (scalarIns [("x", .float 4.0)]) == some [("out", .float 2.0)]
#guard runScalars (mkT "out = dace.float64(k)" [("k", some i64)] [("out", some fl)])
  (scalarIns [("k", .int 2)]) == some [("out", .float 2.0)]
#guard runScalars (mkT "out = int(x)" [("x", some fl)] [("out", some i64)])
  (scalarIns [("x", .float 7.8)]) == some [("out", .int 7)]

-- subscript read (pointer input connector)
def arrIns : Conn → Option ConnIn := fun c =>
  if c = "arr" then some (.array (fun idx => if idx = [2] then some (.int 42) else none))
  else if c = "i" then some (.scalar (.int 2)) else none
#guard runScalars (mkT "lookup = arr[i]" [("arr", some i64), ("i", some i64)] [("lookup", some i64)])
  arrIns == some [("lookup", .int 42)]

-- subscript write (pointer output connector)
#guard (runT (mkT "out[0, 1] = a" [("a", some fl)] [("out", some fl)])
  (scalarIns [("a", .float 7.0)])).map (·.arrayWrites)
  == some [("out", [0, 1], .float 7.0)]

-- stream pop (read) + push
def streamIns : Conn → Option ConnIn := fun c =>
  if c = "s" then some (.stream [.int 1, .int 2]) else none
#guard (runT (mkT "o.push(s)\no.push(s)" [("s", none)] [("o", none)]) streamIns).map (·.pushes)
  == some [("o", [.int 1, .int 2])]
#guard (runT (mkT "o.push(s)\no.push(s)" [("s", none)] [("o", none)]) streamIns).map (·.pops)
  == some [("s", 2)]
-- popping an exhausted stream is undefined
#guard (runT (mkT "o.push(s)\no.push(s)\no.push(s)" [("s", none)] [("o", none)]) streamIns).isNone

-- symbols are visible by name (D46); connectors shadow them
#guard (PyTasklet.run PyTasklet.exprInterp (mkT "out = (a * k)" [("a", some fl)] [("out", some fl)])
  (fun s => if s = "k" then some (.int 3) else none)
  (scalarIns [("a", .float 2.0)])).map (·.scalars) == some [("out", .float 6.0)]
#guard (PyTasklet.run PyTasklet.exprInterp (mkT "out = a" [("a", some fl)] [("out", some fl)])
  (fun s => if s = "a" then some (.int 3) else none)
  (scalarIns [("a", .float 2.0)])).map (·.scalars) == some [("out", .float 2.0)]
-- an unbound name is undefined
#guard (runT (mkT "out = k" [] [("out", some fl)]) (fun _ => none)).isNone

-- non-Python tasklets and unsupported code yield none
#guard (PyTasklet.run PyTasklet.exprInterp
  { code := { code := "x = 1", language := .cpp } } (fun _ => none) (fun _ => none)).isNone
#guard (runT (mkT "while x:\n  pass" [] []) (fun _ => none)).isNone
