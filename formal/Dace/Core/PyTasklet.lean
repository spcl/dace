-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Interp
import Dace.Core.Parse

/-!
# Executable Python-expression tasklets

An executable subset of DaCe's Python tasklet language (`Decisions.md#D28`),
sufficient for the expression-level tasklets that `unparse_tasklet`
(`dace/codegen/targets/cpp.py`) emits and that the corpus contains.

A tasklet body is a sequence of statements separated by newlines or `;`
(comments `#…` and blank lines are ignored).  Supported statements:

* `target = expr` and the augmented forms `target OP= expr`
  (`+= -= *= /= //= %= **= &= |= ^= <<= >>=`), reading the target first;
* `pass`;
* a bare method call `conn.push(e)` / `conn.append(e)` (stream push) or
  `conn.pop()` (stream pop).

A *target* is a connector/local name or a subscript `name[i, j]`.  Compound
statements (`if`/`for`/…) are reported as **unsupported** (the whole body
yields `none`); conditional *expressions* `a if c else b` are supported by the
expression parser.

Reading semantics (`Decisions.md#D13`, `#D14`, `#D46`):
* a local variable reads its last value; a scalar input connector reads its
  bound value; a subscript `a[i]` into a pointer connector reads the element at
  the relative index; each read of a stream input connector pops one available
  value (counted in `pops`), undefined when exhausted; a bare name that is
  neither a local nor a connector reads the symbol of that name;
* assigning a plain name that is an output connector records a scalar write
  (last assignment wins); a subscript on an output connector records an element
  write; `conn.push(e)` records a stream push.

Arithmetic goes through `Expr.eval`'s `Eval.*`; `app` nodes implement the math
library (`math.sqrt`, …), `min/max/abs`, `int()/float()` casts and
`dace.<dtype>(x)` casts.  Output values are coerced to the connector's declared
`Dtype` (float → `Val.float`, int → `Val.int` truncated toward zero, bool →
`Val.bool`).

The parser and evaluator are executable tools (`partial`); they are not
reasoned about.
-/

namespace Dace.PyTasklet

open Dace.Parse (Token)

/-! ## String helpers

The toolchain's `String.take`/`drop` return a `String.Slice` and it lacks
`String.trim`, so these operate through `String.toList`. -/

private def strDrop (s : String) (n : Nat) : String := String.ofList (s.toList.drop n)
private def strDropRight (s : String) (n : Nat) : String :=
  String.ofList (s.toList.take (s.toList.length - n))
private def strTrim (s : String) : String :=
  String.ofList ((s.toList.dropWhile Char.isWhitespace).reverse.dropWhile
    Char.isWhitespace).reverse

/-! ## Casts and the math library -/

/-- C-style cast of a value to an integer: truncation toward zero. -/
def castToInt : Val → Int
  | .int v => v
  | .rat q => q.num.tdiv (q.den : Int)   -- `Int./` is Euclidean; C casts truncate toward zero
  | .float f => (f.toInt64).toInt
  | .bool b => if b then 1 else 0
  | .undef => 0

/-- Coerce an assigned value to an output connector's declared dtype. -/
def coerce : Dtype → Val → Val
  | .float _, v => .float v.toFloat
  | .int _ _, v => .int (castToInt v)
  | .bool, v => .bool (v.toBool?.getD false)
  | .opaque _, v => v

/-- Apply a unary floating-point library function. -/
private def floatFn (f : Float → Float) (v : Val) : Option Val := some (.float (f v.toFloat))

/-- The expression `app` interpretation used by executable tasklets: the math
library, numeric casts and `dace.<dtype>(x)` casts.  Unknown functions are
undefined (`none`). -/
def mathApp : String → List Val → Option Val := fun f args =>
  -- strip a leading `math.` qualifier
  let base := if f.startsWith "math." then strDrop f 5 else f
  match base, args with
  | "sqrt", [v] => floatFn Float.sqrt v
  | "exp", [v] => floatFn Float.exp v
  | "log", [v] => floatFn Float.log v
  | "sin", [v] => floatFn Float.sin v
  | "cos", [v] => floatFn Float.cos v
  | "tan", [v] => floatFn Float.tan v
  | "tanh", [v] => floatFn Float.tanh v
  | "fabs", [v] => floatFn Float.abs v
  | "floor", [v] => some (.int (castToInt (.float (Float.floor v.toFloat))))
  | "ceil", [v] => some (.int (castToInt (.float (Float.ceil v.toFloat))))
  | "pow", [a, b] => some (.float (Float.pow a.toFloat b.toFloat))
  | "abs", [v] => Eval.unary .abs v
  | "min", a :: rest => some (rest.foldl (fun x y => (Eval.arith .min x y).getD x) a)
  | "max", a :: rest => some (rest.foldl (fun x y => (Eval.arith .max x y).getD x) a)
  | "int", [v] => some (.int (castToInt v))
  | "float", [v] => some (.float v.toFloat)
  | "bool", [v] => some (.bool (v.toBool?.getD false))
  | _, [v] =>
    -- `dace.<dtype>(x)` scalar casts
    if f.startsWith "dace." then
      let ty := strDrop f 5
      if ty.startsWith "float" then some (.float v.toFloat)
      else if ty.startsWith "int" || ty.startsWith "uint" then some (.int (castToInt v))
      else if ty.startsWith "bool" then some (.bool (v.toBool?.getD false))
      else none
    else none
  | _, _ => none

/-- The expression interpretation for executable tasklets. -/
def exprInterp : ExprInterp where
  app := mathApp
  readData := fun _ _ => none

/-! ## Statements -/

/-- A supported tasklet statement. -/
inductive Stmt where
  /-- `target = rhs` (`aug = none`) or `target OP= rhs`. -/
  | assign (target : Expr) (aug : Option BinOp) (rhs : Expr)
  /-- A bare expression statement (stream push/pop or a discarded read). -/
  | exprStmt (e : Expr)
  | pass
  /-- `if c₁: b₁ elif c₂: b₂ … else: orelse` (`orelse = []` without `else`). -/
  | ifElse (branches : List (Expr × List Stmt)) (orelse : List Stmt)
  /-- `for var in range(args):` with one to three integer arguments. -/
  | forRange (var : Name) (args : List Expr) (body : List Stmt)
  deriving Inhabited

/-- Statement keywords of compound/flow statements we do not interpret. -/
private def blockKeywords : List String :=
  ["if", "elif", "else", "for", "while", "def", "with", "return", "break",
   "continue", "global", "nonlocal", "import", "from", "class", "yield",
   "raise", "try", "except", "finally", "lambda", "del", "assert", "async",
   "await"]

/-- Map an operator token to the augmented-assignment binary operator. -/
private def augOpOf? : Token → Option BinOp
  | .op "+" => some .add | .op "-" => some .sub | .op "*" => some .mul
  | .op "/" => some .div | .op "//" => some .intFloor | .op "%" => some .mod
  | .op "**" => some .pow | .op "&" => some .bitAnd | .op "|" => some .bitOr
  | .op "^" => some .bitXor | .op "<<" => some .shl | .op ">>" => some .shr
  | _ => none

/-- Find the first top-level `=` (assignment) token; return the tokens before
and after it. -/
private partial def findEq (toks acc : List Token) (depth : Nat) :
    Option (List Token × List Token) :=
  match toks with
  | [] => none
  | t :: rest =>
    if depth = 0 && t = Token.op "=" then some (acc.reverse, rest)
    else
      let d := match t with
        | .lparen | .lbracket => depth + 1
        | .rparen | .rbracket => depth - 1
        | _ => depth
      findEq rest (t :: acc) d

/-- Split a statement's tokens into `(target, augOp?, rhs)` at a top-level `=`. -/
private def splitAssign (toks : List Token) :
    Option (List Token × Option BinOp × List Token) :=
  match findEq toks [] 0 with
  | none => none
  | some (pre, rhs) =>
    match pre.reverse with
    | [] => none
    | last :: revInit =>
      match augOpOf? last with
      | some op => some (revInit.reverse, some op, rhs)
      | none => some (pre, none, rhs)

/-- Strip a `#` comment from a single physical line (no string literals in the
tasklet subset). -/
private def stripComment (l : String) : String := (l.splitOn "#").headD ""

/-- Parse one trimmed, non-empty statement. -/
def parseStmt (line : String) : Option Stmt := do
  if line == "pass" then return .pass
  let toks0 ← Parse.tokenize line
  let toks := toks0.filter (· != Token.eof)
  match toks with
  | Token.ident kw :: _ => if kw ∈ blockKeywords then failure
  | _ => pure ()
  match splitAssign toks with
  | some (pre, aug, rhs) =>
    let tgt ← Parse.parseFromToks pre
    let r ← Parse.parseFromToks rhs
    pure (.assign tgt aug r)
  | none =>
    let e ← Parse.parseFromToks toks
    pure (.exprStmt e)

/-- A logical line: its indentation width and its trimmed text. -/
private abbrev Line := Nat × String

private def strStartsWith (s pre : String) : Bool := pre.toList.isPrefixOf s.toList

/-- The text of a compound-statement header `kw rest:` after `kw`, if `l` is one. -/
private def header? (kw : String) (l : String) : Option String :=
  if l.endsWith ":" && (l == kw ++ ":" || strStartsWith l (kw ++ " ") || strStartsWith l (kw ++ "(")) then
    some (strTrim (strDropRight (strDrop l kw.length) 1))
  else none

/-- `var in range(args)` of a `for` header. -/
private def forHeader? (h : String) : Option (Name × List Expr) := do
  match h.splitOn " in " with
  | [v, it] =>
    let v := strTrim v
    let toks ← Parse.tokenize v
    match toks.filter (· != Token.eof) with
    | [Token.ident name] =>
      match ← Parse.parseFromToks (← Parse.tokenize it) with
      | .app "range" args =>
        let args := Exprs.toList args
        if 1 ≤ args.length && args.length ≤ 3 then pure (name, args) else none
      | _ => none
    | _ => none
  | _ => none

private def parseCond (c : String) : Option Expr := do Parse.parseFromToks (← Parse.tokenize c)

mutual

/-- Parse the statements at indentation `ind` from the front of `ls`; stops at
the first line indented less, returns the remaining lines. -/
private partial def parseBlock (ls : List Line) (ind : Nat) (acc : List Stmt) :
    Option (List Stmt × List Line) :=
  match ls with
  | [] => some (acc.reverse, [])
  | (i, l) :: rest =>
    if i < ind then some (acc.reverse, ls)
    else if i > ind then none
    else if let some c := header? "if" l then do
      let (body, rest) ← parseSuite rest ind
      let (branches, orelse, rest) ← parseElifs rest ind [(← parseCond c, body)]
      parseBlock rest ind (.ifElse branches orelse :: acc)
    else if let some h := header? "for" l then do
      let (v, args) ← forHeader? h
      let (body, rest) ← parseSuite rest ind
      parseBlock rest ind (.forRange v args body :: acc)
    else do
      let stmts ← ((l.splitOn ";").map strTrim |>.filter (!·.isEmpty)).mapM parseStmt
      parseBlock rest ind (stmts.reverse ++ acc)

/-- The indented block after a compound-statement header at indentation `ind`. -/
private partial def parseSuite (ls : List Line) (ind : Nat) : Option (List Stmt × List Line) :=
  match ls with
  | (i, _) :: _ => if i > ind then parseBlock ls i [] else none
  | [] => none

/-- The `elif`/`else` clauses following an `if` block at indentation `ind`. -/
private partial def parseElifs (ls : List Line) (ind : Nat) (branches : List (Expr × List Stmt)) :
    Option (List (Expr × List Stmt) × List Stmt × List Line) :=
  match ls with
  | (i, l) :: rest =>
    if i != ind then some (branches.reverse, [], ls)
    else if let some c := header? "elif" l then do
      let (body, rest) ← parseSuite rest ind
      parseElifs rest ind ((← parseCond c, body) :: branches)
    else if header? "else" l == some "" then do
      let (body, rest) ← parseSuite rest ind
      some (branches.reverse, body, rest)
    else some (branches.reverse, [], ls)
  | [] => some (branches.reverse, [], [])

end

/-- Parse a tasklet body into a statement list (`none` if any statement is
unsupported).  Compound statements are `if`/`elif`/`else` and
`for v in range(…)`, with blocks given by indentation. -/
def parseStmts (code : String) : Option (List Stmt) := do
  let lines : List Line := (code.splitOn "\n").filterMap fun l =>
    let l := stripComment l
    let t := strTrim l
    if t.isEmpty then none
    else some ((l.toList.takeWhile Char.isWhitespace).length, t)
  match lines with
  | [] => pure []
  | (i, _) :: _ =>
    let (stmts, rest) ← parseBlock lines i []
    if rest.isEmpty then pure stmts else none

/-! ## Evaluation -/

/-- Mutable state threaded through a tasklet execution. -/
structure RunState where
  I : ExprInterp
  /-- Symbols visible to the tasklet (`Decisions.md#D46`). -/
  ρ : Valuation
  ins : Conn → Option ConnIn
  outDtypes : List (Conn × Option Dtype)
  locals : List (Conn × Val) := []
  /-- Remaining available values of each stream input connector. -/
  streams : List (Conn × List Val) := []
  pops : List (Conn × Nat) := []
  scalars : List (Conn × Val) := []
  arrayWrites : List (Conn × Idx × Val) := []
  pushes : List (Conn × List Val) := []

abbrev M := StateT RunState Option

/-- Lift an `Option` into the execution monad. -/
def ofOpt {α} (o : Option α) : M α := fun s => o.map (fun a => (a, s))

private def upsert (l : List (String × Val)) (k : String) (v : Val) : List (String × Val) :=
  (l.filter (·.1 != k)) ++ [(k, v)]

private def bumpPop (l : List (Conn × Nat)) (k : Conn) : List (Conn × Nat) :=
  if l.any (·.1 == k) then l.map (fun (p : Conn × Nat) => if p.1 == k then (k, p.2 + 1) else p)
  else l ++ [(k, 1)]

private def appendPush (l : List (Conn × List Val)) (k : Conn) (v : Val) :
    List (Conn × List Val) :=
  if l.any (·.1 == k) then
    l.map (fun (p : Conn × List Val) => if p.1 == k then (k, p.2 ++ [v]) else p)
  else l ++ [(k, [v])]

/-- Pop one value from a stream input connector (`Decisions.md#D14`). -/
def popStream (s : Conn) : M Val := do
  let st ← get
  match st.streams.find? (·.1 == s) with
  | some (_, v :: rest) =>
    set { st with
      streams := st.streams.map (fun (p : Conn × List Val) => if p.1 == s then (s, rest) else p),
      pops := bumpPop st.pops s }
    pure v
  | _ => ofOpt none

/-- Push one value onto a stream output connector, coerced to its dtype. -/
def pushStream (s : Conn) (v : Val) : M Unit := do
  let st ← get
  let v' := match st.outDtypes.find? (·.1 == s) with
    | some (_, some dt) => coerce dt v
    | _ => v
  set { st with pushes := appendPush st.pushes s v' }

/-- Read a bare name: local, previously written output scalar, input
connector, or symbol (`Decisions.md#D46`). -/
def readName (s : Conn) : M Val := do
  let st ← get
  match st.locals.find? (·.1 == s) with
  | some (_, v) => pure v
  | none =>
    match st.scalars.find? (·.1 == s) with
    | some (_, v) => pure v
    | none =>
      match st.ins s with
      | some (.scalar v) => pure v
      | some (.stream _) => popStream s
      | some (.array _) => ofOpt none
      | none => ofOpt (st.ρ s)

/-- Read a subscript `conn[idx]`. -/
def readSub (conn : Name) (idx : Idx) : M Val := do
  let st ← get
  match st.ins conn with
  | some (.array get) => ofOpt (get idx)
  | some (.scalar v) => if idx.all (· == 0) then pure v else ofOpt none
  | _ => ofOpt none

mutual

/-- Evaluate an expression, threading stream state. -/
partial def evalE : Expr → M Val
  | .lit (.int v) => pure (.int v)
  | .lit (.float m e) => pure (.float (Lit.floatVal m e))
  | .lit (.bool b) => pure (.bool b)
  | .sym s => readName s
  | .un op a => do ofOpt (Eval.unary op (← evalE a))
  | .bin op a b => do ofOpt (Eval.arith op (← evalE a) (← evalE b))
  | .ite c t e => do
    let cv ← evalE c
    if (← ofOpt cv.toBool?) then evalE t else evalE e
  | .app f args => evalApp f args
  | .subscript conn idx => do
    let vs ← evalEs idx
    let is ← ofOpt (vs.mapM Val.toInt?)
    readSub conn is
  | .attr _ _ => ofOpt none
  | .approx a _ => evalE a

partial def evalEs : Exprs → M (List Val)
  | .nil => pure []
  | .cons e es => do pure ((← evalE e) :: (← evalEs es))

/-- Evaluate a function/method application. -/
partial def evalApp (f : String) (args : Exprs) : M Val := do
  if f.endsWith ".pop" && (Exprs.toList args).isEmpty then
    popStream (strDropRight f 4)
  else
    let vs ← evalEs args
    ofOpt ((← get).I.app f vs)

end

/-- Read the current value of an assignment target (for augmented assignment). -/
def readTarget : Expr → M Val
  | .sym s => readName s
  | .subscript conn idx => do
    let vs ← evalEs idx
    let is ← ofOpt (vs.mapM Val.toInt?)
    readSub conn is
  | _ => ofOpt none

/-- Store a value into an assignment target. -/
def assignTo (target : Expr) (v : Val) : M Unit := do
  match target with
  | .sym name =>
    let st ← get
    if st.outDtypes.any (·.1 == name) then
      let v' := match st.outDtypes.find? (·.1 == name) with
        | some (_, some dt) => coerce dt v
        | _ => v
      set { st with scalars := upsert st.scalars name v' }
    else
      set { st with locals := upsert st.locals name v }
  | .subscript name idx =>
    let vs ← evalEs idx
    let is ← ofOpt (vs.mapM Val.toInt?)
    let st ← get
    if st.outDtypes.any (·.1 == name) then
      let v' := match st.outDtypes.find? (·.1 == name) with
        | some (_, some dt) => coerce dt v
        | _ => v
      set { st with arrayWrites := st.arrayWrites ++ [(name, is, v')] }
    else ofOpt none
  | _ => ofOpt none

/-- Execute one statement. -/
partial def execStmt : Stmt → M Unit
  | .pass => pure ()
  | .ifElse branches orelse => do
    let rec go : List (Expr × List Stmt) → M Unit
      | [] => orelse.forM execStmt
      | (c, body) :: rest => do
        if (← ofOpt (← evalE c).toBool?) then body.forM execStmt else go rest
    go branches
  | .forRange var args body => do
    let vs ← (← evalEs (Exprs.ofList args)).mapM fun v => ofOpt v.toInt?
    let (lo, hi, step) ← match vs with
      | [hi] => pure (0, hi, 1)
      | [lo, hi] => pure (lo, hi, 1)
      | [lo, hi, step] => pure (lo, hi, step)
      | _ => ofOpt none
    if step = 0 then ofOpt none
    let n := if step > 0 then (hi - lo + step - 1) / step else (lo - hi - step - 1) / (-step)
    for k in List.range n.toNat do
      assignTo (.sym var) (.int (lo + k * step))
      body.forM execStmt
  | .assign target none rhs => do assignTo target (← evalE rhs)
  | .assign target (some op) rhs => do
    let cur ← readTarget target
    let rv ← evalE rhs
    assignTo target (← ofOpt (Eval.arith op cur rv))
  | .exprStmt e =>
    match e with
    | .app f args =>
      if (f.endsWith ".push" || f.endsWith ".append") then
        match Exprs.toList args with
        | [a] => do pushStream (strDropRight f (if f.endsWith ".push" then 5 else 7)) (← evalE a)
        | _ => ofOpt none
      else do let _ ← evalApp f args; pure ()
    | _ => do let _ ← evalE e; pure ()

/-! ## Entry point -/

/-- Build the effect record from the final state. -/
def buildOut (st : RunState) : TaskletOut :=
  { scalars := st.scalars, arrayWrites := st.arrayWrites,
    pushes := st.pushes, pops := st.pops }

/-- Run a Python tasklet under the expression interpretation `I` and the
symbol valuation `ρ`.  Non-Python tasklets and unsupported bodies yield
`none`. -/
def run (I : ExprInterp) : TaskletSem := fun t ρ ins =>
  match t.code.language with
  | .python => do
    let stmts ← parseStmts t.code.code
    let init : RunState :=
      { I, ρ, ins, outDtypes := t.outConns,
        streams := t.inConns.filterMap fun (c, _) =>
          match ins c with
          | some (.stream avail) => some (c, avail)
          | _ => none }
    let (_, st) ← (stmts.forM execStmt).run init
    pure (buildOut st)
  | _ => none

end Dace.PyTasklet

namespace Dace.Interp

/-- The concrete executable interpretation: math-aware expressions, concrete
WCR, and the Python-expression tasklet subset. -/
def executable : Interp where
  expr := PyTasklet.exprInterp
  wcr := WcrOp.applyWith PyTasklet.exprInterp
  tasklet := PyTasklet.run PyTasklet.exprInterp
  library := fun _ _ => none

end Dace.Interp
