-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace.Core.Expr

/-!
# Parsing and printing of symbolic expressions

DaCe serializes SymPy expressions as Python-syntax strings (via
`dace.symbolic.symstr`).  This module provides a tokenizer and a precedence
parser for that language into `Expr`, and a printer back to the same syntax.

Recognized SymPy/DaCe function spellings (`Decisions.md#D22`):
`Min`, `Max` (variadic), `Abs`, `int_floor`, `int_ceil`, `Mod`, `floor`,
`ceiling`, `Not`, `Eq`, `Ne`/`NotEq`, `AND`/`And`, `OR`/`Or`, `IfExpr`,
`Pow`.  Anything else becomes an opaque `Expr.app`.  A typed symbol
`symbol($N, dtype=dace.int64, …)` is the symbol `N` (`Decisions.md#D66`).

The parser is a plain executable tool (`partial`): it is not reasoned about.
-/

namespace Dace.Parse

/-! ## Tokens -/

inductive Token where
  | ident (s : String)
  | int (v : Nat)
  | float (mant : Int) (exp : Int)
  | op (s : String)
  | lparen | rparen | lbracket | rbracket | comma | dot
  | eof
  deriving DecidableEq, Repr, Inhabited

private def isIdentStart (c : Char) : Bool := c.isAlpha || c = '_' || c = '$'
private def isIdentChar (c : Char) : Bool := c.isAlphanum || c = '_'

/-- Character array view of the input; `cs.getD i ' '` is safe past the end. -/
private abbrev Chars := Array Char

private def at? (cs : Chars) (i : Nat) : Char := cs.getD i ' '
private def inBounds (cs : Chars) (i : Nat) : Bool := i < cs.size

/-- Consume a run of decimal digits starting at `i`: value, digit count, next
position. -/
private partial def digits (cs : Chars) (i : Nat) (acc n : Nat) : Nat × Nat × Nat :=
  if inBounds cs i && (at? cs i).isDigit then
    digits cs (i + 1) (acc * 10 + ((at? cs i).toNat - '0'.toNat)) (n + 1)
  else (acc, n, i)

/-- Parse a decimal literal `digits[.digits][e[+-]digits]` starting at `i`.
Returns the token and the next position. -/
private def lexNumber (cs : Chars) (i : Nat) : Token × Nat :=
  let (ip, _, j) := digits cs i 0 0
  -- fractional part
  let (frac, nfrac, j, isFloat) :=
    if inBounds cs j && at? cs j = '.' then
      if inBounds cs (j + 1) && (at? cs (j + 1)).isDigit then
        let (f, n, j'') := digits cs (j + 1) 0 0
        (f, n, j'', true)
      else (0, 0, j + 1, true)
    else (0, 0, j, false)
  -- exponent
  let (expo, j, isFloat) :=
    if inBounds cs j && (at? cs j = 'e' || at? cs j = 'E') then
      let (neg, j2) :=
        if at? cs (j + 1) = '-' then (true, j + 2)
        else if at? cs (j + 1) = '+' then (false, j + 2)
        else (false, j + 1)
      if inBounds cs j2 && (at? cs j2).isDigit then
        let (e, _, j3) := digits cs j2 0 0
        ((if neg then -(e : Int) else e), j3, true)
      else (0, j, isFloat)
    else (0, j, isFloat)
  if isFloat then
    let mant : Int := ip * 10 ^ nfrac + frac
    (Token.float mant (expo - nfrac), j)
  else (Token.int ip, j)

private partial def identEnd (cs : Chars) (j : Nat) : Nat :=
  if inBounds cs j && isIdentChar (at? cs j) then identEnd cs (j + 1) else j

/-- Tokenize; `none` on an unexpected character. -/
partial def tokenize (s : String) : Option (List Token) :=
  let cs : Chars := s.toList.toArray
  let rec go (i : Nat) (acc : List Token) : Option (List Token) :=
    if inBounds cs i then
      let c := at? cs i
      if c.isWhitespace then go (i + 1) acc
      else if c.isDigit then
        let (t, j) := lexNumber cs i
        go j (t :: acc)
      else if isIdentStart c then
        -- serialized symbols carry a `$` prefix (`dace.symbolic.serialize_symbolic`)
        let start := if c = '$' then i + 1 else i
        let j := identEnd cs start
        if j = start then none
        else go j (Token.ident (String.ofList (cs.extract start j).toList) :: acc)
      else
        let two := String.ofList [c, at? cs (i + 1)]
        if two ∈ ["**", "//", "==", "!=", "<=", ">=", "<<", ">>"] then
          go (i + 2) (Token.op two :: acc)
        else
          match c with
          | '(' => go (i + 1) (Token.lparen :: acc)
          | ')' => go (i + 1) (Token.rparen :: acc)
          | '[' => go (i + 1) (Token.lbracket :: acc)
          | ']' => go (i + 1) (Token.rbracket :: acc)
          | ',' => go (i + 1) (Token.comma :: acc)
          | '.' => go (i + 1) (Token.dot :: acc)
          | '+' | '-' | '*' | '/' | '%' | '<' | '>' | '&' | '|' | '^' | '~' | '=' =>
            go (i + 1) (Token.op c.toString :: acc)
          | _ => none
    else some (Token.eof :: acc).reverse
  go 0 []

/-! ## Parser -/

structure PState where
  toks : Array Token
  pos : Nat := 0

abbrev P := StateT PState Option

def peek : P Token := do
  let st ← get
  pure (st.toks.getD st.pos .eof)

def advance : P Unit := modify fun st => { st with pos := st.pos + 1 }

def expect (t : Token) : P Unit := do
  if (← peek) = t then advance else failure

def peekOp (s : String) : P Bool := do
  pure ((← peek) = Token.op s)

/-- Keyword operators (`and`, `or`, `not`, `in`, `is`) arrive as identifiers. -/
def peekKw (s : String) : P Bool := do
  pure ((← peek) = Token.ident s)

/-- Fold a variadic call into nested binary operations. -/
private def foldBin (op : BinOp) : List Expr → Option Expr
  | [] => none
  | [a] => some a
  | a :: rest => (foldBin op rest).map fun r => .bin op a r

/-- Map a SymPy/DaCe function spelling to an `Expr`. -/
def mkCall (f : String) (args : List Expr) : Expr :=
  match f, args with
  | "Min", _ => (foldBin .min args).getD (.app f args)
  | "min", _ => (foldBin .min args).getD (.app f args)
  | "Max", _ => (foldBin .max args).getD (.app f args)
  | "max", _ => (foldBin .max args).getD (.app f args)
  | "Abs", [a] => .un .abs a
  | "abs", [a] => .un .abs a
  | "floor", [a] => .un .floor a
  | "ceiling", [a] => .un .ceil a
  | "ceil", [a] => .un .ceil a
  | "SymExpr", [a, b] => .approx a b
  | "int_floor", [a, b] => .bin .intFloor a b
  | "int_ceil", [a, b] => .bin .intCeil a b
  | "Mod", [a, b] => .bin .mod a b
  | "Pow", [a, b] => .bin .pow a b
  | "pow", [a, b] => .bin .pow a b
  | "Not", [a] => .un .not a
  | "not", [a] => .un .not a
  | "Eq", [a, b] => .bin .eq a b
  | "Ne", [a, b] => .bin .ne a b
  | "NotEq", [a, b] => .bin .ne a b
  | "Unequality", [a, b] => .bin .ne a b
  | "AND", _ => (foldBin .and args).getD (.app f args)
  | "And", _ => (foldBin .and args).getD (.app f args)
  | "OR", _ => (foldBin .or args).getD (.app f args)
  | "Or", _ => (foldBin .or args).getD (.app f args)
  | "IfExpr", [c, t, e] => .ite c t e
  | "Piecewise", _ => .app f args
  | _, _ => .app f args

/-- Dotted name of an attribute chain (`math.sqrt`, `dace.float64`), used to
turn method/qualified calls into opaque `app` nodes. -/
def dottedName? : Expr → Option String
  | .sym s => some s
  | .attr b f => (dottedName? b).map (· ++ "." ++ f)
  | _ => none

mutual

/-- Python conditional expression `a if c else b` (lowest precedence). -/
partial def parseTernary : P Expr := do
  let a ← parseOr
  if ← peekKw "if" then
    advance
    let c ← parseOr
    if ← peekKw "else" then
      advance
      let b ← parseTernary
      pure (.ite c a b)
    else failure
  else pure a

/-- `or` -/
partial def parseOr : P Expr := do
  let mut lhs ← parseAnd
  while ← peekKw "or" do
    advance
    let rhs ← parseAnd
    lhs := .bin .or lhs rhs
  pure lhs

partial def parseAnd : P Expr := do
  let mut lhs ← parseNot
  while ← peekKw "and" do
    advance
    let rhs ← parseNot
    lhs := .bin .and lhs rhs
  pure lhs

partial def parseNot : P Expr := do
  if ← peekKw "not" then
    advance
    let a ← parseNot
    pure (.un .not a)
  else parseCmp

partial def parseCmp : P Expr := do
  let lhs ← parseBitOr
  let t ← peek
  let op? : Option BinOp :=
    match t with
    | .op "==" => some .eq | .op "!=" => some .ne
    | .op "<" => some .lt | .op "<=" => some .le
    | .op ">" => some .gt | .op ">=" => some .ge
    | _ => none
  match op? with
  | some op =>
    advance
    let rhs ← parseBitOr
    -- chained comparisons `a < b < c` are conjunctions in Python
    let mut acc : Expr := .bin op lhs rhs
    let mut prev := rhs
    repeat
      let t2 ← peek
      let op2? : Option BinOp :=
        match t2 with
        | .op "==" => some .eq | .op "!=" => some .ne
        | .op "<" => some .lt | .op "<=" => some .le
        | .op ">" => some .gt | .op ">=" => some .ge
        | _ => none
      match op2? with
      | some op2 =>
        advance
        let r ← parseBitOr
        acc := .bin .and acc (.bin op2 prev r)
        prev := r
      | none => break
    pure acc
  | none => pure lhs

partial def parseBitOr : P Expr := do
  let mut lhs ← parseBitXor
  while ← peekOp "|" do
    advance
    let rhs ← parseBitXor
    lhs := .bin .bitOr lhs rhs
  pure lhs

partial def parseBitXor : P Expr := do
  let mut lhs ← parseBitAnd
  while ← peekOp "^" do
    advance
    let rhs ← parseBitAnd
    lhs := .bin .bitXor lhs rhs
  pure lhs

partial def parseBitAnd : P Expr := do
  let mut lhs ← parseShift
  while ← peekOp "&" do
    advance
    let rhs ← parseShift
    lhs := .bin .bitAnd lhs rhs
  pure lhs

partial def parseShift : P Expr := do
  let mut lhs ← parseAdd
  repeat
    if ← peekOp "<<" then
      advance
      let rhs ← parseAdd
      lhs := .bin .shl lhs rhs
    else if ← peekOp ">>" then
      advance
      let rhs ← parseAdd
      lhs := .bin .shr lhs rhs
    else break
  pure lhs

partial def parseAdd : P Expr := do
  let mut lhs ← parseMul
  repeat
    if ← peekOp "+" then
      advance
      let rhs ← parseMul
      lhs := .bin .add lhs rhs
    else if ← peekOp "-" then
      advance
      let rhs ← parseMul
      lhs := .bin .sub lhs rhs
    else break
  pure lhs

partial def parseMul : P Expr := do
  let mut lhs ← parseUnary
  repeat
    if ← peekOp "*" then
      advance
      let rhs ← parseUnary
      lhs := .bin .mul lhs rhs
    else if ← peekOp "/" then
      advance
      let rhs ← parseUnary
      lhs := .bin .div lhs rhs
    else if ← peekOp "//" then
      advance
      let rhs ← parseUnary
      lhs := .bin .intFloor lhs rhs
    else if ← peekOp "%" then
      advance
      let rhs ← parseUnary
      lhs := .bin .mod lhs rhs
    else break
  pure lhs

partial def parseUnary : P Expr := do
  if ← peekOp "-" then
    advance
    let a ← parseUnary
    pure (match a with
      | .lit (.int v) => .lit (.int (-v))
      | .lit (.float m e) => .lit (.float (-m) e)
      | _ => .un .neg a)
  else if ← peekOp "+" then
    advance
    parseUnary
  else if ← peekOp "~" then
    advance
    let a ← parseUnary
    pure (.un .bitNot a)
  else parsePow

/-- `**` is right-associative and binds tighter than unary minus on its
left operand but *not* on its right (`-a**b = -(a**b)`, `a**-b` allowed). -/
partial def parsePow : P Expr := do
  let base ← parsePostfix
  if ← peekOp "**" then
    advance
    let ex ← parseUnary
    pure (.bin .pow base ex)
  else pure base

partial def parsePostfix : P Expr := do
  let mut e ← parseAtom
  repeat
    match ← peek with
    | .lbracket =>
      advance
      let idx ← parseArgs .rbracket
      match e with
      | .sym d => e := .subscript d idx
      | _ => failure
    | .dot =>
      advance
      match ← peek with
      | .ident f =>
        advance
        if (← peek) = .lparen then
          -- qualified / method call `base.f(args)` → opaque `app "base.f" args`
          advance
          let args ← parseArgs .rparen
          match dottedName? e with
          | some base => e := .app (base ++ "." ++ f) args
          | none => failure
        else e := .attr e f
      | _ => failure
    | _ => break
  pure e

/-- Comma-separated arguments up to (and consuming) `close`. -/
partial def parseArgs (close : Token) : P (List Expr) := do
  if (← peek) = close then advance; pure []
  else
    let mut args : List Expr := []
    repeat
      let a ← parseTernary
      args := args ++ [a]
      match ← peek with
      | .comma => advance
      | t => if t = close then advance; break else failure
    pure args

partial def parseAtom : P Expr := do
  match ← peek with
  | .int v => advance; pure (.lit (.int v))
  | .float m e => advance; pure (.lit (.float m e))
  | .ident "True" => advance; pure (.lit (.bool true))
  | .ident "False" => advance; pure (.lit (.bool false))
  | .ident "symbol" =>
    advance
    if (← peek) = .lparen then
      -- typed symbol `symbol($N, dtype=dace.int64, nonnegative=True, …)`
      -- (`DaceSympySerializer._print_Symbol`): the symbol `N`; its dtype and
      -- assumptions are dropped (`Decisions.md#D22`, `#D1`)
      advance
      match ← peek with
      | .ident x =>
        advance
        parseSymbolKwargs
        pure (.sym x)
      | _ => failure
    else pure (.sym "symbol")
  | .ident f =>
    advance
    if (← peek) = .lparen then
      advance
      let args ← parseArgs .rparen
      pure (mkCall f args)
    else pure (.sym f)
  | .lparen =>
    advance
    let e ← parseTernary
    expect .rparen
    pure e
  | _ => failure

/-- The keyword arguments `, k = v` of a typed symbol up to (and consuming)
the closing parenthesis; the values are parsed and discarded. -/
partial def parseSymbolKwargs : P Unit := do
  repeat
    match ← peek with
    | .rparen => advance; break
    | .comma =>
      advance
      match ← peek with
      | .ident _ =>
        advance
        expect (.op "=")
        let _ ← parseTernary
      | _ => failure
    | _ => failure

end

/-- Parse a DaCe symbolic-expression string. -/
def parseExpr (s : String) : Option Expr := do
  let toks ← tokenize s
  let (e, st) ← (parseTernary).run { toks := toks.toArray }
  if st.toks.getD st.pos .eof = .eof then some e else none

/-- Parse an expression from a token list (a slice of a tokenized statement).
A single `Token.eof` is appended so the precedence parser terminates. -/
def parseFromToks (toks : List Token) : Option Expr := do
  let toks := (toks.filter (· != Token.eof)) ++ [Token.eof]
  let (e, st) ← (parseTernary).run { toks := toks.toArray }
  if st.toks.getD st.pos .eof = .eof then some e else none

/-- Parse a single assignment statement `name = expr` (loop init/update
statements, interstate assignments). -/
def parseAssign (s : String) : Option (Sym × Expr) := do
  let toks ← tokenize s
  match toks with
  | .ident name :: .op "=" :: rest =>
    let (e, st) ← (parseTernary).run { toks := rest.toArray }
    if st.toks.getD st.pos .eof = .eof then some (name, e) else none
  | _ => none

/-! ## Printing -/

/-- Precedence levels (higher binds tighter), following Python. -/
private def prec : Expr → Nat
  | .bin .or _ _ => 1
  | .bin .and _ _ => 2
  | .un .not _ => 3
  | .bin .eq _ _ | .bin .ne _ _ | .bin .lt _ _ | .bin .le _ _ | .bin .gt _ _ | .bin .ge _ _ => 4
  | .bin .bitOr _ _ => 5
  | .bin .bitXor _ _ => 6
  | .bin .bitAnd _ _ => 7
  | .bin .shl _ _ | .bin .shr _ _ => 8
  | .bin .add _ _ | .bin .sub _ _ => 9
  | .bin .mul _ _ | .bin .div _ _ | .bin .intFloor _ _ | .bin .mod _ _ => 10
  | .un .neg _ | .un .bitNot _ => 11
  | .bin .pow _ _ => 12
  | _ => 13

private def binSym : BinOp → Option String
  | .add => some "+" | .sub => some "-" | .mul => some "*" | .div => some "/"
  | .intFloor => some "//" | .mod => some "%" | .pow => some "**"
  | .bitAnd => some "&" | .bitOr => some "|" | .bitXor => some "^"
  | .shl => some "<<" | .shr => some ">>"
  | .and => some "and" | .or => some "or"
  | .eq => some "==" | .ne => some "!=" | .lt => some "<" | .le => some "<="
  | .gt => some ">" | .ge => some ">="
  | .intCeil | .min | .max => none

/-- Print a float literal `mant * 10^exp` in a Python-readable form. -/
def floatLitToString (mant exp : Int) : String :=
  if exp ≥ 0 then s!"{mant * 10 ^ exp.toNat}.0"
  else
    let digits := mant.natAbs.repr
    let n := exp.natAbs
    let padded := if digits.length ≤ n then "".pushn '0' (n - digits.length + 1) ++ digits else digits
    let split := padded.length - n
    let ip := padded.take split
    let fp := padded.drop split
    (if mant < 0 then "-" else "") ++ ip ++ "." ++ fp

/-- Print an expression in DaCe's Python syntax. -/
partial def toPy : Expr → String
  | .lit (.int v) => if v < 0 then s!"({v})" else toString v
  | .lit (.float m e) => floatLitToString m e
  | .lit (.bool b) => if b then "True" else "False"
  | .sym s => s
  | .un .neg a => s!"-{wrap 11 a}"
  | .un .bitNot a => s!"~{wrap 11 a}"
  | .un .not a => s!"not {wrap 3 a}"
  | .un .abs a => s!"Abs({toPy a})"
  | .un .floor a => s!"floor({toPy a})"
  | .un .ceil a => s!"ceiling({toPy a})"
  | .approx a o => s!"SymExpr({toPy a}, {toPy o})"
  | e@(.bin op a b) =>
    match binSym op with
    | some s =>
      let p := prec e
      -- left-assoc: left operand may have equal precedence, right needs strictly higher
      if op = .pow then s!"{wrap (p + 1) a} ** {wrap p b}"
      else s!"{wrap p a} {s} {wrap (p + 1) b}"
    | none =>
      match op with
      | .intCeil => s!"int_ceil({toPy a}, {toPy b})"
      | .min => s!"Min({toPy a}, {toPy b})"
      | .max => s!"Max({toPy a}, {toPy b})"
      | _ => "?"
  | .ite c t e => s!"IfExpr({toPy c}, {toPy t}, {toPy e})"
  | .app f args => s!"{f}({", ".intercalate (args.toList.map toPy)})"
  | .subscript d idx => s!"{d}[{", ".intercalate (idx.toList.map toPy)}]"
  | .attr b f => s!"{wrap 13 b}.{f}"
where
  wrap (p : Nat) (e : Expr) : String :=
    if prec e < p then s!"({toPy e})" else toPy e

end Dace.Parse

namespace Dace.Expr

/-- Parse from DaCe's string syntax. -/
def parse? (s : String) : Option Expr := Dace.Parse.parseExpr s

/-- Parse an assignment statement `name = expr`. -/
def parseAssign? (s : String) : Option (Sym × Expr) := Dace.Parse.parseAssign s

/-- Print in DaCe's string syntax. -/
def toPy (e : Expr) : String := Dace.Parse.toPy e

instance : ToString Expr := ⟨toPy⟩

end Dace.Expr
