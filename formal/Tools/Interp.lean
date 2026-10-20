-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-!
# `lake exe dace-interp`

Run an SDFG or a schedule tree on concrete inputs with the executable
interpreters (`Dace.SDFG.Interp` / `Dace.STree.Interp`, interpretation
`Interp.executable`):

```
lake exe dace-interp sdfg <sdfg.json> <inputs.json> [--fuel N]
lake exe dace-interp stree <stree.json> <inputs.json> [--fuel N]
```

The inputs file has the format produced by `scripts/difftest.py --record`
(`symbols`, `containers` with `dtype`/`shape`/row-major `data`).  On success
the tool prints `{"containers": {...}}` with every non-transient container of
the program and exits with `0`; a semantic error prints
`{"error": "...", "kind": k, "reason": r}` and exits with `1`, where `k` is the
error class (`ErrReason.kind`: `ub`, `malformed` or `unsupported`,
`Decisions.md#D59`) or `tool` for import/read-back failures and missing
outcomes, and `r` the error's constructor (`ErrReason.tag`, e.g. `race`;
`tool` for the tool's own failures); usage errors and unsupported IR kinds
exit with `2`.

Containers are laid out in memory exactly as the semantics addresses them:
element `idx` of container `n` lives at `Loc.elem n o` with `o` computed by
`Ctx.elemOffset` from the descriptor's strides and start offset.  For an SDFG
the descriptors are its `arrays`; for a schedule tree they are the tree's
descriptor repository, through the same context the semantics uses
(`TCtx.ofTree`, i.e. `STree.toSDFG`).
-/

open Dace
open Lean (Json JsonNumber)

namespace DaceInterp

/-! ## Floats as round-trippable decimals

`Float.toString` prints six fractional digits, so floats are formatted from
their exact binary value: the shortest decimal with at most 17 significant
digits that lies strictly within half an ulp of the value (hence parses back
to it under any correctly rounded reader). -/

/-- Sign, integral mantissa and binary exponent of a finite double: `|x| = m · 2^e`. -/
def floatParts (x : Float) : Bool × Nat × Int :=
  let bits := x.toBits
  let neg := (bits >>> 63) != 0
  let ex := ((bits >>> 52) &&& 0x7FF).toNat
  let frac := (bits &&& 0xFFFFFFFFFFFFF).toNat
  if ex = 0 then (neg, frac, -1074) else (neg, frac + 2 ^ 52, (ex : Int) - 1075)

/-- Number of decimal digits of a natural number (`0` has one digit). -/
def numDigits (n : Nat) : Nat := (toString n).length

/-- Round `d` to `p` significant digits (half to even): returns the rounded
digits and the number of digits cut. -/
def roundSig (d p : Nat) : Nat × Nat :=
  let len := numDigits d
  if len ≤ p then (d, 0)
  else
    let cut := len - p
    let base := 10 ^ cut
    let q := d / base
    let r := d % base
    (if 2 * r > base || (2 * r = base && q % 2 = 1) then q + 1 else q, cut)

/-- Strip trailing decimal zeros from `q · 10^e`. -/
def stripZeros : Nat → Nat → Int → Nat × Int
  | 0, q, e => (q, e)
  | fuel + 1, q, e => if q ≠ 0 && q % 10 = 0 then stripZeros fuel (q / 10) (e + 1) else (q, e)

def zeros (n : Nat) : String := String.ofList (List.replicate n '0')

/-- Render `(-1)^neg · q · 10^e` in a form Python's `float` accepts. -/
def fmtDecimal (neg : Bool) (q : Nat) (e : Int) : String :=
  let (q, e) := stripZeros (numDigits q) q e
  let s := toString q
  let len : Int := s.length
  let pos := len + e
  let body :=
    if q = 0 then "0.0"
    else if 0 < pos && pos ≤ 21 then
      if pos ≥ len then s ++ zeros (pos - len).toNat ++ ".0"
      else String.ofList (s.toList.take pos.toNat) ++ "." ++ String.ofList (s.toList.drop pos.toNat)
    else if -6 < pos && pos ≤ 0 then "0." ++ zeros (-pos).toNat ++ s
    else
      let frac := String.ofList (s.toList.drop 1)
      String.ofList (s.toList.take 1) ++ "." ++ (if frac.isEmpty then "0" else frac) ++ "e" ++ toString (pos - 1)
  if neg then "-" ++ body else body

/-- Format a double so that it parses back to the same value. -/
def fmtFloat (x : Float) : String :=
  if x.isNaN then "NaN"
  else if x.isInf then (if x < 0 then "-Infinity" else "Infinity")
  else
    let (neg, m, e) := floatParts x
    if m = 0 then (if neg then "-0.0" else "0.0")
    else
      -- exact value: digits · 10^e10, with the ulp expressed in units of 10^e10
      let (digits, e10, ulp) : Nat × Int × Nat :=
        if e ≥ 0 then (m * 2 ^ e.toNat, 0, 2 ^ e.toNat)
        else (m * 5 ^ (-e).toNat, e, 5 ^ (-e).toNat)
      -- below a power of two the gap is half as large
      let halfGap := if m = 2 ^ 52 then ulp else 2 * ulp
      let ok (p : Nat) : Bool :=
        let (q, cut) := roundSig digits p
        let diff := (Int.ofNat (q * 10 ^ cut) - Int.ofNat digits).natAbs
        4 * diff < halfGap
      let p := ((List.range 17).map (· + 1)).find? ok |>.getD 17
      let (q, cut) := roundSig digits p
      fmtDecimal neg q (e10 + cut)

/-! ## JSON helpers -/

def jstr (s : String) : String := (Json.str s).compress

def fmtVal : Val → String
  | .int v => toString v
  | .bool b => if b then "true" else "false"
  | .float f => fmtFloat f
  | .rat q => fmtFloat (Val.toFloat (.rat q))

/-- Convert a JSON element to the value of a container of NumPy dtype `dtype`. -/
def valOfJson (dtype : String) (j : Json) : Except String Val :=
  if dtype.startsWith "float" then
    match j with
    | .num n => pure (.float n.toFloat)
    | .bool b => pure (.float (if b then 1.0 else 0.0))
    | _ => throw "float element expected"
  else if dtype.startsWith "bool" then
    match j with
    | .bool b => pure (.bool b)
    | .num n => pure (.bool (n.mantissa != 0))
    | _ => throw "bool element expected"
  else
    match j with
    | .num n =>
      if n.exponent = 0 then pure (.int n.mantissa)
      else pure (.int (PyTasklet.castToInt (.float n.toFloat)))
    | .bool b => pure (.int (if b then 1 else 0))
    | _ => throw "integer element expected"

/-- NumPy dtype name of a descriptor dtype. -/
def dtypeName : Dtype → String
  | .float bits => s!"float{bits}"
  | .int bits true => s!"int{bits}"
  | .int bits false => s!"uint{bits}"
  | .bool => "bool"
  | .opaque n => n

/-- One container of the inputs file. -/
structure ContainerIn where
  dtype : String
  shape : List Nat
  data : List Json

/-- Parsed inputs file. -/
structure Inputs where
  symbols : List (Sym × Val)
  containers : List (Name × ContainerIn)

open SDFG.Json (field? entries natOf) in
def parseInputs (text : String) : Except String Inputs := do
  let j ← Json.parse text
  let symbols ← match field? j "symbols" with
    | none => pure []
    | some sj => do
      let es ← entries sj
      es.mapM fun (k, v) => do
        match v with
        | .num n =>
          if n.exponent = 0 then pure (k, Val.int n.mantissa)
          else throw s!"symbol '{k}' is not an integer"
        | .bool b => pure (k, Val.bool b)
        | _ => throw s!"symbol '{k}' is not a number"
  let containers ← match field? j "containers" with
    | none => pure []
    | some cj => do
      let es ← entries cj
      es.mapM fun (name, c) => do
        let dtype ← match field? c "dtype" with
          | some d => d.getStr?
          | none => throw s!"container '{name}' without dtype"
        let shape ← match field? c "shape" with
          | some s => do (← s.getArr?).toList.mapM natOf
          | none => pure []
        let data ← match field? c "data" with
          | some d => do pure (← d.getArr?).toList
          | none => pure []
        pure (name, { dtype, shape, data })
  pure { symbols, containers }

/-- Row-major enumeration of a shape. -/
def shapePoints (shape : List Nat) : List Idx :=
  Range.product (shape.map fun n => (List.range n).map Int.ofNat)

/-! ## Memory set-up and read-back -/

/-- Element offset of `idx` in container `name` as the semantics computes it. -/
def offsetOf (ctx : Ctx) (ρ : Valuation) (name : Name) (idx : Idx) : Except String Int :=
  match ctx.desc? name with
  | none => throw s!"container '{name}' is not declared in the SDFG"
  | some d =>
    match ctx.elemOffset ρ d idx with
    | some o => pure o
    | none => throw s!"undefined element offset {idx} in '{name}'"

/-- Initial memory: every input element at its location; other cells are `Val.int 0`. -/
def buildMem (ctx : Ctx) (ρ : Valuation) (ins : Inputs) : Except String Mem :=
  ins.containers.foldlM (fun m (name, c) => do
    if (ctx.desc? name).isNone then
      -- inputs may mention containers the SDFG does not have; ignore them
      pure m
    else
      let pts := shapePoints c.shape
      (pts.zip c.data).foldlM (fun m (idx, jv) => do
        let v ← valOfJson c.dtype jv
        let o ← offsetOf ctx ρ name idx
        pure (m.write (Loc.elem name o) v)) m) (Mem.empty (.int 0))

/-- Shape of a container as reported: the inputs' shape, else the descriptor's
shape evaluated under `ρ` (`[]` for scalars). -/
def shapeOf (ctx : Ctx) (ρ : Valuation) (name : Name) (d : DataDesc) (ins : Inputs) : Except String (List Nat) :=
  match ins.containers.find? (·.1 = name) with
  | some (_, c) => pure c.shape
  | none =>
    if d.isScalar then pure []
    else
      match d.shape.mapM (Expr.evalInt ctx.I.expr ρ) with
      | some dims => pure (dims.map Int.toNat)
      | none => throw s!"undefined shape of '{name}'"

/-- Serialize the non-transient containers of the SDFG from memory `m`. -/
def renderContainers (ctx : Ctx) (ρ : Valuation) (ins : Inputs) (m : Mem) : Except String String := do
  let entries ← ctx.g.arrays.filterMapM fun (name, d) => do
    if d.transient then pure none
    else
      let shape ← shapeOf ctx ρ name d ins
      let dtype := match ins.containers.find? (·.1 = name) with
        | some (_, c) => c.dtype
        | none => dtypeName d.dtype
      let vals ← (shapePoints shape).mapM fun idx => do
        let o ← offsetOf ctx ρ name idx
        pure (m.read (Loc.elem name o))
      let shapeStr := ", ".intercalate (shape.map toString)
      let dataStr := ", ".intercalate (vals.map fmtVal)
      pure (some (jstr name ++ ": {\"dtype\": " ++ jstr dtype ++ ", \"shape\": [" ++ shapeStr ++
        "], \"data\": [" ++ dataStr ++ "]}"))
  pure ("{\"containers\": {" ++ ", ".intercalate entries ++ "}}")

/-! ## Driver -/

structure Options where
  fuel : Nat := 10000000

def parseOptions : List String → Except String Options
  | [] => pure {}
  | "--fuel" :: n :: rest => do
    let o ← parseOptions rest
    match n.toNat? with
    | some k => pure { o with fuel := k }
    | none => throw s!"bad fuel '{n}'"
  | opt :: _ => throw s!"unknown option '{opt}'"

def usage : String :=
  "usage: dace-interp (sdfg <sdfg.json> | stree <stree.json>) <inputs.json> [--fuel N]"

def errorExit (msg : String) (kind : String := "tool") (reason : String := "tool") : IO UInt32 := do
  IO.println ("{\"error\": " ++ jstr msg ++ ", \"kind\": " ++ jstr kind ++ ", \"reason\": " ++ jstr reason ++ "}")
  return 1

/-- Shared driver of both IR kinds: read the inputs, lay them out in memory
over the containers of `ctx`, run the program with `run` and print the
non-transient containers. -/
def runProgram (ctx : Ctx) (inputsPath : String) (opts : Options)
    (run : Valuation → Mem → Nat → Option (Res Mem)) : IO UInt32 := do
  let ins ← match parseInputs (← IO.FS.readFile inputsPath) with
    | .ok i => pure i
    | .error e => IO.eprintln s!"dace-interp: cannot read inputs: {e}"; return 2
  let ρ₀ : Valuation := fun x => (ins.symbols.find? (·.1 = x)).map (·.2)
  let m₀ ← match buildMem ctx ρ₀ ins with
    | .ok m => pure m
    | .error e => IO.eprintln s!"dace-interp: cannot build memory: {e}"; return 2
  match run ρ₀ m₀ opts.fuel with
  | none => errorExit "out of fuel (or no outcome exists)"
  | some (.err e) => errorExit (toString e) e.kind e.tag
  | some (.ok m) =>
    match renderContainers ctx ρ₀ ins m with
    | .ok out => IO.println out; return 0
    | .error e => errorExit s!"cannot read back results: {e}"

def runSDFG (sdfgPath inputsPath : String) (opts : Options) : IO UInt32 := do
  match ← SDFG.Json.readSDFG sdfgPath with
  | .error e => errorExit s!"cannot import SDFG: {e}"
  | .ok g => runProgram { I := Interp.executable, g, pfx := "" } inputsPath opts (SDFG.run Interp.executable g)

/-- The schedule tree runs in the context of its own descriptor repository
(`TCtx.ofTree`), exactly as `STree.Exec` does. -/
def runSTree (streePath inputsPath : String) (opts : Options) : IO UInt32 := do
  match ← readSTree streePath with
  | .error e => errorExit s!"cannot import schedule tree: {e}"
  | .ok t => runProgram (TCtx.ofTree Interp.executable t).ctx inputsPath opts (STree.run Interp.executable t)

end DaceInterp

open DaceInterp in
def main (args : List String) : IO UInt32 := do
  match args with
  | "sdfg" :: sdfgPath :: inputsPath :: rest =>
    match parseOptions rest with
    | .ok opts => runSDFG sdfgPath inputsPath opts
    | .error e => IO.eprintln s!"dace-interp: {e}\n{usage}"; return 2
  | "stree" :: streePath :: inputsPath :: rest =>
    match parseOptions rest with
    | .ok opts => runSTree streePath inputsPath opts
    | .error e => IO.eprintln s!"dace-interp: {e}\n{usage}"; return 2
  | "sdfg" :: _ | "stree" :: _ =>
    IO.eprintln s!"dace-interp: missing arguments\n{usage}"
    return 2
  | kind :: _ =>
    IO.eprintln s!"dace-interp: unsupported IR kind '{kind}'\n{usage}"
    return 2
  | [] =>
    IO.eprintln usage
    return 2
