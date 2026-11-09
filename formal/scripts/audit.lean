-- Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import Dace

/-!
Dumps the declarations of the `Dace` library (and of an executable, when
`scripts/audit.py` substitutes the import above) as tab-separated lines for
`scripts/audit.py`:

    name  kind  module  firstLine  lastLine  typeHash  ownConstsInType  simp  dependencies

`dependencies` are the library's own constants used by the type or the
value.  The output file is named by the environment variable
`DACE_AUDIT_OUT`.
-/

open Lean Elab Command Meta

namespace DaceAudit

def ownModule? (env : Environment) (n : Name) : Option Name := do
  let idx ← env.getModuleIdxFor? n
  let m := env.header.moduleNames[idx.toNat]!
  if m.getRoot == `Dace || m.getRoot == `Tools then some m else none

def kindOf : ConstantInfo → String
  | .thmInfo _ => "thm"
  | .defnInfo _ => "def"
  | .inductInfo _ => "ind"
  | .ctorInfo _ => "ctor"
  | .recInfo _ => "rec"
  | .opaqueInfo _ => "opaque"
  | .axiomInfo _ => "axiom"
  | .quotInfo _ => "quot"

run_cmd do
  let env ← getEnv
  let some out ← IO.getEnv "DACE_AUDIT_OUT" | throwError "DACE_AUDIT_OUT is not set"
  let simps ← liftCoreM getSimpTheorems
  let mut lines : Array String := #[]
  for (n, ci) in env.constants.map₁.toList do
    let some m := ownModule? env n | continue
    let own := fun (d : Name) => (ownModule? env d).isSome && d != n
    let deps := ci.getUsedConstantsAsSet.toList.filter own
    let inType := (ci.type.getUsedConstants.toList.filter own).length
    let (first, last) := match ← findDeclarationRanges? n with
      | some r => (r.range.pos.line, r.range.endPos.line)
      | none => (0, 0)
    let isSimp := simps.isLemma (.decl n)
    lines := lines.push <| "\t".intercalate
      [toString n, kindOf ci, toString m, toString first, toString last, toString ci.type.hash,
       toString inType, toString isSimp, " ".intercalate (deps.map toString)]
  IO.FS.writeFile out ("\n".intercalate lines.toList ++ "\n")

end DaceAudit
