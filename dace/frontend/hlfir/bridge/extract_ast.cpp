// ============================================================================
// extract_ast.cpp — Build a recursive statement tree from HLFIR.
// ============================================================================
// Statement-level ops become nodes; everything else is expression-level
// infrastructure and gets folded into the expression strings or access lists.
//
// This file used to be a single 2800-line monolith.  It's now split into
// five logical chunks under ``ast/``, each included verbatim below:
//
//   * ``ast/expressions.inc``   — buildExpr + index/designate helpers
//                                 + indexStack + alloca synth names.
//   * ``ast/assigns.inc``       — buildAssignNode / buildCopyNode /
//                                 buildLibCallNode / section builders
//                                 + small type helpers (peelWrappers,
//                                 isArrayRef, …).
//   * ``ast/elementals.inc``    — buildReduceNode +
//                                 buildElementalCountLibcall +
//                                 buildSelectCaseChain + libcall-in-
//                                 elemental materialisation.
//   * ``ast/control_flow.inc``  — buildMergeLibcall +
//                                 buildElementalAssign + cmp predicates
//                                 + buildBoolExpr / buildExprWithSubscripts.
//   * ``ast/dispatch.inc``      — scf.if / scf.while walkers,
//                                 buildAST(Block&) per-op dispatcher,
//                                 and the public extractAST(ModuleOp).
//
// Each chunk is included once in the order shown.  They share this
// translation unit's namespace, includes, and file-static state — so
// existing ``static`` / ``thread_local`` declarations keep their
// original linkage and there's no forward-declaration churn at the
// chunk boundaries.  The ``.inc`` extension is the standard signal
// that these files are NOT translation units (cf. LLVM's ``.inc``
// files, clang's TableGen output) — clangd / IDEs / glob-based
// linters won't try to compile them standalone.  CMakeLists.txt
// deliberately compiles only this file.
// ============================================================================

#include "bridge/extract_ast.h"
#include "bridge/extract_vars.h"
#include "bridge/trace_utils.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <map>
#include <set>
#include <iomanip>
#include <sstream>

namespace hlfir_bridge {

#include "bridge/ast/expressions.inc"
#include "bridge/ast/assigns.inc"
#include "bridge/ast/elementals.inc"
#include "bridge/ast/control_flow.inc"
#include "bridge/ast/dispatch.inc"

}  // namespace hlfir_bridge
