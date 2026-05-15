// extract_ast.cpp  --  public entry point ``extractAST(ModuleOp)``.
//
// AST extraction is split across five sibling translation units under
// ``ast/``  --  expressions, assigns, elementals, control_flow, dispatch.
// They share state through ``ast/ast_helpers.h`` (cross-file function
// declarations + thread-locals).  This file holds only the includes
// that pull the dialect headers in once for the whole bundle.

#include "bridge/extract_ast.h"

#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

#include "bridge/extract_vars.h"
#include "bridge/trace_utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// The AST extraction is split across five sibling translation units
// (``ast/expressions.cpp``, ``ast/assigns.cpp``, ``ast/elementals.cpp``,
// ``ast/control_flow.cpp``, ``ast/dispatch.cpp``).  ``ast_helpers.h``
// declares every cross-file function and shares the thread-local state
// they all read from.
#include "bridge/ast/ast_helpers.h"

namespace hlfir_bridge {}  // namespace hlfir_bridge
