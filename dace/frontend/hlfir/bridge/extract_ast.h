// ============================================================================
// extract_ast.h — Build a recursive statement tree for a Fortran subroutine.
// ============================================================================

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include <cstdint>
#include <string>
#include <vector>

namespace hlfir_bridge {

/// One array access found inside an hlfir.assign expression tree.
struct AccessInfo {
    std::string array_name;
    std::vector<std::string> index_vars;
    bool is_read = false;
    bool is_write = false;
};

/// Recursive AST node.  `kind` discriminates which fields are populated:
///
///   "loop"        — loop_iter, loop_bound, loop_lower, children[]
///   "assign"      — target, expr, target_is_array, accesses[]
///   "conditional" — condition, children[], else_children[]
///   "call"        — callee, call_args[]
struct ASTNode {
    std::string kind;

    // loop
    std::string loop_iter, loop_bound;
    int64_t loop_lower = -1;

    // assign
    std::string target, expr;
    bool target_is_array = false;
    std::vector<AccessInfo> accesses;

    // conditional
    std::string condition;

    // call
    std::string callee;
    std::vector<std::string> call_args;

    // recursive
    std::vector<ASTNode> children, else_children;
};

/// Build the AST for the first func.func found in the module.
std::vector<ASTNode> extractAST(mlir::ModuleOp module);

}  // namespace hlfir_bridge
