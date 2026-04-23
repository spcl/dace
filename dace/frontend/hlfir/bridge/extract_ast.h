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
///
/// `index_vars[d]` is the best-effort source name for dimension `d` (a loop
/// iterator, a scalar symbol, or — for indirect accesses — the array that
/// supplied the index).
///
/// `index_exprs[d]` is a richer expression string (same length as
/// index_vars).  For a loop iterator it matches `index_vars[d]`; for a
/// constant it is the decimal literal; for an indirect access like
/// ``z_kin(edge_idx(jc,1), jk)`` the first entry is ``edge_idx[jc,1]``
/// (Fortran 1-based, square-bracket notation) so the SDFG generator can
/// detect indirection by looking for ``[`` in the expression.
struct AccessInfo {
    std::string array_name;
    std::vector<std::string> index_vars;
    std::vector<std::string> index_exprs;
    bool is_read = false;
    bool is_write = false;
};

/// Recursive AST node.  `kind` discriminates which fields are populated:
///
///   "loop"        — loop_iter, loop_bound, loop_lower, children[]
///   "assign"      — target, expr, target_is_array, accesses[]
///   "while"       — condition, children[]
///   "conditional" — condition, children[], else_children[]
///   "call"        — callee, call_args[]
///   "reduce"      — target, reduce_src, reduce_wcr, reduce_identity,
///                   reduce_axes (empty = whole-array)
///   "copy"        — target, reduce_src (source array name)
///   "memset"      — target (memset always writes zero today; the
///                   MemsetLibraryNode side is hard-coded for 0)
struct ASTNode {
    std::string kind;

    // loop
    std::string loop_iter, loop_bound;
    int64_t loop_lower = -1;

    // assign
    std::string target, expr;
    bool target_is_array = false;
    std::vector<AccessInfo> accesses;

    // conditional / while
    std::string condition;

    // call
    std::string callee;
    std::vector<std::string> call_args;

    // reduce
    std::string reduce_src;       // input array name
    std::string reduce_wcr;       // lambda string, e.g. "lambda a, b: a + b"
    std::string reduce_identity;  // initial-accumulator string, e.g. "0"
    std::vector<int64_t> reduce_axes;  // empty = reduce all dimensions

    // recursive
    std::vector<ASTNode> children, else_children;
};

/// Build the AST for the first func.func found in the module.
std::vector<ASTNode> extractAST(mlir::ModuleOp module);

}  // namespace hlfir_bridge
