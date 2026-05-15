// ============================================================================
// extract_ast.h  --  Build a recursive statement tree for a Fortran subroutine.
// ============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"

namespace hlfir_bridge {

/// One array access found inside an hlfir.assign expression tree.
///
/// `index_vars[d]` is the best-effort source name for dimension `d` (a loop
/// iterator, a scalar symbol, or  --  for indirect accesses  --  the array that
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
///   "loop"         --  loop_iter, loop_bound, loop_lower, children[]
///   "assign"       --  target, expr, target_is_array, accesses[]
///   "while"        --  condition, children[]
///   "conditional"  --  condition, children[], else_children[]
///   "call"         --  callee, call_args[]
///   "reduce"       --  target, reduce_src, reduce_wcr, reduce_identity,
///                   reduce_axes (empty = whole-array)
///   "copy"         --  target, reduce_src (source array name)
///   "memset"       --  target (memset always writes zero today; the
///                   MemsetLibraryNode side is hard-coded for 0)
///   "libcall"      --  target, callee ("matmul" / "transpose" /
///                   "dot_product" / "count" / "merge"), call_args
///                   (1-3 source array names), target_is_array.
///                   ``reduce_axes`` carries the optional ``dim``
///                   for reduction-shaped library nodes (count).
///   "declare_transient"
///                 --  Bridge-synthesised transient array used as a
///                   scratch surface between an ``hlfir.elemental``
///                   walker (per-element loop) and a downstream
///                   reduction / select library node.  Payload:
///                     target            --  transient name (unique gid)
///                     expr              --  dtype string ("int32"...)
///                     accesses[0].index_exprs  --  per-dim shape
///                                       strings (literal int or
///                                       symbol name)
///                   The Python emitter calls
///                   ``sdfg.add_array(name, shape, dtype, transient=True)``
///                   and registers the array in ``builder.arrays``.
///   "break"        --  Fortran EXIT (break out of the enclosing loop).
///                   No payload  --  hlfir_to_sdfg emits a ``BreakBlock``
///                   in the current region.
///   "return"       --  Fortran RETURN (exit the subroutine early).
///                   No payload  --  hlfir_to_sdfg emits a ``ReturnBlock``
///                   at SDFG top level.
struct ASTNode {
  std::string kind;

  // loop
  std::string loop_iter, loop_bound;
  // Lower bound  --  int form covers the common ``DO i = 1, n`` case from
  // fir.do_loop where Flang resolves the lower to a constant.  For
  // symbolic lowers (e.g. ``res(a:b) = ...`` from array-section
  // assign) set ``loop_lower_expr`` to a Fortran expression string;
  // the emitter prefers the string whenever it is non-empty.
  int64_t loop_lower = -1;
  std::string loop_lower_expr;
  // Step (Fortran ``DO i = a, b, c``'s ``c``).  Default 1; -1 for
  // reverse-direction ``DO i = N, 1, -1`` (LU back-substitution
  // pattern).  Other steps are not yet supported.
  int64_t loop_step = 1;

  // assign
  std::string target, expr;
  bool target_is_array = false;
  std::vector<AccessInfo> accesses;

  // conditional / while
  std::string condition;

  // call
  std::string callee;
  std::vector<std::string> call_args;
  // Per-call-arg slice subset, parallel to ``call_args``.  Empty entry =
  // whole array; non-empty entry = a Fortran 1-based slice expression
  // like ``"1:3"`` so emit_libcall can build a sliced memlet
  // (``dot_product(arg1(1:3), arg2(1:3))`` etc.).
  std::vector<std::string> call_arg_subsets;

  // reduce
  std::string reduce_src;            // input array name
  std::string reduce_wcr;            // lambda string, e.g. "lambda a, b: a + b"
  std::string reduce_identity;       // initial-accumulator string, e.g. "0"
  std::vector<int64_t> reduce_axes;  // empty = reduce all dimensions

  // recursive
  std::vector<ASTNode> children, else_children;
};

/// Build the AST for the first func.func found in the module.
std::vector<ASTNode> extractAST(mlir::ModuleOp module);

}  // namespace hlfir_bridge
