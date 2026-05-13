// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Cross-file API for the HLFIR AST extraction layer.
//
// The bridge's AST extraction is split across five translation units
// (expressions / assigns / elementals / control_flow / dispatch), one
// per ``.cpp`` file in this directory.  This header collects:
//
//   * the thread-local state shared between TUs (counters and caches
//     that mint synthetic SDFG names + symbols for the duration of one
//     ``extractAST`` call),
//   * the cross-TU function declarations (signatures + docstrings) so
//     each chunk can call into the others without re-declaring,
//   * the ``NoSubscriptGuard`` RAII helper that scopes the
//     ``kBoolExprNoSubscripts`` flag.
//
// The header is included by every ``ast/*.cpp`` and by
// ``bridge/extract_ast.cpp``.  Globals are ``inline thread_local`` so
// the C++17 single-definition rule holds across TUs without a separate
// ``ast_state.cpp``.
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "bridge/extract_ast.h"
#include "bridge/extract_vars.h"
#include "bridge/trace_utils.h"

namespace hlfir_bridge {

// ============================================================================
// Thread-local state
// ============================================================================

/// Synthetic-scalar registry for ``scf.if`` results.  Every yielded
/// ``scf.if`` value gets a stable name (``__sc_<N>``) so downstream
/// tasklets can reference it as a scalar rather than walking back into
/// the if-region.  Reset between SDFG builds by ``buildAST`` setup.
inline thread_local int kScfValueCounter = 0;
inline thread_local llvm::DenseMap<mlir::Value, std::string> kScfValueMap;

/// Synthetic-scalar registry for un-named ``fir.alloca`` scratch ops
/// (Flang lowers some ``DO WHILE`` counters as bare allocas without a
/// surrounding ``hlfir.declare``).  Names are ``__al_<N>``.
inline thread_local int kAllocaCounter = 0;
inline thread_local llvm::DenseMap<mlir::Operation *, std::string> kAllocaMap;

/// Map from a libcall-producing ``hlfir`` op (``hlfir.matmul`` /
/// ``hlfir.transpose`` / ``hlfir.dot_product``) inlined inside an
/// ``hlfir.elemental`` body to the synthetic transient name that
/// ``buildElementalAssign`` materialises ahead of the elemental loop.
/// ``buildExpr`` consults this map when handling ``hlfir.apply``.
inline thread_local std::map<mlir::Operation *, std::string> kHlfirExprToTransient;

/// Position-array registry: ``__sym_<arr>_<one_based_idx>`` symbol minted by
/// ``buildIndexExpr`` for each ``arr(constant)`` it sees used as an
/// array index or section bound.
inline thread_local std::map<std::pair<std::string, int64_t>, std::string>
    kPosSymbolRegistry;

/// Synthetic-transient counter used by elemental walks
/// (``__libcall_tmp_<N>``).
inline thread_local int kSynthTransientCounter = 0;

/// Counter for libcall-result transients minted alongside an
/// ``hlfir.assign`` to a section / single element.
inline thread_local int kLibTmpCounter = 0;

// ----- buildBoolExpr context flags -----------------------------------------

/// When true, ``buildBoolExpr``'s cmp branches use bare identifiers for
/// array reads (e.g. ``a > b``) instead of subscripted form (``a[i]
/// > b[i]``).  Set during elemental-body walks where the resulting
/// expression is destined for a tasklet body -- emit_tasklet rewrites
/// bare names into per-occurrence connectors and wires the subscripts
/// via memlets.  Interstate-edge condition contexts (the default) use
/// the subscript form.
inline thread_local bool kBoolExprNoSubscripts = false;

/// RAII guard scoping ``kBoolExprNoSubscripts = true``.  Prefer this
/// over the manual ``bool prev = ...; flag = true; ...; flag = prev;``
/// pattern so the flag is restored on any exit path (early return,
/// thrown exception).  All callers that need bare-name mode use this
/// guard (elemental walks, MERGE-of-scalars in ``buildExpr``'s
/// ``arith.select`` ternary, ``buildExpr``'s i1 ``andi`` / ``ori``
/// chain handler).
struct NoSubscriptGuard {
    bool prev;
    NoSubscriptGuard() : prev(kBoolExprNoSubscripts) { kBoolExprNoSubscripts = true; }
    ~NoSubscriptGuard() { kBoolExprNoSubscripts = prev; }
};

/// When true, suppress the ``dace.float32(...)`` wrap around f32
/// constants and f32->f64 converts.  Set inside ``buildBoolExpr`` /
/// ``buildExprWithSubscripts`` -- the resulting string lands in an
/// interstate-edge condition or ConditionalBlock guard, parsed by
/// DaCe's symbolic engine which treats ``dace.float32`` as a free
/// symbol reference (``dace``) followed by an attribute, blowing up
/// with ``KeyError: 'dace'``.  Inside tasklet bodies the wrap is
/// needed and harmless (Python tasklet parser understands the
/// ``dace`` namespace); inside condition strings it isn't, and the
/// f32-vs-f64 precision difference doesn't change comparison
/// outcomes anyway.
inline thread_local bool kSuppressFloatCast = false;

/// RAII guard scoping ``kSuppressFloatCast = true``.
struct SuppressFloatCastGuard {
    bool prev;
    SuppressFloatCastGuard() : prev(kSuppressFloatCast) { kSuppressFloatCast = true; }
    ~SuppressFloatCastGuard() { kSuppressFloatCast = prev; }
};

// ============================================================================
// Cross-file API
// ============================================================================

// All inline so multiple TUs can include this header without ODR
// violations -- bodies live in the chunk that owns each function.

/// Build a Python-syntax expression string for the SSA value ``val``.
/// Bare-name form (no subscripts) for tasklet contexts.  Depth-limited
/// to prevent runaway recursion on malformed IR.  See ``expressions.cpp``.
std::string buildExpr(mlir::Value val, int d);

/// Render an SSA value as a Fortran-1-based index / loop-bound
/// expression.  Recognises arith add/sub/mul/divsi/divui, MAX / MIN
/// (both ``arith.maxsi``/``minsi`` and the ``cmp+select`` lowering on
/// any predicate), constant-indexed array reads (interned via
/// ``internPosSymbol``), and falls back to ``"?"``.  See
/// ``assigns.cpp``.
std::string buildIndexExpr(mlir::Value v, int d);

/// Like ``buildExpr`` but renders array reads with their full
/// ``arr[idx, ...]`` subscripts.  Used by ``buildBoolExpr`` for
/// interstate-edge condition contexts.  See ``control_flow.cpp``.
std::string buildExprWithSubscripts(mlir::Value val, int d);

/// Render an ``i1`` SSA value as a Python boolean expression.
/// Recognises ``arith.cmpf`` / ``arith.cmpi`` (full predicate table),
/// ``arith.andi`` / ``arith.ori`` / ``arith.xori`` chains on i1,
/// ``fir.is_present`` for OPTIONAL ``present(x)``.  Honours the
/// ``kBoolExprNoSubscripts`` / ``kBoolExprBareForAndi`` flags to pick
/// bare-name vs subscripted operand rendering.  See
/// ``control_flow.cpp``.
std::string buildBoolExpr(mlir::Value val, int d);

/// Render the ``dim``-th index expression of an ``hlfir.designate``,
/// applying section / assumed-shape lower-bound rebases.  See
/// ``expressions.cpp``.
std::string buildDesignateIndexExpr(hlfir::DesignateOp dg,
                                    unsigned dim,
                                    mlir::Value idx,
                                    int depth);

/// Per-dim AccessInfo entry produced by ``expandDesignateChain``.
struct DimEntry {
    std::string var;   // identifier for AccessInfo::index_vars
    std::string expr;  // 1-based expression for AccessInfo::index_exprs
};

/// Walk an innermost ``hlfir.designate``'s parent chain (through
/// inlined declare aliases, ``fir.convert`` reshapes, and section
/// parent designates) and produce a per-original-dim ``(var, expr)``
/// list keyed to the underlying array's full rank.  Used by
/// ``buildElementalAssign`` / ``buildAssignNode`` /
/// ``buildElementalCountLibcall`` / ``buildElementalAnyAllReduce``
/// so the access list always matches the underlying array even when
/// the inner designate is a rank-reduced view.  See ``elementals.cpp``.
std::pair<std::string, std::vector<DimEntry>>
expandDesignateChain(hlfir::DesignateOp innermost);

/// Resolve an SSA index to its source name when we're inside an
/// elemental body (the elemental's block argument is a tracked synth
/// iter).  Empty string when no match.  See ``expressions.cpp``.
std::string resolveIndex(mlir::Value idx);

/// Lower ``fir.is_present %v -> i1`` to a Python expression
/// (``"<name>_present"`` for OPTIONAL dummies, constant ``0`` /
/// ``1`` otherwise).  See ``expressions.cpp``.
std::string lowerIsPresent(mlir::Value operand);

/// Mint or look up the synthetic name for a bare ``fir.alloca``
/// scratch value (no surrounding ``hlfir.declare``).  See
/// ``expressions.cpp``.
std::string allocaSynthName(mlir::Value memref);

/// Intern an ``arr(constant)`` index read as the SDFG symbol
/// ``__sym_<arr>_<one_based_idx>``.  The bridge attaches a
/// ``kind="symbol_init"`` AST node so the Python emitter loads the
/// value on an interstate edge at SDFG entry.  See
/// ``expressions.cpp``.
std::string internPosSymbol(const std::string &array, int64_t one_based_idx);

/// Capture the LHS of an ``hlfir.assign`` whose destination is either
/// a bare ``hlfir.declare`` or an ``hlfir.designate`` selecting one
/// element of an array.  Writes the resolved name into ``node.target``
/// and records per-dim ``AccessInfo``.  See ``expressions.cpp``.
void captureElementDesignateWrite(mlir::Value dest, ASTNode &node);

/// Render an ``arith::cmpi`` / ``cmpf`` predicate as a Python
/// comparison operator.  See ``control_flow.cpp``.
std::string cmpiPredStr(mlir::arith::CmpIPredicate p);
std::string cmpfPredStr(mlir::arith::CmpFPredicate p);

/// Walk a block of HLFIR ops into a list of ``ASTNode``s -- the
/// recursive backbone of AST extraction.  See ``elementals.cpp``.
std::vector<ASTNode> buildAST(mlir::Block &block);

/// Resolve the ``d``-th extent of a ``fir.shape`` / ``fir.shape_shift``
/// to a name (constant literal, declared symbol, or synthetic).  Empty
/// when the shape op isn't recognised.  See ``elementals.cpp``.
std::string resolveExtent(mlir::Value shape, unsigned d);

/// Walk an SSA expression tree and append every ``hlfir.designate``
/// read it touches to ``accesses``.  Used by ``buildLibCallNode`` so
/// the libcall's tasklet picks up every input array.  See
/// ``elementals.cpp``.
void collectReadAccesses(mlir::Value v,
                         std::vector<AccessInfo> &accesses,
                         int depth);

/// Map an ``hlfir.matmul`` / ``hlfir.transpose`` / ``hlfir.dot_product``
/// op to the libcall name DaCe's runtime exposes.  See
/// ``elementals.cpp``.
const char *libcallNameForExprOp(mlir::Operation *op);

/// Render the result-type shape of an ``hlfir.expr<...>`` value as
/// per-dim extent strings.  Empty vector when the type isn't an
/// ``hlfir.expr``.  See ``elementals.cpp``.
std::vector<std::string> exprResultShape(mlir::Type ty);

/// Render the result element-type of an ``hlfir.expr<...>`` value as
/// a numpy-style dtype string.  See ``elementals.cpp``.
std::string exprDtypeString(mlir::Type ty);

/// Push / pop ``(blockArg, syntheticName)`` pairs on the elemental
/// index-substitution stack used by ``resolveIndex``.  See
/// ``expressions.cpp``.
std::vector<std::pair<mlir::Value, std::string>> &indexStack();

/// Peel ``fir.ref<...>`` / ``fir.box<...>`` / ``fir.heap<...>`` /
/// ``fir.ptr<...>`` wrappers off a type.  See ``assigns.cpp``.
mlir::Type peelWrappers(mlir::Type t);

/// True iff the type peels to a ``fir.array<...>`` or is an
/// ``hlfir.expr<...>`` value with a non-empty shape.  See
/// ``assigns.cpp``.
bool isArrayRef(mlir::Type t);

/// True iff the value is a constant 0 (any integer / index width).
/// See ``assigns.cpp``.
bool isConstantZero(mlir::Value v);

/// Trace a ``fir.do_loop`` lower-bound SSA value to a constant int.
/// Returns -1 when not a constant.  See ``assigns.cpp``.
int64_t traceLB(mlir::Value v);

/// Walk through ``fir.convert`` to find the underlying section
/// designate (``hlfir.designate`` with triplet indices).  See
/// ``assigns.cpp``.
hlfir::DesignateOp asSectionDesignate(mlir::Value v);

}  // namespace hlfir_bridge
