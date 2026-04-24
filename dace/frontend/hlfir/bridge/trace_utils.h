// ============================================================================
// trace_utils.h — Shared SSA tracing utilities
// ============================================================================
// Walks backward through the def-use chains that Flang emits
// (fir.convert, fir.load, arith.select, hlfir.declare) to recover
// Fortran names and constant values from opaque SSA values.
//
// Used by both the bridge extraction code and by MLIR passes.
// ============================================================================

#pragma once

#include "mlir/IR/Value.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include <llvm/ADT/SmallVector.h>
#include <optional>
#include <string>

namespace hlfir_bridge {

/// Recursion and walk-length budgets for the bridge's SSA tracing and
/// expression reconstruction.  All are defensive guards against
/// pathological IR shapes (malformed input, cyclic defs, runaway
/// inlining) — bumping them never changes semantics on well-formed
/// HLFIR, only reduces false-``?`` fallbacks on deep chains.
namespace limits {

/// Maximum recursion depth for ``buildExpr`` / ``buildExprWithSubscripts``
/// / ``buildBoolExpr``.  Arithmetic trees from composed ``hlfir.elemental``
/// + ``hlfir.apply`` chains fused by ``hlfir-flatten-structs`` can run
/// 40+ deep on real ICON kernels.
inline constexpr int kBuildExprDepth = 128;

/// Maximum recursion depth for ``buildIndexExpr``.  Index expressions
/// stay shallower than general expressions (one index operand per
/// designate dim, narrower op set), but inherit the same budget.
inline constexpr int kBuildIndexExprDepth = 64;

/// Maximum ``fir.convert`` chain length while walking a single SSA
/// value inside ``resolveIndex``.  Flang occasionally stacks several
/// converts for index/integer kind coercions.
inline constexpr int kConvertChainDepth = 32;

/// Maximum wrapper-peel depth for type unwrapping (``fir.box``,
/// ``fir.ref``, ``fir.heap``, ``fir.pointer``).  Nested-pointer
/// chains are rare but legal: ``box<ref<heap<array<...>>>>``.
inline constexpr int kTypeWrapperPeelDepth = 32;

/// Default walk budget for ``traceToDecl`` / ``traceConstInt``.  These
/// are long-running walks through fir.convert / fir.load / designate
/// chains back to the originating declare or constant.
inline constexpr int kTraceToDeclMax = 1024;
inline constexpr int kTraceConstIntMax = 64;

/// Maximum memref-walk depth inside ``asAssumedShapeAlias`` (peels
/// fir.convert ops between the alias declare and the outer declare).
inline constexpr int kAliasMemrefWalkDepth = 32;

}  // namespace limits

/// Name of the shape-hint attribute that PropagateShapes attaches to
/// assumed-shape dummy declares.  Value is an ArrayAttr of StringAttrs,
/// one per dimension (empty string if that dim disagreed across callers).
inline constexpr const char *kShapeHintAttr = "hlfir_bridge.shape_hint";

/// Extract the short Fortran name from Flang's mangled unique name.
///   "_QFcompute_z_v_grad_wEnproma" → "nproma"
std::string extractName(const std::string &mangled);

/// Trace an SSA value backwards to the hlfir.declare / fir.declare that
/// introduced it.  Peels fir.convert → fir.load → arith.select transparently.
/// Returns the Fortran name, or "" if the chain breaks before a declare.
std::string traceToDecl(mlir::Value val, int max = limits::kTraceToDeclMax);

/// Trace an SSA value to a compile-time integer constant through
/// any number of fir.convert wrappings.  nullopt if not constant-foldable.
std::optional<int64_t> traceConstInt(mlir::Value v);

/// Extract extent SSA values from a fir.shape or fir.shape_shift.
/// Returns empty if the operand is neither (or is null).
llvm::SmallVector<mlir::Value, 4> extractExtents(mlir::Value shape);

/// When ``hlfir-inline-all`` splices an assumed-shape callee into its
/// caller, the callee's ``hlfir.declare %arg0`` becomes a second
/// declare that aliases the caller's actual argument with its own
/// (default-1-based) lower bound.  The signature to look for:
///   * no shape operand on ``decl`` (assumed-shape);
///   * ``decl.getMemref()`` comes from a ``fir.convert`` (extent-erasing
///     rebox) whose operand traces back to another ``hlfir.declare``.
/// Returns the outer declare if this pattern fires, else a null
/// handle.  Callers use the return to (a) skip registering the alias
/// as its own SDFG data container, (b) walk index expressions through
/// the inner (callee) frame to the outer (caller) frame.
hlfir::DeclareOp asAssumedShapeAlias(hlfir::DeclareOp decl);

/// Per-dimension lower-bound constants for an ``hlfir.declare``.
/// Returns the constants stored in a ``fir.shape_shift`` operand, or
/// a vector of ``1``s (Fortran default) of length ``rank`` when the
/// declare has no shape / uses a plain ``fir.shape``.  Any dim whose
/// lower bound isn't a compile-time constant comes back as
/// ``std::nullopt``.
std::vector<std::optional<int64_t>> declareLowerBounds(hlfir::DeclareOp decl);

}  // namespace hlfir_bridge
