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
#include <set>
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
///
/// May consult a thread-local override map populated by ``extract_vars``
/// for inlined-callee dummy-arg declares whose default short name would
/// collide with a caller-scope declare (``_QFmainEinp`` vs
/// ``_QFinner_loopsEinp`` both → ``inp``).  Without the override the
/// view-alias edge that links the inlined dummy back to the caller's
/// storage self-loops.
std::string extractName(const std::string &mangled);

/// Register ``mangled → shortName`` so subsequent ``extractName`` calls
/// for that exact mangled name return ``shortName`` instead of the
/// default ``E``-stripped tail.  Used by ``extract_vars`` to break
/// short-name collisions between a caller declare and an inlined-
/// callee dummy declare that aliases the caller's storage.  Per
/// thread.
void setManglingOverride(const std::string &mangled, const std::string &shortName);

/// Drop every mangling-override binding.  Called at the start of each
/// ``extractVariables`` / ``extractAST`` invocation so a previous
/// module's overrides don't leak into the next one.
void clearManglingOverrides();

/// Trace an SSA value backwards to the hlfir.declare / fir.declare that
/// introduced it.  Peels fir.convert → fir.load → arith.select transparently.
/// Returns the Fortran name, or "" if the chain breaks before a declare.
///
/// For Fortran ``ALLOCATABLE`` variables that get re-allocated, every
/// ALLOCATE site materialises a fresh SDFG transient.  The bridge keeps
/// a thread-local "current alias" map (see ``allocAliasFor`` /
/// ``setAllocAlias``) that maps the raw Fortran name (``x``) to the
/// active alias (``x``, ``x_alloc1``, ``x_alloc2``, …) at the current
/// IR position.  ``traceToDecl`` consults this map and returns the
/// alias when set — every read / write site downstream then routes to
/// the correct per-allocation transient without further wiring.
std::string traceToDecl(mlir::Value val, int max = limits::kTraceToDeclMax);

/// Look up the active alias for a raw allocatable name (the unaliased
/// Fortran name returned by walking the declare chain).  Returns the
/// raw name unchanged if no alias is set, the alias otherwise.
std::string allocAliasFor(const std::string &raw);

/// Bind ``raw`` (the Fortran allocatable's base name) to ``alias`` for
/// subsequent ``traceToDecl`` calls.  ``alias == raw`` resets the alias.
/// Per thread.
void setAllocAlias(const std::string &raw, const std::string &alias);

/// Drop every alloc-alias binding (used at module-walk start so each
/// extractAST call sees a clean state).
void clearAllocAliases();

/// Trace an SSA value to a compile-time integer constant through
/// any number of fir.convert wrappings.  nullopt if not constant-foldable.
std::optional<int64_t> traceConstInt(mlir::Value v);

/// Render an integer-typed SSA value as a Python expression string of
/// Fortran scalar names + literals + arithmetic operators.  Used by
/// ``resolveShapeSyms`` to surface a dynamic gather-temp extent
/// (``arith.select(cmpi sgt, addi(subi(load_ub, load_lb), 1), 0)``
/// — Flang's clamped "ub - lb + 1") as e.g. ``"max((endcol - startcol)
/// + 1, 0)"`` for use as a symbolic SDFG-array shape dim.  The leaf
/// scalar names must be promoted to SDFG symbols separately (via
/// ``symbolNames`` in extract_vars) for the resulting expression to
/// resolve.
///
/// Returns the empty string on any pattern the helper doesn't
/// recognise, so callers fall back to their existing ``"?"`` →
/// synthetic-symbol path.
std::string traceExtentExpr(mlir::Value v);

/// Walk the same SSA chain ``traceExtentExpr`` recognises and append
/// every leaf scalar-declare name encountered to ``out``.  Used by
/// extract_vars Pass 2 to promote the scalars (``startcol``,
/// ``endcol``, …) that appear in a gather-temp's shape extent
/// expression -- without this, the leaves stay as length-1 Array
/// scalars and the expression string from ``traceExtentExpr``
/// references undeclared SDFG names.
void collectExtentExprScalars(mlir::Value v, std::set<std::string> &out);

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
