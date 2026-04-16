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
#include <llvm/ADT/SmallVector.h>
#include <optional>
#include <string>

namespace hlfir_bridge {

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
std::string traceToDecl(mlir::Value val, int max = 200);

/// Trace an SSA value to a compile-time integer constant through
/// any number of fir.convert wrappings.  nullopt if not constant-foldable.
std::optional<int64_t> traceConstInt(mlir::Value v);

/// Extract extent SSA values from a fir.shape or fir.shape_shift.
/// Returns empty if the operand is neither (or is null).
llvm::SmallVector<mlir::Value, 4> extractExtents(mlir::Value shape);

}  // namespace hlfir_bridge
