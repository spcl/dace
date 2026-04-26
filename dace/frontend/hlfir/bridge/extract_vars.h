// ============================================================================
// extract_vars.h — Collect and classify every hlfir.declare in a module.
// ============================================================================

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include <string>
#include <vector>

namespace hlfir_bridge {

/// One per hlfir.declare.  Describes a Fortran variable.
///
///   fortran_name   — short Fortran name, e.g. "nproma"
///   mangled_name   — Flang unique name, e.g. "_QFcompute_z_v_grad_wEnproma"
///   intent         — "in" | "out" | "inout" | "" (local)
///   dtype          — "float64" | "float32" | "int32" | "int64" | raw type
///   rank           — number of array dimensions (0 for scalars)
///   is_dynamic     — true if any dim is ? (unknown extent)
///   shape_symbols  — per-dim extent name.  Resolution order:
///                      1. hlfir_bridge.shape_hint attribute (from passes)
///                      2. fir.shape / fir.shape_shift operand
///                      3. synthetic "<var>_d<i>" for assumed-shape (:,:)
///   lower_bounds   — per-dim Fortran lower bound as string
///   role           — "array" | "symbol" | "scalar"
struct VarInfo {
    std::string fortran_name, mangled_name, intent, dtype;
    int rank = 0;
    bool is_dynamic = false;
    std::vector<std::string> shape_symbols;
    std::vector<std::string> lower_bounds;
    std::string role;
};

/// Walk the module and build one VarInfo per hlfir.declare.
std::vector<VarInfo> extractVariables(mlir::ModuleOp module);

/// Per-site name for an allocatable ``ALLOCATE``.  Site 0 keeps the
/// original Fortran name (``x``); site 1+ mints synthetic transient
/// names (``x_alloc1``, ``x_alloc2``, …).  Shared between
/// ``extractVariables`` (which registers the synthetic VarInfos) and
/// ``extractAST`` (which keeps the trace-utils alias map in sync as
/// it walks the IR).
std::string allocAliasName(const std::string &fortran, unsigned site);

}  // namespace hlfir_bridge
