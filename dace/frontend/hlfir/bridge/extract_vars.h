// ============================================================================
// extract_vars.h  --  Collect and classify every hlfir.declare in a module.
// ============================================================================

#pragma once

#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"

namespace hlfir_bridge {

/// One per hlfir.declare.  Describes a Fortran variable.
///
///   fortran_name    --  short Fortran name, e.g. "nproma"
///   mangled_name    --  Flang unique name, e.g. "_QFcompute_z_v_grad_wEnproma"
///   intent          --  "in" | "out" | "inout" | "" (local)
///   dtype           --  "float64" | "float32" | "int32" | "int64" | raw type
///   rank            --  number of array dimensions (0 for scalars)
///   is_dynamic      --  true if any dim is ? (unknown extent)
///   shape_symbols   --  per-dim extent name.  Resolution order:
///                      1. hlfir_bridge.shape_hint attribute (from passes)
///                      2. fir.shape / fir.shape_shift operand
///                      3. synthetic "<var>_d<i>" for assumed-shape (:,:)
///   lower_bounds    --  per-dim Fortran lower bound as string
///   role            --  "array" | "symbol" | "scalar"
struct VarInfo {
  std::string fortran_name, mangled_name, intent, dtype;
  int rank = 0;
  bool is_dynamic = false;
  std::vector<std::string> shape_symbols;
  std::vector<std::string> lower_bounds;
  std::string role;
  /// Compile-time constant data for the read-only constant pool
  /// (Flang's ``_QQro.<shape>x<dtype>.<counter>`` globals).  When
  /// non-empty the SDFG builder synthesises an init state writing
  /// these values into the transient before the kernel body runs.
  /// Empty for ordinary variables.  Value layout: row-major doubles
  /// (one per element)  --  the Python side narrows to the actual
  /// dtype on use.  Booleans surface as 0.0 / 1.0.
  std::vector<double> const_data;
  /// For ``role == "view_alias"`` only.  ``view_source`` is the
  /// underlying array's Fortran name; ``view_subset`` is one entry
  /// per source-array dim in 0-based DaCe form  --  ``"0:4"`` for a
  /// full range, ``"2"`` for a fixed scalar.  The alias surface is
  /// a (possibly rank-changed) re-interpretation of ``view_source``
  /// over the section indicated by ``view_subset``.  ``descriptors``
  /// uses this to stage a copy-in at SDFG entry and a copy-out at
  /// SDFG exit so writes propagate back through the alias.  Set
  /// when Flang emits ``hlfir.declare %converted`` where the
  /// memref ultimately threads through a ``fir.convert`` that
  /// re-shapes a section designate's element type to a different
  /// array shape (Fortran storage-association reshape).
  std::string view_source;
  std::vector<std::string> view_subset;
  /// For ``role == "section_alias"`` only.  One entry per source-array
  /// dim; surviving dims are placeholders ``"_d<N>"`` (N = 0-based
  /// dummy-dim index), dropped scalar dims hold a 0-based DaCe-form
  /// index expression (``"(k)-1"`` for symbolic, ``"<int>"`` for
  /// constant).  The Python builder splices the inlined-body's
  /// dummy index_exprs into the placeholders to produce a full
  /// source-array memlet  --  no separate SDFG view is registered.
  /// Set only when the section is structurally trivial (every triplet
  /// has lo=1, stride=1), so the alias is just a name + index suffix.
  /// Non-trivial sections (strided / sub-range) stay on the
  /// ``view_alias`` path.
  std::vector<std::string> view_dim_map;
};

/// Walk the module and build one VarInfo per hlfir.declare.
std::vector<VarInfo> extractVariables(mlir::ModuleOp module);

/// True iff the allocatable / pointer ``declName`` needs the
/// per-variable ``<declName>_allocated`` int32 tracker scalar  --  i.e.
/// either the kernel body writes it (an ALLOCATE / DEALLOCATE site
/// exists) or reads it (an ``ALLOCATED(arr)`` / ``ASSOCIATED(ptr)``
/// reader exists, lowered to ``fir.box_addr``).  Dummies passed in
/// already-allocated and never queried by ``ALLOCATED(...)`` skip the
/// tracker entirely.
bool needsAllocatedTracker(const std::string &declName, mlir::ModuleOp module);

/// Per-site name for an allocatable ``ALLOCATE``.  Site 0 keeps the
/// original Fortran name (``x``); site 1+ mints synthetic transient
/// names (``x_alloc1``, ``x_alloc2``, ...).  Shared between
/// ``extractVariables`` (which registers the synthetic VarInfos) and
/// ``extractAST`` (which keeps the trace-utils alias map in sync as
/// it walks the IR).
std::string allocAliasName(const std::string &fortran, unsigned site);

}  // namespace hlfir_bridge
