// ============================================================================
// Passes.h — Public API for the hlfir_bridge pass library.
// ============================================================================
// Each pass gets a constructor function here and a registration line in
// Passes.cpp's registerAllBridgePasses().  That single registration call
// wires every pass into MLIR's global registry so they become available
// both through the Python run_passes() entry point and through the
// standalone hlfir-bridge-opt tool.
// ============================================================================

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace hlfir_bridge {

// --- Individual pass constructors ---

/// PropagateShapes: for every fir.call site, find the actual arguments'
/// shape operands and stamp hlfir_bridge.shape_hint on the callee's
/// assumed-shape dummy hlfir.declare ops.  Runs to a fixed point to
/// handle transitive propagation through chains of calls.
std::unique_ptr<mlir::Pass> createPropagateShapesPass();

// --- Registry ---

/// Register every bridge pass with MLIR's global pass registry.
/// Call exactly once per process; safe to call again (idempotent).
void registerAllBridgePasses();

}  // namespace hlfir_bridge
