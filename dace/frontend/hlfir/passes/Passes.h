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
std::unique_ptr<mlir::Pass> createPropagateShapesPass();
std::unique_ptr<mlir::Pass> createInlineAllPass();
std::unique_ptr<mlir::Pass> createFlattenStructsPass();
std::unique_ptr<mlir::Pass> createDefaultIntentPass();

// --- Registry ---

/// Register every bridge pass with MLIR's global pass registry.
void registerAllBridgePasses();

}  // namespace hlfir_bridge
