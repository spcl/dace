// ============================================================================
// Passes.cpp — Registration dispatch for every bridge pass.
// ============================================================================

#include "passes/Passes.h"
#include "mlir/Pass/PassRegistry.h"

namespace hlfir_bridge {

void registerAllBridgePasses() {
    static bool registered = false;
    if (registered) return;
    registered = true;

    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return createPropagateShapesPass();
    });
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return createInlineAllPass();
    });
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return createFlattenStructsPass();
    });
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return createDefaultIntentPass();
    });
}

}  // namespace hlfir_bridge
