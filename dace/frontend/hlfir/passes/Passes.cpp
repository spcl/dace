// ============================================================================
// Passes.cpp — Registration dispatch for every bridge pass.
// ============================================================================

#include "passes/Passes.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace hlfir_bridge {

void registerAllBridgePasses() {
    static bool registered = false;
    if (registered) return;
    registered = true;

    // Upstream MLIR conversion / transform passes we run as part of the
    // default bridge pipeline.  The generated headers expose factories but
    // not registry calls, so we wrap them ourselves.
    mlir::registerPass([]() { return mlir::createLiftControlFlowToSCFPass(); });
    mlir::registerPass([]() { return mlir::createSCCPPass(); });
    mlir::registerPass([]() { return mlir::createCanonicalizerPass(); });
    mlir::registerPass([]() { return mlir::createCSEPass(); });
    mlir::registerPass([]() { return mlir::createSymbolDCEPass(); });
    // Upstream interprocedural inliner — relies on FIRInlinerInterface
    // registered by the Flang dialect to know how to inline fir.call.
    mlir::registerPass([]() { return mlir::createInlinerPass(); });

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
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return createVerifyNoUnresolvedCallsPass();
    });
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return createFoldElementAliasesPass();
    });
}

}  // namespace hlfir_bridge
