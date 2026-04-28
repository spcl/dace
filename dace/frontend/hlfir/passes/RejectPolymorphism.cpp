// ============================================================================
// RejectPolymorphism.cpp — refuse programs that exercise runtime polymorphism.
// ============================================================================
//
// What it rejects:
//   * ``fir.dispatch``    — virtual method call ``obj%method()`` whose
//                           target depends on the runtime type of
//                           ``obj``.
//   * ``fir.select_type`` — ``SELECT TYPE`` block dispatching on
//                           runtime type.
//
// Why we don't try to lower these:
//   Flang ships a ``fir-polymorphic-op`` conversion pass that
//   statically devirtualises some cases.  Linking it pulls in
//   ``libFIRTransforms.a`` + ``libFIRCodeGenDialect.a`` +
//   ``libFIRCodeGen.a`` + ``libFIROpenACCSupport.a`` +
//   ``libFortranSupport.a`` — the LLVMTypeConverter machinery and a
//   transitive OpenACC dialect-registration hook.  Several MB and a
//   sprawling dependency for a feature the bridge doesn't actually
//   want: even with the lowering, surviving dispatches are still
//   unsupportable at the SDFG level.
//
//   Instead we walk for these ops directly and call ``op.emitError``,
//   which makes ``run_passes`` return failure that the Python side
//   raises as ``RuntimeError("run_passes: pipeline failed")``.  The
//   error message points at the source location of the dispatch.
//
// What's still supported:
//   * Non-polymorphic ``CLASS(t)`` declares — ``hlfir.designate
//     %obj{"field"}`` style member access without virtual dispatch.
//     Flang represents these as ``fir.class<T>`` type wrappers; the
//     bridge's ``BaseBoxType`` peel in ``FlattenStructs`` walks
//     through them like ``fir.box<T>``.
//   * Type extensions used purely for static dispatch (``call
//     concrete_proc(obj)`` rather than ``obj%method()``).
// ============================================================================

#include "passes/Passes.h"

#include "flang/Optimizer/Dialect/FIROps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace hlfir_bridge {

namespace {

struct RejectPolymorphismPass
    : public mlir::PassWrapper<RejectPolymorphismPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RejectPolymorphismPass)

    llvm::StringRef getArgument() const final {
        return "hlfir-reject-polymorphism";
    }
    llvm::StringRef getDescription() const final {
        return "Refuse programs that contain ``fir.dispatch`` or "
               "``fir.select_type`` (runtime polymorphism).  The HLFIR "
               "bridge supports CLASS-as-monomorphic-box only.";
    }

    void runOnOperation() override {
        bool failed = false;
        getOperation().walk([&](mlir::Operation *op) {
            if (auto d = mlir::dyn_cast<fir::DispatchOp>(op)) {
                d.emitError(
                    "hlfir-reject-polymorphism: ``fir.dispatch`` op "
                    "(virtual method call ``obj%method()``).  The "
                    "HLFIR bridge does not lower runtime polymorphic "
                    "dispatch — every call site must resolve to a "
                    "concrete subroutine at compile time.  Replace "
                    "``call obj%method(...)`` with a direct call "
                    "``call concrete_method(obj, ...)`` or restructure "
                    "to avoid CLASS dispatch.");
                failed = true;
                return;
            }
            if (auto s = mlir::dyn_cast<fir::SelectTypeOp>(op)) {
                s.emitError(
                    "hlfir-reject-polymorphism: ``fir.select_type`` op "
                    "(SELECT TYPE block).  Runtime type discrimination "
                    "is not lowered.  If the dispatch target is "
                    "statically determinable, restructure the source "
                    "to a static call chain.");
                failed = true;
                return;
            }
            // ``fir-polymorphic-op`` rewrites unresolvable ``fir.dispatch``
            // ops into a type-descriptor walk (``fir.box_tdesc`` + a
            // chain of coordinate_of / load) that loads the binding's
            // proc address and ends in an indirect ``fir.call`` through
            // that pointer.  Catch the lowered shape here too: any
            // surviving ``fir.box_tdesc`` is a marker that the program
            // is still doing runtime dispatch.
            if (auto t = mlir::dyn_cast<fir::BoxTypeDescOp>(op)) {
                t.emitError(
                    "hlfir-reject-polymorphism: ``fir.box_tdesc`` op "
                    "(runtime type-info read for an unresolvable "
                    "polymorphic dispatch).  ``fir-polymorphic-op`` "
                    "lowers truly virtual ``fir.dispatch`` to an "
                    "indirect call through the type-info table; the "
                    "HLFIR bridge does not lower this.  Either resolve "
                    "the dispatch statically (concrete receiver type) "
                    "or restructure to a direct call.");
                failed = true;
                return;
            }
        });
        if (failed) signalPassFailure();
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> createRejectPolymorphismPass() {
    return std::make_unique<RejectPolymorphismPass>();
}

}  // namespace hlfir_bridge
