// ============================================================================
// DefaultIntent.cpp  --  Fill in ``intent_inout`` for dummies that lack an
// intent.
// ============================================================================
// Problem:
//     Fortran lets a subroutine declare a dummy argument without any INTENT
//     attribute; the standard treats it as effectively ``intent(inout)``
//     for callers.  Flang emits such a dummy as
//         hlfir.declare %arg0 dummy_scope %0 {uniq_name = ...} : ...
//     with no ``fortran_attrs`` attribute at all.  Downstream consumers
//     (our extract_vars, code that chooses between transient/non-transient
//     SDFG descriptors, ...) can't tell the difference between "a local"
//     and "a dummy without explicit intent" from the attribute alone.
//
// What this pass does:
//     For every func.func in the module, walk its entry-block arguments,
//     find the hlfir.declare that uses each arg, and if that declare's
//     fortran_attrs has no intent bit set, OR the attribute is absent
//     entirely, set it to include ``intent_inout``.
//
// Why it's safe:
//     The standard requires a dummy to be treated like intent_inout when
//     no intent is given, so canonicalising the IR to say so doesn't
//     change semantics  --  it just makes the subsequent passes / extract
//     code trivially correct.
// ============================================================================

#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

struct DefaultIntentPass
    : public mlir::PassWrapper<DefaultIntentPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DefaultIntentPass)

  llvm::StringRef getArgument() const final { return "hlfir-default-intent"; }
  llvm::StringRef getDescription() const final {
    return "Stamp intent_inout onto dummy-argument hlfir.declare ops "
           "that lack an explicit intent, mirroring Fortran's default.";
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    using Flags = fir::FortranVariableFlagsEnum;
    auto hasIntent = [](Flags f) {
      return bitEnumContainsAny(f, Flags::intent_in) ||
             bitEnumContainsAny(f, Flags::intent_out) ||
             bitEnumContainsAny(f, Flags::intent_inout);
    };

    getOperation().walk([&](mlir::func::FuncOp func) {
      if (func.isExternal()) return;
      auto &block = func.front();
      for (auto arg : block.getArguments()) {
        hlfir::DeclareOp decl;
        for (auto *u : arg.getUsers())
          if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
            decl = d;
            break;
          }
        if (!decl) continue;

        auto current = decl.getFortranAttrs();
        Flags flags = current ? *current : Flags::None;
        if (hasIntent(flags)) continue;

        flags = flags | Flags::intent_inout;
        decl->setAttr("fortran_attrs",
                      fir::FortranVariableFlagsAttr::get(ctx, flags));
      }
    });
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createDefaultIntentPass() {
  return std::make_unique<DefaultIntentPass>();
}

}  // namespace hlfir_bridge
