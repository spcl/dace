// ============================================================================
// FoldElementAliases.cpp — erase declare-over-designate aliases.
// ============================================================================
//
// Motivation:
//     After ``hlfir-inline-all`` splices an elemental (or any scalar-arg
//     procedure) into a caller whose actual arguments are array elements,
//     Flang leaves an ``hlfir.declare`` per callee dummy whose memref is
//     the outer array's ``hlfir.designate`` at the current loop index:
//
//         %e = hlfir.designate %outer (%i) : ref<array<Nxf64>>, index -> ref<f64>
//         %d:2 = hlfir.declare %e {uniq_name = "_QMmodFcalleeEx"} : ...
//         hlfir.assign %val to %d#0 : f64, ref<f64>
//
//     The declare exists only to carry the callee's Fortran name into the
//     inlined body — it aliases one element of the outer array and adds
//     no semantics the caller didn't already have.
//
//     For the SDFG builder this second declare is active poison:
//       * ``extract_vars`` sees a second VarInfo with the callee's name
//         over a scalar ref, and the SDFG arglist grows a stray ``x`` /
//         ``od`` / ``g`` / … scalar that the caller never needs to supply.
//       * The write ``hlfir.assign %val to %d#0`` looks like a plain
//         scalar-store, so our frontend never connects the write back
//         to the outer array at the outer's loop index.
//
// What the pass does:
//     For each ``hlfir.declare`` with no shape operand and whose memref
//     is ``hlfir.designate`` of *another* declare, replace every use of
//     the alias's results with the designate itself and erase the
//     alias.  After this, ``hlfir.assign %val to %d#0`` collapses to
//     ``hlfir.assign %val to %designate`` — a regular indexed store the
//     SDFG builder already handles.
//
// Scope:
//     Only element-alias declares are folded.  Assumed-shape aliases
//     (memref via ``fir.convert`` of a box) are left to the bridge's
//     runtime rebase path (``trace_utils::asAssumedShapeAlias`` +
//     ``extract_ast::buildDesignateIndexExpr``) — their semantics
//     require an index rewrite, not a straight replacement.
// ============================================================================

#include "passes/Passes.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

namespace hlfir_bridge {

namespace {

struct FoldElementAliasesPass
    : public mlir::PassWrapper<FoldElementAliasesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldElementAliasesPass)

    llvm::StringRef getArgument() const final {
        return "hlfir-fold-element-aliases";
    }
    llvm::StringRef getDescription() const final {
        return "Erase hlfir.declare ops whose memref is an hlfir.designate of "
               "another declare (element-scoped aliases left behind by "
               "hlfir-inline-all on elemental / scalar-arg calls).";
    }

    void runOnOperation() override {
        llvm::SmallVector<hlfir::DeclareOp, 16> toErase;

        getOperation().walk([&](hlfir::DeclareOp decl) {
            if (decl.getShape()) return;  // has shape — not an alias
            auto memref = decl.getMemref();
            auto *mrd = memref.getDefiningOp();
            if (!mrd) return;
            auto designate = mlir::dyn_cast<hlfir::DesignateOp>(mrd);
            if (!designate) return;

            // Confirm the designate's base is another declare — without
            // that anchor we don't have a "real" outer storage to point
            // uses at.  (Designates nested inside designates would be
            // handled transitively as each outer round erases another
            // layer.)
            auto *base = designate.getMemref().getDefiningOp();
            if (!base) return;
            if (!mlir::isa<hlfir::DeclareOp>(base)) return;

            decl.getResult(0).replaceAllUsesWith(memref);
            decl.getResult(1).replaceAllUsesWith(memref);
            toErase.push_back(decl);
        });

        for (auto d : toErase)
            d.erase();
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> createFoldElementAliasesPass() {
    return std::make_unique<FoldElementAliasesPass>();
}

}  // namespace hlfir_bridge
