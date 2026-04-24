// ============================================================================
// InlineAll.cpp — Aggressive whole-program inlining.
// ============================================================================
// Problem:
//     For deployment-time specialisation (namelist constant injection +
//     SCCP), we need interprocedural constant propagation across call
//     boundaries.  Rather than building a full context-sensitive SCCP,
//     we flatten the call tree: inline every callee into its caller
//     until only external/intrinsic declarations remain.
//
// Approach:
//     Bottom-up fixed-point iteration over the module.  Each sweep walks
//     all fir.call ops; if the callee has a body, inline it.  Repeat
//     until no inlining occurs (all remaining calls are external).
//
// Assumptions:
//     - No recursive functions.  The pass does not detect cycles; if
//       recursion exists, it will hit the iteration cap and bail out.
//     - Code size explosion is acceptable — the result is meant for
//       specialisation, not direct compilation.
//
// After this pass, the entry function(s) contain the full flattened
// program body, and dead callees can be removed with --symbol-dce.
// ============================================================================

#include "passes/Passes.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Interfaces/CallInterfaces.h"

#define DEBUG_TYPE "inline-all"

namespace hlfir_bridge {

namespace {

// ---------------------------------------------------------------------------
// Inliner interface — accept everything.
// ---------------------------------------------------------------------------

/// Permissive inliner interface that allows inlining any callable into any
/// call site.  We override the legality hooks to always return true, and
/// defer the transformation hooks (``handleTerminator`` / ``handleArgument``
/// / ``handleResult``) to the per-dialect ``DialectInlinerInterface`` that
/// Flang registers for FIR and the core dialects — overriding them here
/// would short-circuit the correct per-op behaviour and can corrupt the IR.
struct AggressiveInlinerInterface : public mlir::InlinerInterface {
    using mlir::InlinerInterface::InlinerInterface;

    bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                         bool wouldBeCloned) const final {
        return true;
    }
    bool isLegalToInline(mlir::Region *dest, mlir::Region *src,
                         bool wouldBeCloned,
                         mlir::IRMapping &valueMapping) const final {
        return true;
    }
    bool isLegalToInline(mlir::Operation *op, mlir::Region *dest,
                         bool wouldBeCloned,
                         mlir::IRMapping &valueMapping) const final {
        return true;
    }
};

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

struct InlineAllPass
    : public mlir::PassWrapper<InlineAllPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlineAllPass)

    llvm::StringRef getArgument() const final {
        return "hlfir-inline-all";
    }
    llvm::StringRef getDescription() const final {
        return "Aggressively inline all non-external callees to produce "
               "a flat, single-function representation.";
    }

    /// One sweep over the module.  Returns the number of call sites inlined.
    unsigned sweep(mlir::ModuleOp module, mlir::SymbolTable &symTab) {
        unsigned inlined = 0;
        AggressiveInlinerInterface interface(module.getContext());

        // Collect call ops first — we'll be mutating the IR during inlining,
        // so we cannot walk and inline simultaneously.
        llvm::SmallVector<fir::CallOp, 64> calls;
        module.walk([&](fir::CallOp call) {
            calls.push_back(call);
        });

        for (auto call : calls) {
            // Skip if the op was erased by a previous inlining in this sweep.
            if (!call->getParentOp()) continue;

            auto sym = call.getCallee();
            if (!sym) continue;  // indirect call

            auto callee = symTab.lookup<mlir::func::FuncOp>(
                sym->getLeafReference());
            if (!callee || callee.isDeclaration()) continue;  // external

            LLVM_DEBUG(llvm::dbgs()
                       << "InlineAll: inlining " << callee.getSymName()
                       << " into "
                       << call->getParentOfType<mlir::func::FuncOp>()
                              .getSymName()
                       << "\n");

            // Perform the inlining.
            auto callIface = mlir::dyn_cast<mlir::CallOpInterface>(call.getOperation());
            auto callableIface = mlir::dyn_cast<mlir::CallableOpInterface>(callee.getOperation());
            if (!callIface || !callableIface) continue;

            // Clone callback for inlineCall: insert cloned blocks BEFORE
            // ``postInsertBlock`` so the layout becomes
            //     [inlineBlock (inlined-into), cloned..., postInsertBlock].
            // Inserting at ``inlineBlock`` instead demotes the caller's
            // original entry block and drops its block-argument list,
            // which then trips func.func's signature verifier.
            auto cloneCallback = [](mlir::OpBuilder &builder, mlir::Region *src,
                                    mlir::Block *inlineBlock, mlir::Block *postBlock,
                                    mlir::IRMapping &mapper, bool shouldClone) {
                if (shouldClone) {
                    src->cloneInto(inlineBlock->getParent(),
                                   postBlock->getIterator(), mapper);
                } else {
                    src->getBlocks().splice(postBlock->getIterator(),
                                            src->getBlocks());
                }
            };

            auto result = mlir::inlineCall(
                interface,
                cloneCallback,
                callIface,
                callableIface,
                &callee.getBody(),
                /*shouldCloneInlinedRegion=*/true);

            if (mlir::succeeded(result)) {
                // The call op is replaced by the inlined body.
                call->erase();
                ++inlined;
            } else {
                LLVM_DEBUG(llvm::dbgs()
                           << "InlineAll: FAILED to inline "
                           << callee.getSymName() << "\n");
            }
        }

        return inlined;
    }

    void runOnOperation() override {
        auto module = getOperation();
        mlir::SymbolTable symTab(module);

        // Fixed-point iteration.  Each round inlines one level of calls;
        // repeat until no more inlining is possible.
        // Cap at 128 rounds — without recursion this is the max call-tree
        // depth; in practice convergence is much faster.
        unsigned totalInlined = 0;
        for (int round = 0; round < 128; ++round) {
            unsigned n = sweep(module, symTab);
            if (n == 0) break;
            totalInlined += n;

            LLVM_DEBUG(llvm::dbgs()
                       << "InlineAll: round " << round
                       << " inlined " << n << " call sites\n");
        }

        LLVM_DEBUG(llvm::dbgs()
                   << "InlineAll: total " << totalInlined
                   << " call sites inlined\n");
    }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createInlineAllPass() {
    return std::make_unique<InlineAllPass>();
}

}  // namespace hlfir_bridge
