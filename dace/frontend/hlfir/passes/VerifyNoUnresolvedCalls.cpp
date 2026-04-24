// ============================================================================
// VerifyNoUnresolvedCalls.cpp — fail fast on missing HLFIR inputs.
// ============================================================================
// Motivation:
//     The multi-file driver parses several ``.hlfir`` files, merges them,
//     and runs ``hlfir-inline-all`` to collapse cross-file call trees into
//     a single flat body.  If the caller forgot to pass one of the
//     dependencies, inlining silently leaves a fir.call to an external
//     declaration — the SDFG builder then produces something wrong
//     (or fails deep in extraction).
//
//     This pass runs right after ``hlfir-inline-all`` and errors out if
//     any fir.call still points at an external declaration that is not
//     on the Flang-runtime / intrinsic allowlist.  The message lists the
//     offending callees so the caller can add the missing HLFIR file.
//
// Allowlist:
//     We tolerate callees that clearly come from the Flang runtime or
//     libm (mathematical intrinsics) — these are resolved at link time
//     against libFortranRuntime / libm and are not expected to have an
//     HLFIR body.
// ============================================================================

#include "passes/Passes.h"

#include "flang/Optimizer/Dialect/FIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace hlfir_bridge {

namespace {

/// True if ``name`` names a runtime / intrinsic entry we don't expect to
/// ship as HLFIR.  Intentionally conservative — we want real unresolved
/// calls to ICON-level kernels to show up, not be swallowed here.
static bool isAllowedCallee(llvm::StringRef name) {
    // Flang runtime: ``_FortranA...`` / ``_Fortran...``.
    if (name.starts_with("_Fortran")) return true;
    // Common libm entries Flang lowers math intrinsics to.
    static const llvm::StringSet<> kLibm = {
        "sinf",  "cosf",  "tanf",  "expf",  "logf",  "sqrtf", "powf",
        "sin",   "cos",   "tan",   "exp",   "log",   "sqrt",  "pow",
        "atan2f","atan2", "hypotf","hypot", "fabsf", "fabs",
    };
    if (kLibm.contains(name)) return true;
    // C stdlib we might legitimately leave in the IR.
    static const llvm::StringSet<> kStd = {"malloc", "free", "abort"};
    if (kStd.contains(name)) return true;
    return false;
}

struct VerifyNoUnresolvedCallsPass
    : public mlir::PassWrapper<VerifyNoUnresolvedCallsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyNoUnresolvedCallsPass)

    llvm::StringRef getArgument() const final {
        return "hlfir-verify-no-unresolved-calls";
    }
    llvm::StringRef getDescription() const final {
        return "Fail if any fir.call still resolves to an external "
               "declaration outside the runtime/intrinsic allowlist.";
    }

    void runOnOperation() override {
        auto module = getOperation();
        mlir::SymbolTable symTab(module);

        llvm::SmallVector<std::string, 4> unresolved;
        auto checkCallee = [&](llvm::StringRef name) {
            auto callee = symTab.lookup<mlir::func::FuncOp>(name);
            if (callee && !callee.isDeclaration()) return;  // fully resolved
            if (isAllowedCallee(name)) return;
            unresolved.push_back(name.str());
        };
        module.walk([&](fir::CallOp call) {
            if (auto sym = call.getCallee())
                checkCallee(sym->getLeafReference().getValue());
        });
        // func.call is the canonical form too — belt-and-suspenders.
        module.walk([&](mlir::func::CallOp call) {
            checkCallee(call.getCallee());
        });

        if (unresolved.empty()) return;

        std::string msg = "unresolved calls after inlining (missing HLFIR inputs?):";
        // Deduplicate while preserving first-seen order so the message
        // isn't noisy when a single missing callee is called many times.
        llvm::StringSet<> seen;
        for (auto &n : unresolved) {
            if (seen.insert(n).second) { msg += " "; msg += n; }
        }
        module.emitError() << msg;
        signalPassFailure();
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> createVerifyNoUnresolvedCallsPass() {
    return std::make_unique<VerifyNoUnresolvedCallsPass>();
}

}  // namespace hlfir_bridge
