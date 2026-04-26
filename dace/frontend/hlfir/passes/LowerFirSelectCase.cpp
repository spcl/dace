// ============================================================================
// LowerFirSelectCase.cpp — Lower fir.select_case to arith.cmp + cf.cond_br.
// ============================================================================
// Problem:
//     ``fir.select_case`` is a multi-successor terminator that the upstream
//     ``mlir::inlineCall`` API mishandles when cloning a callee with one
//     into the caller — the inliner segfaults during block-operand remap.
//     This blocks the bridge's ``hlfir-inline-all`` pass on any module
//     where a ``contains``-nested subroutine uses ``SELECT CASE`` (e.g.
//     ``module/contains/subroutine foo: select case (v) case(a) ... end``).
//
// Fix:
//     Lower every ``fir.select_case`` into a chain of plain ``arith.cmpi``
//     comparisons with ``cf.cond_br`` between intermediate check blocks
//     before ``hlfir-inline-all`` runs.  Once the IR is in plain CFG
//     form, the inliner has no special-cased multi-successor op left to
//     trip on, and ``lift-cf-to-scf`` later reconstitutes the ``scf.if``
//     chain that the bridge's AST emitter consumes via the existing
//     ``scf.IfOp`` handler.
//
// Per-tag lowering (matches the bridge's existing ``buildSelectCaseChain``
// in extract_ast.cpp so the resulting SDFG is identical):
//     #fir.point %v          → ``selector == v``
//     #fir.interval %lo %hi  → ``(selector >= lo) and (selector <= hi)``
//     #fir.lower %lo         → ``selector >= lo``
//     #fir.upper %hi         → ``selector <= hi``
//     unit                   → fall-through (default arm)
//
// Float selectors fall back to ``arith.cmpf oeq`` / ``oge`` / ``ole``.
// String selectors are not handled — Fortran ``CHARACTER`` is not on the
// FaCe roadmap and ``fir.select_case`` over CHARACTER would have arrived
// here as a runtime-call lowering anyway.
// ============================================================================

#include "passes/Passes.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

namespace hlfir_bridge {

namespace {

/// Build the boolean condition for one non-``unit`` case attribute.
/// Returns nullptr if the tag shape isn't recognised — callers should
/// then leave the original ``fir.select_case`` in place.
mlir::Value buildCaseCondition(mlir::OpBuilder &b, mlir::Location loc,
                               mlir::Value selector, mlir::Attribute tag,
                               mlir::ValueRange cmpOps) {
    bool isFloat = mlir::isa<mlir::FloatType>(selector.getType());

    auto cmpEq = [&](mlir::Value v) {
        return isFloat
                   ? b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, selector, v).getResult()
                   : b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, selector, v).getResult();
    };
    auto cmpGe = [&](mlir::Value v) {
        return isFloat
                   ? b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, selector, v).getResult()
                   : b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, selector, v).getResult();
    };
    auto cmpLe = [&](mlir::Value v) {
        return isFloat
                   ? b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, selector, v).getResult()
                   : b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sle, selector, v).getResult();
    };

    if (mlir::isa<fir::PointIntervalAttr>(tag) && cmpOps.size() == 1)
        return cmpEq(cmpOps[0]);
    if (mlir::isa<fir::ClosedIntervalAttr>(tag) && cmpOps.size() == 2) {
        // ``case(b:c)`` lowers as ``(selector >= b) AND (selector <= c)``.
        mlir::Value ge = cmpGe(cmpOps[0]);
        mlir::Value le = cmpLe(cmpOps[1]);
        return b.create<mlir::arith::AndIOp>(loc, ge, le);
    }
    if (mlir::isa<fir::LowerBoundAttr>(tag) && cmpOps.size() == 1)
        return cmpGe(cmpOps[0]);
    if (mlir::isa<fir::UpperBoundAttr>(tag) && cmpOps.size() == 1)
        return cmpLe(cmpOps[0]);
    return mlir::Value();
}

/// Rewrite one ``fir.select_case`` into ``cmp + cf.cond_br`` blocks.
/// Returns ``failure()`` if any case shape is unrecognised — leaves the
/// op untouched so the existing in-bridge ``buildSelectCaseChain``
/// fallback can still handle it (or the next pipeline stage can fail
/// loudly with a useful error rather than silent miscompilation).
mlir::LogicalResult rewriteSelectCase(fir::SelectCaseOp sel) {
    mlir::Location loc = sel.getLoc();
    auto operands = sel.getOperands();
    mlir::Value selector = sel.getSelector(operands);
    auto cases = sel.getCases();

    // Find the default destination (``unit`` case) and pre-compute the
    // list of non-default case indices.
    mlir::Block *defaultDest = nullptr;
    llvm::SmallVector<unsigned, 4> nonDefault;
    for (unsigned i = 0; i < cases.size(); ++i) {
        if (mlir::isa<mlir::UnitAttr>(cases[i]))
            defaultDest = sel.getSuccessor(i);
        else
            nonDefault.push_back(i);
    }
    // Fall-through target when no default is present.  The select_case
    // op was the block's terminator, so "after" it is whatever comes
    // next in the parent function's CFG; if the user wrote no default,
    // Fortran semantics is "no-op when nothing matches".  Use the
    // default (which Flang always emits) — bail otherwise.
    if (!defaultDest) return mlir::failure();

    mlir::Block *parentBlock = sel->getBlock();
    mlir::Region *region = parentBlock->getParent();

    // First non-default case: build the cmp + cond_br BEFORE ``sel``,
    // then erase ``sel``.  Subsequent cases live in fresh blocks chained
    // via cond_br fall-throughs.
    mlir::OpBuilder b(sel);
    mlir::Block *currentCheck = parentBlock;

    for (unsigned k = 0; k < nonDefault.size(); ++k) {
        unsigned i = nonDefault[k];
        auto tag = cases[i];
        mlir::Block *succ = sel.getSuccessor(i);
        auto cmpOps = sel.getCompareOperands(operands, i);
        if (!cmpOps) return mlir::failure();

        b.setInsertionPoint(currentCheck, currentCheck == parentBlock
                                              ? sel->getIterator()
                                              : currentCheck->end());

        mlir::Value cond = buildCaseCondition(b, loc, selector, tag, *cmpOps);
        if (!cond) return mlir::failure();

        bool isLast = (k == nonDefault.size() - 1);
        mlir::Block *failDest;
        if (isLast) {
            failDest = defaultDest;
        } else {
            failDest = b.createBlock(region, std::next(currentCheck->getIterator()));
            // ``createBlock`` sets the insertion point inside the new
            // block; reset to the current check's terminator slot.
        }

        b.setInsertionPoint(currentCheck, currentCheck == parentBlock
                                              ? sel->getIterator()
                                              : currentCheck->end());
        b.create<mlir::cf::CondBranchOp>(loc, cond, succ, failDest);

        currentCheck = failDest;
    }

    // If the original op had ONLY a default case (degenerate), wire the
    // parent block to it directly.
    if (nonDefault.empty()) {
        mlir::OpBuilder bb(sel);
        bb.create<mlir::cf::BranchOp>(loc, defaultDest);
    }

    sel->erase();
    return mlir::success();
}

struct LowerFirSelectCasePass
    : public mlir::PassWrapper<LowerFirSelectCasePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerFirSelectCasePass)

    llvm::StringRef getArgument() const final { return "lower-fir-select-case"; }
    llvm::StringRef getDescription() const final {
        return "Lower fir.select_case to arith.cmp + cf.cond_br chains so "
               "downstream inlining (which mishandles fir.select_case's "
               "block-operand remap and segfaults) works on Fortran "
               "programs that use SELECT CASE inside a callee.";
    }

    void runOnOperation() override {
        // Collect first; rewriting mutates the parent block list.
        llvm::SmallVector<fir::SelectCaseOp, 8> ops;
        getOperation().walk([&](fir::SelectCaseOp op) { ops.push_back(op); });
        for (auto sel : ops) {
            if (mlir::failed(rewriteSelectCase(sel))) {
                // Leave it for a later pass (or the bridge's
                // ``buildSelectCaseChain`` fallback) to handle.
                continue;
            }
        }
    }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createLowerFirSelectCasePass() {
    return std::make_unique<LowerFirSelectCasePass>();
}

}  // namespace hlfir_bridge
