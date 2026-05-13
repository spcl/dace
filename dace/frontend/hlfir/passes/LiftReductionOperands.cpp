// ============================================================================
// LiftReductionOperands.cpp  --  pre-lower reduction intrinsics that appear as
// inline expression operands.
// ============================================================================
//
// Motivation:
//     The bridge's ``buildExpr`` (expressions.cpp) handles scalar-shape
//     ops (arith.*, math.*, hlfir.designate, fir.load, ...) but has no
//     case for array-reducing intrinsics  --  ``hlfir.sum`` / ``maxval`` /
//     ``minval`` / ``product`` / ``any`` / ``all``.  Those map to scalar
//     values but can't be rendered as a tasklet expression: a tasklet
//     reads scalars, not array slices.
//
//     The dispatcher in ``dispatch.cpp`` already handles the top-level
//     case: ``target = MAXVAL(arr)`` routes to ``buildReduceNode`` /
//     ``buildSectionReduceAssign`` / ``buildElementalAnyAllReduce``.  The
//     gap is reductions used as INLINE operands:
//
//         max_vcfl_dyn = MAX(p_diag%max_vcfl_dyn, MAXVAL(vcflmax(s:e)))
//
//     ``buildExpr`` returns ``"?"`` for the inner ``MAXVAL``, the resulting
//     tasklet code ``_out = max(_in_..., ?)`` fails Python ``ast.parse``,
//     and the SDFG can't build.
//
// What the pass does:
//     For each ``hlfir.assign`` in the module, walk its RHS subtree for
//     any reduction op that is NOT the immediate RHS.  For each such
//     "nested" reduction:
//
//         1. Insert ``%tmp = fir.alloca T`` + ``%tmp_decl = hlfir.declare``
//            in the function entry, where T is the reduction's scalar
//            result type.
//         2. Insert ``hlfir.assign <reduction_op_result> to %tmp_decl#0``
//            immediately BEFORE the consuming assign.
//         3. Replace uses of the reduction op's result in the RHS subtree
//            with ``fir.load %tmp_decl#0``.
//
//     After this pass:
//         - The lifted ``temp = MAXVAL(slice)`` is a top-level reduction
//           assign  --  ``buildSectionReduceAssign`` handles it.
//         - The outer ``max_vcfl_dyn = MAX(p_diag_..., load(temp))``
//           sees only a scalar load  --  the existing buildExpr arith.maxnumf
//           handler renders it correctly.
//
// Pipeline position:
//     After ``hlfir-flatten-structs`` (so designate chains on flattened
//     companions are already rewritten) and BEFORE the AST extractor's
//     dispatch (``buildAssignNode`` / ``buildExpr``).  Insertion point in
//     the bridge's DEFAULT_PIPELINE: directly before
//     ``hlfir-default-intent`` is fine.
//
// Out of scope:
//     * ``hlfir.count``  --  already routed through a libcall in the dispatch
//       table; that codepath supports inline use via the libcall's emit
//       path.  If it ever surfaces as a problem, fold in here.
//     * Reductions whose source is an ``hlfir.elemental`` (a compound
//       boolean expression).  The dispatcher's Mode-C path already
//       materialises a transient mask before calling the reduce; that
//       same path covers the lifted top-level case here.
// ============================================================================

#include "passes/Passes.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

namespace hlfir_bridge {

namespace {

/// True iff ``op`` is one of the array-reducing intrinsics that
/// ``buildExpr`` cannot render inline.  ``hlfir.count`` is excluded  --
/// the dispatcher routes it through ``CountLibraryNode`` which handles
/// inline use via the libcall emit path.
static bool isReductionOp(mlir::Operation *op) {
    if (!op) return false;
    return mlir::isa<hlfir::SumOp, hlfir::ProductOp, hlfir::MinvalOp,
                     hlfir::MaxvalOp, hlfir::AnyOp, hlfir::AllOp>(op);
}

/// Find every reduction op transitively used by ``rootOp`` (the RHS of an
/// assign)  --  except the rootOp itself.  Returns them in
/// reverse-postorder so callers process inner reductions before outer.
static void collectNestedReductions(
        mlir::Operation *rootOp,
        llvm::SmallVectorImpl<mlir::Operation *> &out) {
    if (!rootOp) return;
    llvm::SmallVector<mlir::Operation *, 8> stack;
    llvm::SmallPtrSet<mlir::Operation *, 16> seen;
    for (auto v : rootOp->getOperands())
        if (auto *def = v.getDefiningOp())
            if (seen.insert(def).second)
                stack.push_back(def);
    while (!stack.empty()) {
        auto *op = stack.pop_back_val();
        if (isReductionOp(op)) out.push_back(op);
        for (auto v : op->getOperands())
            if (auto *def = v.getDefiningOp())
                if (seen.insert(def).second)
                    stack.push_back(def);
    }
}

struct LiftReductionOperandsPass
    : public mlir::PassWrapper<LiftReductionOperandsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftReductionOperandsPass)

    llvm::StringRef getArgument() const final {
        return "hlfir-lift-reduction-operands";
    }
    llvm::StringRef getDescription() const final {
        return "Lift array-reducing intrinsics that appear as inline "
               "expression operands into a preceding scalar-temp assign, "
               "so the AST extractor sees only top-level reductions and "
               "scalar loads in the consuming expression.";
    }

    void runOnOperation() override {
        // Counter for unique temp names per function.
        llvm::DenseMap<mlir::func::FuncOp, unsigned> liftCounter;

        // Two-pass: collect first, mutate after  --  modifying the IR
        // mid-walk would invalidate iterators.
        struct Job {
            hlfir::AssignOp consumer;
            mlir::Operation *redOp;
        };
        llvm::SmallVector<Job, 16> jobs;

        getOperation().walk([&](hlfir::AssignOp assign) {
            auto rhs = assign.getRhs();
            auto *rhsOp = rhs.getDefiningOp();
            if (!rhsOp) return;
            // If the RHS itself is a reduction, the dispatcher already
            // handles it  --  leave alone.  Only lift NESTED reductions.
            llvm::SmallVector<mlir::Operation *, 4> nested;
            collectNestedReductions(rhsOp, nested);
            for (auto *r : nested) jobs.push_back({assign, r});
        });

        for (auto &job : jobs) lift(job.consumer, job.redOp, liftCounter);
    }

    /// Materialise a scalar local for the reduction op's result, emit
    /// ``hlfir.assign <red> to <local>`` before the consuming assign,
    /// and rewrite the consuming RHS to read from the local.
    void lift(hlfir::AssignOp consumer, mlir::Operation *redOp,
              llvm::DenseMap<mlir::func::FuncOp, unsigned> &liftCounter) {
        auto func = consumer->getParentOfType<mlir::func::FuncOp>();
        if (!func) return;
        if (redOp->getNumResults() != 1) return;
        auto resTy = redOp->getResult(0).getType();
        // Reduction result types are always scalar (i32, f64, logical, ...).
        // Defensive: bail on unexpected shapes.
        if (mlir::isa<fir::SequenceType>(resTy)) return;

        unsigned gid = liftCounter[func]++;
        auto loc = redOp->getLoc();
        auto *ctx = func.getContext();

        // Create the temp local at the function entry block  --  putting
        // it inline at the consuming assign's location works too, but
        // hoisting to entry keeps the pattern uniform with how flang
        // emits other Fortran-source ``REAL :: tmp`` locals.
        mlir::OpBuilder b(&func.front(), func.front().begin());
        auto allocaTy = fir::ReferenceType::get(resTy);
        auto alloca = b.create<fir::AllocaOp>(loc, resTy);
        std::string uniqName = "_QQred_lift_" + std::to_string(gid);
        mlir::NamedAttrList attrs;
        attrs.append("uniq_name", mlir::StringAttr::get(ctx, uniqName));
        // operandSegmentSizes for hlfir.declare: memref + (no shape) +
        // (no typeparams) + (no dummy_scope).
        attrs.append("operandSegmentSizes",
                     b.getDenseI32ArrayAttr({1, 0, 0, 0}));
        auto decl = b.create<hlfir::DeclareOp>(
            loc, mlir::TypeRange{allocaTy, allocaTy},
            mlir::ValueRange{alloca.getResult()}, attrs);

        // Emit the lifted assign and load IMMEDIATELY AFTER the
        // reduction op  --  placing them at the consumer's location would
        // put the load AFTER existing uses of the reduction (e.g.
        // ``arith.cmpf %scalar, %maxval`` followed by
        // ``arith.select`` followed by the assign), and rewriting those
        // earlier uses to reference the load would violate dominance.
        // After-the-reduction placement keeps the load before every
        // existing use; the reduction op itself stays at its original
        // position and the new ``hlfir.assign`` plus ``fir.load`` form
        // a tight pair right behind it.  The dispatcher then sees the
        // lifted assign as a top-level ``temp = REDUCTION(...)`` and
        // routes through the existing reduce-emit machinery; consuming
        // sites read the scalar load uniformly.
        b.setInsertionPointAfter(redOp);
        auto liftedAssign = b.create<hlfir::AssignOp>(
            loc, redOp->getResult(0), decl.getResult(0));
        auto load = b.create<fir::LoadOp>(loc, decl.getResult(0));

        // Replace every existing use of ``redOp`` with the load,
        // EXCEPT the just-emitted ``hlfir.assign`` (which intentionally
        // takes the reduction's original result as its source).
        llvm::SmallPtrSet<mlir::Operation *, 4> exceptions{liftedAssign};
        redOp->getResult(0).replaceAllUsesExcept(load.getResult(),
                                                 exceptions);
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> createLiftReductionOperandsPass() {
    return std::make_unique<LiftReductionOperandsPass>();
}

}  // namespace hlfir_bridge
