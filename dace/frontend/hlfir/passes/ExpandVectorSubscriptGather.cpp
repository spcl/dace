// ============================================================================
// ExpandVectorSubscriptGather.cpp  --  replace hlfir.associate of an elemental
// with an explicit gather temp.
// ============================================================================
//
// Pattern matched (typical for noncontiguous slice arguments):
//
//     %elem = hlfir.elemental %shape : ... -> !hlfir.expr<NxT> { ... }
//     %a:3  = hlfir.associate %elem(%shape) : ... ->
//                 (!fir.ref<!fir.array<NxT>>, !fir.ref<...>, i1)
//     ...uses of %a#0 / %a#1 (e.g. inlined callee dummy declare)...
//     hlfir.end_associate %a#1, %a#2 : ...
//     hlfir.destroy %elem : !hlfir.expr<NxT>
//
// ``hlfir.associate`` materialises an elemental expression into an
// addressable temp so callers (typically inlined callees that took the
// gather as a dummy arg) can take a ``fir.ref`` to it.  The bridge has
// no direct lowering for ``hlfir.associate``: the resulting addressable
// temp has no ``hlfir.declare``, so ``traceToDecl`` can't bind the
// inlined callee's reads to anything ``extract_vars`` registered.
//
// Rewrite to:
//
//     %buf  = fir.alloca !fir.array<NxT>
//     %decl:2 = hlfir.declare %buf(%shape) {uniq_name =
//     "_QF<encl>E__assoc_<n>"} fir.do_loop %i = 1 to N step 1 {
//         %v   = hlfir.apply %elem, %i_idx : ...
//         %dst = hlfir.designate %decl#0 (%i_idx)
//         hlfir.assign %v to %dst
//     }
//     // %a#0 and %a#1 -> %decl#0; %a#2 -> false (no heap copy)
//
// After the rewrite the inlined dummy declare aliasing the temp resolves
// through the same ``hlfir.designate``-of-declare path the bridge already
// handles.  ``hlfir.apply`` on the original elemental keeps the gather
// expression intact so ``buildExpr`` lowers it through the same path as
// any other elemental.
//
// ----------------------------------------------------------------------------
// SCOPE / DESIGN CONSTRAINTS  --  see also dace/frontend/hlfir/README.md
// "Future support map" for user-facing status.
// ----------------------------------------------------------------------------
//
// 1. **Static extent only  --  IR-rewrite limitation, not a DaCe one.**
//    The DaCe-side model permits symbolic extent: inside the gather
//    loop each iteration reassigns the SAME indirection symbol slot
//    (``<arr>_at<gid>``), so we only need ONE slot per syntactic
//    access site  --  never an "array of N symbols".  N can be runtime.
//
//    The static-only restriction is here in the local IR rewrite:
//    emitting a dynamic-extent ``hlfir.declare`` requires careful
//    result-type plumbing (the ``second-result-must-match-memref``
//    verifier rule).  When extent isn't a compile-time constant the
//    pass aborts with a clear error pointing at the source location.
//
// 2. **Rank-N elementals supported (N >= 1).**  Higher-rank gathers
//    (``d(cols2, cols)`` -> 2-D elemental, ``d(:, cols)`` -> 2-D
//    range+index gather) build a nested ``fir.do_loop`` tree
//    matching the elemental's ``fir.shape<N>``.  The elemental body
//    takes one ``index`` block-arg per dim; ``hlfir.apply`` and the
//    destination ``hlfir.designate`` consume all N IVs.
//
// 3. **INTENT(in) gathers only.**  The i1 ``must finalise`` flag from
//    ``hlfir.end_associate`` is hard-coded to ``false``  --  i.e. no
//    scatter-back after the inlined callee returns.  INTENT(out) /
//    INTENT(inout) gathers (rare in real code) would need a paired
//    scatter loop after the call.
//
// 4. **Source must be a literal ``hlfir.elemental``.**  Other
//    expression sources (whole-array intrinsic results, struct field
//    sections fully materialised by an upstream pass) bypass this
//    rewrite.
// ============================================================================

#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

struct ExpandVectorSubscriptGatherPass
    : public mlir::PassWrapper<ExpandVectorSubscriptGatherPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandVectorSubscriptGatherPass)

  llvm::StringRef getArgument() const final {
    return "hlfir-expand-vector-subscript-gather";
  }
  llvm::StringRef getDescription() const final {
    return "Replace hlfir.associate of an hlfir.elemental with an "
           "explicit alloca + gather DO loop so the bridge can name "
           "the materialised temp.";
  }

  void runOnOperation() override {
    unsigned counter = 0;
    llvm::SmallVector<hlfir::AssociateOp, 8> elemTargets;
    llvm::SmallVector<hlfir::AssociateOp, 8> scalarTargets;
    getOperation().walk([&](hlfir::AssociateOp op) {
      auto src = op.getSource();
      if (mlir::isa_and_nonnull<hlfir::ElementalOp>(src.getDefiningOp())) {
        elemTargets.push_back(op);
        return;
      }
      // Scalar value-by-reference associate.  Common shape:
      // ``call f(s, 0.5d0)``  --  flang emits
      // ``%a:3 = hlfir.associate %cst {adapt.valuebyref}`` so
      // the inlined callee's by-ref dummy can take ``%a#0``.
      // The result type is a plain ``fir.ref<T>`` (rank 0); the
      // source is whatever scalar value (constant, fir.load, etc.).
      if (mlir::isa<fir::ReferenceType>(op.getResult(0).getType()) &&
          !mlir::isa<hlfir::ExprType>(src.getType())) {
        scalarTargets.push_back(op);
      }
    });

    for (auto assoc : elemTargets)
      if (mlir::failed(rewrite(assoc, counter++))) return signalPassFailure();
    for (auto assoc : scalarTargets)
      if (mlir::failed(rewriteScalar(assoc, counter++)))
        return signalPassFailure();
  }

  /// Materialise a scalar value-by-reference associate into a local
  /// alloca + ``fir.store`` of the value, so the bridge sees a normal
  /// local transient with a single store rather than a nameless
  /// associate result.  After this pass, the inlined-callee dummy
  /// declare aliasing ``%a#0`` resolves through the standard
  /// ``asAssumedShapeAlias`` chain to the synthesised local.
  mlir::LogicalResult rewriteScalar(hlfir::AssociateOp assoc, unsigned id) {
    auto loc = assoc.getLoc();
    auto src = assoc.getSource();
    auto refTy = mlir::cast<fir::ReferenceType>(assoc.getResult(0).getType());
    auto eleTy = refTy.getEleTy();

    std::string enclName = "anon";
    if (auto fn = assoc->getParentOfType<mlir::func::FuncOp>()) {
      auto sym = fn.getSymName().str();
      if (sym.rfind("_QP", 0) == 0)
        enclName = sym.substr(3);
      else if (sym.rfind("_QM", 0) == 0) {
        auto p = sym.find('P', 3);
        if (p != std::string::npos)
          enclName = sym.substr(3, p - 3) + "_" + sym.substr(p + 1);
        else
          enclName = sym.substr(3);
      } else
        enclName = sym;
    }
    std::string uniqName =
        "_QF" + enclName + "E__assoc_scalar_" + std::to_string(id);

    mlir::OpBuilder b(assoc);
    auto alloca = b.create<fir::AllocaOp>(loc, eleTy);
    auto declOp = b.create<hlfir::DeclareOp>(loc, alloca.getResult(), uniqName,
                                             /*shape=*/mlir::Value{});
    b.create<fir::StoreOp>(loc, src, declOp.getResult(0));

    b.setInsertionPointAfter(assoc);
    auto falseI1 = b.create<mlir::arith::ConstantOp>(loc, b.getBoolAttr(false));
    assoc.getResult(0).replaceAllUsesWith(declOp.getResult(0));
    assoc.getResult(1).replaceAllUsesWith(declOp.getResult(0));
    assoc.getResult(2).replaceAllUsesWith(falseI1.getResult());
    assoc.erase();
    return mlir::success();
  }

 private:
  mlir::LogicalResult rewrite(hlfir::AssociateOp assoc, unsigned id) {
    auto loc = assoc.getLoc();
    auto src = assoc.getSource();
    auto elem = mlir::cast<hlfir::ElementalOp>(src.getDefiningOp());

    auto exprTy = mlir::cast<hlfir::ExprType>(src.getType());
    auto eleTy = exprTy.getElementType();
    auto shapeOper = elem.getShape();
    if (!shapeOper) {
      return assoc.emitError(
          "hlfir-expand-vector-subscript-gather: elemental has no shape "
          "operand  --  cannot determine gather extent");
    }
    auto shapeOp =
        mlir::dyn_cast_or_null<fir::ShapeOp>(shapeOper.getDefiningOp());
    if (!shapeOp) {
      return assoc.emitError(
          "hlfir-expand-vector-subscript-gather: shape operand is not a "
          "fir.shape op  --  unsupported elemental form");
    }

    // Per-dim extent values + static-or-dynamic classification.
    // Rank-N elementals (the common 2-D pattern is
    // ``d(cols2, cols)`` materialising a 2-D gather temp) follow
    // the same shape as the rank-1 case but build a nested
    // ``fir.do_loop`` tree below.
    llvm::SmallVector<mlir::Value, 4> extents(shapeOp.getExtents().begin(),
                                              shapeOp.getExtents().end());
    unsigned rank = (unsigned)extents.size();
    llvm::SmallVector<int64_t, 4> staticExts(
        rank, fir::SequenceType::getUnknownExtent());
    bool allStatic = true;
    for (unsigned i = 0; i < rank; ++i) {
      auto cstExt = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(
          extents[i].getDefiningOp());
      if (cstExt)
        if (auto a = mlir::dyn_cast<mlir::IntegerAttr>(cstExt.getValue()))
          staticExts[i] = a.getInt();
      if (staticExts[i] == fir::SequenceType::getUnknownExtent())
        allStatic = false;
    }

    mlir::OpBuilder b(assoc);

    // 1) fir.alloca + hlfir.declare for the temp.  Static-shape
    //    when every dim is a compile-time constant (preferred  --
    //    easier on downstream passes and the bridge's seq-extent
    //    fallback in ``extract_vars``); dynamic otherwise, with
    //    extent operands threaded through ``shape``.
    auto seqTy = fir::SequenceType::get(staticExts, eleTy);
    fir::AllocaOp alloca;
    if (allStatic) {
      alloca = b.create<fir::AllocaOp>(loc, seqTy);
    } else {
      alloca =
          b.create<fir::AllocaOp>(loc, seqTy, /*uniqName=*/llvm::StringRef{},
                                  /*bindcName=*/llvm::StringRef{},
                                  /*typeparams=*/mlir::ValueRange{},
                                  /*shape=*/mlir::ValueRange{extents});
    }
    // Derive a Flang-style mangled name so extractName + sdfg_name
    // parse it like any other local: ``_QF<encl>E<src>_gather_<n>``.
    // Including the source array name in the temp gives the SDFG a
    // self-documenting transient ("d_gather_0" instead of
    // "__assoc_0")  --  the inlined-callee tasklet body becomes
    // readable.
    std::string enclName = "anon";
    if (auto fn = assoc->getParentOfType<mlir::func::FuncOp>()) {
      auto sym = fn.getSymName().str();
      // ``_QPmain`` -> ``main``; ``_QMmodPfn`` -> ``mod_fn`` (best effort).
      if (sym.rfind("_QP", 0) == 0)
        enclName = sym.substr(3);
      else if (sym.rfind("_QM", 0) == 0) {
        auto p = sym.find('P', 3);
        if (p != std::string::npos)
          enclName = sym.substr(3, p - 3) + "_" + sym.substr(p + 1);
        else
          enclName = sym.substr(3);
      } else
        enclName = sym;
    }
    // Walk the elemental's body to find the source array name.
    // Pattern: yield_element of fir.load of hlfir.designate of a
    // declare; the declare's uniq_name is the source array.
    std::string srcName = "expr";
    if (auto &elemBlock = elem.getRegion().front(); !elemBlock.empty()) {
      for (auto &op : elemBlock) {
        auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op);
        if (!y) continue;
        auto v = y.getElementValue();
        for (int i = 0; i < 128 && v; ++i) {
          auto *d = v.getDefiningOp();
          if (!d) break;
          if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
            v = cv.getValue();
            continue;
          }
          if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
            v = ld.getMemref();
            continue;
          }
          if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
            v = dg.getMemref();
            continue;
          }
          if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
            auto un = dc.getUniqName().str();
            auto p = un.rfind('E');
            if (p != std::string::npos)
              srcName = un.substr(p + 1);
            else
              srcName = un;
            break;
          }
          break;
        }
        break;
      }
    }
    std::string uniqName =
        "_QF" + enclName + "E" + srcName + "_gather_" + std::to_string(id);
    // Result-type convention (matches what flang itself emits for
    // assumed-shape declares):
    //   * all-static: result#0 = result#1 = ``fir.ref<array<...>>``
    //   * any-dynamic: result#0 = ``fir.box<array<...>>`` (HLFIR
    //                 variable form, carries shape), result#1 =
    //                 ``fir.ref<array<...>>`` (raw memref).
    // Use the explicit-result-types builder so we control the
    // shape regardless of which short-form builder rule applies.
    hlfir::DeclareOp declOp;
    auto refTy = fir::ReferenceType::get(seqTy);
    if (allStatic) {
      declOp = b.create<hlfir::DeclareOp>(loc, alloca.getResult(), uniqName,
                                          shapeOper);
    } else {
      auto boxTy = fir::BoxType::get(seqTy);
      declOp = b.create<hlfir::DeclareOp>(
          loc,
          /*resultType0=*/boxTy,
          /*resultType1=*/refTy,
          /*memref=*/alloca.getResult(),
          /*shape=*/shapeOper,
          /*typeparams=*/mlir::ValueRange{},
          /*dummy_scope=*/mlir::Value{},
          /*uniq_name=*/b.getStringAttr(uniqName),
          /*fortran_attrs=*/fir::FortranVariableFlagsAttr{},
          /*data_attr=*/cuf::DataAttributeAttr{});
    }

    // 2) Build the nested gather DO loop tree  --  one ``fir.do_loop``
    //    per dim from outermost (dim 0) to innermost (dim N-1).
    //    Collect the per-dim induction variables; the inner body
    //    applies the elemental and stores into the temp at the
    //    composite index.
    auto idxTy = b.getIndexType();
    (void)idxTy;
    auto c1 = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(1));
    llvm::SmallVector<mlir::Value, 4> ivs;
    for (unsigned i = 0; i < rank; ++i) {
      auto loop = b.create<fir::DoLoopOp>(loc, c1, extents[i], c1,
                                          /*unordered=*/false,
                                          /*finalCountValue=*/false);
      ivs.push_back(loop.getInductionVar());
      b.setInsertionPointToStart(loop.getBody());
    }
    // Inner body: apply + designate + assign.  ``hlfir.apply`` on
    // an N-D elemental takes N index args matching the elemental's
    // block-arg list.
    auto applied =
        b.create<hlfir::ApplyOp>(loc, eleTy, src, mlir::ValueRange{ivs},
                                 /*typeparams=*/mlir::ValueRange{});
    auto dest = b.create<hlfir::DesignateOp>(
        loc, fir::ReferenceType::get(eleTy), declOp.getResult(0),
        mlir::ValueRange{ivs});
    b.create<hlfir::AssignOp>(loc, applied.getResult(), dest.getResult());

    // 3) Replace associate uses.  Match the result-type convention
    //    flang itself uses for ``hlfir.associate``:
    //      result#0 (HLFIR variable, box for dynamic)  -> decl#0
    //      result#1 (raw memref ref) -> decl#1 for dynamic, decl#0
    //                                  for static (where both are ref)
    //      result#2 (must-finalise i1) -> false (no heap allocation)
    b.setInsertionPointAfter(assoc);
    auto falseI1 = b.create<mlir::arith::ConstantOp>(loc, b.getBoolAttr(false));
    assoc.getResult(0).replaceAllUsesWith(declOp.getResult(0));
    assoc.getResult(1).replaceAllUsesWith(allStatic ? declOp.getResult(0)
                                                    : declOp.getResult(1));
    assoc.getResult(2).replaceAllUsesWith(falseI1.getResult());

    // 4) Erase the associate, plus any matching end_associate / destroy
    //    that referenced its results.  After replaceAllUsesWith those
    //    cleanup ops have orphan operands; sweep them in the same
    //    rewrite step so the IR re-verifies cleanly.
    assoc.erase();
    // Clean up cleanup ops that became no-ops.
    getOperation().walk([&](mlir::Operation *op) {
      if (auto endA = mlir::dyn_cast<hlfir::EndAssociateOp>(op)) {
        // After RAUW, end_associate's operand is decl#0  --  not a temp
        // it owns.  Erase it; finalisation is handled by SDFG
        // transient lifetime.
        if (endA.getVar().getDefiningOp() == declOp.getOperation())
          endA.erase();
      }
    });
    // ``hlfir.destroy`` of the elemental can stay  --  DCE drops it later.
    return mlir::success();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createExpandVectorSubscriptGatherPass() {
  return std::make_unique<ExpandVectorSubscriptGatherPass>();
}

}  // namespace hlfir_bridge
