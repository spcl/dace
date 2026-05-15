// ============================================================================
// FoldCopyInOut.cpp  --  fold ``hlfir.copy_in`` / ``hlfir.copy_out`` pairs.
// ============================================================================
//
// Motivation:
//     Flang materialises a non-contiguous slice argument (e.g.
//     ``call inner_loops(INP(I, :), OUT(I, :))`` where ``INP`` is column-
//     major) into a heap-allocated copy via:
//
//         %src   = hlfir.designate %parent (i, lo:hi:1) shape ...
//                : -> !fir.box<!fir.array<NxT>>
//         %cpy:2 = hlfir.copy_in %src to %tempBox
//                : (!fir.box<!fir.array<NxT>>, !fir.ref<!fir.box<...>>)
//                  -> (!fir.box<!fir.array<NxT>>, i1)
//         %addr  = fir.box_addr %cpy#0 : ... -> !fir.ref<!fir.array<NxT>>
//         %alias:2 = hlfir.declare %addr ... { uniq_name = "_QFcalleeEarg" }
//                                              : ...
//         ... uses of %alias#0 / %alias#1 ...
//         hlfir.copy_out %tempBox, %cpy#1 to %src : ...
//
//     The bridge does not model ``hlfir.copy_in`` / ``hlfir.copy_out``  --
//     ``%alias`` becomes an uninitialised transient and writes via the
//     callee never propagate back to ``%parent``.  Tests with this
//     pattern (``memlet_in_map_test``, ``type_array_slice``, the
//     ``noncontiguous_*`` cluster) silently produce wrong values.
//
// What the pass does (simple-section scope):
//     For each ``hlfir.copy_in`` whose source is a ``hlfir.designate``
//     with EXACTLY ONE TRAILING TRIPLET (stride 1) and arbitrary scalar
//     prefix, fold the alias-side accesses back to the parent:
//
//         %alias #0 (j)  ->  %parent (scalars..., j + lo - 1)
//
//     ``copy_in`` / ``copy_out`` and the heap buffer alloca then erase
//     because nothing references them.  The chain below the alias
//     declare reads / writes ``%parent`` directly.
//
// Out of scope (left for follow-ups):
//     * Non-stride-1 triplets (``arr(1:N:2)``).  Would need an index
//       multiply ``+ (j-1)*stride`` instead of the current ``+ (lo-1)``.
//     * Multiple triplets (``arr(:, :)``).  Alias is then multi-D and
//       the per-dim mapping needs more care.
//     * Non-section sources (``copy_in`` of a whole array / scalar)  --
//       those are pathological and the bridge should reject them
//       loudly elsewhere if they ever surface.
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// Walk a designate's per-dim ``isTriplet`` flags + flat index list,
/// extracting the scalar-prefix indices and the single trailing
/// triplet's ``(lo, hi, stride)``.  Returns false when the source
/// shape is outside the simple-section scope (zero or multi triplets,
/// or the cursor logic mismatches the operand layout).
struct SectionShape {
  llvm::SmallVector<mlir::Value, 4> scalars;
  unsigned tripletDim;  // index in is_triplet flag list
  mlir::Value tripLo;
  mlir::Value tripHi;
  mlir::Value tripStride;
};

static bool parseSimpleSection(hlfir::DesignateOp dg, SectionShape &out) {
  auto trip = dg.getIsTripletAttr();
  if (!trip) return false;
  auto tripFlags = trip.asArrayRef();
  if (tripFlags.empty()) return false;

  unsigned tripletCount = 0;
  for (bool t : tripFlags)
    if (t) tripletCount++;
  if (tripletCount != 1) return false;

  auto idxRange = dg.getIndices();
  unsigned cursor = 0;
  for (unsigned i = 0; i < tripFlags.size(); ++i) {
    if (tripFlags[i]) {
      if (cursor + 3 > idxRange.size()) return false;
      out.tripletDim = i;
      out.tripLo = idxRange[cursor];
      out.tripHi = idxRange[cursor + 1];
      out.tripStride = idxRange[cursor + 2];
      cursor += 3;
    } else {
      if (cursor + 1 > idxRange.size()) return false;
      out.scalars.push_back(idxRange[cursor]);
      cursor += 1;
    }
  }
  return true;
}

/// Trace ``v`` to a constant ``index``-typed integer if possible.
/// Walks ``arith.constant`` and ``fir.convert`` shims.  Returns
/// ``std::nullopt`` if the value isn't a constant.
static std::optional<int64_t> traceConstIndex(mlir::Value v) {
  if (!v) return std::nullopt;
  auto *def = v.getDefiningOp();
  if (!def) return std::nullopt;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return ia.getInt();
    return std::nullopt;
  }
  if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def))
    return traceConstIndex(cv.getValue());
  return std::nullopt;
}

/// True when ``v`` is the integer constant ``1`` (any int width / index).
static bool isConstOne(mlir::Value v) {
  auto c = traceConstIndex(v);
  return c.has_value() && *c == 1;
}

struct FoldCopyInOutPass
    : public mlir::PassWrapper<FoldCopyInOutPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldCopyInOutPass)

  llvm::StringRef getArgument() const final { return "hlfir-fold-copy-in-out"; }
  llvm::StringRef getDescription() const final {
    return "Fold hlfir.copy_in / hlfir.copy_out pairs around inlined-callee "
           "alias declares (single-trailing-triplet stride-1 sections only).";
  }

  void runOnOperation() override {
    llvm::SmallVector<hlfir::CopyInOp, 16> copies;
    getOperation().walk([&](hlfir::CopyInOp op) { copies.push_back(op); });
    for (auto cin : copies) tryFold(cin);
  }

  void tryFold(hlfir::CopyInOp cin) {
    // 1) Source must be a section ``hlfir.designate``.
    auto srcDg = cin.getVar().getDefiningOp<hlfir::DesignateOp>();
    if (!srcDg) return;
    SectionShape sec;
    if (!parseSimpleSection(srcDg, sec)) return;
    if (!isConstOne(sec.tripStride)) return;

    // 2) Walk users of ``cin#0`` (the box copy) for the
    // ``fir.box_addr`` and from there for the alias declare.
    fir::BoxAddrOp boxAddr;
    for (auto *u : cin.getResult(0).getUsers()) {
      if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(u)) {
        boxAddr = ba;
        break;
      }
    }
    if (!boxAddr) return;

    hlfir::DeclareOp aliasDecl;
    for (auto *u : boxAddr.getResult().getUsers()) {
      if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
        aliasDecl = d;
        break;
      }
    }
    if (!aliasDecl) return;

    // 3) Source's parent  --  the array we're slicing.  This is
    // the memref of the source designate.
    mlir::Value parent = srcDg.getMemref();
    mlir::OpBuilder b(aliasDecl);
    mlir::Location loc = aliasDecl.getLoc();

    // 4) Rewrite uses of the alias declare's results.  Each
    // designate ``%alias #X (j)`` becomes ``%parent (scalars...,
    // j + lo - 1)``.  Whole-result uses (``hlfir.assign %v to
    // %alias #0`` or ``%alias #0`` passed as a function arg) are
    // not handled in this scope  --  bail if encountered.
    llvm::SmallVector<hlfir::DesignateOp, 8> aliasUseDgs;
    bool foreignUse = false;
    for (mlir::Value res : {aliasDecl.getResult(0), aliasDecl.getResult(1)}) {
      for (auto *u : res.getUsers()) {
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u)) {
          aliasUseDgs.push_back(dg);
        } else {
          foreignUse = true;
        }
      }
    }
    if (foreignUse) return;

    for (auto useDg : aliasUseDgs) rewriteAccess(useDg, parent, sec, b);

    // 5) Erase the chain: alias declare, box_addr, copy_in,
    // any copy_out targeting this copy_in, the alloca for the
    // temp box.  Order matters  --  erase users first, defs last.
    // copy_out has the form ``copy_out %tempBox, %cin#1 to %var``.
    llvm::SmallVector<hlfir::CopyOutOp, 2> copyOuts;
    getOperation().walk([&](hlfir::CopyOutOp op) {
      if (op.getOperand(1) == cin.getResult(1)) copyOuts.push_back(op);
    });

    if (aliasDecl.getResult(0).use_empty() &&
        aliasDecl.getResult(1).use_empty())
      aliasDecl.erase();

    if (boxAddr.getResult().use_empty()) boxAddr.erase();

    for (auto co : copyOuts) co.erase();

    if (cin.getResult(0).use_empty() && cin.getResult(1).use_empty()) {
      mlir::Value temp = cin.getTempBox();
      cin.erase();
      // The temp box is typically a ``fir.alloca`` whose only
      // users were the copy_in / copy_out pair.  Erase if dead.
      if (auto *def = temp.getDefiningOp())
        if (def->use_empty()) def->erase();
    }
  }

  /// Rewrite a single ``hlfir.designate %alias (j_1, ..., j_K)`` use
  /// to the equivalent designate on ``parent`` with the section
  /// indices folded in.  Preserves triplets / shape on the alias-
  /// access side (so ``%alias(1:N:1)`` whole-array becomes
  /// ``%parent(scalars..., 1:N:1)``).
  void rewriteAccess(hlfir::DesignateOp useDg, mlir::Value parent,
                     const SectionShape &sec, mlir::OpBuilder &b) {
    b.setInsertionPoint(useDg);
    auto loc = useDg.getLoc();

    // Shift the leaf indices into the parent's frame.  For each
    // alias dim (which is the alias's own 1..N-style index), the
    // parent index is ``alias_idx + lo - 1``.  Stride-1 only  --
    // the caller filters strides out before reaching here.
    mlir::Value loMinusOne;
    auto loConst = traceConstIndex(sec.tripLo);
    if (loConst && *loConst == 1) {
      // Common Fortran default ``arr(:)`` lo = 1, no shift
      // needed.
      loMinusOne = {};
    } else {
      mlir::Value lo = sec.tripLo;
      // Promote to index if needed.
      if (!lo.getType().isIndex()) {
        lo = b.create<fir::ConvertOp>(loc, b.getIndexType(), lo);
      }
      mlir::Value one = b.create<mlir::arith::ConstantOp>(loc, b.getIndexType(),
                                                          b.getIndexAttr(1));
      loMinusOne = b.create<mlir::arith::SubIOp>(loc, lo, one);
    }

    // Build the new index list: parent's scalar prefix + shifted
    // alias indices + (no scalars after triplet in this scope).
    llvm::SmallVector<mlir::Value, 6> newIndices(sec.scalars.begin(),
                                                 sec.scalars.end());
    // Walk the alias designate's own indices.  Each may be a
    // scalar or a triplet (``%alias(1:N:1)`` whole-array shape).
    // Triplet flags on the alias side carry over verbatim  --  we
    // just need to shift the ``lo`` and ``hi`` values per-triplet.
    auto aliasTripAttr = useDg.getIsTripletAttr();
    auto aliasIdx = useDg.getIndices();
    llvm::SmallVector<bool, 4> newTripFlags;
    // Scalar prefix from the section is non-triplet.
    for (size_t i = 0; i < sec.scalars.size(); ++i)
      newTripFlags.push_back(false);

    if (!aliasTripAttr || aliasTripAttr.asArrayRef().empty()) {
      // No triplets on alias use; all alias-supplied indices
      // are scalars.  Each shifts by ``lo-1`` (only one alias
      // dim corresponds to the source triplet).
      for (auto idx : aliasIdx) {
        mlir::Value shifted = idx;
        if (loMinusOne) {
          if (!idx.getType().isIndex())
            shifted = b.create<fir::ConvertOp>(loc, b.getIndexType(), idx);
          shifted = b.create<mlir::arith::AddIOp>(loc, shifted, loMinusOne);
        }
        newIndices.push_back(shifted);
        newTripFlags.push_back(false);
      }
    } else {
      // Mixed: walk per-dim, push lo/hi/stride for triplets
      // (lo/hi shifted by ``lo-1``), scalar otherwise.
      unsigned cursor = 0;
      for (bool isT : aliasTripAttr.asArrayRef()) {
        if (isT) {
          if (cursor + 3 > aliasIdx.size()) return;
          auto shift = [&](mlir::Value v) {
            if (!loMinusOne) return v;
            mlir::Value vc = v;
            if (!v.getType().isIndex())
              vc = b.create<fir::ConvertOp>(loc, b.getIndexType(), v);
            return (mlir::Value)b.create<mlir::arith::AddIOp>(loc, vc,
                                                              loMinusOne);
          };
          newIndices.push_back(shift(aliasIdx[cursor]));
          newIndices.push_back(shift(aliasIdx[cursor + 1]));
          newIndices.push_back(aliasIdx[cursor + 2]);
          newTripFlags.push_back(true);
          cursor += 3;
        } else {
          if (cursor + 1 > aliasIdx.size()) return;
          mlir::Value shifted = aliasIdx[cursor];
          if (loMinusOne) {
            if (!shifted.getType().isIndex())
              shifted =
                  b.create<fir::ConvertOp>(loc, b.getIndexType(), shifted);
            shifted = b.create<mlir::arith::AddIOp>(loc, shifted, loMinusOne);
          }
          newIndices.push_back(shifted);
          newTripFlags.push_back(false);
          cursor += 1;
        }
      }
    }

    auto newOp = b.create<hlfir::DesignateOp>(
        loc,
        /*result_type=*/useDg.getResult().getType(),
        /*memref=*/parent,
        /*component=*/mlir::StringAttr{},
        /*component_shape=*/mlir::Value{},
        /*indices=*/mlir::ValueRange{newIndices},
        /*is_triplet=*/
        (newTripFlags.empty() ? mlir::DenseBoolArrayAttr{}
                              : b.getDenseBoolArrayAttr(newTripFlags)),
        /*substring=*/mlir::ValueRange{},
        /*complex_part=*/mlir::BoolAttr{},
        /*shape=*/useDg.getShape(),
        /*typeparams=*/mlir::ValueRange{},
        /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
    useDg.getResult().replaceAllUsesWith(newOp.getResult());
    useDg.erase();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createFoldCopyInOutPass() {
  return std::make_unique<FoldCopyInOutPass>();
}

}  // namespace hlfir_bridge
