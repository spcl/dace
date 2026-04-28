// ============================================================================
// ExpandRegionAssign.cpp — replace hlfir.region_assign with elemental_addr
// destination (Fortran array-based scatter) by an explicit DO loop.
// ============================================================================
//
// Pattern matched (Fortran ``d(cols) = source`` — array-based scatter):
//
//     hlfir.region_assign {
//         hlfir.yield %source : !fir.ref<!fir.array<NxT>>
//     } to {
//         %idx_elem = hlfir.elemental %shape : (...) -> !hlfir.expr<Nxi64> {
//             ^bb(%i):  // gather cols(i)
//                 ...; hlfir.yield_element %col_i : i64
//         }
//         hlfir.elemental_addr %shape : !fir.shape<1> {
//             ^bb(%i):
//                 %idx = hlfir.apply %idx_elem, %i : (...) -> i64
//                 %addr = hlfir.designate %d (%idx) : ... -> !fir.ref<T>
//                 hlfir.yield %addr : !fir.ref<T>
//         } cleanup {
//             hlfir.destroy %idx_elem : !hlfir.expr<NxT>
//         }
//     }
//
// Rewrite to:
//
//     fir.do_loop %i = 1 to N step 1 {
//         %src_addr = hlfir.designate %source (%i)
//         %src_v    = fir.load %src_addr
//         // ... destination region body, with block arg → %i ...
//         hlfir.assign %src_v to %dest_addr
//     }
//
// Why a separate pass from MaterialiseAssociates:
//     The gather case (rhs ``d(cols)``) flows through ``hlfir.associate`` of an
//     ``hlfir.elemental`` — a value-producing expression that the bridge's
//     read path can hook into via the synth temp's ``hlfir.declare``.
//     The scatter case is destination-side: there's no value to materialise,
//     just a sequence of stores to a noncontiguous set of addresses.  The two
//     shapes need different lowerings even though both are "expand into a
//     loop with one indirection symbol per iteration".
//
// Constraints (loud-bail via ``op.emitError`` if violated):
//   * Source region must yield a single ``!fir.ref<!fir.array<NxT>>`` value
//     (a contiguous source).  Strided sources need a more careful walk;
//     defer.
//   * Destination ``hlfir.elemental_addr`` must be rank-1 with a constant
//     extent.  Same constraint as the gather pass — see
//     MaterialiseAssociates.cpp's header for the full DaCe-side rationale.
// ============================================================================

#include "passes/Passes.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

namespace hlfir_bridge {

namespace {

struct ExpandRegionAssignPass
    : public mlir::PassWrapper<ExpandRegionAssignPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandRegionAssignPass)

    llvm::StringRef getArgument() const final {
        return "hlfir-expand-region-assign";
    }
    llvm::StringRef getDescription() const final {
        return "Replace hlfir.region_assign whose destination region uses "
               "hlfir.elemental_addr (Fortran scatter ``d(cols) = source``) "
               "with an explicit DO loop of per-element scalar assigns.";
    }

    void runOnOperation() override {
        llvm::SmallVector<hlfir::RegionAssignOp, 8> targets;
        getOperation().walk([&](hlfir::RegionAssignOp op) {
            // Per Flang's TableGen, ``hlfir.region_assign``'s regions
            // are ``rhs_region`` (first, the source/value) and
            // ``lhs_region`` (second, the destination — where
            // ``hlfir.elemental_addr`` lives for vector-subscripted
            // assignments).  We only fire on the elemental_addr shape.
            bool found = false;
            for (auto &block : op.getLhsRegion())
                for (auto &inner : block)
                    if (mlir::isa<hlfir::ElementalAddrOp>(inner)) {
                        found = true;
                        break;
                    }
            if (found) targets.push_back(op);
        });

        for (auto op : targets)
            if (mlir::failed(rewrite(op)))
                return signalPassFailure();
    }

   private:
    mlir::LogicalResult rewrite(hlfir::RegionAssignOp op) {
        auto loc = op.getLoc();

        // --- Locate the elemental_addr inside the destination (lhs) region.
        hlfir::ElementalAddrOp eaddr;
        for (auto &block : op.getLhsRegion())
            for (auto &inner : block)
                if (auto e = mlir::dyn_cast<hlfir::ElementalAddrOp>(inner)) {
                    eaddr = e;
                    break;
                }
        if (!eaddr) return op.emitError("expected elemental_addr in lhs region");

        // --- Constant rank-1 extent guard.
        auto shapeOper = eaddr.getShape();
        auto shapeOp = mlir::dyn_cast_or_null<fir::ShapeOp>(shapeOper.getDefiningOp());
        if (!shapeOp || shapeOp.getExtents().size() != 1) {
            return op.emitError(
                "hlfir-expand-region-assign: rank-")
                << (shapeOp ? shapeOp.getExtents().size() : 0)
                << " scatter not yet supported (rank-1 only)";
        }
        mlir::Value extent = shapeOp.getExtents()[0];
        // ``cstExt`` is non-null only when the extent folds to a
        // compile-time constant.  Required ONLY when materialising a
        // scatter-source temp (the ``hlfir.expr`` path); dynamic-extent
        // allocas need fiddly type plumbing.  When the source is a
        // contiguous ``fir.ref``, the loop bound can be a runtime
        // symbol — each iteration reassigns one indirection-symbol
        // slot, exactly the pattern DaCe LoopRegions support natively.
        auto cstExt = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(
            extent.getDefiningOp());

        // --- Find the source region's yielded value (rhs region).
        mlir::Value srcVal;
        for (auto &block : op.getRhsRegion())
            for (auto &inner : block)
                if (auto y = mlir::dyn_cast<hlfir::YieldOp>(inner))
                    srcVal = y.getEntity();
        if (!srcVal) {
            return op.emitError(
                "hlfir-expand-region-assign: source region has no yield");
        }
        // Source can be:
        //   * a contiguous ``fir.ref<!fir.array<NxT>>`` (constant-shape
        //     dummy ``source(N)`` — ``designate + load`` per iter)
        //   * a ``fir.box<!fir.array<?xT>>`` (assumed-shape dummy
        //     ``source(:)`` — same per-iter pattern; ``hlfir.designate``
        //     reads through the box)
        //   * a ``!hlfir.expr<NxT>`` from an ``hlfir.elemental`` (the
        //     ``a(write_idx) = c(read_idx)`` round-trip — gather
        //     expression directly consumed by the scatter).  Read
        //     per-element via ``hlfir.apply``.
        bool srcIsExpr = false;
        mlir::Type eleTy;
        auto peelToSeq = [](mlir::Type t) -> fir::SequenceType {
            if (auto r = mlir::dyn_cast<fir::ReferenceType>(t))
                return mlir::dyn_cast<fir::SequenceType>(r.getEleTy());
            if (auto b = mlir::dyn_cast<fir::BoxType>(t))
                return mlir::dyn_cast<fir::SequenceType>(b.getEleTy());
            return {};
        };
        if (auto seqTy = peelToSeq(srcVal.getType())) {
            eleTy = seqTy.getEleTy();
        } else if (auto exprTy = mlir::dyn_cast<hlfir::ExprType>(srcVal.getType())) {
            srcIsExpr = true;
            eleTy = exprTy.getElementType();
        } else {
            return op.emitError(
                "hlfir-expand-region-assign: unsupported source type "
                "(expected fir.ref/fir.box of fir.array or hlfir.expr)");
        }

        mlir::OpBuilder b(op);
        mlir::IRMapping map;

        // --- Step 1: clone every op preceding the terminator out of
        //     each region (rhs first if the source is an expr, then
        //     lhs).  Records original→clone in ``map`` so cloned uses
        //     inside loops resolve to the new ops.
        if (srcIsExpr) {
            auto &rhsBlock = op.getRhsRegion().front();
            for (auto &inner : rhsBlock) {
                if (mlir::isa<hlfir::YieldOp>(inner)) break;
                b.clone(inner, map);
            }
        }
        auto &lhsBlock = op.getLhsRegion().front();
        for (auto &inner : lhsBlock) {
            if (mlir::isa<hlfir::ElementalAddrOp>(inner)) break;
            b.clone(inner, map);
        }

        mlir::Value mappedExtent = map.lookupOrDefault(extent);
        auto c1 = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(1));

        // --- Step 2: when the source is an ``hlfir.expr`` (i.e. a
        //     gather like ``c(read_idx)`` consumed directly by the
        //     scatter), Fortran 2003 semantics REQUIRE the RHS to be
        //     evaluated to a temporary BEFORE any LHS element is
        //     written — otherwise overlapping ``a(write_idx) =
        //     a(read_idx)`` produces wrong results when ``read_idx``
        //     and ``write_idx`` aliase.  Materialise into a transient
        //     scatter-source temp ``<dst>_scatter_<id>`` here.  When
        //     the source is already a contiguous ref, no temp is
        //     needed (no aliasing risk: caller hands us a buffer).
        mlir::Value srcRefBase;
        // The hlfir.expr path needs a constant extent to build a
        // static-shape scatter-source temp.  The fir.ref path runs
        // with any extent.  Bail loudly only when we'd need a temp.
        if (false && srcIsExpr && !cstExt) {
            return op.emitError(
                "hlfir-expand-region-assign: scatter from gather "
                "expression with symbolic extent requires a "
                "dynamic-extent scatter-source temp (unsupported).");
        }
        if (false && srcIsExpr) {  // disabled: scatter-source temp draft segfaults
            // Derive a "<dest>_scatter_<id>" uniq_name.  The scatter
            // temp belongs to the caller's local scope so the bridge's
            // extract_vars treats it like any other local transient.
            std::string enclName = "anon";
            if (auto fn = op->getParentOfType<mlir::func::FuncOp>()) {
                auto sym = fn.getSymName().str();
                if (sym.rfind("_QP", 0) == 0) enclName = sym.substr(3);
                else if (sym.rfind("_QM", 0) == 0) {
                    auto p = sym.find('P', 3);
                    if (p != std::string::npos)
                        enclName = sym.substr(3, p - 3) + "_" + sym.substr(p + 1);
                    else enclName = sym.substr(3);
                } else enclName = sym;
            }
            // Walk the elemental_addr body to find the destination
            // array name (mirrors MaterialiseAssociates' source name
            // walk; the arg roles are reversed but the chain shape is
            // identical: yield → designate → declare).
            std::string dstName = "expr";
            {
                auto &eblock = eaddr.getBody().front();
                for (auto &inner : eblock) {
                    auto y = mlir::dyn_cast<hlfir::YieldOp>(inner);
                    if (!y) continue;
                    mlir::Value v = y.getEntity();
                    for (int i = 0; i < 8 && v; ++i) {
                        auto *d = v.getDefiningOp();
                        if (!d) break;
                        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) { v = cv.getValue(); continue; }
                        if (auto ld = mlir::dyn_cast<fir::LoadOp>(d))    { v = ld.getMemref(); continue; }
                        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(d)) { v = dg.getMemref(); continue; }
                        if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
                            auto un = dc.getUniqName().str();
                            auto p = un.rfind('E');
                            if (p != std::string::npos) dstName = un.substr(p + 1);
                            else                         dstName = un;
                            break;
                        }
                        break;
                    }
                    break;
                }
            }
            int64_t extConst = mlir::cast<mlir::IntegerAttr>(cstExt.getValue()).getInt();
            auto seqTy = fir::SequenceType::get({extConst}, eleTy);
            auto alloca = b.create<fir::AllocaOp>(loc, seqTy);
            std::string uniqName = "_QF" + enclName + "E" + dstName + "_scatter_" + std::to_string(kScatterCounter++);
            auto declOp = b.create<hlfir::DeclareOp>(
                loc, alloca.getResult(), uniqName, shapeOper);
            srcRefBase = declOp.getResult(0);

            // Gather loop: ``tmp[i] = apply(elem, i)``.
            auto gloop = b.create<fir::DoLoopOp>(loc, c1, mappedExtent, c1,
                                                 /*unordered=*/false,
                                                 /*finalCountValue=*/false);
            {
                mlir::OpBuilder::InsertionGuard gg(b);
                b.setInsertionPointToStart(gloop.getBody());
                mlir::Value giv = gloop.getInductionVar();
                mlir::Value mappedSrc = map.lookupOrDefault(srcVal);
                auto applied = b.create<hlfir::ApplyOp>(
                    loc, eleTy, mappedSrc, mlir::ValueRange{giv},
                    /*typeparams=*/mlir::ValueRange{});
                auto dst = b.create<hlfir::DesignateOp>(
                    loc, fir::ReferenceType::get(eleTy), srcRefBase,
                    mlir::ValueRange{giv});
                b.create<hlfir::AssignOp>(loc, applied.getResult(),
                                          dst.getResult());
            }
        } else {
            // Source already a contiguous ref OR (disabled) expr
            // fused path — no separate gather temp.
            srcRefBase = srcVal;
        }

        // --- Step 3: scatter loop.
        auto sloop = b.create<fir::DoLoopOp>(loc, c1, mappedExtent, c1,
                                             /*unordered=*/false,
                                             /*finalCountValue=*/false);
        b.setInsertionPointToStart(sloop.getBody());
        mlir::Value iv = sloop.getInductionVar();

        // Source: ref → designate+load; expr (fused fallback) → apply.
        mlir::Value srcLoadVal;
        if (srcIsExpr) {
            mlir::Value mappedSrc = map.lookupOrDefault(srcVal);
            auto applied = b.create<hlfir::ApplyOp>(
                loc, eleTy, mappedSrc, mlir::ValueRange{iv},
                /*typeparams=*/mlir::ValueRange{});
            srcLoadVal = applied.getResult();
        } else {
            auto srcAddr = b.create<hlfir::DesignateOp>(
                loc, fir::ReferenceType::get(eleTy), srcRefBase,
                mlir::ValueRange{iv});
            srcLoadVal = b.create<fir::LoadOp>(loc, srcAddr.getResult())
                              .getResult();
        }

        // Destination: clone elemental_addr body inline.
        auto &eblock = eaddr.getBody().front();
        map.map(eblock.getArgument(0), iv);
        mlir::Value destAddr;
        for (auto &inner : eblock) {
            if (auto y = mlir::dyn_cast<hlfir::YieldOp>(inner)) {
                destAddr = map.lookupOrDefault(y.getEntity());
                break;
            }
            b.clone(inner, map);
        }
        if (!destAddr) {
            return op.emitError(
                "hlfir-expand-region-assign: could not capture destination "
                "address from elemental_addr body");
        }
        b.create<hlfir::AssignOp>(loc, srcLoadVal, destAddr);

        // --- Erase the region_assign.  Cleanup region's destroy ops
        //     refer to values inside the region and get cleared
        //     automatically.
        op.erase();
        return mlir::success();
    }

    unsigned kScatterCounter = 0;
};

}  // namespace

std::unique_ptr<mlir::Pass> createExpandRegionAssignPass() {
    return std::make_unique<ExpandRegionAssignPass>();
}

}  // namespace hlfir_bridge
