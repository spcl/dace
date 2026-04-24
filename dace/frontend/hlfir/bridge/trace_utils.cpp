// ============================================================================
// trace_utils.cpp — Shared SSA tracing utilities
// ============================================================================

#include "bridge/trace_utils.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace hlfir_bridge {

std::string extractName(const std::string &m) {
    auto p = m.rfind('E');
    return p != std::string::npos ? m.substr(p + 1) : m;
}

std::string traceToDecl(mlir::Value val, int max) {
    for (int i = 0; i < max && val; ++i) {
        auto *d = val.getDefiningOp();
        if (!d) break;
        if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
            // Walk through inlined assumed-shape aliases to the outer
            // caller declare so downstream SDFG emission references
            // the real storage by its caller-side name.
            if (auto outer = asAssumedShapeAlias(dc)) {
                val = outer.getResult(0);
                continue;
            }
            return extractName(dc.getUniqName().str());
        }
        if (auto dc = mlir::dyn_cast<fir::DeclareOp>(d))
            return extractName(dc.getUniqName().str());
        if (auto c = mlir::dyn_cast<fir::ConvertOp>(d))
            { val = c.getValue(); continue; }
        if (auto l = mlir::dyn_cast<fir::LoadOp>(d))
            { val = l.getMemref(); continue; }
        if (auto co = mlir::dyn_cast<fir::CoordinateOp>(d))
            { val = co.getRef(); continue; }
        // Section / element designates (``a(lo:hi)``, ``a(i)``) — walk
        // through to the underlying memref so a reduce over an
        // ``hlfir.any %levmask(i_startblk:i_endblk, jk)`` resolves its
        // source array to ``levmask``.
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(d))
            { val = dg.getMemref(); continue; }
        if (auto s = mlir::dyn_cast<mlir::arith::SelectOp>(d))
            { val = s.getTrueValue(); continue; }
        break;
    }
    return "";
}

std::optional<int64_t> traceConstInt(mlir::Value v) {
    for (int i = 0; i < limits::kTraceConstIntMax; ++i) {
        auto *d = v.getDefiningOp();
        if (!d) break;
        if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(d))
            if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
                return ia.getInt();
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d))
            { v = cv.getValue(); continue; }
        // Flang wraps each static extent in a `select extent>0, extent, 0`
        // clamp; follow the true branch to reach the original value.
        if (auto s = mlir::dyn_cast<mlir::arith::SelectOp>(d))
            { v = s.getTrueValue(); continue; }
        break;
    }
    return std::nullopt;
}

hlfir::DeclareOp asAssumedShapeAlias(hlfir::DeclareOp decl) {
    // Signature: no shape operand, memref produced by a fir.convert
    // whose input (possibly behind more converts) is another
    // hlfir.declare.  This is precisely what Flang emits for the
    // callee's dummy_scope declare after hlfir-inline-all splices the
    // callee's body into the caller.
    if (decl.getShape()) return {};
    auto mr = decl.getMemref();
    for (int i = 0; i < limits::kAliasMemrefWalkDepth && mr; ++i) {
        auto *d = mr.getDefiningOp();
        if (!d) break;
        if (auto outer = mlir::dyn_cast<hlfir::DeclareOp>(d))
            return outer;
        if (auto conv = mlir::dyn_cast<fir::ConvertOp>(d)) {
            mr = conv.getValue();
            continue;
        }
        break;
    }
    return {};
}

std::vector<std::optional<int64_t>> declareLowerBounds(hlfir::DeclareOp decl) {
    std::vector<std::optional<int64_t>> lbs;
    auto shape = decl.getShape();
    if (!shape) return lbs;
    auto *def = shape.getDefiningOp();
    if (!def) return lbs;
    if (auto sh = mlir::dyn_cast<fir::ShapeOp>(def)) {
        // Plain fir.shape: every dim defaults to lbound=1.
        for (unsigned i = 0; i < sh.getExtents().size(); ++i)
            lbs.push_back(std::optional<int64_t>(1));
        return lbs;
    }
    if (auto ss = mlir::dyn_cast<fir::ShapeShiftOp>(def)) {
        // shape_shift operands alternate: lb0, ext0, lb1, ext1, ...
        auto ops = ss->getOperands();
        for (unsigned i = 0; i < ops.size(); i += 2)
            lbs.push_back(traceConstInt(ops[i]));
        return lbs;
    }
    return lbs;
}

llvm::SmallVector<mlir::Value, 4> extractExtents(mlir::Value shape) {
    llvm::SmallVector<mlir::Value, 4> result;
    if (!shape) return result;
    auto *def = shape.getDefiningOp();
    if (!def) return result;

    if (auto sh = mlir::dyn_cast<fir::ShapeOp>(def))
        for (auto e : sh.getExtents()) result.push_back(e);

    if (auto ss = mlir::dyn_cast<fir::ShapeShiftOp>(def)) {
        // shape_shift: alternating (lb, extent) — we want odd indices.
        auto ops = ss->getOperands();
        for (unsigned i = 1; i < ops.size(); i += 2)
            result.push_back(ops[i]);
    }

    return result;
}

}  // namespace hlfir_bridge
