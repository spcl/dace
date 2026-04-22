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
        if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d))
            return extractName(dc.getUniqName().str());
        if (auto dc = mlir::dyn_cast<fir::DeclareOp>(d))
            return extractName(dc.getUniqName().str());
        if (auto c = mlir::dyn_cast<fir::ConvertOp>(d))
            { val = c.getValue(); continue; }
        if (auto l = mlir::dyn_cast<fir::LoadOp>(d))
            { val = l.getMemref(); continue; }
        if (auto co = mlir::dyn_cast<fir::CoordinateOp>(d))
            { val = co.getRef(); continue; }
        if (auto s = mlir::dyn_cast<mlir::arith::SelectOp>(d))
            { val = s.getTrueValue(); continue; }
        break;
    }
    return "";
}

std::optional<int64_t> traceConstInt(mlir::Value v) {
    for (int i = 0; i < 10; ++i) {
        auto *d = v.getDefiningOp();
        if (!d) break;
        if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(d))
            if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
                return ia.getInt();
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d))
            { v = cv.getValue(); continue; }
        break;
    }
    return std::nullopt;
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
