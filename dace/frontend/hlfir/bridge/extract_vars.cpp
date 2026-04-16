// ============================================================================
// extract_vars.cpp — Collect and classify every hlfir.declare.
// ============================================================================

#include "bridge/extract_vars.h"
#include "bridge/trace_utils.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

#include <set>

namespace hlfir_bridge {

// ---------------------------------------------------------------------------
// Shape and lower-bound resolution for one declare.
// ---------------------------------------------------------------------------

/// Per-dim extent symbol names.  Resolution order:
///   1. hlfir_bridge.shape_hint attribute (populated by PropagateShapes)
///   2. fir.shape / fir.shape_shift operand (traced via SSA)
///   3. empty — caller fills with synthetics
static std::vector<std::string> resolveShapeSyms(hlfir::DeclareOp decl) {
    std::vector<std::string> syms;

    // (1) Check the attribute set by the shape-propagation pass.
    if (auto hint = decl->getAttrOfType<mlir::ArrayAttr>(kShapeHintAttr)) {
        for (auto a : hint) {
            auto s = mlir::cast<mlir::StringAttr>(a).str();
            syms.push_back(s.empty() ? "?" : s);
        }
        return syms;
    }

    // (2) Trace the shape operand.
    auto shape = decl.getShape();
    if (!shape) return syms;

    if (auto sh = mlir::dyn_cast_or_null<fir::ShapeOp>(shape.getDefiningOp()))
        for (auto ext : sh.getExtents()) {
            auto n = traceToDecl(ext);
            syms.push_back(n.empty() ? "?" : n);
        }

    if (auto ss = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shape.getDefiningOp())) {
        auto ops = ss->getOperands();
        for (unsigned i = 1; i < ops.size(); i += 2) {
            auto n = traceToDecl(ops[i]);
            syms.push_back(n.empty() ? "?" : n);
        }
    }

    return syms;
}

static std::vector<std::string> resolveLowerBounds(hlfir::DeclareOp decl) {
    std::vector<std::string> lbs;
    auto shape = decl.getShape();
    if (!shape) return lbs;

    if (auto sh = mlir::dyn_cast_or_null<fir::ShapeOp>(shape.getDefiningOp()))
        for (unsigned i = 0; i < sh.getExtents().size(); ++i)
            lbs.push_back("1");

    if (auto ss = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shape.getDefiningOp())) {
        auto ops = ss->getOperands();
        for (unsigned i = 0; i < ops.size(); i += 2) {
            if (auto c = traceConstInt(ops[i]))
                lbs.push_back(std::to_string(*c));
            else {
                auto n = traceToDecl(ops[i]);
                lbs.push_back(n.empty() ? "?" : n);
            }
        }
    }

    return lbs;
}

/// Find the fir.do_loop induction variable's Fortran name by looking for
/// `fir.store %block_arg, %alloca` in the loop body.
static std::string traceLoopIter(fir::DoLoopOp loop) {
    for (auto &op : loop.getRegion().front())
        if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
            auto n = traceToDecl(st.getMemref());
            if (!n.empty()) return n;
        }
    return "";
}

// ---------------------------------------------------------------------------
// Main extraction
// ---------------------------------------------------------------------------

std::vector<VarInfo> extractVariables(mlir::ModuleOp module) {
    std::vector<VarInfo> vars;

    // Pass 1: collect every hlfir.declare.
    std::vector<hlfir::DeclareOp> decls;
    module.walk([&](hlfir::DeclareOp op) { decls.push_back(op); });

    // Pass 2a: loop iterators.
    std::set<std::string> loopIters;
    module.walk([&](fir::DoLoopOp lp) {
        auto n = traceLoopIter(lp);
        if (!n.empty()) loopIters.insert(n);
    });

    // Pass 2b: symbol names (scalars appearing in shapes or loop bounds).
    std::set<std::string> symbolNames;
    for (auto &op : decls)
        for (auto &s : resolveShapeSyms(op))
            if (s != "?") symbolNames.insert(s);
    module.walk([&](fir::DoLoopOp lp) {
        auto n = traceToDecl(lp.getUpperBound());
        if (!n.empty()) symbolNames.insert(n);
    });

    // Pass 3: build one VarInfo per declare.
    for (auto &op : decls) {
        VarInfo v;
        v.mangled_name = op.getUniqName().str();
        v.fortran_name = extractName(v.mangled_name);

        // Intent
        if (auto a = op.getFortranAttrs()) {
            auto fa = *a;
            if (bitEnumContainsAny(fa, fir::FortranVariableFlagsEnum::intent_inout))
                v.intent = "inout";
            else if (bitEnumContainsAny(fa, fir::FortranVariableFlagsEnum::intent_in))
                v.intent = "in";
            else if (bitEnumContainsAny(fa, fir::FortranVariableFlagsEnum::intent_out))
                v.intent = "out";
        }

        // Unwrap FIR type wrappers to find element type + rank.
        auto ty = op.getResult(0).getType();
        if (auto b = mlir::dyn_cast<fir::BoxType>(ty)) ty = b.getEleTy();
        if (auto r = mlir::dyn_cast<fir::ReferenceType>(ty)) ty = r.getEleTy();
        if (auto h = mlir::dyn_cast<fir::HeapType>(ty)) ty = h.getEleTy();
        if (auto p = mlir::dyn_cast<fir::PointerType>(ty)) ty = p.getEleTy();
        if (auto seq = mlir::dyn_cast<fir::SequenceType>(ty)) {
            for (auto d : seq.getShape())
                if (d == fir::SequenceType::getUnknownExtent())
                    v.is_dynamic = true;
            v.rank = seq.getShape().size();
            ty = seq.getEleTy();
        }

        // Element type string.
        if (ty.isF64())            v.dtype = "float64";
        else if (ty.isF32())       v.dtype = "float32";
        else if (ty.isInteger(32)) v.dtype = "int32";
        else if (ty.isInteger(64)) v.dtype = "int64";
        else {
            std::string s; llvm::raw_string_ostream os(s);
            ty.print(os); v.dtype = s;
        }

        v.shape_symbols = resolveShapeSyms(op);
        v.lower_bounds  = resolveLowerBounds(op);

        // Assumed-shape fallback: synthesise per-dim symbol names.
        if (v.shape_symbols.empty() && v.rank > 0)
            for (int dim = 0; dim < v.rank; ++dim)
                v.shape_symbols.push_back(
                    v.fortran_name + "_d" + std::to_string(dim));

        // Classify.
        if (v.rank > 0)                             v.role = "array";
        else if (symbolNames.count(v.fortran_name)) v.role = "symbol";
        else if (loopIters.count(v.fortran_name))   v.role = "loop_iter";
        else                                        v.role = "scalar";

        vars.push_back(std::move(v));
    }
    return vars;
}

}  // namespace hlfir_bridge
