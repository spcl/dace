// ============================================================================
// extract_vars.cpp — Collect and classify every hlfir.declare.
// ============================================================================

#include "bridge/extract_vars.h"
#include "bridge/trace_utils.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
            if (!n.empty()) { syms.push_back(n); continue; }
            if (auto c = traceConstInt(ext)) {
                syms.push_back(std::to_string(*c));
                continue;
            }
            syms.push_back("?");
        }

    if (auto ss = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shape.getDefiningOp())) {
        auto ops = ss->getOperands();
        for (unsigned i = 1; i < ops.size(); i += 2) {
            auto n = traceToDecl(ops[i]);
            if (!n.empty()) { syms.push_back(n); continue; }
            if (auto c = traceConstInt(ops[i])) {
                syms.push_back(std::to_string(*c));
                continue;
            }
            syms.push_back("?");
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

/// Walk the defining-op graph backwards from a control-flow condition
/// Value, collecting the Fortran names of every ``hlfir.declare``'d
/// scalar that feeds into it.  Used by pass 2d to promote loop /
/// branch-condition scalars to symbols.
///
/// Recognised shape: ``arith.cmp*`` (the leaf comparison), and the
/// transparent wrappers lift-cf-to-scf emits around it —
/// ``arith.xori/andi/ori/trunci/extui/extsi`` and ``fir.convert``.
/// Stops when it hits a ``fir.load`` (hands off to ``traceToDecl``) or
/// an op it doesn't recognise.
static void collectConditionReads(mlir::Value v, std::set<std::string> &out,
                                  int depth = 0) {
    if (depth > 20 || !v) return;
    auto *def = v.getDefiningOp();
    if (!def) return;
    auto nm = def->getName().getStringRef();

    // Comparison leaves: recurse into both operands to catch both sides
    // of ``i < n`` (i and n both become symbols if they're declared).
    if (nm == "arith.cmpf" || nm == "arith.cmpi") {
        for (auto operand : def->getOperands())
            collectConditionReads(operand, out, depth + 1);
        return;
    }

    // Logical combinators and int casts the lift-cf-to-scf chain emits.
    if (nm == "arith.xori" || nm == "arith.andi" || nm == "arith.ori"
        || nm == "arith.trunci" || nm == "arith.extui" || nm == "arith.extsi"
        || nm == "fir.convert") {
        for (auto operand : def->getOperands())
            collectConditionReads(operand, out, depth + 1);
        return;
    }

    // Scalar read: trace to its declare; every op on the trace chain
    // (fir.load + hlfir.declare) resolves to the Fortran name.
    if (mlir::isa<fir::LoadOp>(def)) {
        auto n = traceToDecl(v);
        if (!n.empty()) out.insert(n);
        return;
    }

    // Anything else (constants, arith.addi used as index arithmetic, …)
    // — trace through traceToDecl as a last resort; it already handles
    // several pass-through ops.
    auto n = traceToDecl(v);
    if (!n.empty()) out.insert(n);
}

// ---------------------------------------------------------------------------
// Main extraction
// ---------------------------------------------------------------------------

std::vector<VarInfo> extractVariables(mlir::ModuleOp module) {
    std::vector<VarInfo> vars;

    // Pass 1: collect every hlfir.declare.
    std::vector<hlfir::DeclareOp> decls;
    module.walk([&](hlfir::DeclareOp op) { decls.push_back(op); });

    // Pass 2a: loop iterators.  A Fortran DO induction variable is
    // always a symbol downstream — the LoopRegion uses it as
    // ``loop_var`` in its init / update / condition expressions, and
    // any ``a(i)`` body uses it as an index (which only symbols may
    // be).  Add to symbolNames directly; there's no reason to keep a
    // separate ``loop_iter`` role when every consumer wants ``symbol``
    // semantics.
    std::set<std::string> symbolNames;
    module.walk([&](fir::DoLoopOp lp) {
        auto n = traceLoopIter(lp);
        if (!n.empty()) symbolNames.insert(n);
    });

    // Pass 2b: shape symbols + do-loop upper bounds.
    for (auto &op : decls)
        for (auto &s : resolveShapeSyms(op))
            if (s != "?") symbolNames.insert(s);
    module.walk([&](fir::DoLoopOp lp) {
        auto n = traceToDecl(lp.getUpperBound());
        if (!n.empty()) symbolNames.insert(n);
    });

    // Pass 2c: scalars used as array indices (``a(i)``) are also symbols.
    // Catches the DO-with-EXIT / DO-WHILE shape where lift-cf-to-scf
    // removed the fir.do_loop that pass 2a would otherwise trace, plus
    // any index-only scalar the user declares by hand.  Writing to a
    // symbol then routes through the interstate-edge path in
    // _emit_assign, which is the state-change DaCe needs to keep the
    // index value live across loop iterations.
    module.walk([&](hlfir::DesignateOp dg) {
        for (auto idx : dg.getIndices()) {
            auto n = traceToDecl(idx);
            if (!n.empty()) symbolNames.insert(n);
        }
    });

    // Pass 2d: scalars read by any control-flow condition are also
    // symbols.  Principle: loop variables, while-loop counters and
    // if-branch guards all go through the symbol / interstate-edge
    // write path so DaCe's condition evaluators see every update.
    // Without this, ``DO WHILE (i < n)`` reads the scalar's initial
    // zero-init and the loop body never runs.
    module.walk([&](mlir::scf::IfOp ifOp) {
        collectConditionReads(ifOp.getCondition(), symbolNames);
    });
    module.walk([&](fir::IfOp ifOp) {
        collectConditionReads(ifOp.getCondition(), symbolNames);
    });
    module.walk([&](mlir::scf::ConditionOp condOp) {
        collectConditionReads(condOp.getCondition(), symbolNames);
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
        else if (ty.isInteger(1))  v.dtype = "bool";
        else if (mlir::isa<fir::LogicalType>(ty)) v.dtype = "bool";
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
        else                                        v.role = "scalar";

        vars.push_back(std::move(v));
    }
    return vars;
}

}  // namespace hlfir_bridge
