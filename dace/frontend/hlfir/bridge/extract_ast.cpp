// ============================================================================
// extract_ast.cpp — Build a recursive statement tree from HLFIR.
// ============================================================================
// Statement-level ops become nodes; everything else is expression-level
// infrastructure and gets folded into the expression strings or access lists.
// ============================================================================

#include "bridge/extract_ast.h"
#include "bridge/trace_utils.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <map>
#include <set>
#include <sstream>

namespace hlfir_bridge {

// ---------------------------------------------------------------------------
// Expression reconstruction
// ---------------------------------------------------------------------------

/// Recursively build a Python-syntax expression string from an SSA value.
/// Depth-limited to 30 to prevent infinite recursion on malformed IR.
static std::string buildExpr(mlir::Value val, int d = 0) {
    if (d > 30) return "?";
    auto *def = val.getDefiningOp();
    if (!def) return "?";

    auto nm = def->getName().getStringRef();

    static const std::map<llvm::StringRef, std::string> ops = {
        {"arith.mulf", " * "}, {"arith.addf", " + "},
        {"arith.subf", " - "}, {"arith.divf", " / "},
        {"arith.muli", " * "}, {"arith.addi", " + "},
    };
    auto it = ops.find(nm);
    if (it != ops.end() && def->getNumOperands() == 2)
        return "(" + buildExpr(def->getOperand(0), d + 1)
                   + it->second
                   + buildExpr(def->getOperand(1), d + 1) + ")";

    if (nm == "arith.negf" && def->getNumOperands() == 1)
        return "(-" + buildExpr(def->getOperand(0), d + 1) + ")";

    if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
        auto mem = ld.getMemref();
        if (auto md = mem.getDefiningOp())
            if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(md))
                return traceToDecl(dg.getMemref());
        auto n = traceToDecl(mem);
        if (!n.empty()) return n;
    }

    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
        if (auto f = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
            std::ostringstream o; o << f.getValueAsDouble(); return o.str();
        }
        if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
            return std::to_string(i.getInt());
    }

    return "?";
}

/// Build a display expression for an index value.  Mirrors Fortran syntax
/// (1-based, square brackets for indirect access) so the Python side can
/// pattern-match on it.  Depth-limited to avoid loops on malformed IR.
static std::string buildIndexExpr(mlir::Value v, int d = 0) {
    if (d > 20 || !v) return "?";
    auto *def = v.getDefiningOp();
    if (!def) return "?";

    // fir.convert is transparent.
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def))
        return buildIndexExpr(conv.getValue(), d + 1);

    // A loaded scalar — either a named variable (loop iter) or an indirect
    // access via hlfir.designate on another array.
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
        auto mem = ld.getMemref();
        if (auto *md = mem.getDefiningOp()) {
            if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(md)) {
                auto arrName = traceToDecl(dg.getMemref());
                if (arrName.empty()) return "?";
                std::string s = arrName + "[";
                bool first = true;
                for (auto idx : dg.getIndices()) {
                    if (!first) s += ",";
                    s += buildIndexExpr(idx, d + 1);
                    first = false;
                }
                s += "]";
                return s;
            }
        }
        auto n = traceToDecl(mem);
        return n.empty() ? "?" : n;
    }

    // Constant integer.
    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
        if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
            return std::to_string(i.getInt());

    return "?";
}

// ---------------------------------------------------------------------------
// Per-statement builders
// ---------------------------------------------------------------------------

static ASTNode buildAssignNode(hlfir::AssignOp assign) {
    ASTNode node;
    node.kind = "assign";

    // --- LHS ---
    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp()) {
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(dd)) {
            node.target = traceToDecl(dg.getMemref());
            node.target_is_array = true;
            AccessInfo wa;
            wa.array_name = node.target;
            wa.is_write = true;
            for (auto idx : dg.getIndices()) {
                auto n = traceToDecl(idx);
                wa.index_vars.push_back(n.empty() ? "?" : n);
                wa.index_exprs.push_back(buildIndexExpr(idx));
            }
            node.accesses.push_back(std::move(wa));
        } else {
            node.target = traceToDecl(dest);
        }
    } else {
        node.target = traceToDecl(dest);
    }

    // --- RHS expression string ---
    auto src = assign.getOperand(0);
    node.expr = buildExpr(src);
    if (node.expr == "?") {
        if (auto d = src.getDefiningOp()) {
            if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(d)) {
                if (auto f = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
                    std::ostringstream o; o << f.getValueAsDouble();
                    node.expr = o.str();
                } else if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
                    node.expr = std::to_string(i.getInt());
            }
        }
    }

    // --- Collect RHS array reads ---
    std::set<mlir::Operation *> visited;
    std::function<void(mlir::Value)> collectReads = [&](mlir::Value v) {
        auto *op = v.getDefiningOp();
        if (!op || visited.count(op)) return;
        visited.insert(op);
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(op)) {
            AccessInfo ra;
            ra.array_name = traceToDecl(dg.getMemref());
            ra.is_read = true;
            for (auto idx : dg.getIndices()) {
                auto n = traceToDecl(idx);
                ra.index_vars.push_back(n.empty() ? "?" : n);
                ra.index_exprs.push_back(buildIndexExpr(idx));
                // Keep descending into the index operand too, so inner
                // indirect loads (edge_idx used below z_kin) are captured as
                // their own AccessInfo entries for extract_vars to see.
                collectReads(idx);
            }
            node.accesses.push_back(std::move(ra));
            return;
        }
        for (auto operand : op->getOperands())
            collectReads(operand);
    };
    collectReads(src);

    return node;
}

static int64_t traceLB(mlir::Value v) {
    if (auto c = traceConstInt(v)) return *c;
    return -1;
}

static std::string traceLoopIter(fir::DoLoopOp loop) {
    for (auto &op : loop.getRegion().front())
        if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
            auto n = traceToDecl(st.getMemref());
            if (!n.empty()) return n;
        }
    return "";
}

// ---------------------------------------------------------------------------
// Block walker
// ---------------------------------------------------------------------------

static std::vector<ASTNode> buildAST(mlir::Block &block) {
    std::vector<ASTNode> nodes;
    for (auto &op : block) {
        if (auto doLoop = mlir::dyn_cast<fir::DoLoopOp>(op)) {
            ASTNode n;
            n.kind = "loop";
            n.loop_iter  = traceLoopIter(doLoop);
            n.loop_bound = traceToDecl(doLoop.getUpperBound());
            if (n.loop_bound.empty()) {
                // Literal integer upper bound (e.g. DO jk = 1, 10) — fall back
                // to the constant value so downstream code doesn't see an
                // empty string.
                if (auto c = traceConstInt(doLoop.getUpperBound()))
                    n.loop_bound = std::to_string(*c);
            }
            n.loop_lower = traceLB(doLoop.getLowerBound());
            n.children   = buildAST(doLoop.getRegion().front());
            nodes.push_back(std::move(n));
            continue;
        }
        if (auto assign = mlir::dyn_cast<hlfir::AssignOp>(op)) {
            nodes.push_back(buildAssignNode(assign));
            continue;
        }
        if (auto ifOp = mlir::dyn_cast<fir::IfOp>(op)) {
            ASTNode n;
            n.kind = "conditional";
            n.condition = "?";
            if (!ifOp.getThenRegion().empty())
                n.children = buildAST(ifOp.getThenRegion().front());
            if (!ifOp.getElseRegion().empty())
                n.else_children = buildAST(ifOp.getElseRegion().front());
            nodes.push_back(std::move(n));
            continue;
        }
        if (auto call = mlir::dyn_cast<fir::CallOp>(op)) {
            ASTNode n;
            n.kind = "call";
            if (auto ref = call.getCallee()) {
                std::string s; llvm::raw_string_ostream os(s);
                ref->print(os); n.callee = s;
            }
            nodes.push_back(std::move(n));
            continue;
        }
    }
    return nodes;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

std::vector<ASTNode> extractAST(mlir::ModuleOp module) {
    std::vector<ASTNode> result;
    module.walk([&](mlir::func::FuncOp func) {
        if (!result.empty()) return;  // first func only
        if (!func.getBody().empty())
            result = buildAST(func.getBody().front());
    });
    return result;
}

}  // namespace hlfir_bridge
