// Translation-unit headers.  ``ast_helpers.h`` carries the cross-TU
// API + thread-local state shared with the other ``ast/*.cpp`` files.
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>
#include <iomanip>
#include <sstream>

#include "bridge/ast/ast_helpers.h"
#include "bridge/ast/ast_internal.h"

namespace hlfir_bridge {

//
// MERGE-libcall + buildElementalAssign + comparison primitives + scf.if
// helpers.  Owns:
//   * buildMergeLibcall — recognises Flang's hlfir.elemental shape
//     for MERGE(t, f, mask) and routes to MergeLibraryNode.
//   * buildElementalAssign — the general elemental walker (where
//     non-MERGE elementals land).
//   * cmpiPredStr / cmpfPredStr — Python-syntax predicate strings.
//   * buildExprWithSubscripts — like buildExpr but keeps explicit
//     a[i-1] subscripts (interstate-edge condition mode).
//   * buildBoolExpr — Python bool expression for arith.cmp* /
//     andi/ori/xori chains, used by both elemental walks and conditionals.
//   * scfSynthName / isScfIfResult / yieldedExpr — helpers for
//     the synthetic-scalar scf.if-result machinery.
//
// This file is included verbatim from extract_ast.cpp via
// #include "bridge/ast/control_flow.cpp" and shares that translation
// unit's namespace, includes, and file-static state.  It MUST NOT be
// added to the build's compile list — CMakeLists.txt deliberately omits
// it.  The split is purely for readability: the AST builder used to
// be a single 2800-line file.
std::vector<ASTNode>
buildMergeLibcall(hlfir::AssignOp assign, hlfir::ElementalOp elem) {
    auto &region = elem.getRegion();
    if (region.empty()) return {};
    auto &block = region.front();

    // Find the yield_element and confirm its operand is an arith.select.
    mlir::Value yielded;
    for (auto &op : block)
        if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op))
            { yielded = y.getElementValue(); break; }
    if (!yielded) return {};
    auto sel = mlir::dyn_cast_or_null<mlir::arith::SelectOp>(yielded.getDefiningOp());
    if (!sel) return {};

    // Each of the three operands must trace back to a fir.load of an
    // hlfir.designate of an hlfir.declare.  ``fir.convert`` wrappers
    // (e.g. ``logical<4> → i1`` for the mask) are transparent.  Bail
    // on anything more elaborate (those go through the generic
    // per-element tasklet path via ``buildElementalAssign``).
    // Operands can be:
    //   * ``fir.load %designate`` — array element (Flang's array path)
    //   * ``fir.load %declare``   — scalar dummy (Flang hoists scalar
    //                                loads outside the elemental for
    //                                broadcast variants 3, 4, 5)
    // Either form resolves to a declared array / scalar by name; the
    // library node's expansion later introspects the incoming memlet's
    // subset to decide per-operand whether to broadcast.
    auto traceLoadSource = [](mlir::Value v) -> std::string {
        // Walk through any fir.convert wrappers at the top.
        for (int i = 0; i < 8; ++i) {
            auto *op = v.getDefiningOp();
            if (!op) return "";
            auto cv = mlir::dyn_cast<fir::ConvertOp>(op);
            if (!cv) break;
            v = cv.getValue();
        }
        auto *op = v.getDefiningOp();
        if (!op) return "";
        auto ld = mlir::dyn_cast<fir::LoadOp>(op);
        if (!ld) return "";
        auto *md = ld.getMemref().getDefiningOp();
        if (!md) return "";
        if (mlir::isa<hlfir::DesignateOp>(md)
            || mlir::isa<hlfir::DeclareOp>(md))
            return traceToDecl(ld.getMemref());
        return "";
    };

    std::string mask_name = traceLoadSource(sel.getCondition());
    std::string t_name    = traceLoadSource(sel.getTrueValue());
    std::string f_name    = traceLoadSource(sel.getFalseValue());
    if (mask_name.empty() || t_name.empty() || f_name.empty()) return {};

    ASTNode lib;
    lib.kind = "libcall";
    lib.callee = "merge";
    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp())
        if (auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            lib.target = extractName(declOp.getUniqName().str());
    if (lib.target.empty()) lib.target = traceToDecl(dest);
    lib.target_is_array = isArrayRef(dest.getType());
    // MergeLibraryNode connector order: ``_t``, ``_f``, ``_mask``.
    lib.call_args.push_back(t_name);
    lib.call_args.push_back(f_name);
    lib.call_args.push_back(mask_name);
    return {std::move(lib)};
}

/// ``b = elementwise_expr(a)`` — the ``hlfir.assign``'s source is an
/// ``hlfir.elemental``.  Synthesise one ``kind="loop"`` ASTNode per shape
/// dimension (synthetic iter names ``ei0``, ``ei1``, …) wrapping a single
/// ``kind="assign"`` child whose RHS is the elemental's body expression
/// with the block args replaced by the synthetic iter names.
///
/// ``buildExpr`` consults ``indexStack()`` to resolve an elemental block
/// arg to its synthetic name, so the inner ``buildAssignNode``-style walk
/// sees ``a[ei0]`` etc. as a normal array read with a normal iter var.
std::vector<ASTNode>
buildElementalAssign(hlfir::AssignOp assign, hlfir::ElementalOp elem) {
    // Target array (LHS of the assign).
    ASTNode inner;
    inner.kind = "assign";
    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp())
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            inner.target = allocAliasFor(extractName(decl.getUniqName().str()));
    if (inner.target.empty()) inner.target = traceToDecl(dest);
    inner.target_is_array = true;

    // Synthetic iter names for each shape dimension.
    auto shape = elem.getShape();
    auto &region = elem.getRegion();
    if (region.empty()) return {};
    auto &block = region.front();
    unsigned rank = block.getNumArguments();

    std::vector<std::string> iter_names;
    iter_names.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
        iter_names.push_back("ei" + std::to_string(i));

    // Push block-arg → synthetic-name pairs so resolveIndex sees them
    // everywhere we walk the body.
    unsigned pushed = 0;
    for (unsigned i = 0; i < rank; ++i) {
        indexStack().push_back({block.getArgument(i), iter_names[i]});
        ++pushed;
    }

    // Inner write access: target[<per-array-dim index>].  When the
    // destination is a section designate ``a(lo:hi)`` we need to add
    // ``(lo - 1)`` to each triplet-iter so the write lands on the right
    // element of the root array (mirrors the same logic in
    // ``buildDesignateIndexExpr`` for reads through nested section
    // designates).  When the designate mixes triplet + scalar dims
    // (e.g. ``res(:, pos(1)+2) = input1 + input2``) the per-array-dim
    // index list is one-per-dim of the underlying array — triplet dims
    // contribute their ``ei_<tDim>`` iter, scalar dims contribute the
    // scalar's Fortran 1-based index expression.  The elemental's rank
    // matches the triplet count, NOT the underlying array's rank.
    //
    // ``LowerAdj`` keeps the constant-fold and symbolic-fallback paths
    // in one place: ``expr`` non-empty → ``(iter + expr - 1)`` form;
    // ``value != 0`` → integer offset; both empty / zero → bare iter.
    struct LowerAdj {
        int64_t value = 0;
        std::string expr;
    };
    AccessInfo wa;
    wa.array_name = inner.target;
    wa.is_write = true;
    if (auto dd = dest.getDefiningOp()) {
        if (auto dstDg = mlir::dyn_cast<hlfir::DesignateOp>(dd)) {
            auto triplets = dstDg.getIsTriplet();
            auto idxOps = dstDg.getIndices();
            unsigned cursor = 0;
            unsigned tDim = 0;
            for (bool isT : triplets) {
                if (isT && tDim < rank && cursor + 3 <= idxOps.size()) {
                    LowerAdj adj;
                    if (auto lo = traceConstInt(idxOps[cursor])) {
                        adj.value = *lo - 1;
                    } else {
                        auto loExpr = buildIndexExpr(idxOps[cursor], 0);
                        if (!loExpr.empty() && loExpr != "?")
                            adj.expr = std::move(loExpr);
                    }
                    std::string ix = iter_names[tDim];
                    if (!adj.expr.empty())
                        ix = "(" + ix + " + " + adj.expr + " - 1)";
                    else if (adj.value > 0)
                        ix = "(" + ix + " + " + std::to_string(adj.value) + ")";
                    else if (adj.value < 0)
                        ix = "(" + ix + " - " + std::to_string(-adj.value) + ")";
                    wa.index_vars.push_back(iter_names[tDim]);
                    wa.index_exprs.push_back(std::move(ix));
                    cursor += 3;
                    tDim++;
                } else if (!isT && cursor < idxOps.size()) {
                    // Scalar dim — thread its (Fortran 1-based) index
                    // expression directly into the write memlet so the
                    // memlet rank matches the underlying array.
                    auto sc = buildIndexExpr(idxOps[cursor], 0);
                    if (sc.empty() || sc == "?") sc = "?";
                    wa.index_vars.push_back(sc);
                    wa.index_exprs.push_back(std::move(sc));
                    cursor += 1;
                } else {
                    // Defensive: skip bad cursor advance to avoid an
                    // infinite loop on malformed input.
                    cursor += isT ? 3 : 1;
                    if (isT) tDim++;
                }
            }
        } else {
            // Bare ``hlfir.declare`` — write across the elemental's full
            // rank (every dim is a triplet covering the array's extent).
            for (unsigned i = 0; i < rank; ++i) {
                wa.index_vars.push_back(iter_names[i]);
                wa.index_exprs.push_back(iter_names[i]);
            }
        }
    } else {
        for (unsigned i = 0; i < rank; ++i) {
            wa.index_vars.push_back(iter_names[i]);
            wa.index_exprs.push_back(iter_names[i]);
        }
    }
    inner.accesses.push_back(std::move(wa));

    // Walk the body's yield_element to produce the RHS string.
    mlir::Value yielded;
    for (auto &op : block)
        if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op))
            { yielded = y.getElementValue(); break; }

    // Pre-walk: any ``hlfir.apply <libcall_expr>`` we encounter inside
    // the elemental body needs the libcall (``hlfir.matmul`` /
    // ``hlfir.transpose`` / …) materialised into a real transient
    // BEFORE the elemental itself runs.  Without this, ``buildExpr``
    // sees an apply whose source isn't an inner elemental and returns
    // ``?``, producing tasklet bodies like ``2 - ?``.  Each libcall op
    // gets a unique ``_libtmp_<gid>`` name + a pair of pre-emitted AST
    // nodes (``declare_transient`` for the descriptor, ``libcall`` for
    // the runtime computation).  ``buildExpr`` then renders the apply
    // as a regular Fortran-style read of the transient.
    std::vector<ASTNode> preNodes;
    if (yielded) {
        std::function<void(mlir::Value, int)> findApplies =
            [&](mlir::Value v, int depth) {
            if (depth > 40 || !v) return;
            auto *op = v.getDefiningOp();
            if (!op) return;
            if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(op)) {
                auto src = apply.getExpr();
                if (auto *srcOp = src.getDefiningOp()) {
                    // Inner elemental → existing path inlines the body.
                    if (mlir::isa<hlfir::ElementalOp>(srcOp)) {
                        findApplies(src, depth + 1);
                        return;
                    }
                    // Recognised libcall expr-producer → materialise.
                    if (const char *callee = libcallNameForExprOp(srcOp)) {
                        if (!kHlfirExprToTransient.count(srcOp)) {
                            std::string tmp =
                                "_libtmp_" + std::to_string(kLibTmpCounter++);
                            kHlfirExprToTransient[srcOp] = tmp;

                            mlir::Type rty = srcOp->getResult(0).getType();
                            auto shape = exprResultShape(rty);

                            ASTNode decl;
                            decl.kind = "declare_transient";
                            decl.target = tmp;
                            decl.expr = exprDtypeString(rty);
                            decl.target_is_array = !shape.empty();
                            AccessInfo shapeInfo;
                            shapeInfo.array_name = tmp;
                            for (auto &s : shape)
                                shapeInfo.index_exprs.push_back(s);
                            decl.accesses.push_back(std::move(shapeInfo));
                            preNodes.push_back(std::move(decl));

                            ASTNode lib;
                            lib.kind = "libcall";
                            lib.target = tmp;
                            lib.target_is_array = !shape.empty();
                            lib.callee = callee;
                            for (auto operand : srcOp->getOperands()) {
                                auto n = traceToDecl(operand);
                                lib.call_args.push_back(n);
                            }
                            preNodes.push_back(std::move(lib));
                        }
                    }
                }
                return;
            }
            for (auto operand : op->getOperands())
                findApplies(operand, depth + 1);
        };
        findApplies(yielded, 0);
    }

    // Walk the yielded expression in tasklet-body mode: any embedded
    // comparisons (``a .eq. b`` inside a MERGE mask, etc.) must produce
    // bare names so emit_tasklet's per-occurrence connector wiring
    // matches.  Without this the bool path emits ``a[ei0-1] == b`` and
    // the bare-name path emits just ``a`` — same array, two different
    // forms in one tasklet body, which leaks ``ei0`` as a free symbol.
    {
        NoSubscriptGuard g;
        inner.expr = yielded ? buildExpr(yielded, 0) : "?";
    }

    // Read accesses.  Unlike plain assigns we must follow hlfir.apply into
    // the referenced hlfir.elemental's body (where the real designate
    // lives) — pushing the apply's index mapping onto indexStack() so the
    // designate sees the same synthetic iter names as the outer elemental.
    if (yielded) {
        // Per-occurrence AccessInfo (depth-limited, no op-identity dedup).
        // emit_tasklet counts array-name regex occurrences in the RHS
        // string; shared SSA values (``x * x``) must yield matching
        // AccessInfo count or downstream wiring strands a connector.
        collectReadAccesses(yielded, inner.accesses, 0);
    }

    // Pop the stack frames we pushed.
    for (unsigned i = 0; i < pushed; ++i)
        indexStack().pop_back();

    // Wrap the inner assign in one ASTNode kind="loop" per rank.  The
    // outermost loop is the result; deeper loops live as its sole child.
    ASTNode current;
    current.kind = "assign";
    current = inner;
    for (int i = rank - 1; i >= 0; --i) {
        ASTNode wrap;
        wrap.kind = "loop";
        wrap.loop_iter = iter_names[i];
        wrap.loop_lower = 1;
        wrap.loop_bound = resolveExtent(shape, i);
        wrap.children.push_back(current);
        current = wrap;
    }
    // Prepend any libcall-result materialisations so the transient is
    // declared and populated before the elemental body reads it.
    if (!preNodes.empty()) {
        preNodes.push_back(std::move(current));
        return preNodes;
    }
    return {current};
}

/// Render an arith::cmpi predicate as a Python comparison operator.  Returns
/// an empty string for signed/unsigned variants we haven't wired up yet.
 std::string cmpiPredStr(mlir::arith::CmpIPredicate p) {
    using P = mlir::arith::CmpIPredicate;
    switch (p) {
    case P::slt: case P::ult: return "<";
    case P::sle: case P::ule: return "<=";
    case P::sgt: case P::ugt: return ">";
    case P::sge: case P::uge: return ">=";
    case P::eq:               return "==";
    case P::ne:               return "!=";
    }
    return "";
}

/// Render an arith::cmpf predicate as a Python comparison operator.  Ordered
/// and unordered predicates both collapse to the same Python operator; NaN
/// handling is beyond what a Python condition string can express, so we
/// accept the lossy mapping.
 std::string cmpfPredStr(mlir::arith::CmpFPredicate p) {
    using P = mlir::arith::CmpFPredicate;
    switch (p) {
    case P::OLT: case P::ULT: return "<";
    case P::OLE: case P::ULE: return "<=";
    case P::OGT: case P::UGT: return ">";
    case P::OGE: case P::UGE: return ">=";
    case P::OEQ: case P::UEQ: return "==";
    case P::ONE: case P::UNE: return "!=";
    default:                  return "";
    }
}

/// Like ``buildExpr`` but keeps explicit array subscripts (``a[(i) - 1]``)
/// when the value is a ``fir.load`` of a ``hlfir.designate``.  Used by
/// ``buildBoolExpr`` so interstate-edge conditions can reference array
/// elements directly — they're evaluated in the caller's frame, not by a
/// tasklet, so they can't rely on memlet-wired connectors.
 std::string buildExprWithSubscripts(mlir::Value val, int d) {
    if (d > limits::kBuildExprDepth || !val) return "?";
    // Output lands in an interstate-edge / ConditionalBlock condition,
    // parsed by DaCe's symbolic engine which can't handle the
    // ``dace.float32(...)`` precision wrap (treats ``dace`` as a free
    // symbol).  Suppress the wrap for the duration of this walk; the
    // f32-vs-f64 distinction doesn't change comparison outcomes.
    SuppressFloatCastGuard floatCastGuard;
    auto *def = val.getDefiningOp();
    if (!def) return "?";

    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def))
        return buildExprWithSubscripts(conv.getValue(), d + 1);

    // fir.load of hlfir.designate: emit 0-based subscripts.
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
        auto mem = ld.getMemref();
        if (auto md = mem.getDefiningOp())
            if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(md)) {
                auto arr = traceToDecl(dg.getMemref());
                auto indices = dg.getIndices();
                if (arr.empty()) return "?";
                if (indices.empty()) return arr;
                std::string s = arr + "[";
                bool first = true;
                for (auto idx : indices) {
                    if (!first) s += ", ";
                    s += "(" + buildIndexExpr(idx, d + 1) + ") - 1";
                    first = false;
                }
                s += "]";
                return s;
            }
    }

    // Binary arith — recurse through the subscript-aware builder.
    static const std::map<llvm::StringRef, std::string> bin_ops = {
        {"arith.mulf", " * "}, {"arith.addf", " + "},
        {"arith.subf", " - "}, {"arith.divf", " / "},
        {"arith.muli", " * "}, {"arith.addi", " + "},
        {"arith.subi", " - "}, {"arith.divsi", " // "}, {"arith.divui", " // "},
    };
    auto nm = def->getName().getStringRef();
    if (auto it = bin_ops.find(nm); it != bin_ops.end() && def->getNumOperands() == 2)
        return "(" + buildExprWithSubscripts(def->getOperand(0), d + 1) + it->second
                   + buildExprWithSubscripts(def->getOperand(1), d + 1) + ")";
    if (nm == "arith.negf" && def->getNumOperands() == 1)
        return "(-" + buildExprWithSubscripts(def->getOperand(0), d + 1) + ")";

    // Fall through to the plain expression builder for anything else
    // (constants, math intrinsics, …).
    return buildExpr(val, d + 1);
}

/// Build a Python-syntax boolean expression for an ``i1`` SSA value.
/// Recognises ``arith.cmpf``, ``arith.cmpi``, ``arith.andi``, ``arith.ori``
/// (used as boolean ops on i1), ``arith.xori`` (boolean xor / ``not`` pattern
/// ``xori %x, true``), and constant booleans.  Opaque inputs fall back to
/// ``buildExpr`` (which may still produce a usable Python expression for the
/// condition, or ``"?"`` when the shape isn't understood).
 std::string buildBoolExpr(mlir::Value val, int d) {
    if (d > limits::kBuildExprDepth) return "?";
    auto *def = val.getDefiningOp();
    if (!def) return "?";

    // Synthetic scalars for scf.if results.  The assignments we emit for
    // yielded values write 0/1 into them, so reading the name as-is is
    // semantically a bool.
    {
        auto it = kScfValueMap.find(val);
        if (it != kScfValueMap.end()) return it->second;
    }

    // ``fir.is_present %x : (!fir.ref<T>) -> i1`` — the runtime query
    // Flang emits for Fortran's ``present(x)`` on an OPTIONAL dummy.
    // ``lowerIsPresent`` (in expressions.inc) walks the operand back
    // through inlined declare aliases to one of: ``fir.absent`` → 0,
    // a host OPTIONAL dummy → its companion ``<name>_present`` symbol,
    // or a mandatory root → 1.
    if (auto isp = mlir::dyn_cast<fir::IsPresentOp>(def)) {
        auto e = lowerIsPresent(isp.getVal());
        if (!e.empty()) return e;
        return "?";
    }

    // fir.convert (i1 <-> i1 kind, i8 -> i1, …) and arith.trunci / extui
    // are transparent here — DaCe codegen treats any non-zero integer as
    // True inside a Python condition, so the cast is a no-op.
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def))
        return buildBoolExpr(conv.getValue(), d + 1);
    auto nm2 = def->getName().getStringRef();
    if (nm2 == "arith.trunci" || nm2 == "arith.extui" || nm2 == "arith.extsi") {
        if (def->getNumOperands() == 1)
            return buildBoolExpr(def->getOperand(0), d + 1);
    }

    // Pick the operand-renderer once for every leaf in this bool tree:
    // tasklet-body context (``kBoolExprNoSubscripts`` set via
    // ``NoSubscriptGuard`` by elemental walks, MERGE-of-scalars, or
    // the i1 ``andi`` / ``ori`` chain handler) wants bare identifiers
    // because emit_tasklet's regex rewrite later turns them into
    // ``_in_a_0`` connectors and wires subscripts through memlets.
    // Interstate-edge / IF-condition contexts (the default) want the
    // explicit ``arr[idx]`` form because the consumer is an expression
    // parser, not a tasklet rewrite.  ``leafExpr`` is reused by the
    // cmp branches AND the last-resort fall-through so every leaf
    // threads through the same rendering decision.
    bool bareNames = kBoolExprNoSubscripts;
    auto leafExpr = [bareNames](mlir::Value v, int d) -> std::string {
        return bareNames ? buildExpr(v, d) : buildExprWithSubscripts(v, d);
    };

    if (auto cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(def)) {
        auto pred = cmpfPredStr(cmp.getPredicate());
        if (pred.empty()) return "?";
        return "(" + leafExpr(cmp.getLhs(), d + 1) + " " + pred + " "
             + leafExpr(cmp.getRhs(), d + 1) + ")";
    }
    if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(def)) {
        auto pred = cmpiPredStr(cmp.getPredicate());
        if (pred.empty()) return "?";
        return "(" + leafExpr(cmp.getLhs(), d + 1) + " " + pred + " "
             + leafExpr(cmp.getRhs(), d + 1) + ")";
    }
    auto nm = def->getName().getStringRef();
    if (nm == "arith.andi" && def->getNumOperands() == 2)
        return "(" + buildBoolExpr(def->getOperand(0), d + 1) + " and "
                   + buildBoolExpr(def->getOperand(1), d + 1) + ")";
    if (nm == "arith.ori" && def->getNumOperands() == 2)
        return "(" + buildBoolExpr(def->getOperand(0), d + 1) + " or "
                   + buildBoolExpr(def->getOperand(1), d + 1) + ")";
    // ``xori %x, true`` is Flang's lowering of ``.not. x``.  Otherwise
    // boolean xor — Python has no operator, use ``!=``.
    if (nm == "arith.xori" && def->getNumOperands() == 2) {
        auto *rhsDef = def->getOperand(1).getDefiningOp();
        if (auto c = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(rhsDef))
            if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
                if (ia.getInt() == 1)
                    return "(not " + buildBoolExpr(def->getOperand(0), d + 1) + ")";
        return "(" + buildBoolExpr(def->getOperand(0), d + 1) + " != "
                   + buildBoolExpr(def->getOperand(1), d + 1) + ")";
    }
    if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
        if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
            return ia.getInt() ? "True" : "False";

    // Bool-tree leaf: any non-bool op reached at the bottom of the
    // recursion (typically a ``fir.load`` of an i1 / fir.logical) goes
    // through ``leafExpr`` so the operand-renderer choice (subscripted
    // vs bare) stays consistent across every leaf in this tree.
    return leafExpr(val, d + 1);
}

/// Faithful ``scf.while`` translator.
///
/// Rather than pattern-matching the shape ``lift-cf-to-scf`` produces, we
/// copy every structural op in the before-region into the AST one-for-one:
///
///   * ``scf.if`` (void)     → ``kind="conditional"`` with recursively walked arms.
///   * ``scf.if -> T``       → same, but we allocate a ``__sc_<id>`` synthetic
///                             int scalar per result; each arm ends with a
///                             ``kind="assign"`` writing the yielded value to
///                             that scalar so downstream reads of the result
///                             find a real SDFG data descriptor.
///   * ``scf.condition(%c)`` → ``if not (%c): break``.
///   * ``hlfir.assign``       → existing ``buildAssignNode`` path.
///
/// Pure-value ops (``arith.cmp*``, ``fir.load``, ``arith.xori``, ``arith.trunci``,
/// ``arith.extui``, ``fir.convert``, …) don't become AST nodes — their values
/// are inlined by ``buildExpr`` / ``buildBoolExpr`` when downstream ops read
/// them.
///
/// The synthetic-scalar trick means the whole translation is compositional:
/// every MLIR op maps to one SDFG primitive, no special cases for EXIT or
/// value-yielding scf.if nestings.  DaCe's IR-level simplification can
/// re-flatten the result if it wants to.
/// Synthetic scalar name for one scf.if result value.  Allocated on first
/// reference; subsequent references return the same name.  DaCe's side
/// auto-declares names starting with ``__sc_``.
std::string scfSynthName(mlir::Value v) {
    auto it = kScfValueMap.find(v);
    if (it != kScfValueMap.end()) return it->second;
    std::string s = "__sc_" + std::to_string(kScfValueCounter++);
    kScfValueMap[v] = s;
    return s;
}

static bool isScfIfResult(mlir::Value v) {
    auto *def = v.getDefiningOp();
    return def && mlir::isa<mlir::scf::IfOp>(def);
}

std::vector<ASTNode> walkSCFBeforeRegion(mlir::Block &block);

/// Helper: convert a yielded value to a string for writing into a synthetic
/// scalar.  scf.yield of an i32 constant / boolean / computed expression —
/// just reuse buildExpr, which traces through arith ops and cast chains.
std::string yieldedExpr(mlir::Value v) {
    auto s = buildExpr(v, 0);
    if (s == "?") s = buildBoolExpr(v, 0);
    return s;
}

}  // namespace hlfir_bridge
