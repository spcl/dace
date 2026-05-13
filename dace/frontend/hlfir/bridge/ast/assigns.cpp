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
// Per-shape assign builders + small type / value helpers.  Owns:
//   * buildIndexExpr body (continuation from expressions.cpp's forward
//     declaration).
//   * buildAssignNode / buildCopyNode / buildMemsetNode /
//     buildLibCallNode  --  one ASTNode per Fortran assignment shape.
//   * buildSectionScalarAssign / buildSectionReduceAssign  --
//     loop-synthesis for arr(lo:hi) = scalar and section-reduce.
//   * Type / value helpers: peelWrappers, isArrayRef,
//     isConstantZero, traceLB, asSectionDesignate.
//
// This file is included verbatim from extract_ast.cpp via
// #include "bridge/ast/assigns.cpp" and shares that translation
// unit's namespace, includes, and file-static state.  It MUST NOT be
// added to the build's compile list  --  CMakeLists.txt deliberately omits
// it.  The split is purely for readability: the AST builder used to
// be a single 2800-line file.
 std::string buildIndexExpr(mlir::Value v, int d) {
    if (d > limits::kBuildIndexExprDepth || !v) return "?";

    // Block args (fir.do_loop induction, hlfir.elemental iter) have no
    // defining op  --  resolve via indexStack() first so inlined elemental
    // bodies that use the block arg directly as a designate index don't
    // fall through to "?".
    if (!v.getDefiningOp()) {
        auto resolved = resolveIndex(v);
        if (!resolved.empty()) return resolved;
        return "?";
    }
    auto *def = v.getDefiningOp();

    // fir.convert is transparent.
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def))
        return buildIndexExpr(conv.getValue(), d + 1);

    // ``hlfir.apply %elem, %i`` used as a designate index (e.g. the
    // gather elemental ``cols(arg2)`` produced for noncontiguous slice
    // arguments).  Inline the referenced elemental's body and recurse
    // on its yielded value so the index renders as ``cols[i]`` rather
    // than the bare iter name.  Mirrors the ``hlfir.apply`` handler in
    // ``buildExpr`` (expressions.cpp).
    if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(def)) {
        auto src = apply.getExpr();
        if (auto *sd = src.getDefiningOp())
            if (auto inner_elem = mlir::dyn_cast<hlfir::ElementalOp>(sd)) {
                auto &ireg = inner_elem.getRegion();
                if (!ireg.empty()) {
                    auto &iblock = ireg.front();
                    auto apply_idxs = apply.getIndices();
                    unsigned pushed = 0;
                    for (unsigned i = 0;
                         i < iblock.getNumArguments() && i < apply_idxs.size();
                         ++i) {
                        auto name = resolveIndex(apply_idxs[i]);
                        indexStack().push_back({iblock.getArgument(i), name});
                        ++pushed;
                    }
                    std::string result = "?";
                    for (auto &iop : iblock)
                        if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(iop)) {
                            result = buildIndexExpr(y.getElementValue(), d + 1);
                            break;
                        }
                    for (unsigned i = 0; i < pushed; ++i)
                        indexStack().pop_back();
                    return result;
                }
            }
    }

    // A loaded scalar  --  either a named variable (loop iter) or an indirect
    // access via hlfir.designate on another array.
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
        auto mem = ld.getMemref();
        if (auto *md = mem.getDefiningOp()) {
            if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(md)) {
                auto arrName = resolveIndex(dg.getMemref());
                if (arrName.empty()) arrName = traceToDecl(dg.getMemref());
                if (arrName.empty()) return "?";
                // Constant-indexed array element used as an index / bound
                // (Fortran ``pos(1):pos(2)`` / ``a(idx(1), j)`` / ...).
                //
                // Two lowerings are available:
                //
                //   (a) ``internPosSymbol``  --  mint a one-shot SDFG
                //       symbol ``__sym_<arr>_<n>`` and prepend a
                //       ``kind="symbol_init"`` AST node that the
                //       Python emitter stages as an interstate-edge
                //       load ``__sym_<arr>_<n> = <arr>[n-1]`` at
                //       SDFG entry.  Every memlet that uses the
                //       symbol is then a closed-form expression DaCe
                //       can simplify, instead of a data reference it
                //       can't represent in subset form.
                //
                //   (b) Fall through to the ``arr[idx]`` form below
                //        --  the per-occurrence indirect-symbol
                //       machinery (``collect_indirect`` /
                //       ``indirect_to_dace`` in
                //       ``builder/access.py``) mints a fresh
                //       ``<arr>_at<gid>`` symbol at the read site.
                //
                // (a) is cheaper (one symbol shared across uses) but
                // ONLY safe when the array's contents don't change
                // after SDFG entry  --  read-only ``parameter``
                // constants and ``intent(in)`` dummies fit; arrays
                // the kernel writes do NOT.  An entry-time read of
                // a not-yet-initialised local captures uninitialised
                // memory, and every subsequent use of the symbol
                // sees garbage.  long_tasklet_test exercises this:
                // ``ind%indices(:) = 1; ... arr(ind%indices(1))``  --
                // (a) loads ``ind_indices[0]`` BEFORE the body's
                // initialiser runs.
                //
                // Gate: if the source declare is the LHS of any
                // ``hlfir.assign`` in the enclosing function (walked
                // through chained ``hlfir.designate`` ops on the
                // LHS), treat it as mutable and fall through to (b).
                auto idxOps = dg.getIndices();
                bool allConstScalar = !idxOps.empty();
                std::vector<int64_t> consts;
                for (auto idx : idxOps) {
                    auto c = traceConstInt(idx);
                    if (!c) { allConstScalar = false; break; }
                    consts.push_back(*c);
                }
                if (allConstScalar && consts.size() == 1) {
                    bool isMutable = false;
                    if (auto srcDecl = mlir::dyn_cast_or_null<hlfir::DeclareOp>(
                            dg.getMemref().getDefiningOp())) {
                        if (auto func = srcDecl->getParentOfType<mlir::func::FuncOp>()) {
                            func.walk([&](hlfir::AssignOp aop) {
                                auto lhs = aop.getLhs();
                                for (int hop = 0; hop < 8 && lhs; ++hop) {
                                    if (lhs == srcDecl.getResult(0) ||
                                        lhs == srcDecl.getResult(1)) {
                                        isMutable = true;
                                        return;
                                    }
                                    auto *ld = lhs.getDefiningOp();
                                    if (!ld) break;
                                    if (auto innerDg = mlir::dyn_cast<hlfir::DesignateOp>(ld)) {
                                        lhs = innerDg.getMemref();
                                        continue;
                                    }
                                    break;
                                }
                            });
                        }
                    }
                    if (!isMutable)
                        return internPosSymbol(arrName, consts[0]);
                    // Mutable source  --  fall through.  The matching
                    // Python-side fix lives in
                    // ``builder/access.py::build_memlet_index``: the
                    // bare-iter-name fallback now defaults to the
                    // bridge-supplied ``index_exprs[dim]`` instead
                    // of ``index_vars[dim]``, so the ``<arr>[idx]``
                    // form rendered below survives all the way to
                    // the memlet without being clobbered by
                    // ``resolveIndex``'s whole-array-name fallback.
                }
                // Multi-dim constant-indexed reads (``pos(1, 2)``) are not
                // yet folded to a symbol; the legacy ``pos[0, 1]`` form
                // below still applies.
                // Non-constant index  --  render as a Fortran 1-based
                // subscript ``arr[idx, ...]``.  The Python emitter
                // converts this to DaCe 0-based form
                // (``arr[(idx) - offset_arr_d0, ...]``) at consumption
                // time  --  loop bounds via ``_fortran_subs_to_dace``,
                // memlets via ``indirect_to_dace``  --  keeping the bridge
                // output uniform regardless of which downstream context
                // ultimately consumes it.
                std::string s = arrName + "[";
                bool first = true;
                unsigned di = 0;
                for (auto idx : dg.getIndices()) {
                    if (!first) s += ",";
                    s += buildDesignateIndexExpr(dg, di, idx, d + 1);
                    first = false;
                    ++di;
                }
                s += "]";
                return s;
            }
        }
        auto n = traceToDecl(mem);
        if (!n.empty()) return n;
        // Last resort: maybe the load memref is the elemental's block arg
        // indirectly (unlikely, but guard).
        return "?";
    }

    // Inside an elemental body, an index value IS a tracked block arg.
    auto resolved = resolveIndex(v);
    if (!resolved.empty()) return resolved;

    // Constant integer.
    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
        if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
            return std::to_string(i.getInt());

    // ``fir.box_dims %arr, %dim``  --  runtime descriptor read on an
    // allocatable / pointer array.  Flang emits this for ``arr(:)``
    // section bounds: each ``:`` triplet's ``lo`` is ``box_dims#0``
    // (lower bound) and the implicit ``hi`` derives from
    // ``box_dims#0 + box_dims#1 - 1`` (extent).  Map back onto the
    // bridge's per-array shape / offset symbols so a whole-array
    // section ``arr(:)`` resolves to ``offset_<arr>_d<K> :
    // (offset_<arr>_d<K> + <arr>_d<K> - 1)``.
    //
    // Only fires for arrays without a visible ``fir.allocmem`` in the
    // module  --  local allocatables (``allocate(x(n))``) have one, and
    // ``extract_vars`` resolves their extent to the real Fortran scalar
    // (``n``) rather than minting an ``<arr>_d<K>`` symbol.  Emitting
    // the synth symbol here for those would surface a missing-symbol
    // KeyError at SDFG build time.  Top-level pointer / allocatable
    // companions produced by ``hlfir-flatten-structs`` (alloc happens
    // outside this scope) have no allocmem and DO carry the ``_d<K>``
    // symbol via ``extract_vars``'s assumed-shape branch.
    if (def->getName().getStringRef() == "fir.box_dims"
            && def->getNumOperands() >= 2) {
        auto resIdx = mlir::cast<mlir::OpResult>(v).getResultNumber();
        if (resIdx == 2) return "1";  // stride  --  bridge requires contiguous
        auto dimC = traceConstInt(def->getOperand(1));
        // Walk operand 0 back to a declare so we can name the array.
        mlir::Value arrayVal = def->getOperand(0);
        for (int hop = 0; hop < limits::kTraceToDeclMax && arrayVal; ++hop) {
            auto *adef = arrayVal.getDefiningOp();
            if (!adef) break;
            if (auto cv = mlir::dyn_cast<fir::ConvertOp>(adef))
                { arrayVal = cv.getValue(); continue; }
            if (auto ld = mlir::dyn_cast<fir::LoadOp>(adef))
                { arrayVal = ld.getMemref(); continue; }
            break;
        }
        auto arrName = traceToDecl(arrayVal);
        if (dimC && !arrName.empty()) {
            // Suppress when an allocmem in the module carries this
            // declare's expected name  --  that path keeps the legacy
            // bounds-fail fallback (``buildCopyNode`` whole-array copy).
            bool hasAllocmem = false;
            if (auto *adef = arrayVal.getDefiningOp()) {
                if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(adef)) {
                    std::string allocName = decl.getUniqName().str() + ".alloc";
                    if (auto mod = decl->getParentOfType<mlir::ModuleOp>()) {
                        mod.walk([&](fir::AllocMemOp a) {
                            if (hasAllocmem) return;
                            if (auto un = a.getUniqName())
                                if (un->str() == allocName)
                                    hasAllocmem = true;
                        });
                    }
                }
            }
            if (!hasAllocmem) {
                std::string suffix = "_d" + std::to_string(*dimC);
                if (resIdx == 0) return "offset_" + arrName + suffix;
                if (resIdx == 1) return arrName + suffix;
            }
        }
    }

    // Integer arithmetic used inside index expressions  --  Flang lowers
    // ``arr(..., nlev-1, ...)`` via ``arith.subi %nlev, %c1``, ``nlev+1``
    // via ``arith.addi``, etc.  Render parenthesised so downstream
    // ``build_memlet_index`` takes the closed-form expression branch.
    auto nm = def->getName().getStringRef();
    static const std::map<llvm::StringRef, std::string> int_bin = {
        {"arith.addi",  " + "},
        {"arith.subi",  " - "},
        {"arith.muli",  " * "},
        {"arith.divsi", " // "}, {"arith.divui", " // "},
    };
    if (auto it = int_bin.find(nm);
        it != int_bin.end() && def->getNumOperands() == 2) {
        return "(" + buildIndexExpr(def->getOperand(0), d + 1)
                   + it->second
                   + buildIndexExpr(def->getOperand(1), d + 1) + ")";
    }

    // ``MAX`` / ``MIN`` in a Fortran index / loop bound expression
    // (``do jk = MAX(3, nrdmax-2), nlev-4``).  Flang's HLFIR lowers
    // these to ``arith.maxsi`` / ``arith.minsi`` (signed integer) and
    // their unsigned twins.  Render as ``max(a, b)`` / ``min(a, b)``
    // -- the same form ``buildExpr`` uses, accepted by both interstate
    // edge assignments (sympy maps to ``Max`` / ``Min``) and the C++
    // codegen.
    static const std::map<llvm::StringRef, std::string> idx_minmax = {
        {"arith.maxsi", "max"}, {"arith.maxui", "max"},
        {"arith.minsi", "min"}, {"arith.minui", "min"},
    };
    if (auto it = idx_minmax.find(nm);
        it != idx_minmax.end() && def->getNumOperands() == 2) {
        return it->second + "("
             + buildIndexExpr(def->getOperand(0), d + 1) + ", "
             + buildIndexExpr(def->getOperand(1), d + 1) + ")";
    }

    // Older flang lowerings (and integer MAX / MIN on some kinds) emit
    // the cmp+select idiom rather than ``arith.maxsi``:
    //
    //   %cmp = arith.cmpi sgt, %a, %b
    //   %r   = arith.select %cmp, %a, %b   ; MAX(a, b)
    //
    // The semantic mapping depends on BOTH the predicate AND which
    // operand of the comparison is selected on the true side -- flang
    // sometimes canonicalises ``cmpi sgt %a %b ... select %cmp %a %b``
    // into ``cmpi slt %b %a ... select %cmp %a %b`` (same MAX semantics
    // but the predicate flipped).  Render directly on the select's
    // true / false values so the polarity is correct regardless of
    // canonicalisation:
    //   pred lt + (true=lhs, false=rhs) -> ``min(lhs, rhs)``
    //   pred lt + (true=rhs, false=lhs) -> ``max(lhs, rhs)``
    //   pred gt + (true=lhs, false=rhs) -> ``max(lhs, rhs)``
    //   pred gt + (true=rhs, false=lhs) -> ``min(lhs, rhs)``
    if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
        auto *cdef = sel.getCondition().getDefiningOp();
        if (auto cmp = mlir::dyn_cast_or_null<mlir::arith::CmpIOp>(cdef)) {
            using P = mlir::arith::CmpIPredicate;
            auto pred = cmp.getPredicate();
            // Strict and inclusive variants collapse to the same
            // min/max semantics (the equal case selects either side
            // -- both equal, so the choice is irrelevant).
            bool is_lt = (pred == P::slt || pred == P::ult
                          || pred == P::sle || pred == P::ule);
            bool is_gt = (pred == P::sgt || pred == P::ugt
                          || pred == P::sge || pred == P::uge);
            if (is_lt || is_gt) {
                bool true_is_lhs = (cmp.getLhs() == sel.getTrueValue());
                bool true_is_rhs = (cmp.getRhs() == sel.getTrueValue());
                const char *fn = nullptr;
                if (is_lt && true_is_lhs)        fn = "min";
                else if (is_lt && true_is_rhs)   fn = "max";
                else if (is_gt && true_is_lhs)   fn = "max";
                else if (is_gt && true_is_rhs)   fn = "min";
                if (fn) {
                    return std::string(fn) + "("
                         + buildIndexExpr(sel.getTrueValue(),  d + 1) + ", "
                         + buildIndexExpr(sel.getFalseValue(), d + 1) + ")";
                }
            }
        }
    }

    return "?";
}

// ---------------------------------------------------------------------------
// Per-statement builders
// ---------------------------------------------------------------------------

ASTNode buildAssignNode(hlfir::AssignOp assign) {
    ASTNode node;
    node.kind = "assign";

    // --- LHS ---
    // For a designate destination, use ``expandDesignateChain`` so
    // view-alias and section-parent chains (Fortran storage-association
    // reshape, slice-into-multi-dim, etc.) decompose into the source
    // array's full-rank coords.  The simple no-chain case still
    // produces the same (target, indices) as the legacy code path.
    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp()) {
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(dd)) {
            auto [arr, dims] = expandDesignateChain(dg);
            node.target = arr.empty() ? traceToDecl(dg.getMemref()) : arr;
            node.target_is_array = true;
            AccessInfo wa;
            wa.array_name = node.target;
            wa.is_write = true;
            for (auto &de : dims) {
                wa.index_vars.push_back(de.var);
                wa.index_exprs.push_back(de.expr);
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
    node.expr = buildExpr(src, 0);
    if (node.expr == "?") {
        if (auto d = src.getDefiningOp()) {
            if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(d)) {
                if (auto f = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
                    std::ostringstream o;
                    o << std::setprecision(17) << f.getValueAsDouble();
                    node.expr = o.str();
                } else if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
                    node.expr = std::to_string(i.getInt());
            }
        }
    }

    // --- Collect RHS array reads ---
    // We emit one AccessInfo per designate *occurrence* in the expression
    // tree  --  not per unique designate op.  emit_tasklet counts array-name
    // regex occurrences in ``assign_node.expr`` and wires one connector
    // per occurrence, so the bridge must supply matching AccessInfo count.
    // For ``g * g`` Flang shares ``%gv = fir.load %gi`` across both mulf
    // operands; without per-occurrence emission the second ``g`` becomes
    // a dangling ``_in_g_1`` connector with no memlet.
    std::function<void(mlir::Value, int)> collectReads =
        [&](mlir::Value v, int depth) {
        if (depth > 40) return;
        auto *op = v.getDefiningOp();
        if (!op) return;
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(op)) {
            AccessInfo ra;
            ra.array_name = traceToDecl(dg.getMemref());
            ra.is_read = true;
            unsigned di = 0;
            for (auto idx : dg.getIndices()) {
                auto n = resolveIndex(idx);
                ra.index_vars.push_back(n.empty() ? "?" : n);
                ra.index_exprs.push_back(buildDesignateIndexExpr(dg, di, idx, 0));
                ++di;
                // Descend into the index operand so inner indirect loads
                // (edge_idx used below z_kin) get their own AccessInfo.
                collectReads(idx, depth + 1);
            }
            node.accesses.push_back(std::move(ra));
            return;
        }
        // ``hlfir.apply %elem, %i``  --  recurse into the referenced
        // elemental's body so reads inside it get tracked.  Mirrors the
        // global ``collectReadAccesses`` handler in elementals.cpp.
        // Without this, ``hlfir.assign (apply elem, i) to dst`` (the
        // shape produced by ``hlfir-expand-vector-subscript-gather``' gather
        // loop) registers no read connectors and the tasklet body
        // references a free-floating array name.
        if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(op)) {
            auto src = apply.getExpr();
            if (auto *sd = src.getDefiningOp()) {
                if (auto inner_elem = mlir::dyn_cast<hlfir::ElementalOp>(sd)) {
                    auto &ireg = inner_elem.getRegion();
                    if (!ireg.empty()) {
                        auto &iblock = ireg.front();
                        auto apply_idxs = apply.getIndices();
                        unsigned pushed = 0;
                        for (unsigned i = 0;
                             i < iblock.getNumArguments() && i < apply_idxs.size();
                             ++i) {
                            auto name = resolveIndex(apply_idxs[i]);
                            indexStack().push_back({iblock.getArgument(i), name});
                            ++pushed;
                        }
                        for (auto &iop : iblock)
                            if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(iop))
                                collectReads(y.getElementValue(), depth + 1);
                        for (unsigned i = 0; i < pushed; ++i)
                            indexStack().pop_back();
                    }
                }
            }
            return;
        }
        // Scalar min/max idiom: ``arith.select(arith.cmpf|cmpi, t, f)``
        // where buildExpr emits ``min(t, f)`` / ``max(t, f)``.  The cmp
        // and the select reference the SAME ``t`` and ``f`` loads, so
        // recursing into both branches would record 4 AccessInfos for
        // the 2 textual operands ``min/max`` emits.  Skip the cmp's
        // operands here so the AccessInfo count matches the textual
        // occurrences emit_tasklet's regex sees.  Falls back to the
        // generic operand walk for any select that doesn't match the
        // idiom (in particular when t != cmp.lhs or f != cmp.rhs, which
        // is the actual conditional-select shape).
        if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
            auto *cdef = sel.getCondition().getDefiningOp();
            bool isMinMax = false;
            if (auto cmp = mlir::dyn_cast_or_null<mlir::arith::CmpFOp>(cdef)) {
                using P = mlir::arith::CmpFPredicate;
                auto pred = cmp.getPredicate();
                bool ok_pred = (pred == P::OLT || pred == P::ULT
                             || pred == P::OGT || pred == P::UGT);
                if (ok_pred && cmp.getLhs() == sel.getTrueValue()
                            && cmp.getRhs() == sel.getFalseValue())
                    isMinMax = true;
            }
            if (!isMinMax) {
                if (auto cmp = mlir::dyn_cast_or_null<mlir::arith::CmpIOp>(cdef)) {
                    using P = mlir::arith::CmpIPredicate;
                    auto pred = cmp.getPredicate();
                    bool ok_pred = (pred == P::slt || pred == P::ult
                                 || pred == P::sgt || pred == P::ugt);
                    if (ok_pred && cmp.getLhs() == sel.getTrueValue()
                                && cmp.getRhs() == sel.getFalseValue())
                        isMinMax = true;
                }
            }
            if (isMinMax) {
                collectReads(sel.getTrueValue(), depth + 1);
                collectReads(sel.getFalseValue(), depth + 1);
                return;
            }
        }
        for (auto operand : op->getOperands())
            collectReads(operand, depth + 1);
    };
    collectReads(src, 0);

    return node;
}

 int64_t traceLB(mlir::Value v) {
    if (auto c = traceConstInt(v)) return *c;
    return -1;
}

/// Peel `fir.ref<...>` / `fir.box<...>` / `fir.heap<...>` / `fir.ptr<...>` wrappers.
 mlir::Type peelWrappers(mlir::Type t) {
    for (int i = 0; i < limits::kTypeWrapperPeelDepth; ++i) {
        mlir::Type next = t;
        if (auto b = mlir::dyn_cast<fir::BoxType>(next))       next = b.getEleTy();
        else if (auto r = mlir::dyn_cast<fir::ReferenceType>(next)) next = r.getEleTy();
        else if (auto h = mlir::dyn_cast<fir::HeapType>(next))      next = h.getEleTy();
        else if (auto p = mlir::dyn_cast<fir::PointerType>(next))   next = p.getEleTy();
        else break;
        t = next;
    }
    return t;
}

/// True iff the MLIR type peels to a ``fir.array<...>`` OR is an
/// ``!hlfir.expr<...>`` value with a non-empty shape.  The latter is
/// what every ``hlfir.elemental`` / ``hlfir.matmul`` / ``hlfir.transpose``
/// produces; the bridge has to treat those as arrays so that
/// ``hlfir.assign %elem to %dst`` routes through the elemental walker
/// (or libcall handler) rather than the section-scalar fallback that
/// only knows how to broadcast a scalar across a slice.
 bool isArrayRef(mlir::Type t) {
    auto peeled = peelWrappers(t);
    if (mlir::isa<fir::SequenceType>(peeled)) return true;
    if (auto e = mlir::dyn_cast<hlfir::ExprType>(peeled))
        return !e.getShape().empty();
    return false;
}

/// True iff ``v`` traces back to an ``arith.constant`` with value zero
/// (integer zero or floating-point +0.0 / -0.0).
 bool isConstantZero(mlir::Value v) {
    auto *def = v.getDefiningOp();
    if (!def) return false;
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def))
        return isConstantZero(cv.getValue());
    if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
        if (auto f = mlir::dyn_cast<mlir::FloatAttr>(c.getValue()))
            return f.getValueAsDouble() == 0.0;
        if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
            return i.getInt() == 0;
    }
    return false;
}

/// ``hlfir.assign %src to %dst`` where both sides are array boxes  --  a
/// whole-array copy.  Emit ``kind="copy"`` and let hlfir_to_sdfg wire a
/// ``standard.CopyLibraryNode``.
ASTNode buildCopyNode(hlfir::AssignOp assign) {
    ASTNode n;
    n.kind = "copy";
    auto dest = assign.getOperand(1);
    // Always route through traceToDecl so the allocatable alias map
    // (set by ``ALLOCATE`` walks in buildAST) takes effect  --  direct
    // ``extractName(decl.getUniqName())`` would skip the alias lookup
    // and stale-bind to the first allocation's name.
    if (auto dd = dest.getDefiningOp())
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            n.target = allocAliasFor(extractName(decl.getUniqName().str()));
    if (n.target.empty()) n.target = traceToDecl(dest);
    n.target_is_array = true;
    n.reduce_src = traceToDecl(assign.getOperand(0));
    return n;
}

/// ``hlfir.assign %zero to %dst`` where source is a constant zero and dest
/// is an array box  --  a zero-fill.  Emit ``kind="memset"`` so
/// hlfir_to_sdfg can wire a ``standard.MemsetLibraryNode``.
/// Return ``dg`` if ``v`` comes from an ``hlfir.designate`` whose
/// ``is_triplet`` attribute marks at least one dimension as a section
/// (lower:upper:stride).  Used by the Phase-1 array-section lowering to
/// split section assignments off from plain indexed designates before
/// the reduce / elemental dispatch.
 hlfir::DesignateOp asSectionDesignate(mlir::Value v) {
    auto *def = v.getDefiningOp();
    if (!def) return {};
    auto dg = mlir::dyn_cast<hlfir::DesignateOp>(def);
    if (!dg) return {};
    for (bool t : dg.getIsTriplet())
        if (t) return dg;
    return {};
}

/// Per-dim spec for an ``hlfir.designate``: either a triplet
/// (lo:hi:stride) or a scalar index.  Used by section helpers to walk
/// LHS / RHS uniformly without re-parsing the flat operand list.
struct DesignateDim {
    bool isTriplet;
    std::string lo;          // buildIndexExpr (Fortran 1-based, 0)  --  triplet only
    std::string hi;          // triplet only
    std::string strideExpr;  // empty when stride literal == 1
    int64_t strideConst = 1; // when strideExpr is empty, the literal stride
    std::string scalarIdx;   // non-triplet only
};

/// Walk a designate's per-dim ``isTriplet`` flags and group its flat
/// index operands accordingly.  Returns ``false`` (and leaves ``out``
/// undefined) when an operand can't be lowered to a string  --  caller
/// must decide whether that's recoverable or a hard error.
static bool parseDesignateDims(hlfir::DesignateOp dg,
                               std::vector<DesignateDim> &out) {
    auto triplets = dg.getIsTriplet();
    auto idxs = dg.getIndices();
    unsigned cursor = 0;
    for (bool isT : triplets) {
        DesignateDim d;
        d.isTriplet = isT;
        if (isT) {
            if (cursor + 3 > idxs.size()) return false;
            d.lo = buildIndexExpr(idxs[cursor], 0);
            d.hi = buildIndexExpr(idxs[cursor + 1], 0);
            if (d.lo.empty() || d.lo == "?" || d.hi.empty() || d.hi == "?")
                return false;
            if (auto sc = traceConstInt(idxs[cursor + 2])) {
                d.strideConst = *sc;
            } else {
                d.strideExpr = buildIndexExpr(idxs[cursor + 2], 0);
                if (d.strideExpr.empty() || d.strideExpr == "?") return false;
            }
            cursor += 3;
        } else {
            if (cursor + 1 > idxs.size()) return false;
            d.scalarIdx = buildIndexExpr(idxs[cursor], 0);
            if (d.scalarIdx.empty() || d.scalarIdx == "?") return false;
            cursor += 1;
        }
        out.push_back(std::move(d));
    }
    return true;
}

/// Lower ``<section_designate> = <scalar>`` as a rank-N nested
/// ``kind="loop"`` wrapper around an inner ``kind="assign"``.  The
/// designate may mix triplet and scalar dims  --  a triplet dim drives
/// a loop and contributes its iter name to the write index; a scalar
/// dim contributes its (Fortran 1-based) index expression directly so
/// e.g. ``res(:, pos(1)) = nlev`` writes ``res[as_0, pos(1)]`` for
/// every ``as_0`` in the slice.
///
/// Returns an empty vector when no triplet dim is present (caller
/// should fall through  --  there is no section to broadcast over).
std::vector<ASTNode> buildSectionScalarAssign(
    hlfir::AssignOp assign, hlfir::DesignateOp dst) {

    std::vector<DesignateDim> dims;
    if (!parseDesignateDims(dst, dims)) return {};
    unsigned tripletRank = 0;
    for (auto &d : dims) if (d.isTriplet) ++tripletRank;
    if (tripletRank == 0) return {};

    std::vector<std::string> iter_names;
    iter_names.reserve(tripletRank);
    for (unsigned i = 0; i < tripletRank; ++i)
        iter_names.push_back("as_" + std::to_string(i));

    auto dstName = traceToDecl(dst.getMemref());
    if (dstName.empty()) return {};

    // Inner assign: target[per-dim] = <scalar_rhs>.  Triplet dims use
    // their iter; scalar dims thread their Fortran 1-based index expr.
    ASTNode inner;
    inner.kind = "assign";
    inner.target = dstName;
    inner.target_is_array = true;
    inner.expr = buildExpr(assign.getOperand(0), 0);

    AccessInfo wa;
    wa.array_name = dstName;
    wa.is_write = true;
    {
        unsigned tDim = 0;
        for (auto &d : dims) {
            if (d.isTriplet) {
                wa.index_vars.push_back(iter_names[tDim]);
                wa.index_exprs.push_back(iter_names[tDim]);
                ++tDim;
            } else {
                wa.index_vars.push_back(d.scalarIdx);
                wa.index_exprs.push_back(d.scalarIdx);
            }
        }
    }
    inner.accesses.push_back(std::move(wa));

    // Bug fix: also collect read accesses on the scalar RHS.
    // ``inner.expr`` is the rendered RHS string (e.g. ``bob(1)``),
    // and Python ``emit_tasklet`` rewrites every array-name occurrence
    // to ``_in_<name>_<N>`` and creates one in_connector + memlet PER
    // ``AccessInfo`` in ``inner.accesses``.  Without read AccessInfos
    // here the connector references in the rewritten code become
    // dangling  --  and DaCe's free-symbol analysis later treats the
    // unbound ``_in_<name>_<N>`` token as an undefined symbol,
    // surfacing as ``KeyError: '_in_bob_0'`` at SDFG construction.
    // Mirrors the ``collectReads`` walker inside ``buildAssignNode``
    // (lines ~304-361 above); kept inline rather than extracted as a
    // free helper because the only shared piece would be the walker
    // and the AccessInfo emission, which already differs in subtle
    // ways (no per-dim ``index_vars`` reset rules at the section level).
    std::function<void(mlir::Value, int)> collectScalarRhsReads =
        [&](mlir::Value v, int depth) {
        if (depth > 40) return;
        auto *op = v.getDefiningOp();
        if (!op) return;
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(op)) {
            AccessInfo ra;
            ra.array_name = traceToDecl(dg.getMemref());
            ra.is_read = true;
            unsigned di = 0;
            for (auto idx : dg.getIndices()) {
                auto n = resolveIndex(idx);
                ra.index_vars.push_back(n.empty() ? "?" : n);
                ra.index_exprs.push_back(buildDesignateIndexExpr(dg, di, idx, 0));
                ++di;
                collectScalarRhsReads(idx, depth + 1);
            }
            inner.accesses.push_back(std::move(ra));
            return;
        }
        for (auto operand : op->getOperands())
            collectScalarRhsReads(operand, depth + 1);
    };
    collectScalarRhsReads(assign.getOperand(0), 0);

    // Wrap descending so the outermost ASTNode is the outermost loop.
    // Lower bound goes into loop_lower_expr (string form) so symbolic
    // lowers like ``res(a:b)`` survive  --  emit_loop prefers it over the
    // int ``loop_lower`` when non-empty.
    ASTNode current = inner;
    {
        unsigned tDim = tripletRank;
        for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
            if (!it->isTriplet) continue;
            --tDim;
            ASTNode wrap;
            wrap.kind = "loop";
            wrap.loop_iter = iter_names[tDim];
            wrap.loop_lower_expr = it->lo;
            wrap.loop_bound      = it->hi;
            wrap.children.push_back(current);
            current = wrap;
        }
    }
    return {current};
}

/// ``<section> = <other_section>`` where both sides are
/// ``hlfir.designate`` ops carrying section info  --  e.g.
/// ``res(nval, pos(1):pos(2)) = input1(nval, pos(3):pos(4))`` from
/// ECRAD's pattern.  Without this the dispatcher would route to
/// ``buildCopyNode`` (whole-array copy) and silently ignore the
/// scalar / triplet structure on each side.
///
/// Strategy: synthesise one outer loop per LHS triplet dim, iterating
/// over the LHS bounds.  Inside, write at the LHS dims (scalar ->
/// scalar index, triplet -> loop iter) and read at the RHS dims
/// (triplet -> loop iter shifted by ``rhs_lo - lhs_lo``).
///
/// Loud-failure contract: returns ``empty`` only when this dispatch
/// arm doesn't apply (src isn't a designate AND isn't a bare-declare
/// matching the LHS triplet count, or neither side has any triplet).
/// Once we've confirmed both sides carry sections, any shape / stride
/// mismatch throws ``std::runtime_error`` rather than falling back to
/// a wrong answer.  The dispatcher relies on this  --  it does NOT have
/// a section-to-section recovery path.
std::vector<ASTNode> buildSectionToSectionAssign(
    hlfir::AssignOp assign, mlir::Value dst) {
    auto srcVal = assign.getOperand(0);
    auto *srcDef = srcVal.getDefiningOp();
    if (!srcDef) return {};

    // Destination form parallels the src dispatch below: a designate
    // with triplets carries its own slicing, while a bare declare
    // means whole-array (synthesise full-extent triplets, lo=1, the
    // hi unused since RHS section bounds drive the loop).  This
    // symmetric handling is what lets ``t0_w = p%pprog(1)%w``
    // (LHS bare decl, RHS section) lower to a per-element loop
    // instead of falling through to ``buildCopyNode`` (which would
    // copy the whole 3D companion).
    auto *dstDef = dst.getDefiningOp();
    if (!dstDef) return {};
    auto dstDg = mlir::dyn_cast<hlfir::DesignateOp>(dstDef);
    auto dstDecl = mlir::dyn_cast<hlfir::DeclareOp>(dstDef);
    if (!dstDg && !dstDecl) return {};

    // ``dstTC`` (triplet count) is set up-front for both forms so the
    // rank-match check below can fire before we parse per-dim details.
    // Bare-decl: every dim is a triplet, so rank == triplet count.
    std::string dstName;
    unsigned dstTC = 0;
    if (dstDg) {
        auto dstTriplets = dstDg.getIsTriplet();
        if (dstTriplets.empty()) return {};
        for (bool t : dstTriplets) if (t) dstTC++;
        if (dstTC == 0) return {};
        dstName = traceToDecl(dstDg.getMemref());
    } else {
        dstName = allocAliasFor(extractName(dstDecl.getUniqName().str()));
        auto ty = dstDecl.getResult(0).getType();
        if (auto b = mlir::dyn_cast<fir::BoxType>(ty))       ty = b.getEleTy();
        if (auto r = mlir::dyn_cast<fir::ReferenceType>(ty)) ty = r.getEleTy();
        if (auto h = mlir::dyn_cast<fir::HeapType>(ty))      ty = h.getEleTy();
        if (auto p = mlir::dyn_cast<fir::PointerType>(ty))   ty = p.getEleTy();
        if (auto seq = mlir::dyn_cast<fir::SequenceType>(ty))
            dstTC = seq.getShape().size();
        if (dstTC == 0) return {};
    }
    if (dstName.empty()) return {};

    // Source can be either a designate (its own slicing) or a bare
    // ``hlfir.declare`` (whole array  --  treated as all-triplet ``(:)``
    // dims with default lower bound 1, stride 1).  The whole-array
    // case is what flang produces for ``res(<section>) = arr`` where
    // ``arr`` is a 1D dummy and the LHS section has matching triplet
    // rank.
    auto srcDg = mlir::dyn_cast<hlfir::DesignateOp>(srcDef);
    auto srcDecl = mlir::dyn_cast<hlfir::DeclareOp>(srcDef);
    if (!srcDg && !srcDecl) return {};

    std::string srcName;
    std::vector<DesignateDim> srcDims;
    unsigned srcTC = 0;
    if (srcDg) {
        srcName = traceToDecl(srcDg.getMemref());
        if (srcName.empty()) return {};
        if (!parseDesignateDims(srcDg, srcDims)) return {};
        for (auto &d : srcDims) if (d.isTriplet) ++srcTC;
    } else {
        // Bare declare: synthesise per-dim full-extent triplets matching
        // the underlying array's rank.  The K-th triplet's lo / hi
        // expressions are the same shape resolveExtent gives back  --
        // here we only need ``lo == "1"`` for the source-shift formula
        // (``rhs_lo - lhs_lo``); the actual bounds aren't used because
        // the loop bounds come from the LHS.
        srcName = allocAliasFor(extractName(srcDecl.getUniqName().str()));
        if (srcName.empty()) return {};
        // Determine src rank from the underlying type.
        auto ty = srcDecl.getResult(0).getType();
        if (auto b = mlir::dyn_cast<fir::BoxType>(ty))       ty = b.getEleTy();
        if (auto r = mlir::dyn_cast<fir::ReferenceType>(ty)) ty = r.getEleTy();
        if (auto h = mlir::dyn_cast<fir::HeapType>(ty))      ty = h.getEleTy();
        if (auto p = mlir::dyn_cast<fir::PointerType>(ty))   ty = p.getEleTy();
        unsigned rank = 0;
        if (auto seq = mlir::dyn_cast<fir::SequenceType>(ty))
            rank = seq.getShape().size();
        if (rank == 0) return {};
        for (unsigned i = 0; i < rank; ++i) {
            DesignateDim d;
            d.isTriplet = true;
            d.lo = "1";
            d.hi = "1";   // unused; loop bounds come from LHS
            srcDims.push_back(std::move(d));
        }
        srcTC = rank;
    }

    // Triplet rank mismatch  --  definitively wrong; ``buildCopyNode``
    // would produce a whole-array copy that silently ignores the
    // mismatched scalar / triplet structure.  Throw loudly.
    if (dstTC != srcTC)
        throw std::runtime_error(
            "section-to-section assign \"" + dstName + " = " + srcName
            + "\": triplet rank mismatch (dst=" + std::to_string(dstTC)
            + ", src=" + std::to_string(srcTC) + ")");

    // Bounds parsing can fail on dynamic-extent allocatables whose
    // triplet operands are ``fir.box_dims`` results (``buildIndexExpr``
    // doesn't lower those yet).  In that case both sides are typically
    // full-extent ``(:)`` over the same backing storage and
    // ``buildCopyNode``'s whole-array copy is correct  --  fall back
    // silently.  Genuine section mismatches (different lo/hi/stride
    // each side) require parseable bounds and are caught below.
    std::vector<DesignateDim> dstDims;
    if (dstDg) {
        if (!parseDesignateDims(dstDg, dstDims)) return {};
    } else {
        // Bare decl: synthesise full-extent triplets per dim, mirroring
        // the bare-source path above.  Loop bounds come from the src
        // section (see boundsSide selection later); the lo="1" here
        // just feeds the per-dim source-shift formula and keeps the
        // memlet index expression Fortran 1-based.
        for (unsigned i = 0; i < dstTC; ++i) {
            DesignateDim d;
            d.isTriplet = true;
            d.lo = "1";
            d.hi = "1";  // unused; loop bounds come from RHS
            dstDims.push_back(std::move(d));
        }
    }

    // Stride: only stride 1 is supported on both sides for now.  Any
    // non-1 stride is rejected loudly so the caller is forced to confront
    // it (silently flattening to stride 1 would compute the wrong
    // elements, e.g. ``arr(1:9:2)`` covering 5 odd indices vs. the first
    // 5 contiguous indices a stride-1 lowering would touch).
    auto dimStrideStr = [](const DesignateDim &d) -> std::string {
        if (!d.strideExpr.empty()) return d.strideExpr;
        return std::to_string(d.strideConst);
    };
    for (size_t i = 0; i < dstDims.size(); ++i)
        if (dstDims[i].isTriplet
            && (!dstDims[i].strideExpr.empty() || dstDims[i].strideConst != 1))
            throw std::runtime_error(
                "section-to-section assign \"" + dstName + " = " + srcName
                + "\": LHS stride " + dimStrideStr(dstDims[i])
                + " on dim " + std::to_string(i) + " not yet supported");
    for (size_t i = 0; i < srcDims.size(); ++i)
        if (srcDims[i].isTriplet
            && (!srcDims[i].strideExpr.empty() || srcDims[i].strideConst != 1))
            throw std::runtime_error(
                "section-to-section assign \"" + dstName + " = " + srcName
                + "\": RHS stride " + dimStrideStr(srcDims[i])
                + " on dim " + std::to_string(i) + " not yet supported");

    std::vector<std::string> iter_names;
    iter_names.reserve(dstTC);
    for (unsigned i = 0; i < dstTC; ++i)
        iter_names.push_back("ss_" + std::to_string(i));

    // Build LHS write-index list (and remember per-tDim lo for the source
    // shift).  Triplet -> loop iter; scalar -> the scalar index expression.
    AccessInfo wa;
    wa.array_name = dstName;
    wa.is_write = true;
    std::vector<std::string> dstLoExprs;
    {
        unsigned tDim = 0;
        for (auto &d : dstDims) {
            if (d.isTriplet) {
                dstLoExprs.push_back(d.lo);
                wa.index_vars.push_back(iter_names[tDim]);
                wa.index_exprs.push_back(iter_names[tDim]);
                tDim++;
            } else {
                wa.index_vars.push_back(d.scalarIdx);
                wa.index_exprs.push_back(d.scalarIdx);
            }
        }
    }

    // Build RHS read-index list, aligning by tDim.  When src_lo == dst_lo
    // (e.g. both ``pos(1):pos(2)``) skip the redundant shift so the
    // memlet stays simple  --  DaCe's symbolic memlet simplifier doesn't
    // always fold ``+ pos(1) - pos(1)``.
    AccessInfo ra;
    ra.array_name = srcName;
    ra.is_read = true;
    {
        unsigned tDim = 0;
        for (auto &d : srcDims) {
            if (d.isTriplet) {
                std::string ix;
                if (d.lo == dstLoExprs[tDim]) {
                    ix = iter_names[tDim];
                } else {
                    ix = "(" + iter_names[tDim] + " + " + d.lo
                       + " - " + dstLoExprs[tDim] + ")";
                }
                ra.index_vars.push_back(iter_names[tDim]);
                ra.index_exprs.push_back(std::move(ix));
                tDim++;
            } else {
                ra.index_vars.push_back(d.scalarIdx);
                ra.index_exprs.push_back(d.scalarIdx);
            }
        }
    }

    // ``inner.expr`` is just the bare source name  --  emit_tasklet's regex
    // scan replaces it with ``_in_<srcName>_0`` and the AccessInfo
    // (index_exprs) builds the memlet subset.  Subscripted forms in
    // ``expr`` would re-index a connector and break codegen.
    ASTNode inner;
    inner.kind = "assign";
    inner.target = dstName;
    inner.target_is_array = true;
    inner.expr = srcName;
    inner.accesses.push_back(std::move(wa));
    inner.accesses.push_back(std::move(ra));

    // Wrap in nested loops over the bounds-bearing side's triplets
    // (Fortran 1-based form so emit_loop's offset_<arr>_d<i>
    // subtraction lands the write at the right element).  When the
    // dst is a bare declare its synthesised dims carry placeholder
    // hi="1"  --  drive the loop from src's section instead.  In the
    // mirror case (bare-decl src, designate dst) dstDims already
    // hold the real bounds, which has been the long-standing path.
    ASTNode current = inner;
    {
        const auto &boundsSide = dstDg ? dstDims : srcDims;
        std::vector<std::pair<std::string, std::string>> bounds;
        for (auto &d : boundsSide)
            if (d.isTriplet) bounds.push_back({d.lo, d.hi});
        for (int i = (int)bounds.size() - 1; i >= 0; --i) {
            ASTNode wrap;
            wrap.kind = "loop";
            wrap.loop_iter = iter_names[i];
            wrap.loop_lower_expr = bounds[i].first;
            wrap.loop_bound      = bounds[i].second;
            wrap.children.push_back(current);
            current = wrap;
        }
    }
    return {current};
}

/// ``hlfir.assign %scalar to %arr_decl`` where the destination is the bare
/// declare for a whole array  --  Fortran's ``res = 3`` (broadcast scalar to
/// every element).  Memset only handles the zero case; for any other
/// constant we synthesise a nested loop that writes the scalar into every
/// element.  Mirrors ``buildSectionScalarAssign``'s shape but iterates
/// from 1 to the declare's extent for each dim instead of ``lo:hi``.
///
/// Returns an empty vector when the destination's shape can't be
/// resolved  --  caller falls back to the default assign handler.
std::vector<ASTNode> buildWholeArrayScalarBroadcast(hlfir::AssignOp assign) {
    auto dst = assign.getOperand(1);
    auto *dDef = dst.getDefiningOp();
    if (!dDef) return {};
    auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dDef);
    if (!decl) return {};
    auto shape = decl.getShape();
    if (!shape) return {};
    auto *shDef = shape.getDefiningOp();
    if (!shDef) return {};

    // Collect per-dim extent operands.  ``fir.shape`` lists them flat;
    // ``fir.shape_shift`` interleaves (lb, ext) pairs  --  pick out the
    // extents (odd indices).
    std::vector<mlir::Value> extents;
    if (auto sh = mlir::dyn_cast<fir::ShapeOp>(shDef)) {
        for (auto e : sh.getExtents()) extents.push_back(e);
    } else if (auto ss = mlir::dyn_cast<fir::ShapeShiftOp>(shDef)) {
        auto ops = ss->getOperands();
        for (unsigned i = 1; i < ops.size(); i += 2) extents.push_back(ops[i]);
    } else {
        return {};
    }
    unsigned rank = extents.size();
    if (rank == 0) return {};

    auto extentString = [](mlir::Value ext) -> std::string {
        // Inline ``resolveExtent``-equivalent: prefer a traced declare,
        // then a literal constant, otherwise a buildIndexExpr fallback.
        if (!ext) return {};
        auto n = traceToDecl(ext);
        if (!n.empty()) return n;
        if (auto c = traceConstInt(ext)) return std::to_string(*c);
        auto idx = buildIndexExpr(ext, 0);
        if (!idx.empty() && idx != "?") return "(" + idx + ")";
        return {};
    };

    std::vector<std::string> bounds;
    bounds.reserve(rank);
    for (unsigned i = 0; i < rank; ++i) {
        auto s = extentString(extents[i]);
        if (s.empty()) return {};
        bounds.push_back(std::move(s));
    }

    std::vector<std::string> iter_names;
    iter_names.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
        iter_names.push_back("ab_" + std::to_string(i));

    ASTNode inner;
    inner.kind = "assign";
    inner.target = traceToDecl(dst);
    inner.target_is_array = true;
    inner.expr = buildExpr(assign.getOperand(0), 0);

    AccessInfo wa;
    wa.array_name = inner.target;
    wa.is_write = true;
    for (unsigned i = 0; i < rank; ++i) {
        wa.index_vars.push_back(iter_names[i]);
        wa.index_exprs.push_back(iter_names[i]);
    }
    inner.accesses.push_back(std::move(wa));

    ASTNode current = inner;
    for (int i = (int)rank - 1; i >= 0; --i) {
        ASTNode wrap;
        wrap.kind = "loop";
        wrap.loop_iter = iter_names[i];
        wrap.loop_lower = 1;
        wrap.loop_bound = bounds[i];
        wrap.children.push_back(current);
        current = wrap;
    }
    return {current};
}

ASTNode buildMemsetNode(hlfir::AssignOp assign) {
    ASTNode n;
    n.kind = "memset";
    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp())
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            n.target = allocAliasFor(extractName(decl.getUniqName().str()));
    if (n.target.empty()) n.target = traceToDecl(dest);
    n.target_is_array = true;
    return n;
}

/// ``target = matmul(a, b)`` / ``transpose(a)`` / ``dot_product(x, y)`` /
/// ``count(mask [,dim])``  --  the source of an hlfir.assign is a first-class
/// hlfir linalg / reduction op.  Emit ``kind="libcall"`` so hlfir_to_sdfg
/// can wire the matching DaCe library node.
///
/// For library nodes that take an integer ``dim`` argument
/// (``hlfir.count`` etc.), the second operand is the dim value;
/// trace it via ``traceConstInt`` and stash it in ``reduce_axes`` (0-based,
/// same convention as ``buildReduceNode``).  ``emit_libcall`` reads it
/// back and converts to Fortran 1-based for the library-node constructor.
ASTNode buildLibCallNode(hlfir::AssignOp assign,
                                mlir::Operation *srcOp,
                                std::string_view callee) {
    ASTNode n;
    n.kind = "libcall";
    n.callee = callee;

    auto dest = assign.getOperand(1);
    captureElementDesignateWrite(dest, n);
    if (!n.target_is_array)
        n.target_is_array = isArrayRef(dest.getType());

    // Linalg ops use call_args for every operand; reduction-style ops
    // (count) treat the first operand as the array source and any
    // remaining numeric operand as a dim/axis arg.
    //
    // ``dot_product(arg1(1:3), arg2(1:3))`` and friends pass an
    // ``hlfir.designate`` with one or more triplets as the operand.
    // Capture the bounds as a parallel ``call_arg_subsets`` entry so
    // emit_libcall can wire a sliced memlet.  Loud-failure contract:
    // once we recognise the operand as a designate with at least one
    // triplet (i.e. NOT a whole-array reference), failing to express
    // the slice throws  --  silently flattening to the whole array would
    // include trailing elements the slice was meant to exclude.
    //
    // Subsets are emitted in DaCe-0-based half-open form ``lo:hi:stride``
    // (with stride omitted when 1).  Symbolic bounds are wrapped in
    // parens when the input expression is non-trivial so e.g.
    // ``__sym_pos_1 - 1`` doesn't bind across an outer subtract.
    auto resolveSliceSubset =
        [&](mlir::Value operand, llvm::StringRef calleeName, unsigned argIdx)
        -> std::pair<std::string, std::string> {
        auto *def = operand.getDefiningOp();
        if (!def) return {traceToDecl(operand), std::string{}};
        auto dg = mlir::dyn_cast<hlfir::DesignateOp>(def);
        if (!dg) return {traceToDecl(operand), std::string{}};
        auto triplets = dg.getIsTriplet();
        if (triplets.empty())
            return {traceToDecl(operand), std::string{}};
        bool anyTriplet = false;
        for (bool t : triplets) if (t) { anyTriplet = true; break; }
        if (!anyTriplet)
            return {traceToDecl(operand), std::string{}};

        auto name = traceToDecl(dg.getMemref());
        if (name.empty())
            throw std::runtime_error(
                "libcall \"" + std::string(calleeName) + "\" arg "
                + std::to_string(argIdx)
                + ": cannot resolve sliced operand's array name");

        std::vector<DesignateDim> dims;
        if (!parseDesignateDims(dg, dims))
            throw std::runtime_error(
                "libcall \"" + std::string(calleeName) + "\" arg "
                + std::to_string(argIdx)
                + ": cannot lower sliced designate of \"" + name + "\"");

        // Walk dims and build the subset expression per dim.  Track the
        // flat-operand cursor manually so we can also peek at the raw
        // mlir::Value for traceConstInt's tighter constant fold (avoids
        // wrapping ``5`` as ``(5 - 1)`` when we can just emit ``4``).
        std::string sub;
        auto idxs = dg.getIndices();
        unsigned flatCursor = 0;
        for (size_t d = 0; d < dims.size(); ++d) {
            if (!sub.empty()) sub += ", ";
            const auto &dim = dims[d];
            if (dim.isTriplet) {
                // DaCe 0-based half-open: lo - 1 : hi : stride.  Drop
                // the explicit ``:1`` stride to keep the common case
                // readable.
                std::string lo0;
                if (auto c = traceConstInt(idxs[flatCursor])) {
                    lo0 = std::to_string(*c - 1);
                } else {
                    lo0 = "(" + dim.lo + " - 1)";
                }
                sub += lo0 + ":" + dim.hi;
                bool strideIsOne = dim.strideExpr.empty()
                                   && dim.strideConst == 1;
                if (!strideIsOne) {
                    sub += ":";
                    sub += dim.strideExpr.empty()
                         ? std::to_string(dim.strideConst)
                         : dim.strideExpr;
                }
                flatCursor += 3;
            } else {
                // Mixed scalar+slice: emit a single-element subset for
                // the scalar dim so the memlet's rank matches the
                // underlying array.
                std::string idx0;
                if (auto c = traceConstInt(idxs[flatCursor])) {
                    idx0 = std::to_string(*c - 1);
                } else {
                    idx0 = "(" + dim.scalarIdx + " - 1)";
                }
                sub += idx0;
                flatCursor += 1;
            }
        }
        return {name, sub};
    };

    auto opName = srcOp->getName().getStringRef();
    bool is_count = (opName == "hlfir.count");
    if (is_count) {
        if (srcOp->getNumOperands() > 0) {
            auto [nm, sub] = resolveSliceSubset(srcOp->getOperand(0), callee, 0);
            n.call_args.push_back(nm);
            n.call_arg_subsets.push_back(sub);
        }
        if (srcOp->getNumOperands() >= 2) {
            auto dim_val = srcOp->getOperand(1);
            if (auto c = traceConstInt(dim_val))
                n.reduce_axes.push_back(*c - 1);   // Fortran 1-based -> 0-based
        }
    } else {
        unsigned argIdx = 0;
        for (auto operand : srcOp->getOperands()) {
            auto [nm, sub] = resolveSliceSubset(operand, callee, argIdx);
            n.call_args.push_back(nm);
            n.call_arg_subsets.push_back(sub);
            ++argIdx;
        }
    }
    return n;
}

/// ``target = sum(a)`` / product / minval / maxval  --  one of the dedicated
/// hlfir reduction ops appears as the source of an hlfir.assign.
///
/// Returned ASTNode carries enough metadata for hlfir_to_sdfg to call
/// ``state.add_reduce(wcr, axes, identity)`` and wire the input / output
/// memlets.  ``axes`` is left empty for whole-array reductions  --  Flang
/// signals that by emitting the reduction op with no ``dim`` operand.
/// Lower ``target = ANY/ALL/SUM/PRODUCT(src(lo:hi, ...))`` as a
/// loop-accumulator: an init-to-identity assign followed by a
/// ``kind="loop"`` whose body ORs / ANDs / sums the next section
/// element into ``target``.  Used when the reduction's input is a
/// section designate  --  DaCe's ``Reduce`` library node would read the
/// whole source array and produce a wrong result.
///
/// Handles the common shape where the destination is a scalar or an
/// element designate (``levelmask(jk)``) and the source has exactly
/// the section dims to loop over; non-section dims of the source
/// thread through via their existing indices (``jk`` here).  Returns
/// an empty vector when the shape doesn't fit so the caller falls
/// back to whole-array ``buildReduceNode``.
std::vector<ASTNode> buildSectionReduceAssign(
    hlfir::AssignOp assign, hlfir::DesignateOp src,
    std::string_view pyOp, std::string_view identity) {

    auto triplets = src.getIsTriplet();
    if (triplets.empty()) return {};
    auto srcIndices = src.getIndices();

    struct DimSpec {
        bool isTriplet = false;
        mlir::Value lo, hi, stride;
        mlir::Value index;
    };
    std::vector<DimSpec> dims;
    unsigned cursor = 0;
    for (bool t : triplets) {
        DimSpec d; d.isTriplet = t;
        if (t) {
            if (cursor + 3 > srcIndices.size()) return {};
            d.lo = srcIndices[cursor++];
            d.hi = srcIndices[cursor++];
            d.stride = srcIndices[cursor++];
        } else {
            if (cursor + 1 > srcIndices.size()) return {};
            d.index = srcIndices[cursor++];
        }
        dims.push_back(d);
    }
    unsigned sectionRank = 0;
    for (auto &d : dims) if (d.isTriplet) sectionRank++;
    if (sectionRank == 0) return {};

    std::vector<std::string> iterNames;
    iterNames.reserve(sectionRank);
    for (unsigned i = 0; i < sectionRank; ++i)
        iterNames.push_back("ar_" + std::to_string(i));

    // Target name + index expressions  --  target may be a scalar (no
    // designate) or an element designate like ``levelmask(jk)``.
    auto dst = assign.getOperand(1);
    std::string tgtName;
    hlfir::DesignateOp tgtDg;
    if (auto *dd = dst.getDefiningOp())
        tgtDg = mlir::dyn_cast<hlfir::DesignateOp>(dd);
    if (tgtDg) tgtName = traceToDecl(tgtDg.getMemref());
    else       tgtName = traceToDecl(dst);
    if (tgtName.empty()) return {};

    AccessInfo tgtWrite;
    tgtWrite.array_name = tgtName;
    tgtWrite.is_write = true;
    if (tgtDg) {
        for (auto idx : tgtDg.getIndices()) {
            auto nm = resolveIndex(idx);
            tgtWrite.index_vars.push_back(nm.empty() ? "?" : nm);
            tgtWrite.index_exprs.push_back(buildIndexExpr(idx, 0));
        }
    }
    bool tgtIsArray = !tgtWrite.index_vars.empty();

    AccessInfo tgtRead = tgtWrite;
    tgtRead.is_write = false;
    tgtRead.is_read = true;

    // Source read  --  full base array name, indexed with section iters
    // for triplet dims and the original indices for non-section dims.
    std::string srcName = traceToDecl(src.getMemref());
    AccessInfo srcRead;
    srcRead.array_name = srcName;
    srcRead.is_read = true;
    unsigned sectionIdx = 0;
    for (auto &d : dims) {
        if (d.isTriplet) {
            srcRead.index_vars.push_back(iterNames[sectionIdx]);
            srcRead.index_exprs.push_back(iterNames[sectionIdx]);
            sectionIdx++;
        } else {
            auto nm = resolveIndex(d.index);
            srcRead.index_vars.push_back(nm.empty() ? "?" : nm);
            srcRead.index_exprs.push_back(buildIndexExpr(d.index, 0));
        }
    }

    // Init assign: target = identity
    ASTNode init;
    init.kind = "assign";
    init.target = tgtName;
    init.target_is_array = tgtIsArray;
    init.expr = std::string(identity);
    init.accesses.push_back(tgtWrite);

    // Accumulate assign: ``target = target <op> src`` for binary ops
    // (``+``, ``*``, ``or``, ``and``); ``target = fn(target, src)`` for
    // function-form reductions (``min``, ``max``).  The latter pattern
    // gives Min/MaxVal section-reduce a tasklet shape that lowers
    // cleanly to ``std::min`` / ``std::max`` via DaCe's symbolic
    // codegen.
    ASTNode acc;
    acc.kind = "assign";
    acc.target = tgtName;
    acc.target_is_array = tgtIsArray;
    bool isFnForm = (pyOp == "min" || pyOp == "max");
    if (isFnForm) {
        acc.expr = std::string(pyOp) + "(" + tgtName + ", " + srcName + ")";
    } else {
        acc.expr = "(" + tgtName + " " + std::string(pyOp) + " " + srcName + ")";
    }
    acc.accesses.push_back(tgtWrite);
    acc.accesses.push_back(tgtRead);
    acc.accesses.push_back(srcRead);

    // Wrap accumulate in one loop per section dim (outermost first,
    // matching buildElementalAssign's convention).
    ASTNode current = acc;
    int revIdx = (int)sectionRank;
    for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
        if (!it->isTriplet) continue;
        --revIdx;
        ASTNode wrap;
        wrap.kind = "loop";
        wrap.loop_iter = iterNames[revIdx];
        wrap.loop_lower_expr = buildIndexExpr(it->lo, 0);
        wrap.loop_bound      = buildIndexExpr(it->hi, 0);
        wrap.children.push_back(current);
        current = wrap;
    }

    return {init, current};
}

}  // namespace hlfir_bridge
