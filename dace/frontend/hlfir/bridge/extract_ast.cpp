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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <map>
#include <set>
#include <sstream>

namespace hlfir_bridge {

// ---------------------------------------------------------------------------
// Elemental index substitution stack
// ---------------------------------------------------------------------------
//
// Flang-lowered ``hlfir.elemental`` bodies use the elemental's block
// argument as the index operand of inner ``hlfir.designate`` ops — that
// index has no ``hlfir.declare`` to trace back to, so traceToDecl returns
// the empty string.  Before we walk an elemental body we push (blockArg,
// syntheticName) pairs onto this stack so our index lookups can resolve
// the block arg to the synthetic loop iter name the emitter will use.
// Supports nesting (elementals composed via hlfir.apply) via LIFO search.

namespace {
std::vector<std::pair<mlir::Value, std::string>> &indexStack() {
    static thread_local std::vector<std::pair<mlir::Value, std::string>> s;
    return s;
}

std::string resolveIndex(mlir::Value idx) {
    // Look up through fir.convert chains since the index might be wrapped.
    mlir::Value cur = idx;
    for (int i = 0; i < 6; ++i) {
        for (auto it = indexStack().rbegin(); it != indexStack().rend(); ++it)
            if (it->first == cur) return it->second;
        if (auto *d = cur.getDefiningOp())
            if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d))
                { cur = cv.getValue(); continue; }
        break;
    }
    return traceToDecl(idx);
}
}  // namespace

// ---------------------------------------------------------------------------
// Expression reconstruction
// ---------------------------------------------------------------------------

/// Recursively build a Python-syntax expression string from an SSA value.
/// Depth-limited to 30 to prevent infinite recursion on malformed IR.
///
/// Handles:
///   * binary / unary arith ops (addf, mulf, subf, divf, addi, muli,
///     negf, minimumf, maximumf)
///   * elementwise math.* ops (math.sin, math.cos, math.sqrt, math.exp,
///     math.log, math.log10, math.tan, math.sinh, math.cosh, math.tanh,
///     math.absf, math.floor, math.ceil, math.erf, math.erfc, math.powf,
///     math.atan, math.atan2, math.asin, math.acos) — emitted as a bare
///     Python call so DaCe's tasklet codegen can resolve the name.
///   * fir.load of hlfir.designate (named variable read)
///   * arith.constant integer / float literals
///   * fir.convert pass-through (numeric kind casts)
///   * hlfir.apply / hlfir.elemental composition (inlined at index)
static std::string buildIndexExpr(mlir::Value v, int d = 0);
static std::string buildExprWithSubscripts(mlir::Value val, int d = 0);
static std::string buildBoolExpr(mlir::Value val, int d = 0);

// Thread-local state for the faithful ``scf.while`` walker.  See the
// block of helpers further down for what these are used for.
static thread_local int kScfValueCounter = 0;
static thread_local llvm::DenseMap<mlir::Value, std::string> kScfValueMap;

// Bare ``fir.alloca`` (no ``hlfir.declare``) → synthetic scalar name.
// Flang uses un-named i32 allocas as scratch counters for the lifted
// ``scf.while`` shape.  Tracking them as synthetic scalars lets
// buildExpr resolve the counter's value inside loop conditions and
// assignments instead of returning ``?``.
static thread_local int kAllocaCounter = 0;
static thread_local llvm::DenseMap<mlir::Operation *, std::string> kAllocaMap;

static std::string allocaSynthName(mlir::Value memref) {
    auto *def = memref.getDefiningOp();
    if (!def) return "";
    auto it = kAllocaMap.find(def);
    if (it != kAllocaMap.end()) return it->second;
    std::string s = "__al_" + std::to_string(kAllocaCounter++);
    kAllocaMap[def] = s;
    return s;
}

static std::string buildExpr(mlir::Value val, int d = 0) {
    if (d > 30) return "?";
    // Synthetic scalars minted for scf.if results: every downstream read of
    // the result Value resolves to the scalar's name, not to walking into
    // the scf.if itself (which has no single defining expression — the
    // value comes from one of two arms).
    {
        auto it = kScfValueMap.find(val);
        if (it != kScfValueMap.end()) return it->second;
    }
    auto *def = val.getDefiningOp();
    if (!def) return "?";

    auto nm = def->getName().getStringRef();

    // Binary arithmetic.
    static const std::map<llvm::StringRef, std::string> bin_ops = {
        {"arith.mulf", " * "}, {"arith.addf", " + "},
        {"arith.subf", " - "}, {"arith.divf", " / "},
        {"arith.muli", " * "}, {"arith.addi", " + "},
        {"arith.subi", " - "}, {"arith.divsi", " // "}, {"arith.divui", " // "},
    };
    if (auto it = bin_ops.find(nm); it != bin_ops.end()
            && def->getNumOperands() == 2) {
        return "(" + buildExpr(def->getOperand(0), d + 1)
                   + it->second
                   + buildExpr(def->getOperand(1), d + 1) + ")";
    }

    if (nm == "arith.negf" && def->getNumOperands() == 1)
        return "(-" + buildExpr(def->getOperand(0), d + 1) + ")";

    // Elementwise min / max — arith.minimumf / maximumf produce IEEE-min/max
    // (NaN-propagating); arith.minnumf / maxnumf are the numeric variants.
    static const std::map<llvm::StringRef, std::string> minmax_ops = {
        {"arith.minimumf", "min"}, {"arith.maximumf", "max"},
        {"arith.minnumf",  "min"}, {"arith.maxnumf",  "max"},
        {"arith.minsi",    "min"}, {"arith.maxsi",    "max"},
        {"arith.minui",    "min"}, {"arith.maxui",    "max"},
    };
    if (auto it = minmax_ops.find(nm); it != minmax_ops.end()
            && def->getNumOperands() == 2) {
        return it->second + "("
             + buildExpr(def->getOperand(0), d + 1) + ", "
             + buildExpr(def->getOperand(1), d + 1) + ")";
    }

    // Elementwise math intrinsics → bare Python names.  DaCe's tasklet
    // codegen maps ``sin``/``cos``/... to ``dace::math::sin`` etc. via
    // ``_ALLOWED_MODULES`` in ``dace/dtypes.py``.  The ``f`` suffix Flang
    // uses (absf / powf / …) is stripped because the runtime wrappers
    // overload on the operand's type.
    static const std::map<llvm::StringRef, std::string> unary_math = {
        {"math.sin",   "sin"},   {"math.cos",   "cos"},
        {"math.tan",   "tan"},
        {"math.asin",  "asin"},  {"math.acos",  "acos"},
        {"math.atan",  "atan"},
        {"math.sinh",  "sinh"},  {"math.cosh",  "cosh"},
        {"math.tanh",  "tanh"},
        {"math.exp",   "exp"},   {"math.log",   "log"},
        {"math.log10", "log10"},
        {"math.sqrt",  "sqrt"},
        {"math.absf",  "abs"},   {"math.absi",  "abs"},
        {"math.floor", "floor"}, {"math.ceil",  "ceil"},
        {"math.erf",   "erf"},   {"math.erfc",  "erfc"},
    };
    if (auto it = unary_math.find(nm); it != unary_math.end()
            && def->getNumOperands() == 1) {
        return it->second + "("
             + buildExpr(def->getOperand(0), d + 1) + ")";
    }

    static const std::map<llvm::StringRef, std::string> binary_math = {
        {"math.powf",  "pow"}, {"math.ipowi", "pow"},
        {"math.atan2", "atan2"},
    };
    if (auto it = binary_math.find(nm); it != binary_math.end()
            && def->getNumOperands() == 2) {
        return it->second + "("
             + buildExpr(def->getOperand(0), d + 1) + ", "
             + buildExpr(def->getOperand(1), d + 1) + ")";
    }

    // fir.convert is transparent at the expression level — Fortran type
    // kind casts (i32 -> i64, f32 -> f64, i64 -> f64 …) don't survive into
    // the tasklet code verbatim.
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def))
        return buildExpr(conv.getValue(), d + 1);

    // MLIR bool/int casts used by lift-cf-to-scf when threading keep-going
    // flags through scf.if yields.  All transparent: the synthetic scalars
    // we mint for scf.if results already hold 0 / 1, and Python handles
    // ``0 != 0`` as False / ``1 != 0`` as True uniformly.
    if (nm == "arith.trunci" || nm == "arith.extui" || nm == "arith.extsi") {
        if (def->getNumOperands() == 1)
            return buildExpr(def->getOperand(0), d + 1);
    }

    // Comparisons flowing into integer casts (``extui %cmp : i1 to i32``
    // yielded to a scf.if result) need to produce a usable expression,
    // not ``?``.  Defer to buildBoolExpr which understands cmpf / cmpi.
    if (nm == "arith.cmpf" || nm == "arith.cmpi") {
        auto b = buildBoolExpr(val, d + 1);
        if (b != "?") return b;
    }
    // ``xori %x, true`` → logical NOT; any other xori → Python ``!=``.
    // MLIR's ``arith.constant true`` stores the i1 value as -1 (all-bits
    // set) on most targets, so match both 1 and -1.
    if (nm == "arith.xori" && def->getNumOperands() == 2) {
        auto *rhs = def->getOperand(1).getDefiningOp();
        if (auto c = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(rhs))
            if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) {
                auto v = ia.getInt();
                if (v == 1 || v == -1)
                    return "(not " + buildExpr(def->getOperand(0), d + 1) + ")";
            }
        return "(" + buildExpr(def->getOperand(0), d + 1) + " != "
                   + buildExpr(def->getOperand(1), d + 1) + ")";
    }

    // Scalar min / max idiom: Flang lowers ``min(a, b)`` on f32/f64 to
    // ``arith.select(arith.cmpf olt, a, b)`` (and ``max`` via ``ogt``).
    // Recognise that shape so the tasklet code gets a bare min/max call.
    if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
        auto *cdef = sel.getCondition().getDefiningOp();
        if (auto cmp = mlir::dyn_cast_or_null<mlir::arith::CmpFOp>(cdef)) {
            auto pred = cmp.getPredicate();
            using P = mlir::arith::CmpFPredicate;
            const char *fn = nullptr;
            if (pred == P::OLT || pred == P::ULT) fn = "min";
            else if (pred == P::OGT || pred == P::UGT) fn = "max";
            if (fn && cmp.getLhs() == sel.getTrueValue()
                   && cmp.getRhs() == sel.getFalseValue()) {
                return std::string(fn) + "("
                     + buildExpr(cmp.getLhs(), d + 1) + ", "
                     + buildExpr(cmp.getRhs(), d + 1) + ")";
            }
        }
        // Same idiom for integer min / max via arith.cmpi.
        if (auto cmp = mlir::dyn_cast_or_null<mlir::arith::CmpIOp>(cdef)) {
            auto pred = cmp.getPredicate();
            using P = mlir::arith::CmpIPredicate;
            const char *fn = nullptr;
            if (pred == P::slt || pred == P::ult) fn = "min";
            else if (pred == P::sgt || pred == P::ugt) fn = "max";
            if (fn && cmp.getLhs() == sel.getTrueValue()
                   && cmp.getRhs() == sel.getFalseValue()) {
                return std::string(fn) + "("
                     + buildExpr(cmp.getLhs(), d + 1) + ", "
                     + buildExpr(cmp.getRhs(), d + 1) + ")";
            }
        }
    }

    if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
        auto mem = ld.getMemref();
        if (auto md = mem.getDefiningOp())
            if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(md))
                return traceToDecl(dg.getMemref());
        auto n = traceToDecl(mem);
        if (!n.empty()) return n;
        // Bare fir.alloca without a hlfir.declare — mint a synthetic
        // scalar name.  Flang uses these as scratch counters for
        // lift-cf-to-scf's lowered DO / DO-WHILE / DO+EXIT shapes.
        if (auto *md = mem.getDefiningOp())
            if (mlir::isa<fir::AllocaOp>(md))
                return allocaSynthName(mem);
    }

    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
        if (auto f = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
            std::ostringstream o; o << f.getValueAsDouble(); return o.str();
        }
        if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
            return std::to_string(i.getInt());
    }

    // hlfir.apply %elem, %i — read one element of an hlfir.elemental expr
    // at a given index.  Inline the referenced elemental's body at the
    // apply site by mapping its block args to the apply's index operands
    // via indexStack(), then recursing into the yield_element operand.
    if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(def)) {
        auto src = apply.getExpr();
        if (auto *srcDef = src.getDefiningOp())
            if (auto elem = mlir::dyn_cast<hlfir::ElementalOp>(srcDef)) {
                auto &region = elem.getRegion();
                if (!region.empty()) {
                    auto &block = region.front();
                    auto apply_idxs = apply.getIndices();
                    unsigned pushed = 0;
                    // Push the apply indices onto the index stack — as
                    // synthetic names if we have them, otherwise pass the
                    // Value through resolveIndex so callers see the same
                    // iter names the outer elemental already set up.
                    for (unsigned i = 0;
                         i < block.getNumArguments() && i < apply_idxs.size();
                         ++i) {
                        auto name = resolveIndex(apply_idxs[i]);
                        indexStack().push_back({block.getArgument(i), name});
                        ++pushed;
                    }
                    std::string result = "?";
                    for (auto &op : block)
                        if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op)) {
                            result = buildExpr(y.getElementValue(), d + 1);
                            break;
                        }
                    for (unsigned i = 0; i < pushed; ++i)
                        indexStack().pop_back();
                    return result;
                }
            }
    }

    return "?";
}

/// Build a display expression for an index value.  Mirrors Fortran syntax
/// (1-based, square brackets for indirect access) so the Python side can
/// pattern-match on it.  Depth-limited to avoid loops on malformed IR.
static std::string buildIndexExpr(mlir::Value v, int d) {
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
                auto arrName = resolveIndex(dg.getMemref());
                if (arrName.empty()) arrName = traceToDecl(dg.getMemref());
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
                auto n = resolveIndex(idx);
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
                auto n = resolveIndex(idx);
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

/// Peel `fir.ref<…>` / `fir.box<…>` / `fir.heap<…>` / `fir.ptr<…>` wrappers.
static mlir::Type peelWrappers(mlir::Type t) {
    for (int i = 0; i < 8; ++i) {
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

/// True iff the MLIR type peels to a ``fir.array<...>``.
static bool isArrayRef(mlir::Type t) {
    return mlir::isa<fir::SequenceType>(peelWrappers(t));
}

/// True iff ``v`` traces back to an ``arith.constant`` with value zero
/// (integer zero or floating-point +0.0 / -0.0).
static bool isConstantZero(mlir::Value v) {
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

/// ``hlfir.assign %src to %dst`` where both sides are array boxes — a
/// whole-array copy.  Emit ``kind="copy"`` and let hlfir_to_sdfg wire a
/// ``standard.CopyLibraryNode``.
static ASTNode buildCopyNode(hlfir::AssignOp assign) {
    ASTNode n;
    n.kind = "copy";
    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp())
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            n.target = extractName(decl.getUniqName().str());
    if (n.target.empty()) n.target = traceToDecl(dest);
    n.target_is_array = true;
    n.reduce_src = traceToDecl(assign.getOperand(0));
    return n;
}

/// ``hlfir.assign %zero to %dst`` where source is a constant zero and dest
/// is an array box — a zero-fill.  Emit ``kind="memset"`` so
/// hlfir_to_sdfg can wire a ``standard.MemsetLibraryNode``.
static ASTNode buildMemsetNode(hlfir::AssignOp assign) {
    ASTNode n;
    n.kind = "memset";
    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp())
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            n.target = extractName(decl.getUniqName().str());
    if (n.target.empty()) n.target = traceToDecl(dest);
    n.target_is_array = true;
    return n;
}

/// ``target = matmul(a, b)`` / ``transpose(a)`` / ``dot_product(x, y)`` —
/// the source of an hlfir.assign is a first-class hlfir linalg op.  Emit
/// ``kind="libcall"`` so hlfir_to_sdfg can wire a ``blas.MatMul`` /
/// ``standard.Transpose`` / ``blas.Dot`` library node.
static ASTNode buildLibCallNode(hlfir::AssignOp assign,
                                mlir::Operation *srcOp,
                                std::string_view callee) {
    ASTNode n;
    n.kind = "libcall";
    n.callee = callee;

    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp())
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            n.target = extractName(decl.getUniqName().str());
    if (n.target.empty()) n.target = traceToDecl(dest);
    n.target_is_array = isArrayRef(dest.getType());

    for (auto operand : srcOp->getOperands())
        n.call_args.push_back(traceToDecl(operand));
    return n;
}

/// ``target = sum(a)`` / product / minval / maxval — one of the dedicated
/// hlfir reduction ops appears as the source of an hlfir.assign.
///
/// Returned ASTNode carries enough metadata for hlfir_to_sdfg to call
/// ``state.add_reduce(wcr, axes, identity)`` and wire the input / output
/// memlets.  ``axes`` is left empty for whole-array reductions — Flang
/// signals that by emitting the reduction op with no ``dim`` operand.
static ASTNode buildReduceNode(hlfir::AssignOp assign, mlir::Operation *redOp,
                               std::string_view wcr,
                               std::string_view identity) {
    ASTNode n;
    n.kind = "reduce";

    // Target (LHS).
    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp())
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            n.target = extractName(decl.getUniqName().str());
    if (n.target.empty()) n.target = traceToDecl(dest);

    // Target is scalar when the reduction produces a scalar; array when a
    // ``dim=`` argument was supplied.  We only fire for the scalar case
    // today so target_is_array stays false.
    n.target_is_array = false;

    // Source array — operand 0 of the reduction op.
    if (redOp->getNumOperands() > 0)
        n.reduce_src = traceToDecl(redOp->getOperand(0));
    n.reduce_wcr = wcr;
    n.reduce_identity = identity;

    // ``hlfir.sum %arr dim %d`` — trace the second operand.  When absent
    // (whole-array reduction) leave reduce_axes empty.
    if (redOp->getNumOperands() >= 2) {
        auto d = redOp->getOperand(1);
        if (auto c = traceConstInt(d))
            // Fortran ``dim`` is 1-based; DaCe axes are 0-based.
            n.reduce_axes.push_back(*c - 1);
    }
    return n;
}

/// Forward declare — called from buildWhileNode to recurse into the body.
static std::vector<ASTNode> buildAST(mlir::Block &block);

/// Synthesise a chain of nested ``kind="conditional"`` AST nodes from a
/// ``fir.select_case`` terminator.  Fortran ``SELECT CASE`` has no direct
/// equivalent in DaCe's control-flow vocabulary, so we fold every case
/// label into a boolean guard and nest the rest in the ``else`` branch.
///
/// Case labels supported (from FIROps.td):
///   - ``#fir.point %v``       → ``x == v``
///   - ``#fir.interval %l %h`` → ``(x >= l) and (x <= h)``
///   - ``#fir.lower %l``       → ``x >= l``
///   - ``#fir.upper %h``       → ``x <= h``
///   - ``unit``                → default (else at the innermost nesting)
///
/// Adjacent cases targeting the same successor block (``case (2, 3, 5)``
/// lowers to three ``fir.point`` cases all pointing at the same ``^bb``)
/// collapse into a single guard whose sub-predicates are OR-joined.
static ASTNode buildSelectCaseChain(fir::SelectCaseOp sel) {
    auto operands = sel.getOperands();
    std::string xExpr = buildExprWithSubscripts(sel.getSelector(operands));

    auto cases = sel.getCases();
    unsigned numCases = cases.size();

    // Per-case metadata for a first pass.
    struct CaseInfo {
        bool isDefault = false;
        std::string guard;
        mlir::Block *dest = nullptr;
    };
    std::vector<CaseInfo> infos;
    infos.reserve(numCases);
    for (unsigned i = 0; i < numCases; ++i) {
        CaseInfo ci;
        ci.dest = sel.getSuccessor(i);
        auto tag = cases[i];
        auto cmpOps = sel.getCompareOperands(operands, i);
        if (mlir::isa<mlir::UnitAttr>(tag)) {
            ci.isDefault = true;
        } else if (mlir::isa<fir::PointIntervalAttr>(tag) && cmpOps && !cmpOps->empty()) {
            ci.guard = "(" + xExpr + " == "
                     + buildExprWithSubscripts((*cmpOps)[0]) + ")";
        } else if (mlir::isa<fir::ClosedIntervalAttr>(tag) && cmpOps && cmpOps->size() >= 2) {
            auto lo = buildExprWithSubscripts((*cmpOps)[0]);
            auto hi = buildExprWithSubscripts((*cmpOps)[1]);
            ci.guard = "((" + xExpr + " >= " + lo + ") and ("
                     + xExpr + " <= " + hi + "))";
        } else if (mlir::isa<fir::LowerBoundAttr>(tag) && cmpOps && !cmpOps->empty()) {
            ci.guard = "(" + xExpr + " >= "
                     + buildExprWithSubscripts((*cmpOps)[0]) + ")";
        } else if (mlir::isa<fir::UpperBoundAttr>(tag) && cmpOps && !cmpOps->empty()) {
            ci.guard = "(" + xExpr + " <= "
                     + buildExprWithSubscripts((*cmpOps)[0]) + ")";
        } else {
            // Unknown shape — emit ``False`` so the case is never taken,
            // keeping the rest of the chain well-formed.
            ci.guard = "False";
        }
        infos.push_back(std::move(ci));
    }

    // Merge runs of non-default cases sharing the same destination block
    // (Fortran ``case (2, 3, 5)`` → three fir.point cases all targeting
    // the same successor).
    struct Group {
        std::string guard;     // OR-joined guards
        mlir::Block *dest = nullptr;
    };
    std::vector<Group> groups;
    std::vector<ASTNode> defaultBody;
    for (auto &ci : infos) {
        if (ci.isDefault) {
            if (ci.dest) defaultBody = buildAST(*ci.dest);
            continue;
        }
        if (!groups.empty() && groups.back().dest == ci.dest) {
            groups.back().guard += " or " + ci.guard;
        } else {
            Group g;
            g.guard = ci.guard;
            g.dest = ci.dest;
            groups.push_back(std::move(g));
        }
    }

    // Build the nested conditional chain from the last non-default group
    // backwards, folding each previous group into the next one's else.
    ASTNode chain;
    bool first = true;
    for (auto it = groups.rbegin(); it != groups.rend(); ++it) {
        ASTNode node;
        node.kind = "conditional";
        node.condition = "(" + it->guard + ")";
        if (it->dest) node.children = buildAST(*it->dest);
        if (first) {
            node.else_children = defaultBody;
            first = false;
        } else {
            node.else_children.push_back(std::move(chain));
        }
        chain = std::move(node);
    }
    // If every case was defaulted away (no non-default labels), fall back
    // to the default body as-is wrapped in a trivial ``if True``.
    if (first) {
        chain.kind = "conditional";
        chain.condition = "True";
        chain.children = defaultBody;
    }
    return chain;
}

/// Resolve the extent of a fir.shape / fir.shape_shift operand at dim `d`,
/// preferring a traced declare name (`"nproma"`), then a literal constant
/// (`"10"`), and falling back to `"?"` if neither is available.
static std::string resolveExtent(mlir::Value shape, unsigned d) {
    if (!shape) return "?";
    auto *def = shape.getDefiningOp();
    if (!def) return "?";
    mlir::Value ext;
    if (auto sh = mlir::dyn_cast<fir::ShapeOp>(def)) {
        if (d >= sh.getExtents().size()) return "?";
        ext = sh.getExtents()[d];
    } else if (auto ss = mlir::dyn_cast<fir::ShapeShiftOp>(def)) {
        auto ops = ss->getOperands();
        unsigned idx = 2 * d + 1;
        if (idx >= ops.size()) return "?";
        ext = ops[idx];
    } else {
        return "?";
    }
    auto n = traceToDecl(ext);
    if (!n.empty()) return n;
    if (auto c = traceConstInt(ext)) return std::to_string(*c);
    return "?";
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
static std::vector<ASTNode>
buildElementalAssign(hlfir::AssignOp assign, hlfir::ElementalOp elem) {
    // Target array (LHS of the assign).
    ASTNode inner;
    inner.kind = "assign";
    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp())
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            inner.target = extractName(decl.getUniqName().str());
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

    // Inner write access: target[ei0, ei1, …].
    AccessInfo wa;
    wa.array_name = inner.target;
    wa.is_write = true;
    for (unsigned i = 0; i < rank; ++i) {
        wa.index_vars.push_back(iter_names[i]);
        wa.index_exprs.push_back(iter_names[i]);
    }
    inner.accesses.push_back(std::move(wa));

    // Walk the body's yield_element to produce the RHS string.
    mlir::Value yielded;
    for (auto &op : block)
        if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op))
            { yielded = y.getElementValue(); break; }
    inner.expr = yielded ? buildExpr(yielded) : "?";

    // Read accesses.  Unlike plain assigns we must follow hlfir.apply into
    // the referenced hlfir.elemental's body (where the real designate
    // lives) — pushing the apply's index mapping onto indexStack() so the
    // designate sees the same synthetic iter names as the outer elemental.
    if (yielded) {
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
                    auto n = resolveIndex(idx);
                    ra.index_vars.push_back(n.empty() ? "?" : n);
                    ra.index_exprs.push_back(buildIndexExpr(idx));
                    collectReads(idx);
                }
                inner.accesses.push_back(std::move(ra));
                return;
            }
            if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(op)) {
                auto src = apply.getExpr();
                if (auto *sd = src.getDefiningOp())
                    if (auto inner_elem = mlir::dyn_cast<hlfir::ElementalOp>(sd)) {
                        auto &ireg = inner_elem.getRegion();
                        if (!ireg.empty()) {
                            auto &iblock = ireg.front();
                            auto apply_idxs = apply.getIndices();
                            unsigned pushed = 0;
                            for (unsigned i = 0;
                                 i < iblock.getNumArguments()
                                      && i < apply_idxs.size();
                                 ++i) {
                                auto name = resolveIndex(apply_idxs[i]);
                                indexStack().push_back(
                                    {iblock.getArgument(i), name});
                                ++pushed;
                            }
                            for (auto &iop : iblock)
                                if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(iop))
                                    collectReads(y.getElementValue());
                            for (unsigned i = 0; i < pushed; ++i)
                                indexStack().pop_back();
                        }
                    }
                return;
            }
            for (auto operand : op->getOperands())
                collectReads(operand);
        };
        collectReads(yielded);
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
    return {current};
}

/// Render an arith::cmpi predicate as a Python comparison operator.  Returns
/// an empty string for signed/unsigned variants we haven't wired up yet.
static std::string cmpiPredStr(mlir::arith::CmpIPredicate p) {
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
static std::string cmpfPredStr(mlir::arith::CmpFPredicate p) {
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
static std::string buildExprWithSubscripts(mlir::Value val, int d) {
    if (d > 30 || !val) return "?";
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
static std::string buildBoolExpr(mlir::Value val, int d) {
    if (d > 30) return "?";
    auto *def = val.getDefiningOp();
    if (!def) return "?";

    // Synthetic scalars for scf.if results.  The assignments we emit for
    // yielded values write 0/1 into them, so reading the name as-is is
    // semantically a bool.
    {
        auto it = kScfValueMap.find(val);
        if (it != kScfValueMap.end()) return it->second;
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

    if (auto cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(def)) {
        auto pred = cmpfPredStr(cmp.getPredicate());
        if (pred.empty()) return "?";
        return "(" + buildExprWithSubscripts(cmp.getLhs(), d + 1) + " " + pred + " "
             + buildExprWithSubscripts(cmp.getRhs(), d + 1) + ")";
    }
    if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(def)) {
        auto pred = cmpiPredStr(cmp.getPredicate());
        if (pred.empty()) return "?";
        return "(" + buildExprWithSubscripts(cmp.getLhs(), d + 1) + " " + pred + " "
             + buildExprWithSubscripts(cmp.getRhs(), d + 1) + ")";
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

    // Last resort: maybe the condition is a scalar bool read or plain expr.
    return buildExpr(val, d + 1);
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
static std::string scfSynthName(mlir::Value v) {
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

static std::vector<ASTNode> walkSCFBeforeRegion(mlir::Block &block);

/// Helper: convert a yielded value to a string for writing into a synthetic
/// scalar.  scf.yield of an i32 constant / boolean / computed expression —
/// just reuse buildExpr, which traces through arith ops and cast chains.
static std::string yieldedExpr(mlir::Value v) {
    auto s = buildExpr(v);
    if (s == "?") s = buildBoolExpr(v);
    return s;
}

static ASTNode buildScfIfAsConditional(mlir::scf::IfOp ifOp) {
    ASTNode c;
    c.kind = "conditional";
    c.condition = buildBoolExpr(ifOp.getCondition());

    auto walkArm = [&](mlir::Region &region) -> std::vector<ASTNode> {
        if (region.empty()) return {};
        auto arm = walkSCFBeforeRegion(region.front());
        // If the scf.if yields values, append one scalar_assign per result
        // reading the matching operand of the arm's scf.yield.
        if (ifOp.getNumResults() > 0) {
            mlir::scf::YieldOp yieldOp;
            for (auto &op : region.front())
                if (auto y = mlir::dyn_cast<mlir::scf::YieldOp>(op)) { yieldOp = y; break; }
            if (yieldOp) {
                for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
                    auto target = scfSynthName(ifOp.getResult(i));
                    auto expr = yieldedExpr(yieldOp.getOperand(i));
                    ASTNode a;
                    a.kind = "assign";
                    a.target = target;
                    a.expr = expr;
                    a.target_is_array = false;
                    arm.push_back(std::move(a));
                }
            }
        }
        return arm;
    };

    c.children = walkArm(ifOp.getThenRegion());
    if (!ifOp.getElseRegion().empty())
        c.else_children = walkArm(ifOp.getElseRegion());
    return c;
}

static std::vector<ASTNode> walkSCFBeforeRegion(mlir::Block &block) {
    std::vector<ASTNode> out;
    for (auto &op : block) {
        if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
            out.push_back(buildScfIfAsConditional(ifOp));
            continue;
        }
        if (auto condOp = mlir::dyn_cast<mlir::scf::ConditionOp>(op)) {
            // ``scf.condition(%c)``: break when %c is false.
            ASTNode guard;
            guard.kind = "conditional";
            auto b = buildBoolExpr(condOp.getCondition());
            guard.condition = "not (" + b + ")";
            ASTNode brk;
            brk.kind = "break";
            guard.children.push_back(std::move(brk));
            out.push_back(std::move(guard));
            continue;
        }
        if (auto assign = mlir::dyn_cast<hlfir::AssignOp>(op)) {
            // Route through the normal assign dispatcher so copy/memset /
            // reduction / elemental shapes stay recognised inside the loop.
            auto src = assign.getOperand(0);
            auto dst = assign.getOperand(1);
            bool dst_is_array = isArrayRef(dst.getType());
            bool src_is_array = isArrayRef(src.getType());
            if (dst_is_array && src_is_array) {
                out.push_back(buildCopyNode(assign));
            } else if (dst_is_array && !src_is_array && isConstantZero(src)) {
                out.push_back(buildMemsetNode(assign));
            } else {
                out.push_back(buildAssignNode(assign));
            }
            continue;
        }
        if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
            // IV / counter bump stores Flang emits inside the lifted
            // scf.while body (``i = i + 1``, ``counter = counter - 1``).
            // Handled uniformly for declared vars and bare-alloca scratch
            // counters.
            auto memref = st.getMemref();
            auto target = traceToDecl(memref);
            if (target.empty())
                if (auto *md = memref.getDefiningOp())
                    if (mlir::isa<fir::AllocaOp>(md))
                        target = allocaSynthName(memref);
            if (target.empty()) continue;
            auto expr = buildExpr(st.getValue());
            // Drop stores whose RHS we couldn't resolve.  These are almost
            // always Flang's implicit IV writeback at the end of a
            // ``fir.do_loop`` body: the stored value is a block arg of the
            // surrounding do-loop that buildExpr can't express on its own
            // — and the regular do-loop emitter already handles the IV
            // through ``initialize_expr`` / ``update_expr``.
            if (expr == "?") continue;
            ASTNode a;
            a.kind = "assign";
            a.target = target;
            a.expr = expr;
            a.target_is_array = false;
            out.push_back(std::move(a));
            continue;
        }
        // Pure-value ops — no AST node, their values flow inline.
    }
    return out;
}

static ASTNode buildWhileNode(mlir::scf::WhileOp whileOp) {
    ASTNode n;
    n.kind = "while";
    n.condition = "True";  // all break decisions live inside the body.

    if (whileOp.getBefore().empty()) return n;
    n.children = walkSCFBeforeRegion(whileOp.getBefore().front());
    return n;
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
            auto src = assign.getOperand(0);
            auto dst = assign.getOperand(1);
            bool dst_is_array = isArrayRef(dst.getType());
            bool src_is_array = isArrayRef(src.getType());

            // Whole-array copy: both sides are array boxes / refs.
            if (dst_is_array && src_is_array) {
                nodes.push_back(buildCopyNode(assign));
                continue;
            }
            // Scalar-zero → array fill: MemsetLibraryNode.
            if (dst_is_array && !src_is_array && isConstantZero(src)) {
                nodes.push_back(buildMemsetNode(assign));
                continue;
            }

            // b = <elementwise-expression>  — Flang wraps the RHS in one or
            // more composed hlfir.elemental ops, the outermost of which is
            // the assign's source.  Synthesise a nested loop over the shape
            // instead of treating it as a scalar assign.
            if (auto *sd = src.getDefiningOp()) {
                if (auto elem = mlir::dyn_cast<hlfir::ElementalOp>(sd)) {
                    for (auto &n : buildElementalAssign(assign, elem))
                        nodes.push_back(std::move(n));
                    continue;
                }
                // Linear-algebra ops are first-class in HLFIR; each lowers
                // to a dedicated DaCe library node.  MatMul's SpecializeMatMul
                // handles matrix-matrix / matrix-vector / vector-matrix via
                // operand rank, so we don't disambiguate here.
                auto srcOpName = sd->getName().getStringRef();
                struct LibEntry {
                    llvm::StringRef op;
                    llvm::StringRef callee;
                };
                static const LibEntry kLibTable[] = {
                    {"hlfir.matmul",      "matmul"},
                    {"hlfir.transpose",   "transpose"},
                    {"hlfir.dot_product", "dot_product"},
                };
                bool libMatched = false;
                for (auto &e : kLibTable) {
                    if (srcOpName == e.op) {
                        nodes.push_back(buildLibCallNode(assign, sd, e.callee.str()));
                        libMatched = true;
                        break;
                    }
                }
                if (libMatched) continue;

                // Scalar reductions land as their own dedicated op; pattern-
                // match each one and hand the shared reduce-lowering helper
                // the right wcr + identity.
                auto opName = sd->getName().getStringRef();
                struct RedEntry {
                    llvm::StringRef op;
                    llvm::StringRef wcr;
                    llvm::StringRef identity;
                };
                static const RedEntry kRedTable[] = {
                    {"hlfir.sum",     "lambda a, b: a + b",   "0"},
                    {"hlfir.product", "lambda a, b: a * b",   "1"},
                    {"hlfir.minval",  "lambda a, b: min(a, b)", "math.inf"},
                    {"hlfir.maxval",  "lambda a, b: max(a, b)", "-math.inf"},
                };
                bool matched = false;
                for (auto &e : kRedTable) {
                    if (opName == e.op) {
                        nodes.push_back(buildReduceNode(
                            assign, sd, e.wcr.str(), e.identity.str()));
                        matched = true;
                        break;
                    }
                }
                if (matched) continue;
            }
            nodes.push_back(buildAssignNode(assign));
            continue;
        }
        if (auto ifOp = mlir::dyn_cast<fir::IfOp>(op)) {
            ASTNode n;
            n.kind = "conditional";
            n.condition = buildBoolExpr(ifOp.getCondition());
            if (!ifOp.getThenRegion().empty())
                n.children = buildAST(ifOp.getThenRegion().front());
            if (!ifOp.getElseRegion().empty())
                n.else_children = buildAST(ifOp.getElseRegion().front());
            nodes.push_back(std::move(n));
            continue;
        }
        if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
            ASTNode n;
            n.kind = "conditional";
            n.condition = buildBoolExpr(ifOp.getCondition());
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
        if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op)) {
            nodes.push_back(buildWhileNode(whileOp));
            continue;
        }
        if (auto sel = mlir::dyn_cast<fir::SelectCaseOp>(op)) {
            nodes.push_back(buildSelectCaseChain(sel));
            continue;
        }
        if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
            // Top-level ``fir.store`` is Flang's lowering for lifted
            // DO / DO-WHILE init (``fir.store %c1 to %i``) and internal
            // scratch counters.  Emit as a plain scalar assign.  Regular
            // ``fir.do_loop``s' internal IV stores never reach here —
            // they live inside the loop's body region, which we walk
            // with the existing do-loop handler that takes care of the
            // IV through ``init_expr`` / ``update_expr``.
            auto memref = st.getMemref();
            auto target = traceToDecl(memref);
            if (target.empty())
                if (auto *md = memref.getDefiningOp())
                    if (mlir::isa<fir::AllocaOp>(md))
                        target = allocaSynthName(memref);
            if (target.empty()) continue;
            auto expr = buildExpr(st.getValue());
            // Drop stores with unresolvable RHS — see note in
            // ``walkSCFBeforeRegion``'s fir.store handler.
            if (expr == "?") continue;
            ASTNode a;
            a.kind = "assign";
            a.target = target;
            a.expr = expr;
            a.target_is_array = false;
            nodes.push_back(std::move(a));
            continue;
        }
    }
    return nodes;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

std::vector<ASTNode> extractAST(mlir::ModuleOp module) {
    // Fresh synthetic-name counters / maps per module so two consecutive
    // extractAST calls don't interleave __sc_5 / __al_2 across unrelated
    // SDFGs.
    kScfValueCounter = 0;
    kScfValueMap.clear();
    kAllocaCounter = 0;
    kAllocaMap.clear();

    std::vector<ASTNode> result;
    module.walk([&](mlir::func::FuncOp func) {
        if (!result.empty()) return;  // first func only
        if (!func.getBody().empty())
            result = buildAST(func.getBody().front());
    });
    return result;
}

}  // namespace hlfir_bridge
