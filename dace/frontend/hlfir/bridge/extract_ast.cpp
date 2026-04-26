// ============================================================================
// extract_ast.cpp — Build a recursive statement tree from HLFIR.
// ============================================================================
// Statement-level ops become nodes; everything else is expression-level
// infrastructure and gets folded into the expression strings or access lists.
// ============================================================================

#include "bridge/extract_ast.h"
#include "bridge/extract_vars.h"
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
#include <iomanip>
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
    for (int i = 0; i < limits::kConvertChainDepth; ++i) {
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

/// Build the index expression string for the ``dim``-th operand of a
/// ``hlfir.designate``, applying the assumed-shape rebase when the
/// designate's base is an inlined alias declare.  Rebase rule (Flang
/// convention: assumed-shape dummies implicitly carry lbound = 1):
///
///     outer_fortran_index = inner_fortran_index + outer_lbound - 1
///
/// so ``arr(i)`` with ``i`` in the callee's 1-based frame becomes
/// ``outer(i + outer_lbound - 1)`` — downstream ``build_memlet_index``
/// then subtracts ``outer_lbound``, net result ``i - 1``, the same
/// 1-based-to-0-based shift the callee view already expected.
static std::string buildDesignateIndexExpr(hlfir::DesignateOp dg,
                                           unsigned dim,
                                           mlir::Value idx,
                                           int depth = 0) {
    std::string raw = buildIndexExpr(idx, depth);
    auto memref = dg.getMemref();
    auto *defOp = memref.getDefiningOp();
    if (!defOp) return raw;
    auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(defOp);
    if (!declOp) return raw;
    auto outer = asAssumedShapeAlias(declOp);
    if (!outer) return raw;
    auto lbs = declareLowerBounds(outer);
    if (dim >= lbs.size() || !lbs[dim]) return raw;
    int64_t adjust = *lbs[dim] - 1;
    if (adjust == 0) return raw;
    if (adjust > 0) return "(" + raw + " + " + std::to_string(adjust) + ")";
    return "(" + raw + " - " + std::to_string(-adjust) + ")";
}

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
    if (d > limits::kBuildExprDepth) return "?";
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

    // Exponentiation ``a ** b``.  Flang's four variants
    // (``math.fpowi`` float**int, ``math.powf`` float**float,
    // ``math.powi`` / ``math.ipowi`` int**int) all surface as the
    // Python ``**`` operator.  A downstream SDFG-level simplify pass
    // recognises ``**`` and rewrites it based on the tasklet's
    // input/output types — no variant marker needed at this layer.
    static const std::set<llvm::StringRef> pow_ops = {
        "math.fpowi", "math.powf", "math.powi", "math.ipowi",
    };
    if (pow_ops.count(nm) && def->getNumOperands() == 2) {
        return "(" + buildExpr(def->getOperand(0), d + 1)
             + " ** "
             + buildExpr(def->getOperand(1), d + 1) + ")";
    }

    // ``hlfir.no_reassoc`` is a transparency wrapper Flang emits around
    // parenthesised subexpressions to prevent the optimizer from
    // reassociating them across ``**`` / ``+`` boundaries.  For our
    // purposes it's a passthrough — recurse into its single operand so
    // we don't strand ``pow`` / ``addf`` results as ``?``.
    if (nm == "hlfir.no_reassoc" && def->getNumOperands() == 1) {
        return buildExpr(def->getOperand(0), d + 1);
    }

    static const std::map<llvm::StringRef, std::string> binary_math = {
        {"math.atan2",    "atan2"},
        // Fortran ``SIGN(a, b)`` on float operands lowers to
        // ``math.copysign``; ``dace::math::copysign`` resolves at the
        // tasklet codegen layer.  Integer SIGN goes through the
        // generic ``arith.select`` ternary fallback (predicate-driven
        // min/max idiom shape).
        {"math.copysign", "copysign"},
    };
    if (auto it = binary_math.find(nm); it != binary_math.end()
            && def->getNumOperands() == 2) {
        return it->second + "("
             + buildExpr(def->getOperand(0), d + 1) + ", "
             + buildExpr(def->getOperand(1), d + 1) + ")";
    }

    // Runtime / LLVM intrinsic calls that Flang sometimes emits for
    // intrinsics it doesn't lower to a ``math.*`` op.  Mapped to bare
    // Python names so DaCe's tasklet codegen routes them through
    // ``dace::math::*`` (or stdlib ``math.*``) the same way ``unary_math``
    // does for the math-dialect form.
    //
    // Notable cases:
    //   * ``math.sinh`` / ``math.cosh`` / ``math.tanh`` exist but Flang
    //     occasionally still emits ``fir.call @sinh`` — recognise both.
    //   * Fortran ``MOD`` / ``MODULO`` lower to ``_FortranAMod*Real{4,8}``
    //     runtime calls; the Python ``math.fmod`` matches Fortran ``MOD``
    //     (truncated quotient) and a ``(a - b * floor(a/b))`` formula
    //     matches ``MODULO`` (floored quotient).
    //   * ``NINT(x)`` lowers to ``llvm.lround``; ``AINT(x)`` to
    //     ``llvm.trunc``; both are supported by DaCe's tasklet codegen
    //     when surfaced as ``round`` / ``trunc`` Python calls.
    if (auto call = mlir::dyn_cast<fir::CallOp>(def)) {
        auto callee = call.getCallee();
        if (callee) {
            llvm::StringRef cname = callee->getRootReference().getValue();
            // Single-arg pass-through to a Python identifier (math /
            // bare runtime calls).
            static const std::map<llvm::StringRef, std::string> unary_calls = {
                {"sinh", "sinh"}, {"cosh", "cosh"}, {"tanh", "tanh"},
                {"asinh", "asinh"}, {"acosh", "acosh"}, {"atanh", "atanh"},
                {"asin", "asin"}, {"acos", "acos"}, {"atan", "atan"},
                {"sin", "sin"}, {"cos", "cos"}, {"tan", "tan"},
                {"exp", "exp"}, {"log", "log"}, {"log10", "log10"},
                {"sqrt", "sqrt"}, {"fabs", "abs"},
                // AINT / ANINT — same-kind real return, value-only round/trunc.
                {"llvm.trunc.f64", "trunc"}, {"llvm.trunc.f32", "trunc"},
                {"llvm.floor.f64", "floor"}, {"llvm.floor.f32", "floor"},
                {"llvm.ceil.f64", "ceil"}, {"llvm.ceil.f32", "ceil"},
                {"llvm.round.f64", "round"}, {"llvm.round.f32", "round"},
                {"llvm.fabs.f64", "abs"}, {"llvm.fabs.f32", "abs"},
            };
            if (auto it = unary_calls.find(cname); it != unary_calls.end()
                    && call.getNumOperands() >= 1) {
                return it->second + "("
                     + buildExpr(call.getOperand(0), d + 1) + ")";
            }
            // Type-converting casts — Fortran NINT(x) / INT(x).
            // Flang emits ``llvm.lround.i{32,64}.f{32,64}`` for NINT
            // (rounded-to-nearest, then truncating cast).  Render as
            // ``dace.int{32,64}(round(x))`` so the rounding stays
            // explicit and the cast lowers to ``static_cast<int{32,64}>``
            // in the C++ codegen.  Plain INT(x) lowers separately via
            // ``fir.convert`` (transparent here) and an integer cast on
            // the Python side; nothing extra needed for that.
            static const std::map<llvm::StringRef, std::string> cast_calls = {
                {"llvm.lround.i32.f64", "dace.int32"},
                {"llvm.lround.i32.f32", "dace.int32"},
                {"llvm.lround.i64.f64", "dace.int64"},
                {"llvm.lround.i64.f32", "dace.int64"},
            };
            if (auto it = cast_calls.find(cname); it != cast_calls.end()
                    && call.getNumOperands() >= 1) {
                return it->second + "(round("
                     + buildExpr(call.getOperand(0), d + 1) + "))";
            }
            // Two-arg ATAN2 runtime fallback.
            if (cname == "atan2" && call.getNumOperands() >= 2) {
                return "atan2(" + buildExpr(call.getOperand(0), d + 1) + ", "
                                + buildExpr(call.getOperand(1), d + 1) + ")";
            }
            // Fortran MOD on real operands — truncated-quotient
            // remainder.  Maps directly to ``std::fmod`` (in ``<cmath>``,
            // pulled in via ``<dace/dace.h>``); integer MOD lowers to
            // ``arith.remsi`` and never reaches this fir.call branch.
            if ((cname == "_FortranAModReal4" || cname == "_FortranAModReal8")
                    && call.getNumOperands() >= 2) {
                return "fmod(" + buildExpr(call.getOperand(0), d + 1) + ", "
                                + buildExpr(call.getOperand(1), d + 1) + ")";
            }
            // Fortran SCALE(x, n) — returns ``x * 2^n``.  Maps to
            // ``dace::math::ldexp`` (templated; ``std::ldexp``
            // internally).  Runtime-call signature is ``(x, n,
            // src_file_ptr, src_line)`` — first two operands are
            // semantic.
            if ((cname == "_FortranAScale4" || cname == "_FortranAScale8")
                    && call.getNumOperands() >= 2) {
                return "ldexp(" + buildExpr(call.getOperand(0), d + 1) + ", "
                                + buildExpr(call.getOperand(1), d + 1) + ")";
            }
            // Fortran EXPONENT(x) — returns ``e`` such that
            // ``x = mantissa * 2^e`` with ``0.5 <= |mantissa| < 1``.
            // ``dace::math::ilogb`` provides this via ``std::frexp``
            // (returns ``int`` directly so callers can use the result
            // in a tasklet-integer context).
            if ((cname == "_FortranAExponent4_4" || cname == "_FortranAExponent8_4"
                 || cname == "_FortranAExponent4_8" || cname == "_FortranAExponent8_8"
                 || cname == "_FortranAExponent4" || cname == "_FortranAExponent8")
                    && call.getNumOperands() >= 1) {
                return "ilogb(" + buildExpr(call.getOperand(0), d + 1) + ")";
            }
            // Fortran MODULO — floored-quotient remainder.
            // ``dace::math::floor_mod`` is the templated helper (uses
            // ``py_mod`` internally; ``floor`` for floats, sign-aware
            // ``((a%b)+b)%b`` for ints).  Required because Python's
            // ``%`` on int floors but C++'s ``%`` on int truncates.
            if ((cname == "_FortranAModuloReal4" || cname == "_FortranAModuloReal8"
                 || cname == "_FortranAModuloInteger4" || cname == "_FortranAModuloInteger8")
                    && call.getNumOperands() >= 2) {
                return "floor_mod(" + buildExpr(call.getOperand(0), d + 1) + ", "
                                    + buildExpr(call.getOperand(1), d + 1) + ")";
            }
        }
    }

    // fir.convert: same-family kind casts (i32→i64, f32→f64, i64→f64)
    // are transparent — Fortran's KIND coercion semantics flow through
    // the tasklet's operand types so the C++ codegen widens for free.
    // Cross-family casts (float ↔ int) are NOT transparent: Fortran's
    // INT(x) / NINT(x) / DBLE(x) / REAL(x) carry semantic intent (cast
    // truncates, NINT rounds, DBLE widens) that the bridge must
    // surface as an explicit ``dace.<ty>(...)`` call so the codegen
    // emits the right ``static_cast``.
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def)) {
        auto inT = conv.getValue().getType();
        auto outT = conv.getRes().getType();
        bool inIsInt = inT.isInteger(8) || inT.isInteger(16)
                       || inT.isInteger(32) || inT.isInteger(64);
        bool outIsInt = outT.isInteger(8) || outT.isInteger(16)
                        || outT.isInteger(32) || outT.isInteger(64);
        bool inIsFloat = mlir::isa<mlir::FloatType>(inT);
        bool outIsFloat = mlir::isa<mlir::FloatType>(outT);
        // Float → integer: explicit truncating cast.  Use ``dace.intN``
        // so the C++ codegen lowers via ``static_cast<int{32,64}>``.
        if (inIsFloat && outIsInt) {
            const char *cast = outT.isInteger(64) ? "dace.int64" : "dace.int32";
            return std::string(cast) + "("
                 + buildExpr(conv.getValue(), d + 1) + ")";
        }
        // Integer → float: same shape — codegen will widen at the
        // arithmetic site.  Tag with ``float64`` / ``float32`` so the
        // intent is explicit when the surrounding op is integer too.
        if (inIsInt && outIsFloat) {
            const char *cast = mlir::cast<mlir::FloatType>(outT).getWidth() == 32
                                   ? "dace.float32" : "dace.float64";
            return std::string(cast) + "("
                 + buildExpr(conv.getValue(), d + 1) + ")";
        }
        // Same family — transparent.
        return buildExpr(conv.getValue(), d + 1);
    }

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
    // ``xori %x, true`` → logical NOT; any other i1 xori → Python ``!=``.
    // For non-i1 operands, ``xori`` is the Fortran ``ieor(a,b)`` bitwise op
    // and lowers to Python ``^``.  MLIR's ``arith.constant true`` stores
    // the i1 value as -1 (all-bits set) on most targets, so match 1 / -1.
    if (nm == "arith.xori" && def->getNumOperands() == 2) {
        bool i1_operands = def->getOperand(0).getType().isInteger(1);
        auto *rhs = def->getOperand(1).getDefiningOp();
        if (i1_operands) {
            if (auto c = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(rhs))
                if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) {
                    auto v = ia.getInt();
                    if (v == 1 || v == -1)
                        return "(not " + buildExpr(def->getOperand(0), d + 1) + ")";
                }
            return "(" + buildExpr(def->getOperand(0), d + 1) + " != "
                       + buildExpr(def->getOperand(1), d + 1) + ")";
        }
        // Bitwise XOR: Fortran ``ieor(a, b)`` and the bitwise-NOT idiom
        // ``xori a, -1`` (Flang's lowering of ``ibclr``'s mask
        // construction step).
        return "(" + buildExpr(def->getOperand(0), d + 1) + " ^ "
                   + buildExpr(def->getOperand(1), d + 1) + ")";
    }

    // Bitwise AND / OR — for non-i1 operands these are ``iand`` / ``ior``
    // (and the building blocks of ``ibclr`` / ``ibset`` / ``ibits`` /
    // ``btest``).  i1 versions feed buildBoolExpr which lifts them to
    // logical ``and`` / ``or``; here we handle the non-bool cases.
    if ((nm == "arith.andi" || nm == "arith.ori") && def->getNumOperands() == 2
            && !def->getOperand(0).getType().isInteger(1)) {
        const char *op = (nm == "arith.andi") ? " & " : " | ";
        return "(" + buildExpr(def->getOperand(0), d + 1) + op
                   + buildExpr(def->getOperand(1), d + 1) + ")";
    }

    // Bit shifts — Fortran ``ishft`` (and the building blocks of
    // ``ibset`` / ``ibclr`` / ``ibits``).  Map shli / shrsi / shrui to
    // Python ``<<`` / ``>>``; Python's ``>>`` is signed by default, so
    // ``shrsi`` and ``shrui`` both render the same on the integer types
    // Flang produces here.
    if ((nm == "arith.shli" || nm == "arith.shrsi" || nm == "arith.shrui")
            && def->getNumOperands() == 2) {
        const char *op = (nm == "arith.shli") ? " << " : " >> ";
        return "(" + buildExpr(def->getOperand(0), d + 1) + op
                   + buildExpr(def->getOperand(1), d + 1) + ")";
    }

    // Integer remainder — ``arith.remsi`` / ``arith.remui`` — used by some
    // Fortran ``mod`` lowerings on integers.
    if ((nm == "arith.remsi" || nm == "arith.remui") && def->getNumOperands() == 2) {
        return "(" + buildExpr(def->getOperand(0), d + 1) + " % "
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
        // Inlined integer Fortran MODULO collapse:
        //
        //   r  = arith.remsi a, b              ; truncated remainder
        //   x  = arith.xori  a, b              ; signed-XOR (sign test)
        //   c1 = arith.cmpi slt, x, 0          ; (a^b) < 0  → signs differ
        //   c2 = arith.cmpi ne, r, 0           ; r != 0
        //   c  = arith.andi c1, c2
        //   ab = arith.addi r, b               ; (r + b)
        //   r' = arith.select c, ab, r         ; floored result
        //
        // Flang inlines this for ``MODULO(int, int)`` instead of
        // emitting a runtime call.  Recognising the shape and emitting
        // a single ``floor_mod(a, b)`` keeps the tasklet expression
        // tight (one connector per operand instead of nine) and uses
        // the existing ``dace::math::floor_mod`` helper.
        do {
            auto trueOp  = sel.getTrueValue().getDefiningOp();
            auto falseOp = sel.getFalseValue().getDefiningOp();
            auto condOp  = sel.getCondition().getDefiningOp();
            auto add = mlir::dyn_cast_or_null<mlir::arith::AddIOp>(trueOp);
            auto rem = mlir::dyn_cast_or_null<mlir::arith::RemSIOp>(falseOp);
            auto andi = mlir::dyn_cast_or_null<mlir::arith::AndIOp>(condOp);
            if (!add || !rem || !andi) break;
            // add = (rem, b)
            auto add_lhs = add.getLhs().getDefiningOp();
            auto add_rem = mlir::dyn_cast_or_null<mlir::arith::RemSIOp>(add_lhs);
            if (!add_rem || add_rem.getResult() != rem.getResult()) break;
            mlir::Value a = rem.getLhs();
            mlir::Value b = rem.getRhs();
            if (add.getRhs() != b) break;
            // andi = (cmpi ne r 0, cmpi slt (xori a b) 0)  -- order-agnostic
            auto cm0 = mlir::dyn_cast_or_null<mlir::arith::CmpIOp>(
                andi.getLhs().getDefiningOp());
            auto cm1 = mlir::dyn_cast_or_null<mlir::arith::CmpIOp>(
                andi.getRhs().getDefiningOp());
            if (!cm0 || !cm1) break;
            auto isNeR = [&](mlir::arith::CmpIOp c) {
                return c.getPredicate() == mlir::arith::CmpIPredicate::ne
                       && c.getLhs() == rem.getResult();
            };
            auto isSltXori = [&](mlir::arith::CmpIOp c) {
                if (c.getPredicate() != mlir::arith::CmpIPredicate::slt) return false;
                auto x = mlir::dyn_cast_or_null<mlir::arith::XOrIOp>(
                    c.getLhs().getDefiningOp());
                return x && ((x.getLhs() == a && x.getRhs() == b)
                          || (x.getLhs() == b && x.getRhs() == a));
            };
            if (!((isNeR(cm0) && isSltXori(cm1)) || (isNeR(cm1) && isSltXori(cm0))))
                break;
            return "floor_mod(" + buildExpr(a, d + 1) + ", "
                                + buildExpr(b, d + 1) + ")";
        } while (false);
        // Generic ternary fallback — Fortran ``MERGE(t, f, mask)`` lowers
        // to a bare ``arith.select`` (and the SIZE/LBOUND/UBOUND clamps
        // Flang inlines as ``(0 > n) ? 0 : n`` use ``arith.select`` on a
        // cmpi whose operand order doesn't match the min/max idiom).
        // Render as Python ``(t if cond else f)``; the C++ codegen
        // accepts the conditional expression.
        std::string condExpr = buildBoolExpr(sel.getCondition(), d + 1);
        if (condExpr == "?")
            condExpr = buildExpr(sel.getCondition(), d + 1);
        return "(" + buildExpr(sel.getTrueValue(), d + 1)
             + " if " + condExpr
             + " else " + buildExpr(sel.getFalseValue(), d + 1) + ")";
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
            // 17 digits round-trips IEEE-754 binary64 exactly — anything
            // less truncates the mantissa and Flang-folded constants
            // (module ``parameter`` literals etc.) come out at f32
            // precision in tasklet code.
            std::ostringstream o; o << std::setprecision(17) << f.getValueAsDouble(); return o.str();
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
    if (d > limits::kBuildIndexExprDepth || !v) return "?";

    // Block args (fir.do_loop induction, hlfir.elemental iter) have no
    // defining op — resolve via indexStack() first so inlined elemental
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

    // Integer arithmetic used inside index expressions — Flang lowers
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
            unsigned di = 0;
            for (auto idx : dg.getIndices()) {
                auto n = resolveIndex(idx);
                wa.index_vars.push_back(n.empty() ? "?" : n);
                wa.index_exprs.push_back(buildDesignateIndexExpr(dg, di, idx));
                ++di;
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
    // tree — not per unique designate op.  emit_tasklet counts array-name
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
                ra.index_exprs.push_back(buildDesignateIndexExpr(dg, di, idx));
                ++di;
                // Descend into the index operand so inner indirect loads
                // (edge_idx used below z_kin) get their own AccessInfo.
                collectReads(idx, depth + 1);
            }
            node.accesses.push_back(std::move(ra));
            return;
        }
        for (auto operand : op->getOperands())
            collectReads(operand, depth + 1);
    };
    collectReads(src, 0);

    return node;
}

static int64_t traceLB(mlir::Value v) {
    if (auto c = traceConstInt(v)) return *c;
    return -1;
}

/// Peel `fir.ref<…>` / `fir.box<…>` / `fir.heap<…>` / `fir.ptr<…>` wrappers.
static mlir::Type peelWrappers(mlir::Type t) {
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
    // Always route through traceToDecl so the allocatable alias map
    // (set by ``ALLOCATE`` walks in buildAST) takes effect — direct
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
/// is an array box — a zero-fill.  Emit ``kind="memset"`` so
/// hlfir_to_sdfg can wire a ``standard.MemsetLibraryNode``.
/// Return ``dg`` if ``v`` comes from an ``hlfir.designate`` whose
/// ``is_triplet`` attribute marks at least one dimension as a section
/// (lower:upper:stride).  Used by the Phase-1 array-section lowering to
/// split section assignments off from plain indexed designates before
/// the reduce / elemental dispatch.
static hlfir::DesignateOp asSectionDesignate(mlir::Value v) {
    auto *def = v.getDefiningOp();
    if (!def) return {};
    auto dg = mlir::dyn_cast<hlfir::DesignateOp>(def);
    if (!dg) return {};
    for (bool t : dg.getIsTriplet())
        if (t) return dg;
    return {};
}

/// Lower ``<section_designate> = <scalar>`` as a rank-N nested
/// ``kind="loop"`` wrapper around an inner ``kind="assign"`` that
/// writes the scalar into ``target[as_0, ..., as_{R-1}]``.  Mirrors
/// the tail of ``buildElementalAssign`` but sources loop bounds from
/// the designate's triplet operands instead of an elemental's shape.
///
/// Returns an empty vector if any non-triplet (single-index) dim is
/// mixed into the designate — Phase 1 doesn't synthesise loops for
/// mixed slice + index designates; those fall back to the caller's
/// default assign handling.
static std::vector<ASTNode> buildSectionScalarAssign(
    hlfir::AssignOp assign, hlfir::DesignateOp dst) {

    llvm::ArrayRef<bool> triplets = dst.getIsTriplet();
    if (triplets.empty()) return {};

    // Split the designate's flat index operand list into per-dim groups
    // of three (lower, upper, stride) for triplet dims.  A non-triplet
    // dim bails Phase 1 — we'd need a separate "fixed at k" index to
    // thread into the inner assign.
    auto indices = dst.getIndices();
    std::vector<std::array<mlir::Value, 3>> triples;
    unsigned cursor = 0;
    for (bool t : triplets) {
        if (!t) return {};
        if (cursor + 3 > indices.size()) return {};
        triples.push_back({indices[cursor], indices[cursor + 1],
                           indices[cursor + 2]});
        cursor += 3;
    }
    if (triples.empty()) return {};

    unsigned rank = triples.size();
    std::vector<std::string> iter_names;
    iter_names.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
        iter_names.push_back("as_" + std::to_string(i));

    // Inner assign: target[iter_names] = <scalar_rhs>.
    ASTNode inner;
    inner.kind = "assign";
    inner.target = traceToDecl(dst.getMemref());
    inner.target_is_array = true;
    inner.expr = buildExpr(assign.getOperand(0));

    AccessInfo wa;
    wa.array_name = inner.target;
    wa.is_write = true;
    for (unsigned i = 0; i < rank; ++i) {
        wa.index_vars.push_back(iter_names[i]);
        wa.index_exprs.push_back(iter_names[i]);
    }
    inner.accesses.push_back(std::move(wa));

    // Wrap descending so the outermost ASTNode is the outermost loop.
    // Lower bound goes into loop_lower_expr (string form) so symbolic
    // lowers like ``res(a:b)`` survive — emit_loop prefers it over the
    // int ``loop_lower`` when non-empty.
    ASTNode current = inner;
    for (int i = rank - 1; i >= 0; --i) {
        ASTNode wrap;
        wrap.kind = "loop";
        wrap.loop_iter = iter_names[i];
        wrap.loop_lower_expr = buildIndexExpr(triples[i][0]);
        wrap.loop_bound      = buildIndexExpr(triples[i][1]);
        wrap.children.push_back(current);
        current = wrap;
    }
    return {current};
}

static ASTNode buildMemsetNode(hlfir::AssignOp assign) {
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
/// ``count(mask [,dim])`` — the source of an hlfir.assign is a first-class
/// hlfir linalg / reduction op.  Emit ``kind="libcall"`` so hlfir_to_sdfg
/// can wire the matching DaCe library node.
///
/// For library nodes that take an integer ``dim`` argument
/// (``hlfir.count`` etc.), the second operand is the dim value;
/// trace it via ``traceConstInt`` and stash it in ``reduce_axes`` (0-based,
/// same convention as ``buildReduceNode``).  ``emit_libcall`` reads it
/// back and converts to Fortran 1-based for the library-node constructor.
static ASTNode buildLibCallNode(hlfir::AssignOp assign,
                                mlir::Operation *srcOp,
                                std::string_view callee) {
    ASTNode n;
    n.kind = "libcall";
    n.callee = callee;

    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp())
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            n.target = allocAliasFor(extractName(decl.getUniqName().str()));
    if (n.target.empty()) n.target = traceToDecl(dest);
    n.target_is_array = isArrayRef(dest.getType());

    // Linalg ops use call_args for every operand; reduction-style ops
    // (count) treat the first operand as the array source and any
    // remaining numeric operand as a dim/axis arg.
    auto opName = srcOp->getName().getStringRef();
    bool is_count = (opName == "hlfir.count");
    if (is_count) {
        if (srcOp->getNumOperands() > 0)
            n.call_args.push_back(traceToDecl(srcOp->getOperand(0)));
        if (srcOp->getNumOperands() >= 2) {
            auto dim_val = srcOp->getOperand(1);
            if (auto c = traceConstInt(dim_val))
                n.reduce_axes.push_back(*c - 1);   // Fortran 1-based → 0-based
        }
    } else {
        for (auto operand : srcOp->getOperands())
            n.call_args.push_back(traceToDecl(operand));
    }
    return n;
}

/// ``target = sum(a)`` / product / minval / maxval — one of the dedicated
/// hlfir reduction ops appears as the source of an hlfir.assign.
///
/// Returned ASTNode carries enough metadata for hlfir_to_sdfg to call
/// ``state.add_reduce(wcr, axes, identity)`` and wire the input / output
/// memlets.  ``axes`` is left empty for whole-array reductions — Flang
/// signals that by emitting the reduction op with no ``dim`` operand.
/// Lower ``target = ANY/ALL/SUM/PRODUCT(src(lo:hi, ...))`` as a
/// loop-accumulator: an init-to-identity assign followed by a
/// ``kind="loop"`` whose body ORs / ANDs / sums the next section
/// element into ``target``.  Used when the reduction's input is a
/// section designate — DaCe's ``Reduce`` library node would read the
/// whole source array and produce a wrong result.
///
/// Handles the common shape where the destination is a scalar or an
/// element designate (``levelmask(jk)``) and the source has exactly
/// the section dims to loop over; non-section dims of the source
/// thread through via their existing indices (``jk`` here).  Returns
/// an empty vector when the shape doesn't fit so the caller falls
/// back to whole-array ``buildReduceNode``.
static std::vector<ASTNode> buildSectionReduceAssign(
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

    // Target name + index expressions — target may be a scalar (no
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
            tgtWrite.index_exprs.push_back(buildIndexExpr(idx));
        }
    }
    bool tgtIsArray = !tgtWrite.index_vars.empty();

    AccessInfo tgtRead = tgtWrite;
    tgtRead.is_write = false;
    tgtRead.is_read = true;

    // Source read — full base array name, indexed with section iters
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
            srcRead.index_exprs.push_back(buildIndexExpr(d.index));
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
        wrap.loop_lower_expr = buildIndexExpr(it->lo);
        wrap.loop_bound      = buildIndexExpr(it->hi);
        wrap.children.push_back(current);
        current = wrap;
    }

    return {init, current};
}

static ASTNode buildReduceNode(hlfir::AssignOp assign, mlir::Operation *redOp,
                               std::string_view wcr,
                               std::string_view identity) {
    ASTNode n;
    n.kind = "reduce";

    // Target (LHS).  ``dest`` may be either a bare ``hlfir.declare`` (whole
    // result variable, e.g. ``out = SUM(a)``) or a ``hlfir.designate``
    // selecting one element of an output array (``res(2) = MINVAL(d)``).
    // For the designate case we capture the index expressions in an
    // AccessInfo so emit_reduce can wire the output memlet to the specific
    // element — without this every reduction in the same routine writes
    // through the whole destination array and the last one wins.
    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp()) {
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd)) {
            n.target = allocAliasFor(extractName(decl.getUniqName().str()));
        } else if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(dd)) {
            n.target = traceToDecl(dg.getMemref());
            n.target_is_array = true;
            AccessInfo wa;
            wa.array_name = n.target;
            wa.is_write = true;
            unsigned di = 0;
            for (auto idx : dg.getIndices()) {
                auto resolved = resolveIndex(idx);
                wa.index_vars.push_back(resolved.empty() ? "?" : resolved);
                wa.index_exprs.push_back(buildDesignateIndexExpr(dg, di, idx));
                ++di;
            }
            n.accesses.push_back(std::move(wa));
        }
    }
    if (n.target.empty()) n.target = traceToDecl(dest);

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

// Process-level monotonic counter for bridge-synthesised transient
// arrays (e.g. the int32 mask used by Mode C of COUNT / SUM / ANY / ALL
// over an inline ``hlfir.elemental``).  Process-level (not per-build)
// so name collisions across multi-file SDFG builds are impossible.
static thread_local int kSynthTransientCounter = 0;

// When true, ``buildBoolExpr``'s cmp / cmpi branches use ``buildExpr``
// instead of ``buildExprWithSubscripts`` for the operands.  Set during
// elemental-body walks (e.g. Mode C COUNT) where the resulting
// expression is destined for a tasklet body — emit_tasklet rewrites
// bare identifiers into per-occurrence connectors and wires the
// subscripts via memlets, so the bridge must emit ``a`` / ``b`` rather
// than ``a[(ei0)-1]`` / ``b[(ei0)-1]``.  Interstate-edge conditions
// (the default) still want the subscript form.
static thread_local bool kBoolExprNoSubscripts = false;

/// Build the AST-node sequence for a Fortran reduction whose source is
/// an inline ``hlfir.elemental`` — the "Mode C" path for COUNT (and the
/// shape that generalises to SUM / ANY / ALL on comparison sources).
///
/// Emits three ASTNodes in order:
///   1. ``kind="declare_transient"`` — a fresh int32 transient sized to
///      the elemental's shape.  ``descriptors.emit_declare_transient``
///      registers the array on the SDFG and in ``builder.arrays``.
///   2. nested ``kind="loop"`` (rank-deep) wrapping a ``kind="assign"``
///      whose target is the transient and whose RHS is
///      ``dace.int32(<elemental yield expression>)``.  The bridge's
///      generic select / cmp / arith machinery walks the yield expr.
///   3. ``kind="libcall"`` to ``CountLibraryNode`` reading the transient
///      and writing the original ``hlfir.assign`` destination.
///
/// The for-loop body has no WCR — the reduction stays inside the
/// library node's expansion (which uses a ``Reduce`` library node, not
/// a WCR-on-tasklet).  When the user's elemental body is more elaborate
/// than a single comparison, the chain-of-tasklets shape still lands
/// inside the loop body as a normal assign; downstream loop-to-map
/// transformations can paralleise the synthesised loop without
/// modifying the rest of the SDFG.
static std::vector<ASTNode>
buildElementalCountLibcall(hlfir::AssignOp assign, hlfir::ElementalOp elem) {
    auto &region = elem.getRegion();
    if (region.empty()) return {};
    auto &block = region.front();
    unsigned rank = block.getNumArguments();
    auto shape = elem.getShape();

    // Mint a fresh transient name shared by the declare, the loop's
    // assign target, and the libcall's source.
    std::string trName = "_count_mask_"
                       + std::to_string(kSynthTransientCounter++);

    // (1) Declare-transient ASTNode.  Shape strings come from
    // ``resolveExtent`` so they handle both literal-int and symbol
    // forms; dtype is fixed at int32 for the count case.
    ASTNode decl;
    decl.kind = "declare_transient";
    decl.target = trName;
    decl.expr = "int32";
    AccessInfo shape_info;
    shape_info.array_name = trName;
    for (unsigned i = 0; i < rank; ++i)
        shape_info.index_exprs.push_back(resolveExtent(shape, i));
    decl.accesses.push_back(std::move(shape_info));

    // (2) Loop+assign filling the transient.  Reuses the existing
    // synthetic-iter / indexStack / collectReads conventions from
    // ``buildElementalAssign`` so designate accesses inside the body
    // see the same iter names as a hand-written per-element loop.
    std::vector<std::string> iter_names;
    iter_names.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
        iter_names.push_back("ei" + std::to_string(i));
    unsigned pushed = 0;
    for (unsigned i = 0; i < rank; ++i) {
        indexStack().push_back({block.getArgument(i), iter_names[i]});
        ++pushed;
    }

    ASTNode inner;
    inner.kind = "assign";
    inner.target = trName;
    inner.target_is_array = true;
    AccessInfo wa;
    wa.array_name = trName;
    wa.is_write = true;
    for (unsigned i = 0; i < rank; ++i) {
        wa.index_vars.push_back(iter_names[i]);
        wa.index_exprs.push_back(iter_names[i]);
    }
    inner.accesses.push_back(std::move(wa));

    // Walk the body's yield to extract the per-element expression.
    // ``buildBoolExpr`` handles arith.cmp* nicely; ``buildExpr``
    // handles arith ops + arith.select.  Wrap in dace.int32(...) so
    // the cast happens at the tasklet level (not at the count node).
    mlir::Value yielded;
    for (auto &op : block)
        if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op))
            { yielded = y.getElementValue(); break; }
    std::string body = "?";
    if (yielded) {
        // Comparisons / boolean expressions go through buildBoolExpr;
        // anything else falls back to buildExpr.  Walk in tasklet-body
        // mode so ``a == b`` becomes ``(a == b)`` (bare names) rather
        // than ``(a[(ei0)-1] == b[(ei0)-1])`` — emit_tasklet's
        // per-occurrence connector wiring expects the bare form.
        bool prev = kBoolExprNoSubscripts;
        kBoolExprNoSubscripts = true;
        std::string b = buildBoolExpr(yielded);
        if (b == "?") b = buildExpr(yielded);
        kBoolExprNoSubscripts = prev;
        body = b;
    }
    inner.expr = "dace.int32(" + body + ")";

    // Per-occurrence read accesses for the assign — duplicate of the
    // walker in ``buildElementalAssign`` since both shapes consume the
    // same yield value.
    if (yielded) {
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
                    ra.index_exprs.push_back(buildDesignateIndexExpr(dg, di, idx));
                    ++di;
                    collectReads(idx, depth + 1);
                }
                inner.accesses.push_back(std::move(ra));
                return;
            }
            // ``fir.load %decl`` direct on an ``hlfir.declare`` result —
            // a scalar dummy argument like ``integer, intent(in) :: lo``
            // has no ``hlfir.designate``; the load reads the whole
            // length-1 dummy.  Emit an empty-index AccessInfo so
            // emit_tasklet wires a ``_in_lo_0 = lo[0]`` memlet.
            if (auto ld = mlir::dyn_cast<fir::LoadOp>(op)) {
                auto mem = ld.getMemref();
                if (auto *md = mem.getDefiningOp()) {
                    if (mlir::isa<hlfir::DeclareOp>(md)) {
                        AccessInfo ra;
                        ra.array_name = traceToDecl(mem);
                        ra.is_read = true;
                        // No subscripts — the 1-element-array dummy
                        // signature lets ``build_memlet_index`` fall
                        // through to a 0-index subset.
                        inner.accesses.push_back(std::move(ra));
                        return;
                    }
                }
            }
            // ``hlfir.apply %elem, %i`` — read one element of an
            // earlier ``hlfir.elemental`` expression.  Flang chains
            // elementals this way for compound boolean expressions
            // (``(a > lo) .and. (a < hi)`` becomes a tree of 5
            // elementals linked by apply).  Walk into the referenced
            // elemental's body so the underlying designate / load
            // accesses get registered.
            if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(op)) {
                auto src = apply.getExpr();
                if (auto *sd = src.getDefiningOp())
                    if (auto inner_elem = mlir::dyn_cast<hlfir::ElementalOp>(sd)) {
                        auto &ireg = inner_elem.getRegion();
                        if (!ireg.empty()) {
                            auto &iblock = ireg.front();
                            auto apply_idxs = apply.getIndices();
                            unsigned pushed_inner = 0;
                            for (unsigned i = 0;
                                 i < iblock.getNumArguments() && i < apply_idxs.size();
                                 ++i) {
                                auto name = resolveIndex(apply_idxs[i]);
                                indexStack().push_back({iblock.getArgument(i), name});
                                ++pushed_inner;
                            }
                            for (auto &iop : iblock)
                                if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(iop))
                                    collectReads(y.getElementValue(), depth + 1);
                            for (unsigned i = 0; i < pushed_inner; ++i)
                                indexStack().pop_back();
                        }
                    }
                return;
            }
            for (auto operand : op->getOperands())
                collectReads(operand, depth + 1);
        };
        collectReads(yielded, 0);
    }

    for (unsigned i = 0; i < pushed; ++i)
        indexStack().pop_back();

    // Wrap the assign in nested loops (outermost first).
    ASTNode current = inner;
    for (int i = (int)rank - 1; i >= 0; --i) {
        ASTNode wrap;
        wrap.kind = "loop";
        wrap.loop_iter = iter_names[i];
        wrap.loop_lower = 1;
        wrap.loop_bound = resolveExtent(shape, i);
        wrap.children.push_back(current);
        current = wrap;
    }

    // (3) libcall to CountLibraryNode reading the transient.  Reuses
    // the same ASTNode shape ``buildLibCallNode`` produces for Mode A.
    ASTNode lib;
    lib.kind = "libcall";
    lib.callee = "count";
    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp())
        if (auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            lib.target = extractName(declOp.getUniqName().str());
    if (lib.target.empty()) lib.target = traceToDecl(dest);
    lib.target_is_array = isArrayRef(dest.getType());
    lib.call_args.push_back(trName);

    return {std::move(decl), std::move(current), std::move(lib)};
}

/// ``b = MERGE(t, f, mask)`` on arrays — Flang lowers as
/// ``hlfir.elemental { hlfir.designate; arith.select; yield_element }``.
/// Detect that exact shape (three loaded designate sources fed into a
/// single ``arith.select``) and emit a ``kind="libcall"`` to
/// ``MergeLibraryNode`` directly.  The library node owns the per-target
/// expansion, so the bridge stays out of the per-element select details.
///
/// Returns an empty vector if the elemental body doesn't match the
/// simple three-designate-load shape; the caller falls back to the
/// generic ``buildElementalAssign`` (which inlines the select into a
/// per-element tasklet via the existing arith.select fallback).
static std::vector<ASTNode>
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
static std::vector<ASTNode>
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
        // Per-occurrence AccessInfo (depth-limited, no op-identity dedup).
        // emit_tasklet counts array-name regex occurrences in the RHS
        // string; shared SSA values (``x * x``) must yield matching
        // AccessInfo count or downstream wiring strands a connector.
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
                    ra.index_exprs.push_back(buildDesignateIndexExpr(dg, di, idx));
                    ++di;
                    collectReads(idx, depth + 1);
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
                                    collectReads(y.getElementValue(), depth + 1);
                            for (unsigned i = 0; i < pushed; ++i)
                                indexStack().pop_back();
                        }
                    }
                return;
            }
            for (auto operand : op->getOperands())
                collectReads(operand, depth + 1);
        };
        collectReads(yielded, 0);
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
    if (d > limits::kBuildExprDepth || !val) return "?";
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
    // The caller ABI supplies a companion flag ``<name>_present``
    // (registered by extract_vars whenever it sees an optional declare);
    // reading the flag IS the boolean expression.
    if (auto isp = mlir::dyn_cast<fir::IsPresentOp>(def)) {
        auto n = traceToDecl(isp.getVal());
        if (!n.empty()) return n + "_present";
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

    if (auto cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(def)) {
        auto pred = cmpfPredStr(cmp.getPredicate());
        if (pred.empty()) return "?";
        // Tasklet-body context (set by buildElementalCountLibcall etc.):
        // strip subscripts so emit_tasklet's identifier-rewrite turns
        // bare ``a`` / ``b`` into connectors ``_in_a_0`` / ``_in_b_0``.
        // Default context (interstate-edge condition) keeps subscripts.
        if (kBoolExprNoSubscripts)
            return "(" + buildExpr(cmp.getLhs(), d + 1) + " " + pred + " "
                 + buildExpr(cmp.getRhs(), d + 1) + ")";
        return "(" + buildExprWithSubscripts(cmp.getLhs(), d + 1) + " " + pred + " "
             + buildExprWithSubscripts(cmp.getRhs(), d + 1) + ")";
    }
    if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(def)) {
        auto pred = cmpiPredStr(cmp.getPredicate());
        if (pred.empty()) return "?";
        if (kBoolExprNoSubscripts)
            return "(" + buildExpr(cmp.getLhs(), d + 1) + " " + pred + " "
                 + buildExpr(cmp.getRhs(), d + 1) + ")";
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

    // Per-block site counter for ``ALLOCATE``-bound stores into an
    // allocatable's box descriptor.  Increments every time we walk past
    // a ``fir.store (fir.embox-of-fir.allocmem) to <decl_box_ref>``;
    // first store keeps the original Fortran name (site 0 alias =
    // identity), subsequent ones bind ``x_alloc1`` / ``x_alloc2`` /
    // … via setAllocAlias so every downstream traceToDecl picks up
    // the per-allocation transient name.  Uses the block-local map
    // keyed by raw declare name so two separate allocatables in the
    // same scope don't share counters.
    std::map<std::string, unsigned> allocSiteCount;
    auto bindAllocSite = [&](mlir::Operation *op) {
        auto store = mlir::dyn_cast<fir::StoreOp>(op);
        if (!store) return false;
        auto valDef = store.getValue().getDefiningOp();
        if (!valDef) return false;
        auto embox = mlir::dyn_cast<fir::EmboxOp>(valDef);
        if (!embox) return false;
        auto allocmem = mlir::dyn_cast_or_null<fir::AllocMemOp>(
            embox.getMemref().getDefiningOp());
        if (!allocmem) return false;
        // Only the user-visible allocs we model — skip embox-of-zero_bits
        // (the empty-init store the bridge already filters out elsewhere).
        auto un = allocmem.getUniqName();
        if (!un || !un->ends_with(".alloc")) return false;
        auto memDef = store.getMemref().getDefiningOp();
        if (!memDef) return false;
        auto decl = mlir::dyn_cast<hlfir::DeclareOp>(memDef);
        if (!decl) return false;
        std::string raw = extractName(decl.getUniqName().str());
        if (raw.empty()) return false;
        unsigned site = allocSiteCount[raw]++;
        setAllocAlias(raw, allocAliasName(raw, site));
        return true;
    };
    for (auto &op : block) {
        // Bind / advance the alloc-alias for this allocatable, then
        // skip the op — the SDFG model treats the storage as live for
        // the whole scope, so we emit no AST node for the alloc-bound
        // store itself.
        if (bindAllocSite(&op)) continue;

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
            if (n.loop_bound.empty()) {
                // Closed-form upper bound (``DO i = 1, n-1`` lowers to
                // ``arith.subi %n, %c1`` which neither traceToDecl nor
                // traceConstInt resolves).  Fall back to the same
                // index-expression renderer the lower-bound branch uses
                // so downstream emitters see ``(n - 1)`` instead of ""``.
                n.loop_bound = buildIndexExpr(doLoop.getUpperBound());
            }
            n.loop_lower = traceLB(doLoop.getLowerBound());
            if (n.loop_lower < 0) {
                // Non-constant lower bound (e.g. ``DO jk = nflatlev, nlev``
                // with ``nflatlev`` a dummy-arg scalar).  Capture the
                // symbolic form so emit_loop can thread it through
                // instead of silently defaulting to 1.
                auto sym = traceToDecl(doLoop.getLowerBound());
                if (!sym.empty())      n.loop_lower_expr = sym;
                else                   n.loop_lower_expr = buildIndexExpr(doLoop.getLowerBound());
            }
            // Elemental-inlined bodies use the fir.do_loop block arg
            // directly as the hlfir.designate index — no fir.store →
            // alloca → fir.load indirection.  traceLoopIter returns ""
            // for that shape; push the block arg onto indexStack() with
            // a synthetic name so resolveIndex() can recover it when the
            // inner designate's index is the raw block arg.
            static thread_local int kDoLoopIterCounter = 0;
            bool pushedBlockArg = false;
            auto &loopBlock = doLoop.getRegion().front();
            if (n.loop_iter.empty() && loopBlock.getNumArguments() > 0) {
                n.loop_iter = "_doit_" + std::to_string(kDoLoopIterCounter++);
                indexStack().push_back({loopBlock.getArgument(0), n.loop_iter});
                pushedBlockArg = true;
            }
            n.children   = buildAST(loopBlock);
            if (pushedBlockArg) indexStack().pop_back();
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

            // Array-section ``res(a:b) = <scalar>`` — detect the LHS
            // hlfir.designate with triplet operands and synthesise a
            // nested loop over the section bounds.  Handled before the
            // elemental dispatch below because Flang emits a plain
            // scalar RHS here (no hlfir.elemental wrapping).
            if (!src_is_array) {
                if (auto sec = asSectionDesignate(dst)) {
                    auto built = buildSectionScalarAssign(assign, sec);
                    if (!built.empty()) {
                        for (auto &built_n : built)
                            nodes.push_back(std::move(built_n));
                        continue;
                    }
                }
            }

            // b = <elementwise-expression>  — Flang wraps the RHS in one or
            // more composed hlfir.elemental ops, the outermost of which is
            // the assign's source.  Synthesise a nested loop over the shape
            // instead of treating it as a scalar assign.
            //
            // Special-case: ``b = MERGE(t, f, mask)`` on arrays lowers to
            // ``hlfir.elemental { hlfir.designate; arith.select;
            // yield_element }``.  Detect that exact shape and route to
            // ``MergeLibraryNode`` directly so the per-element select
            // stays inside the library node's expansion (modular —
            // bridge doesn't inline).  Anything more elaborate falls
            // through to ``buildElementalAssign``'s per-element tasklet
            // path (which uses the generic select/cmp fallback).
            if (auto *sd = src.getDefiningOp()) {
                if (auto elem = mlir::dyn_cast<hlfir::ElementalOp>(sd)) {
                    auto merge_built = buildMergeLibcall(assign, elem);
                    if (!merge_built.empty()) {
                        for (auto &n : merge_built)
                            nodes.push_back(std::move(n));
                        continue;
                    }
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
                    // Fortran ``COUNT(mask [, dim])`` — routed through
                    // ``CountLibraryNode`` so its ``cast → Reduce``
                    // expansion handles the integer-cast and the
                    // per-target reduction lowering.  ``buildLibCallNode``
                    // picks up the optional ``dim`` operand and threads
                    // it through the ASTNode for ``emit_libcall``.
                    {"hlfir.count",       "count"},
                };
                bool libMatched = false;
                for (auto &e : kLibTable) {
                    if (srcOpName == e.op) {
                        // Mode C: ``hlfir.count`` whose first operand is
                        // an ``hlfir.elemental`` (comparison-as-mask /
                        // compound boolean expression).  Synthesise a
                        // transient int32 mask via a per-element loop,
                        // then route through ``CountLibraryNode``.
                        if (e.op == "hlfir.count" && sd->getNumOperands() > 0) {
                            auto mask_src = sd->getOperand(0);
                            if (auto *ms = mask_src.getDefiningOp()) {
                                if (auto elem_src = mlir::dyn_cast<hlfir::ElementalOp>(ms)) {
                                    auto built = buildElementalCountLibcall(assign, elem_src);
                                    if (!built.empty()) {
                                        for (auto &n : built)
                                            nodes.push_back(std::move(n));
                                        libMatched = true;
                                        break;
                                    }
                                }
                            }
                        }
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
                    llvm::StringRef wcr;       // DaCe wcr lambda string
                    llvm::StringRef identity;  // initial accumulator value
                    llvm::StringRef py_op;     // Python binary op for
                                               // section-reduce loop body;
                                               // empty → fall back to
                                               // buildReduceNode (whole-array)
                };
                static const RedEntry kRedTable[] = {
                    {"hlfir.sum",     "lambda a, b: a + b",    "0",     "+"},
                    {"hlfir.product", "lambda a, b: a * b",    "1",     "*"},
                    // Identity strings use the bare ``inf`` token (not
                    // ``math.inf``) so DaCe's cppunparse — which maps
                    // ``inf`` → ``INFINITY`` via _py2c_reserved — emits
                    // a valid C++ literal in the section-reduce init
                    // tasklet.  The whole-array Reduce path's eval()
                    // namespace is patched with ``inf=math.inf`` for
                    // the same string.
                    {"hlfir.minval",  "lambda a, b: min(a, b)", "inf",  "min"},
                    {"hlfir.maxval",  "lambda a, b: max(a, b)", "-inf", "max"},
                    // Logical reductions — ANY / ALL on ``fir.logical``
                    // arrays (ICON's levelmask / maskflag patterns).
                    {"hlfir.any",     "lambda a, b: a or b",   "False",     "or"},
                    {"hlfir.all",     "lambda a, b: a and b",  "True",      "and"},
                    // ``hlfir.count`` is intentionally absent — handled
                    // in ``kLibTable`` above as a ``CountLibraryNode``
                    // libcall (covers Fortran COUNT's int-cast semantics
                    // and the optional ``dim`` argument).
                };
                bool matched = false;
                for (auto &e : kRedTable) {
                    if (opName == e.op) {
                        // If the reduction source is a section designate
                        // (``mask(lo:hi, jk)``) we can't use DaCe's Reduce
                        // node directly — it reduces whole arrays.  Fall
                        // back to a loop-accumulator lowering when a
                        // Python op is available.
                        bool emitted = false;
                        if (!e.py_op.empty() && sd->getNumOperands() > 0) {
                            auto srcVal = sd->getOperand(0);
                            if (auto *srcOp = srcVal.getDefiningOp()) {
                                if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(srcOp)) {
                                    bool hasTrip = false;
                                    for (bool t : dg.getIsTriplet())
                                        if (t) { hasTrip = true; break; }
                                    if (hasTrip) {
                                        auto built = buildSectionReduceAssign(
                                            assign, dg, e.py_op.str(),
                                            e.identity.str());
                                        if (!built.empty()) {
                                            for (auto &bn : built)
                                                nodes.push_back(std::move(bn));
                                            emitted = true;
                                        }
                                    }
                                }
                            }
                        }
                        if (!emitted) {
                            nodes.push_back(buildReduceNode(
                                assign, sd, e.wcr.str(), e.identity.str()));
                        }
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
            // Allocatable deallocate-guard: ``fir.if (alloc_status != 0) {
            // fir.freemem, reset box to zero }``.  Carries no observable
            // side effect in the SDFG model (we treat allocatables as
            // single-allocation transients) — skip the whole construct.
            auto isAllocCleanup = [](mlir::Region &region) {
                if (region.empty()) return false;
                bool hasFreemem = false;
                for (auto &op : region.front()) {
                    auto nm = op.getName().getStringRef();
                    if (nm == "fir.freemem") { hasFreemem = true; continue; }
                    if (nm == "fir.box_addr"  || nm == "fir.zero_bits"
                        || nm == "fir.embox"  || nm == "fir.shape"
                        || nm == "fir.store"  || nm == "fir.load"
                        || nm == "fir.if"     || nm == "fir.result"
                        || nm == "arith.constant")
                        continue;
                    return false;
                }
                return hasFreemem;
            };
            if (isAllocCleanup(ifOp.getThenRegion())
                && (ifOp.getElseRegion().empty() || isAllocCleanup(ifOp.getElseRegion()))) {
                continue;
            }
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
    clearAllocAliases();

    std::vector<ASTNode> result;
    module.walk([&](mlir::func::FuncOp func) {
        if (!result.empty()) return;  // first func only
        if (!func.getBody().empty())
            result = buildAST(func.getBody().front());
    });
    return result;
}

}  // namespace hlfir_bridge
