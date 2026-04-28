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
// Expression-builder primitives.  Owns:
//   * buildExpr (recursive Python-syntax expression rewrite for arith,
//     math.*, fir.load, hlfir.designate, hlfir.apply, …) + its forward
//     declarations.
//   * buildIndexExpr and buildDesignateIndexExpr (Fortran 1-based
//     index renderer with section-parent + assumed-shape rebase, 0).
//   * resolveIndex and indexStack() (elemental-iter substitution).
//   * allocaSynthName (synthetic names for bare fir.alloca scratch).
//   * Thread-local state used by buildExpr itself: kScfValueMap,
//     kAllocaMap, kHlfirExprToTransient.
//
// This file is included verbatim from extract_ast.cpp via
// #include "bridge/ast/expressions.cpp" and shares that translation
// unit's namespace, includes, and file-static state.  It MUST NOT be
// added to the build's compile list — CMakeLists.txt deliberately omits
// it.  The split is purely for readability: the AST builder used to
// be a single 2800-line file.
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

// Lower ``fir.is_present %v -> i1`` to a Python expression.  After
// ``hlfir-inline-all`` flattens an internal subprogram (Fortran's
// ``CONTAINS``), the operand of an inner-scope ``is_present`` walks
// back through one or more inlined ``hlfir.declare`` aliases until it
// roots at either:
//   * ``fir.absent`` — the caller passed nothing → constant ``0``;
//   * a host block-arg whose declare carries ``fortran_attrs<optional>``
//     → emit the companion ``<name>_present`` symbol that
//     ``extract_vars`` registers alongside every host-scope OPTIONAL
//     dummy; the caller binds it to 0 / 1 at SDFG-call time;
//   * any other root (mandatory dummy, local alloca) → constant ``1``,
//     since the storage is unconditionally bound.
// Only the host-scope declare (the one whose memref IS the block-arg)
// decides between ``_present`` and ``1`` — inner-scope inlined
// aliases all carry ``optional`` from the callee's signature, but
// that's bookkeeping, not whether the caller actually passed storage.
// Returns ``""`` when the chain breaks before a recognisable root, so
// callers can fall back to ``?``.
std::string lowerIsPresent(mlir::Value operand) {
    mlir::Value cur = operand;
    hlfir::DeclareOp lastDecl;
    for (int i = 0; i < limits::kTraceToDeclMax && cur; ++i) {
        if (auto *d = cur.getDefiningOp()) {
            if (mlir::isa<fir::AbsentOp>(d)) return "0";
            if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d))
                { cur = cv.getValue(); continue; }
            if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
                lastDecl = dc;
                cur = dc.getMemref();
                continue;
            }
            break;
        }
        // Block argument — that's the storage root.  The declare we
        // most recently walked through is the host-scope alias; its
        // OPTIONAL attribute tells us whether to emit the companion
        // symbol or fold to ``1``.
        if (mlir::isa<mlir::BlockArgument>(cur)) {
            bool isOpt = false;
            if (lastDecl)
                if (auto a = lastDecl.getFortranAttrs())
                    isOpt = bitEnumContainsAny(*a,
                                fir::FortranVariableFlagsEnum::optional);
            if (isOpt) {
                auto n = extractName(lastDecl.getUniqName().str());
                return allocAliasFor(n) + "_present";
            }
            return "1";
        }
        break;
    }
    return "";
}

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
// Cross-chunk helpers (signatures + docstrings live in
// ``bridge/ast/ast_helpers.h``).  Bodies appear later in this file
// or in ``assigns.inc`` / ``control_flow.inc``.


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
/// Emit a Fortran 1-based index expression for one dim of a designate.
/// In the symbolic offset-symbol architecture, every memlet subtracts
/// ``offset_<arr>_d<dim>`` (declared by ``add_descriptors``).  The
/// array-level lower-bound rebase is therefore handled uniformly by
/// the offset symbol after ``sdfg.specialize``.
///
/// What this function still has to add to the raw index:
///   1. **Section-designate parent** (``hlfir.designate %inner (i)``
///      whose memref is ``hlfir.designate %a (lo:hi:stride)``): the
///      child's iter is local to the section, so we add ``(lo - 1)``
///      to map back to the root array's Fortran index.
///   2. **Assumed-shape alias** (``hlfir-inline-all`` splices a
///      callee's assumed-shape view into the caller; the callee's
///      view starts at lb=1 but the underlying storage uses the
///      caller's lb): the existing offset symbol on the resolved
///      OUTER array captures the caller's lb, so the callee's iter
///      ``i`` needs ``(lb_outer - 1)`` added so the final memlet
///      ``(i + lb_outer - 1) - offset_outer_d0`` collapses to
///      ``i - 1`` after specialise (correct for the callee's view).
 std::string buildDesignateIndexExpr(hlfir::DesignateOp dg,
                                           unsigned dim,
                                           mlir::Value idx,
                                           int depth) {
    std::string raw = buildIndexExpr(idx, depth);
    auto memref = dg.getMemref();
    auto *defOp = memref.getDefiningOp();
    if (!defOp) return raw;

    // Section-designate parent contribution.
    if (auto parentDg = mlir::dyn_cast<hlfir::DesignateOp>(defOp)) {
        auto triplets = parentDg.getIsTriplet();
        if (!triplets.empty() && dim < triplets.size() && triplets[dim]) {
            unsigned cursor = 0;
            for (unsigned k = 0; k < dim; ++k)
                cursor += triplets[k] ? 3 : 1;
            auto idxOps = parentDg.getIndices();
            if (cursor < idxOps.size()) {
                if (auto lo = traceConstInt(idxOps[cursor])) {
                    int64_t adjust = *lo - 1;
                    if (adjust > 0)
                        raw = "(" + raw + " + " + std::to_string(adjust) + ")";
                    else if (adjust < 0)
                        raw = "(" + raw + " - " + std::to_string(-adjust) + ")";
                } else {
                    // Section ``lo`` isn't a compile-time constant — typical
                    // shape is ``a(pos(1):pos(2))`` where ``pos(1)`` minted
                    // the symbol ``__sym_pos_1`` via ``buildIndexExpr``'s
                    // load-of-designate path.  Use that closed-form so the
                    // memlet stays expressible: rebase = ``+ (lo - 1)``.
                    auto loExpr = buildIndexExpr(idxOps[cursor], depth + 1);
                    if (!loExpr.empty() && loExpr != "?")
                        raw = "(" + raw + " + " + loExpr + " - 1)";
                }
            }
        }
        return raw;
    }

    // Assumed-shape alias contribution: the callee's view-offset is 1
    // (Fortran default for assumed-shape) but the resolved outer
    // array's offset is the caller's lb.  Add ``(lb_outer - 1)`` so
    // the memlet form gives the right element after specialise.
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

/// Capture the LHS of an ``hlfir.assign`` whose destination is either a
/// bare ``hlfir.declare`` (whole-result variable, e.g. ``out = SUM(a)``)
/// or an ``hlfir.designate`` selecting one element of an array
/// (``res(2) = MINVAL(d)``).  Writes the resolved name into
/// ``node.target`` and, for the designate case, appends a per-dim
/// ``AccessInfo`` so the downstream emitter wires the output memlet to
/// that specific element.  Without this, every libcall / reduction in
/// the routine writes through the whole destination array and the last
/// one wins (or pytest fails with "memlet subset does not match node
/// dimension" when the destination has more dims than the libcall's
/// scalar output).
///
/// Shared by ``buildReduceNode``, ``buildElementalCountLibcall``,
/// ``buildElementalAnyAllReduce`` and ``buildLibCallNode``.
 void captureElementDesignateWrite(mlir::Value dest, ASTNode &node) {
    if (auto dd = dest.getDefiningOp()) {
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd)) {
            node.target = allocAliasFor(extractName(decl.getUniqName().str()));
        } else if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(dd)) {
            node.target = traceToDecl(dg.getMemref());
            node.target_is_array = true;
            AccessInfo wa;
            wa.array_name = node.target;
            wa.is_write = true;
            unsigned di = 0;
            for (auto idx : dg.getIndices()) {
                auto resolved = resolveIndex(idx);
                wa.index_vars.push_back(resolved.empty() ? "?" : resolved);
                wa.index_exprs.push_back(buildDesignateIndexExpr(dg, di, idx, 0));
                ++di;
            }
            node.accesses.push_back(std::move(wa));
        }
    }
    if (node.target.empty()) node.target = traceToDecl(dest);
}

// ---------------------------------------------------------------------------
// Bridge-synthesised name conventions
// ---------------------------------------------------------------------------
//
// Synthetic names the bridge mints during AST extraction.  All start
// with ``_`` or ``__`` (reserved by the bridge) so they cannot collide
// with Flang-mangled Fortran names (which always start with ``_Q``
// followed by a Fortran-form identifier).
//
//   __sym_<arr>_<idx>   eager position-array symbol minted by
//                        ``buildIndexExpr`` for ``load(designate %arr (%const))``;
//                        load happens once at SDFG entry via a
//                        ``kind="symbol_init"`` AST node.
//                        Counter:  kPosSymbolRegistry (per-pair, deterministic).
//   __al_<n>            bare ``fir.alloca`` scratch (no surrounding declare),
//                        used as a synthetic scalar name so loads /
//                        stores of an unnamed alloca have something to
//                        reference in the AST.
//                        Counter:  kAllocaCounter.
//   __sc_<n>            scf.if synthetic scalar — a sink for the i-th
//                        ``scf.if`` result so downstream reads of the
//                        result Value resolve to a single name instead
//                        of recursing into both arms.
//                        Counter:  kScfValueCounter.
//   _count_mask_<n>     Mode-C COUNT mask transient (``COUNT(arr1.eq.arr2)``);
//                        per-element loop fills it, ``CountLibraryNode``
//                        reads it.
//                        Counter:  kSynthTransientCounter.
//   _mask_<n>           Mode-C ANY/ALL mask transient (``ANY(arr1.eq.arr2)``);
//                        same shape as count mask, terminated by a
//                        DaCe Reduce node instead of a libcall.
//                        Counter:  kSynthTransientCounter.
//   _libtmp_<n>         Libcall result transient inside an elemental —
//                        ``hlfir.matmul`` / ``hlfir.transpose`` /
//                        ``hlfir.dot_product`` materialised ahead of
//                        the elemental that consumes it via
//                        ``hlfir.apply``.
//                        Counter:  kLibTmpCounter.
//
// All counters are thread-local and reset in ``extractAST`` (dispatch.inc)
// at module-walk start so two consecutive bridge calls don't inherit
// one another's numbering.
//
// The Python emitter pattern-matches on these prefixes in a few
// places (e.g. indirect-symbol detection); keep the prefixes stable
// or update both sides.
// ---------------------------------------------------------------------------



 std::string allocaSynthName(mlir::Value memref) {
    auto *def = memref.getDefiningOp();
    if (!def) return "";
    auto it = kAllocaMap.find(def);
    if (it != kAllocaMap.end()) return it->second;
    std::string s = "__al_" + std::to_string(kAllocaCounter++);
    kAllocaMap[def] = s;
    return s;
}



/// Look up or mint the SDFG symbol name that stands in for
/// ``<array>(<one_based_idx>)`` (both arguments are Fortran-side
/// names / values).  Same key always yields the same symbol — callers
/// can safely use this anywhere the load result was needed before.
 std::string internPosSymbol(const std::string &array,
                                   int64_t one_based_idx) {
    auto k = std::make_pair(array, one_based_idx);
    auto it = kPosSymbolRegistry.find(k);
    if (it != kPosSymbolRegistry.end()) return it->second;
    std::string s = "__sym_" + array + "_" + std::to_string(one_based_idx);
    kPosSymbolRegistry[k] = s;
    return s;
}

 std::string buildExpr(mlir::Value val, int d) {
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

    // ``ALLOCATED(arr)`` — Flang lowers as
    //   %addr = fir.box_addr (fir.load arr_box) -> heap<...>
    //   %i64  = fir.convert %addr : heap<...> -> i64
    //   %r    = arith.cmpi ne, %i64, %c0_i64
    // Recognise that exact shape (cmpi-ne against constant-zero of a
    // box_addr→convert chain) and read the per-allocatable companion
    // ``<arr>_allocated`` scalar that ``extract_vars`` registers and
    // the AST builder maintains at ALLOCATE / DEALLOCATE sites.
    if (nm == "arith.cmpi" && def->getNumOperands() == 2) {
        auto pred = def->getAttrOfType<mlir::IntegerAttr>("predicate");
        constexpr int64_t kPredNe = 1;   // mlir::arith::CmpIPredicate::ne
        if (pred && pred.getInt() == kPredNe) {
            // Operand 1 must be a constant int 0 (the null pointer
            // sentinel after the heap-addr→i64 cast).
            bool rhsZero = false;
            if (auto c = traceConstInt(def->getOperand(1)))
                rhsZero = (*c == 0);
            if (rhsZero) {
                // Operand 0: peel fir.convert back to find a fir.box_addr.
                mlir::Value cur = def->getOperand(0);
                for (int i = 0; i < limits::kConvertChainDepth && cur; ++i) {
                    auto *cd = cur.getDefiningOp();
                    if (!cd) break;
                    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(cd))
                        { cur = cv.getValue(); continue; }
                    break;
                }
                if (cur) {
                    if (auto *cd = cur.getDefiningOp()) {
                        if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(cd)) {
                            // box_addr's operand is fir.load of a box
                            // ref; trace through that to the declare.
                            auto src = ba.getVal();
                            if (auto *sd = src.getDefiningOp())
                                if (auto ld = mlir::dyn_cast<fir::LoadOp>(sd))
                                    src = ld.getMemref();
                            auto arrName = traceToDecl(src);
                            if (!arrName.empty())
                                return arrName + "_allocated";
                        }
                    }
                }
            }
        }
    }

    // ``fir.box_dims %arr_decl, %dim`` — Flang's lowering for SIZE /
    // LBOUND / UBOUND / SHAPE on assumed-shape (and other boxed)
    // arrays.  Produces a 3-tuple ``(lower_bound, extent, stride)``;
    // each result is read out via an OpResult index, so we map per
    // result number to the corresponding bridge-synthesised symbol.
    //
    // For the underlying array's K-th dim:
    //   * ``#0`` (lower bound) → declared lb if present (``fir.shape_shift``),
    //                            otherwise Fortran-default ``1``.
    //   * ``#1`` (extent)      → ``<arr>_d<K>`` symbol the bridge mints
    //                            for assumed-shape arrays in extract_vars
    //                            (line 426-429).  For explicit-shape arrays
    //                            (``dimension(N)``), the declare's ``fir.shape``
    //                            already carries the constant / symbol; we
    //                            recover it via ``buildIndexExpr`` on the
    //                            extent operand.
    //   * ``#2`` (stride)      → ``1`` (assume contiguous; section
    //                            designates with non-1 stride don't reach
    //                            this path).
    if (nm == "fir.box_dims" && def->getNumOperands() >= 2) {
        auto resIdx = mlir::cast<mlir::OpResult>(val).getResultNumber();
        auto dimOp = def->getOperand(1);
        auto dimC = traceConstInt(dimOp);
        // Walk operand 0 back to the underlying ``hlfir.declare`` so we
        // can read its shape (for explicit-shape) or fall back to the
        // assumed-shape symbol form.
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
        std::string arrName = traceToDecl(arrayVal);
        if (!dimC || arrName.empty()) return "?";
        unsigned dim = static_cast<unsigned>(*dimC);

        // Try to get the extent / lb from the declare's shape operand.
        // For inlined assumed-shape callee aliases (no shape on the
        // inner declare), walk to the outer declare via
        // ``asAssumedShapeAlias`` — it shares storage with the caller
        // and carries the actual shape/lb info.
        mlir::Value shapeVal;
        if (auto *adef = arrayVal.getDefiningOp()) {
            if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(adef)) {
                shapeVal = decl.getShape();
                if (!shapeVal) {
                    if (auto outer = asAssumedShapeAlias(decl))
                        shapeVal = outer.getShape();
                }
            }
        }

        // Lower bound (#0):
        if (resIdx == 0) {
            if (shapeVal) {
                if (auto ss = mlir::dyn_cast<fir::ShapeShiftOp>(
                        shapeVal.getDefiningOp())) {
                    auto ops = ss->getOperands();
                    unsigned lbIdx = 2 * dim;
                    if (lbIdx < ops.size()) {
                        auto s = buildIndexExpr(ops[lbIdx], d + 1);
                        if (!s.empty() && s != "?") return s;
                    }
                }
            }
            return "1";
        }
        // Extent (#1):
        if (resIdx == 1) {
            if (shapeVal) {
                if (auto sh = mlir::dyn_cast<fir::ShapeOp>(
                        shapeVal.getDefiningOp())) {
                    if (dim < sh.getExtents().size()) {
                        auto s = buildIndexExpr(sh.getExtents()[dim], d + 1);
                        if (!s.empty() && s != "?") return s;
                    }
                }
                if (auto ss = mlir::dyn_cast<fir::ShapeShiftOp>(
                        shapeVal.getDefiningOp())) {
                    auto ops = ss->getOperands();
                    unsigned extIdx = 2 * dim + 1;
                    if (extIdx < ops.size()) {
                        auto s = buildIndexExpr(ops[extIdx], d + 1);
                        if (!s.empty() && s != "?") return s;
                    }
                }
            }
            // Assumed-shape (no declare shape) — the bridge synthesised
            // ``<arr>_d<dim>`` in extract_vars.  Same string convention.
            return arrName + "_d" + std::to_string(dim);
        }
        // Stride (#2): contiguous default.
        if (resIdx == 2) return "1";
    }

    // Binary arithmetic.
    static const std::map<llvm::StringRef, std::string> bin_ops = {
        {"arith.mulf", " * "}, {"arith.addf", " + "},
        {"arith.subf", " - "}, {"arith.divf", " / "},
        {"arith.muli", " * "}, {"arith.addi", " + "},
        {"arith.subi", " - "}, {"arith.divsi", " // "}, {"arith.divui", " // "},
        // Fortran COMPLEX arithmetic — flang emits dedicated ops on
        // ``complex<f32>`` / ``complex<f64>`` operands.
        {"fir.addc", " + "}, {"fir.subc", " - "},
        {"fir.mulc", " * "}, {"fir.divc", " / "},
    };
    if (auto it = bin_ops.find(nm); it != bin_ops.end()
            && def->getNumOperands() == 2) {
        return "(" + buildExpr(def->getOperand(0), d + 1)
                   + it->second
                   + buildExpr(def->getOperand(1), d + 1) + ")";
    }

    if (nm == "arith.negf" && def->getNumOperands() == 1)
        return "(-" + buildExpr(def->getOperand(0), d + 1) + ")";

    // Fortran ``conjg(z)`` lowers to:
    //     %im  = fir.extract_value %z, [1] : complex<T> -> T
    //     %neg = arith.negf %im
    //     %r   = fir.insert_value %z, %neg, [1]
    // Recognise the full idiom and emit ``<z>.conjugate()`` so the
    // tasklet renders the Python complex method.  DaCe's tasklet
    // codegen lowers ``.conjugate()`` to ``std::conj`` on
    // ``std::complex<T>``.
    if (auto ins = mlir::dyn_cast<fir::InsertValueOp>(def)) {
        auto coords = ins.getCoor();
        // Fortran ``cmplx(re, im, kind=K)`` lowers to:
        //   %base = fir.undefined complex<T>
        //   %r0   = fir.insert_value %base, %re, [0]
        //   %r1   = fir.insert_value %r0, %im, [1]
        // Recognise the outermost insert at coord [1] whose adt is an
        // insert at coord [0] of an ``fir.undefined`` and emit
        // ``complex(<re>, <im>)``.
        if (coords.size() == 1) {
            if (auto coordAttr = mlir::dyn_cast<mlir::IntegerAttr>(coords[0]))
                if (coordAttr.getInt() == 1) {
                    if (auto inner = mlir::dyn_cast_or_null<fir::InsertValueOp>(
                                         ins.getAdt().getDefiningOp())) {
                        auto innerCoords = inner.getCoor();
                        if (innerCoords.size() == 1)
                            if (auto a0 = mlir::dyn_cast<mlir::IntegerAttr>(innerCoords[0]))
                                if (a0.getInt() == 0)
                                    if (mlir::isa_and_nonnull<fir::UndefOp>(
                                            inner.getAdt().getDefiningOp())) {
                                        // Use the ``re + 1j*im`` form
                                        // rather than ``complex(re, im)``:
                                        // DaCe's tasklet C++ codegen
                                        // doesn't lower a free
                                        // ``complex(...)`` call to the
                                        // ``std::complex`` constructor,
                                        // but it does handle the
                                        // ``1j`` literal arithmetic via
                                        // its complex-arithmetic
                                        // rewrites.
                                        return "(("
                                             + buildExpr(inner.getVal(), d + 1)
                                             + ") + 1j * ("
                                             + buildExpr(ins.getVal(), d + 1)
                                             + "))";
                                    }
                    }
                }
        }
        if (coords.size() == 1) {
            // Coord must be the literal index 1 (the imaginary slot).
            if (auto coordAttr = mlir::dyn_cast<mlir::IntegerAttr>(coords[0]))
                if (coordAttr.getInt() == 1) {
                    auto val = ins.getVal();
                    auto adt = ins.getAdt();
                    if (auto neg = mlir::dyn_cast_or_null<mlir::arith::NegFOp>(
                                       val.getDefiningOp())) {
                        if (auto ext = mlir::dyn_cast_or_null<fir::ExtractValueOp>(
                                           neg.getOperand().getDefiningOp())) {
                            auto extCoords = ext.getCoor();
                            bool extIsImag = false;
                            if (extCoords.size() == 1)
                                if (auto a = mlir::dyn_cast<mlir::IntegerAttr>(extCoords[0]))
                                    extIsImag = (a.getInt() == 1);
                            if (extIsImag && ext.getAdt() == adt) {
                                // Emit ``conj(<expr>)`` — DaCe's tasklet
                                // codegen routes the bare name through
                                // ``dace::math::conj`` (defined in
                                // ``runtime/include/dace/math.h``) which
                                // forwards to ``std::conj`` for both
                                // ``std::complex<float>`` and
                                // ``std::complex<double>``.
                                return "conj(" + buildExpr(adt, d + 1) + ")";
                            }
                        }
                    }
                }
        }
    }

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
        // ``llvm.intr.<op>`` — LLVM-dialect intrinsic ops Flang uses
        // for some unary math (ANINT → ``llvm.intr.round``, AINT
        // → ``llvm.intr.trunc`` on some kinds, etc.).  These are
        // OPS, not function calls; the ``fir::CallOp`` table below
        // handles the ``fir.call @llvm.<op>.f{32,64}`` shape.
        {"llvm.intr.round", "round"},
        {"llvm.intr.trunc", "trunc"},
        {"llvm.intr.floor", "floor"},
        {"llvm.intr.ceil",  "ceil"},
        {"llvm.intr.fabs",  "abs"},
        {"llvm.intr.sqrt",  "sqrt"},
        {"llvm.intr.exp",   "exp"},
        {"llvm.intr.log",   "log"},
        {"llvm.intr.sin",   "sin"},
        {"llvm.intr.cos",   "cos"},
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
                // ``f``-suffixed f32 runtime variants Flang emits for
                // single-precision args: ``sinhf``, ``coshf``, etc.
                // Without these, ``real(4)`` SINH/COSH/TANH lowerings
                // hit the ``?`` fallback and the tasklet body fails to
                // parse.
                {"sinhf", "sinh"}, {"coshf", "cosh"}, {"tanhf", "tanh"},
                {"asinhf", "asinh"}, {"acoshf", "acosh"}, {"atanhf", "atanh"},
                {"asinf", "asin"}, {"acosf", "acos"}, {"atanf", "atan"},
                {"sinf", "sin"}, {"cosf", "cos"}, {"tanf", "tan"},
                {"expf", "exp"}, {"logf", "log"}, {"log10f", "log10"},
                {"sqrtf", "sqrt"}, {"fabsf", "abs"},
                // C99 complex math runtime — flang lowers Fortran
                // SIN/COS/EXP/LOG/SQRT/ABS on COMPLEX(8) to ``c<func>``
                // and on COMPLEX(4) to ``c<func>f``.  DaCe's tasklet
                // codegen has Python ``cmath``-equivalent dispatch via
                // the same bare names.
                {"csin", "sin"}, {"ccos", "cos"}, {"ctan", "tan"},
                {"csinh", "sinh"}, {"ccosh", "cosh"}, {"ctanh", "tanh"},
                {"casin", "asin"}, {"cacos", "acos"}, {"catan", "atan"},
                {"cexp", "exp"}, {"clog", "log"},
                {"csqrt", "sqrt"}, {"cabs", "abs"},
                {"csinf", "sin"}, {"ccosf", "cos"}, {"ctanf", "tan"},
                {"csinhf", "sinh"}, {"ccoshf", "cosh"}, {"ctanhf", "tanh"},
                {"casinf", "asin"}, {"cacosf", "acos"}, {"catanf", "atan"},
                {"cexpf", "exp"}, {"clogf", "log"},
                {"csqrtf", "sqrt"}, {"cabsf", "abs"},
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
            // Complex division — flang lowers ``a / b`` on COMPLEX(8) to
            // ``__divdc3(re_a, im_a, re_b, im_b)`` (and ``__divsc3`` for
            // COMPLEX(4)) for overflow-safe Smith's algorithm.  The 4
            // reals come from ``fir.extract_value`` ops on the loaded
            // complex operands; reconstruct the original complex
            // operand identities and emit ``(complex_a / complex_b)``
            // at the tasklet level.
            if ((cname == "__divdc3" || cname == "__divsc3")
                    && call.getNumOperands() == 4) {
                auto extractSource = [](mlir::Value re, mlir::Value im) -> mlir::Value {
                    auto reOp = mlir::dyn_cast_or_null<fir::ExtractValueOp>(re.getDefiningOp());
                    auto imOp = mlir::dyn_cast_or_null<fir::ExtractValueOp>(im.getDefiningOp());
                    if (!reOp || !imOp) return {};
                    if (reOp.getAdt() != imOp.getAdt()) return {};
                    return reOp.getAdt();
                };
                auto srcA = extractSource(call.getOperand(0), call.getOperand(1));
                auto srcB = extractSource(call.getOperand(2), call.getOperand(3));
                if (srcA && srcB) {
                    return "(" + buildExpr(srcA, d + 1) + " / "
                               + buildExpr(srcB, d + 1) + ")";
                }
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

    // ``fir.is_present`` — Fortran's ``present(x)`` on an OPTIONAL dummy.
    // Used as an integer (``res(1) = present(a)`` after the implicit
    // i1→i32 widening) as well as inside a guarding condition (handled
    // by buildBoolExpr).  Both sites trace through the same helper.
    if (auto isp = mlir::dyn_cast<fir::IsPresentOp>(def)) {
        auto e = lowerIsPresent(isp.getVal());
        if (!e.empty()) return e;
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
    // ``btest``).  i1 versions are Fortran ``.AND.`` / ``.OR.`` chains;
    // route through ``buildBoolExpr`` so they render as Python
    // ``... and ...`` / ``... or ...`` -- the typical shape is a
    // ``LOGICAL :: llo1`` cached as ``llo1 = (a>b) .AND. (c>d) .AND.
    // ...``, where the surrounding ``fir.convert`` to ``!fir.logical<K>``
    // pulls us through ``buildExpr`` rather than ``buildBoolExpr``.
    // ``NoSubscriptGuard`` keeps array reads as bare identifiers -- we
    // are inside a ``buildExpr`` call destined for a tasklet body, so
    // the cmpf / cmpi operands must NOT carry ``arr[idx]`` subscripts
    // (emit_tasklet rewrites bare names into ``_in_arr_N`` connectors
    // and wires subscripts via memlets).
    if ((nm == "arith.andi" || nm == "arith.ori") && def->getNumOperands() == 2) {
        if (def->getOperand(0).getType().isInteger(1)) {
            // Bare-name mode for the cmp-leaf array reads (we're inside
            // ``buildExpr``, the tasklet renderer; emit_tasklet wires
            // ``a[i]`` subscripts through memlets and rewrites the
            // bare ``a`` to a ``_in_a_N`` connector).
            NoSubscriptGuard _g;
            auto b = buildBoolExpr(val, d + 1);
            if (b != "?") return b;
        } else {
            const char *op = (nm == "arith.andi") ? " & " : " | ";
            return "(" + buildExpr(def->getOperand(0), d + 1) + op
                       + buildExpr(def->getOperand(1), d + 1) + ")";
        }
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
        //
        // ``buildExpr`` itself is the tasklet renderer (bare names),
        // so the cond's leaves must also be bare -- ``emit_tasklet``
        // will rewrite the connectors and wire subscripts via memlets.
        // Set ``NoSubscriptGuard`` for the ``buildBoolExpr`` call so
        // every leaf threads through bare-names mode, matching the
        // outer ``buildExpr`` calls for the select's true / false sides.
        std::string condExpr;
        {
            NoSubscriptGuard _g;
            condExpr = buildBoolExpr(sel.getCondition(), d + 1);
        }
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
        // Materialised libcall result (``matmul`` / ``transpose`` / …):
        // ``buildElementalAssign`` has already queued the libcall AST
        // node that writes the result to a synthetic transient; render
        // the apply as just the transient's bare name so emit_tasklet
        // rewrites it to an ``_in_<tmp>_<n>`` connector.  The indexing
        // lives entirely in the AccessInfo that ``collectReads`` adds
        // for this same apply (see the matching branch there).
        if (auto *srcDef = src.getDefiningOp()) {
            auto it = kHlfirExprToTransient.find(srcDef);
            if (it != kHlfirExprToTransient.end()) {
                return it->second;
            }
        }
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

}  // namespace hlfir_bridge
