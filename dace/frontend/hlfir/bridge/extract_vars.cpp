// ============================================================================
// extract_vars.cpp — Collect and classify every hlfir.declare.
// ============================================================================

#include "bridge/extract_vars.h"
#include "bridge/trace_utils.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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

    // Two unknown-extent sentinels reach the shape op as plain
    // ``arith.constant`` operands and must NOT be stringified into
    // the DaCe descriptor (which rejects negative extents with
    // "Found negative shape in Data"):
    //
    //   * ``fir::SequenceType::getUnknownExtent()`` (= INT64_MIN) —
    //     the canonical "this dim is dynamic" marker on
    //     ``fir.array<?xT>`` types.
    //   * ``-1`` — the convention flang uses on the shape op of an
    //     assumed-size dummy (``arr(*)``).  See the IR
    //     ``%shape = fir.shape %c-1 : (index) -> !fir.shape<1>``
    //     emitted for ``real, intent(in) :: src(*)`` — flang picks
    //     ``-1`` rather than ``INT64_MIN`` here because the operand
    //     is an ``index`` value the runtime would otherwise treat as
    //     a real extent.
    //
    // Either case → push ``"?"`` so the per-dim synthetic-name
    // fallback at the caller site mints ``<name>_d<i>``.  Any other
    // negative integer is genuinely invalid and we let it surface
    // (flang shouldn't emit such a thing for legal programs).
    auto pushExtent = [&](mlir::Value ext) {
        auto n = traceToDecl(ext);
        if (!n.empty()) { syms.push_back(n); return; }
        if (auto c = traceConstInt(ext)) {
            if (*c == fir::SequenceType::getUnknownExtent() || *c == -1) {
                syms.push_back("?");
                return;
            }
            syms.push_back(std::to_string(*c));
            return;
        }
        syms.push_back("?");
    };
    if (auto sh = mlir::dyn_cast_or_null<fir::ShapeOp>(shape.getDefiningOp()))
        for (auto ext : sh.getExtents())
            pushExtent(ext);

    if (auto ss = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shape.getDefiningOp())) {
        auto ops = ss->getOperands();
        for (unsigned i = 1; i < ops.size(); i += 2)
            pushExtent(ops[i]);
    }

    return syms;
}

/// Collect every ``fir.allocmem`` whose ``uniq_name`` matches
/// ``<declUniqName>.alloc``, in IR walk order.  Multiple matches indicate
/// that the user wrote more than one ``ALLOCATE`` for the variable
/// (e.g. across an explicit ``DEALLOCATE`` + re-``ALLOCATE``).
static std::vector<fir::AllocMemOp> collectAllocSites(
    const std::string &declName, mlir::ModuleOp module) {
    std::vector<fir::AllocMemOp> sites;
    if (declName.empty()) return sites;
    std::string allocName = declName + ".alloc";
    module.walk([&](fir::AllocMemOp a) {
        auto un = a.getUniqName();
        if (un && un->str() == allocName)
            sites.push_back(a);
    });
    return sites;
}

/// Resolve the runtime shape of one ``fir.allocmem`` site to a symbol
/// name list, the same way ``resolveShapeSyms`` resolves a static
/// declare's shape — trace each size operand to its host declare
/// (preferred), fall back to a constant literal, then to ``?``.
static std::vector<std::string> shapeFromAllocSite(fir::AllocMemOp alloc) {
    std::vector<std::string> syms;
    for (auto sz : alloc.getShape()) {
        auto n = traceToDecl(sz);
        if (!n.empty()) { syms.push_back(n); continue; }
        if (auto c = traceConstInt(sz)) {
            syms.push_back(std::to_string(*c));
            continue;
        }
        syms.push_back("?");
    }
    return syms;
}

/// True iff some ``fir.box_addr`` op in the module reads the
/// allocatable / pointer descriptor of the declare whose
/// (short, post-``extractName``) name is ``shortName``.
/// ``ALLOCATED(arr)`` and ``ASSOCIATED(ptr)`` both lower to
/// ``box_addr(load arr_box) != 0``; if no such reader exists the
/// per-allocatable ``<arr>_allocated`` tracker scalar and its init
/// state are dead weight in the SDFG.
static bool hasAllocatedReader(const std::string &shortName,
                               mlir::ModuleOp module) {
    if (shortName.empty()) return false;
    bool found = false;
    module.walk([&](fir::BoxAddrOp ba) {
        if (found) return;
        // ``box_addr``'s operand is normally a ``fir.load`` of a
        // box-ref; trace through that load to the declare.
        auto src = ba.getVal();
        if (auto *sd = src.getDefiningOp())
            if (auto ld = mlir::dyn_cast<fir::LoadOp>(sd))
                src = ld.getMemref();
        // ``traceToDecl`` returns the short (extracted) name — match
        // against ``shortName``, not the full mangled uniq_name.
        if (traceToDecl(src) == shortName) found = true;
    });
    return found;
}

/// True iff the allocatable / pointer ``declUniqName`` needs the
/// ``<short>_allocated`` tracker scalar — either because some
/// kernel-body code writes it (an ALLOCATE / DEALLOCATE site exists,
/// keyed on the full mangled uniq_name) OR because some kernel-body
/// code reads it (an ``ALLOCATED(arr)`` / ``ASSOCIATED(ptr)`` reader
/// exists, keyed on the short post-``extractName`` name).  Dummy /
/// module-level allocatables passed in already-allocated and never
/// queried by ``ALLOCATED(...)`` skip the tracker entirely.
bool needsAllocatedTracker(const std::string &declUniqName,
                           mlir::ModuleOp module) {
    if (declUniqName.empty()) return false;
    if (!collectAllocSites(declUniqName, module).empty()) return true;
    return hasAllocatedReader(extractName(declUniqName), module);
}

/// First ALLOCATE keeps the allocatable's original Fortran name (so
/// every existing single-allocation test stays green); subsequent
/// allocations mint fresh transient names ``<x>_alloc1``,
/// ``<x>_alloc2``, … one per re-allocation site.
std::string allocAliasName(const std::string &fortran, unsigned site) {
    if (site == 0) return fortran;
    return fortran + "_alloc" + std::to_string(site);
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
    // (fir.load + hlfir.declare) resolves to the Fortran name.  Only
    // collect INTEGER-typed scalars -- float and LOGICAL scalars used
    // in branch conditions (e.g. ``IF (zsupsat > zepsec)``,
    // ``IF (llo1)`` where ``llo1 = (a>b) .AND. (c>d) .AND. ...``) must
    // stay as plain scalars so their assignments route through the
    // tasklet path (which preserves complex RHS like
    // ``MAX((a-b*c)/d, 0)`` or a multi-AND boolean expression); the
    // interstate-edge path used for symbol writes only handles trivial
    // single-array-read RHSs, so promoting a non-integer scalar here
    // drops everything past the first array read in the expression.
    if (mlir::isa<fir::LoadOp>(def)) {
        if (v.getType().isIntOrIndex()) {
            auto n = traceToDecl(v);
            if (!n.empty()) out.insert(n);
        }
        return;
    }

    // Anything else (constants, arith.addi used as index arithmetic, …)
    // — trace through traceToDecl as a last resort; it already handles
    // several pass-through ops.  Same integer-only filter so non-integer
    // scalars don't get promoted to symbols here either.
    if (v.getType().isIntOrIndex()) {
        auto n = traceToDecl(v);
        if (!n.empty()) out.insert(n);
    }
}

// Extract the dense initial values of a ``fir.global ... constant``
// op into a flat ``std::vector<double>`` (row-major).  Returns an
// empty vector when the global isn't a recognisable constant pool
// entry (no initialiser, non-dense init, non-numeric element type).
//
// Background: Flang lowers Fortran array literals like
// ``(/ 2.0d0, 3.0d0, 4.0d0 /)`` to a read-only ``fir.global`` with
// a ``dense<[...]>`` attribute, addressed via ``fir.address_of``.
// We surface the data on the corresponding VarInfo so the SDFG
// builder can synthesise an init state writing those values into
// the transient — the kernel's reads then see the right data
// instead of zeros.
//
// All values widen to ``double`` for transport; the Python side
// narrows to the actual SDFG dtype (``int32`` / ``float32`` / ...)
// at descriptor-write time.
static std::vector<double> extractGlobalInitData(fir::GlobalOp gop) {
    std::vector<double> out;
    if (!gop) return out;
    // Path 1: ``fir.global ... constant`` arrays / scalars whose
    // initialiser lives on the op as a ``DenseElementsAttr`` —
    // Flang's encoding for ``parameter, dimension(...) :: x = (/ … /)``
    // and ``parameter :: x = <literal>``.  Restricted to globals
    // marked ``constant`` because the dense attribute is the
    // canonical static data.
    if (gop.getConstant().value_or(false)) {
        if (auto initOpt = gop.getInitVal()) {
            if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(*initOpt)) {
                auto eleTy = dense.getElementType();
                if (eleTy.isF64()) {
                    for (auto v : dense.getValues<double>()) out.push_back(v);
                } else if (eleTy.isF32()) {
                    for (auto v : dense.getValues<float>()) out.push_back((double)v);
                } else if (eleTy.isInteger(8)) {
                    for (auto v : dense.getValues<int8_t>()) out.push_back((double)v);
                } else if (eleTy.isInteger(16)) {
                    for (auto v : dense.getValues<int16_t>()) out.push_back((double)v);
                } else if (eleTy.isInteger(32)) {
                    for (auto v : dense.getValues<int32_t>()) out.push_back((double)v);
                } else if (eleTy.isInteger(64)) {
                    for (auto v : dense.getValues<int64_t>()) out.push_back((double)v);
                } else if (eleTy.isInteger(1)) {
                    for (auto v : dense.getValues<bool>()) out.push_back(v ? 1.0 : 0.0);
                }
                if (!out.empty()) return out;
            }
        }
    }
    // Path 2: scalar ``fir.global`` (e.g. ``real :: bob = 1`` declared
    // at module scope without ``parameter``).  The initialiser lives
    // in the body as an ``arith.constant`` feeding a ``fir.has_value``
    // terminator — extract the constant attribute, narrowing to a
    // single-element ``out`` vector.
    if (gop.getRegion().empty()) return out;
    for (auto &op : gop.getRegion().front()) {
        auto hv = mlir::dyn_cast<fir::HasValueOp>(op);
        if (!hv) continue;
        auto *def = hv.getResval().getDefiningOp();
        if (!def) return out;
        auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def);
        if (!cst) return out;
        auto attr = cst.getValue();
        if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
            out.push_back(fa.getValueAsDouble());
        } else if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
            out.push_back((double)ia.getInt());
        }
        return out;
    }
    return out;
}

// Trace a declare's memref back to the global it references via
// ``fir.address_of``.  Returns the symbol name (without leading
// ``@``) or empty string if the chain doesn't end at an address_of.
// Walks through ``fir.convert`` shims that flang occasionally
// inserts between the address_of and the declare's memref.
static std::string traceToGlobalSymbol(mlir::Value memref) {
    for (int i = 0; i < 8 && memref; ++i) {
        auto *d = memref.getDefiningOp();
        if (!d) return {};
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
            memref = cv.getValue();
            continue;
        }
        if (auto ad = mlir::dyn_cast<fir::AddrOfOp>(d)) {
            return ad.getSymbol().getRootReference().str();
        }
        return {};
    }
    return {};
}

// ---------------------------------------------------------------------------
// Main extraction
// ---------------------------------------------------------------------------

std::vector<VarInfo> extractVariables(mlir::ModuleOp module) {
    std::vector<VarInfo> vars;

    // Pass 0: disambiguate inlined-callee locals.  Two callees with the
    // same local name (Fortran's auto-generated ``a`` for ``result(a)``,
    // for example) get inlined into a common parent and surface as two
    // ``hlfir.declare`` ops with different full uniq_names but the same
    // ``extractName`` short name.  Downstream code keys SDFG arrays /
    // scalars by the short name; without disambiguation the two
    // declares race on a single access node.  Walk all declares in
    // public functions, group by short name, and rewrite the uniq_name
    // of every duplicate to encode its source-callee scope.
    {
        llvm::StringMap<llvm::SmallVector<hlfir::DeclareOp, 2>> byShort;
        module.walk([&](hlfir::DeclareOp op) {
            auto *fn = op->getParentOfType<mlir::func::FuncOp>().getOperation();
            if (auto f = mlir::dyn_cast_or_null<mlir::func::FuncOp>(fn))
                if (f.isPrivate()) return;
            // Only disambiguate declares backed by a fresh
            // ``fir.alloca`` — those are real own-storage locals
            // (the inlined function-result variable shape we care
            // about).  Aliases (declare-of-declare, embox/convert
            // chain, ``fir.absent``-backed optional dummies, and
            // dummy-arg block-arguments) point at storage that is
            // already named elsewhere; renaming them mints a
            // phantom flat scalar that downstream extract_vars
            // would surface as a top-level program kwarg.
            auto *def = op.getMemref().getDefiningOp();
            if (!def || !mlir::isa<fir::AllocaOp>(def)) return;
            byShort[extractName(op.getUniqName().str())].push_back(op);
        });
        auto getFScope = [](llvm::StringRef un) -> std::string {
            auto eP = un.rfind('E');
            if (eP == llvm::StringRef::npos) return {};
            auto fP = un.rfind('F', eP);
            if (fP == llvm::StringRef::npos || fP + 1 >= eP) return {};
            return un.substr(fP + 1, eP - fP - 1).str();
        };
        for (auto &kv : byShort) {
            auto &group = kv.second;
            if (group.size() < 2) continue;
            // Only rename if duplicates span DIFFERENT F-scopes — that's
            // the inlined-callee collision shape.  Two declares with
            // matching short name AND matching F-scope (one func
            // making two declares for one variable, e.g. shape-hint
            // copies) are legitimate; leave them alone so extract_vars
            // dedup downstream handles them.
            std::string firstScope = getFScope(group.front().getUniqName().str());
            bool sameScope = true;
            for (auto op : group) {
                if (getFScope(op.getUniqName().str()) != firstScope) {
                    sameScope = false;
                    break;
                }
            }
            if (sameScope) continue;
            // Rename each declare whose F-scope differs from the entry
            // function's scope.  The entry's declare keeps its original
            // short name; inlined-callee siblings get
            // ``<callee_scope>_<short>``.  Entry = the single public
            // ``func.func`` left in the module (set_entry_symbol made
            // every other function private).  Match its symbol name
            // tail against the F-scope segment of each declare.
            std::string entryScope;
            for (auto fn : module.getOps<mlir::func::FuncOp>()) {
                if (fn.isPrivate()) continue;
                auto sn = fn.getSymName().str();
                // Symbol like ``_QPmain`` or ``_QMmodPname``: the
                // function-name segment lives between the last ``P``
                // and end-of-string.  Match against ``getFScope``,
                // which pulls the F-segment from a declare uniq_name.
                auto pPos = sn.rfind('P');
                if (pPos == std::string::npos) continue;
                entryScope = sn.substr(pPos + 1);
                break;
            }
            for (auto op : group) {
                auto un = op.getUniqName().str();
                std::string scope = getFScope(un);
                if (scope == entryScope) continue;  // keep entry's name
                auto eP = un.rfind('E');
                std::string shortNm = un.substr(eP + 1);
                std::string newShort = scope + "_" + shortNm;
                std::string newUniq = un.substr(0, eP + 1) + newShort;
                op->setAttr("uniq_name",
                            mlir::StringAttr::get(op.getContext(), newUniq));
            }
        }
    }

    // Pass 1: collect every hlfir.declare.  Skip assumed-shape alias
    // declares inserted by ``hlfir-inline-all`` — they share storage
    // with the caller's outer declare, and downstream SDFG emission
    // routes accesses to the outer name via traceToDecl.  Registering
    // both would give DaCe two non-transient arrays over one buffer.
    std::vector<hlfir::DeclareOp> decls;
    module.walk([&](hlfir::DeclareOp op) {
        // Skip declares inside private functions.  The bridge only
        // builds an SDFG for the single public entry; callees that
        // were already inlined into it leave behind their original
        // bodies as private siblings (kept alive only by a
        // dispatch_table after ``fir-polymorphic-op`` resolved the
        // callsites).  Their dummy declares — typed e.g.
        // ``fir.class<T>`` — would otherwise surface as phantom
        // top-level program args at SDFG-build time.
        auto *parentOp = op->getParentOfType<mlir::func::FuncOp>().getOperation();
        if (auto fn = mlir::dyn_cast_or_null<mlir::func::FuncOp>(parentOp))
            if (fn.isPrivate()) return;
        if (asAssumedShapeAlias(op)) return;
        // Skip Flang-synthesised array-constructor temporaries
        // (``.tmp.arrayctor`` etc.) -- those are heap-allocated buffers
        // that ``dispatch.cpp`` recognises and lowers via per-element
        // assigns to the user's destination.  Registering them here
        // would surface ``.tmp.arrayctor`` on the SDFG and downstream
        // memlet parsing rejects the dotted name.
        if (op.getUniqName().str().find(".tmp.") != std::string::npos) return;
        // Skip Flang-internal type-info metadata declares — these are
        // string descriptors emitted for every derived type and its
        // components (``.n.<typename>``, ``.n.<field>``, ``.b.<type>``,
        // ``.di.<type>``).  They never represent user variables and
        // their dotted names break DaCe's ``NestedDict`` (which
        // interprets dots as nested-key separators).  Filter once
        // here so the rest of the pipeline never sees them.
        {
            auto un = op.getUniqName().str();
            auto p = un.rfind('E');
            llvm::StringRef tail = (p != std::string::npos)
                                       ? llvm::StringRef(un).drop_front(p + 1)
                                       : llvm::StringRef(un);
            if (tail.starts_with(".n.") || tail.starts_with(".b.")
                    || tail.starts_with(".di.") || tail.starts_with(".dt."))
                return;
        }
        // Drop unused SCALAR dummy arguments.  A subroutine like
        // ``subroutine main(arg1, arg2, res1) ; res1 = exp(arg1)``
        // (verbatim-port test pattern) leaves ``arg2`` declared but
        // never read or written; ``hlfir-default-intent`` adds
        // ``intent_inout`` to every dummy, so a "drop only if no
        // explicit intent" guard would keep ``arg2`` and the SDFG
        // signature would break Python callers that (correctly)
        // didn't pass it.
        //
        // Restrict the filter to *scalar* dummies (and to dummies
        // whose declare result has rank 0).  Arrays are kept
        // unconditionally even when ``size(a)``-style references
        // get folded by ``hlfir-propagate-shapes``: the array dummy
        // may be the sole carrier of shape symbols for other dummies
        // (``a(n, m)`` where ``m`` is used as an SDFG symbol via
        // ``a``'s extent), and dropping ``a`` breaks the symbol
        // classification cascade.
        auto resTy = op.getResult(0).getType();
        bool isArrayLike = false;
        for (int i = 0; i < limits::kTypeWrapperPeelDepth; ++i) {
            if (auto bt = mlir::dyn_cast<fir::BoxType>(resTy)) { resTy = bt.getEleTy(); continue; }
            if (auto rt = mlir::dyn_cast<fir::ReferenceType>(resTy)) { resTy = rt.getEleTy(); continue; }
            if (auto ht = mlir::dyn_cast<fir::HeapType>(resTy)) { resTy = ht.getEleTy(); continue; }
            if (auto pt = mlir::dyn_cast<fir::PointerType>(resTy)) { resTy = pt.getEleTy(); continue; }
            break;
        }
        if (mlir::isa<fir::SequenceType>(resTy)) isArrayLike = true;
        if (op.getDummyScope() && !isArrayLike
            && op.getResult(0).use_empty()
            && op.getResult(1).use_empty()) {
            return;
        }
        decls.push_back(op);
    });

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

    // Pass 2b: shape symbols + do-loop bounds (both lower and upper).
    // Lower bounds are promoted symmetrically with upper bounds so
    // ``DO jk = nflatlev, nlev`` recognises ``nflatlev`` as a symbol —
    // otherwise codegen generates an int*-vs-int64_t mismatch in the
    // loop initialiser.
    for (auto &op : decls)
        for (auto &s : resolveShapeSyms(op))
            if (s != "?") symbolNames.insert(s);
    module.walk([&](fir::DoLoopOp lp) {
        auto ub = traceToDecl(lp.getUpperBound());
        if (!ub.empty()) symbolNames.insert(ub);
        auto lb = traceToDecl(lp.getLowerBound());
        if (!lb.empty()) symbolNames.insert(lb);
    });
    // Allocatable shape sources: every ``fir.allocmem`` site's shape
    // operands are runtime extents of the resulting array — promote
    // their traced declares to symbols so ``allocate(x(n))``
    // (without any surrounding do-loop) still flips ``n`` from scalar
    // to symbol.  Bug fix for Phase 5a (allocatable struct members):
    // ``s%w`` allocates only at the explicit ``allocate(s%w(n))``
    // statement and may have no other use of ``n``, so neither
    // ``resolveShapeSyms`` (declare has no shape) nor the do-loop
    // pass picks up ``n``.  Without this walk, ``n`` lands as a
    // ``scalar`` data-descriptor and collides with the symbol the
    // SDFG construction step then tries to emit for the array
    // extent.
    module.walk([&](fir::AllocMemOp am) {
        for (auto sz : am.getShape()) {
            auto n = traceToDecl(sz);
            if (!n.empty()) symbolNames.insert(n);
        }
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
            // An OPTIONAL dummy without an explicit intent is still a
            // dummy -- treat it as ``intent(in)`` by default so
            // descriptors.py doesn't misclassify it as a transient
            // local.  The Fortran spec allows any intent for an
            // unspecified OPTIONAL; ``in`` is the common case (and
            // widens safely to ``inout`` via the caller's own buffer).
            if (v.intent.empty()
                && bitEnumContainsAny(fa, fir::FortranVariableFlagsEnum::optional))
                v.intent = "in";
            // ``REAL(8), VALUE :: x`` is a C-interop scalar passed by
            // value -- equivalent to intent(in) since the callee gets
            // its own copy.  Mark intent so the rank-0 path doesn't
            // misclassify it as a transient.  Below (after the
            // role-classification block) we further promote VALUE
            // scalars to SDFG SYMBOLS so callers can bind them with
            // plain Python int / float instead of a 1-element numpy
            // array.
            if (v.intent.empty()
                && bitEnumContainsAny(fa, fir::FortranVariableFlagsEnum::value))
                v.intent = "in";
        }

        // Unwrap FIR type wrappers to find element type + rank.
        //
        // Plain dummy / local arrays surface a single layer (Box, Ref,
        // Heap, or Ptr) over the SequenceType, so a single sequential
        // unwrap suffices.  Allocatable declares add two extra layers
        // (``ref<box<heap<array<…>>>>``); loop through the wrappers
        // only when the declare is allocatable so POINTER and other
        // box-typed dummies stay rank-0 (scalar passthrough).
        auto ty = op.getResult(0).getType();
        bool isAllocatableAttr = false;
        bool isPointerAttr = false;
        if (auto a = op.getFortranAttrs()) {
            if (bitEnumContainsAny(*a, fir::FortranVariableFlagsEnum::allocatable))
                isAllocatableAttr = true;
            if (bitEnumContainsAny(*a, fir::FortranVariableFlagsEnum::pointer))
                isPointerAttr = true;
        }
        // Pointer declares peel the same way as allocatables: their
        // declared type is ``ref<box<ptr<array<?xT>>>>`` and we want
        // the inner array's element type + rank for the SDFG
        // descriptor.  Without peeling, a top-level ``real, pointer
        // :: w(:)`` (or a Phase 5b flat companion ``s_w`` for a
        // pointer struct member) ends up classified as a scalar of
        // dtype ``!fir.box<!fir.ptr<...>>`` — useless to the SDFG.
        //
        // Guard: only peel pointer declares whose results are
        // actually USED downstream.  Pointer declares with all-empty
        // results survive ``hlfir-rewrite-pointer-assigns`` only as
        // dangling artifacts (rebind successfully collapsed → all
        // reads forwarded → declare is dead but not yet erased) or
        // as cross-procedure / unsupported-target leftovers.  Peel
        // always exposes a phantom rank>0 array on the SDFG
        // signature; without the guard, even a successfully-collapsed
        // pointer demanded its own ``_d0`` symbols.
        bool peelPointer = false;
        if (isPointerAttr) {
            if (!op.getResult(0).use_empty() || !op.getResult(1).use_empty())
                peelPointer = true;
        }
        if (isAllocatableAttr || peelPointer) {
            for (int peel = 0; peel < 6; ++peel) {
                if (auto b = mlir::dyn_cast<fir::BoxType>(ty))       { ty = b.getEleTy(); continue; }
                if (auto r = mlir::dyn_cast<fir::ReferenceType>(ty)) { ty = r.getEleTy(); continue; }
                if (auto h = mlir::dyn_cast<fir::HeapType>(ty))      { ty = h.getEleTy(); continue; }
                if (auto p = mlir::dyn_cast<fir::PointerType>(ty))   { ty = p.getEleTy(); continue; }
                break;
            }
        } else {
            if (auto b = mlir::dyn_cast<fir::BoxType>(ty)) ty = b.getEleTy();
            if (auto r = mlir::dyn_cast<fir::ReferenceType>(ty)) ty = r.getEleTy();
            if (auto h = mlir::dyn_cast<fir::HeapType>(ty)) ty = h.getEleTy();
            if (auto p = mlir::dyn_cast<fir::PointerType>(ty)) ty = p.getEleTy();
        }
        // Capture the SequenceType's per-dim extents as a fallback for
        // ``shape_symbols``: a declare synthesised by ``hlfir-flatten-structs``
        // for a per-field array carries the shape only in the type
        // (``!fir.array<5x5x5xf32>``), not as an explicit ``fir.shape``
        // operand.  Without this, ``resolveShapeSyms`` returns empty
        // and the fallback assumed-shape ``<name>_d<i>`` synth fires —
        // but those synth symbols would be unwired because the extent
        // is statically known.
        std::vector<std::string> seqExtents;
        if (auto seq = mlir::dyn_cast<fir::SequenceType>(ty)) {
            for (auto d : seq.getShape()) {
                if (d == fir::SequenceType::getUnknownExtent()) {
                    v.is_dynamic = true;
                    seqExtents.push_back("?");
                } else {
                    seqExtents.push_back(std::to_string(d));
                }
            }
            v.rank = seq.getShape().size();
            ty = seq.getEleTy();
        }

        // Element type string.
        if (ty.isF64())            v.dtype = "float64";
        else if (ty.isF32())       v.dtype = "float32";
        else if (ty.isInteger(8))  v.dtype = "int8";   // Fortran INTEGER(1)
        else if (ty.isInteger(16)) v.dtype = "int16";  // Fortran INTEGER(2)
        else if (ty.isInteger(32)) v.dtype = "int32";
        else if (ty.isInteger(64)) v.dtype = "int64";
        // Fortran ``COMPLEX(kind)`` lowers to ``mlir::ComplexType`` over
        // an ``f32`` / ``f64`` element.  DaCe has native ``complex64`` /
        // ``complex128`` dtypes that match numpy's ABI.
        else if (auto ct = mlir::dyn_cast<mlir::ComplexType>(ty)) {
            auto et = ct.getElementType();
            if (et.isF32())      v.dtype = "complex64";
            else if (et.isF64()) v.dtype = "complex128";
            else { std::string s; llvm::raw_string_ostream os(s);
                   ty.print(os); v.dtype = s; }
        }
        // MLIR ``i1`` and Fortran ``LOGICAL(KIND=N)`` (any kind) both
        // surface as ``bool`` on the SDFG signature (= ``np.bool_`` =
        // C++ ``bool``, 1 byte).  Element-wise boolean ops in tasklets
        // render as ``bool`` operations directly — no ``(x != 0)``
        // truthiness coercion needed.  The caller-side bindings
        // wrapper translates between the original ``LOGICAL(KIND=N)``
        // image and the SDFG's bool layout at the Fortran boundary.
        else if (ty.isInteger(1))  v.dtype = "bool";
        else if (mlir::isa<fir::LogicalType>(ty)) {
            v.dtype = "bool";
        }
        else if (mlir::isa<fir::RecordType>(ty)) {
            // Drop ALL ``fir.RecordType`` declares.  Two cases:
            //
            //   1. Flang-internal type-info metadata
            //      (``_QM__fortran_type_info...`` tables, component
            //      descriptors named ``.b.<type>.<field>``) — never
            //      user-visible.
            //   2. User struct that escaped ``hlfir-flatten-structs``
            //      (the pass handles flat-member structs and nested
            //      records; allocatable-member structs and other
            //      shapes are out of scope).
            //
            // Either way the struct does not belong on the SDFG
            // signature: writing the raw MLIR type string into
            // ``v.dtype`` would produce broken descriptors that
            // downstream codegen would only sometimes tolerate.  Skip
            // the VarInfo entirely; downstream ``traceToDecl`` reads
            // through the per-field declares the pass did lower.  A
            // loud-failure throw here would be ideal but regresses
            // tests that exploit the accidental-success path, so the
            // loud check lives in a dedicated unit test instead.
            continue;
        }
        else {
            std::string s; llvm::raw_string_ostream os(s);
            ty.print(os); v.dtype = s;
        }

        v.shape_symbols = resolveShapeSyms(op);
        v.lower_bounds  = resolveLowerBounds(op);

        // SequenceType-extent fallback: a declare with no ``fir.shape``
        // operand (e.g. one synthesised by ``hlfir-flatten-structs`` for
        // a per-field array) still carries concrete extents in its type.
        // Use them when ``resolveShapeSyms`` came back empty so the
        // SDFG signature gets literal shape (``[5,5,5]``) instead of a
        // free symbol per dim that the caller must bind manually.
        if (v.shape_symbols.empty() && !seqExtents.empty()) {
            v.shape_symbols = seqExtents;
            if (v.lower_bounds.size() != v.shape_symbols.size())
                v.lower_bounds.assign(v.shape_symbols.size(), "1");
        }

        // Allocatable: hlfir.declare has no shape; pull it from the
        // matching ``fir.allocmem`` site(s).  One ALLOCATE → use the
        // first site for ``x``'s shape.  Multiple ALLOCATEs (re-
        // allocation across an explicit DEALLOCATE) → register one
        // extra synthetic VarInfo per additional site, named
        // ``x_alloc1``, ``x_alloc2``, … (allocAliasName); the bridge's
        // alias map (see extract_ast.cpp) will route per-site reads /
        // writes to the right transient at AST-build time.
        bool isAllocatable = false;
        if (auto a = op.getFortranAttrs())
            if (bitEnumContainsAny(*a, fir::FortranVariableFlagsEnum::allocatable))
                isAllocatable = true;
        std::vector<fir::AllocMemOp> allocSites;
        if (isAllocatable && v.rank > 0)
            allocSites = collectAllocSites(v.mangled_name, module);
        if (!allocSites.empty()
            && (v.shape_symbols.empty()
                || std::all_of(v.shape_symbols.begin(), v.shape_symbols.end(),
                               [](const std::string &s){ return s == "?"; }))) {
            auto from_alloc = shapeFromAllocSite(allocSites.front());
            if (!from_alloc.empty()) {
                v.shape_symbols = std::move(from_alloc);
                if (v.lower_bounds.size() != v.shape_symbols.size())
                    v.lower_bounds.assign(v.shape_symbols.size(), "1");
            }
        }

        // Assumed-shape fallback: synthesise per-dim symbol names.
        // Two entry shapes:
        //   * ``shape_symbols`` is empty entirely (no shape op on the
        //     declare) — synthesize all dims.
        //   * ``shape_symbols`` has per-dim ``"?"`` slots (an
        //     unknown-extent sentinel reached us, e.g. assumed-size
        //     ``arr(*)``) — replace just the unresolved slots, keep
        //     the resolved ones.
        if (v.shape_symbols.empty() && v.rank > 0)
            for (int dim = 0; dim < v.rank; ++dim)
                v.shape_symbols.push_back(
                    v.fortran_name + "_d" + std::to_string(dim));
        else
            for (size_t dim = 0; dim < v.shape_symbols.size(); ++dim)
                if (v.shape_symbols[dim] == "?")
                    v.shape_symbols[dim] =
                        v.fortran_name + "_d" + std::to_string(dim);

        // Classify.
        if (v.rank > 0)                             v.role = "array";
        else if (symbolNames.count(v.fortran_name)) v.role = "symbol";
        else                                        v.role = "scalar";

        // OPTIONAL dummy → companion presence flag.  Fortran's
        // ``present(x)`` lowers to ``fir.is_present %x -> i1``, and the
        // bridge renders that as the name ``<x>_present``.  Register a
        // symbol VarInfo for that name here so callers see it on the
        // SDFG signature (non-zero = present, 0 = absent).  We register
        // it BEFORE pushing v, since the caller position should follow
        // the Fortran dummy order — the flag sits alongside its host.
        bool isOptional = false;
        if (auto a = op.getFortranAttrs()) {
            if (bitEnumContainsAny(*a, fir::FortranVariableFlagsEnum::optional))
                isOptional = true;
        }
        if (isOptional) {
            VarInfo pv;
            pv.fortran_name = v.fortran_name + "_present";
            pv.mangled_name = v.mangled_name + "_present";
            pv.dtype        = "int32";  // plain Fortran integer
            pv.rank         = 0;
            pv.intent       = "in";
            pv.role         = "symbol";
            vars.push_back(std::move(pv));
        }

        // Companion ``<arr>_allocated`` int32 transient for every
        // allocatable.  The AST builder writes ``1`` at each ALLOCATE
        // site and ``0`` at each DEALLOCATE site so the Fortran
        // ``ALLOCATED(arr)`` intrinsic — which Flang lowers to
        // ``box_addr(load arr_box) != 0`` — can read this scalar
        // instead of inspecting the descriptor's heap pointer (which
        // DaCe's data model doesn't surface).  Initial value is 0
        // (DaCe default for transient scalars).
        if (isAllocatable && needsAllocatedTracker(v.mangled_name, module)) {
            // Role ``symbol`` (not ``scalar``) so writes land on
            // interstate edges and reads see the latest value across
            // state boundaries.  A plain transient scalar would let
            // DaCe's intra-state DAG scheduler interleave the
            // ALLOCATE-time write with surrounding ``ALLOCATED(arr)``
            // reads, producing the wrong intermediate value.  Symbols
            // also auto-register on the SDFG signature, so no extra
            // ``add_symbol`` plumbing is needed.
            VarInfo av;
            av.fortran_name = v.fortran_name + "_allocated";
            av.mangled_name = v.mangled_name + "_allocated";
            av.dtype        = "int32";
            av.rank         = 0;
            av.intent       = "";
            av.role         = "symbol";
            symbolNames.insert(av.fortran_name);
            vars.push_back(std::move(av));
        }

        // For an allocatable with N ALLOCATE sites, register N-1
        // additional synthetic transients alongside the primary
        // VarInfo.  Each gets the per-site shape (n1, n2, …) and the
        // ``x_allocK`` alias name; the AST builder will redirect reads
        // / writes after the K-th ALLOCATE through this name.
        if (allocSites.size() > 1) {
            for (unsigned site = 1; site < allocSites.size(); ++site) {
                VarInfo av;
                av.fortran_name  = allocAliasName(v.fortran_name, site);
                av.mangled_name  = v.mangled_name + "_alloc" + std::to_string(site);
                av.intent        = "";        // local transient, no caller-side ABI
                av.dtype         = v.dtype;
                av.rank          = v.rank;
                av.is_dynamic    = v.is_dynamic;
                av.shape_symbols = shapeFromAllocSite(allocSites[site]);
                if (av.shape_symbols.size() < (size_t)av.rank)
                    av.shape_symbols.assign(av.rank, "?");
                av.lower_bounds.assign(av.shape_symbols.size(), "1");
                av.role          = "array";
                vars.push_back(std::move(av));
            }
        }

        // Init-value detection.  Two shapes feed the same path:
        //   * ``parameter`` declares pointing at ``fir.global ... constant``
        //     — the read-only constant pool Flang synthesises for array
        //     / scalar literals.
        //   * Plain module-data declares pointing at ``fir.global`` with
        //     a ``fir.has_value`` body init (Fortran's ``real :: bob = 1``
        //     at module scope, no ``parameter`` attribute).
        // ``extractGlobalInitData`` covers both.  The SDFG side treats
        // the data as the transient's initial-value vector; writes to
        // the variable still flow through normally.
        std::string sym = traceToGlobalSymbol(op.getMemref());
        if (!sym.empty()) {
            auto gop = module.lookupSymbol<fir::GlobalOp>(sym);
            v.const_data = extractGlobalInitData(gop);
        }

        vars.push_back(std::move(v));
    }
    return vars;
}

}  // namespace hlfir_bridge
