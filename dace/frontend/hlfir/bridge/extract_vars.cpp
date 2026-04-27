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
    // collect INTEGER-typed scalars -- float scalars used in branch
    // conditions (e.g. ``IF (zsupsat > zepsec)``) must stay as plain
    // scalars so their assignments route through the tasklet path
    // (which preserves complex RHS like ``MAX((a-b*c)/d, 0)``); the
    // interstate-edge path used for symbol writes only handles trivial
    // RHSs, so promoting a float scalar here drops the MAX expression.
    if (mlir::isa<fir::LoadOp>(def)) {
        if (v.getType().isIntOrIndex()) {
            auto n = traceToDecl(v);
            if (!n.empty()) out.insert(n);
        }
        return;
    }

    // Anything else (constants, arith.addi used as index arithmetic, …)
    // — trace through traceToDecl as a last resort; it already handles
    // several pass-through ops.  Same integer-only filter so float
    // scalars don't get promoted to symbols here either.
    if (v.getType().isIntOrIndex()) {
        auto n = traceToDecl(v);
        if (!n.empty()) out.insert(n);
    }
}

// ---------------------------------------------------------------------------
// Main extraction
// ---------------------------------------------------------------------------

std::vector<VarInfo> extractVariables(mlir::ModuleOp module) {
    std::vector<VarInfo> vars;

    // Pass 1: collect every hlfir.declare.  Skip assumed-shape alias
    // declares inserted by ``hlfir-inline-all`` — they share storage
    // with the caller's outer declare, and downstream SDFG emission
    // routes accesses to the outer name via traceToDecl.  Registering
    // both would give DaCe two non-transient arrays over one buffer.
    std::vector<hlfir::DeclareOp> decls;
    module.walk([&](hlfir::DeclareOp op) {
        if (asAssumedShapeAlias(op)) return;
        // Drop unused SCALAR dummy arguments.  A subroutine like
        // ``subroutine main(arg1, arg2, res1) ; res1 = exp(arg1)``
        // (verbatim-port test pattern) leaves ``arg2`` declared but
        // never read or written; with ``hlfir-default-intent`` adding
        // ``intent_inout`` to every dummy, the older "drop only if
        // no explicit intent" guard kept ``arg2`` and the SDFG
        // signature broke Python callers that (correctly) didn't
        // pass it.
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
        // Heap, or Ptr) over the SequenceType, so the original
        // sequential ``if``s suffice — preserving every non-allocatable
        // declare's existing classification.  Allocatable declares add
        // two extra layers (``ref<box<heap<array<…>>>>``); only loop
        // through the wrappers when we know the declare is allocatable
        // so POINTER and other box-typed dummies keep their previous
        // (rank = 0 → scalar passthrough) classification.
        auto ty = op.getResult(0).getType();
        bool isAllocatableAttr = false;
        if (auto a = op.getFortranAttrs())
            if (bitEnumContainsAny(*a, fir::FortranVariableFlagsEnum::allocatable))
                isAllocatableAttr = true;
        if (isAllocatableAttr) {
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
        // operand.  Without this, ``resolveShapeSyms`` returns empty and
        // we fall through to the assumed-shape ``<name>_d<i>`` synth —
        // but those symbols are never wired to anything because the
        // extent is statically known.
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
        else if (ty.isInteger(32)) v.dtype = "int32";
        else if (ty.isInteger(64)) v.dtype = "int64";
        // 1-bit int is the MLIR ``i1`` bool.  Surface as ``uint8`` rather
        // than ``bool`` so numpy / DaCe / f2py all agree on a 1-byte
        // storage layout — saves the caller-side dtype coercion.
        else if (ty.isInteger(1))  v.dtype = "uint8";
        // Fortran ``LOGICAL(kind)`` → ``kind``-byte storage.  Default
        // LOGICAL is 4 bytes (LOGICAL(4)) which matches f2py's ABI
        // convention; surface sized-logicals as the equivalent signed
        // integer so mask arrays round-trip without dtype gymnastics.
        else if (auto lt = mlir::dyn_cast<fir::LogicalType>(ty)) {
            auto kind = lt.getFKind();
            if (kind == 1)      v.dtype = "uint8";
            else if (kind == 4) v.dtype = "int32";
            else if (kind == 8) v.dtype = "int64";
            else                v.dtype = "uint8";
        }
        else if (mlir::isa<fir::RecordType>(ty)) {
            // Drop ALL ``fir.RecordType`` declares.  Two cases:
            //
            //   1. Flang-internal type-info metadata
            //      (``_QM__fortran_type_info...`` tables, component
            //      descriptors named ``.b.<type>.<field>``) — never
            //      user-visible.
            //   2. User struct that escaped ``hlfir-flatten-structs``
            //      (Phase 1 handles flat-member structs only; nested,
            //      allocatable-member, and array-of-struct cases fall
            //      to Phases 2–4).
            //
            // Either way we don't surface the struct on the SDFG
            // signature: the previous behaviour of writing the raw
            // MLIR type string into ``v.dtype`` produced broken
            // descriptors that downstream codegen sometimes tolerated
            // by accident.  Skip the VarInfo entirely; downstream
            // ``traceToDecl`` reads through the per-field declares
            // the pass DID lower.  A loud-failure throw here would
            // be ideal but currently regresses tests that exploit the
            // accidental-success path, so the loud check lives in a
            // dedicated unit test (``derived_type_test.py``) instead.
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
        if (v.shape_symbols.empty() && v.rank > 0)
            for (int dim = 0; dim < v.rank; ++dim)
                v.shape_symbols.push_back(
                    v.fortran_name + "_d" + std::to_string(dim));

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
        if (isAllocatable) {
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

        vars.push_back(std::move(v));
    }
    return vars;
}

}  // namespace hlfir_bridge
