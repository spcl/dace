// ============================================================================
// RewritePointerAssigns.cpp — collapse Fortran pointer rebinding under the
// strict-no-aliasing assumption.
// ============================================================================
//
// The bridge assumes distinct names never alias the same storage at runtime
// (a relaxation Fortran's strict semantics do NOT grant — the language
// allows ``POINTER`` rebinds to overlap with ``TARGET`` declarations and
// even with each other).  Under this relaxation, ``tmp => target`` is a
// pure rename: every read or write of ``tmp`` after the rebind is an
// access to ``target``'s storage.  We materialise that rename by
// rewriting all uses of the pointer declare to the target declare.
//
// Why now (before flatten-structs):
//   * Pointer declares carry ``fir.box<fir.ptr<...>>`` types.  Letting
//     them survive into flatten-structs would either inflate the
//     all-or-nothing flatten gate (every pointer member becomes a
//     non-flat member) or require treating a pointer slot as another
//     allocatable-style runtime-shape variable.  Collapsing the alias
//     here keeps flatten-structs's input clean: just declares + scalar
//     stores / loads, no ``fir.box`` indirection on what is effectively
//     a renamed reference.
//   * Downstream allocatable / pointer struct-member lowering only
//     needs to deal with TRUE runtime-shape members (POINTER /
//     ALLOCATABLE arrays as struct fields), not with name-aliasing
//     pointer locals.  Splitting these two concerns simplifies both
//     passes.
//
// Pattern matched:
//   %ptrAlloca  = fir.alloca !fir.box<!fir.ptr<T>>
//   %nullPtr    = fir.zero_bits !fir.ptr<T>
//   %nullBox    = fir.embox %nullPtr
//   fir.store %nullBox to %ptrAlloca                  -- initial nullify
//   %ptrDecl:2  = hlfir.declare %ptrAlloca {pointer, uniq_name="...tmp"}
//   %targetBox  = fir.embox %target_decl#0  (or fir.embox of a designate
//                                            of a TARGET'd struct field)
//   fir.store %targetBox to %ptrDecl#0                -- the => rebind
//
// Reads then look like:
//   %boxLoad   = fir.load %ptrDecl#0 : !fir.ref<box<ptr<T>>>
//   %addr      = fir.box_addr %boxLoad : (box<ptr<T>>) -> !fir.ptr<T>
//   %v         = fir.load %addr : !fir.ptr<T>
//
// Rewrite under no-alias:
//   Replace every ``fir.box_addr`` whose memref chain traces back to
//   ``ptrDecl`` with the original ``target_decl#0`` (a ``fir.ref<T>``).
//   The intermediate ``fir.load %ptrDecl#0`` and ``fir.box_addr`` ops
//   become dead and are erased.  ``ptrDecl`` itself, the alloca, the
//   zero_bits + initial embox + initial store, and the rebind store
//   all become dead and are erased.
//
// Warning emitted on every firing:
//   ``hlfir-rewrite-pointer-assigns: collapsing pointer rebind <ptr> =>
//     <target> under the strict-no-aliasing assumption.  Fortran allows
//     aliased pointer access; if your program relies on alias semantics
//     this rewrite is unsafe.``
//
// FIR/HLFIR box & shape primer (essential for understanding the
// rebind variants below)
// =============================================================
// Pointer rebinds operate on box-typed values.  The exact wrapper
// shape and shape-encoding choice flang makes determines whether
// the bridge can collapse the rebind safely.
//
//  Wrapper types (outer → inner):
//   * ``fir.ref<T>``         — plain pointer-to-T, no metadata.
//   * ``fir.box<T>``         — descriptor: data pointer + shape /
//                              stride / type info.  ``T`` is the
//                              element type or ``fir.ptr/heap`` of
//                              an array type.
//   * ``fir.ptr<T>``         — Fortran POINTER indirection.  Box
//                              over ``ptr<...>`` carries the
//                              POINTER attribute.
//   * ``fir.heap<T>``        — Fortran ALLOCATABLE indirection.
//                              Box over ``heap<...>`` is what
//                              ``allocate`` allocates into.
//
//  Shape ops on the box/declare:
//   * ``fir.shape  %e0, %e1, …``           — extents only.  Lower
//                                             bounds default to 1.
//                                             Bounds-preserving
//                                             rebox.
//   * ``fir.shift  %lo0, %lo1, …``         — lower bounds only.
//                                             Used for bounds REMAP
//                                             on rebox (the user
//                                             wrote ``ptr(0:..) =>
//                                             src``).
//   * ``fir.shape_shift %e0,%lo0, %e1,%lo1,…``
//                                          — both extents + lower
//                                             bounds.  Also
//                                             remap-carrying.
//
//  Rebind value forms (right-hand side of ``fir.store ... to %ptrDecl#0``):
//   * ``embox(declare)``            — plain target.  ``ptr => x``
//                                      where ``x`` is a whole
//                                      variable.
//   * ``embox(designate(d, slice))`` — section target.  Flang may
//                                      still wrap with a
//                                      ``fir.rebox`` to retype
//                                      ``box<array<...>>`` →
//                                      ``box<ptr<array<...>>>``.
//   * ``rebox(embox(... designate(d, slice) ...))``
//                                   — section target through a
//                                      type retag.  This is the
//                                      ``ptr => arr(1:n)`` shape
//                                      this pass forwards.
//   * ``embox(designate(d, scalar_idx))``
//                                   — element target.  Bail
//                                      (loud-failure) — different
//                                      semantics from a section.
//   * ``embox(zero_bits)``          — initial nullify; skipped.
//
// Unsupported cases (loud-failure contract)
// =========================================
// The pass emits ``signalPassFailure()`` + a clear error rather
// than silently producing wrong code in the following cases.
// Every case below has at least one unit test in
// ``baseline_pointer_assign_test.py`` that pins the contract.
//
//  1. INTERLEAVED REBIND/READ:
//      ``ptr => A; use ptr; ptr => B; use ptr``.
//      The first ``use ptr`` observes ``A``, the second ``B``.
//      Single-substitution would lose that semantics.
//      DETECTION: walk all ``fir.load %ptrDecl#0`` ops and check
//      if any sits between two non-nullify rebind stores.
//      DEAD-STORE rebinds (``ptr => A; ptr => B; use ptr``, no
//      reads between) are NOT this case — only the LAST rebind
//      is observable, the earlier one is a dead store and we
//      take the LAST one as the effective rebind.
//
//  2. ELEMENT-FORM REBIND TARGET:
//      ``ptr => arr(i)`` for a SCALAR pointer rebound to one
//      array element.  IR shape: ``embox(designate(d, scalar_idx))``
//      where the designate has ``isTriplet=[false, …]``.
//      Different semantics from a section: a SECTION (``ptr =>
//      arr(i:i)``) creates a rank-1 view of length 1; an ELEMENT
//      (``ptr => arr(i)``) creates a SCALAR alias.  Forwarding
//      the section box would shift the type but downstream
//      designate(load(ptr))(j) wouldn't make sense.
//      DETECTION: the source designate has indices and no
//      triplet flags.
//
//  3. BOUNDS REMAP:
//      ``ptr(0:n-1) => src(1:n)``.  The user pointer's lower
//      bound differs from the section box's natural ``lo=1``.
//      Flang encodes the remap by emitting a ``fir.shift`` (or
//      ``fir.shape_shift``) operand on the rebox.  Forwarding
//      the rebound box value as-is would silently shift every
//      read by ``remap_lo - 1`` (off-by-N indexing on every
//      access through the pointer).
//      DETECTION: any ``fir.rebox`` in the rebind value's
//      definition chain has a ``shape`` operand defined by
//      ``fir.shift`` or ``fir.shape_shift``.  A plain
//      ``fir.shape`` (extents only) is bounds-preserving and
//      safe to forward.
//
//  4. REBOX SLICE OPERAND:
//      Flang doesn't typically generate this for pointer
//      rebinds, but defensively reject any ``fir.rebox`` whose
//      ``slice`` operand is non-empty — would mean an additional
//      stride/section overlay we don't model.
//
// Survivor declares (NOT loud failures)
// =====================================
// A pointer declare that survives this pass with live uses is
// passed through to ``extract_vars``, which gates pointer-attr
// peeling on use-emptiness — so a pointer declare with no live
// uses gets erased here, and one with live uses (cross-procedure
// pointer dummy, complex chained target the pass couldn't
// recognise, etc.) stays as a SCALAR passthrough downstream
// rather than a phantom rank>0 array on the SDFG signature.
// This keeps the pipeline smooth for cases the pass doesn't
// recognise without forcing them all to be loud-failures.
//
// Inlined-callee aliases
// ======================
// When ``hlfir-inline-all`` splices a module-contained call's
// body into the caller, the callee's pointer dummy declare
// becomes a fresh ``hlfir.declare %callerDecl#0 dummy_scope %dsc
// {pointer, uniq_name="..."}`` whose ``memref`` operand is the
// caller's ``ptrDecl#0``.  The alias has the same ``pointer``
// attribute and would be processed independently, finding no
// in-scope rebind store of its own and surviving as live.  We
// avoid that by collecting alias declares (any pointer declare
// whose memref traces back to ``ptrDecl`` through
// ``hlfir.declare`` / ``fir.convert`` chains) at rewrite time
// and forwarding their loads in lockstep with the parent's.
// ============================================================================

#include "passes/Passes.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace hlfir_bridge {

namespace {

/// Trace ``v`` back to the ``hlfir.declare`` op it ultimately references,
/// walking through the embox / convert / designate / load chain Flang
/// inserts around a TARGET'd entity.  Returns null on an unrecognised
/// chain.
static hlfir::DeclareOp traceTarget(mlir::Value v) {
    for (int i = 0; i < 16 && v; ++i) {
        auto *def = v.getDefiningOp();
        if (!def) return {};
        if (auto e = mlir::dyn_cast<fir::EmboxOp>(def))     { v = e.getMemref(); continue; }
        if (auto c = mlir::dyn_cast<fir::ConvertOp>(def))   { v = c.getValue(); continue; }
        if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(def)) return d;
        // Designate-of-target (``s%a`` where ``s`` is TARGET): the
        // designate's memref leads back to the struct's declare; we
        // would need component-aware rewriting for that case.  Return
        // null so the pass leaves the rebind alone.  ``flatten-structs``
        // will turn ``s%a`` into a direct flat declare, after which a
        // future iteration of this pass could pick it up.
        if (mlir::isa<hlfir::DesignateOp>(def)) return {};
        return {};
    }
    return {};
}

/// Recognise a SLICE-target rebind: the stored value walks back
/// through ``fir.rebox`` / ``fir.convert`` to a section-form
/// ``hlfir.designate`` of some parent declare.  Returns the
/// designate on match, null otherwise.  The slice variant is what
/// ``w => src(1:n)`` lowers to: a section view box that gets
/// reboxed to ``box<ptr<...>>`` before being stored to the pointer
/// declare's box-ref slot.  Distinct from ``traceTarget`` (which
/// only handles plain ``embox(declare)``) because the slice's bounds
/// must be preserved — collapsing to ``parent_decl#0`` would lose
/// them.
static hlfir::DesignateOp traceSliceTarget(mlir::Value v) {
    for (int i = 0; i < 16 && v; ++i) {
        auto *def = v.getDefiningOp();
        if (!def) return {};
        if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def))    { v = rb.getBox(); continue; }
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def))  { v = cv.getValue(); continue; }
        if (auto e = mlir::dyn_cast<fir::EmboxOp>(def))     { v = e.getMemref(); continue; }
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(def)) {
            // Must be a triplet section (some-dim has triplet=true).
            // Element-form designate on the rebind path would mean
            // something else (taking address of a single element);
            // reject it here.
            for (bool t : dg.getIsTriplet()) if (t) return dg;
            return {};
        }
        return {};
    }
    return {};
}

struct RewritePointerAssignsPass
    : public mlir::PassWrapper<RewritePointerAssignsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RewritePointerAssignsPass)

    llvm::StringRef getArgument() const final {
        return "hlfir-rewrite-pointer-assigns";
    }
    llvm::StringRef getDescription() const final {
        return "Collapse Fortran ``ptr => target`` rebinds under the "
               "strict-no-aliasing assumption: every use of ``ptr`` after "
               "the rebind becomes a use of ``target``.  Pinned rule: the "
               "bridge assumes distinct names never alias.";
    }

    void runOnOperation() override {
        // Collect candidates first; rewriting mutates the IR and would
        // invalidate a fused walk.
        llvm::SmallVector<hlfir::DeclareOp, 8> ptrDecls;
        getOperation().walk([&](hlfir::DeclareOp d) {
            auto attrs = d.getFortranAttrs();
            if (!attrs) return;
            if (bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::pointer))
                ptrDecls.push_back(d);
        });

        for (auto ptrDecl : ptrDecls) rewrite(ptrDecl);

        // Sweep: any pointer declare with use_empty results after
        // the rewrites is dead — erase to keep extract_vars clean.
        // Pointer declares that survived with live uses are passed
        // through to extract_vars, which gates pointer-attr peeling
        // on use-emptiness so a never-collapsed pointer stays as a
        // scalar passthrough rather than a phantom rank>0 array.
        // (Cross-procedure pointer dummy rebinds, complex chained
        // targets, and other unsupported rebind shapes all flow
        // through this path; they surface as either a working
        // scalar passthrough or a clean downstream error rather than
        // a hard pass-failure here.)
        getOperation().walk([&](hlfir::DeclareOp d) {
            auto attrs = d.getFortranAttrs();
            if (!attrs) return;
            if (!bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::pointer))
                return;
            if (d.getResult(0).use_empty() && d.getResult(1).use_empty())
                d.erase();
        });
    }

   private:
    void rewrite(hlfir::DeclareOp ptrDecl) {
        // Find the rebind store(s): ``fir.store %targetBox to
        // %ptrDecl#0``.  Three forms:
        //   * Initial nullify (``embox(zero_bits)``): skipped.
        //   * Plain target  (``embox(declare)``):           collapse.
        //   * Slice target  (``rebox(designate(declare))``): forward.
        //
        // Loud-failure cases (we abort the pass with an emitError so
        // the bridge surfaces a clean unsupported message rather
        // than silently producing wrong code):
        //
        //   * Multiple non-nullify rebinds in scope (``ptr => A; ...;
        //     ptr => B``) — would silently bind every read to the
        //     FIRST rebind's target.  Same for conditional rebinds
        //     across branches.
        //   * Element-form designate target (``ptr => arr(i)``,
        //     scalar pointer rebound to one element) — different IR
        //     shape than the supported slice rebind.
        //   * Bounds remap (``ptr(0:n-1) => src(1:n)``) — flang adds
        //     a ``fir.shift`` operand on the rebox to record the
        //     remapped lower bound.  Forwarding the rebind value
        //     as-is would silently produce off-by-(remap_lo-1)
        //     indices on every read.
        // Collect non-nullify rebind stores in IR order.  Multiple
        // sequential stores before any read are fine — only the LAST
        // one is observable, all earlier ones are dead-store
        // rebinds.  Rebinds INTERLEAVED with reads (a read between
        // two stores) bail loudly because the bridge can't pick a
        // single coherent collapse target.
        llvm::SmallVector<fir::StoreOp, 4> nonNullifyStores;
        llvm::SmallVector<fir::LoadOp, 4> loads;
        for (auto *u : ptrDecl.getResult(0).getUsers()) {
            if (auto ld = mlir::dyn_cast<fir::LoadOp>(u))
                loads.push_back(ld);
        }
        fir::StoreOp rebindStore;
        hlfir::DeclareOp targetDecl;     // plain-target path
        mlir::Value sliceRebindValue;    // slice-target path: store.getValue()
        for (auto *u : ptrDecl.getResult(0).getUsers()) {
            auto st = mlir::dyn_cast<fir::StoreOp>(u);
            if (!st) continue;
            auto *valDef = st.getValue().getDefiningOp();

            // Skip the initial nullify on either rebind form.
            if (auto embox = mlir::dyn_cast_or_null<fir::EmboxOp>(valDef))
                if (mlir::isa_and_nonnull<fir::ZeroOp>(
                        embox.getMemref().getDefiningOp()))
                    continue;

            nonNullifyStores.push_back(st);
        }

        // Order the non-nullify stores in IR-walk order so "last"
        // means last observable rebind.
        std::sort(nonNullifyStores.begin(), nonNullifyStores.end(),
                  [](fir::StoreOp a, fir::StoreOp b) {
                      return a->isBeforeInBlock(b);
                  });

        // Interleaved-rebind detection: a read between two rebinds
        // observes the EARLIER target — collapsing to one would lose
        // that semantics.  Bail loudly.
        if (nonNullifyStores.size() > 1) {
            for (auto ld : loads) {
                for (size_t k = 1; k < nonNullifyStores.size(); ++k) {
                    if (nonNullifyStores[k - 1]->isBeforeInBlock(ld) &&
                        ld->isBeforeInBlock(nonNullifyStores[k])) {
                        ld.emitError(
                            "hlfir-rewrite-pointer-assigns: pointer "
                            "``" + ptrDecl.getUniqName().str() + "`` "
                            "is read between two rebind sites — "
                            "collapsing would silently bind every "
                            "read to one target.  Refactor to use "
                            "distinct pointer variables, or guard "
                            "the single rebind site behind a runtime "
                            "selection that the bridge can lower.");
                        signalPassFailure();
                        return;
                    }
                }
            }
        }

        // Walk the LAST non-nullify store as the effective rebind.
        // Earlier stores are dead-store rebinds (no observable
        // reads between them); the existing alloca-store cleanup at
        // the end of this function will erase them.
        for (auto st : llvm::reverse(nonNullifyStores)) {
            if (rebindStore) continue;  // already chosen
            auto *valDef = st.getValue().getDefiningOp();

            // Element-form designate target: ``ptr => arr(i)`` (scalar
            // pointer rebound to a single element).  Detected as
            // ``embox`` of a designate whose isTriplet is all-false.
            if (auto embox = mlir::dyn_cast_or_null<fir::EmboxOp>(valDef)) {
                if (auto dg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
                        embox.getMemref().getDefiningOp())) {
                    bool anyTrip = false;
                    for (bool t : dg.getIsTriplet()) if (t) { anyTrip = true; break; }
                    if (!anyTrip && !dg.getIndices().empty()) {
                        st.emitError(
                            "hlfir-rewrite-pointer-assigns: element-form "
                            "designate target (``ptr => arr(i)`` for a "
                            "scalar pointer) not yet supported.  Refactor to "
                            "rebind a section (``ptr => arr(i:i)``) or use a "
                            "scalar variable.");
                        signalPassFailure();
                        return;
                    }
                }
            }

            // Plain-target path.
            if (auto embox = mlir::dyn_cast_or_null<fir::EmboxOp>(valDef)) {
                if (auto target = traceTarget(embox.getMemref())) {
                    rebindStore = st;
                    targetDecl  = target;
                    continue;
                }
            }
            // Slice-target path.
            if (auto sliceDg = traceSliceTarget(st.getValue())) {
                // Bounds remap detection: a ``fir.rebox`` with a
                // ``fir.shift`` operand re-bases the lower bound.
                // Walk the chain: if any rebox in the rebind value's
                // definition path carries a non-empty ``shift``
                // operand, abort.
                mlir::Value v = st.getValue();
                for (int i = 0; i < 16 && v; ++i) {
                    auto *def = v.getDefiningOp();
                    if (!def) break;
                    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def)) {
                        // Bounds remap: flang encodes the remapped
                        // lower bound on the rebox's ``shape`` operand
                        // when it's a ``fir.shift`` or
                        // ``fir.shape_shift`` (both carry per-dim lo
                        // values).  A plain ``fir.shape`` (extents
                        // only) is bounds-preserving and safe.  A
                        // ``slice`` operand on the rebox would also
                        // be a remap but flang doesn't generate that
                        // form for pointer rebinds.
                        if (mlir::Value shape = rb.getShape()) {
                            auto *shapeDef = shape.getDefiningOp();
                            if (mlir::isa_and_nonnull<fir::ShiftOp>(shapeDef) ||
                                mlir::isa_and_nonnull<fir::ShapeShiftOp>(shapeDef)) {
                                st.emitError(
                                    "hlfir-rewrite-pointer-assigns: pointer "
                                    "rebind with bounds remap (``ptr(<lo>:..) "
                                    "=> src(..)``) not supported — flang "
                                    "encodes the remapped lower bound on the "
                                    "rebox's shift operand and forwarding the "
                                    "rebound box would silently shift every "
                                    "read by ``remap_lo - 1``.");
                                signalPassFailure();
                                return;
                            }
                        }
                        if (rb.getSlice()) {
                            st.emitError(
                                "hlfir-rewrite-pointer-assigns: pointer "
                                "rebind with rebox slice operand not "
                                "supported.");
                            signalPassFailure();
                            return;
                        }
                        v = rb.getBox();
                        continue;
                    }
                    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) { v = cv.getValue(); continue; }
                    break;
                }
                rebindStore = st;
                sliceRebindValue = st.getValue();
                (void)sliceDg;
                continue;
            }
        }
        if (!rebindStore) return;
        if (!targetDecl && !sliceRebindValue) return;
        // (Multi-rebind detection: handled above by the
        // interleaved-rebind check.  Sequential dead-store rebinds
        // are fine — we use the LAST one as the effective rebind
        // and the earlier stores get cleaned up with the alloca.)

        // Inlined-callee alias collapse: any other pointer declare in
        // the function whose memref traces back to ``ptrDecl`` (via
        // ``hlfir.declare`` chain) is an alias of the same storage —
        // typically the inlined dummy of a module-contained call that
        // received our pointer as an argument.  Without redirecting
        // its uses, the alias's loads stay live, extract_vars surfaces
        // it as an independent rank>0 array, and the SDFG ends up
        // demanding extra ``<alias>_d0`` symbols.  Collect them now
        // so the rewrite below redirects their loads in lockstep.
        llvm::SmallVector<hlfir::DeclareOp, 4> aliasDecls;
        if (auto func = ptrDecl->getParentOfType<mlir::func::FuncOp>()) {
            func.walk([&](hlfir::DeclareOp other) {
                if (other == ptrDecl) return;
                auto attrs = other.getFortranAttrs();
                if (!attrs) return;
                if (!bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::pointer))
                    return;
                // Walk other.getMemref() back through hlfir.declare /
                // fir.convert chain and check if it reaches ptrDecl's
                // results.
                mlir::Value mr = other.getMemref();
                for (int i = 0; i < 8 && mr; ++i) {
                    if (mr == ptrDecl.getResult(0) || mr == ptrDecl.getResult(1)) {
                        aliasDecls.push_back(other);
                        return;
                    }
                    auto *d = mr.getDefiningOp();
                    if (!d) return;
                    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) { mr = cv.getValue(); continue; }
                    if (auto inner = mlir::dyn_cast<hlfir::DeclareOp>(d)) { mr = inner.getMemref(); continue; }
                    return;
                }
            });
        }

        if (targetDecl) {
            ptrDecl.emitWarning()
                << "hlfir-rewrite-pointer-assigns: collapsing pointer "
                << "rebind under the strict-no-aliasing assumption "
                << "(target: " << targetDecl.getUniqName().str() << ").  "
                << "Fortran allows aliased pointer access; if your program "
                << "relies on alias semantics this rewrite is unsafe.";
        } else {
            ptrDecl.emitWarning()
                << "hlfir-rewrite-pointer-assigns: forwarding pointer "
                << "rebind ``" << ptrDecl.getUniqName().str()
                << " => <section>`` under the strict-no-aliasing "
                << "assumption.  Section indexing on the pointer is "
                << "rewritten to read directly through the rebound "
                << "box; if your program relies on alias semantics "
                << "this rewrite is unsafe.";
        }

        // Unified rewrite: forward every ``fir.load %ptrDecl#0`` to
        // the rebind value (the box stored at the rebind site).  Both
        // plain-target (``embox(declare)``) and slice-target
        // (``rebox(designate(parent, slice))``) end up with the same
        // shape — a box whose data pointer + shape describe the
        // target's storage — so loads and downstream
        // ``hlfir.designate`` ops can pull straight through to the
        // rebound value, skipping the alloca round-trip.
        //
        // Plain-target legacy fast path: when the consumer of the
        // load is ``fir.box_addr`` (the simple scalar / whole-array
        // case before the bridge generalised to designate-over-box),
        // collapse to the target's raw ref directly — that's how the
        // pre-Phase-5b code worked, and downstream consumers may
        // still depend on the resulting ``ref<T>`` type rather than a
        // ``box<...>``.
        //
        // SSA dominance: the load sites use the loaded box AFTER the
        // store, so substituting the store's input value (which
        // dominates the store) is dominance-correct for any load that
        // comes after the store.  Loads BEFORE the rebind would be
        // reads of an unbound pointer — undefined behaviour we don't
        // handle.
        llvm::SmallVector<mlir::Operation *, 8> deadReaders;
        mlir::Value rebindValue = rebindStore.getValue();
        if (targetDecl) {
            // Try the legacy box_addr fast path first.
            mlir::Value targetRef = targetDecl.getResult(0);
            for (auto *u : ptrDecl.getResult(0).getUsers()) {
                auto ld = mlir::dyn_cast<fir::LoadOp>(u);
                if (!ld) continue;
                bool consumedByBoxAddr = false;
                for (auto *uu : ld.getResult().getUsers()) {
                    auto ba = mlir::dyn_cast<fir::BoxAddrOp>(uu);
                    if (!ba) continue;
                    consumedByBoxAddr = true;
                    mlir::Value replacement = targetRef;
                    if (replacement.getType() != ba.getResult().getType()) {
                        mlir::OpBuilder b(ba);
                        replacement = b.create<fir::ConvertOp>(
                            ba.getLoc(), ba.getResult().getType(), targetRef);
                    }
                    ba.getResult().replaceAllUsesWith(replacement);
                    deadReaders.push_back(ba);
                }
                // If any user was NOT box_addr (e.g. hlfir.designate),
                // forward the load to the rebind value so it picks up
                // a proper box.  Type matches: load returns the same
                // ``box<ptr<...>>`` type the rebind store deposits.
                if (!ld->use_empty() && ld->isBeforeInBlock(rebindStore) == false) {
                    ld.getResult().replaceAllUsesWith(rebindValue);
                }
                if (consumedByBoxAddr || ld->use_empty())
                    deadReaders.push_back(ld);
            }
        } else {
            // Pure slice-target: forward every load directly.
            for (auto *u : ptrDecl.getResult(0).getUsers()) {
                auto ld = mlir::dyn_cast<fir::LoadOp>(u);
                if (!ld) continue;
                if (!ld->isBeforeInBlock(rebindStore)) {
                    ld.getResult().replaceAllUsesWith(rebindValue);
                    deadReaders.push_back(ld);
                }
            }
        }
        for (auto *op : deadReaders)
            if (op->use_empty()) op->erase();

        // Apply the same forwarding to every aliased pointer declare:
        // each load of the alias's box-ref returns the same box value
        // we just collapsed; replacing it lets the alias's
        // ``hlfir.designate`` users land on the rebound parent too.
        for (auto alias : aliasDecls) {
            llvm::SmallVector<mlir::Operation *, 4> aliasDead;
            for (auto *u : alias.getResult(0).getUsers()) {
                auto ld = mlir::dyn_cast<fir::LoadOp>(u);
                if (!ld) continue;
                ld.getResult().replaceAllUsesWith(rebindValue);
                aliasDead.push_back(ld);
            }
            for (auto *op : aliasDead)
                if (op->use_empty()) op->erase();
            // Erase the alias declare if it's now use-free.
            if (alias.getResult(0).use_empty() &&
                alias.getResult(1).use_empty())
                alias.erase();
        }

        // Erase the rebind store + the entire alloca/init chain feeding
        // ptrDecl.  The alloca's only remaining users at this point are
        // the ``hlfir.declare`` itself and the (now dead) initial
        // nullify chain.
        rebindStore.erase();

        mlir::Value ptrAlloca = ptrDecl.getMemref();
        // Sweep dead init ops in reverse so each erase's only users are
        // already gone.  Pattern:
        //   %a = fir.alloca
        //   %z = fir.zero_bits
        //   %e = fir.embox %z
        //   fir.store %e to %a   -- initial nullify
        //   hlfir.declare %a
        llvm::SmallVector<mlir::Operation *, 4> deadInit;
        for (auto *u : ptrAlloca.getUsers()) {
            if (auto st = mlir::dyn_cast<fir::StoreOp>(u))
                deadInit.push_back(st);
        }
        for (auto *op : deadInit) op->erase();

        if (ptrDecl.getResult(0).use_empty() &&
            ptrDecl.getResult(1).use_empty())
            ptrDecl.erase();

        // Sweep the dangling embox + zero_bits + alloca if no users
        // remain.
        if (auto *def = ptrAlloca.getDefiningOp())
            if (def->use_empty()) def->erase();
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> createRewritePointerAssignsPass() {
    return std::make_unique<RewritePointerAssignsPass>();
}

}  // namespace hlfir_bridge
