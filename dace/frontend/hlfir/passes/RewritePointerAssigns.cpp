// ============================================================================
// RewritePointerAssigns.cpp  --  collapse Fortran pointer rebinding under the
// strict-no-aliasing assumption.
// ============================================================================
//
// The bridge assumes distinct names never alias the same storage at runtime
// (a relaxation Fortran's strict semantics do NOT grant  --  the language
// allows ``POINTER`` rebinds to overlap with ``TARGET`` declarations and
// even with each other).  Under this relaxation, ``tmp => target`` is a
// pure rename: every read or write of ``tmp`` after the rebind is an
// access to ``target``'s storage.  We materialise that rename by
// rewriting all uses of the pointer declare to the target declare.
//
// Uniform "always rebase to parent" strategy (TARGET design)
// ===========================================================
// Every pointer rebind has the same logical shape:
//
//     ptr => <parent>(<chain-indices>)
//
// where ``<parent>`` is a single ``hlfir.declare`` (the original
// caller-side or local-side TARGET storage) and ``<chain-indices>`` is
// a possibly-empty list of designate steps (whole-array, section
// triplets, scalar element selection) flang lowered between the
// rebind value and the parent declare.
//
// The rewrite is the same for every shape: replace each access through
// the pointer with a direct designate over the PARENT, merging the
// chain's indices with the access (the Fortran source author's
// ``p(i, j)``):
//
//   ``ptr => x``               =>  ``p(i, j)``     ->  ``x(i, j)``
//   ``ptr => x(2:5)``          =>  ``p(i)``        ->  ``x(i + 1)``
//   ``ptr => x(:, j)``         =>  ``p(i)``        ->  ``x(i, j)``
//   ``ptr => arr(i)`` (scalar) =>  ``p`` (scalar)  ->  ``arr(i)``
//   ``ptr => s%a`` (after flatten) =>  ``p(i)``    ->  ``s_a(i)``
//   ``ptr => s%a(2:5)`` (after flatten) => ``p(i)`` ->  ``s_a(i + 1)``
//
// All of these reduce to the same operation:
//   1. Walk the rebind value through ``fir.embox`` / ``fir.rebox`` /
//      ``fir.convert`` and ``hlfir.designate`` ops to find the parent
//      declare and capture each designate's (indices, isTriplet)
//      record into a CHAIN.
//   2. For every downstream access through ``fir.load %ptrDecl#0``
//      (designate user OR box_addr user) build a fresh designate
//      over the parent with merged indices: triplet positions
//      consume one access index (rebased by ``lo - 1``); scalar
//      positions take the chain's literal value verbatim; whole-
//      array (no chain entry) passes the access indices through
//      untouched.
//   3. Erase the load, the rebind store, the alloca / init chain,
//      and the pointer declare.
//
// This collapses the per-variant special cases (plain target,
// slice target, element rebind) into one rewrite step.  The
// box_addr legacy fast path stays only for the empty-chain whole-
// scalar case where the user expects a raw ``!fir.ptr<T>`` (before
// any designate).  Inlined-callee pointer aliases get the same
// per-load designate-rewrite applied independently.
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
// ============================================================================
// I-level design  --  uniform "rebase to parent" rewrite
// ============================================================================
//
// Every Fortran pointer rebind has the same logical shape:
//
//     ptr => <parent>(<chain>)
//
// where ``<parent>`` is a single ``hlfir.declare`` (the original
// caller-side or local-side TARGET storage) and ``<chain>`` is a
// possibly-empty list of ``hlfir.designate`` steps (whole-array,
// section triplets, scalar element selection) flang lowered between
// the rebind value and the parent.  After this pass, every access
// through ``ptr`` lands on a direct ``hlfir.designate`` of the
// parent with indices merged from the chain and the access:
//
//   ``ptr => x``               =>  ``p(i, j)``     ->  ``x(i, j)``
//   ``ptr => x(2:5)``          =>  ``p(i)``        ->  ``x(i + 1)``
//   ``ptr => x(:, j)``         =>  ``p(i)``        ->  ``x(i, j)``
//   ``ptr => arr(i)`` (scalar) =>  ``p`` (scalar)  ->  ``arr(i)``
//   ``ptr => s%a`` (after flatten) =>  ``p(i)``    ->  ``s_a(i)``
//   ``ptr => s%a(2:5)`` (after flatten) => ``p(i)`` ->  ``s_a(i + 1)``
//
// All variants reduce to the same three-step rewrite.  Throughout
// this pass, ``access_indices`` means "the indices supplied by the
// Fortran source author's access through the pointer"  --  e.g.
// ``p(i, j)`` produces ``access_indices = [i, j]``.  Spelt out
// "access" rather than "user" because MLIR already has ``user`` as
// the SSA-downstream consumer of a value (``Op->getUsers()``); the
// two are unrelated and the bare term ``user`` would be
// ambiguous in a pass-level comment.
//
//   1. Walk the rebind value through ``fir.embox`` / ``fir.rebox`` /
//      ``fir.convert`` and ``hlfir.designate`` ops to find the
//      parent declare and capture each designate's
//      (indices, isTriplet) record into a CHAIN.
//   2. For every downstream access through ``fir.load %ptrDecl#0``
//       --  ``hlfir.designate`` users (array pointers) AND
//      ``fir.box_addr`` users (scalar pointers)  --  build a fresh
//      designate over the parent with merged indices.  Triplet
//      positions in the chain consume one access index (rebased
//      by ``lo - 1``); scalar positions take the chain's literal
//      value verbatim; whole-array (no chain entry) passes the
//      access indices through untouched.
//   3. Erase the load + chain (all dead after the rewrite),
//      the rebind store, the alloca / init chain, and the
//      pointer declare.
//
// Helper interface (defined below):
//
//   struct RebindChain {
//       hlfir::DeclareOp         parent;   // root TARGET declare
//       SmallVector<hlfir::DesignateOp, 2> chain;  // walks-back order:
//                                                  // outermost designate first
//   };
//
//   /// Trace a rebind value through embox/rebox/convert/designate
//   /// chains to the parent declare; returns ``parent == nullptr``
//   /// if the chain doesn't end at a declare.
//   static RebindChain traceRebindChain(mlir::Value rebindValue);
//
//   /// ``access_indices``  --  the indices supplied by the Fortran
//   /// source author's access through the pointer (e.g.
//   /// ``p(i, j)`` -> access_indices = [i, j]).  "Access"
//   /// disambiguates from MLIR's SSA ``user`` (Op->getUsers()).
//   /// Compose them with the chain's per-step indices into a
//   /// flat index list over the parent's storage.  The result
//   /// list has one entry per parent dim, ready to drop into a
//   /// new ``hlfir.designate %parent (...)`` op.  Emits any rebase
//   /// arithmetic (``access_idx + lo - 1``) at the supplied
//   /// builder / loc.  Returns false if the merge can't be
//   /// expressed (rare: section-of-section with overlapping
//   /// triplet / access-index counts that don't reconcile).
//   static bool mergeIndices(const RebindChain &c,
//                            mlir::ValueRange access_indices,
//                            mlir::OpBuilder &b, mlir::Location loc,
//                            SmallVectorImpl<mlir::Value> &out);
//
// Bail-loud guards (preflight, run BEFORE the rewrite):
//
//   * INTERLEAVED REBIND/READ  --  ``ptr => A; use; ptr => B; use``.
//     A read between two distinct rebinds observes the EARLIER
//     target; collapsing to one would lose that semantics.
//     Sequential dead-store rebinds (no reads between) are fine  --
//     the last rebind is the only observable one.
//   * BOUNDS REMAP  --  ``ptr(0:n-1) => src(1:n)``.  The user pointer's
//     lower bound differs from the section box's natural ``lo=1``.
//     Flang emits a ``fir.shift`` / ``fir.shape_shift`` operand on
//     the rebox to record the remap; forwarding silently would
//     shift every access by ``remap_lo - 1``.
//   * REBOX SLICE OPERAND  --  defensive reject (flang doesn't
//     typically emit this for pointer rebinds; would mean an
//     additional stride/section overlay we don't model).
//
// Each guard is independent of the rewrite: preflight scans the
// rebind value's chain BEFORE invoking ``traceRebindChain``.
//
// FIR/HLFIR box & shape primer (essential context for the chain
// walker)
// =============================================================
// Pointer rebinds operate on box-typed values.  The exact wrapper
// shape and shape-encoding choice flang makes determines whether
// the bridge can collapse the rebind safely.
//
//  Wrapper types (outer -> inner):
//   * ``fir.ref<T>``          --  plain pointer-to-T, no metadata.
//   * ``fir.box<T>``          --  descriptor: data pointer + shape /
//                              stride / type info.
//   * ``fir.ptr<T>``          --  Fortran POINTER indirection.
//   * ``fir.heap<T>``         --  Fortran ALLOCATABLE indirection.
//
//  Shape ops on the box/declare:
//   * ``fir.shape``        --  extents only; bounds default to 1.
//   * ``fir.shift``        --  lower bounds only (REMAP marker).
//   * ``fir.shape_shift``  --  both extents and bounds (also REMAP).
//
//  Rebind value forms collapsed into the unified path:
//   * ``embox(declare)``                           --  chain = []
//   * ``embox(designate(declare, indices))``       --  chain = [dg]
//   * ``rebox(embox(... designate ...))``          --  chain = [dg]
//                                                   (rebox is a
//                                                   metadata retag,
//                                                   bounds-preserving
//                                                   if its shape is a
//                                                   plain fir.shape)
//   * ``embox(designate(d, scalar_idx))``          --  chain = [dg]
//                                                   (element rebind:
//                                                   access_indices
//                                                   empty, chain
//                                                   provides all
//                                                   indices)
//   * ``embox(zero_bits)``                         --  initial nullify;
//                                                   skipped (no
//                                                   rebind store).
//
// Survivor declares (NOT loud failures)
// =====================================
// A pointer declare that survives this pass with live uses is
// passed through to ``extract_vars``, which gates pointer-attr
// peeling on use-emptiness  --  so a pointer declare with no live
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
// caller's ``ptrDecl#0``.  The unified rewrite walks each alias
// declare's loads in lockstep with the parent's, applying the
// same merge to every downstream user.
// ============================================================================

#include "passes/Passes.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace hlfir_bridge {

namespace {

/// One step of an ``hlfir.designate`` chain captured during a
/// rebind-value trace.  Records the indices and per-dim triplet
/// flags exactly as flang lowered them, so ``mergeIndices`` can
/// recombine them with the access (the Fortran source author's
/// ``p(i, j)``) without re-walking the IR.
struct ChainStep {
    /// The original designate op  --  kept for source-loc / verifier
    /// hints; not strictly required for the merge.
    hlfir::DesignateOp dg;
    /// Indices in HLFIR designate operand order: each triplet dim
    /// contributes 3 entries (lo, hi, step); each scalar dim
    /// contributes 1.  ``triplets[d]`` says which case ``d`` is.
    llvm::SmallVector<mlir::Value, 6> indices;
    llvm::SmallVector<bool, 4> triplets;
};

/// Output of ``traceRebindChain``: the parent declare and the chain
/// of designate steps that produced the rebind value.  ``parent``
/// is null when the trace doesn't terminate at an ``hlfir.declare``
/// (rare; the rewriter bails for those cases).
///
/// Chain order is walks-back: ``chain[0]`` is the OUTERMOST
/// designate (closest to the rebind value), ``chain.back()`` is
/// the INNERMOST (closest to the parent).  Since hlfir.designate
/// composes inside-out (``designate(designate(parent, A), B)``
/// applies B to the result of A), the OUTERMOST step is the one
/// the access (the Fortran source author's ``p(i, j)``) binds
/// against  --  its triplet positions consume access indices first.
/// ``mergeIndices`` walks the chain in outer-first order to apply
/// access indices at the right level.
struct RebindChain {
    hlfir::DeclareOp parent;
    llvm::SmallVector<ChainStep, 2> chain;
};

/// Trace a rebind value back through ``fir.embox``/``fir.rebox``/
/// ``fir.convert``/``hlfir.designate`` ops to the originating
/// ``hlfir.declare``.  Each designate encountered is captured into
/// the chain so ``mergeIndices`` can compose them with the
/// access indices (see the term explanation on
/// ``mergeIndices``).  Returns ``parent == nullptr`` when the
/// chain hits something the rewriter doesn't model (e.g. an
/// ``hlfir.declare`` with no parent storage, or an unsupported
/// op shape).
static RebindChain traceRebindChain(mlir::Value v) {
    RebindChain out;
    for (int i = 0; i < 16 && v; ++i) {
        auto *def = v.getDefiningOp();
        if (!def) return out;
        if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def))   { v = rb.getBox(); continue; }
        if (auto eb = mlir::dyn_cast<fir::EmboxOp>(def))   { v = eb.getMemref(); continue; }
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) { v = cv.getValue(); continue; }
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(def)) {
            ChainStep step;
            step.dg = dg;
            step.indices.assign(dg.getIndices().begin(), dg.getIndices().end());
            for (bool t : dg.getIsTriplet()) step.triplets.push_back(t);
            out.chain.push_back(std::move(step));
            v = dg.getMemref();
            continue;
        }
        if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
            out.parent = dc;
            return out;
        }
        // Unsupported op in the chain  --  return with parent=null so
        // the caller bails.
        return out;
    }
    return out;
}

/// Compose ``access_indices`` with the chain's per-step indices
/// into a flat index list over the parent's storage, suitable for
/// ``hlfir.designate %parent (...result...)``.
///
/// ``access_indices``  --  the indices supplied by the Fortran source
/// author's access through the pointer (e.g. ``p(i, j)`` ->
/// ``access_indices = [i, j]``).  Spelt "access" rather than "user"
/// because MLIR already uses ``user`` for the SSA-downstream
/// consumer of a value (``Op->getUsers()``); the two are unrelated.
///
/// Algorithm: walk the chain INNER-first (the chain itself is
/// stored in walks-back / outer-first order, so we iterate
/// ``rbegin..rend``).  At each step:
///   * triplet positions consume one access index (with rebase);
///   * scalar  positions take the chain's literal value verbatim.
/// The output of one step becomes the access-index list for the
/// next-outer step.  After the outermost step's merge, the
/// resulting list has one entry per parent dim and is ready to
/// drop into ``hlfir.designate %parent (...)``.
///
/// In practice the bridge rarely sees chains of length > 1 because
/// ``hlfir-flatten-structs`` has already collapsed component
/// chains; the recursion is for completeness (section-of-section).
///
/// Empty chain -> ``access_indices`` pass through unchanged (the
/// whole-rebind case ``ptr => x``; ``p(i, j)`` |-> ``x(i, j)``).
///
/// Returns ``true`` when the merge is well-defined.  ``false`` when
/// triplet / access-index counts don't reconcile (rare; the
/// rewriter leaves the access alone in that case).
///
/// Index rebase: a triplet ``(lo, hi, step)`` with ``lo == 1``
/// passes the access index through unchanged.  Otherwise emits
/// ``access_idx + (lo - 1)`` as plain ``arith.addi`` over
/// ``index``.  The ``step`` and ``hi`` are not used in element
/// rebasing (they only shape extents, which DaCe gets from the
/// parent declare's own shape).
static bool mergeIndices(const RebindChain &c,
                         mlir::ValueRange access_indices,
                         mlir::OpBuilder &b, mlir::Location loc,
                         llvm::SmallVectorImpl<mlir::Value> &out) {
    if (c.chain.empty()) {
        for (auto v : access_indices) out.push_back(v);
        return true;
    }
    auto idxTy = b.getIndexType();
    auto toIndex = [&](mlir::Value v) {
        if (v.getType() == idxTy) return v;
        return b.create<fir::ConvertOp>(loc, idxTy, v).getResult();
    };
    auto rebase = [&](mlir::Value access_idx, mlir::Value lo) -> mlir::Value {
        // Constant-fold ``lo == 1`` to keep the IR clean.
        if (auto loCst = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(
                lo.getDefiningOp())) {
            if (auto a = mlir::dyn_cast<mlir::IntegerAttr>(loCst.getValue())) {
                if (a.getInt() == 1) return toIndex(access_idx);
            }
        }
        mlir::Value aIdx  = toIndex(access_idx);
        mlir::Value loIdx = toIndex(lo);
        auto c1 = b.create<mlir::arith::ConstantOp>(loc, idxTy, b.getIndexAttr(1));
        auto adj = b.create<mlir::arith::SubIOp>(loc, loIdx, c1.getResult());
        return b.create<mlir::arith::AddIOp>(loc, aIdx, adj.getResult())
            .getResult();
    };

    // Apply each chain step in INNER-first order.  Walk
    // ``chain.back()`` (innermost) first; result becomes the input
    // for the next-outer step (towards chain[0]).  Final result is
    // the index list against the parent's storage.
    llvm::SmallVector<mlir::Value, 6> cur(access_indices.begin(),
                                          access_indices.end());
    for (auto it = c.chain.rbegin(); it != c.chain.rend(); ++it) {
        const ChainStep &s = *it;
        llvm::SmallVector<mlir::Value, 6> next;
        unsigned cursor = 0;          // walks s.indices
        unsigned accessCursor = 0;    // walks cur (the access-index list)
        for (unsigned d = 0; d < s.triplets.size(); ++d) {
            if (s.triplets[d]) {
                if (cursor + 2 >= s.indices.size() ||
                    accessCursor >= cur.size()) {
                    return false;
                }
                next.push_back(rebase(cur[accessCursor], s.indices[cursor]));
                cursor += 3;          // lo, hi, step
                ++accessCursor;
            } else {
                if (cursor >= s.indices.size()) return false;
                next.push_back(toIndex(s.indices[cursor]));
                cursor += 1;
            }
        }
        if (accessCursor != cur.size()) return false;
        cur = std::move(next);
    }
    out.append(cur.begin(), cur.end());
    return true;
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
        // the rewrites is dead  --  erase to keep extract_vars clean.
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
        //     ptr => B``)  --  would silently bind every read to the
        //     FIRST rebind's target.  Same for conditional rebinds
        //     across branches.
        //   * Element-form designate target (``ptr => arr(i)``,
        //     scalar pointer rebound to one element)  --  different IR
        //     shape than the supported slice rebind.
        //   * Bounds remap (``ptr(0:n-1) => src(1:n)``)  --  flang adds
        //     a ``fir.shift`` operand on the rebox to record the
        //     remapped lower bound.  Forwarding the rebind value
        //     as-is would silently produce off-by-(remap_lo-1)
        //     indices on every read.
        // Collect non-nullify rebind stores in IR order.  Multiple
        // sequential stores before any read are fine  --  only the LAST
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
        // observes the EARLIER target  --  collapsing to one would lose
        // that semantics.  Bail loudly.
        if (nonNullifyStores.size() > 1) {
            for (auto ld : loads) {
                for (size_t k = 1; k < nonNullifyStores.size(); ++k) {
                    if (nonNullifyStores[k - 1]->isBeforeInBlock(ld) &&
                        ld->isBeforeInBlock(nonNullifyStores[k])) {
                        ld.emitError(
                            "hlfir-rewrite-pointer-assigns: pointer "
                            "``" + ptrDecl.getUniqName().str() + "`` "
                            "is read between two rebind sites  --  "
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

        // Pick the LAST non-nullify store as the effective rebind.
        // Earlier stores are dead-store rebinds (no observable reads
        // between them); the alloca-store cleanup at the end of this
        // function will erase them.
        if (!nonNullifyStores.empty())
            rebindStore = nonNullifyStores.back();
        if (!rebindStore) return;

        // Preflight bail-loud guards on the rebind value's chain.
        // Each guard is independent of the chain trace below so
        // unsupported shapes surface a clean error rather than a
        // miscompile.
        //   * BOUNDS REMAP  --  ``fir.rebox`` with ``fir.shift`` /
        //     ``fir.shape_shift`` operand encodes a remapped lower
        //     bound (``ptr(0:..) => src``).  Forwarding silently
        //     shifts every access by ``remap_lo - 1``.
        //   * REBOX SLICE OPERAND  --  defensive; flang doesn't emit
        //     this for pointer rebinds today, but it would mean an
        //     extra stride/section overlay we don't model.
        for (mlir::Value v = rebindStore.getValue(); v;) {
            auto *def = v.getDefiningOp();
            if (!def) break;
            if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def)) {
                if (mlir::Value shape = rb.getShape()) {
                    auto *shapeDef = shape.getDefiningOp();
                    if (mlir::isa_and_nonnull<fir::ShiftOp>(shapeDef) ||
                        mlir::isa_and_nonnull<fir::ShapeShiftOp>(shapeDef)) {
                        rebindStore.emitError(
                            "hlfir-rewrite-pointer-assigns: pointer "
                            "rebind with bounds remap (``ptr(<lo>:..) "
                            "=> src(..)``) not supported  --  flang "
                            "encodes the remapped lower bound on the "
                            "rebox's shift operand and forwarding the "
                            "rebound box would silently shift every "
                            "read by ``remap_lo - 1``.");
                        signalPassFailure();
                        return;
                    }
                }
                if (rb.getSlice()) {
                    rebindStore.emitError(
                        "hlfir-rewrite-pointer-assigns: pointer "
                        "rebind with rebox slice operand not "
                        "supported.");
                    signalPassFailure();
                    return;
                }
                v = rb.getBox(); continue;
            }
            if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) { v = cv.getValue(); continue; }
            break;
        }

        // Trace the rebind value to (parent, chain).  Bail out if the
        // chain doesn't terminate at an ``hlfir.declare``  --  leave the
        // pointer declare alive; downstream extract_vars treats it as
        // a scalar passthrough.
        RebindChain chain = traceRebindChain(rebindStore.getValue());
        if (!chain.parent) return;

        // Inlined-callee alias collapse: any other pointer declare in
        // the function whose memref traces back to ``ptrDecl`` (via
        // ``hlfir.declare`` chain) is an alias of the same storage  --
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

        ptrDecl.emitWarning()
            << "hlfir-rewrite-pointer-assigns: collapsing pointer "
            << "rebind ``" << ptrDecl.getUniqName().str()
            << " => " << chain.parent.getUniqName().str()
            << "(...chain...)`` under the strict-no-aliasing "
            << "assumption.  Every access through the pointer is "
            << "rewritten to a direct designate of the parent's "
            << "storage; if your program relies on alias semantics "
            << "this rewrite is unsafe.";

        // Unified rewrite: for every ``fir.load %ptrDecl#0`` (and
        // every load through an aliased pointer declare  --  the
        // inlined-callee shape), walk its users and rewrite each:
        //
        //   * ``hlfir.designate %loaded (access_indices)``
        //     -> ``hlfir.designate %parent (mergeIndices(chain,
        //                                               access_indices))``
        //   * ``fir.box_addr %loaded``
        //     -> ``%parent.getResult(0)`` if chain is empty (whole
        //                                rebind), else a direct
        //                                designate over parent
        //                                using the chain's indices
        //                                with no access-index
        //                                contribution (element
        //                                rebind / scalar view).
        //                                Any type mismatch with
        //                                the box_addr's result
        //                                is bridged with a
        //                                ``fir.convert``.
        //
        // SSA dominance: load sites use the loaded box AFTER the
        // store, so substituting the store's input value is
        // dominance-correct for any load that comes after the
        // store.  Loads BEFORE the rebind would be reads of an
        // unbound pointer  --  undefined behaviour we don't model.
        llvm::SmallVector<mlir::Operation *, 8> deadReaders;

        // Helper closure: rewrites all users of one load.
        auto rewriteLoadUsers = [&](fir::LoadOp ld) {
            // Skip loads that happen BEFORE the rebind in the same
            // block (reads of an unbound pointer).  Loads in nested
            // blocks (typical: inside a do_loop body) report as
            // "different blocks"  --  treat them as "after" since they
            // can only execute after the enclosing block reaches
            // the loop.
            if (ld->getBlock() == rebindStore->getBlock() &&
                ld->isBeforeInBlock(rebindStore))
                return;
            // Snapshot users  --  we rewrite in place and the user
            // list mutates as we go.
            llvm::SmallVector<mlir::Operation *, 4> userOps;
            for (auto *uu : ld.getResult().getUsers()) userOps.push_back(uu);
            for (auto *uu : userOps) {
                if (auto userDg = mlir::dyn_cast<hlfir::DesignateOp>(uu)) {
                    mlir::OpBuilder b(userDg);
                    auto loc = userDg.getLoc();
                    llvm::SmallVector<mlir::Value, 6> merged;
                    if (!mergeIndices(chain, userDg.getIndices(), b, loc, merged))
                        continue;  // leave userDg alive; bail-loud
                                   // path or downstream surfaces
                                   // the unsupported shape
                    auto newDg = b.create<hlfir::DesignateOp>(
                        loc,
                        /*result_type=*/userDg.getResult().getType(),
                        /*memref=*/chain.parent.getResult(0),
                        /*indices=*/mlir::ValueRange{merged});
                    userDg.getResult().replaceAllUsesWith(newDg.getResult());
                    deadReaders.push_back(userDg);
                    continue;
                }
                if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(uu)) {
                    mlir::OpBuilder b(ba);
                    auto loc = ba.getLoc();
                    mlir::Value replacement;
                    if (chain.chain.empty()) {
                        // Whole rebind  --  box_addr resolves directly
                        // to the parent's ref.
                        replacement = chain.parent.getResult(0);
                    } else {
                        // Chained rebind  --  build a designate over
                        // the parent using the chain's own indices
                        // with no access-index contribution.  This
                        // is the element-rebind shape
                        // (``ptr => arr(i)``) and the rare
                        // scalar-view-of-section.
                        llvm::SmallVector<mlir::Value, 6> merged;
                        if (!mergeIndices(chain, /*access_indices=*/{},
                                          b, loc, merged))
                            continue;
                        // Result type follows the original box_addr's
                        // result; designate yields a ref into the
                        // parent's storage at the chain's position.
                        replacement = b.create<hlfir::DesignateOp>(
                            loc, ba.getResult().getType(),
                            chain.parent.getResult(0),
                            mlir::ValueRange{merged}).getResult();
                    }
                    if (replacement.getType() != ba.getResult().getType()) {
                        replacement = b.create<fir::ConvertOp>(
                            loc, ba.getResult().getType(), replacement);
                    }
                    ba.getResult().replaceAllUsesWith(replacement);
                    deadReaders.push_back(ba);
                    continue;
                }
                // Other user shapes (rare)  --  leave alone.  The
                // surviving load + ptr declare keep them
                // resolvable downstream as a scalar passthrough.
            }
            // Always queue the load  --  use_empty is checked at
            // sweep time.  Without this, the load is checked here
            // while user ops still reference it and is never
            // pushed for erase, leaving the pointer declare with a
            // live user.
            deadReaders.push_back(ld);
        };

        // Snapshot loads of the primary pointer declare.
        llvm::SmallVector<fir::LoadOp, 4> snapshotLoads;
        for (auto *u : ptrDecl.getResult(0).getUsers())
            if (auto ld = mlir::dyn_cast<fir::LoadOp>(u))
                snapshotLoads.push_back(ld);
        for (auto ld : snapshotLoads) rewriteLoadUsers(ld);

        // Same walk for each aliased pointer declare  --  every load
        // returns the same box value we just rewrote, so rewriting
        // its users via the same chain lands them on the rebound
        // parent too.
        llvm::SmallVector<hlfir::DeclareOp, 4> aliasesToErase;
        for (auto alias : aliasDecls) {
            llvm::SmallVector<fir::LoadOp, 4> aliasLoads;
            for (auto *u : alias.getResult(0).getUsers())
                if (auto ld = mlir::dyn_cast<fir::LoadOp>(u))
                    aliasLoads.push_back(ld);
            for (auto ld : aliasLoads) rewriteLoadUsers(ld);
            aliasesToErase.push_back(alias);
        }

        // Erase user ops first (they hold the only uses on each
        // load), then the loads themselves, in iteration order.
        // ``op->use_empty()`` at the moment of erase decides whether
        // each is safe to drop.
        for (auto *op : deadReaders)
            if (op->use_empty()) op->erase();

        // Erase use-empty alias declares.
        for (auto alias : aliasesToErase) {
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
