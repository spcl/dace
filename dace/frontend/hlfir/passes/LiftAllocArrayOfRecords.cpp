// ============================================================================
// LiftAllocArrayOfRecords.cpp — lift alloc-array-of-records struct members
// into top-level companions with a leading runtime-extent dim.
// ============================================================================
//
// Motivation:
//     Fortran pattern (canonical ICON shape, e.g. ``mo_nonhydro_types.f90``):
//
//         type t_inner
//           real(kind=8), pointer, contiguous :: w(:, :, :)
//           real(kind=8), pointer, contiguous :: rho(:, :, :)
//           ! ... more pointer-array members ...
//         end type t_inner
//         type t_outer
//           type(t_inner), allocatable :: items(:)   ! alloc-array of records
//         end type t_outer
//
//         type(t_outer) :: p_nh
//         allocate(p_nh%items(N))
//         do n = 1, N
//           p_nh%items(n)%w => <storage slice>
//           ! ...
//         end do
//         ! User code (potentially with runtime element index ``jg``):
//         val = p_nh%items(jg)%w(jc, jk, jb)
//
//     The bridge's existing FlattenStructs pass bails on
//     ``box<heap|ptr<seq<? x record>>>`` members — there's no static
//     way to enumerate the alloc-array's elements.  This pass closes
//     the gap by rewriting every access chain
//     ``<root>%<member>(<idx>)%<inner_leaf>(<inner_indices>)``
//     into a direct access on a synthesised top-level companion of
//     shape ``(N_member, *inner_leaf_shape)``.
//
//     After the rewrite:
//         val = p_nh_items_w(jg, jc, jk, jb)
//     where ``p_nh_items_w`` is a new top-level declare with the
//     leading dim sized by a synth symbol ``<root>_<member>_size``.
//     The caller (SDFG bindings layer) is responsible for allocating
//     the flat companion and threading the alloc-count symbol.
//
//     The setup code (``allocate(p_nh%items(N))``, per-element pointer
//     rebinds, ``deallocate``) becomes dead — no downstream consumer
//     reads through the original AoS path after the rewrite — and
//     gets cleaned up by symbol-dce / canonicalize in the pipeline.
//
// Uniformity contract:
//     Every alloc-array element is assumed to have the same inner-leaf
//     shape across the alloc-array's extent.  ICON satisfies this; the
//     pass cannot statically prove it in general.  Runtime enforcement
//     is implicit via the bindings layer (caller passes one flat
//     companion of declared shape `(N_member, *inner_leaf_shape)`).
//
// Pipeline position:
//     After ``hlfir-inline-all`` (so the inlined-callee element-alias
//     declares — ``hlfir.declare %elem ...{uniq_name = "<callee>"}``
//     wrapping ``hlfir.designate %items_box (%idx)`` — are visible)
//     and BEFORE ``hlfir-flatten-structs`` (so flatten sees clean
//     top-level arrays after the lift).
//
// Out of scope:
//     - Whole-AoS copies (``a = p%items``): hard-fail in validation.
//     - Non-uniform per-element shape: trust the caller (runtime
//       contract via bindings).  No static check is general.
//     - Reallocate-in-kernel of the alloc-array: hard-fail.
// ============================================================================

#include "passes/Passes.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace hlfir_bridge {

namespace {

// ---------------------------------------------------------------------------
// Type predicates
// ---------------------------------------------------------------------------

/// Recognise an alloc-array or pointer-array struct member whose inner
/// element type is itself a record:
///     box<heap<seq<? x record>>>     ! type(t), allocatable :: f(:)
///     box<ptr<seq<? x record>>>      ! type(t), pointer     :: f(:)
///
/// Returns the inner element RecordType when matched, null otherwise.
static fir::RecordType allocOrPtrArrayOfRecordsMember(mlir::Type t) {
    auto box = mlir::dyn_cast<fir::BoxType>(t);
    if (!box) return {};
    mlir::Type inner;
    if (auto h = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
        inner = h.getEleTy();
    else if (auto p = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
        inner = p.getEleTy();
    else
        return {};
    auto seq = mlir::dyn_cast<fir::SequenceType>(inner);
    if (!seq) return {};
    return mlir::dyn_cast<fir::RecordType>(seq.getEleTy());
}

/// Peel ``fir.ref<...>`` / ``fir.box<...>`` / ``fir.heap<...>`` /
/// ``fir.ptr<...>`` wrappers to find the underlying RecordType.
/// Returns null if no record is reachable.
static fir::RecordType peelToRecord(mlir::Type t) {
    for (int i = 0; i < 8 && t; ++i) {
        if (auto r = mlir::dyn_cast<fir::RecordType>(t)) return r;
        if (auto rt = mlir::dyn_cast<fir::ReferenceType>(t)) { t = rt.getEleTy(); continue; }
        if (auto bt = mlir::dyn_cast<fir::BoxType>(t))       { t = bt.getEleTy(); continue; }
        if (auto ht = mlir::dyn_cast<fir::HeapType>(t))      { t = ht.getEleTy(); continue; }
        if (auto pt = mlir::dyn_cast<fir::PointerType>(t))   { t = pt.getEleTy(); continue; }
        break;
    }
    return {};
}

// ---------------------------------------------------------------------------
// Phase 1 — Discovery
// ---------------------------------------------------------------------------

/// A discovered (parent_decl, member_name) → inner-record pairing for an
/// alloc-array-of-records access site.  Multiple designate ops in the
/// same function for the same (parent, member) collapse to one entry —
/// they all access the same logical companion.
struct LiftTarget {
    hlfir::DeclareOp parent;          // The struct holding the alloc-array member.
    std::string memberName;            // The member field name.
    fir::RecordType innerRecord;       // The element record type.
};

/// Walk a function and collect every distinct (parent_decl, member)
/// pair where the member is alloc-array-of-records.  Multiple uses
/// (different designate sites) of the same pair collapse.
static void discoverLiftTargets(mlir::func::FuncOp func,
                                llvm::SmallVectorImpl<LiftTarget> &out) {
    // Key: "<parent_op_address>::<member_name>" → index into out.
    // Op pointers are stable within a function walk.
    llvm::StringMap<unsigned> seen;

    func.walk([&](hlfir::DesignateOp dg) {
        auto compAttr = dg.getComponentAttr();
        if (!compAttr) return;
        std::string memberName = compAttr.str();

        // The designate's result type — must peel to a box-over-seq-of-record.
        auto resTy = dg.getResult().getType();
        // For ``%dg = hlfir.designate %parent{"<member>"}``, the result
        // type is ``ref<box<heap<seq<? x record>>>>``.
        mlir::Type peeled = resTy;
        if (auto rt = mlir::dyn_cast<fir::ReferenceType>(peeled))
            peeled = rt.getEleTy();
        auto innerRec = allocOrPtrArrayOfRecordsMember(peeled);
        if (!innerRec) return;

        // Walk the memref back to the parent's hlfir.declare.
        mlir::Value mr = dg.getMemref();
        hlfir::DeclareOp parentDecl;
        for (int hops = 0; hops < 8 && mr; ++hops) {
            auto *def = mr.getDefiningOp();
            if (!def) break;
            if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
                parentDecl = d;
                break;
            }
            // Peel intermediate aliases / wrappers.
            if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) { mr = cv.getValue(); continue; }
            if (auto bx = mlir::dyn_cast<fir::EmboxOp>(def))   { mr = bx.getMemref(); continue; }
            if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def))   { mr = rb.getBox(); continue; }
            if (auto ld = mlir::dyn_cast<fir::LoadOp>(def))    { mr = ld.getMemref(); continue; }
            break;
        }
        if (!parentDecl) return;

        // Dedupe.
        std::string key = std::to_string(
                              reinterpret_cast<uintptr_t>(parentDecl.getOperation()))
                          + "::" + memberName;
        auto [it, inserted] = seen.try_emplace(key,
                                               static_cast<unsigned>(out.size()));
        if (inserted) {
            LiftTarget t;
            t.parent = parentDecl;
            t.memberName = memberName;
            t.innerRecord = innerRec;
            out.push_back(std::move(t));
        }
    });
}

// ---------------------------------------------------------------------------
// Pass driver
// ---------------------------------------------------------------------------

struct LiftAllocArrayOfRecordsPass
    : public mlir::PassWrapper<LiftAllocArrayOfRecordsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftAllocArrayOfRecordsPass)

    llvm::StringRef getArgument() const final {
        return "hlfir-lift-alloc-array-of-records";
    }
    llvm::StringRef getDescription() const final {
        return "Lift `type(t), allocatable :: f(:)` (and pointer "
               "variants) struct members into top-level companions with "
               "a leading runtime-extent dim.  Rewrites every "
               "`<root>%<member>(<idx>)%<inner_leaf>(<...>)` chain into "
               "a direct access on the companion.";
    }

    void runOnOperation() override {
        getOperation().walk([&](mlir::func::FuncOp func) {
            llvm::SmallVector<LiftTarget, 4> targets;
            discoverLiftTargets(func, targets);

            // TODO Phase 2: synthesise companions.
            // TODO Phase 3: rewrite chains.
            // TODO Phase 4: cleanup.
            //
            // For now the pass is discovery-only — if any targets are
            // found, FlattenStructs's existing opaque-skip for
            // alloc-array-of-records members lets the outer struct
            // flatten with the member as a no-op slot, and the bridge
            // surfaces the un-lowered access chain as the existing
            // KeyError at SDFG build.  Subsequent commits land the
            // rewrite.
            (void)targets;
        });
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> createLiftAllocArrayOfRecordsPass() {
    return std::make_unique<LiftAllocArrayOfRecordsPass>();
}

}  // namespace hlfir_bridge
