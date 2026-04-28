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
    }

   private:
    void rewrite(hlfir::DeclareOp ptrDecl) {
        // Find the rebind store: ``fir.store %targetBox to %ptrDecl#0``
        // whose stored value is a ``fir.embox`` of something OTHER than
        // a ``fir.zero_bits`` (the latter is the initial nullify).
        fir::StoreOp rebindStore;
        hlfir::DeclareOp targetDecl;
        for (auto *u : ptrDecl.getResult(0).getUsers()) {
            auto st = mlir::dyn_cast<fir::StoreOp>(u);
            if (!st) continue;
            auto *valDef = st.getValue().getDefiningOp();
            auto embox = mlir::dyn_cast_or_null<fir::EmboxOp>(valDef);
            if (!embox) continue;
            // Skip the initial ``fir.embox %fir.zero_bits`` nullify.
            if (mlir::isa_and_nonnull<fir::ZeroOp>(
                    embox.getMemref().getDefiningOp()))
                continue;
            auto target = traceTarget(embox.getMemref());
            if (!target) continue;
            rebindStore = st;
            targetDecl  = target;
            break;
        }
        if (!rebindStore || !targetDecl) return;

        ptrDecl.emitWarning()
            << "hlfir-rewrite-pointer-assigns: collapsing pointer "
            << "rebind under the strict-no-aliasing assumption "
            << "(target: " << targetDecl.getUniqName().str() << ").  "
            << "Fortran allows aliased pointer access; if your program "
            << "relies on alias semantics this rewrite is unsafe.";

        mlir::Value targetRef = targetDecl.getResult(0);

        // Rewrite every reader chain ``fir.load ptrDecl#0 → fir.box_addr``
        // to yield ``targetRef`` directly.  The fir.box_addr's result has
        // type ``fir.ptr<T>`` while ``targetRef`` is ``fir.ref<T>``;
        // insert a fir.convert when downstream uses care.
        llvm::SmallVector<mlir::Operation *, 8> deadReaders;
        for (auto *u : ptrDecl.getResult(0).getUsers()) {
            auto ld = mlir::dyn_cast<fir::LoadOp>(u);
            if (!ld) continue;
            for (auto *uu : ld.getResult().getUsers()) {
                auto ba = mlir::dyn_cast<fir::BoxAddrOp>(uu);
                if (!ba) continue;
                mlir::Value replacement = targetRef;
                if (replacement.getType() != ba.getResult().getType()) {
                    mlir::OpBuilder b(ba);
                    replacement = b.create<fir::ConvertOp>(
                        ba.getLoc(), ba.getResult().getType(), targetRef);
                }
                ba.getResult().replaceAllUsesWith(replacement);
                deadReaders.push_back(ba);
            }
            deadReaders.push_back(ld);
        }
        for (auto *op : deadReaders)
            if (op->use_empty()) op->erase();

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
