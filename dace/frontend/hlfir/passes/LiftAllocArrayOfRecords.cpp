// ============================================================================
// LiftAllocArrayOfRecords.cpp  --  lift alloc-array-of-records struct members
// into direct accesses on the pointer-rebind target storage.
// ============================================================================
//
// Motivation:
//     Fortran pattern (canonical ICON shape, see
//     ``icon-artifacts/solve_nh/solve_nh_fake.f90:1576``):
//
//         type t_inner
//           real(kind=8), pointer, contiguous :: w(:, :, :)
//         end type t_inner
//         type t_outer
//           type(t_inner), allocatable :: items(:)   ! alloc-array of records
//         end type t_outer
//
//         type(t_outer) :: p_nh
//         real(kind=8), target :: storage(...)
//         allocate(p_nh%items(N))
//         do n = 1, N
//           p_nh%items(n)%w => storage(..., n)
//         end do
//         ! User code (with const or runtime element index ``jg``):
//         val = p_nh%items(jg)%w(jc, jk, jb)
//
//     FlattenStructs bails on ``box<heap|ptr<seq<? x record>>>`` members
//     because there's no static way to enumerate the alloc-array's
//     elements.  This pass closes the gap by exploiting a fundamental
//     fact: the runtime pointer rebinds inside the kernel bind each
//     element's pointer member to a slice of an existing storage array.
//     After those rebinds, ``p_nh%items(jg)%w(jc, jk, jb)`` is
//     observably equivalent to ``storage(jc, jk, jb, jg)`` (or some
//     other re-shaped slice access).
//
// What the pass does:
//
//     1. **Discovery**  --  walk for ``hlfir.designate %X{"<member>"}``
//        where the result's element type peels to
//        ``box<heap|ptr<seq<? x record>>>``.  Collect every distinct
//        ``(parent_decl, member_name)`` pair.
//
//     2. **Rebind tracking**  --  walk for ``fir.store %embox to
//        %dest`` where ``%dest`` is the per-element-pointer field
//        ``hlfir.designate (hlfir.designate (load
//        %parent_member))(%idx){"<inner>"}``. Parse the embox source (a
//        designate over an existing storage declare).  Record:
//           ``(parent_decl, member_name, element_idx_value, inner_member_name)
//             -> (storage_decl, slice_chain)``
//
//     3. **Access rewrite**  --  walk for ``fir.load %addr`` where
//        ``%addr`` is the same per-element-pointer field shape as in
//        Phase 2.  For each load of the box, find downstream
//        ``hlfir.designate %loaded_box (<access_indices>)`` ops.
//        Look up the matching rebind by tuple and replace the terminal
//        designate with a new designate over the storage parent that
//        merges the slice indices with the access indices.
//
//     4. **Cleanup**  --  erase the now-orphan setup ops:
//          * ``fir.call @_FortranAAllocatableAllocate / Deallocate``
//            whose first arg traces back to the parent member.
//          * The rebind ``fir.store`` ops (their target designates
//            have no remaining users).
//          * Walk the function once more to delete dead designate /
//            load / embox / fir.shape chains rooted in the parent
//            member.  Trust the canonicalizer that runs after this
//            pass to clean up anything we miss.
//
// Pipeline position:
//     After ``hlfir-inline-all`` (so inlined-callee element-alias
//     declares are visible) and BEFORE ``hlfir-flatten-structs`` (so
//     flatten sees clean top-level arrays after the lift).
//
// Uniformity contract:
//     Each element's inner pointer-member must be rebound (the
//     pattern requires the user to perform the rebind before any
//     access).  Non-uniform per-element shape is not statically
//     verified  --  trust the user-side contract.
//
// Out of scope:
//     - Multiple rebinds for the same `(idx, inner)` slot interleaved
//       with accesses: bail loudly with ``emitError``.
//     - Whole-AoS copies (``a = p%items``): hard-fail in validation.
//     - Reallocate-in-kernel of the alloc-array: hard-fail (the
//       discovery's allocate-site check rejects multi-allocate).
// ============================================================================

#include "bridge/trace_utils.h"  // limits::kAliasMemrefWalkDepth
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

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

// ---------------------------------------------------------------------------
// Per-(parent, member) lift state
// ---------------------------------------------------------------------------

/// A discovered ``(parent_decl, member_name)`` pair.  This is the
/// outer-struct slot we're lifting away.
struct LiftTarget {
  hlfir::DeclareOp parent;      // The struct holding the alloc-array member.
  std::string memberName;       // The member field name.
  fir::RecordType innerRecord;  // The element record type.
};

/// One rebind site: ``parent%member(idx)%inner_member => storage_chain``.
///
/// ``idx`` is the SSA value selecting the element (const or runtime).
/// ``sliceOp`` is the ``hlfir.designate`` op that produces the slice
/// of the parent storage, OR null if the bind target is the storage
/// declare itself (whole-array bind).
struct RebindInfo {
  fir::StoreOp storeOp;
  mlir::Value elemIdx;           // SSA value of the element index.
  std::string innerMemberName;   // The inner record's field name.
  hlfir::DeclareOp storageDecl;  // The backing storage declare.
  hlfir::DesignateOp sliceOp;    // Designate of the slice (may be null).
  // When the rebind store is inside a ``fir.do_loop`` whose induction
  // variable feeds ``elemIdx``, this rebind is a "wildcard"  --  it
  // symbolically establishes the equivalence
  //   ``parent%member(<I>)%inner === storage(<slice with I substituted>)``
  // for ANY value of ``I``.  Access sites with a runtime elemIdx
  // (not a constant, not the same SSA as any rebind's elemIdx) match
  // the wildcard rebind and substitute their own elemIdx for the
  // loop iter in the slice.
  bool isLoopIter = false;
  mlir::Value loopIterStorage;  // The alloca / declare that the
                                // loop counter writes into; used
                                // to identify slice scalars that
                                // reference the iter.
};

// ---------------------------------------------------------------------------
// Phase 1  --  Discovery
// ---------------------------------------------------------------------------

/// Walk a function and collect every distinct ``(parent_decl, member)``
/// pair where the member is alloc-array-of-records.
static void discoverLiftTargets(mlir::func::FuncOp func,
                                llvm::SmallVectorImpl<LiftTarget> &out) {
  llvm::StringMap<unsigned> seen;

  func.walk([&](hlfir::DesignateOp dg) {
    auto compAttr = dg.getComponentAttr();
    if (!compAttr) return;
    std::string memberName = compAttr.str();

    auto resTy = dg.getResult().getType();
    mlir::Type peeled = resTy;
    if (auto rt = mlir::dyn_cast<fir::ReferenceType>(peeled))
      peeled = rt.getEleTy();
    auto innerRec = allocOrPtrArrayOfRecordsMember(peeled);
    if (!innerRec) return;

    mlir::Value mr = dg.getMemref();
    hlfir::DeclareOp parentDecl;
    for (int hops = 0; hops < limits::kAliasMemrefWalkDepth && mr; ++hops) {
      auto *def = mr.getDefiningOp();
      if (!def) break;
      if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
        parentDecl = d;
        break;
      }
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
        mr = cv.getValue();
        continue;
      }
      if (auto bx = mlir::dyn_cast<fir::EmboxOp>(def)) {
        mr = bx.getMemref();
        continue;
      }
      if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def)) {
        mr = rb.getBox();
        continue;
      }
      if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
        mr = ld.getMemref();
        continue;
      }
      break;
    }
    if (!parentDecl) return;

    std::string key =
        std::to_string(reinterpret_cast<uintptr_t>(parentDecl.getOperation())) +
        "::" + memberName;
    auto [it, inserted] =
        seen.try_emplace(key, static_cast<unsigned>(out.size()));
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
// Chain matching helpers
// ---------------------------------------------------------------------------

/// Match a value of the shape
///   ``hlfir.designate (load (hlfir.designate %parent{"<member>"})) (%idx)``
/// against a known ``(parent, member)`` pair.  On match, returns the
/// SSA value of the element index (``%idx``) AND the inner element-
/// selection designate op.  On non-match, returns ``{null, null}``.
struct ElemMatch {
  mlir::Value elemIdx;
  hlfir::DesignateOp elemDg;
};

static ElemMatch matchElementSelect(mlir::Value v, hlfir::DeclareOp parent,
                                    llvm::StringRef memberName) {
  ElemMatch r{};
  // Peel through any intermediate ``hlfir.declare`` aliases  --  inlined
  // callees materialise these on per-element selections (e.g. the
  // inlined ``stuff`` alias declare between ``designate(elem_idx)``
  // and ``designate{"<inner>"}``).
  for (int hops = 0; hops < limits::kAliasMemrefWalkDepth && v; ++hops) {
    auto *vd = v.getDefiningOp();
    if (!vd) return r;
    auto aliasDecl = mlir::dyn_cast<hlfir::DeclareOp>(vd);
    if (!aliasDecl) break;
    v = aliasDecl.getMemref();
  }
  auto *def = v.getDefiningOp();
  if (!def) return r;
  auto elemDg = mlir::dyn_cast<hlfir::DesignateOp>(def);
  if (!elemDg) return r;
  // Element-select designate: has no component name, takes a single
  // index operand.
  if (elemDg.getComponentAttr()) return r;
  auto idxs = elemDg.getIndices();
  if (idxs.size() != 1) return r;

  // The memref must trace back to load(designate %parent{"<member>"}).
  mlir::Value mr = elemDg.getMemref();
  for (int hops = 0; hops < limits::kAliasMemrefWalkDepth && mr; ++hops) {
    auto *mdef = mr.getDefiningOp();
    if (!mdef) return r;
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(mdef)) {
      auto *ldef = ld.getMemref().getDefiningOp();
      if (auto memberDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(ldef)) {
        auto comp = memberDg.getComponentAttr();
        if (!comp || comp.str() != memberName.str()) return r;
        // Check the memberDg's memref is parent's result.
        if (memberDg.getMemref() != parent.getResult(0) &&
            memberDg.getMemref() != parent.getResult(1))
          return r;
        r.elemIdx = idxs[0];
        r.elemDg = elemDg;
        return r;
      }
      return r;
    }
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(mdef)) {
      mr = cv.getValue();
      continue;
    }
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(mdef)) {
      mr = rb.getBox();
      continue;
    }
    return r;
  }
  return r;
}

/// Match ``hlfir.designate <elem_value>{"<inner_member>"}``.
/// Returns the inner-member designate op, or null.
static hlfir::DesignateOp matchInnerMember(mlir::Value v) {
  auto *def = v.getDefiningOp();
  if (!def) return {};
  auto dg = mlir::dyn_cast<hlfir::DesignateOp>(def);
  if (!dg) return {};
  if (!dg.getComponentAttr()) return {};
  return dg;
}

/// Walk an embox source value to find a (storage_decl, slice_op) pair.
/// Handles:
///   * ``fir.embox %slice`` where ``%slice = hlfir.designate %decl (...)``.
///   * ``fir.rebox %inner_box`` similarly.
static std::pair<hlfir::DeclareOp, hlfir::DesignateOp> parseEmboxSource(
    mlir::Value emboxVal) {
  auto *def = emboxVal.getDefiningOp();
  while (def) {
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def)) {
      def = rb.getBox().getDefiningOp();
      continue;
    }
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      def = cv.getValue().getDefiningOp();
      continue;
    }
    break;
  }
  auto embox = mlir::dyn_cast_or_null<fir::EmboxOp>(def);
  if (!embox) return {{}, {}};

  mlir::Value src = embox.getMemref();
  auto *srcDef = src.getDefiningOp();
  if (!srcDef) return {{}, {}};

  // The embox source may be a designate of a storage declare, or the
  // declare itself.
  if (auto sliceDg = mlir::dyn_cast<hlfir::DesignateOp>(srcDef)) {
    // Walk the slice's memref back to the storage declare.
    mlir::Value mr = sliceDg.getMemref();
    for (int hops = 0; hops < limits::kAliasMemrefWalkDepth && mr; ++hops) {
      auto *mdef = mr.getDefiningOp();
      if (!mdef) break;
      if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(mdef)) return {d, sliceDg};
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(mdef)) {
        mr = cv.getValue();
        continue;
      }
      break;
    }
    return {{}, {}};
  }
  if (auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(srcDef))
    return {declOp, hlfir::DesignateOp{}};
  return {{}, {}};
}

// ---------------------------------------------------------------------------
// Phase 2  --  Rebind tracking
// ---------------------------------------------------------------------------

/// Walk a value back through ``fir.convert`` / ``fir.load`` peels and
/// return the underlying ``hlfir.declare`` (or ``fir.alloca``) memref
/// if the chain bottoms out at one, else null.  Used to identify when
/// an SSA value is "the loop counter alloca" so a rebind inside a
/// ``fir.do_loop`` can be recognised as a wildcard rebind.
static mlir::Value traceToCounterAlloca(mlir::Value v) {
  for (int i = 0; i < limits::kConvertChainDepth && v; ++i) {
    auto *def = v.getDefiningOp();
    if (!def) return {};
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      v = cv.getValue();
      continue;
    }
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
      return ld.getMemref();
    }
    return {};
  }
  return {};
}

/// True iff ``op`` has a ``fir.do_loop`` ancestor anywhere up the
/// region tree.
static fir::DoLoopOp findEnclosingDoLoop(mlir::Operation *op) {
  for (auto *p = op->getParentOp(); p; p = p->getParentOp())
    if (auto dl = mlir::dyn_cast<fir::DoLoopOp>(p)) return dl;
  return {};
}

/// Walk a function and collect every rebind store targeting an
/// inner-pointer-member of ``target.parent``'s alloc-array-of-records.
static void collectRebinds(mlir::func::FuncOp func, LiftTarget &target,
                           llvm::SmallVectorImpl<RebindInfo> &out) {
  func.walk([&](fir::StoreOp store) {
    // The dest must be the inner-member designate.
    auto destDg = matchInnerMember(store.getMemref());
    if (!destDg) return;

    // The dest's memref must be the element-select chain.
    auto em = matchElementSelect(destDg.getMemref(), target.parent,
                                 target.memberName);
    if (!em.elemDg) return;

    // The stored value must be an embox of a slice / declare.
    auto [storageDecl, sliceDg] = parseEmboxSource(store.getValue());
    if (!storageDecl) return;

    RebindInfo info;
    info.storeOp = store;
    info.elemIdx = em.elemIdx;
    info.innerMemberName = destDg.getComponentAttr().str();
    info.storageDecl = storageDecl;
    info.sliceOp = sliceDg;

    // Detect loop-iter rebind: store is inside a ``fir.do_loop``
    // AND the elemIdx traces back through ``convert/load`` to a
    // counter alloca that is itself written by the loop's iter
    // arg.  When this holds, the rebind is a TEMPLATE valid for
    // any element index.
    if (auto dl = findEnclosingDoLoop(store)) {
      (void)dl;  // The loop op itself isn't needed; only the
                 // counter alloca that ``elemIdx`` loads from.
      if (auto counter = traceToCounterAlloca(em.elemIdx)) {
        info.isLoopIter = true;
        info.loopIterStorage = counter;
      }
    }
    out.push_back(std::move(info));
  });
}

// ---------------------------------------------------------------------------
// Phase 3  --  Access rewrite
// ---------------------------------------------------------------------------

/// Try to look up a rebind matching ``(elemIdx, innerMemberName)``.
/// Idx matching is structural: same SSA value, OR same constant.
///
/// Returns the first matching rebind, or null on no match.
static RebindInfo *findRebind(llvm::SmallVectorImpl<RebindInfo> &rebinds,
                              mlir::Value elemIdx,
                              llvm::StringRef innerMemberName) {
  // Try same-value match first.
  for (auto &r : rebinds) {
    if (r.innerMemberName != innerMemberName.str()) continue;
    if (r.elemIdx == elemIdx) return &r;
  }
  // Constant-value fallback.
  std::optional<int64_t> needed;
  if (auto *def = elemIdx.getDefiningOp())
    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
        needed = ia.getInt();
  if (needed) {
    for (auto &r : rebinds) {
      if (r.innerMemberName != innerMemberName.str()) continue;
      if (auto *def = r.elemIdx.getDefiningOp())
        if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
          if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
            if (ia.getInt() == *needed) return &r;
    }
  }
  // Wildcard (loop-iter template) fallback  --  symbolic match.
  //
  // When a rebind is inside a ``fir.do_loop`` whose iter feeds the
  // rebind's elemIdx, the rebind establishes a symbolic template
  // ``parent%member(<I>)%inner === storage(<slice with I substituted>)``
  // for any element index ``I``.  Match on inner_member name alone
  // and let ``buildLiftedAccess`` substitute the access's elemIdx
  // into the slice's loop-iter-referencing scalar slots.
  for (auto &r : rebinds) {
    if (!r.isLoopIter) continue;
    if (r.innerMemberName != innerMemberName.str()) continue;
    return &r;
  }
  return nullptr;
}

/// Materialise an access through the lifted-out chain as a direct
/// designate over the rebind's storage parent.
///
/// Inputs:
///   * ``accessDg``  --  the user-side designate selecting one element
///     of the rebound pointer-array (e.g. ``%w_box(%c1, %c1)``).  Its
///     result type is ``ref<T>`` for some scalar T.
///   * ``rebind``  --  the matching rebind site.
///   * ``elemIdx``  --  the per-access element index value (matches the
///     rebind's same-element).
///
/// Returns a fresh ``hlfir::DesignateOp`` whose result has the SAME
/// type as ``accessDg`` and accesses the equivalent location in the
/// rebind's storage parent.  Insertion point is set immediately
/// before ``accessDg``.
///
/// Strategy: build new indices = ``rebind.sliceOp``'s scalar indices
/// PLUS ``accessDg``'s indices.  For example:
///   * rebind ``storage(:, :, jg)`` (slice = triplet, triplet, scalar=jg)
///   * access ``w_box(i, j)`` (indices = i, j)
///   * merged: ``storage(i, j, jg)``
///
/// The triplet entries in the slice get consumed by the access
/// indices (rebased by ``lo - 1`` if non-1, but for ``(:, :, jg)``
/// the lower bounds are 1, so no rebase).  Scalar entries pass
/// through verbatim from the slice.
static mlir::Value buildLiftedAccess(mlir::OpBuilder &b,
                                     hlfir::DesignateOp accessDg,
                                     RebindInfo &rebind,
                                     mlir::Value accessElemIdx) {
  auto loc = accessDg.getLoc();
  b.setInsertionPoint(accessDg);

  auto accessIdxs = accessDg.getIndices();
  llvm::SmallVector<mlir::Value, 6> newIdxs;

  // For a wildcard (loop-iter) rebind, the slice's scalar slot that
  // references the loop counter alloca gets substituted with the
  // access-site element index.  Helper: given a slice scalar value,
  // walk through ``fir.convert`` / ``fir.load`` to see if it traces
  // to the same counter alloca recorded on the rebind.  If yes,
  // return the access elemIdx (with a type-matching ``fir.convert``
  // if the slice slot's i64-ness disagrees with the access elemIdx).
  auto substIfLoopIter = [&](mlir::Value sliceScalar) -> mlir::Value {
    if (!rebind.isLoopIter) return sliceScalar;
    if (!rebind.loopIterStorage) return sliceScalar;
    if (auto traced = traceToCounterAlloca(sliceScalar)) {
      if (traced == rebind.loopIterStorage) {
        // Type-match: if the slice scalar is i64 and the access
        // elemIdx is i32 (or vice versa), insert a convert.
        if (accessElemIdx.getType() != sliceScalar.getType()) {
          return b
              .create<fir::ConvertOp>(loc, sliceScalar.getType(), accessElemIdx)
              .getResult();
        }
        return accessElemIdx;
      }
    }
    return sliceScalar;
  };

  if (rebind.sliceOp) {
    auto sliceIdxs = rebind.sliceOp.getIndices();
    auto sliceTriplets = rebind.sliceOp.getIsTriplet();
    unsigned cursor = 0;
    unsigned accessCursor = 0;
    for (unsigned k = 0; k < sliceTriplets.size(); ++k) {
      if (sliceTriplets[k]) {
        // Triplet  --  consume one access index.
        if (accessCursor >= accessIdxs.size())
          return {};  // Shape mismatch: bail.
        newIdxs.push_back(accessIdxs[accessCursor++]);
        cursor += 3;
      } else {
        // Scalar  --  pass through from the slice, unless this is
        // a wildcard rebind and the scalar references the loop
        // counter (in which case substitute access elemIdx).
        newIdxs.push_back(substIfLoopIter(sliceIdxs[cursor]));
        cursor += 1;
      }
    }
    // Any remaining access indices (shouldn't happen  --  the slice
    // should have one triplet per access index): bail.
    if (accessCursor != accessIdxs.size()) return {};
  } else {
    // Whole-array bind: storage parent is accessed directly with
    // the access indices.
    for (auto idx : accessIdxs) newIdxs.push_back(idx);
  }

  // Build the result designate over the storage parent.  Use the
  // short-form builder which handles all the operand-segment-sizes
  // and ``is_triplet`` attributes (all-false for an element-access
  // designate, which is what we always produce).
  auto declRef = rebind.storageDecl.getResult(0);
  auto newDg = b.create<hlfir::DesignateOp>(loc, accessDg.getResult().getType(),
                                            declRef, mlir::ValueRange{newIdxs});
  return newDg.getResult();
}

/// Walk a function for every access through the lifted chain and
/// rewrite each one.  Returns true if any rewrite happened.
static bool rewriteAccesses(mlir::func::FuncOp func, LiftTarget &target,
                            llvm::SmallVectorImpl<RebindInfo> &rebinds) {
  // Pattern to match the access shape:
  //   ``%addr = hlfir.designate %inner_box (<access_indices>)``
  //   where ``%inner_box = fir.load %field_addr``
  //   where ``%field_addr = hlfir.designate %elem{"<inner>"}``
  //   where ``%elem = hlfir.designate (load (designate %parent{"<member>"}))
  //   (%idx)``
  bool changed = false;
  mlir::OpBuilder b(func.getContext());

  struct Job {
    hlfir::DesignateOp accessDg;
    RebindInfo *rebind;
    mlir::Value accessElemIdx;
  };
  llvm::SmallVector<Job, 8> jobs;

  func.walk([&](hlfir::DesignateOp accessDg) {
    // Element-access designate: no component, has indices.
    if (accessDg.getComponentAttr()) return;
    if (accessDg.getIndices().empty()) return;

    // Walk the memref: must be a load of an inner-member designate.
    auto *mdef = accessDg.getMemref().getDefiningOp();
    auto ld = mlir::dyn_cast_or_null<fir::LoadOp>(mdef);
    if (!ld) return;
    auto innerDg = matchInnerMember(ld.getMemref());
    if (!innerDg) return;
    auto em = matchElementSelect(innerDg.getMemref(), target.parent,
                                 target.memberName);
    if (!em.elemDg) return;

    // Find the matching rebind.
    RebindInfo *r =
        findRebind(rebinds, em.elemIdx, innerDg.getComponentAttr().str());
    if (!r) return;

    jobs.push_back({accessDg, r, em.elemIdx});
  });

  for (auto &job : jobs) {
    auto newVal =
        buildLiftedAccess(b, job.accessDg, *job.rebind, job.accessElemIdx);
    if (!newVal) continue;
    job.accessDg.getResult().replaceAllUsesWith(newVal);
    job.accessDg.erase();
    changed = true;
  }
  return changed;
}

// ---------------------------------------------------------------------------
// Phase 3b  --  alias-without-rebind redirection
// ---------------------------------------------------------------------------
//
// Some test patterns (notably ``type_arg`` with callee inlining) have
// NO explicit rebind in scope  --  the user calls a subroutine that takes
// the inner pointer-array as an assumed-shape dummy.  After
// ``hlfir-inline-all``, Flang materialises an alias declare on the
// loaded pointer box (``%my_arr = hlfir.declare (fir.rebox (fir.load
// %w_field)))``).  That alias is what the bridge already registers
// as a top-level variable.
//
// But Flang also emits a SECOND access through the bare designate
// chain (the original Fortran-source-author's read in main).  That
// chain doesn't go through the inlined alias  --  it walks the
// ``designate{member} -> load -> designate(idx) -> designate{inner_member}
// -> load -> designate(access_indices)`` shape directly.  ``traceToDecl``
// on its terminal element returns the OUTER struct's name (``p_prog``)
// instead of the inlined alias's name (``my_arr``), and the bridge
// emits an access against ``p_prog`` which isn't a registered array.
//
// Fix: walk the function for inlined-callee alias declares whose
// memref chain matches ``rebox?(load(designate(elemSelect, member)))``
// on a known (parent, member) target.  Record
// ``(parent, member, elemIdx_const, inner_member) -> alias_decl``.
// Then walk for access chains matching the same shape but ending
// at the bare ``fir.load`` (no alias declare).  Rewrite the terminal
// designate to use the alias declare's result instead.

struct AliasInfo {
  hlfir::DeclareOp aliasDecl;   // The inlined-callee alias declare.
  mlir::Value elemIdx;          // The element index value.
  std::string innerMemberName;  // The inner member name.
};

/// Walk a function and find every inlined-callee alias declare whose
/// memref chain matches the (parent, member) target.
static void collectAliases(mlir::func::FuncOp func, LiftTarget &target,
                           llvm::SmallVectorImpl<AliasInfo> &out) {
  func.walk([&](hlfir::DeclareOp decl) {
    if (decl == target.parent) return;
    // The alias's memref must peel back through rebox / load to
    // an inner-member designate chain.
    mlir::Value mr = decl.getMemref();
    for (int hops = 0; hops < limits::kAliasMemrefWalkDepth && mr; ++hops) {
      auto *mdef = mr.getDefiningOp();
      if (!mdef) return;
      if (auto rb = mlir::dyn_cast<fir::ReboxOp>(mdef)) {
        mr = rb.getBox();
        continue;
      }
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(mdef)) {
        mr = cv.getValue();
        continue;
      }
      if (auto ld = mlir::dyn_cast<fir::LoadOp>(mdef)) {
        // Got the load  --  the load's memref is an inner-member
        // designate.
        auto innerDg = matchInnerMember(ld.getMemref());
        if (!innerDg) return;
        auto em = matchElementSelect(innerDg.getMemref(), target.parent,
                                     target.memberName);
        if (!em.elemDg) return;
        AliasInfo info;
        info.aliasDecl = decl;
        info.elemIdx = em.elemIdx;
        info.innerMemberName = innerDg.getComponentAttr().str();
        out.push_back(std::move(info));
        return;
      }
      return;
    }
  });
}

/// Find an alias matching ``(elemIdx, innerMemberName)``, using the
/// same SSA-value-or-constant matching as ``findRebind``.
static AliasInfo *findAlias(llvm::SmallVectorImpl<AliasInfo> &aliases,
                            mlir::Value elemIdx,
                            llvm::StringRef innerMemberName) {
  for (auto &a : aliases) {
    if (a.innerMemberName != innerMemberName.str()) continue;
    if (a.elemIdx == elemIdx) return &a;
  }
  std::optional<int64_t> needed;
  if (auto *def = elemIdx.getDefiningOp())
    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
        needed = ia.getInt();
  if (!needed) return nullptr;
  for (auto &a : aliases) {
    if (a.innerMemberName != innerMemberName.str()) continue;
    if (auto *def = a.elemIdx.getDefiningOp())
      if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
        if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
          if (ia.getInt() == *needed) return &a;
  }
  return nullptr;
}

/// Walk the function for direct-chain accesses that match an alias's
/// shape but use the bare load chain instead of the alias declare.
/// Rewrite each one to use the alias's result.
static bool redirectAccessesToAliases(
    mlir::func::FuncOp func, LiftTarget &target,
    llvm::SmallVectorImpl<AliasInfo> &aliases) {
  if (aliases.empty()) return false;
  bool changed = false;
  mlir::OpBuilder b(func.getContext());

  llvm::SmallVector<std::pair<hlfir::DesignateOp, AliasInfo *>, 8> jobs;
  func.walk([&](hlfir::DesignateOp accessDg) {
    // Element access: no component, has indices.
    if (accessDg.getComponentAttr()) return;
    if (accessDg.getIndices().empty()) return;
    // Memref must be a load of inner-member designate.
    auto *mdef = accessDg.getMemref().getDefiningOp();
    auto ld = mlir::dyn_cast_or_null<fir::LoadOp>(mdef);
    if (!ld) return;
    auto innerDg = matchInnerMember(ld.getMemref());
    if (!innerDg) return;
    auto em = matchElementSelect(innerDg.getMemref(), target.parent,
                                 target.memberName);
    if (!em.elemDg) return;
    // Find matching alias.
    AliasInfo *a =
        findAlias(aliases, em.elemIdx, innerDg.getComponentAttr().str());
    if (!a) return;
    // Skip if this access is already rooted at the alias.
    if (em.elemDg == nullptr) return;  // guard for null
    jobs.push_back({accessDg, a});
  });

  for (auto &[accessDg, a] : jobs) {
    // Build replacement: ``hlfir.designate %alias (accessIdxs...)``.
    b.setInsertionPoint(accessDg);
    llvm::SmallVector<mlir::Value, 4> idxs(accessDg.getIndices().begin(),
                                           accessDg.getIndices().end());
    auto newDg = b.create<hlfir::DesignateOp>(
        accessDg.getLoc(), accessDg.getResult().getType(),
        a->aliasDecl.getResult(0), mlir::ValueRange{idxs});
    accessDg.getResult().replaceAllUsesWith(newDg.getResult());
    accessDg.erase();
    changed = true;
  }
  return changed;
}

// ---------------------------------------------------------------------------
// Phase 4  --  Cleanup
// ---------------------------------------------------------------------------

/// After rewrites, the rebind store / embox / inner-member designate /
/// elem-select / load chains have no remaining users for the lifted
/// member.  Erase the rebind stores and the runtime allocate /
/// deallocate calls so downstream passes don't choke.
static void cleanup(mlir::func::FuncOp func, LiftTarget &target,
                    llvm::SmallVectorImpl<RebindInfo> &rebinds) {
  // Erase rebind stores (their dest designates have no users now).
  for (auto &r : rebinds) {
    if (r.storeOp) r.storeOp.erase();
  }

  // Erase Fortran runtime allocate / deallocate calls touching the
  // member.  The first argument is a fir.convert of the member-
  // designate's result; check that the underlying designate matches.
  llvm::SmallVector<fir::CallOp, 4> deadCalls;
  func.walk([&](fir::CallOp call) {
    auto callee = call.getCallee();
    if (!callee) return;
    auto name = callee->getRootReference().getValue();
    if (name != "_FortranAAllocatableAllocate" &&
        name != "_FortranAAllocatableDeallocate" &&
        name != "_FortranAAllocatableSetBounds")
      return;
    if (call.getNumOperands() == 0) return;
    mlir::Value arg = call.getOperand(0);
    for (int hops = 0; hops < limits::kAliasMemrefWalkDepth && arg; ++hops) {
      auto *def = arg.getDefiningOp();
      if (!def) break;
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
        arg = cv.getValue();
        continue;
      }
      if (auto memberDg = mlir::dyn_cast<hlfir::DesignateOp>(def)) {
        auto comp = memberDg.getComponentAttr();
        if (comp && comp.str() == target.memberName &&
            (memberDg.getMemref() == target.parent.getResult(0) ||
             memberDg.getMemref() == target.parent.getResult(1))) {
          deadCalls.push_back(call);
        }
      }
      break;
    }
  });
  for (auto c : deadCalls) c.erase();
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
           "variants) struct members by tracing the user's pointer "
           "rebinds and rewriting every "
           "`<root>%<member>(<idx>)%<inner>(<...>)` access to the "
           "rebind target's storage directly.";
  }

  void runOnOperation() override {
    getOperation().walk([&](mlir::func::FuncOp func) {
      llvm::SmallVector<LiftTarget, 4> targets;
      discoverLiftTargets(func, targets);
      for (auto &t : targets) {
        llvm::SmallVector<RebindInfo, 4> rebinds;
        collectRebinds(func, t, rebinds);
        if (!rebinds.empty()) {
          rewriteAccesses(func, t, rebinds);
          cleanup(func, t, rebinds);
          continue;
        }
        // No explicit rebind in scope  --  fall back to alias
        // redirection.  This handles the callee-inlining case
        // where Flang materialises an alias declare on the
        // loaded pointer-array; redirect direct-chain accesses
        // to that alias.
        llvm::SmallVector<AliasInfo, 4> aliases;
        collectAliases(func, t, aliases);
        if (!aliases.empty()) redirectAccessesToAliases(func, t, aliases);
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createLiftAllocArrayOfRecordsPass() {
  return std::make_unique<LiftAllocArrayOfRecordsPass>();
}

}  // namespace hlfir_bridge
