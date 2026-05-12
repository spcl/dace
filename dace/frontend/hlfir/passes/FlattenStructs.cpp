// ============================================================================
// FlattenStructs.cpp — Array-of-Structs → Struct-of-Arrays at the HLFIR level.
// ============================================================================
//
// Goal
// ----
// Eliminate Fortran derived types from the IR before SDFG construction so the
// SDFG sees only flat per-member arrays.  Real-world ICON / ECRAD / QE code
// wraps many arrays in a struct (``type :: state_t; real(8) :: u(...), v(...)
// end type``) and passes the struct across subroutine boundaries.  DaCe's
// SDFG model handles flat arrays beautifully and structures awkwardly, so
// we rewrite at the HLFIR level where every struct access is a chain of
// ``hlfir.designate`` ops we can pattern-match.
//
// Design analogy: DaCe's ``StructToContainerGroups`` pass
// (``dace/transformation/passes/struct_to_container_group.py``) does the
// same job at the SDFG level.  Same recursive walk over record members,
// same SoA naming model, same outer-shape concatenation for array-of-
// struct.  The two passes are post-/pre-SDFG mirrors of one transform.
//
// Three flattening shapes
// -----------------------
// 1. **Scalar struct, all flat members.**  ``type t :: real
//    u(M); real v(N)`` produces ``<base>_u`` of shape ``(M)`` and
//    ``<base>_v`` of shape ``(N)``.  Single-level designate rewrite.
//
// 2. **Array-of-struct (AoS) with array members.**
//    ``type(t), dimension(K) :: A`` where each member is itself an
//    array shape concatenates outer × inner: ``A_u`` of shape
//    ``(K, M)``, ``A_v`` of shape ``(K, N)``.  ``A(i)%u(j)`` rewrites
//    to ``A_u(i, j)`` — outer + inner indices merged in
//    ``rewriteDesignate``.  ``A(i)%u`` (whole-member access without
//    inner indices) rewrites to a triplet section ``A_u(i, 1:M:1)``.
//
// 3. **Nested record.**  Members that are themselves
//    record types unfold recursively; the leaf is whatever scalar /
//    array-of-scalar terminates the chain.  ``o%inner%x(j)`` rewrites
//    to a single flat ``o_inner_x(j)``.  ``collectFlatLeaves`` walks
//    every path to a flat leaf and ``rewriteDesignateChain`` walks
//    back through the designate chain to identify the matching
//    ``leafBase`` entry.
//
// Cross-subroutine handling
// -------------------------
// Struct dummy arguments get the same treatment.  ``replaceStructArg``
// inserts one block arg per member (or per leaf for nested) into the
// function signature and renames the function with ``_soa`` suffix.
// Inlined callee dummy declares that alias the outer struct via
// ``hlfir-inline-all`` are followed via ``collectFrom`` recursing
// through ``hlfir.declare`` users — the inlined alias chain is
// transparent to the rewrite.  ``recordStructArgEntry`` writes a
// ``hlfir.flatten_plan`` attribute the bindings emitter consumes to
// generate caller-side pack/unpack wrappers.
//
// Static-shape assumption
// -----------------------
// Every member shape and outer-array extent must fold to a
// compile-time constant — except for the allocatable-array member
// case (Phase 5a) below.  Pointer members and AoS-with-allocatable
// members are still out of scope and surface as
// loud-failure throws at ``extract_vars`` (``fir.RecordType``
// reaches a declare).
//
// Phase 5a — allocatable scalar-struct local member
// -------------------------------------------------
// ``type t :: real, allocatable :: w(:)`` paired with a LOCAL
// ``type(t) :: s`` instance flattens to a flat top-level
// allocatable ``s_w`` (declare carrying ``fortran_attrs =
// #fir.var_attrs<allocatable>``) plus per-allocate-site renames so
// flang's ``fir.allocmem`` op (originally named after the member's
// module scope, e.g. ``_QMlibEw.alloc``) appears under
// ``s_w.alloc`` — the convention the bridge's ``collectAllocSites``
// walks.  Companion change: ``extract_vars.cpp`` pass 2b also walks
// every ``fir.allocmem``'s shape operands and promotes the traced
// declares to symbols, so ``allocate(s%w(n))`` (without any
// surrounding do-loop) doesn't leave ``n`` as a scalar that
// collides with the array-extent symbol downstream.
//
// Phase 5a is gated to: scalar-outer (no AoS) + allocatable / pointer
// array member.  Phase 5b extended to dummy-arg structs and pointer
// members.  AoS-with-allocatable, nested-struct-allocatables, and
// reallocation-inside-kernel for AoS members are still deferred to
// Phase 5c.
//
// Phase 5c — AoS + allocatable members
// ------------------------------------
// ``type t :: real, allocatable :: w(:); type(t) :: A(N)`` — each
// batch instance ``A(i)`` owns its own runtime descriptor for
// ``A(i)%w``.  Two sub-cases share one logical contract
// (padding-to-max), but the IR shape and helpers differ.
//
// 5c-A — local instance, kernel-internal allocate (compile-time uniform)
//   When ``A`` is a local ``fir.alloca`` and every
//   ``allocate(A(i)%w(M))`` site uses the same compile-time constant
//   ``M``, ``aosAllocUniformConstSize`` returns ``M`` and we synthesise
//   a fully static companion ``A_w : ref<array<N x M x T>>``.  The
//   per-instance allocate / freemem chain becomes dead and is erased
//   by ``eraseAosAllocDeallocChain``.  Read-side pattern
//   ``fir.load + designate(loaded, j)`` is folded into a direct
//   2-index designate over the new flat declare by
//   ``collapseAosAllocReads``; whole-component assigns
//   (``A(i)%w = scalar``) are rewritten to row-section assigns
//   (``A_w(i, 1:M:1) = ...``) by ``rewriteAosWholeMemberAssign`` so
//   the existing concat path doesn't broadcast across all rows.
//
// 5c-B (inlined) — module-contained kernel after ``hlfir-inline-all``
//   When the AoS+allocatable struct is the dummy of a module-contained
//   subroutine, ``hlfir-inline-all`` splices the body in and the
//   inlined dummy becomes an alias declare carrying ``dummy_scope``.
//   ``collapseAosAllocReads`` follows the alias chain
//   (``hlfir.declare`` → ``fir.embox`` / ``fir.convert``) back to the
//   original declare so reads inside the inlined body are still
//   collapsed.
//
// 5c-B (true SDFG-boundary) — ``intent(inout)`` AoS struct dummy
//   When the AoS+allocatable struct is the dummy of the SDFG entry
//   itself, the per-instance sizes are runtime-determined and
//   generally differ.  ``replaceStructArg`` inserts two block args
//   per allocatable member:
//     * ``cap_<base>_<m>`` of type ``ref<index>`` — runtime cap
//     * ``<base>_<m>`` of type ``ref<array<N x ?xT>>`` — 2D buffer
//   It synthesises a declare for each, with ``uniq_name = "cap_..."``
//   on the cap declare so ``traceToDecl`` resolves the data declare's
//   inner extent to ``cap_<base>_<m>`` on the SDFG signature.
//   ``recordAosAllocEntry`` emits one ``aos_alloc=True`` FlattenEntry
//   per allocatable member; ``recordStructArgEntry`` takes an
//   exclude-set so non-allocatable siblings are still covered by a
//   separate aliasable entry (mixed structs are split into one
//   per-member aos_alloc entry plus one regular entry).
//
// Bindings-side contract for 5c-B (true boundary).  Stamped in the
// recipe's ``aos_alloc=True`` + ``cap_symbol`` fields and consumed
// by ``bindings/loop_copy.py``:
//   1. cap = max_i(merge(size(A(i)%w), 0, allocated(A(i)%w)))
//   2. allocate(A_w(N, cap)); zero-init.
//   3. Per i with allocated(A(i)%w): A_w(i, 1:size(A(i)%w)) = A(i)%w.
//   4. Call SDFG with the buffer + cap symbol.
//   5. On intent(out)/(inout) and per allocated row: copy back
//      A(i)%w = A_w(i, 1:size(A(i)%w)).
//   6. deallocate(A_w).
// Saved policy: NO runtime ``allocated()`` checks inside the SDFG —
// the bindings handle every allocation query.  Mixed allocation
// states are allowed; unallocated rows stay zero-padded and the
// user's program logic must avoid reading them.  Empty-batch
// sentinel (``cap == 0 → 1``) keeps the buffer non-degenerate.
//
// 5c-C — kernel-internal reallocation (NOT YET SUPPORTED)
//   When the kernel itself runs ``allocate(A(i)%w(N_i))`` (e.g. the
//   struct comes in ``intent(out)`` with no live data), the
//   bindings-time max is unknown.  Two follow-up directions:
//     TODO-1: HLFIR shape-discovery pre-pass that interprets each
//             ``allocate`` as size-discovery, collects all ``N_i``,
//             computes ``max(N_i)``, then re-runs normally.
//             Requires re-runnability of the discovery body.
//     TODO-2: F90 source-level rewrite that lifts each per-instance
//             allocate into a single max-sized pre-allocation.
//             Cleaner than the runtime two-pass approach but requires
//             understanding user code's scope / lifetime semantics.
//
// Things this pass deliberately does NOT do
// -----------------------------------------
// * Truly virtual polymorphic dispatch — handled separately by
//   ``fir-polymorphic-op`` (devirtualises) and
//   ``hlfir-reject-polymorphism`` (loud-fails on residuals).  This
//   pass peels ``fir.class<T>`` like ``fir.box<T>`` so monomorphic
//   CLASS receivers flatten through the same path as TYPE.
// * Nested struct with allocatable members at depth > 1
//   (``outer%inner%w(:)``) — needs the nested-record path to also
//   recognise allocatables on inner records.
// * Reallocation inside the kernel for AoS-allocatable companions
//   (Phase 5c-C TODO-1 / TODO-2 above).
//
// Naming caveat
// -------------
// Per-leaf names join the path with ``_``: ``base_member1_member2``.
// This is ambiguous if user code happens to name a struct field
// ``inner_x`` AND another field ``inner`` with subfield ``x`` —
// both would map to ``base_inner_x``.  Fortran style discourages
// underscores in field names so the collision risk is small in
// practice.  DaCe's container-groups pass uses delimited prefixes
// (``__CG_/__CA_/__m_``) to avoid this; we'd need to migrate the
// recipe consumers in lockstep to switch.
// ============================================================================

#include "passes/Passes.h"
#include "bridge/trace_utils.h"  // traceConstInt for AoS+allocatable size resolution

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace hlfir_bridge {

namespace {

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

/// Strip one layer of fir.box / fir.class / fir.ref / fir.heap /
/// fir.pointer.  ``fir.box<T>`` and ``fir.class<T>`` share the
/// ``fir::BaseBoxType`` base — peeling either via that common base
/// lets monomorphic CLASS declares flatten through the same rewrite
/// path as non-polymorphic TYPE declares.  Surviving virtual
/// dispatch is caught by ``hlfir-reject-polymorphism``, not here.
static mlir::Type unwrapOne(mlir::Type t) {
    if (auto x = mlir::dyn_cast<fir::BaseBoxType>(t))   return x.getEleTy();
    if (auto x = mlir::dyn_cast<fir::ReferenceType>(t)) return x.getEleTy();
    if (auto x = mlir::dyn_cast<fir::HeapType>(t))      return x.getEleTy();
    if (auto x = mlir::dyn_cast<fir::PointerType>(t))   return x.getEleTy();
    return t;
}

/// Walk through every wrapper until we hit a non-wrapper.
static mlir::Type unwrapAll(mlir::Type t) {
    for (;;) {
        auto inner = unwrapOne(t);
        if (inner == t) return t;
        t = inner;
    }
}

static bool isSimpleScalar(mlir::Type t) {
    if (t.isF32() || t.isF64()) return true;
    // Match every integer width that ``extract_vars.cpp`` knows how
    // to map to a DaCe dtype (int8/16/32/64) so the predicates agree.
    if (t.isInteger(8) || t.isInteger(16)
            || t.isInteger(32) || t.isInteger(64))
        return true;
    // Fortran ``LOGICAL(KIND=N)`` lowers to ``fir.logical<N>`` — a
    // distinct MLIR type from IntegerType.  Storage is N bytes (1, 2,
    // 4, 8); ``extract_vars.cpp`` maps each kind to the matching
    // ``int<N*8>`` dtype.  The kind-preserving mapping is required at
    // the SDFG layer because the flat companion's array stride /
    // total_size depend on element bytes; the bindings wrapper does
    // ``.TRUE.``/``.FALSE.`` ↔ ``1``/``0`` conversion at the Fortran
    // caller boundary.
    if (mlir::isa<fir::LogicalType>(t)) return true;
    return false;
}

/// Recognise an allocatable-array OR pointer-array struct member:
///   * ``real, allocatable :: w(:)``  → ``fir.box<fir.heap<fir.array<?xT>>>``
///   * ``real, pointer     :: w(:)``  → ``fir.box<fir.ptr<fir.array<?xT>>>``
///
/// Both share the same outer wrapper shape (a runtime descriptor on
/// the struct slot); only the inner indirection type differs (``heap``
/// vs ``ptr``).  Under the bridge's strict-no-aliasing assumption the
/// two are interchangeable for downstream lowering: each instance
/// holds a (data pointer + shape) descriptor, the static type of the
/// slot is the box, and the dynamic extent lives in the descriptor.
/// Allocate / freemem flow only fires on the heap variant; pointer
/// rebinds are collapsed by ``hlfir-rewrite-pointer-assigns`` (now
/// extended to handle the slice-target form for both top-level
/// pointers and pointer struct-member rebinds).
static bool isAllocatableArrayMember(mlir::Type t) {
    auto box = mlir::dyn_cast<fir::BoxType>(t);
    if (!box) return false;
    mlir::Type inner;
    if (auto heap = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
        inner = heap.getEleTy();
    else if (auto ptr = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
        inner = ptr.getEleTy();
    else
        return false;
    auto seq = mlir::dyn_cast<fir::SequenceType>(inner);
    if (!seq) return false;
    return isSimpleScalar(seq.getEleTy());
}

/// Recognise an allocatable-scalar OR pointer-scalar struct member:
///   * ``real, allocatable :: a``  → ``fir.box<fir.heap<T>>``
///   * ``real, pointer     :: a``  → ``fir.box<fir.ptr<T>>``
/// Sibling of ``isAllocatableArrayMember`` for rank-0 allocatables /
/// pointers.  These appear in nested struct hierarchies (e.g. an
/// inner record holds a scalar allocatable field); admitting them to
/// ``collectFlatLeaves`` lets ``replaceStructArgNested`` produce a
/// ``box<heap<T>>`` leaf with the appropriate FortranAttr.
static bool isAllocatableScalarMember(mlir::Type t) {
    auto box = mlir::dyn_cast<fir::BoxType>(t);
    if (!box) return false;
    mlir::Type inner;
    if (auto heap = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
        inner = heap.getEleTy();
    else if (auto ptr = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
        inner = ptr.getEleTy();
    else
        return false;
    if (mlir::isa<fir::SequenceType>(inner)) return false;
    return isSimpleScalar(inner);
}

/// Scalar or array-of-scalar (or allocatable array-of-scalar).  Used
/// both for struct members (when the enclosing struct is a scalar)
/// and for the final companion pointee type.
static bool isFlatMemberType(mlir::Type t) {
    if (isSimpleScalar(t)) return true;
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(t))
        return isSimpleScalar(seq.getEleTy());
    if (isAllocatableArrayMember(t)) return true;
    return false;
}

/// Pull the enclosed RecordType out of a declared HLFIR type and report
/// whether it is wrapped in an outer fir.array (array-of-struct case).
/// Returns null if the peeled type is not a record.
static fir::RecordType peelToRecord(mlir::Type declaredTy, bool &outerIsArray,
                                    llvm::SmallVectorImpl<int64_t> &outerShape) {
    outerIsArray = false;
    outerShape.clear();
    auto peeled = unwrapAll(declaredTy);
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(peeled)) {
        outerIsArray = true;
        for (auto d : seq.getShape()) outerShape.push_back(d);
        peeled = seq.getEleTy();
    }
    return mlir::dyn_cast<fir::RecordType>(peeled);
}

/// Compute the companion pointee for a given (outer, member) pairing.
/// Returns null if unsupported.
///
/// Outer-is-array AND member-is-array case (``s(N)%w(M1, M2, ...)``):
/// concatenate the two shape vectors into a single fir.array<N, M1, M2,
/// ...>.  Fortran derived types have a SINGLE declared shape per member
/// that applies to every instance, so per-instance offset uniformity is
/// automatic — no per-element check needed.
static mlir::Type companionPointee(bool outerIsArray,
                                   llvm::ArrayRef<int64_t> outerShape,
                                   mlir::Type memberTy) {
    bool memberIsArray = mlir::isa<fir::SequenceType>(memberTy);
    if (outerIsArray && memberIsArray) {
        auto memSeq = mlir::cast<fir::SequenceType>(memberTy);
        llvm::SmallVector<int64_t, 6> concat(outerShape.begin(), outerShape.end());
        for (auto d : memSeq.getShape())
            concat.push_back(d);
        return fir::SequenceType::get(concat, memSeq.getEleTy());
    }
    if (outerIsArray)
        return fir::SequenceType::get(outerShape, memberTy);
    return memberTy;  // scalar struct: pass the member through verbatim
}

/// Rebuild `shell`'s wrappers around a new inner type.  Used when we need
/// to mirror the original declare's result-0 wrapping (e.g. fir.box<array<...>>)
/// with the element type replaced.  ``fir.class<T>`` rebuilds as
/// ``fir.class<newT>`` to preserve the polymorphic tag (degrades
/// gracefully to ``fir.box`` only if explicit).
static mlir::Type rewrapWith(mlir::Type shell, mlir::Type newInner) {
    if (auto x = mlir::dyn_cast<fir::ClassType>(shell))
        return fir::ClassType::get(rewrapWith(x.getEleTy(), newInner));
    if (auto x = mlir::dyn_cast<fir::BoxType>(shell))
        return fir::BoxType::get(rewrapWith(x.getEleTy(), newInner));
    if (auto x = mlir::dyn_cast<fir::ReferenceType>(shell))
        return fir::ReferenceType::get(rewrapWith(x.getEleTy(), newInner));
    if (auto x = mlir::dyn_cast<fir::HeapType>(shell))
        return fir::HeapType::get(rewrapWith(x.getEleTy(), newInner));
    if (auto x = mlir::dyn_cast<fir::PointerType>(shell))
        return fir::PointerType::get(rewrapWith(x.getEleTy(), newInner));
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(shell))
        return fir::SequenceType::get(seq.getShape(), newInner);
    return newInner;
}

/// Emit a fir.shape for a static extent list, inserting arith.constant ops
/// for each extent.  Returns the shape SSA value.  Empty extents → null
/// (scalar — no shape needed).
static mlir::Value emitStaticShape(mlir::OpBuilder &b, mlir::Location loc,
                                   llvm::ArrayRef<int64_t> extents) {
    if (extents.empty()) return {};
    auto idxTy = b.getIndexType();
    llvm::SmallVector<mlir::Value, 4> dims;
    for (auto e : extents) {
        dims.push_back(b.create<mlir::arith::ConstantOp>(
            loc, idxTy, b.getIndexAttr(e)));
    }
    auto shapeTy = fir::ShapeType::get(b.getContext(), extents.size());
    return b.create<fir::ShapeOp>(loc, shapeTy, dims);
}

/// Extract the extents if `t` peels to a fir.array with all-static dims,
/// else return an empty vector.
static llvm::SmallVector<int64_t, 4> staticArrayExtents(mlir::Type t) {
    llvm::SmallVector<int64_t, 4> out;
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(t)) {
        for (auto d : seq.getShape()) {
            if (d == fir::SequenceType::getUnknownExtent()) return {};
            out.push_back(d);
        }
    }
    return out;
}

/// Build the operandSegmentSizes attribute expected on hlfir.declare.
/// hlfir.declare has four operand segments in this order: memref, shape,
/// typeparams, dummy_scope.  We only ever construct declares with a memref
/// (and optionally a shape) in this pass — the remaining two segments are
/// always zero.
static mlir::NamedAttribute declareSegments(mlir::OpBuilder &b, bool hasShape) {
    llvm::SmallVector<int32_t, 4> sizes{1, hasShape ? 1 : 0, 0, 0};
    return b.getNamedAttr("operandSegmentSizes",
                          b.getDenseI32ArrayAttr(sizes));
}

/// True if every member is flat (scalar or array-of-scalar) and we can
/// synthesise a companion pointee for every (outer, member) pair.  AoS
/// outers concatenate outer × inner extents in ``companionPointee``.
static bool allMembersFlattenable(fir::RecordType rec, bool /*outerIsArray*/) {
    for (auto &pair : rec.getTypeList()) {
        if (!isFlatMemberType(pair.second)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Nested struct flattening helpers
// ---------------------------------------------------------------------------
//
// A nested record type (``type(outer_t) :: o`` whose member ``inner`` is
// itself ``type(inner_t)``) flattens by walking ALL paths from root to a
// flat leaf and synthesising one ``hlfir.declare`` per leaf.  Naming
// follows the path: ``o_inner_x``, ``o_inner_y``, etc.  The original
// member-by-member rewrite at the top level still works because each
// nested ``hlfir.designate`` chain unwinds through a sequence of
// component selectors that we walk in ``rewriteDesignateChain``.
//
// FlatLeaf records one such leaf:
//   * ``path``    — successive component names from the outermost
//                   record down to the leaf.  Joined with ``_`` for
//                   the synthesised declare's uniq_name suffix.
//   * ``leafTy``  — the leaf's type (scalar or fir.array<scalar>).
struct FlatLeaf {
    llvm::SmallVector<std::string, 4> path;
    mlir::Type                         leafTy;
};

/// Walk a ``RecordType`` recursively and append every flat leaf to
/// ``out``.  Returns false if any leaf is non-flat (i.e. cannot be
/// reached through a chain of pure-record + flat steps); on false the
/// pass falls back to its single-level path.  Limit recursion depth
/// to guard against unexpectedly deep nesting (Fortran allows up to a
/// realistic ~10).
static constexpr int kFlattenMaxDepth = 12;

/// Recursively walk a record type and append every reachable flat
/// leaf to ``out``.  Returns false if any path bottoms out at a
/// non-flat shape (allocatable / pointer member, dynamic-extent
/// inner array, etc.); on false the caller falls back to a
/// non-nested rewrite and the un-flattened struct surfaces a
/// loud failure downstream.
///
/// Three member shapes are recognised at each level:
///   * **flat member** (scalar / static-shape array of scalar) —
///     contributes one leaf with its intrinsic shape preserved
///     (the ``outerDims`` accumulated above are prepended so
///     intermediate ``array<N x RecordType>`` levels concat into
///     the leaf's flat companion shape).
///   * **pure record** (``RecordType`` directly) — recurses with
///     no shape contribution.
///   * **array of records** (``array<N x RecordType>``) — recurses
///     into the inner record after pushing ``N`` onto
///     ``outerDims``; every leaf produced by that recursion
///     inherits ``N`` as a leading dim.  This is what enables
///     ``p_prog%pprog(i)%w(j, k)`` (where ``pprog: type(t)(10)``
///     is an array-of-struct member) to flatten to a 3D companion
///     ``p_prog_pprog_w`` of shape ``(10, 5, 5)``.
/// Recognise a pointer/allocatable-to-record member (``type(t),
/// pointer :: p`` / ``type(t), allocatable :: p`` with scalar
/// pointee).  Used only by ``collectFlatLeaves``'s cycle handling
/// — the bridge cannot navigate through such a pointer to its
/// pointee (would require concrete pointer-aliasing analysis), but
/// it can safely IGNORE the field when the user code never reads
/// through it.  Returns the pointed-to RecordType when matched,
/// null otherwise.
static fir::RecordType pointerToRecordMember(mlir::Type t) {
    auto box = mlir::dyn_cast<fir::BoxType>(t);
    if (!box) return {};
    mlir::Type inner;
    if (auto h = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
        inner = h.getEleTy();
    else if (auto p = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
        inner = p.getEleTy();
    else
        return {};
    return mlir::dyn_cast<fir::RecordType>(inner);
}

/// Recognise ``type(T), allocatable :: f(:)`` or ``type(T), pointer ::
/// f(:)`` — i.e. an alloc/pointer wrapper over an array of records.
/// Companion of ``pointerToRecordMember`` for the array-shaped case.
/// Returns the inner element ``RecordType`` when matched.
///
/// Treated as opaque by ``collectFlatLeaves`` for the same reason
/// pointer-to-record-scalar is: the bridge can't pre-allocate a flat
/// companion for "all records reachable through this descriptor"
/// without runtime alloc-count info.  Access through such a member
/// (``p_prog%pprog(<idx>)%...``) is handled by recognising the
/// inlined-callee element-alias declare that Flang emits after
/// ``hlfir-inline-all`` and flattening *that* declare instead.
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

static bool collectFlatLeaves(fir::RecordType rec,
                              llvm::SmallVectorImpl<std::string> &prefix,
                              llvm::SmallVectorImpl<int64_t> &outerDims,
                              llvm::SmallVectorImpl<FlatLeaf> &out,
                              llvm::SmallPtrSetImpl<mlir::Type> &visited,
                              int depth = 0) {
    if (depth > kFlattenMaxDepth) return false;
    // Mark this record as in-progress so a downstream pointer member
    // whose pointee re-enters the same type (mutual recursion: ``type a_t
    // { type(b_t), pointer :: b }; type b_t { type(a_t) :: a }``) is
    // recognised as a parent pointer rather than infinite-recursed
    // through.  Pointers to records that close a cycle through any
    // ancestor are treated as opaque — no leaf emitted, no failure
    // raised.  Code that actually navigates through such a pointer
    // (``s%b%a%w``) is out of scope for this admission path; the user
    // contract is that the pointer is either unused or points back to
    // the parent instance.
    visited.insert(rec);
    auto guard = llvm::make_scope_exit([&]() { visited.erase(rec); });
    for (auto &pair : rec.getTypeList()) {
        prefix.push_back(pair.first);
        // ``type(T), pointer :: f`` / ``type(T), allocatable :: f`` is
        // opaque to the leaf walker — we don't have a flat
        // representation for "all the records reachable through this
        // pointer".  Skip silently rather than fail the whole
        // flatten; downstream the only code paths that navigate
        // through such a pointer are the cycle-collapse rewrite (the
        // user-contract case ``s%b%a%w === s%w``) or genuine multi-
        // instance pointer chases (out of scope per the parent-
        // pointer contract).  Either way, the flat leaf set is what
        // matters here, and the pointer doesn't contribute one.
        if (pointerToRecordMember(pair.second)) {
            prefix.pop_back();
            continue;
        }
        // Admit allocatable/pointer scalars alongside the regular flat
        // shapes — ``replaceStructArgNested``'s BoxType leaf branch
        // already produces the right declare for either rank.
        if (isFlatMemberType(pair.second)
            || isAllocatableScalarMember(pair.second)) {
            FlatLeaf leaf;
            leaf.path.assign(prefix.begin(), prefix.end());
            // Compose the leaf's flat companion shape:
            //   outerDims (accumulated array-of-record dims walked
            //   on the way down) ++ memberDims (the leaf member's
            //   own intrinsic shape, if any).
            mlir::Type leafEle = pair.second;
            llvm::SmallVector<int64_t, 4> memberDims;
            if (auto seq = mlir::dyn_cast<fir::SequenceType>(leafEle)) {
                for (auto d : seq.getShape()) {
                    if (d == fir::SequenceType::getUnknownExtent()) {
                        prefix.pop_back();
                        return false;  // dynamic extents in the
                                       // leaf require a runtime
                                       // shape we don't synthesise
                                       // in this path.
                    }
                    memberDims.push_back(d);
                }
                leafEle = seq.getEleTy();
            }
            if (outerDims.empty() && memberDims.empty()) {
                // Pure scalar leaf — no array wrapper.
                leaf.leafTy = leafEle;
            } else {
                llvm::SmallVector<int64_t, 6> shape(outerDims.begin(),
                                                    outerDims.end());
                shape.append(memberDims.begin(), memberDims.end());
                leaf.leafTy = fir::SequenceType::get(shape, leafEle);
            }
            out.push_back(std::move(leaf));
        } else if (auto innerRec = mlir::dyn_cast<fir::RecordType>(pair.second)) {
            if (!collectFlatLeaves(innerRec, prefix, outerDims, out, visited,
                                   depth + 1)) {
                prefix.pop_back();
                return false;
            }
        } else if (auto seq = mlir::dyn_cast<fir::SequenceType>(pair.second)) {
            // Array-of-record member: recurse INTO the inner record
            // with the outer extents pushed on so each leaf inherits
            // them as leading dims.  Bail on dynamic extents — those
            // would need a runtime-shape companion the synth path
            // doesn't yet emit.
            auto innerRec = mlir::dyn_cast<fir::RecordType>(seq.getEleTy());
            if (!innerRec) {
                prefix.pop_back();
                return false;
            }
            llvm::SmallVector<int64_t, 4> theseDims;
            for (auto d : seq.getShape()) {
                if (d == fir::SequenceType::getUnknownExtent()) {
                    prefix.pop_back();
                    return false;
                }
                theseDims.push_back(d);
            }
            for (auto d : theseDims) outerDims.push_back(d);
            bool ok = collectFlatLeaves(innerRec, prefix, outerDims, out,
                                        visited, depth + 1);
            for (size_t i = 0; i < theseDims.size(); ++i) outerDims.pop_back();
            if (!ok) {
                prefix.pop_back();
                return false;
            }
        } else {
            // Member is e.g. allocatable / pointer — not flattenable
            // through this path.  Bail so the pass leaves the
            // struct untouched and the loud-failure throw in
            // extract_vars points at the right gap.
            prefix.pop_back();
            return false;
        }
        prefix.pop_back();
    }
    return true;
}

/// Top-level entry point for the flat-leaf walker.  Internal
/// callers always start with empty ``outerDims``.  Forwards to
/// the recursive form above.
static bool collectFlatLeaves(fir::RecordType rec,
                              llvm::SmallVectorImpl<std::string> &prefix,
                              llvm::SmallVectorImpl<FlatLeaf> &out,
                              int depth = 0) {
    llvm::SmallVector<int64_t, 4> outerDims;
    llvm::SmallPtrSet<mlir::Type, 4> visited;
    return collectFlatLeaves(rec, prefix, outerDims, out, visited, depth);
}

/// Entry point that threads a caller-provided ``outerDims`` (used by
/// ``splitLocal`` / the AoS-allocatable pre-flatten check to seed the
/// outer record's array extents).
static bool collectFlatLeaves(fir::RecordType rec,
                              llvm::SmallVectorImpl<std::string> &prefix,
                              llvm::SmallVectorImpl<int64_t> &outerDims,
                              llvm::SmallVectorImpl<FlatLeaf> &out,
                              int depth = 0) {
    llvm::SmallPtrSet<mlir::Type, 4> visited;
    return collectFlatLeaves(rec, prefix, outerDims, out, visited, depth);
}

/// Detect a "jagged" scalar-struct: every member is a 1-D array of the same
/// scalar element type, and at least two members have different extents.
///
/// When true, the struct is packed into a single 2-D companion of shape
/// ``[numMembers x max(extents)]`` — an ELLPACK-style padded representation
/// used when per-member flattening would produce differently-shaped siblings.
/// Scalar and non-1-D members are reported unsupported, so the caller falls
/// back to the per-member path.
static bool isJaggedScalarStruct(fir::RecordType rec, mlir::Type &eleTy,
                                 llvm::SmallVectorImpl<int64_t> &extents) {
    eleTy = nullptr;
    extents.clear();

    for (auto &pair : rec.getTypeList()) {
        auto seq = mlir::dyn_cast<fir::SequenceType>(pair.second);
        if (!seq) return false;
        auto shape = seq.getShape();
        if (shape.size() != 1) return false;
        if (shape[0] == fir::SequenceType::getUnknownExtent()) return false;
        auto se = seq.getEleTy();
        if (!isSimpleScalar(se)) return false;
        if (!eleTy) eleTy = se;
        else if (eleTy != se) return false;
        extents.push_back(shape[0]);
    }
    if (extents.size() < 2) return false;

    for (size_t i = 1; i < extents.size(); ++i)
        if (extents[i] != extents[0]) return true;
    return false;  // all uniform — per-member path handles it cleanly
}

// ---------------------------------------------------------------------------
// Fortran-name helpers — drive FlattenPlan construction in the pass
// ---------------------------------------------------------------------------

/// Extract the user-visible Fortran variable name from a Flang uniq_name.
///
/// Flang's mangled uniq_name carries the enclosing scope:
///     ``_QF<sub>E<var>``          — dummy/local in subroutine ``<sub>``
///     ``_QM<mod>F<sub>E<var>``    — in module ``<mod>``, subroutine ``<sub>``
///     ``_QF<sub>E<var>_component``  — nested cases exist but keep the ``E``
///                                    as the last separator for the outer
///                                    user name.
///
/// Grabbing everything after the *last* ``E`` gives the declared
/// Fortran name intact for the common case and degrades gracefully
/// (returns the full string) for unfamiliar mangling schemes.
static std::string demangleVarName(llvm::StringRef uniqName) {
    auto epos = uniqName.rfind('E');
    if (epos == llvm::StringRef::npos) return uniqName.str();
    return uniqName.substr(epos + 1).str();
}

/// Map a Flang intent flag to the writeback_intent string the binding
/// emitter expects (``in`` / ``out`` / ``inout`` / ``""``).  The
/// emitter uses ``inout`` and ``out`` to gate copy-out code — ``in``
/// and empty are both read-only.
static std::string extractIntent(
    std::optional<fir::FortranVariableFlagsEnum> flagsOpt) {
    if (!flagsOpt) return "";
    auto flags = *flagsOpt;
    auto has = [&](fir::FortranVariableFlagsEnum f) {
        return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(f)) != 0;
    };
    if (has(fir::FortranVariableFlagsEnum::intent_inout)) return "inout";
    if (has(fir::FortranVariableFlagsEnum::intent_out))   return "out";
    if (has(fir::FortranVariableFlagsEnum::intent_in))    return "in";
    return "";
}

/// Pretty-print a Flang element type as the Fortran scratch dtype the
/// Python ``FlattenRecipe`` carries (``float64`` / ``float32`` /
/// ``int32`` / ``int64``).  Returns an empty string for types we don't
/// map; the caller typically falls back to ``float64`` in that case.
static std::string dtypeName(mlir::Type t) {
    if (t.isF32()) return "float32";
    if (t.isF64()) return "float64";
    if (t.isInteger(32)) return "int32";
    if (t.isInteger(64)) return "int64";
    return "";
}

/// Element type of a member — unwraps fir.array to its element, or
/// returns the scalar itself.  Used to pick the recipe dtype.
static mlir::Type memberElementType(mlir::Type memTy) {
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(memTy))
        return seq.getEleTy();
    return memTy;
}

/// Return the rank of a member type (0 for scalars).
static int memberRank(mlir::Type memTy) {
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(memTy))
        return seq.getShape().size();
    return 0;
}

// ---------------------------------------------------------------------------
// Shared designate rewrite
// ---------------------------------------------------------------------------

/// Redirect a single hlfir.designate whose base is `oldBase` onto the per-
/// member companion.  Works for bare field access (no indices) and for
/// indexed access; clones the original op for the latter so indices, shape,
/// and remaining attributes survive.
/// Walk back through a chain of ``hlfir.designate`` ops to collect the
/// component names from outermost to innermost.  The chain anchor is the
/// first non-designate operand (typically the original ``hlfir.declare``
/// of the struct root).  Returns the joined "_" path on success and an
/// empty string if the chain doesn't end in pure component selectors.
/// Walk a chain of ``hlfir.designate`` ops back from the leaf up
/// to the underlying ``hlfir.declare``, collecting:
///   * ``path``                  — outer-first list of component names.
///   * ``intermediateIndices``   — outer-first list of indices that
///                                 appeared on NON-LEAF designates
///                                 (i.e. on intermediate steps of
///                                 the chain).  Empty for the
///                                 simple case where only the leaf
///                                 carries indices.
///
/// Two chain shapes are handled by separate downstream paths:
///
///   1. **Leaf-only indices** (the original case): every
///      intermediate designate is a pure ``{component}`` selector,
///      and any indices live on the leaf itself.  Caller clones
///      the leaf and swaps its memref to the flat companion —
///      preserving triplet sections, shape operands, and any
///      other leaf-side attributes.
///
///   2. **Intermediate indices** (array-of-record member): the
///      chain has a ``designate(idx)`` step between component
///      designates, e.g. ``p_prog%pprog(i)%w(j, k)``.  Caller
///      builds a fresh designate over the flat companion with
///      indices merged across all chain steps.  Triplet sections
///      on intermediate steps aren't in scope here (rare; would
///      need separate handling).
///
/// Returns the joined ``"a_b_c"`` path key on success (matching the
/// FlatLeaf naming the synth produces); empty string if the chain
/// has no component step at all, or if a triplet section appears
/// at a non-leaf level.
static std::string walkDesignateChain(
    hlfir::DesignateOp leaf,
    llvm::SmallVectorImpl<mlir::Value> &intermediateIndices) {
    llvm::SmallVector<std::string, 4> compsRev;
    llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> intermediateIdxGroupsRev;
    hlfir::DesignateOp cur = leaf;
    for (int i = 0; i < kFlattenMaxDepth && cur; ++i) {
        mlir::StringAttr compAttr;
        for (auto nm : {"component_name", "component"})
            if (auto a = cur->getAttrOfType<mlir::StringAttr>(nm)) {
                compAttr = a;
                break;
            }
        if (compAttr) compsRev.push_back(compAttr.getValue().str());
        bool isLeaf = (cur == leaf);
        if (!isLeaf) {
            // Intermediate steps must be plain (no triplets).
            // Triplet sections on intermediate levels would mean a
            // non-uniform slice through the array-of-record path
            // (e.g. ``p_prog%pprog(2:5)%w(j)``); not in scope.
            //
            // ``getIsTriplet()`` returns a nullable
            // ``DenseBoolArrayAttr`` — iterating a null attr (when
            // the designate carries no isTriplet, the common case
            // for component-only or scalar-index designates) is a
            // crash, so guard via the raw attr accessor first.
            if (auto trip = cur.getIsTripletAttr())
                for (bool t : trip.asArrayRef()) if (t) return "";
            llvm::SmallVector<mlir::Value, 4> these(
                cur.getIndices().begin(), cur.getIndices().end());
            intermediateIdxGroupsRev.push_back(std::move(these));
        }
        // Walk to parent.
        auto memref = cur.getMemref();
        cur = mlir::dyn_cast_or_null<hlfir::DesignateOp>(memref.getDefiningOp());
    }
    if (compsRev.empty()) return "";
    // Reverse to outer-first.  Components join with "_" to match
    // FlatLeaf.path's canonical form.
    std::string joined;
    for (auto it = compsRev.rbegin(); it != compsRev.rend(); ++it) {
        if (!joined.empty()) joined += "_";
        joined += *it;
    }
    for (auto it = intermediateIdxGroupsRev.rbegin();
         it != intermediateIdxGroupsRev.rend(); ++it)
        intermediateIndices.append(it->begin(), it->end());
    return joined;
}

/// Backwards-compatible wrapper used by callers that only need the
/// path (no merged indices) — keeps the original entry point shape
/// while ``walkDesignateChain`` is the canonical implementation.
static std::string designateChainPath(hlfir::DesignateOp leaf,
                                      hlfir::DesignateOp &outAnchor) {
    llvm::SmallVector<mlir::Value, 4> ignored;
    auto joined = walkDesignateChain(leaf, ignored);
    outAnchor = leaf;
    return joined;
}

/// Path-prefix accumulated while tracing an alias declare back to its
/// underlying decl: outermost component first.  Surfaces when an
/// inlined-callee dummy aliases ``decl`` through
/// ``hlfir.declare → fir.convert → fir.embox → hlfir.designate*``
/// chains.  ``rewriteDesignateChain`` prepends this prefix so a leaf
/// rooted at the alias designs into the same flat companion as a
/// leaf rooted directly at ``decl``.
struct AliasPrefix {
    llvm::SmallVector<std::string, 4> path;
    llvm::SmallVector<mlir::Value, 4> indices;
};

/// Rewrite a multi-level ``hlfir.designate`` chain ending at ``leaf``
/// (e.g. ``designate{"x"}.designate{"inner"} %o`` for ``o%inner%x``)
/// to read directly from the path-flattened declare named in
/// ``leafBase``.  ``leaf`` may carry indices (``a(i,j)``) — those are
/// preserved.  ``aliasPrefixes`` lets the rewriter prepend a buried
/// prefix when the chain bottoms out at an inlined-callee alias
/// declare whose source threads through embox/convert into a
/// designate chain rooted at ``decl`` (the type_arg2 / type_array
/// shape).  Returns true if the rewrite fired.
static bool rewriteDesignateChain(
    hlfir::DesignateOp leaf,
    const llvm::StringMap<mlir::Value> &leafBase,
    const llvm::DenseMap<mlir::Value, AliasPrefix> *aliasPrefixes = nullptr) {

    llvm::SmallVector<mlir::Value, 4> intermediateIndices;
    std::string path = walkDesignateChain(leaf, intermediateIndices);

    // Augment with any alias-prefix attached to the chain's root
    // designate's memref (the declare that the innermost ``cur``
    // selects from).
    if (aliasPrefixes) {
        hlfir::DesignateOp cur = leaf;
        for (int i = 0; i < kFlattenMaxDepth && cur; ++i) {
            auto memref = cur.getMemref();
            auto nextDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
                memref.getDefiningOp());
            if (!nextDg) {
                auto pit = aliasPrefixes->find(memref);
                if (pit != aliasPrefixes->end()) {
                    const auto &pref = pit->second;
                    std::string joined;
                    for (auto &c : pref.path) {
                        if (!joined.empty()) joined += "_";
                        joined += c;
                    }
                    if (!path.empty()) {
                        if (!joined.empty()) joined += "_";
                        joined += path;
                    }
                    path = std::move(joined);
                    llvm::SmallVector<mlir::Value, 4> merged(
                        pref.indices.begin(), pref.indices.end());
                    merged.append(intermediateIndices.begin(),
                                  intermediateIndices.end());
                    intermediateIndices = std::move(merged);
                }
                break;
            }
            cur = nextDg;
        }
    }

    if (path.empty()) return false;
    auto it = leafBase.find(path);
    if (it == leafBase.end()) return false;
    auto newBase = it->second;

    // Leaf-only path (no intermediate indices).  Preserves the
    // leaf's full shape — including triplet sections, shape
    // operand, complex_part, etc. — by cloning and rewiring just
    // the memref + clearing the component attrs.  Whole-leaf
    // access (``base{"a"}{"b"}`` with no indices) just RAUWs.
    if (intermediateIndices.empty()) {
        if (leaf.getIndices().empty()) {
            leaf.getResult().replaceAllUsesWith(newBase);
            leaf.erase();
            return true;
        }
        mlir::OpBuilder rb(leaf);
        auto *clone = rb.clone(*leaf.getOperation());
        clone->setOperand(0, newBase);
        clone->removeAttr("component");
        clone->removeAttr("component_name");
        leaf.getResult().replaceAllUsesWith(clone->getResult(0));
        leaf.erase();
        return true;
    }

    // Intermediate-indices path (array-of-record member surfaced
    // by ``collectFlatLeaves``'s extra outerDims).  Build a fresh
    // designate over the flat companion with intermediate +
    // leaf indices merged in outer-first order.  No triplets at
    // intermediate levels (walker bails on that).  Whether the
    // leaf itself has triplets is rare in this shape — a section
    // on the innermost array of a record-of-record-of-... — and
    // is also out of scope; bail to keep the contract narrow.
    if (auto leafTrip = leaf.getIsTripletAttr())
        for (bool t : leafTrip.asArrayRef()) if (t) return false;

    // Whole-component-array access surfaced through the chain
    // (``p_prog%pprog(i)%w`` with leaf having a ``shape`` operand
    // and no own indices, but result type ``ref<array<M1, M2, ...>>``).
    // The flat companion is a higher-rank array (intermediate dims
    // ++ inner dims).  Replacing the leaf with a plain N-index
    // designate where N = #intermediates only would crash the
    // verifier (rank mismatch) — instead emit a section designate
    // ``flat(idx_1, ..., 1:M_1:1, 1:M_2:1)`` so the result keeps
    // the leaf's array shape while the outer scalar indices pin
    // the record element.
    mlir::OpBuilder rb(leaf);
    auto loc = leaf.getLoc();
    if (leaf.getIndices().empty()) {
        if (auto memberSeqTy = mlir::dyn_cast<fir::SequenceType>(
                fir::unwrapRefType(leaf.getResult().getType()))) {
            auto idxTy = rb.getIndexType();
            auto c1 = rb.create<mlir::arith::ConstantOp>(
                loc, idxTy, rb.getIndexAttr(1));
            llvm::SmallVector<mlir::Value, 8> sliceIndices;
            llvm::SmallVector<bool, 4> isTriplet;
            for (auto idx : intermediateIndices) {
                sliceIndices.push_back(idx);
                isTriplet.push_back(false);
            }
            for (auto d : memberSeqTy.getShape()) {
                if (d == fir::SequenceType::getUnknownExtent())
                    return false;
                auto cN = rb.create<mlir::arith::ConstantOp>(
                    loc, idxTy, rb.getIndexAttr(d));
                sliceIndices.push_back(c1.getResult());
                sliceIndices.push_back(cN.getResult());
                sliceIndices.push_back(c1.getResult());
                isTriplet.push_back(true);
            }
            auto newOp = rb.create<hlfir::DesignateOp>(
                loc,
                /*result_type=*/leaf.getResult().getType(),
                /*memref=*/newBase,
                /*component=*/mlir::StringAttr{},
                /*component_shape=*/mlir::Value{},
                /*indices=*/mlir::ValueRange{sliceIndices},
                /*is_triplet=*/rb.getDenseBoolArrayAttr(isTriplet),
                /*substring=*/mlir::ValueRange{},
                /*complex_part=*/mlir::BoolAttr{},
                /*shape=*/leaf.getShape(),
                /*typeparams=*/mlir::ValueRange{},
                /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
            leaf.getResult().replaceAllUsesWith(newOp.getResult());
            leaf.erase();
            return true;
        }
    }

    llvm::SmallVector<mlir::Value, 6> merged(intermediateIndices.begin(),
                                              intermediateIndices.end());
    for (auto v : leaf.getIndices()) merged.push_back(v);
    auto newOp = rb.create<hlfir::DesignateOp>(
        loc,
        leaf.getResult().getType(),
        newBase,
        mlir::ValueRange{merged});
    leaf.getResult().replaceAllUsesWith(newOp.getResult());
    leaf.erase();
    return true;
}

/// Trace ``other``'s memref back through ``hlfir.declare`` /
/// ``fir.convert`` / ``fir.embox`` / ``hlfir.designate`` ops, building
/// the ``(path, indices)`` prefix that an alias root buries.  When an
/// inlined-callee dummy aliases ``decl`` via
/// ``convert(embox(designate{"w"}(designate{"pprog"}(i))))``, a leaf
/// rooted at the alias declare needs the ``("pprog", "w") + [i]``
/// prefix prepended so the rewrite designs into the right flat
/// companion.  Triplet sections on intermediate steps are out of scope.
static std::optional<AliasPrefix>
traceAliasPrefixToDecl(hlfir::DeclareOp other, hlfir::DeclareOp decl) {
    llvm::SmallVector<std::string, 4> pathRev;
    llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> indexGroupsRev;
    mlir::Value mr = other.getMemref();
    for (int i = 0; i < kFlattenMaxDepth && mr; ++i) {
        auto *d = mr.getDefiningOp();
        if (!d) return std::nullopt;
        if (auto outer = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
            if (outer == decl) {
                AliasPrefix info;
                for (auto it = pathRev.rbegin(); it != pathRev.rend(); ++it)
                    info.path.push_back(*it);
                for (auto it = indexGroupsRev.rbegin();
                     it != indexGroupsRev.rend(); ++it)
                    info.indices.append(it->begin(), it->end());
                return info;
            }
            // Intermediate declare (a previously-inlined alias).
            // Continue walking from its source — declares act as
            // identity wrappers around their memref operand for the
            // purposes of alias tracking.
            mr = outer.getMemref();
            continue;
        }
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
            mr = cv.getValue();
            continue;
        }
        if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
            mr = eb.getMemref();
            continue;
        }
        // Inlined-callee alias declares: the alias's memref is a
        // ``fir.load`` of the parent declare's address (possibly
        // through one or more ``fir.rebox`` reshapes for CLASS<heap<T>>
        // → CLASS<T> peels in OOP code, or POINTER box reshapes).
        // Walking through both is the minimum to recognise these as
        // aliases of the parent.
        if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
            mr = ld.getMemref();
            continue;
        }
        if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) {
            mr = rb.getBox();
            continue;
        }
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
            mlir::StringAttr compAttr;
            for (auto nm : {"component_name", "component"})
                if (auto a = dg->getAttrOfType<mlir::StringAttr>(nm)) {
                    compAttr = a;
                    break;
                }
            if (compAttr) pathRev.push_back(compAttr.getValue().str());
            if (auto trip = dg.getIsTripletAttr())
                for (bool t : trip.asArrayRef())
                    if (t) return std::nullopt;
            llvm::SmallVector<mlir::Value, 4> these(
                dg.getIndices().begin(), dg.getIndices().end());
            indexGroupsRev.push_back(std::move(these));
            mr = dg.getMemref();
            continue;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

/// Walk every ``hlfir.designate`` chain rooted at ``decl`` (or any
/// inlined-callee alias of it) and rewrite each leaf to the matching
/// flat companion in ``leafBase``.  Shared between the local-declare
/// path (``splitLocal``) and the dummy-arg path
/// (``replaceStructArgNested``) — both produce the same ``leafBase``
/// shape and need the same rewrite logic.
static void rewriteChainsRootedAt(hlfir::DeclareOp decl,
                                  const llvm::StringMap<mlir::Value> &leafBase) {
    auto func = decl->getParentOfType<mlir::func::FuncOp>();
    if (!func) return;

    // Discover declares that alias ``decl`` (inlined-callee dummies
    // whose memref chain leads back to it).  Each gets its buried
    // path + scalar prefix recorded for the chain rewriter.
    llvm::DenseSet<mlir::Value> equivalentRoots;
    llvm::DenseMap<mlir::Value, AliasPrefix> aliasPrefixes;
    equivalentRoots.insert(decl.getResult(0));
    equivalentRoots.insert(decl.getResult(1));
    func.walk([&](hlfir::DeclareOp other) {
        if (other == decl) return;
        if (auto info = traceAliasPrefixToDecl(other, decl)) {
            equivalentRoots.insert(other.getResult(0));
            equivalentRoots.insert(other.getResult(1));
            if (!info->path.empty()) {
                aliasPrefixes[other.getResult(0)] = *info;
                aliasPrefixes[other.getResult(1)] = *info;
            }
        }
    });

    // Find each chain's leaf — a designate whose users are NOT
    // themselves designates (otherwise we'd rewrite a parent and
    // lose the inner part of the chain) — then verify the chain
    // bottoms out at one of the equivalent roots.
    llvm::SmallVector<hlfir::DesignateOp, 16> chainLeaves;
    func.walk([&](hlfir::DesignateOp dg) {
        bool hasDesignateUser = false;
        for (auto *u : dg.getResult().getUsers())
            if (mlir::isa<hlfir::DesignateOp>(u)) {
                hasDesignateUser = true;
                break;
            }
        if (hasDesignateUser) return;
        // Walk the memref chain back to an equivalent root.  Peel
        // through ``hlfir.designate`` (intermediate component / index
        // selects), ``fir.load`` (loaded box from an allocatable /
        // pointer declare slot), ``fir.rebox`` (class<heap<T>> →
        // class<T> and similar peels), and ``fir.convert``.  The
        // load + rebox peels are what catch direct-access reads on
        // a CLASS-allocatable in main scope:
        //   ``%load = fir.load %decl#0`` then
        //   ``%dg = hlfir.designate %load{"<field>"}``.
        // Without those peels, %dg's memref (the load result) is
        // not an equivalent root and the chain stays un-rewritten.
        mlir::Value v = dg.getMemref();
        for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
            if (equivalentRoots.contains(v)) {
                chainLeaves.push_back(dg);
                return;
            }
            auto *def = v.getDefiningOp();
            if (!def) break;
            if (auto dg2 = mlir::dyn_cast<hlfir::DesignateOp>(def)) { v = dg2.getMemref(); continue; }
            if (auto ld = mlir::dyn_cast<fir::LoadOp>(def))         { v = ld.getMemref(); continue; }
            if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def))        { v = rb.getBox(); continue; }
            if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def))      { v = cv.getValue(); continue; }
            break;
        }
    });
    for (auto dg : chainLeaves)
        rewriteDesignateChain(dg, leafBase, &aliasPrefixes);
}

static void rewriteDesignate(
    hlfir::DesignateOp dg,
    const llvm::StringMap<mlir::Value> &memberBase,
    const llvm::StringSet<> &concatMembers = {}) {

    // The hlfir.designate op prints the component as ``{"name"}`` but stores
    // it under the attribute key ``component_name`` — depending on the HLFIR
    // tablegen spelling.  Tolerate either key so we don't silently no-op.
    mlir::StringAttr compAttr;
    for (auto nm : {"component_name", "component"}) {
        if (auto a = dg->getAttrOfType<mlir::StringAttr>(nm)) {
            compAttr = a;
            break;
        }
    }
    if (!compAttr) return;

    auto it = memberBase.find(compAttr.getValue());
    if (it == memberBase.end()) return;
    auto newBase = it->second;

    // Concat case (``s(N)%w(M, ...)``): the parent op is an indexed
    // designate without component (the per-element access on the outer
    // array-of-struct).  Merge the parent's outer indices with this
    // designate's member indices so the new designate is a flat
    // multi-dim access on the concatenated companion.
    bool isConcat = concatMembers.count(compAttr.getValue());
    if (isConcat) {
        auto parentMemref = dg.getMemref();
        auto parentDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
            parentMemref.getDefiningOp());
        if (parentDg) {
            // Verify parent is a pure indexed access (no component).
            bool parentHasComponent = false;
            for (auto nm : {"component_name", "component"})
                if (parentDg->getAttrOfType<mlir::StringAttr>(nm)) {
                    parentHasComponent = true;
                    break;
                }
            if (!parentHasComponent && !parentDg.getIndices().empty()) {
                mlir::OpBuilder rb(dg);
                if (!dg.getIndices().empty()) {
                    // Element access: ``A(i)%w(j, k)`` → flat
                    // designate ``A_w(i, j, k)``.
                    llvm::SmallVector<mlir::Value, 8> mergedIndices;
                    for (auto idx : parentDg.getIndices()) mergedIndices.push_back(idx);
                    for (auto idx : dg.getIndices())       mergedIndices.push_back(idx);
                    auto newOp = rb.create<hlfir::DesignateOp>(
                        dg.getLoc(),
                        dg.getResult().getType(),
                        newBase,
                        mlir::ValueRange{mergedIndices});
                    dg.getResult().replaceAllUsesWith(newOp.getResult());
                    dg.erase();
                    if (parentDg.getResult().use_empty()) parentDg.erase();
                    return;
                }
                // Whole-component access: ``A(i)%w`` → flat section
                // ``A_w(i, 1:M:1, 1:M:1, ...)`` — scalar outer index,
                // triplet over every inner dim.  The result type
                // stays the original member type (``ref<array<M, ...>>``).
                auto memberSeqTy = mlir::dyn_cast<fir::SequenceType>(
                    fir::unwrapRefType(dg.getResult().getType()));
                if (!memberSeqTy) {
                    // Whole scalar component (rare) — same as
                    // empty-indices old behaviour: replace use.
                    dg.getResult().replaceAllUsesWith(newBase);
                    dg.erase();
                    if (parentDg.getResult().use_empty()) parentDg.erase();
                    return;
                }
                auto loc = dg.getLoc();
                auto idxTy = rb.getIndexType();
                auto c1 = rb.create<mlir::arith::ConstantOp>(
                    loc, idxTy, rb.getIndexAttr(1));
                llvm::SmallVector<mlir::Value, 8> sliceIndices;
                llvm::SmallVector<bool, 4> isTriplet;
                for (auto idx : parentDg.getIndices()) {
                    sliceIndices.push_back(idx);
                    isTriplet.push_back(false);
                }
                for (auto d : memberSeqTy.getShape()) {
                    if (d == fir::SequenceType::getUnknownExtent()) {
                        // Cannot construct a static-bound triplet —
                        // bail to the safe fallback.
                        return;
                    }
                    auto cN = rb.create<mlir::arith::ConstantOp>(
                        loc, idxTy, rb.getIndexAttr(d));
                    sliceIndices.push_back(c1.getResult());
                    sliceIndices.push_back(cN.getResult());
                    sliceIndices.push_back(c1.getResult());
                    isTriplet.push_back(true);
                }
                // Build via the long-form ``hlfir.designate`` builder
                // so we can pass ``is_triplet`` directly.
                auto newOp = rb.create<hlfir::DesignateOp>(
                    loc,
                    /*result_type=*/dg.getResult().getType(),
                    /*memref=*/newBase,
                    /*component=*/mlir::StringAttr{},
                    /*component_shape=*/mlir::Value{},
                    /*indices=*/mlir::ValueRange{sliceIndices},
                    /*is_triplet=*/rb.getDenseBoolArrayAttr(isTriplet),
                    /*substring=*/mlir::ValueRange{},
                    /*complex_part=*/mlir::BoolAttr{},
                    /*shape=*/dg.getShape(),
                    /*typeparams=*/mlir::ValueRange{},
                    /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
                dg.getResult().replaceAllUsesWith(newOp.getResult());
                dg.erase();
                if (parentDg.getResult().use_empty()) parentDg.erase();
                return;
            }
        }
        // Parent isn't an indexed designate — fall through to the
        // single-rewrite path (probably a whole-array reference).
    }

    if (dg.getIndices().empty()) {
        dg.getResult().replaceAllUsesWith(newBase);
        dg.erase();
        return;
    }

    mlir::OpBuilder rb(dg);
    auto *clone = rb.clone(*dg.getOperation());
    clone->setOperand(0, newBase);
    clone->removeAttr("component");
    clone->removeAttr("component_name");
    dg.getResult().replaceAllUsesWith(clone->getResult(0));
    dg.erase();
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

struct FlattenStructsPass
    : public mlir::PassWrapper<FlattenStructsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenStructsPass)

    llvm::StringRef getArgument() const final {
        return "hlfir-flatten-structs";
    }
    llvm::StringRef getDescription() const final {
        return "Flatten derived types with flat members into per-member "
               "companions (AoS → SoA), rewriting struct-typed dummy "
               "arguments, renaming the function, and splitting local "
               "allocations.";
    }

    /// Collected FlattenEntry dicts — stamped on the module at the end
    /// of ``runOnOperation`` as the ``hlfir.flatten_plan`` attribute.
    llvm::SmallVector<mlir::Attribute, 4> planEntries;

    void runOnOperation() override {
        planEntries.clear();
        getOperation().walk(
            [this](mlir::func::FuncOp f) { flattenFunc(f); });
        if (planEntries.empty()) return;

        // Stamp the plan as ``hlfir.flatten_plan = {entries = [...]}``.
        // The binding emitter / bridge later reads this attribute back to
        // reconstruct the Python FlattenPlan object.
        auto *ctx = getOperation().getContext();
        mlir::Builder b(ctx);
        auto entries = b.getArrayAttr(planEntries);
        auto plan = b.getDictionaryAttr({
            b.getNamedAttr("entries", entries)
        });
        getOperation()->setAttr("hlfir.flatten_plan", plan);
    }

    /// Append one FlattenEntry dict to ``planEntries`` describing the
    /// just-performed struct-dummy split.  Covers the *per-member* path
    /// (``replaceStructArg``); the jagged-ELLPACK path is omitted from
    /// the plan — callers of that path fall back to the looped copy-in
    /// emission without plan metadata.
    ///
    /// ``excludeMembers`` lists member names already covered by a
    /// separate ``aos_alloc=True`` entry (see ``recordAosAllocEntry``);
    /// they're skipped here so the plan has exactly one recipe per flat
    /// companion.  When every member is excluded the function emits
    /// nothing.
    void recordStructArgEntry(hlfir::DeclareOp argDecl, fir::RecordType rec,
                              llvm::StringRef intentStr,
                              bool outerIsArray = false,
                              llvm::ArrayRef<int64_t> outerShape = {},
                              const llvm::StringSet<> &excludeMembers = {}) {
        auto *ctx = argDecl.getContext();
        mlir::Builder b(ctx);
        auto mkStr = [&](llvm::StringRef s) -> mlir::Attribute {
            return b.getStringAttr(s);
        };

        std::string outerName = demangleVarName(argDecl.getUniqName());
        // Outer type: dump the declared type as MLIR text — the Python
        // side uses it only for commentary, so round-tripping the MLIR
        // form is sufficient.
        std::string outerType;
        {
            llvm::raw_string_ostream os(outerType);
            argDecl.getResult(0).getType().print(os);
        }

        llvm::SmallVector<mlir::Attribute, 4> flatNames;
        llvm::SmallVector<mlir::Attribute, 4> readExprs;
        // All members of one recipe share a dtype in the current model.
        // Record the element dtype of the first flat member (they match
        // by construction — the per-member path rejects ragged member
        // dtypes upstream in ``allMembersFlattenable``).
        std::string scratchDtype = "float64";
        int64_t maxRank = 0;

        // For AoS dummy args the outer index dim(s) prepend the
        // member's index dims.  So for ``A(i)%w(j, k)``, the recipe's
        // total rank is outer_rank + member_rank, and the read expr
        // is ``A($i1)%w($i2, $i3)``.
        unsigned outerRank = outerIsArray ? (unsigned)outerShape.size() : 0u;

        // Track the first non-excluded member name so the shape-expr
        // sampling below uses one whose flat companion actually carries
        // the member-dim extents.
        std::string firstMember;

        for (auto &pair : rec.getTypeList()) {
            llvm::StringRef memName = pair.first;
            if (excludeMembers.count(memName)) continue;
            mlir::Type memTy = pair.second;
            int memRank = memberRank(memTy);
            int totalRank = (int)outerRank + memRank;
            if (totalRank > maxRank) maxRank = totalRank;

            std::string flat = (outerName + "_" + memName).str();
            flatNames.push_back(mkStr(flat));
            if (firstMember.empty()) firstMember = memName.str();

            // read_expr: ``<outer>($i1, ..., $iOR)%<member>($iOR+1, ..., $iTotal)``
            // Scalar outer + scalar member: just ``<outer>%<member>``.
            std::string read = outerName.c_str();
            if (outerRank > 0) {
                read += "(";
                for (unsigned i = 1; i <= outerRank; ++i) {
                    if (i > 1) read += ", ";
                    read += "$i" + std::to_string(i);
                }
                read += ")";
            }
            read += "%";
            read += memName.str();
            if (memRank > 0) {
                read += "(";
                for (int i = 1; i <= memRank; ++i) {
                    if (i > 1) read += ", ";
                    read += "$i" + std::to_string((int)outerRank + i);
                }
                read += ")";
            }
            readExprs.push_back(mkStr(read));

            if (std::string dt = dtypeName(memberElementType(memTy)); !dt.empty())
                scratchDtype = dt;
        }

        // Nothing to record when every member was excluded — the
        // companion entries already cover them.
        if (flatNames.empty()) return;

        // Shape exprs for the recipe.  For AoS dummy args the leading
        // ``outerRank`` dims come from the outer struct array itself
        // (``size(outer, dim=i)``); the remaining dims come from
        // ``size(outer(1)%<first_member>, dim=j)``.  For scalar-outer
        // structs all dims are member dims.
        llvm::SmallVector<mlir::Attribute, 4> shapeExprs;
        if (maxRank > 0 && !firstMember.empty()) {
            for (unsigned i = 1; i <= outerRank; ++i) {
                std::string s = "size(" + outerName
                                + ", dim=" + std::to_string((int)i) + ")";
                shapeExprs.push_back(mkStr(s));
            }
            int memDimsToEmit = maxRank - (int)outerRank;
            // Sample one instance for member-dim sizes.  Per Fortran
            // type semantics, every instance has the same member shape.
            std::string sampleOuter = outerName.c_str();
            if (outerRank > 0) {
                sampleOuter += "(";
                for (unsigned i = 0; i < outerRank; ++i) {
                    if (i) sampleOuter += ", ";
                    sampleOuter += "1";
                }
                sampleOuter += ")";
            }
            for (int i = 1; i <= memDimsToEmit; ++i) {
                std::string s = ("size(" + sampleOuter + "%" + firstMember
                                 + ", dim=" + std::to_string(i) + ")");
                shapeExprs.push_back(mkStr(s));
            }
        }

        auto recipe = b.getDictionaryAttr({
            b.getNamedAttr("flat_names",    b.getArrayAttr(flatNames)),
            b.getNamedAttr("read_exprs",    b.getArrayAttr(readExprs)),
            b.getNamedAttr("write_expr",    mkStr("")),
            b.getNamedAttr("rank",          b.getI64IntegerAttr(maxRank)),
            b.getNamedAttr("shape_exprs",   b.getArrayAttr(shapeExprs)),
            b.getNamedAttr("aliasable",     b.getBoolAttr(true)),
            b.getNamedAttr("scratch_dtype", mkStr(scratchDtype)),
            b.getNamedAttr("aos_alloc",     b.getBoolAttr(false)),
            b.getNamedAttr("cap_symbol",    mkStr("")),
        });

        auto entry = b.getDictionaryAttr({
            b.getNamedAttr("outer_expr",       mkStr(outerName)),
            b.getNamedAttr("outer_type",       mkStr(outerType)),
            b.getNamedAttr("writeback_intent", mkStr(intentStr)),
            b.getNamedAttr("recipe",           recipe),
        });
        planEntries.push_back(entry);
    }

    /// Phase 5c-B (true SDFG-boundary): emit one FlattenEntry per
    /// AoS+allocatable member.  The bindings layer pads to max
    /// per-instance size, populates ``cap_<base>_<member>`` from the
    /// pack-in loop, and ships a 2D buffer ``<base>_<member>(N, cap)``
    /// to the SDFG.  The bridge declares the data block-arg with
    /// type ``ref<array<N x ?xT>>`` so the inner extent surfaces as a
    /// runtime symbol (``cap_<base>_<member>``) on the SDFG signature.
    void recordAosAllocEntry(hlfir::DeclareOp argDecl, fir::RecordType rec,
                             llvm::StringRef memName,
                             llvm::StringRef intentStr,
                             llvm::ArrayRef<int64_t> outerShape) {
        auto *ctx = argDecl.getContext();
        mlir::Builder b(ctx);
        auto mkStr = [&](llvm::StringRef s) -> mlir::Attribute {
            return b.getStringAttr(s);
        };

        std::string outerName = demangleVarName(argDecl.getUniqName());
        std::string outerType;
        {
            llvm::raw_string_ostream os(outerType);
            argDecl.getResult(0).getType().print(os);
        }

        // Locate the member type so we can record its dtype.
        mlir::Type memTy;
        for (auto &pair : rec.getTypeList())
            if (pair.first == memName) { memTy = pair.second; break; }

        std::string scratchDtype = "float64";
        if (memTy)
            if (std::string dt = dtypeName(memberElementType(memTy)); !dt.empty())
                scratchDtype = dt;

        std::string flatName = outerName + "_" + memName.str();
        std::string capName  = "cap_" + flatName;
        unsigned outerRank = (unsigned)outerShape.size();

        // read_expr: ``<outer>($i1, ..., $iOR)%<member>($i_OR+1)``.
        // We always treat the allocatable member as 1-D for now (the
        // inner extent is the cap symbol — runtime-determined).
        std::string read = outerName;
        read += "(";
        for (unsigned i = 1; i <= outerRank; ++i) {
            if (i > 1) read += ", ";
            read += "$i" + std::to_string(i);
        }
        read += ")%";
        read += memName.str();
        read += "($i" + std::to_string((int)outerRank + 1) + ")";

        llvm::SmallVector<mlir::Attribute, 1> flatNames{mkStr(flatName)};
        llvm::SmallVector<mlir::Attribute, 1> readExprs{mkStr(read)};

        // shape_exprs: ``size(<outer>, dim=i)`` for each outer dim,
        // then the cap symbol for the inner.  The bindings layer's
        // ``_build_symbol_assigns`` skips the cap symbol because the
        // pack-in code computes it directly.
        llvm::SmallVector<mlir::Attribute, 2> shapeExprs;
        for (unsigned i = 1; i <= outerRank; ++i) {
            std::string s = "size(" + outerName
                            + ", dim=" + std::to_string((int)i) + ")";
            shapeExprs.push_back(mkStr(s));
        }
        shapeExprs.push_back(mkStr(capName));

        int64_t totalRank = (int64_t)outerRank + 1;

        auto recipe = b.getDictionaryAttr({
            b.getNamedAttr("flat_names",    b.getArrayAttr(flatNames)),
            b.getNamedAttr("read_exprs",    b.getArrayAttr(readExprs)),
            b.getNamedAttr("write_expr",    mkStr("")),
            b.getNamedAttr("rank",          b.getI64IntegerAttr(totalRank)),
            b.getNamedAttr("shape_exprs",   b.getArrayAttr(shapeExprs)),
            b.getNamedAttr("aliasable",     b.getBoolAttr(false)),
            b.getNamedAttr("scratch_dtype", mkStr(scratchDtype)),
            b.getNamedAttr("aos_alloc",     b.getBoolAttr(true)),
            b.getNamedAttr("cap_symbol",    mkStr(capName)),
        });

        auto entry = b.getDictionaryAttr({
            b.getNamedAttr("outer_expr",       mkStr(outerName)),
            b.getNamedAttr("outer_type",       mkStr(outerType)),
            b.getNamedAttr("writeback_intent", mkStr(intentStr)),
            b.getNamedAttr("recipe",           recipe),
        });
        planEntries.push_back(entry);
    }

    // -------------------------------------------------------------------
    // Function-level orchestration
    // -------------------------------------------------------------------

    void flattenFunc(mlir::func::FuncOp func) {
        if (func.isExternal()) return;
        // Skip private functions.  The bridge always builds an SDFG for
        // the single public entry; callees have been inlined into it.
        // Private siblings (kept alive only by a dispatch_table after
        // ``fir-polymorphic-op`` resolved every call site) would
        // otherwise pollute the module-level flatten_plan with phantom
        // CLASS dummies whose flat names look like top-level program
        // args at extract time.
        if (func.isPrivate()) return;

        // Step 0: decompose struct-valued ``hlfir.assign`` ops into
        // per-leaf assigns BEFORE the per-member declare rewrite runs.
        // ``val%var = indices`` (where both sides are entire struct
        // values) becomes one ``hlfir.assign`` per leaf of the struct
        // type; the existing designate-rewrite path then folds each
        // leaf assign into a flat ``val_var_<leaf> = indices_<leaf>``.
        decomposeStructAssigns(func);

        // Step 1: collect struct-typed dummy arguments, rewrite them in
        // one pass over the original index list so mutations (insertArgument /
        // eraseArgument) don't invalidate later iterations.
        bool rewroteArgs = planAndReplaceStructArgs(func);

        if (rewroteArgs) {
            auto &block = func.front();
            auto newInputs = llvm::to_vector(block.getArgumentTypes());
            func.setType(mlir::FunctionType::get(
                func.getContext(), newInputs,
                func.getFunctionType().getResults()));
            mlir::SymbolTable::setSymbolName(
                func, (func.getName() + "_soa").str());
        }

        // Step 2: local allocations of struct types.
        llvm::SmallVector<hlfir::DeclareOp, 8> work;
        func.walk([&](hlfir::DeclareOp d) {
            if (isLocallyFlattenable(d)) work.push_back(d);
        });
        for (auto d : work) splitLocal(d);
    }

    /// Decompose every struct-valued ``hlfir.assign`` in ``func`` into
    /// per-leaf assigns.  Source pattern (e.g. Fortran ``val%var =
    /// indices`` where both sides are whole struct values):
    ///
    ///   hlfir.assign %indices_struct to %val_var_struct : type<T>
    ///
    /// This pass walks the leaf set of ``T`` (via ``collectFlatLeaves``)
    /// and emits one ``hlfir.designate``-and-``hlfir.assign`` chain per
    /// leaf, copying the matching path from src to dst:
    ///
    ///   %src_leaf = hlfir.designate %indices_struct {"path0"}{"path1"}
    ///   %dst_leaf = hlfir.designate %val_var_struct  {"path0"}{"path1"}
    ///   hlfir.assign %src_leaf to %dst_leaf : <leaf_ty>
    ///
    /// The downstream per-member designate rewrite (``rewriteDesignate``
    /// / ``rewriteDesignateChain``) then folds each leaf chain into the
    /// flat-name form ``val_var_<path0>_<path1> = indices_<path0>_<path1>``.
    /// Array leaves stay whole-array assigns; scalar leaves stay scalar
    /// assigns.
    ///
    /// Out of scope: array-of-struct copies (whole-AoS-to-AoS).  Those
    /// would need to wrap each per-leaf assign in an outer-dim DO loop
    /// — separate work.
    void decomposeStructAssigns(mlir::func::FuncOp func) {
        llvm::SmallVector<hlfir::AssignOp, 16> targets;
        func.walk([&](hlfir::AssignOp op) {
            auto src = op.getRhs();
            auto dst = op.getLhs();
            bool srcIsRec = mlir::isa<fir::RecordType>(unwrapAll(src.getType()));
            bool dstIsRec = mlir::isa<fir::RecordType>(unwrapAll(dst.getType()));
            if (srcIsRec || dstIsRec) targets.push_back(op);
        });
        for (auto op : targets) decomposeStructAssign(op);
    }

    void decomposeStructAssign(hlfir::AssignOp op) {
        auto src = op.getRhs();
        auto dst = op.getLhs();

        bool outerIsArray = false;
        llvm::SmallVector<int64_t, 4> outerShape;
        auto rec = peelToRecord(dst.getType(), outerIsArray, outerShape);
        if (!rec) {
            // Try src side instead.
            rec = peelToRecord(src.getType(), outerIsArray, outerShape);
            if (!rec) return;
        }
        // AoS → AoS struct copy is out of scope (would need an outer
        // index loop wrapping each leaf assign).  Leave the assign
        // alone; downstream gates flag it.
        if (outerIsArray) return;

        llvm::SmallVector<std::string, 4> prefix;
        llvm::SmallVector<FlatLeaf, 8> leaves;
        if (!collectFlatLeaves(rec, prefix, leaves)) return;

        mlir::OpBuilder b(op);
        auto loc = op.getLoc();

        // Build a designate chain over ``base`` following the path
        // components in ``leaf.path``.  Resolves the per-step result
        // type by looking up each component in the running record
        // type's member list.
        auto buildLeafDesignate = [&](mlir::Value base,
                                      const FlatLeaf &leaf) -> mlir::Value {
            mlir::Value cur = base;
            for (auto &component : leaf.path) {
                auto curRec = mlir::dyn_cast<fir::RecordType>(unwrapAll(cur.getType()));
                if (!curRec) return {};
                mlir::Type fieldTy;
                for (auto &p : curRec.getTypeList()) {
                    if (p.first == component) { fieldTy = p.second; break; }
                }
                if (!fieldTy) return {};
                auto refFieldTy = fir::ReferenceType::get(fieldTy);
                auto componentAttr = mlir::StringAttr::get(b.getContext(), component);
                // Array-valued field needs a fir.shape operand for the
                // hlfir.designate verifier ("shape must be provided if
                // and only if the result is an array that is not a box
                // address").  Static extents only — dynamic-extent
                // record members aren't reachable through
                // ``collectFlatLeaves`` anyway.
                mlir::Value fieldShape;
                if (auto seq = mlir::dyn_cast<fir::SequenceType>(fieldTy)) {
                    auto exts = staticArrayExtents(seq);
                    if (exts.empty()) return {};
                    fieldShape = emitStaticShape(b, loc, exts);
                }
                auto newOp = b.create<hlfir::DesignateOp>(
                    loc,
                    /*resultType0=*/refFieldTy,
                    /*memref=*/cur,
                    /*component=*/componentAttr,
                    /*component_shape=*/fieldShape,
                    /*indices=*/mlir::ValueRange{},
                    /*is_triplet=*/mlir::DenseBoolArrayAttr{},
                    /*substring=*/mlir::ValueRange{},
                    /*complex_part=*/mlir::BoolAttr{},
                    /*shape=*/fieldShape,
                    /*typeparams=*/mlir::ValueRange{},
                    /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
                cur = newOp.getResult();
            }
            return cur;
        };

        for (auto &leaf : leaves) {
            mlir::Value lhsLeaf = buildLeafDesignate(dst, leaf);
            mlir::Value rhsLeaf = buildLeafDesignate(src, leaf);
            if (!lhsLeaf || !rhsLeaf) {
                // One of the chains failed to resolve a component;
                // bail out for this assign — leave it intact and let
                // downstream gates flag it loudly.
                return;
            }
            // For scalar leaves, load the source ref first so
            // ``hlfir.assign`` carries a value RHS (matching the
            // standard scalar-assign shape that downstream extract_ast
            // recognises).  Array leaves stay as ``ref<array>``-to-
            // ``ref<array>`` whole-array copy.
            mlir::Value rhsValue = rhsLeaf;
            if (isSimpleScalar(leaf.leafTy)) {
                rhsValue = b.create<fir::LoadOp>(loc, rhsLeaf).getResult();
            }
            b.create<hlfir::AssignOp>(loc, rhsValue, lhsLeaf);
        }
        op.erase();
    }

    /// Returns true if any struct-typed dummy argument was rewritten.
    bool planAndReplaceStructArgs(mlir::func::FuncOp func) {
        auto &block = func.front();

        struct Plan {
            hlfir::DeclareOp argDecl;
            fir::RecordType  rec;
            bool             jagged = false;
            mlir::Type       jaggedEleTy;
            llvm::SmallVector<int64_t, 4> jaggedExtents;
            bool             outerIsArray = false;
            llvm::SmallVector<int64_t, 4> outerShape;
            // Members that must take the Phase 5c-B AoS+allocatable
            // path (cap+data block-arg pair, runtime inner extent).
            llvm::SmallVector<std::string, 2> aosAllocMembers;
            // Nested branch: when ``rec`` has any member that's itself
            // a record, ``allMembersFlattenable`` returns false and the
            // single-level flat path bails.  Instead, walk every leaf
            // path and replace the single struct dummy with one block
            // arg per leaf.  Static-shape leaves only at first cut.
            bool             nested = false;
            llvm::SmallVector<FlatLeaf, 8> leaves;
        };

        // Keep plans sorted by ORIGINAL argument index.
        llvm::SmallVector<std::pair<unsigned, Plan>, 4> plans;
        for (unsigned i = 0, n = block.getNumArguments(); i < n; ++i) {
            auto arg = block.getArgument(i);
            hlfir::DeclareOp argDecl;
            for (auto *u : arg.getUsers())
                if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
                    argDecl = d;
                    break;
                }
            if (!argDecl) continue;

            bool outerIsArray = false;
            llvm::SmallVector<int64_t, 4> outerShape;
            auto rec = peelToRecord(argDecl.getResult(0).getType(),
                                    outerIsArray, outerShape);
            if (!rec) continue;
            // AoS dummy args: the local rewrite already handles the
            // concat shape; the dummy-arg path needs the per-member
            // block-arg insertion + recipe entry to match.  Static
            // outer extent only — dynamic-extent AoS dummies require
            // a fresh symbol per padded dim (out of scope).
            if (outerIsArray) {
                for (auto d : outerShape)
                    if (d == fir::SequenceType::getUnknownExtent()) {
                        outerIsArray = false;  // fallthrough = bail
                        break;
                    }
                if (!outerIsArray) continue;
            }

            Plan p;
            p.argDecl = argDecl;
            p.rec     = rec;
            p.outerIsArray = outerIsArray;
            p.outerShape.assign(outerShape.begin(), outerShape.end());
            if (!outerIsArray && isJaggedScalarStruct(rec, p.jaggedEleTy, p.jaggedExtents))
                p.jagged = true;
            else if (!allMembersFlattenable(rec, outerIsArray)) {
                // Nested branch: the struct has at least one record-
                // typed member.  Walk every leaf path; if every leaf is
                // a flat type with static extents we can replace the
                // single struct dummy with one block arg per leaf.
                // Outer-array nested (``type(t)::s(N)`` where ``t`` is
                // nested) is left for a follow-up — the dummy-arg block-
                // arg shape would need extra outer-dim handling.
                if (outerIsArray) continue;
                llvm::SmallVector<std::string, 4> prefix;
                llvm::SmallVector<FlatLeaf, 8> leaves;
                if (!collectFlatLeaves(rec, prefix, leaves)) continue;
                p.nested = true;
                p.leaves = std::move(leaves);
            }
            // Phase 5b: dummy struct args with allocatable members
            // flatten the same way as local instances — each
            // allocatable member becomes a flat top-level allocatable
            // companion (``<base>_<member>``) and the bindings layer
            // marshals it across the call boundary.
            //
            // Phase 5c-B (true SDFG-boundary): AoS + allocatable
            // members get the padding-to-max contract.  Each such
            // member becomes a 2D buffer ``A_<member>(N, cap)`` plus
            // a runtime cap symbol; the bindings layer computes the
            // cap by max-ing per-instance ``size()`` values, packs
            // each allocated row's live region into the buffer, and
            // unpacks back on intent(out)/(inout).
            if (outerIsArray) {
                for (auto &pair : rec.getTypeList())
                    if (isAllocatableArrayMember(pair.second))
                        p.aosAllocMembers.push_back(pair.first);
            }
            plans.push_back({i, p});
        }

        if (plans.empty()) return false;

        // Walk plans in reverse so lower indices aren't invalidated by
        // higher-index erases.  Each replace either mutates the argument
        // list in place (insert-then-erase) or bails out without changes.
        for (auto &entry : llvm::reverse(plans)) {
            auto idx = entry.first;
            auto &p  = entry.second;
            if (p.jagged) {
                replaceStructArgJagged(func, idx, p.argDecl, p.rec,
                                       p.jaggedEleTy, p.jaggedExtents);
                // Jagged path is not represented in the plan yet.
                continue;
            }
            if (p.nested) {
                // Phase 2 dummy-arg extension: replace the nested struct
                // dummy with one block arg per leaf.  Bindings-side
                // ``FlattenEntry`` emission for ``outer_kind="dummy_nested"``
                // is a separate follow-up — Python-side callers can pass
                // the flat companions directly via kwargs today; the
                // Fortran caller wrapper needs the recipe to pack the
                // nested struct's path-form members on its end.
                replaceStructArgNested(func, idx, p.argDecl, p.leaves);
                continue;
            }
            // Record the entry BEFORE the declare is erased.  If
            // ``replaceStructArg`` bails out (dangling users on the
            // old declare), the entry still describes the intended
            // recipe — but the SDFG won't carry the flat members so
            // the emitter will just skip it downstream.
            std::string intentStr = extractIntent(p.argDecl.getFortranAttrs());
            llvm::StringSet<> aosAllocSet;
            for (auto &m : p.aosAllocMembers) aosAllocSet.insert(m);
            // Phase 5c-B: emit one aos_alloc=True entry per AoS+
            // allocatable member.  Then emit the regular entry
            // covering the non-allocatable members (skipped via the
            // exclude set).
            for (auto &m : p.aosAllocMembers)
                recordAosAllocEntry(p.argDecl, p.rec, m, intentStr,
                                    p.outerShape);
            recordStructArgEntry(p.argDecl, p.rec, intentStr,
                                 p.outerIsArray, p.outerShape,
                                 aosAllocSet);
            replaceStructArg(func, idx, p.argDecl, p.rec,
                             p.outerIsArray, p.outerShape,
                             aosAllocSet);
        }
        return true;
    }

    // -------------------------------------------------------------------
    // Struct dummy arguments
    // -------------------------------------------------------------------

    void replaceStructArg(mlir::func::FuncOp func, unsigned argIdx,
                          hlfir::DeclareOp argDecl, fir::RecordType rec,
                          bool outerIsArray = false,
                          llvm::ArrayRef<int64_t> outerShape = {},
                          const llvm::StringSet<> &aosAllocMembers = {}) {
        auto &block = func.front();
        auto loc = argDecl.getLoc();
        auto *ctx = func.getContext();
        auto baseName = argDecl.getUniqName().str();
        auto demangledBase = demangleVarName(baseName);

        // Insert new block args right after the old one so the argument order
        // tracks the original member order.  Insertion shifts indices >= pos
        // by 1, so we insert sequentially at argIdx+1, argIdx+2, …
        llvm::StringMap<mlir::Value> memberBase;
        llvm::StringMap<mlir::Value> aosAllocFlatBase;
        llvm::StringSet<> concatMembers;
        unsigned memberCount = 0;
        for (auto &pair : rec.getTypeList()) {
            auto memName = pair.first;
            auto memTy   = pair.second;

            // Phase 5c-B (true SDFG boundary): AoS + allocatable member.
            // Insert two block args — the runtime cap (``index``) then a
            // 2D data buffer ``ref<array<N x ?xT>>``.  Build a declare
            // for each, with the cap declare's name = ``cap_<base>_<m>``
            // so ``traceToDecl`` resolves the inner extent to that
            // symbol on the SDFG signature.  ``collapseAosAllocReads``
            // afterwards rewrites every ``fir.load + designate``
            // chain on the original member box into a direct 2-index
            // designate over the new flat declare.
            if (aosAllocMembers.count(memName)) {
                auto box = mlir::cast<fir::BoxType>(memTy);
                mlir::Type eleTy;
                if (auto heap = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
                    eleTy = mlir::cast<fir::SequenceType>(heap.getEleTy()).getEleTy();
                else if (auto ptr = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
                    eleTy = mlir::cast<fir::SequenceType>(ptr.getEleTy()).getEleTy();
                else
                    continue;

                // Pointee shape: outer extents (static) × {?} (cap, runtime).
                llvm::SmallVector<int64_t, 4> exts(outerShape.begin(), outerShape.end());
                exts.push_back(fir::SequenceType::getUnknownExtent());
                auto pointee = fir::SequenceType::get(exts, eleTy);
                auto refTy = fir::ReferenceType::get(pointee);
                // HLFIR's variable-form result for an array with any
                // dynamic extent must be a ``!fir.box``; flang itself
                // emits the same shape for explicit-shape dummies whose
                // last extent comes from a runtime ``n`` (see how
                // ``real(8), intent(inout) :: x(3, n)`` lowers — the
                // declare returns ``(!fir.box<!fir.array<3x?xf64>>,
                // !fir.ref<!fir.array<3x?xf64>>)``).  Match that pair so
                // the verifier accepts the synthesised declare.
                auto boxTy = fir::BoxType::get(pointee);

                auto idxTy = mlir::IndexType::get(ctx);
                auto idxRefTy = fir::ReferenceType::get(idxTy);

                std::string flatName = demangledBase + "_" + memName;
                std::string capName  = "cap_" + flatName;

                // Insert cap arg (block.getArgument as an ``index``
                // value passed by reference so the bindings layer can
                // populate it from the wrapper).
                unsigned capArgIdx = argIdx + 1 + memberCount;
                block.insertArgument(capArgIdx, idxRefTy, loc);
                auto capArg = block.getArgument(capArgIdx);
                ++memberCount;

                unsigned dataArgIdx = argIdx + 1 + memberCount;
                block.insertArgument(dataArgIdx, refTy, loc);
                auto dataArg = block.getArgument(dataArgIdx);
                ++memberCount;

                mlir::OpBuilder b(&block, std::next(argDecl->getIterator()));
                b.setInsertionPoint(argDecl);

                // Cap declare: scalar ``ref<index>`` with uniq_name
                // ``cap_<base>_<member>``.  The bridge's
                // ``traceToDecl`` will resolve the data declare's
                // shape extent to this name.
                mlir::NamedAttrList capAttrs;
                capAttrs.append("uniq_name",
                                mlir::StringAttr::get(ctx, capName));
                capAttrs.append(declareSegments(b, /*hasShape=*/false));
                auto capDecl = b.create<hlfir::DeclareOp>(
                    loc, mlir::TypeRange{idxRefTy, idxRefTy},
                    mlir::ValueRange{capArg}, capAttrs);

                // Load the cap value to use it in the data declare's
                // shape op.
                auto capVal = b.create<fir::LoadOp>(
                    loc, capDecl.getResult(0)).getResult();

                // Build the shape op: outer (static) + cap (runtime).
                llvm::SmallVector<mlir::Value, 4> dims;
                for (auto e : outerShape)
                    dims.push_back(b.create<mlir::arith::ConstantOp>(
                        loc, idxTy, b.getIndexAttr(e)).getResult());
                dims.push_back(capVal);
                auto shapeTy = fir::ShapeType::get(ctx, dims.size());
                auto shape = b.create<fir::ShapeOp>(loc, shapeTy, dims).getResult();

                mlir::NamedAttrList attrs;
                attrs.append("uniq_name",
                             mlir::StringAttr::get(ctx, flatName));
                attrs.append(declareSegments(b, /*hasShape=*/true));

                auto newDecl = b.create<hlfir::DeclareOp>(
                    loc, mlir::TypeRange{boxTy, refTy},
                    mlir::ValueRange{dataArg, shape}, attrs);

                aosAllocFlatBase[memName] = newDecl.getResult(0);
                continue;
            }

            auto pointee = companionPointee(outerIsArray, outerShape, memTy);
            if (!pointee) continue;  // defensive; caller already checked
            auto refTy = fir::ReferenceType::get(pointee);

            bool memberIsArray = mlir::isa<fir::SequenceType>(memTy);
            bool concat = outerIsArray && memberIsArray;
            if (concat) concatMembers.insert(memName);

            unsigned newArgIdx = argIdx + 1 + memberCount;
            block.insertArgument(newArgIdx, refTy, loc);
            auto newArg = block.getArgument(newArgIdx);

            mlir::OpBuilder b(&block, std::next(argDecl->getIterator()));
            b.setInsertionPoint(argDecl);

            // Array members need a fir.shape operand for the declare to verify.
            // For AoS+memberArray (concat), build the concat shape from
            // the outer dims followed by the member's static extents.
            auto extents = staticArrayExtents(pointee);
            mlir::Value shape = emitStaticShape(b, loc, extents);

            llvm::SmallVector<mlir::Value, 2> operands;
            operands.push_back(newArg);
            if (shape) operands.push_back(shape);

            mlir::NamedAttrList attrs;
            attrs.append("uniq_name",
                         mlir::StringAttr::get(ctx,
                                               baseName + "_" + memName));
            attrs.append(declareSegments(b, /*hasShape=*/shape != nullptr));

            auto newDecl = b.create<hlfir::DeclareOp>(
                loc, mlir::TypeRange{refTy, refTy},
                mlir::ValueRange(operands), attrs);

            memberBase[memName] = newDecl.getResult(0);
            ++memberCount;
        }

        // Phase 5c-B AoS+allocatable: rewrite every per-instance
        // ``fir.load + hlfir.designate`` chain into a 2-index
        // designate over the new flat declare.  Run BEFORE the plain-
        // member designate sweep so the alloc-member's parent designates
        // (``A(i)``) are still alive.
        for (auto &kv : aosAllocFlatBase) {
            stripReallocOnAosMember(argDecl, kv.first());
            collapseAosAllocReads(argDecl, kv.first(), kv.second);
        }

        // Rewrite designates on the struct declare.  For AoS, the
        // direct user is an indexed designate (no component) on the
        // outer array; the actual component-designate is its child.
        llvm::SmallVector<hlfir::DesignateOp, 16> designates;
        for (auto *u : argDecl.getResult(0).getUsers()) {
            auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u);
            if (!dg) continue;
            bool hasComponent = false;
            for (auto nm : {"component_name", "component"})
                if (dg->getAttrOfType<mlir::StringAttr>(nm)) {
                    hasComponent = true;
                    break;
                }
            if (hasComponent) {
                designates.push_back(dg);
                continue;
            }
            // Pure indexed designate (A(i) on AoS dummy) — collect children.
            for (auto *cu : dg.getResult().getUsers())
                if (auto cdg = mlir::dyn_cast<hlfir::DesignateOp>(cu))
                    designates.push_back(cdg);
        }
        for (auto dg : designates) rewriteDesignate(dg, memberBase, concatMembers);

        // Erase the old declare; if other ops still reference its results
        // something went sideways and we leave the block arg in place rather
        // than breaking the IR.
        if (!argDecl.getResult(0).use_empty() ||
            !argDecl.getResult(1).use_empty())
            return;
        argDecl.erase();
        if (!block.getArgument(argIdx).use_empty()) return;
        block.eraseArgument(argIdx);
    }

    /// Replace a NESTED struct dummy arg with one block arg per flat
    /// leaf.  Mirrors ``replaceStructArg`` for the single-level case
    /// but consumes ``collectFlatLeaves`` output to handle arbitrary
    /// nesting depth.  Static-shape leaves only — dynamic-extent or
    /// allocatable leaves are left for the Phase 5b nested follow-up
    /// (the leaf walker bails on those upstream).
    ///
    /// For each leaf with path ``[a, b, c]`` and type ``leafTy``:
    ///   * Insert a block arg of type ``ref<leafTy>`` after the
    ///     original struct dummy.
    ///   * Synthesise an ``hlfir.declare`` with
    ///     ``uniq_name = <base>_a_b_c`` and a static shape operand
    ///     when ``leafTy`` is an array.
    ///
    /// Then walks every chain rooted at the original ``argDecl`` (or
    /// any inlined-callee alias of it) and rewrites it via the shared
    /// ``rewriteChainsRootedAt`` helper.  Erases the old declare and
    /// the original block arg if all uses cleared.
    void replaceStructArgNested(mlir::func::FuncOp func, unsigned argIdx,
                                hlfir::DeclareOp argDecl,
                                llvm::ArrayRef<FlatLeaf> leaves) {
        auto &block = func.front();
        auto loc = argDecl.getLoc();
        auto *ctx = func.getContext();
        auto baseName = argDecl.getUniqName().str();
        auto demangledBase = demangleVarName(baseName);

        llvm::StringMap<mlir::Value> leafBase;
        unsigned leafCount = 0;
        for (auto &leaf : leaves) {
            // Build the joined-path key (matches ``rewriteDesignateChain``'s
            // ``walkDesignateChain`` output).
            std::string joinedKey;
            for (unsigned i = 0; i < leaf.path.size(); ++i) {
                if (i) joinedKey += "_";
                joinedKey += leaf.path[i];
            }
            std::string suffix = joinedKey;

            auto leafTy = leaf.leafTy;
            auto refTy = fir::ReferenceType::get(leafTy);

            unsigned newArgIdx = argIdx + 1 + leafCount;
            block.insertArgument(newArgIdx, refTy, loc);
            auto newArg = block.getArgument(newArgIdx);
            ++leafCount;

            mlir::OpBuilder b(argDecl);

            // Array leaves need a fir.shape operand on the declare;
            // dynamic-extent leaves were filtered upstream by
            // ``collectFlatLeaves``.  Allocatable / pointer leaves
            // (``box<heap<array<?>>>`` / ``box<ptr<array<?>>>``) carry
            // their shape in the descriptor at runtime — no explicit
            // shape op, but the Fortran ``allocatable`` / ``pointer``
            // attr must be set so ``extract_vars`` peels through every
            // wrapper to find the inner SequenceType (rank > 0
            // classification).
            mlir::Value leafShape;
            fir::FortranVariableFlagsAttr fortranAttrs;
            if (auto seq = mlir::dyn_cast<fir::SequenceType>(leafTy)) {
                llvm::SmallVector<int64_t, 4> exts;
                for (auto d : seq.getShape()) {
                    if (d == fir::SequenceType::getUnknownExtent()) return;
                    exts.push_back(d);
                }
                leafShape = emitStaticShape(b, loc, exts);
            } else if (auto box = mlir::dyn_cast<fir::BoxType>(leafTy)) {
                // Allocatable / pointer leaf — the box wraps a
                // ``fir.heap`` (allocatable) or ``fir.ptr`` (pointer).
                bool isPointer = mlir::isa<fir::PointerType>(box.getEleTy());
                fortranAttrs = fir::FortranVariableFlagsAttr::get(
                    ctx,
                    isPointer ? fir::FortranVariableFlagsEnum::pointer
                              : fir::FortranVariableFlagsEnum::allocatable);
            }

            llvm::SmallVector<mlir::Value, 2> operands{newArg};
            if (leafShape) operands.push_back(leafShape);
            mlir::NamedAttrList attrs;
            attrs.append("uniq_name",
                         mlir::StringAttr::get(ctx,
                                               demangledBase + "_" + suffix));
            if (fortranAttrs) attrs.append("fortran_attrs", fortranAttrs);
            attrs.append(declareSegments(b, /*hasShape=*/leafShape != nullptr));

            auto newDecl = b.create<hlfir::DeclareOp>(
                loc, mlir::TypeRange{refTy, refTy},
                mlir::ValueRange(operands), attrs);

            leafBase[joinedKey] = newDecl.getResult(0);
        }

        // Reuse the chain-rewrite machinery from the local-instance
        // path: walks every designate chain rooted at ``argDecl`` (or
        // any inlined-callee alias of it) and folds it down to the
        // matching flat companion.
        rewriteChainsRootedAt(argDecl, leafBase);

        // Erase the original struct declare + block arg if all uses
        // cleared.  If something still references the old declare,
        // leave both in place so the IR stays valid.
        if (!argDecl.getResult(0).use_empty()
            || !argDecl.getResult(1).use_empty())
            return;
        argDecl.erase();
        if (!block.getArgument(argIdx).use_empty()) return;
        block.eraseArgument(argIdx);
    }

    /// Pack a jagged scalar-struct argument (1-D array members of same scalar
    /// type with differing extents) into a single 2-D companion of shape
    /// ``[numMembers x max(extents)]``.  Access to member `m` at index `j`
    /// becomes ``combined(rowIdx(m), j)`` — an ELLPACK-style padded view.
    void replaceStructArgJagged(mlir::func::FuncOp func, unsigned argIdx,
                                hlfir::DeclareOp argDecl, fir::RecordType rec,
                                mlir::Type eleTy,
                                llvm::ArrayRef<int64_t> extents) {
        auto &block = func.front();
        auto loc = argDecl.getLoc();
        auto *ctx = func.getContext();
        auto baseName = argDecl.getUniqName().str();

        int64_t maxExt = 0;
        for (auto e : extents) if (e > maxExt) maxExt = e;
        int64_t rows = extents.size();

        auto combinedTy  = fir::SequenceType::get({rows, maxExt}, eleTy);
        auto combinedRef = fir::ReferenceType::get(combinedTy);

        // New block argument right after the old struct arg.
        block.insertArgument(argIdx + 1, combinedRef, loc);
        auto combinedArg = block.getArgument(argIdx + 1);

        mlir::OpBuilder b(argDecl);
        mlir::NamedAttrList attrs;
        attrs.append("uniq_name",
                     mlir::StringAttr::get(ctx, baseName + "_packed"));
        attrs.append(declareSegments(b, /*hasShape=*/false));
        auto combinedDecl = b.create<hlfir::DeclareOp>(
            loc, mlir::TypeRange{combinedRef, combinedRef},
            mlir::ValueRange{combinedArg}, attrs);

        // Per-member aliased view into a single row of the combined array.
        // fir.coordinate_of rank-reduces the 2-D combined ref to a 1-D row
        // ref; fir.convert then bridges the row's max-extent type to the
        // member's original extent type so downstream hlfir.designate uses
        // type-check unchanged.
        llvm::StringMap<mlir::Value> memberBase;
        int64_t rowIdx = 0;
        auto idxTy = b.getIndexType();
        for (auto &pair : rec.getTypeList()) {
            auto memName = pair.first;
            auto memSeq  = mlir::cast<fir::SequenceType>(pair.second);
            int64_t ext  = memSeq.getShape()[0];

            auto rowConst = b.create<mlir::arith::ConstantOp>(
                loc, idxTy, b.getIndexAttr(rowIdx));

            auto rowRefTy = fir::ReferenceType::get(
                fir::SequenceType::get({maxExt}, eleTy));
            auto rowPtr = b.create<fir::CoordinateOp>(
                loc, rowRefTy, combinedDecl.getResult(0),
                mlir::ValueRange{rowConst});

            auto memberRefTy = fir::ReferenceType::get(
                fir::SequenceType::get({ext}, eleTy));
            auto casted = b.create<fir::ConvertOp>(
                loc, memberRefTy, rowPtr.getResult());

            memberBase[memName] = casted.getResult();
            ++rowIdx;
        }

        // Rewrite each component-selecting designate to the member's view.
        llvm::SmallVector<hlfir::DesignateOp, 8> designates;
        for (auto *u : argDecl.getResult(0).getUsers())
            if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u))
                designates.push_back(dg);
        for (auto dg : designates) rewriteDesignate(dg, memberBase);

        if (!argDecl.getResult(0).use_empty() ||
            !argDecl.getResult(1).use_empty())
            return;
        argDecl.erase();
        if (!block.getArgument(argIdx).use_empty()) return;
        block.eraseArgument(argIdx);
    }

    // -------------------------------------------------------------------
    // Local struct allocations
    // -------------------------------------------------------------------

    // -------------------------------------------------------------------
    // Allocatable struct member helpers (Phase 5a)
    // -------------------------------------------------------------------
    //
    // When we replace ``s%w`` (allocatable member) with a flat
    // top-level allocatable ``s_w``, the user's ``allocate(s%w(N))``
    // statement still lowers via flang to a ``fir.allocmem`` op whose
    // ``uniq_name`` attribute points at the MEMBER's namespace
    // (``_QMlibEw.alloc``) — independent of the enclosing struct's
    // declare scope.  The bridge's ``collectAllocSites`` matches
    // allocate sites by ``<declUniqName>.alloc``, so without renaming,
    // the flat declare's allocate site is invisible and the SDFG
    // ends up with an unbound runtime extent symbol.
    //
    // ``renameMemberAllocmems`` walks the original struct declare's
    // direct designate users with component name == ``memName``, and
    // for each, walks store users.  Any allocmem reaching that store
    // through ``fir.embox`` (the standard allocate lowering shape)
    // gets its ``uniq_name`` rewritten to ``<flatName>.alloc``.
    //
    // Caveats
    // -------
    // * Only the direct designate users of ``decl`` are walked.
    //   Aliases through inlined-callee declares would need the same
    //   alias-following machinery the main rewrite uses; we don't
    //   support cross-call ``allocate(s%w(...))`` yet.  This matches
    //   the AoS/parametric-dim phase boundaries documented above.
    // * Multiple allocate sites for the same member (an allocate +
    //   deallocate + re-allocate cycle, for example) all get the
    //   same flat name.  ``allocAliasName`` in extract_vars.cpp then
    //   mints ``<flat>_alloc1``, ``<flat>_alloc2``, … per site.
    void renameMemberAllocmems(hlfir::DeclareOp decl,
                               llvm::StringRef memName,
                               llvm::StringRef flatName) {
        auto *ctx = decl.getContext();
        std::string newAlloc = (flatName + ".alloc").str();

        // Bug fix: Phase 5a only walked ``decl``'s direct users.  When
        // the ``allocate(s%w(n))`` call sits inside an internal
        // subprogram, ``hlfir-inline-all`` splices the callee body in
        // and the designate of ``s%w`` ends up rooted at an inlined
        // alias declare (its memref traces back through ``fir.embox``
        // / ``fir.convert`` chains to ``decl``), not at ``decl``
        // itself.  Walk both ``decl`` directly AND every aliasing
        // declare that resolves back to it, mirroring the same
        // alias-following machinery the main ``splitLocal``
        // designate-collector uses.
        llvm::SmallVector<hlfir::DeclareOp, 8> roots{decl};
        llvm::DenseSet<mlir::Operation*> seen{decl.getOperation()};
        std::function<bool(mlir::Value)> resolvesTo = [&](mlir::Value v) -> bool {
            for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
                auto *d = v.getDefiningOp();
                if (!d) return false;
                if (auto outer = mlir::dyn_cast<hlfir::DeclareOp>(d))
                    return outer == decl;
                if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) { v = cv.getValue(); continue; }
                if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) { v = eb.getMemref(); continue; }
                return false;
            }
            return false;
        };
        // Walk all hlfir.declare ops in the enclosing function and
        // collect those that alias ``decl`` (memref chains through
        // ``embox`` / ``convert`` / another ``declare`` back to it).
        if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
            func.walk([&](hlfir::DeclareOp other) {
                if (other == decl) return;
                if (seen.count(other.getOperation())) return;
                if (resolvesTo(other.getMemref())) {
                    seen.insert(other.getOperation());
                    roots.push_back(other);
                }
            });
        }

        for (auto root : roots) {
            for (auto *u : root.getResult(0).getUsers()) {
                auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u);
                if (!dg) continue;
                mlir::StringAttr compAttr;
                for (auto nm : {"component_name", "component"})
                    if (auto a = dg->getAttrOfType<mlir::StringAttr>(nm)) {
                        compAttr = a;
                        break;
                    }
                if (!compAttr || compAttr.getValue() != memName) continue;
                for (auto *du : dg.getResult().getUsers()) {
                    auto store = mlir::dyn_cast<fir::StoreOp>(du);
                    if (!store) continue;
                    // Trace the stored value back to its
                    // ``fir.allocmem`` through the standard ``embox``
                    // wrapping (and possibly a ``fir.convert`` for
                    // box-shape canonicalisation).
                    mlir::Value v = store.getValue();
                    for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
                        auto *d = v.getDefiningOp();
                        if (!d) break;
                        if (auto am = mlir::dyn_cast<fir::AllocMemOp>(d)) {
                            am->setAttr("uniq_name",
                                        mlir::StringAttr::get(ctx, newAlloc));
                            break;
                        }
                        if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) { v = eb.getMemref(); continue; }
                        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) { v = cv.getValue(); continue; }
                        break;
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // AoS + allocatable helpers (Phase 5c-A)
    // -------------------------------------------------------------------
    //
    // ``aosAllocUniformConstSize(decl, memName)`` walks every
    // ``fir.allocmem`` op whose target traces back through ``embox`` /
    // ``convert`` to a designate of ``decl(<i>){<memName>}`` and checks
    // that all such allocate sites use the SAME compile-time constant
    // size operand.  Returns the constant on uniform match, ``nullopt``
    // otherwise.
    //
    // The "uniform constant" gate is what makes Phase 5c-A's
    // padding-to-max trivial: every ``A(i)%<member>`` allocates the
    // same M elements, so the flat companion is statically
    // ``A_<member>(N, M)``, the bindings layer doesn't need to compute
    // ``max`` at runtime, and the kernel-internal ``allocate`` sites
    // become semantic no-ops over the pre-existing 2D buffer.
    //
    // Caveats
    // -------
    // * Element-form designate of ``decl`` (``A(i)`` with a SCALAR
    //   index per outer dim) is the only path we walk.  Section-form
    //   designates of the AoS outer (``A(1:N)``) wouldn't match —
    //   they'd be compiler-generated whole-array assigns, not
    //   per-instance allocates.
    // * Sites whose size operand isn't a constant (e.g.
    //   ``allocate(A(i)%w(some_runtime_var))``) cause us to bail —
    //   that's the variable-runtime-size case (5c-B / 5c-C).
    /// Collapse the ``fir.load (designate of A(i){memName}) →
    /// hlfir.designate (loaded, j)`` read pattern into a direct
    /// 2-index ``hlfir.designate flatBase (i, j)`` over the Phase
    /// 5c-A companion.
    ///
    /// Why: the original IR threads every read of ``A(i)%w(j)``
    /// through the box descriptor — flang emits ``fir.load %ref``
    /// to fetch the descriptor, then ``hlfir.designate %loaded
    /// (j)`` to index inside the box.  After flatten replaces
    /// ``%ref`` with a plain ``ref<array<NxMxT>>``, ``fir.load``
    /// loads the *whole* 2D value rather than a box, and the
    /// inner ``designate (j)`` indexes it as if it were 1-D.
    /// We leapfrog by replacing the entire chain with a direct
    /// ``hlfir.designate flatBase (i, j)`` against the new ref.
    ///
    /// Mirrors the strategy in ``hlfir-rewrite-pointer-assigns``
    /// (forward-substitute a multi-step load chain into a single
    /// direct access against the rewrite target).
    ///
    /// Caveats
    /// -------
    /// * Only walks element-form parent designates (``A(i)``,
    ///   single scalar index per outer dim).  Section-form parent
    ///   designates (``A(1:N)%w``) are a different shape and not
    ///   handled here.
    /// * Only walks element-form INNER designates
    ///   (``loaded(j)``).  Whole-component reads
    ///   (``hlfir.assign x to <designate of A(i){w}>``) take a
    ///   separate section-rewrite path that's not yet wired.
    /// * If any reader is a ``fir.box_addr`` rather than a
    ///   designate (the path the existing pointer rewrite
    ///   handles), the chain is left alone — the bridge's
    ///   downstream handling for ``box_addr`` returns a
    ///   ``fir.ptr<...>`` that doesn't match the static 2D
    ///   companion.  Future TODO if real code hits this.
    void collapseAosAllocReads(hlfir::DeclareOp decl,
                               llvm::StringRef memName,
                               mlir::Value flatBase) {
        // Phase 5c-B: also recognise inlined-callee aliased declares.
        // When ``hlfir-inline-all`` splices a module-contained
        // ``call kernel(A)`` body into the caller, the kernel's
        // ``A`` dummy becomes a fresh ``hlfir.declare %caller_A_decl
        // dummy_scope %dsc {uniq_name="..."}`` aliasing the same
        // storage.  Designates inside the inlined kernel body are
        // rooted at the alias's results, not ``decl``'s — without
        // following alias chains we'd miss every read inside the
        // inlined call.
        auto isDeclOrAlias = [&](mlir::Value v) -> bool {
            for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
                if (v == decl.getResult(0) || v == decl.getResult(1))
                    return true;
                auto *d = v.getDefiningOp();
                if (!d) return false;
                if (auto inner = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
                    v = inner.getMemref();
                    continue;
                }
                if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
                    v = eb.getMemref();
                    continue;
                }
                if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
                    v = cv.getValue();
                    continue;
                }
                return false;
            }
            return false;
        };
        if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
            // Two-stage erase so dependencies tear down cleanly:
            // (1) eraseInner — the rewritten inner designates
            //     (still hold a use on ``load`` until erased)
            // (2) eraseRest  — load, memDg, parent (sweep after the
            //     inner designates are gone so the use_empty checks
            //     trigger)
            llvm::SmallVector<mlir::Operation*, 16> eraseInner;
            llvm::SmallVector<mlir::Operation*, 16> eraseRest;
            func.walk([&](fir::LoadOp load) {
                // Is this a load of a per-instance member designate?
                auto memDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
                    load.getMemref().getDefiningOp());
                if (!memDg) return;
                mlir::StringAttr compAttr;
                for (auto nm : {"component_name", "component"})
                    if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
                        compAttr = a; break;
                    }
                if (!compAttr || compAttr.getValue() != memName) return;
                auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
                    memDg.getMemref().getDefiningOp());
                if (!parent) return;
                if (!isDeclOrAlias(parent.getMemref()))
                    return;
                // parent must be element-form (no triplets).
                for (bool t : parent.getIsTriplet()) if (t) return;
                if (parent.getIndices().empty()) return;

                // Collect the outer indices (typically a single
                // scalar i for a 1-D AoS).
                llvm::SmallVector<mlir::Value, 4> outerIdx(
                    parent.getIndices().begin(), parent.getIndices().end());

                // For each user of the loaded box: if it's an
                // element-form designate, rewrite to a direct
                // 2-index designate over flatBase.
                for (auto *u : load.getResult().getUsers()) {
                    auto inner = mlir::dyn_cast<hlfir::DesignateOp>(u);
                    if (!inner) continue;
                    // Element form (no triplets, has indices).
                    bool anyTrip = false;
                    for (bool t : inner.getIsTriplet()) if (t) { anyTrip = true; break; }
                    if (anyTrip) continue;
                    if (inner.getIndices().empty()) continue;

                    mlir::OpBuilder b(inner);
                    auto idxTy = b.getIndexType();
                    auto toIndex = [&](mlir::Value v) {
                        if (v.getType() == idxTy) return v;
                        return b.create<fir::ConvertOp>(
                            inner.getLoc(), idxTy, v).getResult();
                    };
                    llvm::SmallVector<mlir::Value, 4> mergedIdx;
                    for (auto v : outerIdx) mergedIdx.push_back(toIndex(v));
                    for (auto v : inner.getIndices()) mergedIdx.push_back(toIndex(v));

                    // Build the new designate.  Result type stays
                    // the inner designate's result type (element ref).
                    auto newDg = b.create<hlfir::DesignateOp>(
                        inner.getLoc(),
                        inner.getResult().getType(),
                        flatBase,
                        mlir::ValueRange{mergedIdx});
                    inner.getResult().replaceAllUsesWith(newDg.getResult());
                    eraseInner.push_back(inner);
                }
                // The load + the member/parent designate chain become
                // dead once the inner designate is erased.  Schedule
                // them for the second sweep.
                eraseRest.push_back(load);
                eraseRest.push_back(memDg);
                eraseRest.push_back(parent);
            });
            for (auto *op : eraseInner)
                if (op->use_empty()) op->erase();
            for (auto *op : eraseRest)
                if (op->use_empty()) op->erase();
        }
    }

    /// Rewrite whole-component ``hlfir.assign``s whose LHS is
    /// ``<designate of A(i){memName}>`` into row-section assigns on
    /// the flat 2D companion: ``A_<member>(i, 1:M:1) = rhs``.
    /// Without this, the existing concat path replaces the LHS with
    /// the bare flat declare (the whole 2D), and the scalar ``rhs``
    /// gets broadcast across ALL rows — silently corrupting
    /// previously-written rows.
    ///
    /// Element-form assigns (``A(i)%w(j) = ...``) are NOT in scope
    /// here — those go through the element designate + the
    /// existing concat path, which already merges parent + inner
    /// indices correctly.  Only whole-component assigns
    /// (``A(i)%w = scalar`` or ``A(i)%w = src(:)``) need the
    /// section-rewrite treatment.
    /// Look up the allocate-site size for a specific outer index of
    /// an AoS-allocatable member.  Returns the constant size used in
    /// ``allocate(A(i)%<member>(N_i))`` when ``i`` matches
    /// ``targetIdx`` and the size is a compile-time constant.
    /// Returns ``nullopt`` if the matching allocate isn't found or
    /// its size isn't constant — in which case the caller falls back
    /// to the global cap ``M``.
    static std::optional<int64_t> aosAllocSizeAt(
            hlfir::DeclareOp decl, llvm::StringRef memName,
            int64_t targetIdx) {
        std::optional<int64_t> found;
        if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
            func.walk([&](hlfir::DesignateOp memDg) -> mlir::WalkResult {
                mlir::StringAttr compAttr;
                for (auto nm : {"component_name", "component"})
                    if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
                        compAttr = a; break;
                    }
                if (!compAttr || compAttr.getValue() != memName)
                    return mlir::WalkResult::advance();
                auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
                    memDg.getMemref().getDefiningOp());
                if (!parent) return mlir::WalkResult::advance();
                if (parent.getMemref() != decl.getResult(0) &&
                    parent.getMemref() != decl.getResult(1))
                    return mlir::WalkResult::advance();
                for (bool t : parent.getIsTriplet()) if (t)
                    return mlir::WalkResult::advance();
                if (parent.getIndices().size() != 1)
                    return mlir::WalkResult::advance();
                auto pi = traceConstInt(parent.getIndices().front());
                if (!pi || *pi != targetIdx)
                    return mlir::WalkResult::advance();
                for (auto *u : memDg.getResult().getUsers()) {
                    auto store = mlir::dyn_cast<fir::StoreOp>(u);
                    if (!store) continue;
                    mlir::Value v = store.getValue();
                    for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
                        auto *d = v.getDefiningOp();
                        if (!d) break;
                        if (auto am = mlir::dyn_cast<fir::AllocMemOp>(d)) {
                            auto sizes = am.getShape();
                            if (sizes.size() == 1) {
                                if (auto sz = traceConstInt(sizes.front())) {
                                    found = *sz;
                                    return mlir::WalkResult::interrupt();
                                }
                            }
                            return mlir::WalkResult::interrupt();
                        }
                        if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) { v = eb.getMemref(); continue; }
                        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) { v = cv.getValue(); continue; }
                        break;
                    }
                }
                return mlir::WalkResult::advance();
            });
        }
        return found;
    }

    void rewriteAosWholeMemberAssign(hlfir::DeclareOp decl,
                                     llvm::StringRef memName,
                                     mlir::Value flatBase,
                                     int64_t M) {
        if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
            llvm::SmallVector<hlfir::AssignOp, 4> dead;
            func.walk([&](hlfir::AssignOp op) {
                auto memDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
                    op.getLhs().getDefiningOp());
                if (!memDg) return;
                mlir::StringAttr compAttr;
                for (auto nm : {"component_name", "component"})
                    if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
                        compAttr = a; break;
                    }
                if (!compAttr || compAttr.getValue() != memName) return;
                auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
                    memDg.getMemref().getDefiningOp());
                if (!parent) return;
                if (parent.getMemref() != decl.getResult(0) &&
                    parent.getMemref() != decl.getResult(1))
                    return;
                // parent must be element-form (no triplets).
                for (bool t : parent.getIsTriplet()) if (t) return;
                if (parent.getIndices().empty()) return;

                // Per-instance section bound.  When the outer index
                // is a compile-time constant we can match it to a
                // specific allocate site and use that site's size as
                // the section bound — needed for the jagged case
                // (``A(1)%val(3)`` vs ``A(2)%val(4)``) so each row
                // assign writes only its live region instead of
                // splatting up to the global cap.  Falls back to the
                // cap when the index is symbolic or no matching
                // allocate is found.
                int64_t sectionBound = M;
                if (parent.getIndices().size() == 1) {
                    if (auto pi = traceConstInt(parent.getIndices().front())) {
                        if (auto specific = aosAllocSizeAt(decl, memName, *pi))
                            sectionBound = *specific;
                    }
                }

                // Build A_w(parent_idx, 1:sectionBound:1) section designate.
                mlir::OpBuilder b(op);
                auto loc = op.getLoc();
                auto idxTy = b.getIndexType();
                auto toIndex = [&](mlir::Value v) {
                    if (v.getType() == idxTy) return v;
                    return b.create<fir::ConvertOp>(loc, idxTy, v).getResult();
                };
                auto c1 = b.create<mlir::arith::ConstantOp>(
                    loc, idxTy, b.getIndexAttr(1));
                auto cBound = b.create<mlir::arith::ConstantOp>(
                    loc, idxTy, b.getIndexAttr(sectionBound));

                llvm::SmallVector<mlir::Value, 6> indices;
                llvm::SmallVector<bool, 4> tripletFlags;
                for (auto v : parent.getIndices()) {
                    indices.push_back(toIndex(v));
                    tripletFlags.push_back(false);
                }
                indices.push_back(c1.getResult());
                indices.push_back(cBound.getResult());
                indices.push_back(c1.getResult());
                tripletFlags.push_back(true);

                // Result type: box<array<sectionBound x T>> — a row
                // view shaped to match the per-instance live region.
                auto flatTy = mlir::cast<fir::ReferenceType>(
                    flatBase.getType()).getEleTy();
                auto flatSeq = mlir::cast<fir::SequenceType>(flatTy);
                auto eleTy = flatSeq.getEleTy();
                auto rowSeqTy = fir::SequenceType::get({sectionBound}, eleTy);
                auto boxTy = fir::BoxType::get(rowSeqTy);

                auto newShape = b.create<fir::ShapeOp>(
                    loc, mlir::ValueRange{cBound.getResult()}).getResult();

                auto sectionDg = b.create<hlfir::DesignateOp>(
                    loc,
                    /*resultType0=*/boxTy,
                    /*memref=*/flatBase,
                    /*component=*/mlir::StringAttr{},
                    /*component_shape=*/mlir::Value{},
                    /*indices=*/mlir::ValueRange{indices},
                    /*is_triplet=*/b.getDenseBoolArrayAttr(tripletFlags),
                    /*substring=*/mlir::ValueRange{},
                    /*complex_part=*/mlir::BoolAttr{},
                    /*shape=*/newShape,
                    /*typeparams=*/mlir::ValueRange{},
                    /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});

                // Build the new assign with the section LHS.  The RHS
                // stays as-is (scalar or src array of matching shape).
                b.create<hlfir::AssignOp>(loc, op.getRhs(), sectionDg.getResult());
                dead.push_back(op);
            });
            for (auto op : dead) op.erase();
        }
    }

    /// Strip the ``realloc`` attribute from ``hlfir.assign`` ops
    /// whose LHS designates an AoS-allocatable member after
    /// flattening.  Phase 5c-A turns the LHS into a static array
    /// section; ``realloc`` is only valid when the LHS is genuinely
    /// allocatable, and the op's verifier rejects it otherwise.
    void stripReallocOnAosMember(hlfir::DeclareOp decl,
                                 llvm::StringRef memName) {
        if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
            func.walk([&](hlfir::AssignOp op) {
                mlir::Value lhs = op.getLhs();
                auto memDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
                    lhs.getDefiningOp());
                if (!memDg) return;
                mlir::StringAttr compAttr;
                for (auto nm : {"component_name", "component"})
                    if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
                        compAttr = a; break;
                    }
                if (!compAttr || compAttr.getValue() != memName) return;
                auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
                    memDg.getMemref().getDefiningOp());
                if (!parent) return;
                if (parent.getMemref() != decl.getResult(0) &&
                    parent.getMemref() != decl.getResult(1))
                    return;
                op.setRealloc(false);
            });
        }
    }

    /// Erase the kernel-internal allocate / deallocate chain for an
    /// AoS allocatable member after Phase 5c-A flattening.  The 2D
    /// buffer is now pre-allocated at static shape, so each
    /// ``allocate(A(i)%<member>(M))`` becomes a no-op:
    ///   * ``fir.store (embox(allocmem)) to <designate>`` — erase.
    ///   * ``fir.allocmem`` itself — erase if dead (no other users).
    ///   * ``fir.embox`` — erase if dead.
    ///   * ``fir.freemem`` (matching deallocate) — erase.
    ///   * Any subsequent ``fir.zero_bits`` + ``fir.embox`` + store
    ///     pattern (the post-deallocate "set descriptor to null"
    ///     sequence flang inserts) — erase.
    void eraseAosAllocDeallocChain(hlfir::DeclareOp decl,
                                   llvm::StringRef memName) {
        llvm::SmallVector<mlir::Operation*, 16> deadOps;
        if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
            func.walk([&](hlfir::DesignateOp memDg) {
                mlir::StringAttr compAttr;
                for (auto nm : {"component_name", "component"})
                    if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
                        compAttr = a; break;
                    }
                if (!compAttr || compAttr.getValue() != memName) return;
                auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
                    memDg.getMemref().getDefiningOp());
                if (!parent) return;
                if (parent.getMemref() != decl.getResult(0) &&
                    parent.getMemref() != decl.getResult(1))
                    return;
                for (auto *u : memDg.getResult().getUsers())
                    if (auto st = mlir::dyn_cast<fir::StoreOp>(u))
                        deadOps.push_back(st);
            });
            // Also collect ``fir.freemem`` ops whose source traces
            // back through ``fir.box_addr`` + ``fir.load`` of the
            // member designate.
            func.walk([&](fir::FreeMemOp fm) {
                mlir::Value v = fm.getHeapref();
                for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
                    auto *d = v.getDefiningOp();
                    if (!d) break;
                    if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(d)) { v = ba.getVal(); continue; }
                    if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) { v = ld.getMemref(); continue; }
                    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) { v = cv.getValue(); continue; }
                    if (auto memDg = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
                        mlir::StringAttr compAttr;
                        for (auto nm : {"component_name", "component"})
                            if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
                                compAttr = a; break;
                            }
                        if (compAttr && compAttr.getValue() == memName) {
                            auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
                                memDg.getMemref().getDefiningOp());
                            if (parent && (parent.getMemref() == decl.getResult(0) ||
                                           parent.getMemref() == decl.getResult(1))) {
                                deadOps.push_back(fm);
                            }
                        }
                        break;
                    }
                    break;
                }
            });
        }
        // Erase stores / freemem; the embox / allocmem / etc. become
        // dead and get swept by canonicalisation downstream.  The
        // wrapping do-loop's body (e.g. the per-instance
        // ``deallocate`` loop) becomes a stub of just iv bookkeeping
        // and dead box-load ops — but the IR-level loop op stays.
        // We don't try to erase the loop here (its result types
        // don't trivially substitute into init args).  Instead the
        // SDFGBuilder's post-gen sweep adds a single empty state to
        // any zero-block CFG region, so the resulting empty
        // ``LoopRegion`` validates cleanly.
        for (auto *op : deadOps) op->erase();
    }

    /// Result of scanning all ``allocate(A(i)%<m>(N))`` sites for a
    /// single member: the per-batch padding size (the max of all
    /// constant N values seen), and a uniform-flag set when every
    /// site uses the same constant.  Returns ``nullopt`` iff any
    /// site is non-constant or there are no sites.
    struct AosAllocConstSize {
        int64_t padTo;     ///< pad-to-max companion column count
        bool    uniform;   ///< true iff every allocate uses the same constant
    };

    static std::optional<AosAllocConstSize> aosAllocMaxConstSize(
            hlfir::DeclareOp decl, llvm::StringRef memName) {
        std::optional<int64_t> maxSeen, minSeen;
        bool anySite = false;
        // Walk all hlfir.designate ops that index into ``decl`` via an
        // outer-element form (``A(i)``) and then component-select the
        // target member.  For each, follow store users and recover the
        // allocate site.
        //
        // Generalised from "uniform-only" to "max-of-constants": a
        // genuinely jagged AoS like batched CSR (``allocate(A(1)%val(3))``
        // and ``allocate(A(2)%val(4))``) flattens to a max-padded
        // companion ``A_val(N, max)``.  The kernel's element-wise
        // accesses (``A(i)%val(j)``) work uniformly because ``j``
        // never exceeds the per-instance live size by program logic;
        // the padding columns stay unread.  Whole-component assigns
        // (``A(i)%w = scalar``) still fire the section-rewrite path —
        // see ``rewriteAosWholeMemberAssign`` — but only when the
        // result is uniform (otherwise the per-instance live size
        // differs from the cap and the simple ``1:M:1`` triplet
        // would over-write padding with stale data).
        if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
            mlir::WalkResult result = func.walk([&](hlfir::DesignateOp memDg) -> mlir::WalkResult {
                // Check this is the member designate (A(i){memName}).
                mlir::StringAttr compAttr;
                for (auto nm : {"component_name", "component"})
                    if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
                        compAttr = a; break;
                    }
                if (!compAttr || compAttr.getValue() != memName)
                    return mlir::WalkResult::advance();

                // Parent must be a per-instance designate of decl.
                auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
                    memDg.getMemref().getDefiningOp());
                if (!parent) return mlir::WalkResult::advance();
                if (parent.getMemref() != decl.getResult(0) &&
                    parent.getMemref() != decl.getResult(1))
                    return mlir::WalkResult::advance();
                // parent should be element-form (no triplets).
                for (bool t : parent.getIsTriplet()) if (t)
                    return mlir::WalkResult::advance();

                // Look for an allocate-store chain on memDg.
                for (auto *u : memDg.getResult().getUsers()) {
                    auto store = mlir::dyn_cast<fir::StoreOp>(u);
                    if (!store) continue;
                    mlir::Value v = store.getValue();
                    for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
                        auto *d = v.getDefiningOp();
                        if (!d) break;
                        if (auto am = mlir::dyn_cast<fir::AllocMemOp>(d)) {
                            // Recover the size: allocmem typically takes
                            // a single ``%size`` operand for a 1-D array.
                            auto sizes = am.getShape();
                            if (sizes.size() != 1) {
                                maxSeen.reset();
                                return mlir::WalkResult::interrupt();
                            }
                            auto sz = traceConstInt(sizes.front());
                            if (!sz) {
                                maxSeen.reset();
                                return mlir::WalkResult::interrupt();
                            }
                            anySite = true;
                            if (!maxSeen || *sz > *maxSeen) maxSeen = *sz;
                            if (!minSeen || *sz < *minSeen) minSeen = *sz;
                            break;
                        }
                        if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) { v = eb.getMemref(); continue; }
                        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) { v = cv.getValue(); continue; }
                        break;
                    }
                }
                return mlir::WalkResult::advance();
            });
            if (result.wasInterrupted()) return std::nullopt;
        }
        if (!anySite || !maxSeen) return std::nullopt;
        return AosAllocConstSize{*maxSeen, *minSeen == *maxSeen};
    }

    static bool isLocallyFlattenable(hlfir::DeclareOp decl) {
        auto *def = decl.getMemref().getDefiningOp();
        auto alloca = mlir::dyn_cast_or_null<fir::AllocaOp>(def);
        if (!alloca) return false;
        // Static shape only — building a companion fir.alloca for a runtime-
        // sized array would need the original shape operands threaded through.
        if (alloca.getNumOperands() != 0) return false;

        bool outerIsArray = false;
        llvm::SmallVector<int64_t, 4> outerShape;
        auto rec = peelToRecord(decl.getResult(0).getType(),
                                outerIsArray, outerShape);
        if (!rec) return false;
        for (auto d : outerShape)
            if (d == fir::SequenceType::getUnknownExtent()) return false;
        // Phase 5c-A: AoS + allocatable member.  Permitted when every
        // ``allocate(A(i)%<member>(M))`` site uses the SAME compile-
        // time constant ``M`` across all instances.  Then the flat
        // companion ``A_<member>(N, M)`` is fully static and the
        // alloc / dealloc / load / designate machinery becomes
        // semantic no-ops over the pre-allocated 2D buffer.  The
        // read-side rewrite (``collapseAosAllocReads``) leapfrogs
        // ``fir.load + hlfir.designate (loaded, j)`` into a direct
        // 2-index ``hlfir.designate flatBase (i, j)``.
        if (outerIsArray) {
            for (auto &pair : rec.getTypeList()) {
                if (!isAllocatableArrayMember(pair.second)) continue;
                if (!aosAllocMaxConstSize(decl, pair.first).has_value())
                    return false;
            }
        }
        if (allMembersFlattenable(rec, outerIsArray)) return true;
        // Nested-struct fallback.  ``collectFlatLeaves`` recurses
        // through ``RecordType`` and ``array<N x RecordType>``
        // members, building each leaf's flat companion shape.  When
        // the outer declare is itself an array of a nested record
        // (``type(t) :: s(3)`` with ``t`` nested), the same walker
        // accepts: thread the outer extents in as the initial
        // ``outerDims`` so every leaf's flat companion concatenates
        // them as leading dims (matching what ``splitLocal`` does
        // when it builds the alloca below).
        llvm::SmallVector<std::string, 4> prefix;
        llvm::SmallVector<int64_t, 4> initialDims(outerShape.begin(),
                                                   outerShape.end());
        llvm::SmallVector<FlatLeaf, 8> leaves;
        return collectFlatLeaves(rec, prefix, initialDims, leaves);
    }

    void splitLocal(hlfir::DeclareOp decl) {
        mlir::OpBuilder b(decl);
        auto *ctx = b.getContext();
        auto loc  = decl.getLoc();

        bool outerIsArray = false;
        llvm::SmallVector<int64_t, 4> outerShape;
        auto rec = peelToRecord(decl.getResult(0).getType(),
                                outerIsArray, outerShape);
        auto shape = decl.getShape();
        auto baseName = decl.getUniqName().str();

        // Nested-record path: walk the path-leaf set and synthesise
        // one declare per leaf, indexed by the path-joined name
        // (``o_inner_x``).  The single-level path below stays the
        // hot path for non-nested structs.
        //
        // When the outer declare is itself an array of a nested
        // record (``type(t) :: s(3)`` where ``t`` is nested), the
        // outer extents thread into ``collectFlatLeaves`` as the
        // initial ``outerDims`` so each leaf's flat companion
        // concatenates them as leading dims (e.g. ``s(3)%w(5,5)``
        // collapses to ``s_w`` of shape ``(3, 5, 5)``).
        bool nested = !allMembersFlattenable(rec, outerIsArray);
        if (nested) {
            llvm::SmallVector<std::string, 4> prefix;
            llvm::SmallVector<int64_t, 4> initialDims(outerShape.begin(),
                                                       outerShape.end());
            llvm::SmallVector<FlatLeaf, 8> leaves;
            if (!collectFlatLeaves(rec, prefix, initialDims, leaves)) return;

            llvm::StringMap<mlir::Value> leafBase;
            for (auto &leaf : leaves) {
                auto memTy = leaf.leafTy;
                auto newAlloca = b.create<fir::AllocaOp>(loc, memTy);
                auto declTy = fir::ReferenceType::get(memTy);
                std::string suffix;
                std::string joinedKey;
                for (unsigned i = 0; i < leaf.path.size(); ++i) {
                    if (i) { suffix += "_"; joinedKey += "_"; }
                    suffix    += leaf.path[i];
                    joinedKey += leaf.path[i];
                }

                // Array leaves need a fir.shape operand on the declare.
                // ``hlfir.declare op of array entity with a raw address
                // base must have a shape operand``.  We derive extents
                // from the leaf type itself; static extents only.
                mlir::Value leafShape;
                if (auto seq = mlir::dyn_cast<fir::SequenceType>(memTy)) {
                    llvm::SmallVector<int64_t, 4> exts;
                    bool allStatic = true;
                    for (auto d : seq.getShape()) {
                        if (d == fir::SequenceType::getUnknownExtent()) {
                            allStatic = false;
                            break;
                        }
                        exts.push_back(d);
                    }
                    if (!allStatic) return;  // dynamic extent unsupported
                    leafShape = emitStaticShape(b, loc, exts);
                }

                llvm::SmallVector<mlir::Value, 2> operands{newAlloca};
                if (leafShape) operands.push_back(leafShape);
                mlir::NamedAttrList attrs;
                attrs.append("uniq_name",
                             mlir::StringAttr::get(ctx, baseName + "_" + suffix));
                attrs.append(declareSegments(b, /*hasShape=*/leafShape != nullptr));
                auto newDecl = b.create<hlfir::DeclareOp>(
                    loc, mlir::TypeRange{declTy, declTy},
                    mlir::ValueRange(operands), attrs);
                leafBase[joinedKey] = newDecl.getResult(0);
            }

            // Walk every hlfir.designate chain rooted at ``decl`` (or
            // any inlined-callee alias of it) and rewrite each leaf to
            // the matching flat companion in ``leafBase``.  Shared
            // helper with the dummy-arg nested path
            // (``replaceStructArgNested``) — both sides build the same
            // ``leafBase`` and need the same chain-rewrite logic.
            rewriteChainsRootedAt(decl, leafBase);

            if (decl.getResult(0).use_empty()
                    && decl.getResult(1).use_empty()) {
                auto *allocaOp = decl.getMemref().getDefiningOp();
                decl.erase();
                if (allocaOp && allocaOp->use_empty()) allocaOp->erase();
            }
            return;
        }

        // Single-level path — every member is flat directly.
        llvm::StringMap<mlir::Value> memberBase;
        // Track which members are AoS-with-array-members so the
        // designate rewriter knows to merge outer + inner indices.
        llvm::StringSet<> concatMembers;
        for (auto &pair : rec.getTypeList()) {
            auto memName = pair.first;
            auto memTy   = pair.second;

            // Phase 5c-A: AoS + allocatable member with uniform
            // constant allocate size.  Synth a fully static 2D
            // companion ``A_<member>(N, M)`` and erase the per-
            // instance ``fir.allocmem`` / ``fir.freemem`` chain
            // (the buffer is pre-allocated at the static N*M shape;
            // the kernel's ``allocate(A(i)%w(M))`` becomes a
            // semantic no-op).
            if (isAllocatableArrayMember(memTy) && outerIsArray) {
                auto sizeOpt = aosAllocMaxConstSize(decl, memName);
                if (!sizeOpt) continue;  // gate already verified; defensive
                int64_t M = sizeOpt->padTo;
                bool sizeUniform = sizeOpt->uniform;
                auto box = mlir::cast<fir::BoxType>(memTy);
                mlir::Type eleTy;
                if (auto heap = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
                    eleTy = mlir::cast<fir::SequenceType>(heap.getEleTy()).getEleTy();
                else if (auto ptr = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
                    eleTy = mlir::cast<fir::SequenceType>(ptr.getEleTy()).getEleTy();
                else
                    continue;

                // Concat shape: outerShape × {M}.
                llvm::SmallVector<int64_t, 4> exts(outerShape.begin(), outerShape.end());
                exts.push_back(M);
                auto pointee = fir::SequenceType::get(exts, eleTy);
                auto refTy = fir::ReferenceType::get(pointee);

                auto newAlloca = b.create<fir::AllocaOp>(loc, pointee);
                mlir::Value memberShape = emitStaticShape(b, loc, exts);

                std::string flatName = baseName + "_" + memName;
                mlir::NamedAttrList attrs;
                attrs.append("uniq_name", mlir::StringAttr::get(ctx, flatName));
                attrs.append(declareSegments(b, /*hasShape=*/true));

                llvm::SmallVector<mlir::Value, 2> ops{newAlloca.getResult(), memberShape};
                auto newDecl = b.create<hlfir::DeclareOp>(
                    loc, mlir::TypeRange{refTy, refTy},
                    mlir::ValueRange{ops}, attrs);
                memberBase[memName] = newDecl.getResult(0);
                concatMembers.insert(memName);  // tells designate rewriter
                                                // to merge outer + inner indices

                // Strip ``realloc`` from any ``hlfir.assign`` whose
                // LHS targets this member: after flatten the LHS is a
                // static array section, not an allocatable, and the
                // assign op's verifier rejects ``realloc=true`` on
                // non-allocatable LHS.
                stripReallocOnAosMember(decl, memName);

                // Rewrite whole-component assigns (``A(i)%w = ...``)
                // into row-section assigns (``A_w(i, 1:N_i:1) = ...``).
                // ``rewriteAosWholeMemberAssign`` resolves the
                // section bound per-assign: if the parent's outer
                // index is a compile-time constant it looks up the
                // matching allocate's size (handles the jagged case
                // ``A(1)%val(3)`` / ``A(2)%val(4)`` where each row
                // wants its own live size); otherwise falls back to
                // the global cap ``M``.  Must run BEFORE
                // ``rewriteDesignate`` sweeps the parent chain,
                // otherwise the concat path's whole-component branch
                // would replace the LHS with the bare flat 2D ref and
                // the assign would broadcast across all rows.
                (void)sizeUniform;  // kept for potential future
                                    // gating; the per-assign size
                                    // resolution covers both cases.
                rewriteAosWholeMemberAssign(decl, memName,
                                            newDecl.getResult(0), M);

                // Collapse ``fir.load + hlfir.designate (loaded, j)``
                // chains into direct 2-index designates over the new
                // companion.  Must run BEFORE ``rewriteDesignate``
                // sweeps the parent designate chain, otherwise the
                // load + inner designate would be left dangling
                // against the rewritten (now plain ref) parent.
                collapseAosAllocReads(decl, memName, newDecl.getResult(0));

                // Erase the per-instance allocate / freemem chain.
                // Each ``allocate(A(i)%<member>(M))`` lowers to:
                //   %alloc = fir.allocmem !fir.array<?xT>, %M
                //   %box   = fir.embox %alloc(%shape)
                //   fir.store %box to <designate of A(i){memName}>
                // and each ``deallocate`` to ``fir.freemem``.  After
                // synth, the 2D buffer is already there; the chain
                // becomes dead.  Erase the stores first (the box
                // value has no other consumer), then sweep the
                // dangling allocmem / embox / freemem.
                eraseAosAllocDeallocChain(decl, memName);
                continue;
            }

            // Allocatable / pointer array members get a parallel
            // synthesis path (Phase 5a + 5b).  The companion is a
            // top-level allocatable / pointer: ``fir.alloca
            // <box<heap|ptr<array<?xT>>>>`` plus a declare carrying
            // the matching fortran_attr.  Skipped for AoS outers —
            // those go through the Phase 5c-A path above.
            if (isAllocatableArrayMember(memTy) && !outerIsArray) {
                auto allocaTy = memTy;  // box<heap|ptr<array<?xT>>>
                auto refTy    = fir::ReferenceType::get(allocaTy);
                auto newAlloca = b.create<fir::AllocaOp>(loc, allocaTy);

                // Pick the right fortran_attr based on the member's
                // inner indirection: ``fir.heap`` → ALLOCATABLE,
                // ``fir.ptr`` → POINTER.  Downstream queries
                // (extract_vars peel-through, the bridge's
                // allocatable / pointer handling) key on this flag.
                auto box = mlir::cast<fir::BoxType>(memTy);
                bool isPointer = mlir::isa<fir::PointerType>(box.getEleTy());
                auto attrFlag = isPointer
                    ? fir::FortranVariableFlagsEnum::pointer
                    : fir::FortranVariableFlagsEnum::allocatable;

                std::string flatName = baseName + "_" + memName;
                mlir::NamedAttrList attrs;
                attrs.append("uniq_name",
                             mlir::StringAttr::get(ctx, flatName));
                attrs.append("fortran_attrs",
                             fir::FortranVariableFlagsAttr::get(ctx, attrFlag));
                attrs.append(declareSegments(b, /*hasShape=*/false));

                auto newDecl = b.create<hlfir::DeclareOp>(
                    loc, mlir::TypeRange{refTy, refTy},
                    mlir::ValueRange{newAlloca.getResult()}, attrs);
                memberBase[memName] = newDecl.getResult(0);

                // Bug fix (Phase 5a): flang names the per-allocate
                // ``fir.allocmem`` op after the MEMBER's module scope
                // (e.g. ``_QMlibEw.alloc`` for ``module lib :: type t
                // :: real, allocatable :: w(:)``), not after the
                // enclosing struct's local declare scope.  The bridge
                // collects allocate sites by matching
                // ``<declUniqName>.alloc`` (see ``collectAllocSites``
                // in extract_vars.cpp).  Without renaming, the flat
                // ``_QFmainEs_w`` declare won't find ``_QMlibEw.alloc``
                // and the SDFG ends up with a free symbol
                // ``s_w_d0`` that nothing binds.
                //
                // Find every ``fir.allocmem`` reaching this member's
                // designate via embox + store and rename it to
                // ``<flat_uniq_name>.alloc``.  Walk pre-rewrite: at
                // this point the designate of ``%struct{"<memName>"}``
                // still exists; the store of the embox-of-allocmem
                // targets it.
                renameMemberAllocmems(decl, memName, flatName);
                continue;
            }

            auto pointee = companionPointee(outerIsArray, outerShape, memTy);
            if (!pointee) continue;

            auto newAlloca = b.create<fir::AllocaOp>(loc, pointee);

            bool memberIsArray = mlir::isa<fir::SequenceType>(memTy);
            bool concat = outerIsArray && memberIsArray;
            mlir::Type res1Ty = fir::ReferenceType::get(pointee);
            // For the concat case, both result types must be the flat
            // ref — using ``rewrapWith`` would produce the nested form
            // ``ref<array<N x array<M, ...>>>`` which the verifier
            // rejects against the alloca's flat ref.
            mlir::Type res0Ty = concat
                ? res1Ty
                : rewrapWith(decl.getResult(0).getType(), memTy);

            // Pick the shape operand.  Concat members need a fresh
            // ``fir.shape`` over the concatenated extent list — the
            // original ``decl.getShape()`` only carries the outer
            // dim(s).
            mlir::Value memberShape = shape;
            if (concat) {
                auto memSeq = mlir::cast<fir::SequenceType>(memTy);
                llvm::SmallVector<int64_t, 6> exts(outerShape.begin(), outerShape.end());
                bool allStatic = true;
                for (auto d : memSeq.getShape()) {
                    if (d == fir::SequenceType::getUnknownExtent()) {
                        allStatic = false;
                        break;
                    }
                    exts.push_back(d);
                }
                if (!allStatic) continue;
                memberShape = emitStaticShape(b, loc, exts);
                concatMembers.insert(memName);
            } else if (!outerIsArray && memberIsArray) {
                // Scalar outer + array member: the new declare needs a
                // shape over the MEMBER's own extents.  The outer
                // declare's ``shape`` is null for a plain
                // ``type(t) :: x``, so without a fresh ``fir.shape``
                // the synthesised ``hlfir.declare`` for
                // ``x_arr_field`` would have a raw address base AND
                // no shape operand — which the verifier rejects with
                // "must have a shape operand that is a shape or
                // shapeshift".
                auto memSeq = mlir::cast<fir::SequenceType>(memTy);
                llvm::SmallVector<int64_t, 4> exts;
                bool allStatic = true;
                for (auto d : memSeq.getShape()) {
                    if (d == fir::SequenceType::getUnknownExtent()) {
                        allStatic = false;
                        break;
                    }
                    exts.push_back(d);
                }
                if (!allStatic) continue;
                memberShape = emitStaticShape(b, loc, exts);
            }

            llvm::SmallVector<mlir::Value, 2> operands;
            operands.push_back(newAlloca);
            if (memberShape) operands.push_back(memberShape);

            mlir::NamedAttrList attrs;
            attrs.append("uniq_name",
                         mlir::StringAttr::get(ctx,
                                               baseName + "_" + memName));
            attrs.append(declareSegments(b, /*hasShape=*/memberShape != nullptr));

            auto newDecl = b.create<hlfir::DeclareOp>(
                loc, mlir::TypeRange{res0Ty, res1Ty},
                mlir::ValueRange(operands), attrs);

            memberBase[memName] = newDecl.getResult(0);
        }

        // Collect designates to rewrite.  For non-concat members the
        // direct user is the component-designate.  For concat members
        // the direct user is an INDEXED designate (no component) on
        // the outer array; the actual component-designate is its
        // child.  We also walk transparently through:
        //   * ``hlfir.declare`` aliases — inlined-callee dummy declares
        //     that share the outer's storage.
        //   * ``fir.embox`` / ``fir.convert`` chains — the wrapping
        //     flang inserts when an inlined callee takes a
        //     ``CLASS(t)`` (or assumed-shape ``TYPE(t)``) dummy.  The
        //     outer concrete declare gets emboxed to ``fir.box<t>``,
        //     converted to ``fir.class<t>``, and the inlined declare
        //     is over the converted value.
        llvm::SmallVector<hlfir::DesignateOp, 16> designates;
        llvm::SmallVector<hlfir::DeclareOp, 8> aliasDecls;
        llvm::SmallVector<mlir::Operation*, 4> wrapperOps;
        std::function<void(mlir::Value)> collectFrom = [&](mlir::Value root) {
            for (auto *u : root.getUsers()) {
                if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u)) {
                    bool hasComponent = false;
                    for (auto nm : {"component_name", "component"})
                        if (dg->getAttrOfType<mlir::StringAttr>(nm)) {
                            hasComponent = true;
                            break;
                        }
                    if (hasComponent) {
                        designates.push_back(dg);
                    } else {
                        for (auto *cu : dg.getResult().getUsers())
                            if (auto cdg = mlir::dyn_cast<hlfir::DesignateOp>(cu))
                                designates.push_back(cdg);
                    }
                } else if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
                    aliasDecls.push_back(dc);
                    collectFrom(dc.getResult(0));
                    if (dc.getResult(1) != dc.getResult(0))
                        collectFrom(dc.getResult(1));
                } else if (mlir::isa<fir::EmboxOp>(u)
                           || mlir::isa<fir::ConvertOp>(u)) {
                    wrapperOps.push_back(u);
                    for (auto v : u->getResults()) collectFrom(v);
                }
            }
        };
        collectFrom(decl.getResult(0));
        for (auto dg : designates) rewriteDesignate(dg, memberBase, concatMembers);
        for (auto a : aliasDecls)
            if (a.getResult(0).use_empty() && a.getResult(1).use_empty())
                a.erase();
        // Sweep wrapper ops in REVERSE so each step's only users
        // (the next op down the chain) are already gone before we
        // try to erase its source.
        for (auto *w : llvm::reverse(wrapperOps))
            if (llvm::all_of(w->getResults(),
                             [](mlir::Value v) { return v.use_empty(); }))
                w->erase();

        if (decl.getResult(0).use_empty() && decl.getResult(1).use_empty()) {
            auto *allocaOp = decl.getMemref().getDefiningOp();
            decl.erase();
            if (allocaOp && allocaOp->use_empty()) allocaOp->erase();
        }
    }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createFlattenStructsPass() {
    return std::make_unique<FlattenStructsPass>();
}

}  // namespace hlfir_bridge
