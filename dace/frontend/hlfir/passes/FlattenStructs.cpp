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
// compile-time constant.  Dynamic-extent struct members (Fortran
// ``allocatable`` / ``pointer`` components) are out of scope:
// they need a fresh SDFG symbol per padded dim plus runtime
// max-computation in the bindings wrapper.  ``isLocallyFlattenable``
// and the dummy-arg gate at ``planAndReplaceStructArgs`` filter
// these out so the bridge sees the unflattened struct and emits a
// loud-failure throw at ``extract_vars`` (``fir.RecordType``
// reaches a declare).
//
// Things this pass deliberately does NOT do
// -----------------------------------------
// * Truly virtual polymorphic dispatch — handled separately by
//   ``fir-polymorphic-op`` (devirtualises) and
//   ``hlfir-reject-polymorphism`` (loud-fails on residuals).  This
//   pass peels ``fir.class<T>`` like ``fir.box<T>`` so monomorphic
//   CLASS receivers flatten through the same path as TYPE.
// * Allocatable / pointer members — out of scope.
// * Cross-boundary AoS with allocatable members — combination of
//   allocatable support + bindings padding-to-max work.
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
    return t.isF32() || t.isF64() || t.isInteger(32) || t.isInteger(64);
}

/// Scalar or array-of-scalar.  Used both for struct members (when the
/// enclosing struct is a scalar) and for the final companion pointee type.
static bool isFlatMemberType(mlir::Type t) {
    if (isSimpleScalar(t)) return true;
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(t))
        return isSimpleScalar(seq.getEleTy());
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

static bool collectFlatLeaves(fir::RecordType rec,
                              llvm::SmallVectorImpl<std::string> &prefix,
                              llvm::SmallVectorImpl<FlatLeaf> &out,
                              int depth = 0) {
    if (depth > kFlattenMaxDepth) return false;
    for (auto &pair : rec.getTypeList()) {
        prefix.push_back(pair.first);
        if (isFlatMemberType(pair.second)) {
            FlatLeaf leaf;
            leaf.path.assign(prefix.begin(), prefix.end());
            leaf.leafTy = pair.second;
            out.push_back(std::move(leaf));
        } else if (auto innerRec = mlir::dyn_cast<fir::RecordType>(pair.second)) {
            if (!collectFlatLeaves(innerRec, prefix, out, depth + 1)) {
                prefix.pop_back();
                return false;
            }
        } else {
            // Member is e.g. an array of struct, or an allocatable —
            // not flattenable here.  Bail out so the pass leaves the
            // struct untouched and the loud-failure throw in
            // extract_vars points at the right gap.
            prefix.pop_back();
            return false;
        }
        prefix.pop_back();
    }
    return true;
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
static std::string designateChainPath(hlfir::DesignateOp leaf,
                                      hlfir::DesignateOp &outAnchor) {
    llvm::SmallVector<std::string, 4> parts;
    hlfir::DesignateOp cur = leaf;
    for (int i = 0; i < kFlattenMaxDepth && cur; ++i) {
        mlir::StringAttr compAttr;
        for (auto nm : {"component_name", "component"})
            if (auto a = cur->getAttrOfType<mlir::StringAttr>(nm)) {
                compAttr = a;
                break;
            }
        if (!compAttr) {
            // Reached a non-component designate (a subscripted access
            // ``a(i,j)``) — that's only valid as the LEAF of the chain,
            // i.e. the very first call.  Stop here.
            break;
        }
        parts.push_back(compAttr.getValue().str());
        outAnchor = cur;
        // Walk to the parent; if it's another designate keep going.
        auto memref = cur.getMemref();
        cur = mlir::dyn_cast_or_null<hlfir::DesignateOp>(memref.getDefiningOp());
    }
    if (parts.empty()) return "";
    // Reverse to outermost-first order matching FlatLeaf.path.
    std::reverse(parts.begin(), parts.end());
    std::string joined;
    for (unsigned i = 0; i < parts.size(); ++i) {
        if (i) joined += "_";
        joined += parts[i];
    }
    return joined;
}

/// Rewrite a multi-level ``hlfir.designate`` chain ending at ``leaf``
/// (e.g. ``designate{"x"}.designate{"inner"} %o`` for ``o%inner%x``)
/// to read directly from the path-flattened declare named in
/// ``leafBase``.  ``leaf`` may carry indices (``a(i,j)``) — those are
/// preserved.  Returns true if the rewrite fired.
static bool rewriteDesignateChain(
    hlfir::DesignateOp leaf,
    const llvm::StringMap<mlir::Value> &leafBase) {

    hlfir::DesignateOp anchor;
    std::string path = designateChainPath(leaf, anchor);
    if (path.empty()) return false;
    auto it = leafBase.find(path);
    if (it == leafBase.end()) return false;
    auto newBase = it->second;

    // ``leaf`` is the INNERMOST (component or component-with-indices) op.
    // Its result is what the rest of the IR consumes.
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
    void recordStructArgEntry(hlfir::DeclareOp argDecl, fir::RecordType rec,
                              llvm::StringRef intentStr,
                              bool outerIsArray = false,
                              llvm::ArrayRef<int64_t> outerShape = {}) {
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

        for (auto &pair : rec.getTypeList()) {
            llvm::StringRef memName = pair.first;
            mlir::Type memTy = pair.second;
            int memRank = memberRank(memTy);
            int totalRank = (int)outerRank + memRank;
            if (totalRank > maxRank) maxRank = totalRank;

            std::string flat = (outerName + "_" + memName).str();
            flatNames.push_back(mkStr(flat));

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

        // Shape exprs for the recipe.  For AoS dummy args the leading
        // ``outerRank`` dims come from the outer struct array itself
        // (``size(outer, dim=i)``); the remaining dims come from
        // ``size(outer(1)%<first_member>, dim=j)``.  For scalar-outer
        // structs all dims are member dims.
        llvm::SmallVector<mlir::Attribute, 4> shapeExprs;
        if (maxRank > 0 && !rec.getTypeList().empty()) {
            llvm::StringRef first = rec.getTypeList()[0].first;
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
                std::string s = ("size(" + sampleOuter + "%" + first.str()
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
                auto newOp = b.create<hlfir::DesignateOp>(
                    loc,
                    /*resultType0=*/refFieldTy,
                    /*memref=*/cur,
                    /*component=*/componentAttr,
                    /*component_shape=*/mlir::Value{},
                    /*indices=*/mlir::ValueRange{},
                    /*is_triplet=*/mlir::DenseBoolArrayAttr{},
                    /*substring=*/mlir::ValueRange{},
                    /*complex_part=*/mlir::BoolAttr{},
                    /*shape=*/mlir::Value{},
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
            b.create<hlfir::AssignOp>(loc, rhsLeaf, lhsLeaf);
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
            else if (!allMembersFlattenable(rec, outerIsArray))
                continue;
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
            // Record the entry BEFORE the declare is erased.  If
            // ``replaceStructArg`` bails out (dangling users on the
            // old declare), the entry still describes the intended
            // recipe — but the SDFG won't carry the flat members so
            // the emitter will just skip it downstream.
            std::string intentStr = extractIntent(p.argDecl.getFortranAttrs());
            recordStructArgEntry(p.argDecl, p.rec, intentStr,
                                 p.outerIsArray, p.outerShape);
            replaceStructArg(func, idx, p.argDecl, p.rec,
                             p.outerIsArray, p.outerShape);
        }
        return true;
    }

    // -------------------------------------------------------------------
    // Struct dummy arguments
    // -------------------------------------------------------------------

    void replaceStructArg(mlir::func::FuncOp func, unsigned argIdx,
                          hlfir::DeclareOp argDecl, fir::RecordType rec,
                          bool outerIsArray = false,
                          llvm::ArrayRef<int64_t> outerShape = {}) {
        auto &block = func.front();
        auto loc = argDecl.getLoc();
        auto *ctx = func.getContext();
        auto baseName = argDecl.getUniqName().str();

        // Insert new block args right after the old one so the argument order
        // tracks the original member order.  Insertion shifts indices >= pos
        // by 1, so we insert sequentially at argIdx+1, argIdx+2, …
        llvm::StringMap<mlir::Value> memberBase;
        llvm::StringSet<> concatMembers;
        unsigned memberCount = 0;
        for (auto &pair : rec.getTypeList()) {
            auto memName = pair.first;
            auto memTy   = pair.second;
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
        if (allMembersFlattenable(rec, outerIsArray)) return true;
        // Nested-struct fallback: no outer array wrapper.  Walk the
        // path-leaf set and check every leaf flat.
        if (outerIsArray) return false;
        llvm::SmallVector<std::string, 4> prefix;
        llvm::SmallVector<FlatLeaf, 8> leaves;
        return collectFlatLeaves(rec, prefix, leaves);
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
        bool nested = !allMembersFlattenable(rec, outerIsArray);
        if (nested) {
            if (outerIsArray) return;  // array-of-nested unsupported
            llvm::SmallVector<std::string, 4> prefix;
            llvm::SmallVector<FlatLeaf, 8> leaves;
            if (!collectFlatLeaves(rec, prefix, leaves)) return;

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

            // Walk the original declare's users (which are component
            // designates) and follow the chain to the LEAF designate.
            // We can't simply walk decl.getResult(0).getUsers because
            // those are the FIRST-level designates; we need the
            // innermost one.  Approach: find every hlfir.designate in
            // the function whose ultimate ancestor (through the chain)
            // is ``decl`` and rewrite the chain.
            llvm::SmallVector<hlfir::DesignateOp, 16> chainLeaves;
            decl->getParentOfType<mlir::func::FuncOp>().walk(
                [&](hlfir::DesignateOp dg) {
                    // A leaf is a designate whose users are NOT themselves
                    // hlfir.designate ops (otherwise we'd rewrite a parent
                    // and lose the chain).
                    bool hasDesignateUser = false;
                    for (auto *u : dg.getResult().getUsers())
                        if (mlir::isa<hlfir::DesignateOp>(u)) {
                            hasDesignateUser = true;
                            break;
                        }
                    if (hasDesignateUser) return;
                    // Walk the chain to verify it ends at decl.
                    hlfir::DesignateOp cur = dg;
                    for (int i = 0; i < kFlattenMaxDepth && cur; ++i) {
                        auto memref = cur.getMemref();
                        if (memref == decl.getResult(0)) {
                            chainLeaves.push_back(dg);
                            return;
                        }
                        cur = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
                            memref.getDefiningOp());
                    }
                });
            for (auto dg : chainLeaves)
                rewriteDesignateChain(dg, leafBase);

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
