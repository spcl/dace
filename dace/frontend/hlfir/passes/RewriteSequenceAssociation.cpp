// ============================================================================
// RewriteSequenceAssociation.cpp  --  collapse Fortran's sequence-association
// adapter (scalar element passed to an array dummy) into an explicit slice.
// ============================================================================
//
// The Fortran 2003 standard (section12.4.1.5) lets a caller pass a single
// element of an array where the formal expects an explicit-shape or
// assumed-size array  --  the formal then sees ``N`` consecutive elements
// starting at the given element.  Flang lowers this with a deterministic
// IR shape:
//
//   %elt   = hlfir.designate %parent (%idx)            : !fir.ref<T>
//   %arr   = fir.convert     %elt                      : !fir.ref<T> ->
//   !fir.ref<!fir.array<?xT>> %decl  = hlfir.declare   %arr (%shape) {...} :
//   (!fir.ref<!fir.array<?xT>>, !fir.shape<1>) -> ...
//   ... uses of %decl#0 / #1 ...
//
// The ``fir.convert`` from a rank-0 ref to a rank-1 ref is the
// sequence-association marker  --  flang inserts it ONLY for this case;
// other rank changes lower through ``fir.embox`` / ``fir.rebox`` /
// ``hlfir.designate`` triplet sections, never through
// ``fir.convert``.  The scalar source's defining op is always an
// element-form ``hlfir.designate`` (no triplet, no shape operand).
//
// Detection signal is therefore deterministic, not heuristic.  See
// the references at the bottom of this comment for the IR shape and
// language semantics.
//
// Rewrite (this pass)
// -------------------
// Replace the converted view with a real section designate of
// ``%parent`` covering ``[idx, idx + N - 1]``:
//
//   %lo    = arith.constant <idx>                           : index
//   %hi    = arith.addi    %lo, <N - 1>                     : index
//   %sec   = hlfir.designate %parent (lo:hi:1) shape ...    :
//   !fir.box<!fir.array<NxT>>
//
// Every use of ``%decl#0`` / ``%decl#1`` is rewritten to the section's
// result.  The bridge's existing per-element designate / box-aware
// reduction lowering then handles the section like any other sliced
// view.  The declare, convert, and element-form designate become dead
// and are erased.
//
// Supported variants
// ------------------
// The Fortran-language patterns below all surface as the same IR
// adapter shape (rank-0 -> rank-1 ``fir.convert`` of an element designate)
//  --  the variants differ only in how the formal's *extent* and the
// *parent rank* travel into the IR.  Each row corresponds to a unit
// test in ``tests/hlfir/rewrite_sequence_association_test.py``.
//
//   1. Constant literal extent: ``f(d(11), 5)``.
//      Extent reaches the inlined callee through flang's
//      ``__assoc_scalar`` adapter (alloca + single ``fir.store`` + load).
//      ``traceStoredConstant`` recovers the literal.
//
//   2. Constant arithmetic extent: ``f(d(11), 2*K + 1)`` with ``K``
//      itself a literal.  Extent expression folds through ``arith.muli``
//      / ``addi`` / ``subi`` / ``divsi`` after recursive trace.
//
//   3. Module / parameter constant extent: ``integer, parameter ::
//      NMAX = 50`` then ``f(d(11), NMAX)``.  Folds through
//      ``fir.address_of @global`` + the global's ``fir.global``
//      initialiser body.
//
//   4. Runtime-symbolic extent: ``f(d(11), sz)`` with ``sz`` computed at
//      runtime, OR extent pulled from another array's descriptor via
//      ``fir.box_dims``.  Cannot fold to a constant; we fall back to a
//      runtime-extent section ``box<array<?xT>>`` whose triplet upper
//      bound is the original extent value (``hi = lo + extent - 1``).
//      The bridge's section-aware reductions handle the runtime hi via
//      its existing buildIndexExpr machinery.
//
//   5. Multi-dim parent (the QE / BLAS pattern): ``ZGEMM(d(1, j), ldd,
//      ...)`` where ``d`` is rank-2.  Source designate has ``K`` scalar
//      indices.  We emit a section that places a triplet on the FIRST
//      dimension and passes the remaining indices through as scalars,
//      yielding a rank-1 view of length ``ldd`` along the column-major
//      contiguous dimension (e.g. ``d(1:ldd, j)`` for ``ZGEMM(d(1,j),
//      ldd, ...)``).
//
// Pipeline placement
// ------------------
// Runs AFTER ``hlfir-inline-all`` (so the inlined callee body's
// declare-of-converted-ref is visible) and BEFORE
// ``hlfir-flatten-structs`` (so the section view feeds into the
// usual designate-rewrite path).  The pass is a no-op on programs
// that don't use sequence association, so leaving it in the default
// pipeline costs nothing on programs that don't trigger it.
//
// Co-dependent change: ``bridge/ast/dispatch.cpp`` peels through
// ``fir.convert`` chains BEFORE pattern-matching ``hlfir.sum`` /
// ``minval`` / ``any`` / ``all`` / etc. against an ``hlfir.designate``.
// After this pass fires, the section it emits is sometimes wrapped in
// a box-shape canonicalisation convert (``box<array<NxT>>`` ->
// ``box<array<?xT>>``) before the reduction op consumes it.  Without
// the peel, the bridge would fall back to the whole-array reduce path
// instead of the section-aware one and produce wrong results.
//
// Caveats / NOT handled
// ---------------------
// * Insertion point is the line *after* ``formalDecl``.  This is
//   required for the runtime-symbolic case (variant 4): the extent
//   Value is defined inside the inlined callee and dominates only
//   uses below the declare.  The constant-N path inserts there too
//   for symmetry  --  constants don't have dominance issues anyway.
//
// * Out-of-bounds slices are NOT validated.  A program like
//   ``f(d(2, j), 10)`` with ``d(8, N)`` requests 10 elements starting
//   at ``d(2,j)`` which crosses the column boundary; we emit the
//   syntactic ``d(2:11, j)`` even though dim-1 only goes up to 8.
//   Sequence association is column-major-flatten by definition (and
//   under the strict-no-aliasing assumption the bridge already
//   makes); the user's program is the source of truth for whether
//   the slice is well-defined.  We don't try to multi-column unfold.
//
// * The pass walks the WHOLE module (all ``fir.convert`` ops in every
//   function).  Multiple sequence-association sites all rewrite
//   independently in a single pass.  Re-running the pass is a no-op:
//   the rewritten section designate is in triplet form, which
//   ``matchSeqAdapter`` rejects up-front.
//
// * CHARACTER substring sequence association  --  ``call sub(s(i:j))``
//   passing a substring slice  --  is a DIFFERENT IR shape (uses the
//   ``substring`` operand on ``hlfir.designate``).  We don't handle
//   it; it surfaces as the un-rewritten convert and the bridge's
//   downstream gate flags it.
//
// * Stride > 1 at the call site is impossible  --  Fortran call sites
//   can only pass a single element ``d(idx)``, not a strided
//   reference.  Triplet-form actuals (``d(lo:hi:s)``) are NOT
//   sequence association  --  they're real array sections that flang
//   lowers through ``hlfir.designate`` triplet without a convert.
//   Handled by the ``isTriplet -> reject`` early-out in
//   ``matchSeqAdapter``.
//
// * ``N == 0`` (constant): pass bails (returns without rewriting) so
//   no zero-extent section gets emitted.  ``N == 0`` (symbolic): we
//   emit the section unchanged; a runtime extent of 0 yields an
//   empty triplet which the bridge's section-reduce loop handles
//   natively (zero-iteration accumulator).
//
// * The pass requires a ``fir.shape`` defining op on the formal
//   declare's shape operand.  Programs whose extent comes from a
//   non-``fir.shape`` source (extremely rare  --  typically the
//   canonicaliser has already produced ``fir.shape``) are left
//   alone.
//
// References
// ----------
// * HLFIR design overview, ``hlfir.designate`` / ``hlfir.associate``
//   semantics  --  https://flang.llvm.org/docs/HighLevelFIR.html
// * Current HLFIR spec (main)  --
// https://github.com/llvm/llvm-project/blob/main/flang/docs/HighLevelFIR.md
// * Variable / Expression value concepts (LLVM 18.1.0)  --
//   https://releases.llvm.org/18.1.0/tools/flang/docs/HighLevelFIR.html
// * Fortran Discourse: explicit-shape and assumed-size arrays + sequence
// association  --
//   https://fortran-lang.discourse.group/t/explicit-shape-and-assumed-size-arrays-and-sequence-association/2783
// * FIR Language Reference (``fir.convert`` operand-shape rules)  --
//   https://flang.llvm.org/docs/FIRLangRef.html
// * Representation of Fortran function calls (LLVM 13.0.0)  --
//   https://releases.llvm.org/13.0.0/tools/flang/docs/Calls.html
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

// Forward decl: mutual recursion between traceConstIndex and
// traceStoredConstant.
static std::optional<int64_t> traceConstIndex(mlir::Value v);

/// Recover a constant value stored into an alloca.  Used to fold
/// flang's ``__assoc_scalar`` adapter (literal-passed-by-reference at a
/// call site): the literal lands in an alloca via a single
/// ``fir.store`` and is read back through ``fir.load`` of the formal
/// dummy's declare.  Walks through any number of ``hlfir.declare``
/// re-declares aliasing the same underlying alloca.  Returns nullopt
/// if the alloca has multiple stores (genuinely runtime-varying
/// value) or the stored value isn't itself a foldable constant.
static std::optional<int64_t> traceStoredConstant(mlir::Value memref) {
  // Walk through hlfir.declare wrappers down to the underlying alloca.
  mlir::Value root = memref;
  for (int i = 0; i < 8; ++i) {
    auto *d = root.getDefiningOp();
    if (!d) return std::nullopt;
    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      root = decl.getMemref();
      continue;
    }
    if (mlir::isa<fir::AllocaOp>(d) || mlir::isa<fir::AllocMemOp>(d)) break;
    return std::nullopt;
  }
  // Collect all alias views (the alloca itself plus any hlfir.declare
  // re-declares of it) and look for a single fir.store among their users.
  llvm::SmallVector<mlir::Value, 4> aliases{root};
  for (auto *u : root.getUsers()) {
    if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
      aliases.push_back(d.getResult(0));
      aliases.push_back(d.getResult(1));
      for (auto *u2 : d.getResult(0).getUsers()) {
        if (auto d2 = mlir::dyn_cast<hlfir::DeclareOp>(u2)) {
          aliases.push_back(d2.getResult(0));
          aliases.push_back(d2.getResult(1));
        }
      }
    }
  }
  fir::StoreOp uniqueStore;
  int storeCount = 0;
  for (auto a : aliases) {
    for (auto *u : a.getUsers()) {
      if (auto s = mlir::dyn_cast<fir::StoreOp>(u)) {
        ++storeCount;
        uniqueStore = s;
      }
    }
  }
  if (storeCount != 1 || !uniqueStore) return std::nullopt;
  return traceConstIndex(uniqueStore.getValue());
}

/// Walk a ``fir.global`` to its initialiser constant.  Used to fold
/// ``integer, parameter :: NMAX = 50`` references  --  flang lowers
/// these to a ``fir.address_of @<global>`` + ``fir.load`` chain.  The
/// global's body is a tiny region whose terminating ``fir.has_value``
/// op carries the initial value.
static std::optional<int64_t> traceGlobalInitialiser(mlir::SymbolRefAttr name,
                                                     mlir::Operation *anchor) {
  auto module = anchor->getParentOfType<mlir::ModuleOp>();
  if (!module) return std::nullopt;
  auto sym = module.lookupSymbol(name.getLeafReference());
  auto global = mlir::dyn_cast_or_null<fir::GlobalOp>(sym);
  if (!global) return std::nullopt;
  if (global.getRegion().empty()) return std::nullopt;
  for (auto &op : global.getRegion().front()) {
    if (auto hv = mlir::dyn_cast<fir::HasValueOp>(op))
      return traceConstIndex(hv.getResval());
  }
  return std::nullopt;
}

/// Trace an MLIR Value back to a constant ``index`` value if possible.
/// Walks every fold-to-constant pattern flang emits for the formal's
/// extent operand on a sequence-association adapter:
///
///   * ``arith.constant`` (terminal).
///   * ``fir.convert`` (transparent  --  same value, different ABI type).
///   * ``arith.addi`` / ``subi`` / ``muli`` / ``divsi`` of constants  --
///     covers ``real :: arr(2*K+1)`` etc. when the operands fold.
///   * ``arith.select sgt(x,0), x, 0``  --  flang's nonneg-extent clamp
///     wrapping every dynamic-shape operand.
///   * ``fir.load`` of an alloca with a single dominating ``fir.store``
///     of a foldable value  --  the ``__assoc_scalar`` literal adapter
///     and the ``parameter`` constant address-of pattern.
///   * ``fir.address_of @global`` followed by ``fir.load``  --  module-
///     level ``parameter`` constants whose initialiser is itself a
///     constant.  The initialiser is read from the global's
///     ``fir.has_value`` terminator.
///
/// Returns nullopt for every other op (most importantly ``fir.load`` of
/// a multi-store variable, ``fir.box_dims``, or any reference that's
/// genuinely runtime-varying).  The caller falls back to the symbolic
/// rewrite in that case.
static std::optional<int64_t> traceConstIndex(mlir::Value v) {
  for (int i = 0; i < 32 && v; ++i) {
    auto *def = v.getDefiningOp();
    if (!def) return std::nullopt;
    if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
        return ia.getInt();
      return std::nullopt;
    }
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      v = cv.getValue();
      continue;
    }
    if (auto add = mlir::dyn_cast<mlir::arith::AddIOp>(def)) {
      auto a = traceConstIndex(add.getLhs());
      auto b2 = traceConstIndex(add.getRhs());
      if (a && b2) return *a + *b2;
      return std::nullopt;
    }
    if (auto sub = mlir::dyn_cast<mlir::arith::SubIOp>(def)) {
      auto a = traceConstIndex(sub.getLhs());
      auto b2 = traceConstIndex(sub.getRhs());
      if (a && b2) return *a - *b2;
      return std::nullopt;
    }
    if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(def)) {
      auto a = traceConstIndex(mul.getLhs());
      auto b2 = traceConstIndex(mul.getRhs());
      if (a && b2) return *a * *b2;
      return std::nullopt;
    }
    if (auto div = mlir::dyn_cast<mlir::arith::DivSIOp>(def)) {
      auto a = traceConstIndex(div.getLhs());
      auto b2 = traceConstIndex(div.getRhs());
      if (a && b2 && *b2 != 0) return *a / *b2;
      return std::nullopt;
    }
    if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
      // Both branches agree -> done.
      auto t = traceConstIndex(sel.getTrueValue());
      auto f = traceConstIndex(sel.getFalseValue());
      if (t && f && *t == *f) return *t;
      // Otherwise resolve via the condition: ``select sgt(x,0), x, 0``
      // (flang's nonneg-extent clamp) folds when x is a known constant.
      if (auto cmp = mlir::dyn_cast_or_null<mlir::arith::CmpIOp>(
              sel.getCondition().getDefiningOp())) {
        auto lhs = traceConstIndex(cmp.getLhs());
        auto rhs = traceConstIndex(cmp.getRhs());
        if (!lhs || !rhs) return std::nullopt;
        using P = mlir::arith::CmpIPredicate;
        bool taken = false;
        switch (cmp.getPredicate()) {
          case P::eq:
            taken = (*lhs == *rhs);
            break;
          case P::ne:
            taken = (*lhs != *rhs);
            break;
          case P::slt:
            taken = (*lhs < *rhs);
            break;
          case P::sle:
            taken = (*lhs <= *rhs);
            break;
          case P::sgt:
            taken = (*lhs > *rhs);
            break;
          case P::sge:
            taken = (*lhs >= *rhs);
            break;
          default:
            return std::nullopt;
        }
        return taken ? t : f;
      }
      return std::nullopt;
    }
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
      // Try the ``__assoc_scalar`` (alloca + single store) path first.
      if (auto fromStore = traceStoredConstant(ld.getMemref()))
        return fromStore;
      // Fall through to the address_of @global path.
      mlir::Value mem = ld.getMemref();
      for (int j = 0; j < 8 && mem; ++j) {
        auto *d = mem.getDefiningOp();
        if (!d) return std::nullopt;
        if (auto ao = mlir::dyn_cast<fir::AddrOfOp>(d))
          return traceGlobalInitialiser(ao.getSymbol(), def);
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
          mem = cv.getValue();
          continue;
        }
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
          mem = decl.getMemref();
          continue;
        }
        return std::nullopt;
      }
      return std::nullopt;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

/// Recognise a SeqAssociation adapter: ``fir.convert`` from
/// ``fir.ref<T>`` (rank 0) to ``fir.ref<fir.array<...xT>>`` (rank 1+),
/// where the source is an element-form ``hlfir.designate``.  Returns
/// the source designate on match, null otherwise.
static hlfir::DesignateOp matchSeqAdapter(fir::ConvertOp conv) {
  auto srcTy = conv.getValue().getType();
  auto dstTy = conv.getResult().getType();
  auto srcRef = mlir::dyn_cast<fir::ReferenceType>(srcTy);
  auto dstRef = mlir::dyn_cast<fir::ReferenceType>(dstTy);
  if (!srcRef || !dstRef) return {};
  auto srcEle = srcRef.getEleTy();
  auto dstEle = dstRef.getEleTy();
  auto dstSeq = mlir::dyn_cast<fir::SequenceType>(dstEle);
  if (!dstSeq) return {};
  if (mlir::isa<fir::SequenceType>(srcEle)) return {};
  if (srcEle != dstSeq.getEleTy()) return {};
  auto dg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
      conv.getValue().getDefiningOp());
  if (!dg) return {};
  // Must be element form (any-triplet -> already a section, skip).
  for (bool b : dg.getIsTriplet())
    if (b) return {};
  if (dg.getIndices().empty()) return {};
  return dg;
}

struct RewriteSequenceAssociationPass
    : public mlir::PassWrapper<RewriteSequenceAssociationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RewriteSequenceAssociationPass)

  llvm::StringRef getArgument() const final {
    return "hlfir-rewrite-sequence-association";
  }
  llvm::StringRef getDescription() const final {
    return "Replace Fortran sequence-association adapters "
           "(scalar element passed to array dummy) with an "
           "explicit section designate of the parent array.";
  }

  void runOnOperation() override {
    llvm::SmallVector<fir::ConvertOp, 8> targets;
    getOperation().walk([&](fir::ConvertOp c) {
      if (matchSeqAdapter(c)) targets.push_back(c);
    });
    for (auto c : targets) rewrite(c);
  }

 private:
  void rewrite(fir::ConvertOp conv) {
    auto eltDg = matchSeqAdapter(conv);
    if (!eltDg) return;

    // Step 1: find the hlfir.declare that consumes the convert.
    // There should be exactly one (the formal's declare).
    hlfir::DeclareOp formalDecl;
    for (auto *u : conv.getResult().getUsers()) {
      if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
        formalDecl = d;
        break;
      }
    }
    if (!formalDecl) return;

    // Step 2: pull the formal's declared length from the declare's
    // shape.  Folds to a compile-time constant when possible (the
    // common case  --  literal extent, ``parameter`` constant,
    // constant arithmetic); otherwise we fall back to the runtime
    // extent value and emit a ``box<array<?xT>>`` section.
    auto shapeVal = formalDecl.getShape();
    if (!shapeVal) return;
    auto shapeOp =
        mlir::dyn_cast_or_null<fir::ShapeOp>(shapeVal.getDefiningOp());
    if (!shapeOp || shapeOp.getExtents().size() != 1) return;
    mlir::Value extentVal = shapeOp.getExtents()[0];
    auto Nopt = traceConstIndex(extentVal);
    if (Nopt && *Nopt <= 0) return;

    // Step 3: pull the element start index from the source
    // designate.  ``hlfir.designate`` for an element form has one
    // scalar operand per parent dimension.  For a rank-K parent
    // (the QE / BLAS pattern: ``f(d(1, j), ldd)`` with ``d``
    // rank-2) we place the triplet on the FIRST dimension and
    // pass the remaining indices through unchanged  --  Fortran's
    // column-major contiguity makes that the dimension the formal
    // strides over.
    if (eltDg.getIndices().empty()) return;
    if (!eltDg.getIsTriplet().empty())
      for (bool t : eltDg.getIsTriplet())
        if (t) return;
    mlir::Value lo = eltDg.getIndices()[0];

    // Step 4: build the section designate over the parent array.
    // Insert AFTER the formal declare  --  for the runtime-symbolic
    // path the extent Value is only defined inside the inlined
    // callee scope (typically as ``arith.select sgt(load(dz), 0),
    // load(dz), 0``) which dominates the declare but not the
    // ``fir.convert`` call site.  Inserting after the declare keeps
    // both paths legal under SSA dominance.
    mlir::OpBuilder b(formalDecl->getNextNode());
    auto loc = conv.getLoc();
    auto idxTy = b.getIndexType();

    auto toIndex = [&](mlir::Value v) {
      if (v.getType() == idxTy) return v;
      return b.create<fir::ConvertOp>(loc, idxTy, v).getResult();
    };
    mlir::Value loIdx = toIndex(lo);
    auto c1 = b.create<mlir::arith::ConstantOp>(loc, idxTy, b.getIndexAttr(1));

    // Decide constant-N vs runtime-N variant.  The runtime variant
    // emits a ``box<array<?xT>>`` and computes ``hi`` from the
    // original extent value at runtime.
    mlir::Value hi;
    mlir::Value sectionShape;
    fir::SequenceType sectionSeqTy;
    auto convDstSeq = mlir::cast<fir::SequenceType>(
        mlir::cast<fir::ReferenceType>(conv.getResult().getType()).getEleTy());
    auto eleTy = convDstSeq.getEleTy();
    if (Nopt) {
      int64_t N = *Nopt;
      auto cN =
          b.create<mlir::arith::ConstantOp>(loc, idxTy, b.getIndexAttr(N));
      auto cNm1 =
          b.create<mlir::arith::ConstantOp>(loc, idxTy, b.getIndexAttr(N - 1));
      hi = b.create<mlir::arith::AddIOp>(loc, loIdx, cNm1).getResult();
      sectionShape =
          b.create<fir::ShapeOp>(loc, mlir::ValueRange{cN.getResult()})
              .getResult();
      sectionSeqTy = fir::SequenceType::get({N}, eleTy);
    } else {
      // Runtime-extent path.  ``hi = lo + extent - 1`` with the
      // extent value carried straight from the formal declare.
      mlir::Value extentIdx = toIndex(extentVal);
      auto extentMinus1 =
          b.create<mlir::arith::SubIOp>(loc, extentIdx, c1.getResult())
              .getResult();
      hi = b.create<mlir::arith::AddIOp>(loc, loIdx, extentMinus1).getResult();
      sectionShape =
          b.create<fir::ShapeOp>(loc, mlir::ValueRange{extentIdx}).getResult();
      sectionSeqTy = fir::SequenceType::get(
          {fir::SequenceType::getUnknownExtent()}, eleTy);
    }
    auto boxTy = fir::BoxType::get(sectionSeqTy);

    // Pass-through scalar indices for parent dims 2..K.  Empty for
    // rank-1 parents (the textbook sequence-association case);
    // populated for the QE pattern's rank-2+ parent.
    auto parent = eltDg.getMemref();
    llvm::SmallVector<mlir::Value, 6> tripletOps{loIdx, hi, c1.getResult()};
    llvm::SmallVector<bool, 6> isTriplet{true};
    for (size_t i = 1; i < eltDg.getIndices().size(); ++i) {
      tripletOps.push_back(toIndex(eltDg.getIndices()[i]));
      isTriplet.push_back(false);
    }
    auto sectionDg = b.create<hlfir::DesignateOp>(
        loc,
        /*resultType0=*/boxTy,
        /*memref=*/parent,
        /*component=*/mlir::StringAttr{},
        /*component_shape=*/mlir::Value{},
        /*indices=*/mlir::ValueRange{tripletOps},
        /*is_triplet=*/b.getDenseBoolArrayAttr(isTriplet),
        /*substring=*/mlir::ValueRange{},
        /*complex_part=*/mlir::BoolAttr{},
        /*shape=*/sectionShape,
        /*typeparams=*/mlir::ValueRange{},
        /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});

    // Step 5: redirect uses of the formal declare's results to the
    // new section view.  The declare emits two results (box view
    // + raw ref); both should resolve to the section.  When the
    // result types differ, insert a fir.convert.
    mlir::Value sectionResult = sectionDg.getResult();
    auto rewireResult = [&](mlir::Value r) {
      if (r.use_empty()) return;
      mlir::Value replacement = sectionResult;
      if (r.getType() != sectionResult.getType()) {
        // Builder ``b`` already sits after the formal declare,
        // so its insertion point is right after the section.
        replacement = b.create<fir::ConvertOp>(loc, r.getType(), sectionResult)
                          .getResult();
      }
      r.replaceAllUsesWith(replacement);
    };
    rewireResult(formalDecl.getResult(0));
    rewireResult(formalDecl.getResult(1));

    // Erase the declare, the convert, and the element designate
    // (if no other users remain).
    formalDecl.erase();
    conv.erase();
    if (eltDg.getResult().use_empty()) eltDg.erase();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createRewriteSequenceAssociationPass() {
  return std::make_unique<RewriteSequenceAssociationPass>();
}

}  // namespace hlfir_bridge
