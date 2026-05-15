// ============================================================================
// trace_utils.cpp  --  Shared SSA tracing utilities
// ============================================================================

#include "bridge/trace_utils.h"

#include <algorithm>
#include <unordered_map>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace hlfir_bridge {

// Disambiguation overrides for Fortran short-name collisions across
// inlined scopes.  When ``hlfir-inline-all`` splices a callee's body
// into the caller, both the caller's argument declare
// (``_QFmainEinp``) and the callee's dummy declare
// (``_QFinner_loopsEinp``) end up in one function with the same
// trailing short name (``inp``).  Without disambiguation,
// ``builder.arrays`` keys collide and view-alias linking edges
// self-loop.  ``extract_vars`` populates this map with
// ``mangled -> unique_short_name`` for the colliding entries; every
// subsequent ``extractName`` call resolves to the unique form.
static thread_local std::unordered_map<std::string, std::string>
    kManglingOverride;

void setManglingOverride(const std::string &mangled,
                         const std::string &shortName) {
  kManglingOverride[mangled] = shortName;
}

void clearManglingOverrides() { kManglingOverride.clear(); }

std::string extractName(const std::string &m) {
  auto it = kManglingOverride.find(m);
  if (it != kManglingOverride.end()) return it->second;
  auto p = m.rfind('E');
  std::string name = p != std::string::npos ? m.substr(p + 1) : m;
  // Sanitize dots  --  flang emits compiler-generated globals like
  // ``_QQro.4xi4.0`` (read-only constant pool for array literals)
  // whose names contain ``.``.  DaCe's ``NestedDict`` reserves
  // ``.`` as a nested-key separator and rejects dotted keys
  // outright.  Fortran identifiers can't contain ``.``, so
  // replacing every ``.`` with ``_`` is collision-free w.r.t.
  // user names.  Done at the boundary (extractName is the
  // canonical "MLIR mangled -> Python-side name" helper) so the
  // raw mangled names in the IR stay intact.
  std::replace(name.begin(), name.end(), '.', '_');
  return name;
}

// Allocatable re-allocation alias map.  Keyed by the raw Fortran name
// (what the declare chain alone would resolve to).  Updated as the
// bridge's IR walker passes ``fir.allocmem``-bound ``fir.store`` ops
// (see extract_ast.cpp); read by ``traceToDecl`` so every downstream
// access lands on the currently-live SDFG transient.
static thread_local std::unordered_map<std::string, std::string> kAllocAlias;

std::string allocAliasFor(const std::string &raw) {
  auto it = kAllocAlias.find(raw);
  return it == kAllocAlias.end() ? raw : it->second;
}

void setAllocAlias(const std::string &raw, const std::string &alias) {
  if (alias == raw)
    kAllocAlias.erase(raw);
  else
    kAllocAlias[raw] = alias;
}

void clearAllocAliases() { kAllocAlias.clear(); }

std::string traceToDecl(mlir::Value val, int max) {
  for (int i = 0; i < max && val; ++i) {
    auto *d = val.getDefiningOp();
    if (!d) break;
    if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      // Walk through inlined assumed-shape aliases to the outer
      // caller declare so downstream SDFG emission references
      // the real storage by its caller-side name.
      if (auto outer = asAssumedShapeAlias(dc)) {
        val = outer.getResult(0);
        continue;
      }
      return allocAliasFor(extractName(dc.getUniqName().str()));
    }
    if (auto dc = mlir::dyn_cast<fir::DeclareOp>(d))
      return allocAliasFor(extractName(dc.getUniqName().str()));
    if (auto c = mlir::dyn_cast<fir::ConvertOp>(d)) {
      val = c.getValue();
      continue;
    }
    if (auto l = mlir::dyn_cast<fir::LoadOp>(d)) {
      val = l.getMemref();
      continue;
    }
    if (auto co = mlir::dyn_cast<fir::CoordinateOp>(d)) {
      val = co.getRef();
      continue;
    }
    // ``fir.rebox`` retypes an existing box (e.g. section view box
    // -> ``box<ptr<...>>`` for a Fortran ``ptr => slice`` rebind);
    // it doesn't change the underlying storage.  Walk through so a
    // downstream designate over the reboxed value still resolves
    // to the parent's name.  Same role as the
    // ``hlfir-rewrite-pointer-assigns`` slice-target forwarding:
    // pointer reads land back on the parent array.
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) {
      val = rb.getBox();
      continue;
    }
    if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
      val = eb.getMemref();
      continue;
    }
    // ``fir.box_addr`` extracts the data pointer from a descriptor
    // (heap / ptr underlying the box).  Flang emits it for every
    // allocatable / pointer dereference; the underlying storage
    // name is the box's source declare, so we walk through.
    if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(d)) {
      val = ba.getVal();
      continue;
    }
    // Section / element designates (``a(lo:hi)``, ``a(i)``)  --  walk
    // through to the underlying memref so a reduce over an
    // ``hlfir.any %levmask(i_startblk:i_endblk, jk)`` resolves its
    // source array to ``levmask``.
    if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
      val = dg.getMemref();
      continue;
    }
    if (auto s = mlir::dyn_cast<mlir::arith::SelectOp>(d)) {
      val = s.getTrueValue();
      continue;
    }
    break;
  }
  return "";
}

std::optional<int64_t> traceConstInt(mlir::Value v) {
  for (int i = 0; i < limits::kTraceConstIntMax; ++i) {
    auto *d = v.getDefiningOp();
    if (!d) break;
    if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(d))
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
        return ia.getInt();
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      v = cv.getValue();
      continue;
    }
    // Flang wraps each static extent in a `select extent>0, extent, 0`
    // clamp; follow the true branch to reach the original value.
    // Restrict to that exact shape -- the false value must be the
    // constant 0 -- so we don't accidentally follow Fortran ``MAX``
    // / ``MIN`` (also lowered as ``arith.select`` over a cmp) and
    // collapse a non-constant bound to its first operand.
    if (auto s = mlir::dyn_cast<mlir::arith::SelectOp>(d)) {
      auto *fdef = s.getFalseValue().getDefiningOp();
      bool false_is_zero = false;
      if (fdef) {
        if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(fdef))
          if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
            false_is_zero = (ia.getInt() == 0);
      }
      if (false_is_zero) {
        v = s.getTrueValue();
        continue;
      }
    }
    break;
  }
  return std::nullopt;
}

std::string traceExtentExpr(mlir::Value v) {
  if (!v) return "";
  auto *def = v.getDefiningOp();
  if (!def) return "";

  // Transparent peels.
  if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def))
    return traceExtentExpr(cv.getValue());

  // Constant integer literal.
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return std::to_string(ia.getInt());

  // Load of a Fortran scalar -- render as its short name.  Loads of
  // designate-of-array (``cols(1)``) aren't expected in extent
  // chains for the canonical Flang ``ub - lb + 1`` shape, so we
  // bail rather than mint a position symbol.
  if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
    auto mem = ld.getMemref();
    auto *md = mem.getDefiningOp();
    if (!md) return "";
    if (mlir::isa<hlfir::DeclareOp>(md) || mlir::isa<fir::DeclareOp>(md)) {
      return traceToDecl(mem);
    }
    return "";
  }

  // Flang's clamp wrapper for a Fortran triplet extent:
  //   %cmp = arith.cmpi sgt, %ext, %c0
  //   %r   = arith.select %cmp, %ext, %c0
  // Array extents are non-negative by construction in valid Fortran,
  // so the clamp is dead defensive code at runtime.  Drop the wrap
  // and return the underlying extent expression directly  --  keeps
  // SDFG shapes and view subsets readable (``klon`` not
  // ``max(klon, 0)``) and lets sympy fold downstream arithmetic.
  if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
    auto *cdef = sel.getCondition().getDefiningOp();
    auto cmp = cdef ? mlir::dyn_cast<mlir::arith::CmpIOp>(cdef) : nullptr;
    bool falseIsZero = false;
    if (auto *fdef = sel.getFalseValue().getDefiningOp())
      if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(fdef))
        if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
          falseIsZero = (ia.getInt() == 0);
    if (cmp && falseIsZero &&
        cmp.getPredicate() == mlir::arith::CmpIPredicate::sgt) {
      return traceExtentExpr(sel.getTrueValue());
    }
    return "";
  }

  // Binary integer arithmetic.  Render parenthesised so the result
  // composes cleanly when nested.
  auto nm = def->getName().getStringRef();
  static const std::map<llvm::StringRef, std::string> bin = {
      {"arith.addi", " + "},   {"arith.subi", " - "},   {"arith.muli", " * "},
      {"arith.divsi", " // "}, {"arith.divui", " // "},
  };
  if (auto it = bin.find(nm); it != bin.end() && def->getNumOperands() == 2) {
    auto l = traceExtentExpr(def->getOperand(0));
    auto r = traceExtentExpr(def->getOperand(1));
    if (l.empty() || r.empty()) return "";
    return "(" + l + it->second + r + ")";
  }

  return "";
}

void collectExtentExprScalars(mlir::Value v, std::set<std::string> &out) {
  if (!v) return;
  auto *def = v.getDefiningOp();
  if (!def) return;
  if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
    collectExtentExprScalars(cv.getValue(), out);
    return;
  }
  if (mlir::isa<mlir::arith::ConstantOp>(def)) return;
  if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
    auto mem = ld.getMemref();
    if (auto *md = mem.getDefiningOp())
      if (mlir::isa<hlfir::DeclareOp>(md) || mlir::isa<fir::DeclareOp>(md)) {
        auto n = traceToDecl(mem);
        if (!n.empty()) out.insert(n);
      }
    return;
  }
  if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
    // Walk both branches; cmp condition leaves are already
    // covered by the operands themselves.
    collectExtentExprScalars(sel.getTrueValue(), out);
    collectExtentExprScalars(sel.getFalseValue(), out);
    return;
  }
  auto nm = def->getName().getStringRef();
  if ((nm == "arith.addi" || nm == "arith.subi" || nm == "arith.muli" ||
       nm == "arith.divsi" || nm == "arith.divui" || nm == "arith.cmpi") &&
      def->getNumOperands() == 2) {
    collectExtentExprScalars(def->getOperand(0), out);
    collectExtentExprScalars(def->getOperand(1), out);
    return;
  }
}

hlfir::DeclareOp asAssumedShapeAlias(hlfir::DeclareOp decl) {
  // Signature: memref produced by another ``hlfir.declare`` (possibly
  // behind ``fir.convert`` rebox ops).  This is precisely what Flang
  // emits for the callee's dummy_scope declare after
  // ``hlfir-inline-all`` splices the callee's body into the caller  --
  // the callee declare aliases the caller's outer declare for
  // both assumed-shape (no shape operand on the inner declare) and
  // fixed-shape (the inner declare carries its own copy of the
  // callee-side shape) callees, the only difference being whether
  // the inner declare reissues a shape.  Either way the storage is
  // shared and downstream tracing should walk to the outer declare.
  auto mr = decl.getMemref();
  for (int i = 0; i < limits::kAliasMemrefWalkDepth && mr; ++i) {
    auto *d = mr.getDefiningOp();
    if (!d) break;
    if (auto outer = mlir::dyn_cast<hlfir::DeclareOp>(d)) return outer;
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      mr = conv.getValue();
      continue;
    }
    // Internal-subprogram inlining wraps the outer fixed-shape array
    // in a ``fir.embox`` so the inlined assumed-shape callee sees a
    // ``fir.box``.  Peel through to the underlying declare.
    if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
      mr = eb.getMemref();
      continue;
    }
    // Inlined-callee aliases on CLASS-allocatable / box-typed
    // dummies: the alias's memref comes from a ``fir.load`` of the
    // caller's box-slot declare, possibly preceded by ``fir.rebox``
    // peels (CLASS<heap<T>> -> CLASS<T>).  Walking through both is
    // what catches monomorphic CLASS subroutine dummies as
    // aliases of their caller-side allocatable.
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
      mr = ld.getMemref();
      continue;
    }
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) {
      mr = rb.getBox();
      continue;
    }
    break;
  }
  return {};
}

std::vector<std::optional<int64_t>> declareLowerBounds(hlfir::DeclareOp decl) {
  std::vector<std::optional<int64_t>> lbs;
  auto shape = decl.getShape();
  if (!shape) return lbs;
  auto *def = shape.getDefiningOp();
  if (!def) return lbs;
  if (auto sh = mlir::dyn_cast<fir::ShapeOp>(def)) {
    // Plain fir.shape: every dim defaults to lbound=1.
    for (unsigned i = 0; i < sh.getExtents().size(); ++i)
      lbs.push_back(std::optional<int64_t>(1));
    return lbs;
  }
  if (auto ss = mlir::dyn_cast<fir::ShapeShiftOp>(def)) {
    // shape_shift operands alternate: lb0, ext0, lb1, ext1, ...
    auto ops = ss->getOperands();
    for (unsigned i = 0; i < ops.size(); i += 2)
      lbs.push_back(traceConstInt(ops[i]));
    return lbs;
  }
  return lbs;
}

llvm::SmallVector<mlir::Value, 4> extractExtents(mlir::Value shape) {
  llvm::SmallVector<mlir::Value, 4> result;
  if (!shape) return result;
  auto *def = shape.getDefiningOp();
  if (!def) return result;

  if (auto sh = mlir::dyn_cast<fir::ShapeOp>(def))
    for (auto e : sh.getExtents()) result.push_back(e);

  if (auto ss = mlir::dyn_cast<fir::ShapeShiftOp>(def)) {
    // shape_shift: alternating (lb, extent)  --  we want odd indices.
    auto ops = ss->getOperands();
    for (unsigned i = 1; i < ops.size(); i += 2) result.push_back(ops[i]);
  }

  return result;
}

}  // namespace hlfir_bridge
