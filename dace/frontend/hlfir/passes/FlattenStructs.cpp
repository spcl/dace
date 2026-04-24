// ============================================================================
// FlattenStructs.cpp — Array-of-Structs → Struct-of-Arrays at the HLFIR level.
// ============================================================================
// Motivation:
//     ICON code (e.g. the velocity-tendencies kernel) wraps many arrays in a
//     single Fortran derived type and passes the struct around:
//
//         type :: state_t
//             real(8) :: u(nproma, nlev), v(nproma, nlev)
//             real(8) :: w(nproma, nlev), p(nproma, nlev)
//         end type
//         subroutine kernel(st, ...)
//             type(state_t), intent(inout) :: st
//             ... st%u(i,j) ...
//
//     The SDFG representation does not model nested structs.  We prefer a
//     flat argument list: one array per struct member.  Doing the rewrite at
//     HLFIR (rather than as a post-SDFG pass) keeps the SDFG side oblivious
//     to struct data.
//
// What the pass does, per function:
//     1. Struct-typed dummy arguments:
//        (a) Jagged structs — all members are 1-D arrays of the same scalar
//            type but with differing extents: pack into one ELLPACK-style
//            companion of shape ``[numMembers x max(extents)]`` and alias
//            each member to a row slice via fir.coordinate_of + fir.convert.
//        (b) Otherwise (uniform shapes or mixed scalar + array members):
//            insert one block argument per member, synthesise a
//            hlfir.declare for each, and rewrite component-selecting
//            designates onto the per-member companion.
//        After every struct argument has been flattened, refresh the
//        function type and rename the function to "<orig>_soa".
//
//     2. For each local fir.alloca of a struct (scalar or array-of-scalar-
//        struct) with flat members: synthesise per-member fir.alloca +
//        hlfir.declare companions, rewrite designates, and erase the
//        original.
//
// Flat-member rule:
//     A member is "flat" if its type is a simple scalar (f32/f64/i32/i64) or
//     a fir.array whose element type is a simple scalar.  Members that are
//     themselves derived types or arrays-of-derived-types are out of scope
//     and abort the rewrite for that declare.
//
// Out of scope:
//     - Array-of-struct where the struct has array members (would need shape
//       concatenation of the two rank axes).
//     - Deferred-shape / allocatable array members in a struct argument
//       (would need fir.box and run-time shape propagation).
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

namespace hlfir_bridge {

namespace {

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

/// Strip one layer of fir.box / fir.ref / fir.heap / fir.pointer.
static mlir::Type unwrapOne(mlir::Type t) {
    if (auto x = mlir::dyn_cast<fir::BoxType>(t))       return x.getEleTy();
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
static mlir::Type companionPointee(bool outerIsArray,
                                   llvm::ArrayRef<int64_t> outerShape,
                                   mlir::Type memberTy) {
    bool memberIsArray = mlir::isa<fir::SequenceType>(memberTy);
    if (outerIsArray && memberIsArray) return {};
    if (outerIsArray)
        return fir::SequenceType::get(outerShape, memberTy);
    return memberTy;  // scalar struct: pass the member through verbatim
}

/// Rebuild `shell`'s wrappers around a new inner type.  Used when we need
/// to mirror the original declare's result-0 wrapping (e.g. fir.box<array<...>>)
/// with the element type replaced.
static mlir::Type rewrapWith(mlir::Type shell, mlir::Type newInner) {
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
/// synthesise a companion pointee for every (outer, member) pair.
static bool allMembersFlattenable(fir::RecordType rec, bool outerIsArray) {
    for (auto &pair : rec.getTypeList()) {
        if (!isFlatMemberType(pair.second)) return false;
        if (outerIsArray && mlir::isa<fir::SequenceType>(pair.second))
            return false;  // array-of-struct-with-array-members: skip
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
static void rewriteDesignate(
    hlfir::DesignateOp dg,
    const llvm::StringMap<mlir::Value> &memberBase) {

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
    /// the plan for now — callers of that path should fall back to the
    /// looped copy-in emission without plan metadata.
    void recordStructArgEntry(hlfir::DeclareOp argDecl, fir::RecordType rec,
                              llvm::StringRef intentStr) {
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

        for (auto &pair : rec.getTypeList()) {
            llvm::StringRef memName = pair.first;
            mlir::Type memTy = pair.second;
            int rank = memberRank(memTy);
            if (rank > maxRank) maxRank = rank;

            std::string flat = (outerName + "_" + memName).str();
            flatNames.push_back(mkStr(flat));

            // read_expr: "<outer>%<member>($i1, $i2, ..., $iN)".
            // Scalars skip the ``($iN)`` suffix entirely.
            std::string read = (outerName + "%" + memName).str();
            if (rank > 0) {
                read += "(";
                for (int i = 1; i <= rank; ++i) {
                    if (i > 1) read += ", ";
                    read += "$i" + std::to_string(i);
                }
                read += ")";
            }
            readExprs.push_back(mkStr(read));

            if (std::string dt = dtypeName(memberElementType(memTy)); !dt.empty())
                scratchDtype = dt;
        }

        // Shape exprs for the recipe: one ``size(outer%<first_member>,
        // dim=i)`` per loop dimension.  The first member is chosen
        // arbitrarily — the non-jagged path guarantees matching shapes
        // across members so which one we size against is immaterial.
        llvm::SmallVector<mlir::Attribute, 4> shapeExprs;
        if (maxRank > 0 && !rec.getTypeList().empty()) {
            llvm::StringRef first = rec.getTypeList()[0].first;
            for (int i = 1; i <= maxRank; ++i) {
                std::string s = ("size(" + outerName + "%" + first
                                 + ", dim=" + std::to_string(i) + ")").str();
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

        // Phase 1: collect struct-typed dummy arguments, rewrite them in
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

        // Phase 2: local allocations of struct types.
        llvm::SmallVector<hlfir::DeclareOp, 8> work;
        func.walk([&](hlfir::DeclareOp d) {
            if (isLocallyFlattenable(d)) work.push_back(d);
        });
        for (auto d : work) splitLocal(d);
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
            if (outerIsArray) continue;

            Plan p;
            p.argDecl = argDecl;
            p.rec     = rec;
            if (isJaggedScalarStruct(rec, p.jaggedEleTy, p.jaggedExtents))
                p.jagged = true;
            else if (!allMembersFlattenable(rec, /*outerIsArray=*/false))
                continue;
            plans.push_back({i, p});
        }

        if (plans.empty()) return false;

        // Walk plans in reverse so earlier indices aren't invalidated by
        // earlier erases.  Each replace either mutates the argument list
        // in place (insert-then-erase) or bails out without changes.
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
            recordStructArgEntry(p.argDecl, p.rec, intentStr);
            replaceStructArg(func, idx, p.argDecl, p.rec);
        }
        return true;
    }

    // -------------------------------------------------------------------
    // Phase 1: struct dummy arguments
    // -------------------------------------------------------------------

    void replaceStructArg(mlir::func::FuncOp func, unsigned argIdx,
                          hlfir::DeclareOp argDecl, fir::RecordType rec) {
        auto &block = func.front();
        auto loc = argDecl.getLoc();
        auto *ctx = func.getContext();
        auto baseName = argDecl.getUniqName().str();

        // Insert new block args right after the old one so the argument order
        // tracks the original member order.  Insertion shifts indices >= pos
        // by 1, so we insert sequentially at argIdx+1, argIdx+2, …
        llvm::StringMap<mlir::Value> memberBase;
        unsigned memberCount = 0;
        for (auto &pair : rec.getTypeList()) {
            auto memName = pair.first;
            auto memTy   = pair.second;
            auto pointee = companionPointee(/*outerIsArray=*/false, {}, memTy);
            if (!pointee) continue;  // defensive; caller already checked
            auto refTy = fir::ReferenceType::get(pointee);

            unsigned newArgIdx = argIdx + 1 + memberCount;
            block.insertArgument(newArgIdx, refTy, loc);
            auto newArg = block.getArgument(newArgIdx);

            mlir::OpBuilder b(&block, std::next(argDecl->getIterator()));
            b.setInsertionPoint(argDecl);

            // Array members need a fir.shape operand for the declare to verify.
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

        // Rewrite designates on the struct declare.
        llvm::SmallVector<hlfir::DesignateOp, 8> designates;
        for (auto *u : argDecl.getResult(0).getUsers())
            if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u))
                designates.push_back(dg);
        for (auto dg : designates) rewriteDesignate(dg, memberBase);

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
    // Phase 2: local struct allocations
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
        return allMembersFlattenable(rec, outerIsArray);
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

        llvm::StringMap<mlir::Value> memberBase;
        for (auto &pair : rec.getTypeList()) {
            auto memName = pair.first;
            auto memTy   = pair.second;
            auto pointee = companionPointee(outerIsArray, outerShape, memTy);
            if (!pointee) continue;

            auto newAlloca = b.create<fir::AllocaOp>(loc, pointee);

            // Declare result-0 mirrors the original declare's wrapping but
            // with the element type swapped; result-1 is always the raw ref.
            auto res0Ty = rewrapWith(decl.getResult(0).getType(), memTy);
            auto res1Ty = fir::ReferenceType::get(pointee);

            llvm::SmallVector<mlir::Value, 2> operands;
            operands.push_back(newAlloca);
            if (shape) operands.push_back(shape);

            mlir::NamedAttrList attrs;
            attrs.append("uniq_name",
                         mlir::StringAttr::get(ctx,
                                               baseName + "_" + memName));
            attrs.append(declareSegments(b, /*hasShape=*/shape != nullptr));

            auto newDecl = b.create<hlfir::DeclareOp>(
                loc, mlir::TypeRange{res0Ty, res1Ty},
                mlir::ValueRange(operands), attrs);

            memberBase[memName] = newDecl.getResult(0);
        }

        llvm::SmallVector<hlfir::DesignateOp, 8> designates;
        for (auto *u : decl.getResult(0).getUsers())
            if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u))
                designates.push_back(dg);
        for (auto dg : designates) rewriteDesignate(dg, memberBase);

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
