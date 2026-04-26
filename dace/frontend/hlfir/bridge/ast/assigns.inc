// ============================================================================
// ast/assigns.inc — included from extract_ast.cpp; NOT a separate TU.
// ============================================================================
//
// Per-shape assign builders + small type / value helpers.  Owns:
//   * buildIndexExpr body (continuation from expressions.cpp's forward
//     declaration).
//   * buildAssignNode / buildCopyNode / buildMemsetNode /
//     buildLibCallNode — one ASTNode per Fortran assignment shape.
//   * buildSectionScalarAssign / buildSectionReduceAssign —
//     loop-synthesis for arr(lo:hi) = scalar and section-reduce.
//   * Type / value helpers: peelWrappers, isArrayRef,
//     isConstantZero, traceLB, asSectionDesignate.
//
// This file is included verbatim from extract_ast.cpp via
// #include "bridge/ast/assigns.cpp" and shares that translation
// unit's namespace, includes, and file-static state.  It MUST NOT be
// added to the build's compile list — CMakeLists.txt deliberately omits
// it.  The split is purely for readability: the AST builder used to
// be a single 2800-line file.
// ============================================================================



/// Build a display expression for an index value.  Mirrors Fortran syntax
/// (1-based, square brackets for indirect access) so the Python side can
/// pattern-match on it.  Depth-limited to avoid loops on malformed IR.
inline std::string buildIndexExpr(mlir::Value v, int d) {
    if (d > limits::kBuildIndexExprDepth || !v) return "?";

    // Block args (fir.do_loop induction, hlfir.elemental iter) have no
    // defining op — resolve via indexStack() first so inlined elemental
    // bodies that use the block arg directly as a designate index don't
    // fall through to "?".
    if (!v.getDefiningOp()) {
        auto resolved = resolveIndex(v);
        if (!resolved.empty()) return resolved;
        return "?";
    }
    auto *def = v.getDefiningOp();

    // fir.convert is transparent.
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def))
        return buildIndexExpr(conv.getValue(), d + 1);

    // A loaded scalar — either a named variable (loop iter) or an indirect
    // access via hlfir.designate on another array.
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
        auto mem = ld.getMemref();
        if (auto *md = mem.getDefiningOp()) {
            if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(md)) {
                auto arrName = resolveIndex(dg.getMemref());
                if (arrName.empty()) arrName = traceToDecl(dg.getMemref());
                if (arrName.empty()) return "?";
                // Constant-indexed array element used as an index / bound
                // (Fortran ``pos(1):pos(2)`` / ``a(idx(1), j)`` / …).  Mint
                // an SDFG symbol ``__sym_<arr>_<n>`` and return the symbol
                // name; the registry's contents become ``kind="symbol_init"``
                // AST nodes prepended to the body in ``extractAST``, which
                // the Python emitter stages as an interstate-edge load
                // ``__sym_<arr>_<n> = <arr>[n-1]`` at SDFG entry.  The
                // memlet subset using the symbol is then a closed-form
                // expression DaCe can simplify, instead of a data
                // reference DaCe can't represent.
                auto idxOps = dg.getIndices();
                bool allConstScalar = !idxOps.empty();
                std::vector<int64_t> consts;
                for (auto idx : idxOps) {
                    auto c = traceConstInt(idx);
                    if (!c) { allConstScalar = false; break; }
                    consts.push_back(*c);
                }
                if (allConstScalar && consts.size() == 1)
                    return internPosSymbol(arrName, consts[0]);
                // Multi-dim constant-indexed reads (``pos(1, 2)``) are not
                // yet folded to a symbol; the legacy ``pos[0, 1]`` form
                // below still applies.
                // Non-constant index — render as a Fortran 1-based
                // subscript ``arr[idx, …]``.  The Python emitter
                // converts this to DaCe 0-based form
                // (``arr[(idx) - offset_arr_d0, …]``) at consumption
                // time — loop bounds via ``_fortran_subs_to_dace``,
                // memlets via ``indirect_to_dace`` — keeping the bridge
                // output uniform regardless of which downstream context
                // ultimately consumes it.
                std::string s = arrName + "[";
                bool first = true;
                unsigned di = 0;
                for (auto idx : dg.getIndices()) {
                    if (!first) s += ",";
                    s += buildDesignateIndexExpr(dg, di, idx, d + 1);
                    first = false;
                    ++di;
                }
                s += "]";
                return s;
            }
        }
        auto n = traceToDecl(mem);
        if (!n.empty()) return n;
        // Last resort: maybe the load memref is the elemental's block arg
        // indirectly (unlikely, but guard).
        return "?";
    }

    // Inside an elemental body, an index value IS a tracked block arg.
    auto resolved = resolveIndex(v);
    if (!resolved.empty()) return resolved;

    // Constant integer.
    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
        if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
            return std::to_string(i.getInt());

    // Integer arithmetic used inside index expressions — Flang lowers
    // ``arr(..., nlev-1, ...)`` via ``arith.subi %nlev, %c1``, ``nlev+1``
    // via ``arith.addi``, etc.  Render parenthesised so downstream
    // ``build_memlet_index`` takes the closed-form expression branch.
    auto nm = def->getName().getStringRef();
    static const std::map<llvm::StringRef, std::string> int_bin = {
        {"arith.addi",  " + "},
        {"arith.subi",  " - "},
        {"arith.muli",  " * "},
        {"arith.divsi", " // "}, {"arith.divui", " // "},
    };
    if (auto it = int_bin.find(nm);
        it != int_bin.end() && def->getNumOperands() == 2) {
        return "(" + buildIndexExpr(def->getOperand(0), d + 1)
                   + it->second
                   + buildIndexExpr(def->getOperand(1), d + 1) + ")";
    }

    return "?";
}

// ---------------------------------------------------------------------------
// Per-statement builders
// ---------------------------------------------------------------------------

static ASTNode buildAssignNode(hlfir::AssignOp assign) {
    ASTNode node;
    node.kind = "assign";

    // --- LHS ---
    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp()) {
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(dd)) {
            node.target = traceToDecl(dg.getMemref());
            node.target_is_array = true;
            AccessInfo wa;
            wa.array_name = node.target;
            wa.is_write = true;
            unsigned di = 0;
            for (auto idx : dg.getIndices()) {
                auto n = resolveIndex(idx);
                wa.index_vars.push_back(n.empty() ? "?" : n);
                wa.index_exprs.push_back(buildDesignateIndexExpr(dg, di, idx));
                ++di;
            }
            node.accesses.push_back(std::move(wa));
        } else {
            node.target = traceToDecl(dest);
        }
    } else {
        node.target = traceToDecl(dest);
    }

    // --- RHS expression string ---
    auto src = assign.getOperand(0);
    node.expr = buildExpr(src);
    if (node.expr == "?") {
        if (auto d = src.getDefiningOp()) {
            if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(d)) {
                if (auto f = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
                    std::ostringstream o;
                    o << std::setprecision(17) << f.getValueAsDouble();
                    node.expr = o.str();
                } else if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
                    node.expr = std::to_string(i.getInt());
            }
        }
    }

    // --- Collect RHS array reads ---
    // We emit one AccessInfo per designate *occurrence* in the expression
    // tree — not per unique designate op.  emit_tasklet counts array-name
    // regex occurrences in ``assign_node.expr`` and wires one connector
    // per occurrence, so the bridge must supply matching AccessInfo count.
    // For ``g * g`` Flang shares ``%gv = fir.load %gi`` across both mulf
    // operands; without per-occurrence emission the second ``g`` becomes
    // a dangling ``_in_g_1`` connector with no memlet.
    std::function<void(mlir::Value, int)> collectReads =
        [&](mlir::Value v, int depth) {
        if (depth > 40) return;
        auto *op = v.getDefiningOp();
        if (!op) return;
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(op)) {
            AccessInfo ra;
            ra.array_name = traceToDecl(dg.getMemref());
            ra.is_read = true;
            unsigned di = 0;
            for (auto idx : dg.getIndices()) {
                auto n = resolveIndex(idx);
                ra.index_vars.push_back(n.empty() ? "?" : n);
                ra.index_exprs.push_back(buildDesignateIndexExpr(dg, di, idx));
                ++di;
                // Descend into the index operand so inner indirect loads
                // (edge_idx used below z_kin) get their own AccessInfo.
                collectReads(idx, depth + 1);
            }
            node.accesses.push_back(std::move(ra));
            return;
        }
        for (auto operand : op->getOperands())
            collectReads(operand, depth + 1);
    };
    collectReads(src, 0);

    return node;
}

static int64_t traceLB(mlir::Value v) {
    if (auto c = traceConstInt(v)) return *c;
    return -1;
}

/// Peel `fir.ref<…>` / `fir.box<…>` / `fir.heap<…>` / `fir.ptr<…>` wrappers.
static mlir::Type peelWrappers(mlir::Type t) {
    for (int i = 0; i < limits::kTypeWrapperPeelDepth; ++i) {
        mlir::Type next = t;
        if (auto b = mlir::dyn_cast<fir::BoxType>(next))       next = b.getEleTy();
        else if (auto r = mlir::dyn_cast<fir::ReferenceType>(next)) next = r.getEleTy();
        else if (auto h = mlir::dyn_cast<fir::HeapType>(next))      next = h.getEleTy();
        else if (auto p = mlir::dyn_cast<fir::PointerType>(next))   next = p.getEleTy();
        else break;
        t = next;
    }
    return t;
}

/// True iff the MLIR type peels to a ``fir.array<...>`` OR is an
/// ``!hlfir.expr<...>`` value with a non-empty shape.  The latter is
/// what every ``hlfir.elemental`` / ``hlfir.matmul`` / ``hlfir.transpose``
/// produces; the bridge has to treat those as arrays so that
/// ``hlfir.assign %elem to %dst`` routes through the elemental walker
/// (or libcall handler) rather than the section-scalar fallback that
/// only knows how to broadcast a scalar across a slice.
static bool isArrayRef(mlir::Type t) {
    auto peeled = peelWrappers(t);
    if (mlir::isa<fir::SequenceType>(peeled)) return true;
    if (auto e = mlir::dyn_cast<hlfir::ExprType>(peeled))
        return !e.getShape().empty();
    return false;
}

/// True iff ``v`` traces back to an ``arith.constant`` with value zero
/// (integer zero or floating-point +0.0 / -0.0).
static bool isConstantZero(mlir::Value v) {
    auto *def = v.getDefiningOp();
    if (!def) return false;
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def))
        return isConstantZero(cv.getValue());
    if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
        if (auto f = mlir::dyn_cast<mlir::FloatAttr>(c.getValue()))
            return f.getValueAsDouble() == 0.0;
        if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
            return i.getInt() == 0;
    }
    return false;
}

/// ``hlfir.assign %src to %dst`` where both sides are array boxes — a
/// whole-array copy.  Emit ``kind="copy"`` and let hlfir_to_sdfg wire a
/// ``standard.CopyLibraryNode``.
static ASTNode buildCopyNode(hlfir::AssignOp assign) {
    ASTNode n;
    n.kind = "copy";
    auto dest = assign.getOperand(1);
    // Always route through traceToDecl so the allocatable alias map
    // (set by ``ALLOCATE`` walks in buildAST) takes effect — direct
    // ``extractName(decl.getUniqName())`` would skip the alias lookup
    // and stale-bind to the first allocation's name.
    if (auto dd = dest.getDefiningOp())
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            n.target = allocAliasFor(extractName(decl.getUniqName().str()));
    if (n.target.empty()) n.target = traceToDecl(dest);
    n.target_is_array = true;
    n.reduce_src = traceToDecl(assign.getOperand(0));
    return n;
}

/// ``hlfir.assign %zero to %dst`` where source is a constant zero and dest
/// is an array box — a zero-fill.  Emit ``kind="memset"`` so
/// hlfir_to_sdfg can wire a ``standard.MemsetLibraryNode``.
/// Return ``dg`` if ``v`` comes from an ``hlfir.designate`` whose
/// ``is_triplet`` attribute marks at least one dimension as a section
/// (lower:upper:stride).  Used by the Phase-1 array-section lowering to
/// split section assignments off from plain indexed designates before
/// the reduce / elemental dispatch.
static hlfir::DesignateOp asSectionDesignate(mlir::Value v) {
    auto *def = v.getDefiningOp();
    if (!def) return {};
    auto dg = mlir::dyn_cast<hlfir::DesignateOp>(def);
    if (!dg) return {};
    for (bool t : dg.getIsTriplet())
        if (t) return dg;
    return {};
}

/// Lower ``<section_designate> = <scalar>`` as a rank-N nested
/// ``kind="loop"`` wrapper around an inner ``kind="assign"`` that
/// writes the scalar into ``target[as_0, ..., as_{R-1}]``.  Mirrors
/// the tail of ``buildElementalAssign`` but sources loop bounds from
/// the designate's triplet operands instead of an elemental's shape.
///
/// Returns an empty vector if any non-triplet (single-index) dim is
/// mixed into the designate — Phase 1 doesn't synthesise loops for
/// mixed slice + index designates; those fall back to the caller's
/// default assign handling.
static std::vector<ASTNode> buildSectionScalarAssign(
    hlfir::AssignOp assign, hlfir::DesignateOp dst) {

    llvm::ArrayRef<bool> triplets = dst.getIsTriplet();
    if (triplets.empty()) return {};

    // Split the designate's flat index operand list into per-dim groups
    // of three (lower, upper, stride) for triplet dims.  A non-triplet
    // dim bails Phase 1 — we'd need a separate "fixed at k" index to
    // thread into the inner assign.
    auto indices = dst.getIndices();
    std::vector<std::array<mlir::Value, 3>> triples;
    unsigned cursor = 0;
    for (bool t : triplets) {
        if (!t) return {};
        if (cursor + 3 > indices.size()) return {};
        triples.push_back({indices[cursor], indices[cursor + 1],
                           indices[cursor + 2]});
        cursor += 3;
    }
    if (triples.empty()) return {};

    unsigned rank = triples.size();
    std::vector<std::string> iter_names;
    iter_names.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
        iter_names.push_back("as_" + std::to_string(i));

    // Inner assign: target[iter_names] = <scalar_rhs>.
    ASTNode inner;
    inner.kind = "assign";
    inner.target = traceToDecl(dst.getMemref());
    inner.target_is_array = true;
    inner.expr = buildExpr(assign.getOperand(0));

    AccessInfo wa;
    wa.array_name = inner.target;
    wa.is_write = true;
    for (unsigned i = 0; i < rank; ++i) {
        wa.index_vars.push_back(iter_names[i]);
        wa.index_exprs.push_back(iter_names[i]);
    }
    inner.accesses.push_back(std::move(wa));

    // Wrap descending so the outermost ASTNode is the outermost loop.
    // Lower bound goes into loop_lower_expr (string form) so symbolic
    // lowers like ``res(a:b)`` survive — emit_loop prefers it over the
    // int ``loop_lower`` when non-empty.
    ASTNode current = inner;
    for (int i = rank - 1; i >= 0; --i) {
        ASTNode wrap;
        wrap.kind = "loop";
        wrap.loop_iter = iter_names[i];
        wrap.loop_lower_expr = buildIndexExpr(triples[i][0]);
        wrap.loop_bound      = buildIndexExpr(triples[i][1]);
        wrap.children.push_back(current);
        current = wrap;
    }
    return {current};
}

static ASTNode buildMemsetNode(hlfir::AssignOp assign) {
    ASTNode n;
    n.kind = "memset";
    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp())
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            n.target = allocAliasFor(extractName(decl.getUniqName().str()));
    if (n.target.empty()) n.target = traceToDecl(dest);
    n.target_is_array = true;
    return n;
}

/// ``target = matmul(a, b)`` / ``transpose(a)`` / ``dot_product(x, y)`` /
/// ``count(mask [,dim])`` — the source of an hlfir.assign is a first-class
/// hlfir linalg / reduction op.  Emit ``kind="libcall"`` so hlfir_to_sdfg
/// can wire the matching DaCe library node.
///
/// For library nodes that take an integer ``dim`` argument
/// (``hlfir.count`` etc.), the second operand is the dim value;
/// trace it via ``traceConstInt`` and stash it in ``reduce_axes`` (0-based,
/// same convention as ``buildReduceNode``).  ``emit_libcall`` reads it
/// back and converts to Fortran 1-based for the library-node constructor.
static ASTNode buildLibCallNode(hlfir::AssignOp assign,
                                mlir::Operation *srcOp,
                                std::string_view callee) {
    ASTNode n;
    n.kind = "libcall";
    n.callee = callee;

    auto dest = assign.getOperand(1);
    if (auto dd = dest.getDefiningOp())
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
            n.target = allocAliasFor(extractName(decl.getUniqName().str()));
    if (n.target.empty()) n.target = traceToDecl(dest);
    n.target_is_array = isArrayRef(dest.getType());

    // Linalg ops use call_args for every operand; reduction-style ops
    // (count) treat the first operand as the array source and any
    // remaining numeric operand as a dim/axis arg.
    auto opName = srcOp->getName().getStringRef();
    bool is_count = (opName == "hlfir.count");
    if (is_count) {
        if (srcOp->getNumOperands() > 0)
            n.call_args.push_back(traceToDecl(srcOp->getOperand(0)));
        if (srcOp->getNumOperands() >= 2) {
            auto dim_val = srcOp->getOperand(1);
            if (auto c = traceConstInt(dim_val))
                n.reduce_axes.push_back(*c - 1);   // Fortran 1-based → 0-based
        }
    } else {
        for (auto operand : srcOp->getOperands())
            n.call_args.push_back(traceToDecl(operand));
    }
    return n;
}

/// ``target = sum(a)`` / product / minval / maxval — one of the dedicated
/// hlfir reduction ops appears as the source of an hlfir.assign.
///
/// Returned ASTNode carries enough metadata for hlfir_to_sdfg to call
/// ``state.add_reduce(wcr, axes, identity)`` and wire the input / output
/// memlets.  ``axes`` is left empty for whole-array reductions — Flang
/// signals that by emitting the reduction op with no ``dim`` operand.
/// Lower ``target = ANY/ALL/SUM/PRODUCT(src(lo:hi, ...))`` as a
/// loop-accumulator: an init-to-identity assign followed by a
/// ``kind="loop"`` whose body ORs / ANDs / sums the next section
/// element into ``target``.  Used when the reduction's input is a
/// section designate — DaCe's ``Reduce`` library node would read the
/// whole source array and produce a wrong result.
///
/// Handles the common shape where the destination is a scalar or an
/// element designate (``levelmask(jk)``) and the source has exactly
/// the section dims to loop over; non-section dims of the source
/// thread through via their existing indices (``jk`` here).  Returns
/// an empty vector when the shape doesn't fit so the caller falls
/// back to whole-array ``buildReduceNode``.
static std::vector<ASTNode> buildSectionReduceAssign(
    hlfir::AssignOp assign, hlfir::DesignateOp src,
    std::string_view pyOp, std::string_view identity) {

    auto triplets = src.getIsTriplet();
    if (triplets.empty()) return {};
    auto srcIndices = src.getIndices();

    struct DimSpec {
        bool isTriplet = false;
        mlir::Value lo, hi, stride;
        mlir::Value index;
    };
    std::vector<DimSpec> dims;
    unsigned cursor = 0;
    for (bool t : triplets) {
        DimSpec d; d.isTriplet = t;
        if (t) {
            if (cursor + 3 > srcIndices.size()) return {};
            d.lo = srcIndices[cursor++];
            d.hi = srcIndices[cursor++];
            d.stride = srcIndices[cursor++];
        } else {
            if (cursor + 1 > srcIndices.size()) return {};
            d.index = srcIndices[cursor++];
        }
        dims.push_back(d);
    }
    unsigned sectionRank = 0;
    for (auto &d : dims) if (d.isTriplet) sectionRank++;
    if (sectionRank == 0) return {};

    std::vector<std::string> iterNames;
    iterNames.reserve(sectionRank);
    for (unsigned i = 0; i < sectionRank; ++i)
        iterNames.push_back("ar_" + std::to_string(i));

    // Target name + index expressions — target may be a scalar (no
    // designate) or an element designate like ``levelmask(jk)``.
    auto dst = assign.getOperand(1);
    std::string tgtName;
    hlfir::DesignateOp tgtDg;
    if (auto *dd = dst.getDefiningOp())
        tgtDg = mlir::dyn_cast<hlfir::DesignateOp>(dd);
    if (tgtDg) tgtName = traceToDecl(tgtDg.getMemref());
    else       tgtName = traceToDecl(dst);
    if (tgtName.empty()) return {};

    AccessInfo tgtWrite;
    tgtWrite.array_name = tgtName;
    tgtWrite.is_write = true;
    if (tgtDg) {
        for (auto idx : tgtDg.getIndices()) {
            auto nm = resolveIndex(idx);
            tgtWrite.index_vars.push_back(nm.empty() ? "?" : nm);
            tgtWrite.index_exprs.push_back(buildIndexExpr(idx));
        }
    }
    bool tgtIsArray = !tgtWrite.index_vars.empty();

    AccessInfo tgtRead = tgtWrite;
    tgtRead.is_write = false;
    tgtRead.is_read = true;

    // Source read — full base array name, indexed with section iters
    // for triplet dims and the original indices for non-section dims.
    std::string srcName = traceToDecl(src.getMemref());
    AccessInfo srcRead;
    srcRead.array_name = srcName;
    srcRead.is_read = true;
    unsigned sectionIdx = 0;
    for (auto &d : dims) {
        if (d.isTriplet) {
            srcRead.index_vars.push_back(iterNames[sectionIdx]);
            srcRead.index_exprs.push_back(iterNames[sectionIdx]);
            sectionIdx++;
        } else {
            auto nm = resolveIndex(d.index);
            srcRead.index_vars.push_back(nm.empty() ? "?" : nm);
            srcRead.index_exprs.push_back(buildIndexExpr(d.index));
        }
    }

    // Init assign: target = identity
    ASTNode init;
    init.kind = "assign";
    init.target = tgtName;
    init.target_is_array = tgtIsArray;
    init.expr = std::string(identity);
    init.accesses.push_back(tgtWrite);

    // Accumulate assign: ``target = target <op> src`` for binary ops
    // (``+``, ``*``, ``or``, ``and``); ``target = fn(target, src)`` for
    // function-form reductions (``min``, ``max``).  The latter pattern
    // gives Min/MaxVal section-reduce a tasklet shape that lowers
    // cleanly to ``std::min`` / ``std::max`` via DaCe's symbolic
    // codegen.
    ASTNode acc;
    acc.kind = "assign";
    acc.target = tgtName;
    acc.target_is_array = tgtIsArray;
    bool isFnForm = (pyOp == "min" || pyOp == "max");
    if (isFnForm) {
        acc.expr = std::string(pyOp) + "(" + tgtName + ", " + srcName + ")";
    } else {
        acc.expr = "(" + tgtName + " " + std::string(pyOp) + " " + srcName + ")";
    }
    acc.accesses.push_back(tgtWrite);
    acc.accesses.push_back(tgtRead);
    acc.accesses.push_back(srcRead);

    // Wrap accumulate in one loop per section dim (outermost first,
    // matching buildElementalAssign's convention).
    ASTNode current = acc;
    int revIdx = (int)sectionRank;
    for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
        if (!it->isTriplet) continue;
        --revIdx;
        ASTNode wrap;
        wrap.kind = "loop";
        wrap.loop_iter = iterNames[revIdx];
        wrap.loop_lower_expr = buildIndexExpr(it->lo);
        wrap.loop_bound      = buildIndexExpr(it->hi);
        wrap.children.push_back(current);
        current = wrap;
    }

    return {init, current};
}
