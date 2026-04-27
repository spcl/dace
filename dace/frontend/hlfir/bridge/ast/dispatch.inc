// ============================================================================
// ast/dispatch.inc — included from extract_ast.cpp; NOT a separate TU.
// ============================================================================
//
// Per-op dispatcher.  Owns:
//   * buildScfIfAsConditional — scf.if → ASTNode kind=conditional.
//   * walkSCFBeforeRegion — the faithful scf.while walker.
//   * buildWhileNode — scf.while → ASTNode kind=while.
//   * traceLoopIter — find a fir.do_loop's induction var.
//   * buildAST(Block&) — the per-op switch that walks an MLIR block,
//     picks the right shape builder for each hlfir.assign /
//     fir.do_loop / fir.if / etc., and wires alloc-alias
//     binds for fir.allocmem-bound stores.
//   * extractAST(ModuleOp) — public entry point; calls buildAST
//     on the first func.func body and returns the AST.
//
// This file is included verbatim from extract_ast.cpp via
// #include "bridge/ast/dispatch.cpp" and shares that translation
// unit's namespace, includes, and file-static state.  It MUST NOT be
// added to the build's compile list — CMakeLists.txt deliberately omits
// it.  The split is purely for readability: the AST builder used to
// be a single 2800-line file.
// ============================================================================



static ASTNode buildScfIfAsConditional(mlir::scf::IfOp ifOp) {
    ASTNode c;
    c.kind = "conditional";
    c.condition = buildBoolExpr(ifOp.getCondition());

    auto walkArm = [&](mlir::Region &region) -> std::vector<ASTNode> {
        if (region.empty()) return {};
        auto arm = walkSCFBeforeRegion(region.front());
        // If the scf.if yields values, append one scalar_assign per result
        // reading the matching operand of the arm's scf.yield.
        if (ifOp.getNumResults() > 0) {
            mlir::scf::YieldOp yieldOp;
            for (auto &op : region.front())
                if (auto y = mlir::dyn_cast<mlir::scf::YieldOp>(op)) { yieldOp = y; break; }
            if (yieldOp) {
                for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
                    auto target = scfSynthName(ifOp.getResult(i));
                    auto expr = yieldedExpr(yieldOp.getOperand(i));
                    ASTNode a;
                    a.kind = "assign";
                    a.target = target;
                    a.expr = expr;
                    a.target_is_array = false;
                    arm.push_back(std::move(a));
                }
            }
        }
        return arm;
    };

    c.children = walkArm(ifOp.getThenRegion());
    if (!ifOp.getElseRegion().empty())
        c.else_children = walkArm(ifOp.getElseRegion());
    return c;
}

static std::vector<ASTNode> walkSCFBeforeRegion(mlir::Block &block) {
    std::vector<ASTNode> out;
    for (auto &op : block) {
        if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
            out.push_back(buildScfIfAsConditional(ifOp));
            continue;
        }
        if (auto condOp = mlir::dyn_cast<mlir::scf::ConditionOp>(op)) {
            // ``scf.condition(%c)``: break when %c is false.
            ASTNode guard;
            guard.kind = "conditional";
            auto b = buildBoolExpr(condOp.getCondition());
            guard.condition = "not (" + b + ")";
            ASTNode brk;
            brk.kind = "break";
            guard.children.push_back(std::move(brk));
            out.push_back(std::move(guard));
            continue;
        }
        if (auto assign = mlir::dyn_cast<hlfir::AssignOp>(op)) {
            // Route through the normal assign dispatcher so copy/memset /
            // reduction / elemental shapes stay recognised inside the loop.
            auto src = assign.getOperand(0);
            auto dst = assign.getOperand(1);
            bool dst_is_array = isArrayRef(dst.getType());
            bool src_is_array = isArrayRef(src.getType());
            if (dst_is_array && src_is_array) {
                out.push_back(buildCopyNode(assign));
            } else if (dst_is_array && !src_is_array && isConstantZero(src)) {
                out.push_back(buildMemsetNode(assign));
            } else {
                out.push_back(buildAssignNode(assign));
            }
            continue;
        }
        if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
            // IV / counter bump stores Flang emits inside the lifted
            // scf.while body (``i = i + 1``, ``counter = counter - 1``).
            // Handled uniformly for declared vars and bare-alloca scratch
            // counters.
            auto memref = st.getMemref();
            auto target = traceToDecl(memref);
            if (target.empty())
                if (auto *md = memref.getDefiningOp())
                    if (mlir::isa<fir::AllocaOp>(md))
                        target = allocaSynthName(memref);
            if (target.empty()) continue;
            auto expr = buildExpr(st.getValue());
            // Drop stores whose RHS we couldn't resolve.  These are almost
            // always Flang's implicit IV writeback at the end of a
            // ``fir.do_loop`` body: the stored value is a block arg of the
            // surrounding do-loop that buildExpr can't express on its own
            // — and the regular do-loop emitter already handles the IV
            // through ``initialize_expr`` / ``update_expr``.
            if (expr == "?") continue;
            ASTNode a;
            a.kind = "assign";
            a.target = target;
            a.expr = expr;
            a.target_is_array = false;
            out.push_back(std::move(a));
            continue;
        }
        // Pure-value ops — no AST node, their values flow inline.
    }
    return out;
}

static ASTNode buildWhileNode(mlir::scf::WhileOp whileOp) {
    ASTNode n;
    n.kind = "while";
    n.condition = "True";  // all break decisions live inside the body.

    if (whileOp.getBefore().empty()) return n;
    n.children = walkSCFBeforeRegion(whileOp.getBefore().front());
    return n;
}

static std::string traceLoopIter(fir::DoLoopOp loop) {
    for (auto &op : loop.getRegion().front())
        if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
            auto n = traceToDecl(st.getMemref());
            if (!n.empty()) return n;
        }
    return "";
}

// ---------------------------------------------------------------------------
// Block walker
// ---------------------------------------------------------------------------

static std::vector<ASTNode> buildAST(mlir::Block &block) {
    std::vector<ASTNode> nodes;

    // Per-block site counter for ``ALLOCATE``-bound stores into an
    // allocatable's box descriptor.  Increments every time we walk past
    // a ``fir.store (fir.embox-of-fir.allocmem) to <decl_box_ref>``;
    // first store keeps the original Fortran name (site 0 alias =
    // identity), subsequent ones bind ``x_alloc1`` / ``x_alloc2`` /
    // … via setAllocAlias so every downstream traceToDecl picks up
    // the per-allocation transient name.  Uses the block-local map
    // keyed by raw declare name so two separate allocatables in the
    // same scope don't share counters.
    std::map<std::string, unsigned> allocSiteCount;
    // Returns the allocatable's raw Fortran name on a successful match
    // (so the caller can emit a state-change ``<name>_allocated = 1``
    // ASTNode), empty string otherwise.
    auto bindAllocSite = [&](mlir::Operation *op) -> std::string {
        auto store = mlir::dyn_cast<fir::StoreOp>(op);
        if (!store) return {};
        auto valDef = store.getValue().getDefiningOp();
        if (!valDef) return {};
        auto embox = mlir::dyn_cast<fir::EmboxOp>(valDef);
        if (!embox) return {};
        auto allocmem = mlir::dyn_cast_or_null<fir::AllocMemOp>(
            embox.getMemref().getDefiningOp());
        if (!allocmem) return {};
        // Only the user-visible allocs we model — skip embox-of-zero_bits
        // (the empty-init store the bridge already filters out elsewhere).
        auto un = allocmem.getUniqName();
        if (!un || !un->ends_with(".alloc")) return {};
        auto memDef = store.getMemref().getDefiningOp();
        if (!memDef) return {};
        auto decl = mlir::dyn_cast<hlfir::DeclareOp>(memDef);
        if (!decl) return {};
        std::string raw = extractName(decl.getUniqName().str());
        if (raw.empty()) return {};
        unsigned site = allocSiteCount[raw]++;
        setAllocAlias(raw, allocAliasName(raw, site));
        return raw;
    };
    auto emitAllocStateChange = [&](const std::string &name, int value) {
        ASTNode n;
        n.kind = "assign";
        n.target = name + "_allocated";
        n.target_is_array = false;
        n.expr = std::to_string(value);
        nodes.push_back(std::move(n));
    };
    for (auto &op : block) {
        // Bind / advance the alloc-alias for this allocatable, then
        // emit a state-change ``<name>_allocated = 1`` so downstream
        // ``ALLOCATED(arr)`` reads see the right value.  The ALLOCATE
        // store itself produces no other observable side effect in the
        // SDFG model — we treat allocatables as live for the whole
        // scope.
        if (auto allocName = bindAllocSite(&op); !allocName.empty()) {
            emitAllocStateChange(allocName, 1);
            continue;
        }

        // Standalone ``fir.freemem`` — Flang's DEALLOCATE expansion at
        // top level (the trailing ``fir.if (alloc_status != 0) { … }``
        // is the implicit end-of-scope cleanup, handled separately as
        // ``isAllocCleanup``).  Trace through ``fir.box_addr`` and
        // ``fir.load`` to find the underlying ``hlfir.declare`` and
        // emit ``<rawname>_allocated = 0`` against the declare's RAW
        // Fortran name (NOT the current alloc-alias) so multi-site
        // allocatables ``x → x_alloc1 → x_alloc2`` still funnel state
        // updates through the original ``x_allocated`` symbol.
        if (auto fm = mlir::dyn_cast<fir::FreeMemOp>(&op)) {
            mlir::Value cur = fm.getHeapref();
            for (int i = 0; i < limits::kConvertChainDepth && cur; ++i) {
                auto *cd = cur.getDefiningOp();
                if (!cd) break;
                if (auto cv = mlir::dyn_cast<fir::ConvertOp>(cd))
                    { cur = cv.getValue(); continue; }
                if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(cd))
                    { cur = ba.getVal(); continue; }
                if (auto ld = mlir::dyn_cast<fir::LoadOp>(cd))
                    { cur = ld.getMemref(); continue; }
                break;
            }
            std::string name;
            if (cur)
                if (auto *cd = cur.getDefiningOp())
                    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(cd))
                        name = extractName(decl.getUniqName().str());
            if (!name.empty()) emitAllocStateChange(name, 0);
            continue;
        }

        if (auto doLoop = mlir::dyn_cast<fir::DoLoopOp>(op)) {
            ASTNode n;
            n.kind = "loop";
            n.loop_iter  = traceLoopIter(doLoop);
            // Bound resolution.  ``traceToDecl`` is a useful shortcut for
            // the scalar-variable case (``DO i = 1, n`` where ``n`` is a
            // dummy scalar) but it is wrong for array-element loads:
            // ``DO j = row_ptr(i), row_ptr(i+1)-1`` would otherwise resolve
            // to the bare name ``row_ptr`` because the load chain bottoms
            // out at the array's declare.  Detect that case and route
            // through ``buildIndexExpr`` so the bound is rendered as the
            // proper subscripted expression ``row_ptr[(i) - offset_row_ptr_d0]``.
            auto isArrayElementLoad = [](mlir::Value v) -> bool {
                for (int i = 0; i < 32 && v; ++i) {
                    auto *d = v.getDefiningOp();
                    if (!d) break;
                    if (mlir::isa<hlfir::DesignateOp>(d)) return true;
                    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) { v = cv.getValue(); continue; }
                    if (auto ld = mlir::dyn_cast<fir::LoadOp>(d))   { v = ld.getMemref(); continue; }
                    break;
                }
                return false;
            };
            if (auto c = traceConstInt(doLoop.getUpperBound())) {
                n.loop_bound = std::to_string(*c);
            } else if (!isArrayElementLoad(doLoop.getUpperBound())) {
                n.loop_bound = traceToDecl(doLoop.getUpperBound());
            }
            if (n.loop_bound.empty())
                n.loop_bound = buildIndexExpr(doLoop.getUpperBound());
            n.loop_lower = traceLB(doLoop.getLowerBound());
            if (n.loop_lower < 0) {
                // Non-constant lower bound (``DO jk = nflatlev, nlev`` /
                // ``DO j = row_ptr(i), …``).  Capture the symbolic form
                // so emit_loop can thread it through instead of silently
                // defaulting to 1.
                if (!isArrayElementLoad(doLoop.getLowerBound())) {
                    auto sym = traceToDecl(doLoop.getLowerBound());
                    if (!sym.empty()) n.loop_lower_expr = sym;
                }
                if (n.loop_lower_expr.empty())
                    n.loop_lower_expr = buildIndexExpr(doLoop.getLowerBound());
            }
            // Step.  Reverse-direction ``DO i = N, 1, -1`` (LU
            // back-substitution) carries step -1; the bridge needs
            // this to flip init/cond/update in emit_loop.  Constant
            // steps only — symbolic-step loops would silently default
            // to step=1 and produce a wrong-direction iteration if
            // the symbol is actually negative, so throw loudly when
            // the step is non-constant AND non-trivial (i.e. not the
            // default ``%c1``).
            if (auto stepC = traceConstInt(doLoop.getStep())) {
                n.loop_step = *stepC;
            } else {
                throw std::runtime_error(
                    "fir.do_loop with non-constant step — bridge "
                    "currently lowers only constant-step loops. The "
                    "step's sign decides forward-vs-reverse codegen; "
                    "with a symbolic step we'd silently default to +1 "
                    "and produce wrong-direction iteration when the "
                    "symbol is negative.");
            }
            // Elemental-inlined bodies use the fir.do_loop block arg
            // directly as the hlfir.designate index — no fir.store →
            // alloca → fir.load indirection.  traceLoopIter returns ""
            // for that shape; push the block arg onto indexStack() with
            // a synthetic name so resolveIndex() can recover it when the
            // inner designate's index is the raw block arg.
            static thread_local int kDoLoopIterCounter = 0;
            bool pushedBlockArg = false;
            auto &loopBlock = doLoop.getRegion().front();
            if (n.loop_iter.empty() && loopBlock.getNumArguments() > 0) {
                n.loop_iter = "_doit_" + std::to_string(kDoLoopIterCounter++);
                indexStack().push_back({loopBlock.getArgument(0), n.loop_iter});
                pushedBlockArg = true;
            }
            n.children   = buildAST(loopBlock);
            if (pushedBlockArg) indexStack().pop_back();
            nodes.push_back(std::move(n));
            continue;
        }
        if (auto assign = mlir::dyn_cast<hlfir::AssignOp>(op)) {
            auto src = assign.getOperand(0);
            auto dst = assign.getOperand(1);
            bool dst_is_array = isArrayRef(dst.getType());
            bool src_is_array = isArrayRef(src.getType());

            // ``hlfir.expr``-valued sources (any HLFIR op whose result
            // type peels to ``!hlfir.expr<...>``: ``hlfir.elemental``,
            // ``hlfir.matmul``, ``hlfir.transpose``, ``hlfir.dot_product``,
            // ``hlfir.count``, ``hlfir.sum``, …) are array-typed but
            // NOT array refs — they have no memory backing.
            // ``buildCopyNode`` would call ``traceToDecl`` on them and
            // get an empty name; route them to the elemental / libcall
            // handlers below instead.  This is what makes
            // ``res(:) = a(:) - b(:)`` (Flang-generated elemental) and
            // ``res = COUNT(mask, dim=1)`` (libcall returning expr)
            // work when the bridge previously fell through to a
            // degenerate copy.
            bool src_is_hlfir_expr = false;
            if (auto srcOp = src.getDefiningOp()) {
                if (mlir::isa<hlfir::ExprType>(peelWrappers(srcOp->getResult(0).getType())))
                    src_is_hlfir_expr = true;
            }

            // Section-to-section assign — both sides are non-trivial
            // ``hlfir.designate``s with at least one triplet dim.  Walk
            // the structure explicitly because ``buildCopyNode`` would
            // otherwise treat it as a whole-array copy and silently
            // ignore scalar dims and slice offsets (e.g.
            // ``res(nval, pos(1):pos(2)) = a(nval, pos(3):pos(4))``).
            if (dst_is_array && src_is_array && !src_is_hlfir_expr) {
                if (auto dstSec = asSectionDesignate(dst)) {
                    auto built = buildSectionToSectionAssign(assign, dstSec);
                    if (!built.empty()) {
                        for (auto &n : built)
                            nodes.push_back(std::move(n));
                        continue;
                    }
                }
            }
            // Whole-array copy: both sides are array boxes / refs (and
            // not an ``hlfir.expr``-producer that we want to walk into).
            if (dst_is_array && src_is_array && !src_is_hlfir_expr) {
                nodes.push_back(buildCopyNode(assign));
                continue;
            }
            // Scalar-zero → array fill: MemsetLibraryNode.
            if (dst_is_array && !src_is_array && isConstantZero(src)) {
                nodes.push_back(buildMemsetNode(assign));
                continue;
            }

            // Array-section ``res(a:b) = <scalar>`` — detect the LHS
            // hlfir.designate with triplet operands and synthesise a
            // nested loop over the section bounds.  Handled before the
            // elemental dispatch below because Flang emits a plain
            // scalar RHS here (no hlfir.elemental wrapping).
            if (!src_is_array) {
                if (auto sec = asSectionDesignate(dst)) {
                    auto built = buildSectionScalarAssign(assign, sec);
                    if (!built.empty()) {
                        for (auto &built_n : built)
                            nodes.push_back(std::move(built_n));
                        continue;
                    }
                }
                // ``res = <non-zero scalar>`` — broadcast across the whole
                // array.  Memset already handled zero above; synthesise a
                // nested loop here for any other constant / scalar RHS.
                if (dst_is_array) {
                    auto built = buildWholeArrayScalarBroadcast(assign);
                    if (!built.empty()) {
                        for (auto &built_n : built)
                            nodes.push_back(std::move(built_n));
                        continue;
                    }
                }
            }

            // b = <elementwise-expression>  — Flang wraps the RHS in one or
            // more composed hlfir.elemental ops, the outermost of which is
            // the assign's source.  Synthesise a nested loop over the shape
            // instead of treating it as a scalar assign.
            //
            // Special-case: ``b = MERGE(t, f, mask)`` on arrays lowers to
            // ``hlfir.elemental { hlfir.designate; arith.select;
            // yield_element }``.  Detect that exact shape and route to
            // ``MergeLibraryNode`` directly so the per-element select
            // stays inside the library node's expansion (modular —
            // bridge doesn't inline).  Anything more elaborate falls
            // through to ``buildElementalAssign``'s per-element tasklet
            // path (which uses the generic select/cmp fallback).
            //
            // Implicit Fortran kind/type conversion: ``logical :: res``
            // assigned from an integer-valued ``COUNT`` (or any libcall
            // returning a different kind from the destination) puts a
            // ``fir.convert`` between the libcall and the assign.  Peel
            // it here so the dispatch below pattern-matches the real
            // producer, not the convert.
            mlir::Value srcPeeled = src;
            while (auto *pdef = srcPeeled.getDefiningOp()) {
                if (auto cv = mlir::dyn_cast<fir::ConvertOp>(pdef)) {
                    srcPeeled = cv.getValue();
                    continue;
                }
                break;
            }
            if (auto *sd = srcPeeled.getDefiningOp()) {
                if (auto elem = mlir::dyn_cast<hlfir::ElementalOp>(sd)) {
                    auto merge_built = buildMergeLibcall(assign, elem);
                    if (!merge_built.empty()) {
                        for (auto &n : merge_built)
                            nodes.push_back(std::move(n));
                        continue;
                    }
                    for (auto &n : buildElementalAssign(assign, elem))
                        nodes.push_back(std::move(n));
                    continue;
                }
                // Linear-algebra ops are first-class in HLFIR; each lowers
                // to a dedicated DaCe library node.  MatMul's SpecializeMatMul
                // handles matrix-matrix / matrix-vector / vector-matrix via
                // operand rank, so we don't disambiguate here.
                auto srcOpName = sd->getName().getStringRef();
                struct LibEntry {
                    llvm::StringRef op;
                    llvm::StringRef callee;
                };
                static const LibEntry kLibTable[] = {
                    {"hlfir.matmul",      "matmul"},
                    {"hlfir.transpose",   "transpose"},
                    {"hlfir.dot_product", "dot_product"},
                    // Fortran ``COUNT(mask [, dim])`` — routed through
                    // ``CountLibraryNode`` so its ``cast → Reduce``
                    // expansion handles the integer-cast and the
                    // per-target reduction lowering.  ``buildLibCallNode``
                    // picks up the optional ``dim`` operand and threads
                    // it through the ASTNode for ``emit_libcall``.
                    {"hlfir.count",       "count"},
                };
                bool libMatched = false;
                for (auto &e : kLibTable) {
                    if (srcOpName == e.op) {
                        // Mode C: ``hlfir.count`` whose first operand is
                        // an ``hlfir.elemental`` (comparison-as-mask /
                        // compound boolean expression).  Synthesise a
                        // transient int32 mask via a per-element loop,
                        // then route through ``CountLibraryNode``.
                        if (e.op == "hlfir.count" && sd->getNumOperands() > 0) {
                            auto mask_src = sd->getOperand(0);
                            if (auto *ms = mask_src.getDefiningOp()) {
                                if (auto elem_src = mlir::dyn_cast<hlfir::ElementalOp>(ms)) {
                                    auto built = buildElementalCountLibcall(assign, elem_src);
                                    if (!built.empty()) {
                                        for (auto &n : built)
                                            nodes.push_back(std::move(n));
                                        libMatched = true;
                                        break;
                                    }
                                }
                            }
                        }
                        nodes.push_back(buildLibCallNode(assign, sd, e.callee.str()));
                        libMatched = true;
                        break;
                    }
                }
                if (libMatched) continue;

                // Scalar reductions land as their own dedicated op; pattern-
                // match each one and hand the shared reduce-lowering helper
                // the right wcr + identity.
                auto opName = sd->getName().getStringRef();
                struct RedEntry {
                    llvm::StringRef op;
                    llvm::StringRef wcr;       // DaCe wcr lambda string
                    llvm::StringRef identity;  // initial accumulator value
                    llvm::StringRef py_op;     // Python binary op for
                                               // section-reduce loop body;
                                               // empty → fall back to
                                               // buildReduceNode (whole-array)
                };
                static const RedEntry kRedTable[] = {
                    {"hlfir.sum",     "lambda a, b: a + b",    "0",     "+"},
                    {"hlfir.product", "lambda a, b: a * b",    "1",     "*"},
                    // Identity strings use the bare ``inf`` token (not
                    // ``math.inf``) so DaCe's cppunparse — which maps
                    // ``inf`` → ``INFINITY`` via _py2c_reserved — emits
                    // a valid C++ literal in the section-reduce init
                    // tasklet.  The whole-array Reduce path's eval()
                    // namespace is patched with ``inf=math.inf`` for
                    // the same string.
                    {"hlfir.minval",  "lambda a, b: min(a, b)", "inf",  "min"},
                    {"hlfir.maxval",  "lambda a, b: max(a, b)", "-inf", "max"},
                    // Logical reductions — ANY / ALL on ``fir.logical``
                    // arrays (ICON's levelmask / maskflag patterns).
                    {"hlfir.any",     "lambda a, b: a or b",   "False",     "or"},
                    {"hlfir.all",     "lambda a, b: a and b",  "True",      "and"},
                    // ``hlfir.count`` is intentionally absent — handled
                    // in ``kLibTable`` above as a ``CountLibraryNode``
                    // libcall (covers Fortran COUNT's int-cast semantics
                    // and the optional ``dim`` argument).
                };
                bool matched = false;
                for (auto &e : kRedTable) {
                    if (opName == e.op) {
                        // If the reduction source is a section designate
                        // (``mask(lo:hi, jk)``) we can't use DaCe's Reduce
                        // node directly — it reduces whole arrays.  Fall
                        // back to a loop-accumulator lowering when a
                        // Python op is available.
                        bool emitted = false;
                        if (!e.py_op.empty() && sd->getNumOperands() > 0) {
                            auto srcVal = sd->getOperand(0);
                            if (auto *srcOp = srcVal.getDefiningOp()) {
                                if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(srcOp)) {
                                    bool hasTrip = false;
                                    for (bool t : dg.getIsTriplet())
                                        if (t) { hasTrip = true; break; }
                                    if (hasTrip) {
                                        auto built = buildSectionReduceAssign(
                                            assign, dg, e.py_op.str(),
                                            e.identity.str());
                                        if (!built.empty()) {
                                            for (auto &bn : built)
                                                nodes.push_back(std::move(bn));
                                            emitted = true;
                                        }
                                    }
                                }
                            }
                        }
                        // Mode-C for ``hlfir.any`` / ``hlfir.all``: the
                        // reduction's source is an ``hlfir.elemental``
                        // (compound boolean expression), so ``traceToDecl``
                        // returns "" and the plain Reduce path explodes
                        // with ``reduction source '' not registered``.
                        // Materialise the elemental into a transient mask
                        // via a per-element loop (same pattern as Mode-C
                        // COUNT) and route the Reduce over the transient.
                        if (!emitted && (e.op == "hlfir.any" || e.op == "hlfir.all")
                                && sd->getNumOperands() > 0) {
                            auto srcVal = sd->getOperand(0);
                            if (auto *srcOp = srcVal.getDefiningOp())
                                if (auto elem_src = mlir::dyn_cast<hlfir::ElementalOp>(srcOp)) {
                                    auto built = buildElementalAnyAllReduce(
                                        assign, elem_src, e.wcr.str(), e.identity.str());
                                    if (!built.empty()) {
                                        for (auto &bn : built)
                                            nodes.push_back(std::move(bn));
                                        emitted = true;
                                    }
                                }
                        }
                        if (!emitted) {
                            nodes.push_back(buildReduceNode(
                                assign, sd, e.wcr.str(), e.identity.str()));
                        }
                        matched = true;
                        break;
                    }
                }
                if (matched) continue;
            }
            nodes.push_back(buildAssignNode(assign));
            continue;
        }
        if (auto ifOp = mlir::dyn_cast<fir::IfOp>(op)) {
            // Allocatable deallocate-guard: ``fir.if (alloc_status != 0) {
            // fir.freemem, reset box to zero }``.  Carries no observable
            // side effect in the SDFG model (we treat allocatables as
            // single-allocation transients) — skip the whole construct.
            auto isAllocCleanup = [](mlir::Region &region) {
                if (region.empty()) return false;
                bool hasFreemem = false;
                for (auto &op : region.front()) {
                    auto nm = op.getName().getStringRef();
                    if (nm == "fir.freemem") { hasFreemem = true; continue; }
                    if (nm == "fir.box_addr"  || nm == "fir.zero_bits"
                        || nm == "fir.embox"  || nm == "fir.shape"
                        || nm == "fir.store"  || nm == "fir.load"
                        || nm == "fir.if"     || nm == "fir.result"
                        || nm == "arith.constant")
                        continue;
                    return false;
                }
                return hasFreemem;
            };
            if (isAllocCleanup(ifOp.getThenRegion())
                && (ifOp.getElseRegion().empty() || isAllocCleanup(ifOp.getElseRegion()))) {
                continue;
            }
            ASTNode n;
            n.kind = "conditional";
            n.condition = buildBoolExpr(ifOp.getCondition());
            if (!ifOp.getThenRegion().empty())
                n.children = buildAST(ifOp.getThenRegion().front());
            if (!ifOp.getElseRegion().empty())
                n.else_children = buildAST(ifOp.getElseRegion().front());
            nodes.push_back(std::move(n));
            continue;
        }
        if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
            ASTNode n;
            n.kind = "conditional";
            n.condition = buildBoolExpr(ifOp.getCondition());
            if (!ifOp.getThenRegion().empty())
                n.children = buildAST(ifOp.getThenRegion().front());
            if (!ifOp.getElseRegion().empty())
                n.else_children = buildAST(ifOp.getElseRegion().front());
            nodes.push_back(std::move(n));
            continue;
        }
        if (auto call = mlir::dyn_cast<fir::CallOp>(op)) {
            ASTNode n;
            n.kind = "call";
            if (auto ref = call.getCallee()) {
                std::string s; llvm::raw_string_ostream os(s);
                ref->print(os); n.callee = s;
            }
            nodes.push_back(std::move(n));
            continue;
        }
        if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op)) {
            nodes.push_back(buildWhileNode(whileOp));
            continue;
        }
        if (auto sel = mlir::dyn_cast<fir::SelectCaseOp>(op)) {
            nodes.push_back(buildSelectCaseChain(sel));
            continue;
        }
        if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
            // Top-level ``fir.store`` is Flang's lowering for lifted
            // DO / DO-WHILE init (``fir.store %c1 to %i``) and internal
            // scratch counters.  Emit as a plain scalar assign.  Regular
            // ``fir.do_loop``s' internal IV stores never reach here —
            // they live inside the loop's body region, which we walk
            // with the existing do-loop handler that takes care of the
            // IV through ``init_expr`` / ``update_expr``.
            auto memref = st.getMemref();
            auto target = traceToDecl(memref);
            if (target.empty())
                if (auto *md = memref.getDefiningOp())
                    if (mlir::isa<fir::AllocaOp>(md))
                        target = allocaSynthName(memref);
            if (target.empty()) continue;
            auto expr = buildExpr(st.getValue());
            // Drop stores with unresolvable RHS — see note in
            // ``walkSCFBeforeRegion``'s fir.store handler.
            if (expr == "?") continue;
            ASTNode a;
            a.kind = "assign";
            a.target = target;
            a.expr = expr;
            a.target_is_array = false;
            nodes.push_back(std::move(a));
            continue;
        }
    }
    return nodes;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

std::vector<ASTNode> extractAST(mlir::ModuleOp module) {
    // Fresh synthetic-name counters / maps per module so two consecutive
    // extractAST calls don't interleave __sc_5 / __al_2 across unrelated
    // SDFGs.
    kScfValueCounter = 0;
    kScfValueMap.clear();
    kAllocaCounter = 0;
    kAllocaMap.clear();
    kHlfirExprToTransient.clear();
    kLibTmpCounter = 0;
    kPosSymbolRegistry.clear();
    kSynthTransientCounter = 0;
    // Defensive: ``kBoolExprNoSubscripts`` is a context flag that should
    // be ``false`` at module-walk start.  Mode-C helpers toggle it via
    // an RAII guard, but if a previous extractAST was aborted mid-walk
    // (an exception reaches here), the flag could still be set.
    kBoolExprNoSubscripts = false;
    clearAllocAliases();

    std::vector<ASTNode> result;
    module.walk([&](mlir::func::FuncOp func) {
        if (!result.empty()) return;  // first func only
        if (!func.getBody().empty())
            result = buildAST(func.getBody().front());
    });

    // Prepend one ``kind="symbol_init"`` node per registered position
    // symbol.  Each such node tells the Python emitter to add the
    // symbol to the SDFG and stage an interstate-edge load
    // ``<symbol> = <array>[<one_based_idx> - 1]`` ahead of the body.
    // Stable order (sorted by symbol name) keeps the emitted SDFG
    // deterministic across runs.
    // Prepend one ``<arr>_allocated = 0`` init per allocatable so that
    // a ``res = ALLOCATED(arr)`` read BEFORE the first ALLOCATE returns
    // the correct ``0`` instead of whatever DaCe leaves in the
    // uninitialised transient scalar.  Walks the module's declares and
    // collects every one with the ``allocatable`` Fortran attribute;
    // sorted so the order is deterministic across runs.
    {
        std::vector<std::string> allocNames;
        module.walk([&](hlfir::DeclareOp op) {
            auto attrs = op.getFortranAttrs();
            if (!attrs) return;
            if (!bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::allocatable))
                return;
            std::string raw = extractName(op.getUniqName().str());
            if (!raw.empty()) allocNames.push_back(std::move(raw));
        });
        std::sort(allocNames.begin(), allocNames.end());
        allocNames.erase(std::unique(allocNames.begin(), allocNames.end()),
                         allocNames.end());
        if (!allocNames.empty()) {
            std::vector<ASTNode> initNodes;
            initNodes.reserve(allocNames.size());
            for (const auto &n : allocNames) {
                ASTNode init;
                init.kind = "assign";
                init.target = n + "_allocated";
                init.target_is_array = false;
                init.expr = "0";
                initNodes.push_back(std::move(init));
            }
            initNodes.insert(initNodes.end(), result.begin(), result.end());
            result = std::move(initNodes);
        }
    }

    if (!kPosSymbolRegistry.empty()) {
        std::vector<std::tuple<std::string, std::string, int64_t>> entries;
        for (auto &kv : kPosSymbolRegistry) {
            const auto &arr = kv.first.first;
            int64_t idx = kv.first.second;
            entries.emplace_back(kv.second, arr, idx);
        }
        std::sort(entries.begin(), entries.end());
        std::vector<ASTNode> initNodes;
        initNodes.reserve(entries.size());
        for (auto &e : entries) {
            ASTNode init;
            init.kind = "symbol_init";
            init.target = std::get<0>(e);          // symbol name
            init.expr = std::get<1>(e);            // source array name
            init.loop_lower = std::get<2>(e);      // 1-based idx (0 if multi-dim
                                                   // packed name; the array name
                                                   // already encodes the dims)
            initNodes.push_back(std::move(init));
        }
        initNodes.insert(initNodes.end(), result.begin(), result.end());
        result = std::move(initNodes);
    }
    return result;
}
