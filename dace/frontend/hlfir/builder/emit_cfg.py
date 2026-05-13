"""Control-flow-graph emission: assign, loop, while, conditional.

These produce the SDFG's CFG skeleton  --  LoopRegions, ConditionalBlocks,
and interstate-edge state-change assignments for symbol writes.  The
actual per-element compute lives in ``emit_tasklet``; this module only
stitches states and regions together.
"""
from __future__ import annotations

import re

import dace
from dace import InterstateEdge
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion

from dace.frontend.hlfir.builder.access import (
    array_read_to_dace_expr,
    collect_indirect,
    find_array_subscripts,
    indirect_to_dace,
)
from dace.frontend.hlfir.builder.context import _Ctx
from dace.frontend.hlfir.builder.descriptors import auto_declare_synth
from dace.frontend.hlfir.builder.emit_tasklet import assign_reads_array, emit_tasklet


def _anchor_views_referenced_in_expr(builder, expr: str, region, pre, sdfg):
    """Ensure every ``view_alias`` array referenced (by name) in ``expr``
    has at least one real AccessNode in a state upstream of the
    interstate edge that will carry ``expr``.

    DaCe's framecode scans interstate-edge free_symbols and synthesises
    a bare ``AccessNode`` if the symbol is an array name with no real
    node yet  --  which then trips ``allocate_view -> get_view_edge ->
    state.in_edges`` because the synthetic node isn't in any state.
    The anchor state's call to ``acc()`` registers the view's
    ``src -> view`` linking memlet, so framecode finds a real instance
    first in topological order.

    Section-alias dummies are excluded  --  they have no SDFG descriptor
    and ``expr`` should have already been rewritten through
    ``_rewrite_section_aliases_in_expr`` before this runs.

    Returns the (possibly updated) ``pre`` state to chain off.
    """
    if not isinstance(expr, str):
        return pre
    from dace.frontend.hlfir.builder.access import acc
    view_aliases = {nm for nm, v in builder.arrays.items() if getattr(v, 'role', '') == 'view_alias'}
    if not view_aliases:
        return pre
    referenced = [nm for nm in view_aliases if re.search(rf'\b{re.escape(nm)}\b', expr)]
    if not referenced:
        return pre
    anchor = region.add_state(f"view_anchor_{builder.nid()}")
    region.add_edge(pre, anchor, InterstateEdge())
    for nm in referenced:
        acc(builder, anchor, nm)
    return anchor


def _rewrite_section_aliases_in_expr(builder, expr: str) -> str:
    """Rewrite ``dummy[i, j]`` to ``source[i, j, k_const]`` for every
    section_alias dummy referenced in ``expr``.

    Used by emit_cond / emit_loop when condition / bound expressions
    get lifted onto interstate-edge assignments  --  the dummy itself has
    no SDFG descriptor, so a bare ``dummy`` symbol in the edge's free
    symbols would trip ``sdfg.arglist`` (KeyError) and DaCe's
    allocation-lifetime tracker.

    The input subscripts are 0-based DaCe-form (built by
    ``buildExprWithSubscripts`` as ``(idx) - 1``); ``view_dim_map``'s
    scalar slots are 1-based Fortran-form, so we subtract 1 when
    splicing them in.
    """
    if not isinstance(expr, str) or '[' not in expr:
        return expr
    section_dummies = {nm for nm, v in builder.arrays.items() if getattr(v, 'role', '') == 'section_alias'}
    if not section_dummies:
        return expr
    matches = list(find_array_subscripts(expr, builder.arrays))
    if not matches:
        return expr
    out = expr
    for start, end, arr, parts in reversed(matches):
        if arr not in section_dummies:
            continue
        v = builder.arrays[arr]
        new_parts = []
        for slot in v.view_dim_map:
            if slot.startswith('_d'):
                try:
                    d_idx = int(slot[2:])
                except ValueError:
                    d_idx = len(new_parts)
                new_parts.append(parts[d_idx] if d_idx < len(parts) else '0')
            else:
                new_parts.append(f"({slot}) - 1")
        out = out[:start] + f"{v.view_source}[{', '.join(new_parts)}]" + out[end:]
    return out


def emit_assign(builder, ctx: '_Ctx', n, region):
    """Scalar or symbol assignment.

    Routes by target kind:
      * ``symbols``    -> interstate-edge assignment that forces a new state.
      * ``array``      -> tasklet via ``emit_tasklet`` with per-occurrence
                         array-read connectors.
      * ``scalar`` whose RHS reads an array element (``s = d(2,1) + 1.0``)
                         -> same tasklet path; the subscripted read needs a
                         real memlet so the codegen sees a connector, not
                         a bare array-pointer identifier.
      * plain ``scalar`` (``i = i + 1``, ``c = 0.5``) -> queued on
                         ``ctx.pending`` for ``emit_scalar_assign`` at
                         flush time.
    """
    # Synthetic scalars (``__sc_N`` / ``__al_N``) from the faithful
    # scf.while walker don't come in as ``hlfir.declare`` ops, so
    # ``add_descriptors`` never saw them.  Register on first assign.
    auto_declare_synth(builder, n.target, ctx)
    if n.target in builder.symbols:
        # Symbol-target reads of an array (``ci0 = icidx(je, jb, 1)``  --
        # the scalar-staged indirection load) need the bridge's bare-name
        # ``n.expr`` (just ``"icidx"``) reconstructed into a full DaCe
        # subscript expression with Fortran->0-based offsets and
        # iter_map remap; otherwise the interstate edge would assign the
        # whole array to a scalar symbol.  Plain symbol writes
        # (``i = i + 1``) keep ``n.expr`` verbatim.
        if assign_reads_array(n, builder.arrays):
            rhs = array_read_to_dace_expr(builder, n, ctx.iter_map)
        else:
            rhs = n.expr
            # Scalar I/O convention: ``intent(inout)`` / ``intent(out)``
            # scalar dummies register in the SDFG as length-1 ``Array``
            # descriptors (so the caller's binding has a writable slot);
            # ``intent(in)`` scalars register as ``Scalar``.  The C ABI
            # binds an Array as ``T*`` and a Scalar as ``T`` (after
            # DaCe's auto-deref), so a bare ``<name>`` reference on the
            # RHS of a symbol-target interstate-edge assignment renders
            # correctly as ``indices_end = endidx`` only when ``endidx``
            # is a Scalar.  For length-1 Arrays we need an explicit
            # ``<name>[0]`` so the codegen sees a scalar value, not the
            # bare pointer.
            rhs_name = rhs.strip() if isinstance(rhs, str) else None
            if rhs_name and rhs_name in ctx.sdfg.arrays:
                desc = ctx.sdfg.arrays[rhs_name]
                if type(desc).__name__ == 'Array':
                    shape = getattr(desc, 'shape', None)
                    if shape is not None and tuple(shape) == (1, ):
                        rhs = f"{rhs_name}[0]"
        ctx.flush(builder)
        ctx.ensure(region)
        dst = region.add_state(f"post_{n.target}_{builder.nid()}")
        region.add_edge(ctx.cur, dst, InterstateEdge(assignments={n.target: rhs}))
        ctx.cur = dst
        return
    if n.target_is_array or assign_reads_array(n, builder.arrays):
        ctx.flush(builder, region)
        ctx.ensure(region)
        # Inline indirect accesses: ``vn(iqidx(je,jb,1), jk, iqblk(je,jb,1))``
        # inside an IF body skips ``emit_loop``'s batch path, so the
        # indirect symbols would otherwise never get minted and the
        # memlet subset would carry the bare array name (which DaCe
        # codegen renders as a pointer-vs-int multiply).  Mint them
        # here, chained one-symbol-per-state so inner indirects are
        # available to outer ones.
        indirect_syms = collect_indirect(builder, [n])
        if indirect_syms:
            for expr, sym in indirect_syms.items():
                rhs = indirect_to_dace(builder, expr, ctx.iter_map, indirect_syms)
                if sym not in ctx.sdfg.symbols:
                    ctx.sdfg.add_symbol(sym, dace.int64)
                nxt = region.add_state(f"sym_{sym}_{builder.nid()}")
                region.add_edge(ctx.cur, nxt, InterstateEdge(assignments={sym: rhs}))
                ctx.cur = nxt
        emit_tasklet(builder, ctx.cur, n, builder.nid(), ctx.iter_map, indirect_syms or None)
        return
    ctx.pending.append((n.target, n.expr))


def emit_symbol_init(builder, ctx: '_Ctx', n, region):
    """Stage a position-array -> SDFG-symbol read at SDFG entry.

    The bridge mints one of these for every ``arr(constant)`` it sees
    used as an array index or section bound (e.g. ``a(pos(1):pos(2))``).
    ``n.target`` is the symbol name (``__sym_pos_1``), ``n.expr`` the
    source array name (``pos``), and ``n.loop_lower`` the 1-based
    Fortran index.  We add the symbol to the SDFG and emit an
    interstate edge ``__sym_pos_1 = pos[0]`` so every memlet whose
    subset references the symbol resolves to a closed-form expression
    rather than a data lookup DaCe can't represent in subset form.
    """
    sym, arr, one_based = n.target, n.expr, int(n.loop_lower)
    if sym not in ctx.sdfg.symbols:
        ctx.sdfg.add_symbol(sym, dace.int64)
    ctx.flush(builder)
    ctx.ensure(region)
    dst = region.add_state(f"sym_init_{sym}_{builder.nid()}")
    # If ``arr`` is a Scalar on the SDFG (the bridge folds length-1
    # transients to Scalar), ``arr[0]`` is invalid -- a Scalar has no
    # subscript.  Drop the subscript so the interstate edge reads the
    # Scalar value directly.  Non-scalar arrays keep the usual 0-based
    # Fortran-to-DaCe index conversion.
    from dace.data import Scalar as _Scalar
    src_desc = ctx.sdfg.arrays.get(arr)
    if isinstance(src_desc, _Scalar):
        read_expr = arr
    else:
        read_expr = f"{arr}[{one_based - 1}]"
    region.add_edge(ctx.cur, dst, InterstateEdge(assignments={sym: read_expr}))
    ctx.cur = dst


def _fortran_subs_to_dace(expr, builder):
    """Rewrite every ``<arr>[<idx>, ...]`` substring in ``expr`` to
    DaCe 0-based form ``<arr>[(<idx>) - offset_<arr>_d<i>, ...]`` for
    each known array.  Used by ``emit_loop`` to convert Fortran-form
    bound expressions (e.g. ``row_ptr[(i_0+1)]``) into valid DaCe
    subscripts before they land in a LoopRegion's init / cond.
    Non-array names (or untracked synthesised arrays) are left
    unchanged.  Walks brackets balanced via ``find_array_subscripts``
    so nested subscripts are handled correctly."""
    if not isinstance(expr, str) or '[' not in expr:
        return expr
    matches = list(find_array_subscripts(expr, builder.arrays))
    if not matches:
        return expr
    out = []
    cursor = 0
    for start, end, arr, parts in matches:
        out.append(expr[cursor:start])
        new_inner = ", ".join(f"({p}) - offset_{arr}_d{d}" for d, p in enumerate(parts))
        out.append(f"{arr}[{new_inner}]")
        cursor = end
    out.append(expr[cursor:])
    return "".join(out)


def _is_trivial_bound(expr: str) -> bool:
    """A bound / condition expression is trivial when it's a bare
    identifier or a single integer literal -- hoisting it to a symbol
    would be pure ceremony.  Anything with operators, brackets, or
    whitespace is non-trivial and gets hoisted."""
    s = expr.strip()
    if not s:
        return True
    # Bare integer literal (incl. signed).
    if s.lstrip('-+').isdigit():
        return True
    # Bare identifier (single name, no operators, no brackets).
    if all(ch.isalnum() or ch == '_' for ch in s) and not s[0].isdigit():
        return True
    return False


def emit_loop(builder, ctx: '_Ctx', n, region, iter_map=None):
    """Fortran DO loop -> LoopRegion with exact Fortran bounds."""
    # Flush any pending scalar assigns from earlier siblings INTO the
    # parent region.  Without ``region`` here, ``ctx.flush`` would land
    # them in ``ctx.sdfg`` (the top-level SDFG)  --  disconnected from the
    # nested loop and orphaned: e.g. ``acc = 0.0d0`` ahead of an inner
    # ``do j = ...`` would surface as a duplicate top-level ``s_*``
    # state with no incoming edge, making the parent CFG's start block
    # ambiguous.
    ctx.flush(builder, region)
    # The bridge no longer uniquifies loop iter names -- the
    # ``UniqueLoopIterators`` post-pass (run from ``SDFGBuilder.build()``
    # via ``_run_post_gen_passes``) renames every ``LoopRegion.loop_var``
    # to a globally-unique ``_loop_it_<N>`` symbol and propagates the
    # rename through the body.  ``emit_loop`` therefore uses the
    # source-Fortran iter name verbatim, and ``iter_map`` is the
    # identity map kept here only so the few callers that still pipe
    # expressions through ``rename_iters`` see a no-op rather than a
    # missing dict.
    if iter_map is None:
        iter_map = dict(ctx.iter_map) if ctx.iter_map else {}

    uid = n.loop_iter

    # ``arr[idx]`` (Fortran 1-based) -> DaCe 0-based form so the
    # LoopRegion's init / cond hit the correct element.
    bound = _fortran_subs_to_dace(n.loop_bound, builder)
    lower_expr = (_fortran_subs_to_dace(n.loop_lower_expr, builder) if n.loop_lower_expr else '')
    lower = lower_expr if lower_expr else (n.loop_lower if n.loop_lower >= 0 else 1)

    # Hoist non-trivial bounds onto pre-LoopRegion symbols so the
    # LoopRegion's init / cond carry only symbol names -- the bridge
    # then doesn't need to embed expression-rewrite logic in bound
    # rendering (the hoisted assignment goes through the same
    # interstate-edge symbol-staging path indirect-array reads use).
    # Bare-symbol bounds are skipped; the staging would be pure noise.
    bound_expr_str = str(bound)
    lower_expr_str = str(lower)
    if not _is_trivial_bound(bound_expr_str):
        sym = f"loopend_{builder.nid()}"
        if sym not in ctx.sdfg.symbols:
            ctx.sdfg.add_symbol(sym, dace.int64)
        ctx.ensure(region)
        nxt = region.add_state(f"pre_{sym}")
        region.add_edge(ctx.cur, nxt, InterstateEdge(assignments={sym: bound_expr_str}))
        ctx.cur = nxt
        bound = sym
    if not _is_trivial_bound(lower_expr_str):
        sym = f"loopbegin_{builder.nid()}"
        if sym not in ctx.sdfg.symbols:
            ctx.sdfg.add_symbol(sym, dace.int64)
        ctx.ensure(region)
        nxt = region.add_state(f"pre_{sym}")
        region.add_edge(ctx.cur, nxt, InterstateEdge(assignments={sym: lower_expr_str}))
        ctx.cur = nxt
        lower = sym

    iter_map = {**iter_map, n.loop_iter: uid}

    # ``DO i = a, b, step`` semantics.  Flang's ``fir.do_loop``
    # carries (lower, upper, step) literally  --  for forward step the
    # iter walks lower->upper inclusive; for negative step the
    # MLIR-level lower is actually the START (e.g. NCLV-1 for ``DO
    # JN = NCLV-1, 1, -1``) and upper is the END (1).  The bridge
    # passes them through as ``loop_lower`` and ``loop_bound`` without
    # reordering, so emit_loop is responsible for picking the right
    # one as init.
    step = getattr(n, 'loop_step', 1)

    if step >= 0:
        loop = LoopRegion(
            label=f"loop_{uid}_{builder.nid()}",
            condition_expr=f"{uid} < {bound} + 1",
            loop_var=uid,
            initialize_expr=f"{uid} = {lower}",
            update_expr=(f"{uid} = {uid} + 1" if step == 1 else f"{uid} = {uid} + {step}"),
        )
    else:
        # Reverse: ``loop_lower`` is the START (the larger value),
        # ``loop_bound`` is the END (the smaller value).  Iter walks
        # from lower DOWN to bound, inclusive.
        loop = LoopRegion(
            label=f"loop_{uid}_{builder.nid()}",
            condition_expr=f"{uid} >= {bound}",
            loop_var=uid,
            initialize_expr=f"{uid} = {lower}",
            update_expr=(f"{uid} = {uid} - 1" if step == -1 else f"{uid} = {uid} + {step}"),
        )
    region.add_node(loop)
    if ctx.cur is not None:
        region.add_edge(ctx.cur, loop, InterstateEdge())
    ctx.cur = loop

    # Cache .children once  --  nanobind copies on every access.
    children = n.children

    child_loops = [c for c in children if c.kind == "loop"]
    child_assigns = [c for c in children if c.kind == "assign"]
    # Anything beyond nested DO loops and plain assignments (IF/ELSE,
    # WHILE, reductions, library-node calls, ...) forces the generic
    # state-machine walk  --  the flat ``body`` tasklet path can't host
    # interstate edges.
    has_structured = any(c.kind not in ("loop", "assign") for c in children)

    if has_structured:
        inner_ctx = _Ctx(ctx.sdfg, builder)
        inner_ctx.iter_map = iter_map
        body_start = loop.add_state(f"body_{builder.nid()}", is_start_block=True)
        inner_ctx.cur = body_start
        builder._emit(inner_ctx, list(children), loop)
        inner_ctx.flush(builder, loop)
    elif child_loops:
        inner_ctx = _Ctx(ctx.sdfg, builder)
        inner_ctx.iter_map = iter_map
        for c in children:
            if c.kind == "loop":
                emit_loop(builder, inner_ctx, c, loop, iter_map)
            elif c.kind == "assign":
                emit_assign(builder, inner_ctx, c, loop)
        inner_ctx.flush(builder)
    elif child_assigns:
        # Inline indirect accesses (``z_kin(edge_idx(jc,k), jk)``) mint a
        # fresh ``<arr>_at<gid>`` SDFG symbol per occurrence; the value is
        # assigned on an interstate edge so a new state is forced before
        # the compute tasklet runs.
        indirect_syms = collect_indirect(builder, child_assigns)

        # Scalar-staged indirection (``ci0 = icidx(je, jb, 1); w(ci0,...)``):
        # the bridge classifies ``ci0`` as a symbol (it feeds an
        # ``hlfir.designate`` index downstream), so the assign cannot land
        # as a tasklet  --  DaCe has no array named ``ci0``.  Lift each such
        # assign onto the same pre->body interstate edge that hosts the
        # inline indirect symbols; the consuming tasklet then reads the
        # symbol value uniformly.  The compute tasklets run on the
        # remaining (array-target) child assigns.
        symbol_assigns = [a for a in child_assigns if a.target in builder.symbols]
        compute_assigns = [a for a in child_assigns if a.target not in builder.symbols]

        # Serialise sibling assigns that share an array as RW  --  an inlined
        # elemental body like ``f = g*g; g = g/(1+g)`` puts both tasklets
        # in one state with no dataflow edge between them; because both
        # access nodes back the same non-transient storage, the scheduler
        # can reorder the write ahead of the read and clobber the value.
        # Use one state per assign whenever such a hazard exists.
        def _raw_hazard(assigns) -> bool:
            write_names_so_far = set()
            for a in assigns:
                reads = {ac.array_name for ac in a.accesses if ac.is_read}
                if reads & write_names_so_far:
                    return True
                for ac in a.accesses:
                    if ac.is_write:
                        write_names_so_far.add(ac.array_name)
                    if ac.is_read and ac.array_name in {ac2.array_name for ac2 in a.accesses if ac2.is_write}:
                        # self-update within one assign  --  fine (handled by
                        # the write-sink logic in emit_tasklet), but any
                        # later sibling that reads the same name must be
                        # in a new state so it sees the updated value.
                        write_names_so_far.add(ac.array_name)
            # Also catch later-writer / earlier-reader patterns that would
            # otherwise race within a single state.
            later_writes = set()
            for a in reversed(assigns):
                reads = {ac.array_name for ac in a.accesses if ac.is_read}
                if reads & later_writes:
                    return True
                for ac in a.accesses:
                    if ac.is_write:
                        later_writes.add(ac.array_name)
            return False

        serialise = _raw_hazard(compute_assigns)

        # For each indirect symbol, emit an ``InterstateEdge`` carrying
        # its assignment.  Nested indirection (``idx1[idx2[idx3[i]]]``)
        # mints one symbol per level, innermost-first; placing each one
        # on its OWN interstate edge in that order lets the outer levels
        # read the inner symbol's value through DaCe's normal symbol
        # propagation -- a single edge with all assignments triggers the
        # race-condition validator because edge-side assignments are
        # treated as parallel.  Empty intermediate states are fine; DaCe
        # collapses the chain at codegen time.
        per_sym_assigns: list[tuple[str, str]] = []
        for expr, sym in indirect_syms.items():
            rhs = indirect_to_dace(builder, expr, iter_map, indirect_syms)
            per_sym_assigns.append((sym, rhs))
            if sym not in ctx.sdfg.symbols:
                ctx.sdfg.add_symbol(sym, dace.int64)
        symbol_assign_pairs: list[tuple[str, str]] = []
        for a in symbol_assigns:
            symbol_assign_pairs.append((a.target, array_read_to_dace_expr(builder, a, iter_map)))

        if per_sym_assigns or symbol_assign_pairs:
            pre = loop.add_state(f"pre_{builder.nid()}")
            cur = pre
            # Indirect symbols first (innermost -> outermost).  Each
            # gets a fresh state so its assignment can reference the
            # symbol set on the previous edge.
            for sym, rhs in per_sym_assigns:
                nxt = loop.add_state(f"sym_{sym}_{builder.nid()}")
                loop.add_edge(cur, nxt, InterstateEdge(assignments={sym: rhs}))
                cur = nxt
            # Stage-staged scalar->symbol writes (``ci0 = icidx(je, jb, 1)``)
            # don't have ordering constraints among themselves, so they
            # can share one final edge.
            body = loop.add_state('body')
            edge = InterstateEdge()
            for tgt, rhs in symbol_assign_pairs:
                edge.assignments[tgt] = rhs
            loop.add_edge(cur, body, edge)
        else:
            body = loop.add_state('body')

        if not serialise:
            for idx, a in enumerate(compute_assigns):
                emit_tasklet(builder, body, a, idx, iter_map, indirect_syms)
        else:
            prev = body
            for idx, a in enumerate(compute_assigns):
                if idx == 0:
                    emit_tasklet(builder, prev, a, idx, iter_map, indirect_syms)
                    continue
                nxt = loop.add_state(f"body_{builder.nid()}")
                loop.add_edge(prev, nxt, InterstateEdge())
                emit_tasklet(builder, nxt, a, idx, iter_map, indirect_syms)
                prev = nxt


def emit_while(builder, ctx: '_Ctx', n, region):
    """Fortran ``DO WHILE``  --  lifted by ``lift-cf-to-scf`` into scf.while
    and extracted as ``kind="while"``.  Emit a DaCe LoopRegion whose
    condition is ``True`` (the bridge's faithful walker folds any
    break-on-false into a ``break`` child node inside the body).
    """
    ctx.flush(builder)
    # ``?`` is the bridge's placeholder for an unextractable condition.
    # Default to ``True`` so ast.parse succeeds and leaves the faithful
    # structure visible in the SDFG for inspection.
    cond = n.condition if n.condition and n.condition != "?" else "True"
    loop = LoopRegion(label=f"while_{builder.nid()}", condition_expr=cond)
    region.add_node(loop)
    if ctx.cur is not None:
        region.add_edge(ctx.cur, loop, InterstateEdge())
    ctx.cur = loop

    body_start = loop.add_state(f"while_body_{builder.nid()}", is_start_block=True)
    inner_ctx = _Ctx(ctx.sdfg, builder)
    inner_ctx.cur = body_start
    builder._emit(inner_ctx, list(n.children), loop)
    inner_ctx.flush(builder, loop)


def emit_cond(builder, ctx: '_Ctx', n, region):
    """``if (cond) then ... else ... end if`` -> ``ConditionalBlock`` with
    a ``ControlFlowRegion`` per branch.  Subsequent statements land in a
    fresh successor state wired from the block.
    """
    ctx.flush(builder, region)
    ctx.ensure(region)
    pre = ctx.cur

    cond = n.condition if n.condition and n.condition != "?" else "True"
    # Section-alias dummies (trivial section slices) have no SDFG
    # descriptor  --  rewrite ``dummy[i, j]`` references in the condition
    # to ``source[i, j, k_const]`` via the view_dim_map.  Without this,
    # the interstate-edge assignment carries the dummy name as a free
    # symbol and ``sdfg.arglist`` raises a KeyError when scanning.
    cond = _rewrite_section_aliases_in_expr(builder, cond)
    # Scalar OUTPUTS land as size-1 Arrays on the SDFG signature, so
    # referring to a bare name in a branch condition would pick up the
    # array pointer.  Subscript each one to read element 0.  Scalar
    # INPUTS (``intent(in)`` / ``VALUE``) are true Scalars and need no
    # subscript -- they're addressable as the bare name in C++.
    for nm, v in builder.scalars.items():
        if v.intent in ('out', 'inout'):
            cond = re.sub(rf'\b{re.escape(nm)}\b', f"{nm}[0]", cond)

    # Hoist non-trivial conditions to a pre-state symbol so the
    # ConditionalBlock branch carries only a symbol name -- one path
    # for every IF lowering, no per-branch expression-rewrite logic.
    # Trivial cases (a bare name or ``True`` / ``False``) skip the
    # staging.
    if not _is_trivial_bound(cond):
        sym = f"if_cond_{builder.nid()}"
        if sym not in ctx.sdfg.symbols:
            ctx.sdfg.add_symbol(sym, dace.int64)
        # If the condition references any view_alias array, anchor it
        # in a state upstream of the interstate-edge assignment so
        # DaCe's framecode finds a real AccessNode first.
        pre = _anchor_views_referenced_in_expr(builder, cond, region, pre, ctx.sdfg)
        nxt = region.add_state(f"pre_{sym}")
        region.add_edge(pre, nxt, InterstateEdge(assignments={sym: cond}))
        pre = nxt
        ctx.cur = nxt
        cond = sym

    uid = builder.nid()
    cond_block = ConditionalBlock(f"if_{uid}")
    region.add_node(cond_block, ensure_unique_name=True)
    if pre is not None:
        region.add_edge(pre, cond_block, InterstateEdge())

    def _populate_branch(label: str, children: list) -> ControlFlowRegion:
        branch = ControlFlowRegion(label, sdfg=ctx.sdfg)
        inner = _Ctx(ctx.sdfg, builder)
        inner.iter_map = ctx.iter_map
        builder._emit(inner, children, branch)
        inner.flush(builder, branch)
        # An empty branch (e.g. the EXIT arm of a Flang-lowered DO+EXIT)
        # still needs a start block, otherwise the validator complains.
        if len(branch.nodes()) == 0:
            branch.add_state(f"{label}_noop", is_start_block=True)
        return branch

    then_region = _populate_branch(f"if_{uid}_then", list(n.children))
    cond_block.add_branch(cond, then_region)

    else_children = list(n.else_children)
    if else_children:
        else_region = _populate_branch(f"if_{uid}_else", else_children)
        cond_block.add_branch(None, else_region)

    # The ConditionalBlock is itself the "current" control-flow node;
    # subsequent statements get a fresh state edge-connected to it.
    ctx.cur = cond_block
