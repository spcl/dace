"""Control-flow-graph emission: assign, loop, while, conditional.

These produce the SDFG's CFG skeleton — LoopRegions, ConditionalBlocks,
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
    rename_iters,
)
from dace.frontend.hlfir.builder.context import _Ctx
from dace.frontend.hlfir.builder.descriptors import auto_declare_synth
from dace.frontend.hlfir.builder.emit_tasklet import assign_reads_array, emit_tasklet


def emit_assign(builder, ctx: '_Ctx', n, region):
    """Scalar or symbol assignment.

    Routes by target kind:
      * ``symbols``    → interstate-edge assignment that forces a new state.
      * ``array``      → tasklet via ``emit_tasklet`` with per-occurrence
                         array-read connectors.
      * ``scalar`` whose RHS reads an array element (``s = d(2,1) + 1.0``)
                         → same tasklet path; the subscripted read needs a
                         real memlet so the codegen sees a connector, not
                         a bare array-pointer identifier.
      * plain ``scalar`` (``i = i + 1``, ``c = 0.5``) → queued on
                         ``ctx.pending`` for ``emit_scalar_assign`` at
                         flush time.
    """
    # Synthetic scalars (``__sc_N`` / ``__al_N``) from the faithful
    # scf.while walker don't come in as ``hlfir.declare`` ops, so
    # ``add_descriptors`` never saw them.  Register on first assign.
    auto_declare_synth(builder, n.target, ctx)
    if n.target in builder.symbols:
        # Symbol-target reads of an array (``ci0 = icidx(je, jb, 1)`` —
        # the scalar-staged indirection load) need the bridge's bare-name
        # ``n.expr`` (just ``"icidx"``) reconstructed into a full DaCe
        # subscript expression with Fortran→0-based offsets and
        # iter_map remap; otherwise the interstate edge would assign the
        # whole array to a scalar symbol.  Plain symbol writes
        # (``i = i + 1``) keep ``n.expr`` verbatim.
        if assign_reads_array(n, builder.arrays):
            rhs = array_read_to_dace_expr(builder, n, ctx.iter_map)
        else:
            rhs = n.expr
        ctx.flush(builder)
        ctx.ensure(region)
        dst = region.add_state(f"post_{n.target}_{builder.nid()}")
        region.add_edge(ctx.cur, dst, InterstateEdge(assignments={n.target: rhs}))
        ctx.cur = dst
        return
    if n.target_is_array or assign_reads_array(n, builder.arrays):
        ctx.flush(builder, region)
        ctx.ensure(region)
        emit_tasklet(builder, ctx.cur, n, builder.nid(), ctx.iter_map)
        return
    ctx.pending.append((n.target, n.expr))


def emit_symbol_init(builder, ctx: '_Ctx', n, region):
    """Stage a position-array → SDFG-symbol read at SDFG entry.

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
    region.add_edge(ctx.cur, dst, InterstateEdge(assignments={sym: f"{arr}[{one_based - 1}]"}))
    ctx.cur = dst


def _fortran_subs_to_dace(expr, builder):
    """Rewrite every ``<arr>[<idx>, …]`` substring in ``expr`` to
    DaCe 0-based form ``<arr>[(<idx>) - offset_<arr>_d<i>, …]`` for
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


def emit_loop(builder, ctx: '_Ctx', n, region, iter_map=None):
    """Fortran DO loop → LoopRegion with exact Fortran bounds."""
    # Flush any pending scalar assigns from earlier siblings INTO the
    # parent region.  Without ``region`` here, ``ctx.flush`` would land
    # them in ``ctx.sdfg`` (the top-level SDFG) — disconnected from the
    # nested loop and orphaned: e.g. ``acc = 0.0d0`` ahead of an inner
    # ``do j = ...`` would surface as a duplicate top-level ``s_*``
    # state with no incoming edge, making the parent CFG's start block
    # ambiguous.
    ctx.flush(builder, region)
    if iter_map is None:
        iter_map = {}

    uid = f"{n.loop_iter}_{builder.nid()}"

    # Apply the OUTER iter_map to bound expressions BEFORE adding our
    # own rename: the outer loop's iter (e.g. ``i`` → ``i_0``) may
    # appear inside our bound (``do j = row_ptr(i), row_ptr(i+1)-1``),
    # but our own iter ``j`` cannot legally appear in our bounds.
    # Any embedded ``arr[idx]`` (Fortran 1-based) is then converted to
    # DaCe 0-based form so the LoopRegion's init / cond hit the correct
    # element.
    bound = _fortran_subs_to_dace(rename_iters(n.loop_bound, iter_map), builder)
    lower_expr = (_fortran_subs_to_dace(rename_iters(n.loop_lower_expr, iter_map), builder)
                  if n.loop_lower_expr else '')
    lower = lower_expr if lower_expr else (n.loop_lower if n.loop_lower >= 0 else 1)

    iter_map = {**iter_map, n.loop_iter: uid}

    loop = LoopRegion(
        label=f"loop_{uid}",
        condition_expr=f"{uid} < {bound} + 1",
        loop_var=uid,
        initialize_expr=f"{uid} = {lower}",
        update_expr=f"{uid} = {uid} + 1",
    )
    region.add_node(loop)
    if ctx.cur is not None:
        region.add_edge(ctx.cur, loop, InterstateEdge())
    ctx.cur = loop

    # Cache .children once — nanobind copies on every access.
    children = n.children
    child_loops = [c for c in children if c.kind == "loop"]
    child_assigns = [c for c in children if c.kind == "assign"]
    # Anything beyond nested DO loops and plain assignments (IF/ELSE,
    # WHILE, reductions, library-node calls, …) forces the generic
    # state-machine walk — the flat ``body`` tasklet path can't host
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
        # as a tasklet — DaCe has no array named ``ci0``.  Lift each such
        # assign onto the same pre→body interstate edge that hosts the
        # inline indirect symbols; the consuming tasklet then reads the
        # symbol value uniformly.  The compute tasklets run on the
        # remaining (array-target) child assigns.
        symbol_assigns = [a for a in child_assigns if a.target in builder.symbols]
        compute_assigns = [a for a in child_assigns if a.target not in builder.symbols]

        # Serialise sibling assigns that share an array as RW — an inlined
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
                        # self-update within one assign — fine (handled by
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

        edge_assigns = {}
        for expr, sym in indirect_syms.items():
            edge_assigns[sym] = indirect_to_dace(builder, expr, iter_map)
            if sym not in ctx.sdfg.symbols:
                ctx.sdfg.add_symbol(sym, dace.int64)
        for a in symbol_assigns:
            edge_assigns[a.target] = array_read_to_dace_expr(builder, a, iter_map)

        if edge_assigns:
            pre = loop.add_state(f"pre_{builder.nid()}")
            body = loop.add_state('body')
            loop.add_edge(pre, body, InterstateEdge(assignments=edge_assigns))
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
    """Fortran ``DO WHILE`` — lifted by ``lift-cf-to-scf`` into scf.while
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
    """``if (cond) then ... else ... end if`` → ``ConditionalBlock`` with
    a ``ControlFlowRegion`` per branch.  Subsequent statements land in a
    fresh successor state wired from the block.
    """
    ctx.flush(builder, region)
    ctx.ensure(region)
    pre = ctx.cur

    cond = n.condition if n.condition and n.condition != "?" else "True"
    # Substitute Fortran iterator names with their unique DaCe loop-var
    # names (``i_0`` etc.) picked by the enclosing ``emit_loop``.
    for fname, uname in ctx.iter_map.items():
        cond = re.sub(rf'\b{re.escape(fname)}\b', uname, cond)
    # Scalars with intent land as size-1 Arrays on the SDFG signature,
    # so referring to a bare name in a branch condition would pick up
    # the array pointer.  Subscript each one to read element 0.
    for nm, v in builder.scalars.items():
        if v.intent:
            cond = re.sub(rf'\b{re.escape(nm)}\b', f"{nm}[0]", cond)

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
