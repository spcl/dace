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
    collect_indirect,
    indirect_to_dace,
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
        ctx.flush(builder)
        ctx.ensure(region)
        dst = region.add_state(f"post_{n.target}_{builder.nid()}")
        region.add_edge(ctx.cur, dst, InterstateEdge(assignments={n.target: n.expr}))
        ctx.cur = dst
        return
    if n.target_is_array or assign_reads_array(n, builder.arrays):
        ctx.flush(builder, region)
        ctx.ensure(region)
        emit_tasklet(builder, ctx.cur, n, builder.nid(), ctx.iter_map)
        return
    ctx.pending.append((n.target, n.expr))


def emit_loop(builder, ctx: '_Ctx', n, region, iter_map=None):
    """Fortran DO loop → LoopRegion with exact Fortran bounds."""
    ctx.flush(builder)
    if iter_map is None:
        iter_map = {}

    uid = f"{n.loop_iter}_{builder.nid()}"
    iter_map = {**iter_map, n.loop_iter: uid}

    bound = n.loop_bound
    # Prefer the string form when non-empty (section-assign / symbolic
    # lowers); fall back to the int form for fir.do_loop bounds that
    # Flang resolved to a constant.
    lower = n.loop_lower_expr if n.loop_lower_expr else (n.loop_lower if n.loop_lower >= 0 else 1)

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
        # Indirect accesses in the body turn into fresh SDFG symbols; the
        # value is assigned on an interstate edge so a new state is forced
        # before the compute tasklet runs.
        indirect_syms = collect_indirect(builder, child_assigns)
        if indirect_syms:
            pre = loop.add_state(f"pre_{builder.nid()}")
            body = loop.add_state('body')
            assigns = {sym: indirect_to_dace(builder, expr, iter_map) for expr, sym in indirect_syms.items()}
            for sym in indirect_syms.values():
                if sym not in ctx.sdfg.symbols:
                    ctx.sdfg.add_symbol(sym, dace.int64)
            loop.add_edge(pre, body, InterstateEdge(assignments=assigns))
        else:
            body = loop.add_state('body')
        for idx, a in enumerate(child_assigns):
            emit_tasklet(builder, body, a, idx, iter_map, indirect_syms)


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
