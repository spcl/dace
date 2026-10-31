# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The statement model of a wavefront candidate: an outer loop whose body is a chain of statements.

A statement is an inner ``LoopRegion``, a one-parameter top-level Map, or a top-level copy with one
varying axis. One instance of a statement is one point ``(u, v)`` of its domain, ``u`` the outer
iterator and ``v`` the statement's own. Every memlet the instance touches is an :class:`Access`
over its cells; enclosing loops and maps inside the instance are existential dimensions of it.
"""

from dataclasses import dataclass, field
from typing import Any

import sympy

from dace import SDFG, data, subsets
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.analysis import cfg
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.narrowing import as_expr, free_symbols, ndrange_exprs, simplified
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.interstate.loop_to_map import carried_local_transients
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.parallelization.wavefront import polyhedron as poly

#: Suffix ``BreakAntiDependence`` gives the per-iteration anti-dependence snapshot it inserts
#: (``arr`` -> ``arr_split_snap``); the skew folds such a snapshot back into the live array.
SPLIT_SNAP_SUFFIX = "_split_snap"

#: ``(name, lo, hi)`` of an existential dimension; ``None`` bounds leave that side open.
Dim = tuple[str, sympy.Expr | None, sympy.Expr | None]


@dataclass
class Access:
    """One memlet of a statement instance: cells ``lo[a] <= c[a] <= hi[a]`` of ``data``, an open side ``None``."""

    data: str
    write: bool
    lo: list[sympy.Expr | None]
    hi: list[sympy.Expr | None]
    ctx: list[Dim]
    guard: list[sympy.Expr]
    snapshot: bool = False

    def key(self) -> tuple[Any, ...]:
        return (self.data, self.write, str(self.lo), str(self.hi), str(self.ctx), str(self.guard), self.snapshot)


@dataclass
class Statement:
    """One statement of the outer body; ``order`` is its position in program order."""

    order: int
    kind: str  # "loop" | "map" | "copy"
    block: LoopRegion | SDFGState
    var: str
    lo: sympy.Expr
    hi: sympy.Expr
    node: nodes.MapEntry | None = None
    edge: MultiConnectorEdge[Memlet] | None = None
    accesses: list[Access] = field(default_factory=list)


@dataclass
class Snapshot:
    """A ``live -> snap`` copy state the skew absorbs: reads of ``snap`` become reads of ``live``."""

    order: int
    state: SDFGState
    snap: str
    live: str
    window: subsets.Range


@dataclass
class Body:
    """The outer loop ``u in [u_lo, u_hi]`` and its statements in program order."""

    outer: LoopRegion
    u: str
    u_lo: sympy.Expr
    u_hi: sympy.Expr
    statements: list[Statement]
    snapshots: list[Snapshot]
    #: Symbols assigned anywhere inside the outer loop: per-iteration values, never parameters.
    assigned: set[str]


def interval(lo: Any, hi: Any, step: Any) -> tuple[sympy.Expr | None, sympy.Expr | None]:
    """The inclusive hull of ``lo, lo + step, ..`` towards ``hi``; open when the step's sign is unknown."""
    st = simplified(as_expr(step))
    if st.is_Integer and int(st) > 0:
        return as_expr(lo), as_expr(hi)
    if st.is_Integer and int(st) < 0:
        return as_expr(hi), as_expr(lo)
    return None, None


def loop_header(loop: LoopRegion) -> tuple[sympy.Expr, sympy.Expr, sympy.Expr] | None:
    """``(init, inclusive end, stride)`` of a for-loop, or ``None``."""
    if not loop.loop_variable:
        return None
    lo = loop_analysis.get_init_assignment(loop)
    hi = loop_analysis.get_loop_end(loop)
    st = loop_analysis.get_loop_stride(loop)
    if lo is None or hi is None or st is None:
        return None
    return as_expr(lo), as_expr(hi), simplified(as_expr(st))


def unit_loop_bounds(loop: LoopRegion) -> tuple[sympy.Expr, sympy.Expr] | None:
    """``(init, inclusive end)`` of a non-inverted unit-stride for-loop, else ``None``."""
    header = loop_header(loop)
    if header is None or loop.inverted or header[2] != 1:
        return None
    return header[0], header[1]


def loop_context(block: ControlFlowBlock, stop: ControlFlowRegion) -> list[Dim]:
    """The for-loops strictly between ``block`` and ``stop``, as dimensions over their hulls."""
    ctx: list[Dim] = []
    g = block.parent_graph
    while g is not None and g is not stop:
        if isinstance(g, LoopRegion) and g.loop_variable:
            header = loop_header(g)
            if header is None or g.inverted:
                ctx.append((g.loop_variable, None, None))
            else:
                ctx.append((g.loop_variable, *interval(*header)))
        g = g.parent_graph
    return ctx


def scope_context(state: SDFGState, node: nodes.Node, stop: nodes.EntryNode | None) -> list[Dim] | None:
    """The map parameters enclosing ``node`` below ``stop``, over their hulls; ``None`` under a non-map scope."""
    ctx: list[Dim] = []
    cur = state.entry_node(node)
    while cur is not None and cur is not stop:
        if not isinstance(cur, nodes.MapEntry):
            return None
        for param, (lo, hi, step) in zip(cur.map.params, ndrange_exprs(cur.map.range)):
            ctx.append((param, *interval(lo, hi, step)))
        cur = state.entry_node(cur)
    return ctx


def branch_guard(state: SDFGState, stop: ControlFlowRegion) -> list[sympy.Expr]:
    """The branch conditions ``state`` runs under inside ``stop``, each ``>= 0``; empty when not convex."""
    return poly.constraints_from_condition(cfg.collect_enclosing_conditions(state, stop=stop)) or []


def cells(subset: subsets.Subset) -> tuple[list[sympy.Expr | None], list[sympy.Expr | None]]:
    """Per-axis inclusive hull of ``subset``."""
    lo: list[sympy.Expr | None] = []
    hi: list[sympy.Expr | None] = []
    for start, end, step in ndrange_exprs(subset):
        a, b = interval(start, end, step) if start != end else (start, end)
        lo.append(a)
        hi.append(b)
    return lo, hi


def edge_accesses(state: SDFGState, e: MultiConnectorEdge[Memlet]) -> list[tuple[str, bool, subsets.Subset]] | None:
    """``(container, is_write, subset)`` of what the memlet ``e`` itself moves. An edge into or out of a scope
    boundary moves nothing of its own (the leaves inside say what is touched), nor does the pass-through of a
    container into an access node of the same container. ``None`` when a side is unreadable."""
    m = e.data
    if m is None or m.is_empty():
        return []
    src, dst = e.src, e.dst
    src_an, dst_an = isinstance(src, nodes.AccessNode), isinstance(dst, nodes.AccessNode)
    if isinstance(src, nodes.ExitNode) or isinstance(dst, nodes.EntryNode):
        if isinstance(src, nodes.CodeNode) or isinstance(dst, nodes.CodeNode):
            return None  # a code node feeding a scope through no container
        return []
    out: list[tuple[str, bool, subsets.Subset | None]] = []
    if src_an and dst_an:
        out = [(src.data, False, m.get_src_subset(e, state)), (dst.data, True, m.get_dst_subset(e, state))]
    elif src_an and isinstance(dst, nodes.CodeNode):
        out = [(src.data, False, m.get_src_subset(e, state))]
    elif isinstance(src, nodes.CodeNode) and dst_an:
        out = [(dst.data, True, m.get_dst_subset(e, state))]
    elif isinstance(src, nodes.EntryNode) and isinstance(dst, nodes.CodeNode):
        out = [(m.data, False, m.subset)]
    elif isinstance(src, nodes.CodeNode) and isinstance(dst, nodes.ExitNode):
        out = [(m.data, True, m.subset)]
    elif isinstance(src, nodes.EntryNode) and dst_an:
        if m.data != dst.data:  # copy-in to a scope-local container
            out = [(m.data, False, m.subset), (dst.data, True, m.get_dst_subset(e, state))]
    elif src_an and isinstance(dst, nodes.ExitNode):
        if m.data != src.data:  # copy-out of a scope-local container
            out = [(src.data, False, m.get_src_subset(e, state)), (m.data, True, m.subset)]
    elif not (isinstance(src, nodes.EntryNode) or isinstance(dst, nodes.ExitNode)):
        return None
    full = []
    for name, write, sub in out:
        if sub is None:  # the implicit side of a copy spans the whole container
            sub = subsets.Range.from_array(state.sdfg.arrays[name])
        full.append((name, write, sub))
    return full


def anchor(e: MultiConnectorEdge[Memlet]) -> nodes.Node:
    """The node whose scope encloses what ``e`` moves."""
    for n in (e.dst, e.src):
        if isinstance(n, nodes.CodeNode):
            return n
    return e.src if isinstance(e.src, nodes.AccessNode) else e.dst


def state_accesses(
    state: SDFGState,
    edges: list[MultiConnectorEdge[Memlet]],
    stop_scope: nodes.EntryNode | None,
    outer_ctx: list[Dim],
    guard: list[sympy.Expr],
) -> list[Access] | None:
    """The accesses of ``edges``, each under the maps below ``stop_scope`` that enclose it."""
    out: list[Access] = []
    for e in edges:
        found = edge_accesses(state, e)
        if found is None:
            return None
        if not found:
            continue
        ctx = scope_context(state, anchor(e), stop_scope)
        if ctx is None:
            return None
        for name, write, subset in found:
            lo, hi = cells(subset)
            out.append(Access(name, write, lo, hi, [*outer_ctx, *ctx], guard))
    return out


def body_chain(outer: LoopRegion) -> list[ControlFlowBlock] | None:
    """``outer``'s blocks in order when they form one unconditional chain without assignments."""
    start = outer.start_block
    chain: list[ControlFlowBlock] = []
    seen: set[ControlFlowBlock] = set()
    block = start
    while block is not None:
        if block in seen:
            return None
        seen.add(block)
        chain.append(block)
        out = outer.out_edges(block)
        if len(out) > 1:
            return None
        if not out:
            break
        if out[0].data.assignments or not out[0].data.is_unconditional():
            return None
        block = out[0].dst
    if len(chain) != outer.number_of_nodes():
        return None
    return chain


def split_snapshot_window(state: SDFGState) -> subsets.Range | None:
    """The window a pure ``arr -> arr_split_snap`` identity copy state covers, else ``None``."""
    ns = list(state.nodes())
    edges = list(state.edges())
    if len(ns) != 2 or len(edges) != 1 or not all(isinstance(n, nodes.AccessNode) for n in ns):
        return None
    e = edges[0]
    if e.dst.data != f"{e.src.data}{SPLIT_SNAP_SUFFIX}" or e.data is None or e.data.is_empty():
        return None
    src_desc, dst_desc = state.sdfg.arrays.get(e.src.data), state.sdfg.arrays.get(e.dst.data)
    if src_desc is None or dst_desc is None or len(src_desc.shape) != len(dst_desc.shape):
        return None
    if any(simplified(a - b) != 0 for a, b in zip(src_desc.shape, dst_desc.shape)):
        return None
    src_sub = e.data.get_src_subset(e, state)
    dst_sub = e.data.get_dst_subset(e, state)
    if not isinstance(src_sub, subsets.Range) or (dst_sub is not None and dst_sub != src_sub):
        return None
    if any(simplified(step - 1) != 0 for (_, _, step) in ndrange_exprs(src_sub)):
        return None
    return src_sub


def copy_axis(state: SDFGState, e: MultiConnectorEdge[Memlet]) -> tuple[int, int, sympy.Expr] | None:
    """``(src axis, dst axis, width)`` of a copy that varies along exactly one axis on each side."""
    src_sub, dst_sub = e.data.get_src_subset(e, state), e.data.get_dst_subset(e, state)
    if src_sub is None or dst_sub is None:
        return None
    sides = []
    for sub in (src_sub, dst_sub):
        varying = [
            (ax, simplified(end - start + 1))
            for ax, (start, end, step) in enumerate(ndrange_exprs(sub))
            if start != end
        ]
        if len(varying) != 1 or any(step != 1 for _, _, step in ndrange_exprs(sub)):
            return None
        sides.append(varying[0])
    if simplified(sides[0][1] - sides[1][1]) != 0:
        return None
    return sides[0][0], sides[1][0], sides[0][1]


def copy_var(order: int) -> str:
    """The iterator name a copy statement's element index is modelled under."""
    return f"_wf_copy_{order}"


def state_statements(state: SDFGState, order: int) -> list[Statement] | None:
    """The map and copy statements of a top-level state in a topological order, or ``None`` if it holds other work."""
    found: list[tuple[int, Statement]] = []
    rank = {n: i for i, n in enumerate(sdutil.dfs_topological_sort(state))}
    for n in state.nodes():
        if state.entry_node(n) is not None:
            continue
        if isinstance(n, nodes.MapEntry):
            if len(n.map.params) != 1 or any(not c.startswith("IN_") for c in n.in_connectors):
                return None  # several axes, or a range read from data
            lo, hi, step = ndrange_exprs(n.map.range)[0]
            if step != 1:
                return None
            found.append((rank[n], Statement(0, "map", state, n.map.params[0], lo, hi, node=n)))
        elif isinstance(n, nodes.MapExit):
            continue
        elif not isinstance(n, nodes.AccessNode):
            return None
    for e in state.edges():
        if not (isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode)):
            continue
        if state.entry_node(e.src) is not None or e.data is None or e.data.is_empty():
            continue
        axes = copy_axis(state, e)
        if axes is None:
            return None
        found.append((rank[e.dst], Statement(0, "copy", state, "", as_expr(0), simplified(axes[2] - 1), edge=e)))
    found.sort(key=lambda item: item[0])
    out = []
    for i, (_, st) in enumerate(found):
        st.order = order + i
        if st.kind == "copy":
            st.var = copy_var(st.order)
        out.append(st)
    return out


def map_accesses(st: Statement) -> list[Access] | None:
    """The accesses one iteration of a map statement makes."""
    state = st.block
    assert isinstance(state, SDFGState) and st.node is not None
    scope = state.scope_subgraph(st.node, include_entry=True, include_exit=True)
    return state_accesses(state, list(scope.edges()), st.node, [], [])


def copy_accesses(st: Statement) -> list[Access] | None:
    """The read and the write of one element of a copy statement."""
    state, e = st.block, st.edge
    assert isinstance(state, SDFGState) and e is not None
    axes = copy_axis(state, e)
    if axes is None:
        return None
    c = as_expr(st.var)
    out = []
    for name, write, sub, ax in (
        (e.src.data, False, e.data.get_src_subset(e, state), axes[0]),
        (e.dst.data, True, e.data.get_dst_subset(e, state), axes[1]),
    ):
        lo, hi = cells(sub)
        lo[ax] = hi[ax] = simplified(lo[ax] + c)
        out.append(Access(name, write, lo, hi, [], []))
    return out


def loop_accesses(st: Statement) -> list[Access] | None:
    """The accesses one iteration of a loop statement makes, over every state below it."""
    loop = st.block
    assert isinstance(loop, LoopRegion)
    out: list[Access] = []
    for state in loop.states():
        found = state_accesses(state, list(state.edges()), None, loop_context(state, loop), branch_guard(state, loop))
        if found is None:
            return None
        out += found
    return out


def private_containers(st: Statement, sdfg: SDFG, occurrences: dict[str, set[SDFGState]]) -> set[str]:
    """Transients every instance of ``st`` writes before it reads them and nothing else touches."""
    if st.kind == "copy":
        return set()
    if st.kind == "map":
        state = st.block
        assert isinstance(state, SDFGState) and st.node is not None
        inside = set(state.scope_subgraph(st.node, include_entry=False, include_exit=False).nodes())
        names = {n.data for n in inside if isinstance(n, nodes.AccessNode)}
        return {
            name
            for name in names
            if sdfg.arrays[name].transient
            and occurrences.get(name, set()) == {state}
            and all(n in inside for n in state.data_nodes() if n.data == name)
        }
    loop = st.block
    assert isinstance(loop, LoopRegion)
    states = set(loop.states())
    candidates = {
        name
        for state in states
        for name in (n.data for n in state.data_nodes())
        if sdfg.arrays[name].transient and occurrences.get(name, set()) <= states
    }
    if not candidates:
        return set()
    return candidates - carried_local_transients(loop, candidates)


def occurrence_index(sdfg: SDFG) -> dict[str, set[SDFGState]]:
    """Every container of ``sdfg`` mapped to the states holding an access node of it."""
    occ: dict[str, set[SDFGState]] = {}
    for state in sdfg.states():
        for n in state.data_nodes():
            occ.setdefault(n.data, set()).add(state)
    return occ


def assigned_symbols(outer: LoopRegion) -> set[str]:
    """Symbols an interstate edge inside ``outer`` assigns."""
    return {name for e in outer.all_interstate_edges(recursive=True) for name in e.data.assignments}


def extract_body(outer: LoopRegion, sdfg: SDFG, occurrences: dict[str, set[SDFGState]]) -> Body | None:
    """The statement model of ``outer``, or ``None`` when its body is not a chain of statements."""
    bounds = unit_loop_bounds(outer)
    if bounds is None:
        return None
    chain = body_chain(outer)
    if chain is None:
        return None
    statements: list[Statement] = []
    snapshots: list[Snapshot] = []
    order = 0
    for block in chain:
        if isinstance(block, LoopRegion):
            vb = unit_loop_bounds(block)
            if vb is None:
                return None
            statements.append(Statement(order, "loop", block, block.loop_variable, vb[0], vb[1]))
            order += 1
        elif isinstance(block, SDFGState):
            window = split_snapshot_window(block)
            if window is not None:
                e = next(iter(block.edges()))
                snapshots.append(Snapshot(order, block, e.dst.data, e.src.data, window))
                order += 1
                continue
            found = state_statements(block, order)
            if found is None:
                return None
            statements += found
            order += len(found)
        else:
            return None
    if not any(st.kind == "loop" for st in statements):
        return None  # every map statement is parallel at fixed ``u``: no wavefront to expose
    for st in statements:
        found = {"loop": loop_accesses, "map": map_accesses, "copy": copy_accesses}[st.kind](st)
        if found is None:
            return None
        private = private_containers(st, sdfg, occurrences)
        dedup: dict[tuple[Any, ...], Access] = {}
        for acc in found:
            if acc.data in private:
                continue
            desc = sdfg.arrays.get(acc.data)
            if desc is None or isinstance(desc, (data.View, data.Reference)):
                return None
            dedup.setdefault(acc.key(), acc)
        st.accesses = list(dedup.values())
    if not attribute_snapshots(snapshots, statements, occurrences):
        return None
    return Body(outer, outer.loop_variable, bounds[0], bounds[1], statements, snapshots, assigned_symbols(outer))


def is_time_step_loop(body: Body) -> bool:
    """Whether the outer iterator indexes nothing: every iteration re-applies the same statements to the same
    cells, a loop over kernel invocations."""
    exprs: list[Any] = []
    for st in body.statements:
        exprs += [st.lo, st.hi]
        for acc in st.accesses:
            exprs += [*acc.lo, *acc.hi, *acc.guard] + [b for _, lo, hi in acc.ctx for b in (lo, hi)]
    return not any(body.u in free_names(e) for e in exprs if e is not None)


def attribute_snapshots(
    snapshots: list[Snapshot], statements: list[Statement], occurrences: dict[str, set[SDFGState]]
) -> bool:
    """Turn every snapshot read into a read of its live array; ``False`` when a snapshot has another use."""
    for snap in snapshots:
        readers = [st for st in statements if any(a.data == snap.snap for a in st.accesses)]
        if any(st.order < snap.order for st in readers):
            return False  # read before this iteration's copy: it holds the previous iteration's values
        allowed = {snap.state} | {s for st in readers for s in statement_states(st)}
        if not occurrences.get(snap.snap, set()) <= allowed:
            return False
        for st in readers:
            for acc in st.accesses:
                if acc.data == snap.snap:
                    if acc.write:
                        return False
                    acc.data, acc.snapshot = snap.live, True
    return True


def statement_states(st: Statement) -> list[SDFGState]:
    """The states a statement spans."""
    if isinstance(st.block, LoopRegion):
        return list(st.block.states())
    return [st.block]


def reaches(st_a: Statement, st_b: Statement) -> bool:
    """Whether ``st_a``'s dataflow precedes ``st_b``'s inside their shared state."""
    state = st_a.block
    assert isinstance(state, SDFGState)
    start = state.exit_node(st_a.node) if st_a.node is not None else st_a.edge.dst
    goal = st_b.node if st_b.node is not None else st_b.edge.src
    seen = {start}
    stack = [start]
    while stack:
        n = stack.pop()
        if n is goal:
            return True
        for e in state.out_edges(n):
            if e.dst not in seen:
                seen.add(e.dst)
                stack.append(e.dst)
    return False


def sequenced(st_a: Statement, st_b: Statement) -> bool:
    """Whether every instance of ``st_a`` at one step may run before every instance of ``st_b`` at the same step."""
    if st_a.order >= st_b.order:
        return False
    if st_a.block is not st_b.block:
        return True
    return reaches(st_a, st_b)


def free_names(expr: Any) -> set[str]:
    return {s.name for s in free_symbols(simplified(as_expr(expr)))}


__all__ = [
    "Access",
    "Body",
    "Snapshot",
    "Statement",
    "extract_body",
    "is_time_step_loop",
    "occurrence_index",
    "sequenced",
]
