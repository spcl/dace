# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drop a loop-carried store that a LATER iteration overwrites before anyone reads it."""

from typing import Any, NamedTuple

import sympy

from dace import SDFG, properties, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.narrowing import as_basic, as_expr
from dace.sdfg.state import LoopRegion, SDFGState
from dace.subsets import Subset
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis


class CarriedStore(NamedTuple):
    """A store whose value a later iteration overwrites unread."""

    name: str
    dead_edge: MultiConnectorEdge[Memlet]
    distance: int


def symbol_named(expr: sympy.Basic, name: str) -> sympy.Symbol | None:
    """The symbol OBJECT called ``name`` inside ``expr``, or ``None``."""
    return next((s for s in expr.free_symbols if str(s) == name), None)


def constant_offset_on_axis(subset: Subset, loop_var: str) -> tuple[int, int] | None:
    """``(axis, offset)`` for a subset that is the single point ``i + offset`` on exactly one axis and loop-invariant
    everywhere else, else ``None``."""
    found: tuple[int, int] | None = None
    for axis, (begin, end, step) in enumerate(subset.ndrange()):
        begin, end = symbolic.pystr_to_symbolic(begin), symbolic.pystr_to_symbolic(end)
        if (
            symbolic.simplify(as_expr(end) - as_expr(begin)) != 0
            or symbolic.simplify(as_expr(symbolic.pystr_to_symbolic(step)) - 1) != 0
        ):
            return None
        ivar = symbol_named(begin, loop_var)
        if ivar is None:
            continue  # loop-invariant index: fine on any axis, and not the carried one
        if found is not None:
            return None  # the loop variable steers two axes; the overlap is not a shift
        offset = symbolic.simplify(begin - ivar)
        if not as_basic(offset).is_Integer:
            return None
        found = (axis, int(as_expr(offset)))
    return found


def body_is_analyzable(loop: LoopRegion) -> list[SDFGState] | None:
    """The body as a straight-line chain of states, or ``None`` for anything this pass will not read."""
    blocks = list(loop.nodes())
    if not blocks or any(not isinstance(b, SDFGState) for b in blocks):
        return None
    for edge in loop.edges():
        if edge.data.assignments or not edge.data.is_unconditional():
            return None
    if any(loop.out_degree(b) > 1 or loop.in_degree(b) > 1 for b in blocks):
        return None
    order: list[SDFGState] = []
    node = loop.start_block
    seen = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        order.append(node)
        succ = list(loop.successors(node))
        node = succ[0] if succ else None
    if len(order) != len(blocks):
        return None  # not one chain
    if any(isinstance(n, nodes.NestedSDFG) for st in order for n in st.nodes()):
        return None
    return order


def find_killed_store(loop: LoopRegion, body: list[SDFGState], stride: int) -> tuple[SDFGState, CarriedStore] | None:
    """The one store the loop provably overwrites unread, with the state owning it, or ``None``."""
    ambiguous: set = set()
    stores: dict[str, list[tuple[SDFGState, Any, int, int]]] = {}
    for state in body:
        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode):
                continue
            if state.sdfg.arrays[node.data].transient:
                ambiguous.add(node.data)
            for edge in state.in_edges(node):
                written = edge.data.get_dst_subset(edge, state)
                # Another output of the tasklet still needs it, which the drop could not keep valid.
                shared = isinstance(edge.src, nodes.Tasklet) and len(edge.src.out_connectors) > 1
                # A dynamic store may not happen, so it kills nothing.
                if edge.data.wcr is not None or edge.data.dynamic or written is None or shared:
                    ambiguous.add(node.data)
                    continue
                spot = constant_offset_on_axis(written, loop.loop_variable)
                if spot is None:
                    ambiguous.add(node.data)
                    continue
                stores.setdefault(node.data, []).append((state, edge, spot[0], spot[1]))

    for name, found in stores.items():
        if name in ambiguous or len(found) < 2:
            continue
        for dead_state, dead_edge, dead_axis, dead_off in found:
            for _, kill_edge, kill_axis, kill_off in found:
                if kill_edge is dead_edge or kill_axis != dead_axis or kill_off >= dead_off:
                    continue
                # The other axes name one element only if they are the same.
                dead_at, kill_at = dead_edge.data.subset.ranges, kill_edge.data.subset.ranges
                if any(a != b for d, (a, b) in enumerate(zip(dead_at, kill_at)) if d != dead_axis):
                    continue
                if (dead_off - kill_off) % stride != 0:
                    continue
                if not reads_are_clear(
                    body, dead_state, name, loop.loop_variable, dead_axis, kill_off, dead_off, dead_edge
                ):
                    continue
                return dead_state, CarriedStore(
                    name=name,
                    dead_edge=dead_edge,
                    distance=(dead_off - kill_off) // stride,
                )
    return None


def reads_are_clear(
    body: list[SDFGState],
    dead_state: SDFGState,
    name: str,
    loop_var: str,
    axis: int,
    kill_off: int,
    dead_off: int,
    dead_edge: MultiConnectorEdge[Memlet],
) -> bool:
    """No read of ``name`` observes the dead store -- in a later iteration, or in this one."""
    for state in body:
        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode) or node.data != name:
                continue
            for edge in state.out_edges(node):
                read = edge.data.get_src_subset(edge, state)
                if read is None:
                    return False
                spot = constant_offset_on_axis(read, loop_var)
                if spot is None or spot[0] != axis:
                    return False
                if kill_off < spot[1] < dead_off:
                    return False
                if spot[1] == kill_off:
                    # The kill READS the location it writes (``a[i] += ...``, which the frontend lowers to an explicit
                    # read/write rather than a WCR).
                    return False
    # A read at the dead store's OWN offset is only safe when it is that store's read-modify-write input
    producer = dead_edge.src
    for state in body:
        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode) or node.data != name:
                continue
            for edge in state.out_edges(node):
                read = edge.data.get_src_subset(edge, state)
                if read is None:
                    return False
                spot = constant_offset_on_axis(read, loop_var)
                if spot is None or spot[0] != axis or spot[1] != dead_off:
                    continue
                order = body.index(state) - body.index(dead_state)
                if order < 0:
                    # Strictly BEFORE the store: it reads what was there on entry.
                    continue
                if order > 0 or not reaches(state, edge.dst, producer):
                    return False
    # Any later state reading the array at all is a read-after-write within the iteration.
    for state in body[body.index(dead_state) + 1 :]:
        if any(isinstance(n, nodes.AccessNode) and n.data == name and state.out_degree(n) > 0 for n in state.nodes()):
            return False
    downstream, frontier = set(), [dead_edge.dst]
    while frontier:
        for out in dead_state.out_edges(frontier.pop()):
            if out.dst not in downstream:
                downstream.add(out.dst)
                frontier.append(out.dst)
    return not any(isinstance(n, nodes.AccessNode) and n.data == name for n in downstream)


def reaches(state: SDFGState, start: nodes.Node, target: nodes.Node) -> bool:
    """``True`` iff ``target`` is ``start`` or lies downstream of it inside ``state``."""
    seen, frontier = {id(start)}, [start]
    while frontier:
        node = frontier.pop()
        if node is target:
            return True
        for edge in state.out_edges(node):
            if id(edge.dst) not in seen:
                seen.add(id(edge.dst))
                frontier.append(edge.dst)
    return False


def drop_store(state: SDFGState, edge: MultiConnectorEdge[Memlet]) -> None:
    """Remove a write edge and everything that existed only to feed it."""
    producer = edge.src
    state.remove_edge(edge)
    frontier = [producer, edge.dst]
    while frontier:
        node = frontier.pop()
        if node not in state.nodes():
            continue
        if state.out_degree(node) > 0:
            continue
        if isinstance(node, nodes.AccessNode) and not state.sdfg.arrays[node.data].transient:
            if state.in_degree(node) > 0:
                continue  # a real store into a program argument -- not ours to remove
        elif not isinstance(node, (nodes.Tasklet, nodes.AccessNode)):
            continue
        for in_edge in list(state.in_edges(node)):
            state.remove_edge(in_edge)
            frontier.append(in_edge.src)
        state.remove_node(node)


@properties.make_properties
class DeadCarriedStoreElimination(ppl.Pass):
    """Peel the tail iterations whose store survives, then drop the store from the rest."""

    CATEGORY: str = "Optimization Preparation"

    max_peel = properties.Property(
        dtype=int,
        default=4,
        desc=(
            "Largest kill distance to act on. The peel costs one copy of the body per iteration, "
            "so a far-reaching kill would trade a store for more code than it saves; it also "
            "bounds the assumption that the loop runs more times than it peels."
        ),
    )

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        from dace.transformation.interstate.loop_peeling import LoopPeeling

        count = 0
        for loop in [
            b
            for g in sdfg.all_sdfgs_recursive()
            for b in g.all_control_flow_regions(recursive=True)
            if isinstance(b, LoopRegion)
        ]:
            stride = loop_analysis.get_loop_stride(loop)
            if stride is None or not symbolic.pystr_to_symbolic(stride).is_Integer:
                continue
            stride = int(as_expr(symbolic.pystr_to_symbolic(stride)))
            # A descending loop kills FORWARD; the tail peel anchors on the loop end, an iterate only at stride one.
            if stride != 1 or loop.inverted:
                continue
            body = body_is_analyzable(loop)
            if body is None:
                continue
            match = find_killed_store(loop, body, stride)
            if match is None or not 1 <= match[1].distance <= int(self.max_peel):
                continue
            # Peel FIRST: the tail iterations keep the store, and if the peel refuses (an
            # infeasible or too-short loop) nothing has been removed yet.
            try:
                LoopPeeling().apply_to(
                    sdfg=loop.sdfg,
                    loop=loop,
                    verify=False,
                    options={"count": match[1].distance, "begin": False, "guarded": True},
                )
            except Exception:  # noqa: BLE001 -- an unpeelable loop keeps its store
                continue
            # ``apply_to`` rewrites the region in place, so re-find the store in what remains
            # rather than trusting an edge handle the peel may have replaced.
            body = body_is_analyzable(loop)
            if body is None:
                continue
            again = find_killed_store(loop, body, stride)
            if again is None:
                continue
            drop_store(again[0], again[1].dead_edge)
            count += 1
        if count:
            # The store's operands are computed into TRANSIENTS, often several states upstream (``a_slice``,
            # ``a_slice_times_d_slice``), and that chain still reads ``a[i + 1]`` while the loop writes ``a[i]`` -- a
            # loop-carried dependence that keeps LoopToMap away from a loop which no longer has one.
            from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination

            ppl.Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
        return count or None

    def report(self, pass_retval: int) -> str:
        return f"Dropped {pass_retval} loop-carried store(s) a later iteration overwrites."
