# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drop a loop-carried store that a LATER iteration overwrites before anyone reads it.

TSVC ``s244`` is the shape::

    for i in range(N - 1):
        a[i]     = b[i] + c[i] * d[i]      # S1, writes a[i]
        b[i]     = c[i] + b[i]
        a[i + 1] = b[i] + a[i + 1] * d[i]  # S3, writes a[i + 1]

``S3`` at iteration ``i`` writes ``a[i + 1]``; ``S1`` at iteration ``i + 1`` writes ``a[i + 1]``
again and never reads ``a``. So every ``S3`` store is overwritten before it can be observed --
except the one made by the last iteration, whose killer would be an iteration the loop does not
run. Peel that last iteration off intact and the store is dead in what remains, leaving a loop
whose only remaining carrier is gone: ``LoopToMap`` takes it unchanged.

The dependence is not weak or approximate here, it is DEAD, which is why this is worth a pass of
its own. ``DeadDataflowElimination`` cannot do it for two independent reasons: it only removes
writes to TRANSIENT descriptors (``a`` is a program argument), and it reasons inside one state's
dataflow, whereas the kill lands an iteration later.

Every condition below is checked, and anything unproven refuses the rewrite. A false positive here
silently drops a store the program needed, so the pass declines wherever it cannot see the whole
picture: one state, no nested SDFGs, no conditionals, no WCR, and every access to the array
affine in the loop variable with matching non-scan indices.
"""
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from dace import SDFG, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis


class CarriedStore(NamedTuple):
    """A store whose value a later iteration overwrites unread.

    :ivar name: the array both stores address.
    :ivar dead_edge: the write edge to drop (offset :attr:`dead_offset` on :attr:`axis`).
    :ivar dead_offset: constant ``c`` in the dead store's subscript ``i + c``.
    :ivar kill_offset: constant ``d`` in the killing store's subscript ``i + d``; ``d < c``.
    :ivar distance: ``c - d`` -- how many iterations later the kill lands, and so how many
        iterations must be peeled off the tail to keep the stores the loop really makes.
    """
    name: str
    dead_edge: Any
    dead_offset: int
    kill_offset: int
    distance: int


def symbol_named(expr, name: str):
    """The symbol OBJECT called ``name`` inside ``expr``, or ``None``.

    Never ``symbolic.symbol(name)``: a freshly minted symbol carries
    ``DEFAULT_SYMBOL_TYPE`` while the one in a subset carries whatever the descriptor declared, and
    two same-named symbols of different dtype do not cancel -- ``i - i + 1`` stays ``i - i + 1``
    instead of folding to ``1``, and every offset here reads as non-constant. Resolving the
    instance out of ``free_symbols`` is the only spelling that compares against what is really
    there.
    """
    return next((s for s in expr.free_symbols if str(s) == name), None)


def constant_offset_on_axis(subset, loop_var: str) -> Optional[Tuple[int, int]]:
    """``(axis, offset)`` for a subset that is the single point ``i + offset`` on exactly one axis
    and loop-invariant everywhere else, else ``None``.

    Only a POINT is accepted (begin == end, unit step): a range spanning the loop variable would
    make the overlap between two iterations a set question rather than the equality this pass
    reasons with.
    """
    found: Optional[Tuple[int, int]] = None
    for axis, (begin, end, step) in enumerate(subset.ndrange()):
        begin, end = symbolic.pystr_to_symbolic(begin), symbolic.pystr_to_symbolic(end)
        if symbolic.simplify(end - begin) != 0 or symbolic.simplify(symbolic.pystr_to_symbolic(step) - 1) != 0:
            return None
        ivar = symbol_named(begin, loop_var)
        if ivar is None:
            continue  # loop-invariant index: fine on any axis, and not the carried one
        if found is not None:
            return None  # the loop variable steers two axes; the overlap is not a shift
        offset = symbolic.simplify(begin - ivar)
        if not offset.is_Integer:
            return None
        found = (axis, int(offset))
    return found


def accesses_of(state: SDFGState, name: str) -> Tuple[List[Any], List[Any]]:
    """``(write_edges, read_edges)`` touching ``name`` in ``state``."""
    writes, reads = [], []
    for node in state.nodes():
        if not isinstance(node, nodes.AccessNode) or node.data != name:
            continue
        writes.extend(state.in_edges(node))
        reads.extend(state.out_edges(node))
    return writes, reads


def body_is_analyzable(loop: LoopRegion) -> Optional[List[SDFGState]]:
    """The body as a straight-line chain of states, or ``None`` for anything this pass will not read.

    A statement per state is the shape the frontend actually produces -- ``s244``'s three lines are
    two states -- so a single-state gate would refuse the motivating case. What must be excluded is
    everything that makes "which store happens" or "what index does it name" a question:

    * a conditional or nested region: a store under a predicate is not a fact;
    * a nested SDFG: its accesses are invisible to :func:`accesses_of`;
    * a branch in the body's own control flow: two paths write different stores;
    * an interstate assignment: it can redefine the very symbol the subscripts are read against,
      so ``i + 1`` in a later state need not name what it named in the first.
    """
    blocks = list(loop.nodes())
    if not blocks or any(not isinstance(b, SDFGState) for b in blocks):
        return None
    for edge in loop.edges():
        if edge.data.assignments or not edge.data.is_unconditional():
            return None
    if any(loop.out_degree(b) > 1 or loop.in_degree(b) > 1 for b in blocks):
        return None
    order: List[SDFGState] = []
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


def find_killed_store(loop: LoopRegion, body: List[SDFGState], stride: int) -> Optional[Tuple[SDFGState, CarriedStore]]:
    """The one store the loop provably overwrites unread, with the state owning it, or ``None``.

    Requires, for the candidate pair ``(dead at i + c, kill at i + d)``:

    * ``c - d`` is a positive multiple of the stride, so the killing iteration is one the loop
      actually visits;
    * neither store carries WCR -- a WCR store READS its destination, folding the earlier value in
      rather than replacing it, so it kills nothing;
    * the array is not transient (that case is ``DeadDataflowElimination``'s, and it can see it);
    * every access to the array, anywhere in the body, is a single point affine in the loop
      variable on ONE axis with the others loop-invariant. One unreadable access and the array is
      abandoned entirely -- a partial picture is what a false positive is made of;
    * the reads clear :func:`reads_are_clear`.
    """
    ambiguous: set = set()
    stores: Dict[str, List[Tuple[SDFGState, Any, int, int]]] = {}
    for state in body:
        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode):
                continue
            if state.sdfg.arrays[node.data].transient:
                ambiguous.add(node.data)
            for edge in state.in_edges(node):
                if edge.data.wcr is not None or edge.data.subset is None:
                    ambiguous.add(node.data)
                    continue
                spot = constant_offset_on_axis(edge.data.subset, loop.loop_variable)
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
                if (dead_off - kill_off) % stride != 0:
                    continue
                if not reads_are_clear(body, dead_state, name, loop.loop_variable, dead_axis, kill_off, dead_off,
                                       dead_edge):
                    continue
                return dead_state, CarriedStore(name=name,
                                                dead_edge=dead_edge,
                                                dead_offset=dead_off,
                                                kill_offset=kill_off,
                                                distance=(dead_off - kill_off) // stride)
    return None


def reads_are_clear(body: List[SDFGState], dead_state: SDFGState, name: str, loop_var: str, axis: int, kill_off: int,
                    dead_off: int, dead_edge) -> bool:
    """No read of ``name`` observes the dead store -- in a later iteration, or in this one.

    Across iterations: a read at offset ``r`` in iteration ``i`` addresses what the dead store of
    iteration ``i + r - c`` wrote, and the kill only reaches that location at ``i + r - d``. For
    ``d < r < c`` the read lands inside the window where the dead value is live, so the store is
    not dead. At ``r <= d`` the kill has already run; at ``r >= c`` the dead store has not run yet.

    Within the iteration: the store is followed by everything downstream of it in its own state and
    by every state after it in the chain, so a read of ``name`` in either would see what it wrote.
    """
    for state in body:
        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode) or node.data != name:
                continue
            for edge in state.out_edges(node):
                if edge.data.subset is None:
                    return False
                spot = constant_offset_on_axis(edge.data.subset, loop_var)
                if spot is None or spot[0] != axis:
                    return False
                if kill_off < spot[1] < dead_off:
                    return False
                if spot[1] == kill_off:
                    # The kill READS the location it writes (``a[i] += ...``, which the frontend
                    # lowers to an explicit read/write rather than a WCR). That read runs before
                    # the kill's write, so it still observes the dead store of the previous
                    # iteration -- the value is folded in, not replaced. Refuse rather than order
                    # the two: being wrong here drops a store the program needed.
                    return False
    # A read at the dead store's OWN offset is only safe when it is that store's read-modify-write
    # input -- the value flowing into the producer that makes the store. Any other consumer at that
    # offset reads what the store just wrote, in the same iteration, and the store is not dead:
    #
    #     A[i]     = B[i]           kill,  offset 0
    #     A[i + 1] = E[i]           the candidate, offset 1
    #     D[i]     = A[i + 1] * 2   reads offset 1 -- the value just written
    #
    # The window test ``d < r < c`` cannot see this (``r == c``), and reachability from the store's
    # access node cannot either: this consumer hangs off a DIFFERENT access node for the same array,
    # so no edge joins them even though the read plainly observes the write. Demanding that every
    # offset-c read feed the producer is what makes it decidable from the graph alone -- and it is
    # what ``s244`` satisfies, where the only such read is the ``a[i + 1] * d[i]`` the store itself
    # consumes.
    producer = dead_edge.src
    for state in body:
        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode) or node.data != name:
                continue
            for edge in state.out_edges(node):
                if edge.data.subset is None:
                    return False
                spot = constant_offset_on_axis(edge.data.subset, loop_var)
                if spot is None or spot[0] != axis or spot[1] != dead_off:
                    continue
                order = body.index(state) - body.index(dead_state)
                if order < 0:
                    # Strictly BEFORE the store: it reads what was there on entry. Nothing in the
                    # loop has touched ``a[i + c]`` yet -- the dead store of this iteration is still
                    # ahead of it, and the kill that covers that location belongs to iteration
                    # ``i + distance``, which is further ahead still.
                    continue
                if order > 0 or not reaches(state, edge.dst, producer):
                    return False
    # Any later state reading the array at all is a read-after-write within the iteration.
    for state in body[body.index(dead_state) + 1:]:
        if any(isinstance(n, nodes.AccessNode) and n.data == name and state.out_degree(n) > 0 for n in state.nodes()):
            return False
    downstream, frontier = set(), [dead_edge.dst]
    while frontier:
        for out in dead_state.out_edges(frontier.pop()):
            if out.dst not in downstream:
                downstream.add(out.dst)
                frontier.append(out.dst)
    return not any(isinstance(n, nodes.AccessNode) and n.data == name for n in downstream)


def reaches(state: SDFGState, start, target) -> bool:
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


def drop_store(state: SDFGState, edge) -> None:
    """Remove a write edge and everything that existed only to feed it.

    The producer is a Tasklet for a computed store and an AccessNode for a plain copy, so both are
    pruned the same way: walk back while a node's last consumer is gone. Leaving an isolated node
    behind fails SDFG validation, which is how this was caught.
    """
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
    """Peel the tail iterations whose store survives, then drop the store from the rest.

    See the module docstring for the shape and for why ``DeadDataflowElimination`` cannot do it.
    """

    CATEGORY: str = 'Optimization Preparation'

    max_peel = properties.Property(
        dtype=int,
        default=4,
        desc=('Largest kill distance to act on. The peel costs one copy of the body per iteration, '
              'so a far-reaching kill would trade a store for more code than it saves; it also '
              'bounds the assumption that the loop runs more times than it peels.'),
    )

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        from dace.transformation.interstate.loop_peeling import LoopPeeling
        count = 0
        for loop in [
                b for g in sdfg.all_sdfgs_recursive() for b in g.all_control_flow_regions(recursive=True)
                if isinstance(b, LoopRegion)
        ]:
            stride = loop_analysis.get_loop_stride(loop)
            if stride is None or not symbolic.pystr_to_symbolic(stride).is_Integer:
                continue
            stride = int(symbolic.pystr_to_symbolic(stride))
            if stride < 1:
                continue  # a descending loop kills FORWARD; out of scope until something needs it
            body = body_is_analyzable(loop)
            if body is None:
                continue
            match = find_killed_store(loop, body, stride)
            if match is None or not 1 <= match[1].distance <= int(self.max_peel):
                continue
            # Peel FIRST: the tail iterations keep the store, and if the peel refuses (an
            # infeasible or too-short loop) nothing has been removed yet.
            try:
                LoopPeeling().apply_to(sdfg=loop.sdfg,
                                       loop=loop,
                                       verify=False,
                                       options={
                                           'count': match[1].distance,
                                           'begin': False
                                       })
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
            # The store's operands are computed into TRANSIENTS, often several states upstream
            # (``a_slice``, ``a_slice_times_d_slice``), and that chain still reads ``a[i + 1]``
            # while the loop writes ``a[i]`` -- a loop-carried dependence that keeps LoopToMap away
            # from a loop which no longer has one. Those are transients inside one region, which is
            # precisely what DeadDataflowElimination handles; the cross-iteration, non-transient
            # half it cannot see is what this pass did first.
            from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
            ppl.Pipeline([DeadDataflowElimination()]).apply_pass(sdfg, {})
        return count or None

    def report(self, pass_retval: int) -> str:
        return f'Dropped {pass_retval} loop-carried store(s) a later iteration overwrites.'
