# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Sink the store of a running value into a scan that is read only at its last element.

Shape (cp2k ``cumsum(...)[-1]``)::

    for i in range(N):              for i in range(N):
        acc = acc + x[i]      ->        acc = acc + x[i]
        scan[i] = acc               scan[N - 1] = acc

Every element but the last is dead, so the per-iteration stores go. The loop is pinned sequential:
a cumsum fixes the summation order, and reassociating it changes results (cp2k flips a branch).

Contract, per loop ``L`` (``i`` from 0, step 1, last value ``e``, no break/continue) and transient
array ``scan``:

* the store ``scan[..., i, ...] = acc`` is in a state directly in ``L``, copies a one-element
  ``acc``, and every other index of it is a constant;
* ``acc`` is written in ``L`` only in that state, and not after the store;
* ``scan`` is written in ``L`` only by the store and read only outside ``L``, always at the store's
  index with ``i := e`` (interstate reads included); it is not named in loop or branch code, and the
  symbols of ``e`` are never assigned.

Zero trips: ``i`` starts at 0, so ``e < 0`` and the original read ``scan[e]`` is out of bounds.
"""

from typing import Any

from dace import SDFG, data, properties, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.lift_loop_carried_reduction import _copy_input_connector


def _control_flow_reads(sdfg: SDFG) -> tuple[set[str], dict[str, list[tuple[Any, Range]]]]:
    """Names in loop/branch code of ``sdfg`` (not nested SDFGs), and the array reads on its interstate edges with
    the region owning each edge."""
    names: set[str] = set()
    reads: dict[str, list[tuple[Any, Range]]] = {}
    for region in sdfg.all_control_flow_regions():
        for edge in region.edges():
            for memlet in edge.data.get_read_memlets(sdfg.arrays):
                reads.setdefault(memlet.data, []).append((region, memlet.subset))
        if isinstance(region, LoopRegion):
            for code in (region.loop_condition, region.init_statement, region.update_statement):
                if code is not None:
                    names |= code.get_free_symbols()
        elif isinstance(region, ConditionalBlock):
            for condition, _ in region.branches:
                if condition is not None:
                    names |= condition.get_free_symbols()
    return names, reads


def _assigned_symbols(sdfg: SDFG) -> set[str]:
    """Symbols assigned on an interstate edge or bound by a loop of ``sdfg``."""
    names: set[str] = set()
    for edge in sdfg.all_interstate_edges():
        names |= set(edge.data.assignments)
    for region in sdfg.all_control_flow_regions():
        if isinstance(region, LoopRegion) and region.loop_variable:
            names.add(region.loop_variable)
    return names


def _same_subset(a: Range, b: Range) -> bool:
    if a == b:
        return True
    if len(a) != len(b):
        return False
    return all(
        all(symbolic.equal(x, y, is_length=False) is True for x, y in zip(ra, rb)) for ra, rb in zip(a.ranges, b.ranges)
    )


def _store_source(state: SDFGState, edge: MultiConnectorEdge[Memlet]) -> nodes.AccessNode | None:
    """The one-element access node the store ``edge`` copies, directly or through a copy tasklet."""
    src = edge.src
    if isinstance(src, nodes.Tasklet):
        conn = _copy_input_connector(src)
        if conn is None or state.in_degree(src) != 1 or state.out_degree(src) != 1:
            return None
        src_edge = state.in_edges(src)[0]
        if src_edge.dst_conn != conn:
            return None
        src = src_edge.src
    if not isinstance(src, nodes.AccessNode):
        return None
    desc = state.sdfg.arrays[src.data]
    if not isinstance(desc, (data.Scalar, data.Array)) or isinstance(desc, data.View) or desc.total_size != 1:
        return None
    return src


def _last_subset(subset: Range, loop_var: str, end) -> Range | None:
    """``subset`` with ``loop_var := end`` if it is a point indexed by ``loop_var`` in exactly one dimension."""
    if subset.num_elements() != 1:
        return None
    axis = None
    for k, (b, e, s) in enumerate(subset.ranges):
        free = set(symbolic.symlist(b))
        if free == {loop_var} and str(b) == loop_var:
            if axis is not None:
                return None
            axis = k
        elif free:
            return None
    if axis is None:
        return None
    ranges = list(subset.ranges)
    ranges[axis] = (end, end, 1)
    return Range(ranges)


@properties.make_properties
class SinkLastOnlyScanStore(ppl.Pass):
    """Replace a loop's stores into a scan read only at its last element by one store after it; pin the loop."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        count = 0
        for g in sdfg.all_sdfgs_recursive():
            loops = [r for r in g.all_control_flow_regions() if isinstance(r, LoopRegion)]
            if loops:
                count += self._apply_sdfg(g, loops)
        return count or None

    def _apply_sdfg(self, sdfg: SDFG, loops: list[LoopRegion]) -> int:
        # One walk: every access node of a transient array, by name
        accesses: dict[str, list[tuple[SDFGState, nodes.AccessNode]]] = {}
        for state in sdfg.states():
            for node in state.data_nodes():
                desc = sdfg.arrays[node.data]
                if desc.transient and type(desc) is data.Array:
                    accesses.setdefault(node.data, []).append((state, node))
        if not accesses:
            return 0
        named, edge_reads = _control_flow_reads(sdfg)
        assigned = _assigned_symbols(sdfg)
        count = 0
        for loop in loops:
            count += self._apply_loop(sdfg, loop, accesses, named, edge_reads, assigned)
        return count

    def _apply_loop(self, sdfg, loop, accesses, named, edge_reads, assigned) -> int:
        loop_var = loop.loop_variable
        if not loop_var or loop_analysis.loop_jumps(loop):
            return 0
        init, stride, end = (
            loop_analysis.get_init_assignment(loop),
            loop_analysis.get_loop_stride(loop),
            loop_analysis.get_loop_end(loop),
        )
        if init is None or stride is None or end is None:
            return 0
        if (
            symbolic.equal(init, 0, is_length=False) is not True
            or symbolic.equal(stride, 1, is_length=False) is not True
        ):
            return 0
        if set(symbolic.symlist(end)) & assigned:
            return 0
        loop_states = set(loop.states())
        loop_regions = set(loop.all_control_flow_regions())
        stores = []  # (state, store edge, acc node, last subset)
        for state in loop.nodes():
            if not isinstance(state, SDFGState):
                continue
            for node in state.data_nodes():
                if node.data not in accesses or node.data in named or state.in_degree(node) != 1:
                    continue
                edge = state.in_edges(node)[0]
                if edge.data.wcr is not None or edge.data.data is None:
                    continue
                last = _last_subset(edge.data.get_dst_subset(edge, state), loop_var, end)
                acc = _store_source(state, edge) if last is not None else None
                if acc is None or not self._scan_ok(sdfg, node, last, loop_states, accesses[node.data]):
                    continue
                if any(r in loop_regions or not _same_subset(sub, last) for r, sub in edge_reads.get(node.data, ())):
                    continue
                if not self._acc_final(state, acc, loop_states):
                    continue
                stores.append((state, edge, acc, last))
        if not stores:
            return 0
        loop.pinned_sequential = True
        after = loop.parent_graph.add_state_after(loop, label=f"{loop.label}_last")
        for state, edge, acc, last in stores:
            scan = edge.dst
            if isinstance(edge.src, nodes.Tasklet):
                tasklet = edge.src
                state.remove_node(tasklet)
            else:
                state.remove_edge(edge)
            for node in (scan, acc):
                if state.degree(node) == 0:
                    state.remove_node(node)
            src = after.add_read(acc.data)
            dst = after.add_write(scan.data)
            after.add_nedge(
                src, dst, Memlet(data=scan.data, subset=last, other_subset=Range.from_array(sdfg.arrays[acc.data]))
            )
            accesses[scan.data] = [(s, n) for s, n in accesses[scan.data] if n is not scan] + [(after, dst)]
        return len(stores)

    @staticmethod
    def _scan_ok(sdfg, store_node, last, loop_states, scan_accesses) -> bool:
        """``scan`` is written in the loop only by the store and read only outside it, at ``last``."""
        desc = sdfg.arrays[store_node.data]
        if any(symbolic.equal(o, 0, is_length=False) is not True for o in desc.offset):
            return False
        for state, node in scan_accesses:
            in_loop = state in loop_states
            if in_loop and node is not store_node and state.in_degree(node) > 0:
                return False
            for out in state.out_edges(node):
                if out.data.is_empty():
                    continue
                if in_loop:
                    return False
                read = out.data.get_src_subset(out, state)
                if read is None or not _same_subset(read, last):
                    return False
        return True

    @staticmethod
    def _acc_final(store_state, acc, loop_states) -> bool:
        """``acc`` is written in the loop only in ``store_state``, and not downstream of the store's read."""
        downstream = set(store_state.bfs_nodes(acc)) - {acc}
        for state in loop_states:
            for node in state.data_nodes():
                if node.data != acc.data or state.in_degree(node) == 0:
                    continue
                if state is not store_state or node in downstream:
                    return False
        return True

    def report(self, pass_retval: int) -> str:
        return f"Sank {pass_retval} store(s) into scans read only at their last element."
