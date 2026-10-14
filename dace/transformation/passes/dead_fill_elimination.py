# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drop a memset whose array the next map overwrites element for element before anyone reads it.

``s = np.zeros(N); for i in range(N): s[i] = r[i] - alpha * v[i]`` zeroes ``s`` only for the map
to replace every zero. Inside an iterative solver's loop (``bicgstab`` holds two of them) that is
an N-element store per iteration with no observer, and one the CPU transfer rule runs on a single
core because the loop re-enters it.

The memset is dead when, in its own state:

* it fills the whole of a transient array, and its access node passes no data on -- every edge out
  of it is an empty ordering edge;
* a later access node of the same array, reachable from it, takes a map's write that covers every
  element exactly once: the inner write is ``array[p0, p1, ...]`` with one distinct map parameter
  per axis, each ranging over that axis' whole extent, without WCR;
* the map does not read the array, and every other access node of the array in the state lies
  downstream of the overwrite.

Anything else (a partial or indirect write, a conditional one, a read in between, a nested SDFG
writing it) keeps the memset.
"""

from typing import Any

from dace import SDFG, data, properties, symbolic
from dace.libraries.standard.nodes.fill import FillLibraryNode
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl


def covers_every_element(state: SDFGState, written: nodes.AccessNode, desc: data.Array) -> bool:
    """Whether ``written``'s only writer is a map storing each element of ``desc`` exactly once, unread."""
    in_edges = state.in_edges(written)
    if len(in_edges) != 1 or not isinstance(in_edges[0].src, nodes.MapExit):
        return False
    outer = in_edges[0]
    if outer.data.dynamic or outer.data.wcr is not None or outer.src_conn is None:
        return False
    exit_node = outer.src
    entry = state.entry_node(exit_node)
    inner = list(state.in_edges_by_connector(exit_node, "IN_" + outer.src_conn[len("OUT_") :]))
    if len(inner) != 1 or inner[0].data.dynamic or inner[0].data.wcr is not None or inner[0].data.subset is None:
        return False
    ranges = dict(zip(entry.map.params, entry.map.range.ranges))
    seen = set()
    for (begin, end, step), extent in zip(inner[0].data.subset.ranges, desc.shape):
        param = str(begin)
        if begin != end or param not in ranges or param in seen:
            return False
        seen.add(param)
        low, high, stride = ranges[param]
        if symbolic.simplify(low) != 0 or stride != 1 or symbolic.simplify(high + 1 - extent) != 0:
            return False
    # One parameter per axis and no other: a parameter left over repeats the stores, so it is still a map whose
    # iterations outnumber the elements -- kept, not reasoned about.
    if len(seen) != len(desc.shape) or len(entry.map.params) != len(seen):
        return False
    return all(e.data.data != written.data for e in state.in_edges(entry))


def dead_fill(state: SDFGState, fill: FillLibraryNode) -> nodes.AccessNode | None:
    """The access node ``fill`` writes, when the fill is dead; ``None`` otherwise."""
    if any(not e.data.is_empty() for e in state.in_edges(fill)):
        return None
    out_edges = state.out_edges(fill)
    if len(out_edges) != 1 or not isinstance(out_edges[0].dst, nodes.AccessNode):
        return None
    zeroed = out_edges[0].dst
    sdfg = state.sdfg
    desc = sdfg.arrays[zeroed.data]
    if not desc.transient or not isinstance(desc, data.Array) or isinstance(desc, data.View):
        return None
    subset = out_edges[0].data.subset
    if subset is None or symbolic.simplify(subset.num_elements() - desc.total_size) != 0:
        return None
    if any(not e.data.is_empty() for e in state.out_edges(zeroed)) or len(state.in_edges(zeroed)) != 1:
        return None
    others = [n for n in state.data_nodes() if n.data == zeroed.data and n is not zeroed]
    reachable = set(state.bfs_nodes(zeroed))
    for written in others:
        if written not in reachable or not covers_every_element(state, written, desc):
            continue
        downstream = set(state.bfs_nodes(written))
        if all(n is written or n in downstream for n in others):
            return zeroed
    return None


@properties.make_properties
class DeadFillElimination(ppl.Pass):
    """Remove every memset a map overwrites in full, unread, in the same state."""

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Nodes)

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Drop the dead memsets of ``sdfg`` and its nested SDFGs.

        :returns: how many memsets were removed, or ``None`` if none were.
        """
        removed = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                for fill in [n for n in state.nodes() if isinstance(n, FillLibraryNode)]:
                    zeroed = dead_fill(state, fill)
                    if zeroed is None:
                        continue
                    state.remove_node(fill)
                    state.remove_node(zeroed)
                    removed += 1
        return removed or None
