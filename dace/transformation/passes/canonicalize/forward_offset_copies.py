# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Read a shifted copy's source at the shifted index instead of keeping the copy (index-remap forwarding).

A stencil written with shifted operands copies its grid once per shift::

    q_i[0:n, :, :] = q[3:n + 3, :, :]        # fv3_dycore: q_i, q_im1, q_j, q_jm1
    ... = f(q_i[i, j, k], q_im1[i, j, k])

The copy is a full-size transient written once and read at the index it was taken at, so every read of
``q_i[x]`` is ``q[x + 3]`` as long as nothing writes ``q`` between the copy and the read. Array elimination folds
such a copy only when every reader shares its state; this pass forwards it across states too, after which the
transient and the copy are gone.

Contract, per copy edge ``A -> B``:

* ``B`` is a transient array written only by this copy, over its whole extent, with the source range of the same
  shape (an offset, no reshape); it is not a view, reaches no nested SDFG, and no interstate edge or condition
  names it;
* every reader of ``B`` is in a state of the copy's control-flow region (an acyclic one), reachable from it;
* no block of that region on a path from the copy to a reader writes ``A``, nor does the copy's state or a
  reader's state.
"""

from collections import defaultdict
from typing import Any

from dace import SDFG, data, properties, subsets
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def _writes(block: ControlFlowBlock, name: str) -> bool:
    """Whether ``block`` (any state inside it) writes ``name``."""
    states = (
        [block]
        if isinstance(block, SDFGState)
        else [b for b in block.all_control_flow_blocks() if isinstance(b, SDFGState)]
    )
    return any(st.in_degree(n) > 0 for st in states for n in st.data_nodes() if n.data == name)


@properties.make_properties
@transformation.explicit_cf_compatible
class ForwardOffsetCopies(ppl.Pass):
    """Forward every offset copy the module contract admits to its readers."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.AccessNodes)

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        forwarded = 0
        for g in sdfg.all_sdfgs_recursive():
            forwarded += self.apply_to_sdfg(g)
        return forwarded or None

    def apply_to_sdfg(self, sdfg: SDFG) -> int:
        named = set()
        for edge in sdfg.all_interstate_edges():
            named |= {str(s) for s in edge.data.free_symbols}
        for region in sdfg.all_control_flow_regions():
            named |= region.used_symbols(all_symbols=True, with_contents=False)
        forwarded = 0
        changed = True
        while changed:
            changed = False
            accesses: dict[str, list[tuple[SDFGState, nodes.AccessNode]]] = defaultdict(list)
            for st in sdfg.states():
                for an in st.data_nodes():
                    accesses[an.data].append((st, an))
            for st in sdfg.states():
                for edge in st.edges():
                    offset = self.forwardable(sdfg, st, edge, accesses, named)
                    if offset is not None:
                        self.forward(sdfg, st, edge, offset, accesses[edge.dst.data])
                        forwarded += 1
                        changed = True
                        break
                if changed:
                    break
        return forwarded

    @staticmethod
    def forwardable(sdfg: SDFG, state: SDFGState, edge, accesses, named: set[str]) -> list | None:
        """The per-dimension offset ``source - copy`` when ``edge`` is a copy the contract admits, else ``None``."""
        src, dst = edge.src, edge.dst
        if not isinstance(src, nodes.AccessNode) or not isinstance(dst, nodes.AccessNode) or src.data == dst.data:
            return None
        if edge.data.is_empty() or edge.data.wcr is not None:
            return None
        a, b = sdfg.arrays[src.data], sdfg.arrays[dst.data]
        if not isinstance(b, data.Array) or isinstance(b, data.View) or isinstance(a, data.View) or not b.transient:
            return None
        if len(a.shape) != len(b.shape) or dst.data in named:
            return None
        src_subset, dst_subset = edge.data.get_src_subset(edge, state), edge.data.get_dst_subset(edge, state)
        if not isinstance(src_subset, subsets.Range) or not isinstance(dst_subset, subsets.Range):
            return None
        if dst_subset != subsets.Range.from_array(b) or any(s != 1 for _, _, s in src_subset.ranges):
            return None
        if any(x != y for x, y in zip(src_subset.size(), dst_subset.size())):
            return None
        uses = accesses[dst.data]
        if any(n is not dst and st.in_degree(n) > 0 for st, n in uses) or state.in_degree(dst) != 1:
            return None
        region: ControlFlowRegion = state.parent_graph
        readers = [st for st, n in uses if n is not dst]
        if any(st.parent_graph is not region for st in readers):
            return None
        # Every leaf of a reader's memlet trees is code or another access node, never a nested SDFG
        for st, n in uses:
            for out in st.out_edges(n):
                if any(isinstance(t.dst, nodes.NestedSDFG) for t in st.memlet_tree(out).leaves()):
                    return None
        if _writes_besides(state, src.data, edge) or any(_writes(st, src.data) for st in set(readers) - {state}):
            return None
        after = _reach(region, state, region.successors)
        if state in after:  # a cycle through the copy
            return None
        for reader in set(readers) - {state}:
            if reader not in after:
                return None
            between = after & _reach(region, reader, region.predecessors)
            if any(_writes(block, src.data) for block in between):
                return None
        return [b0 - d0 for (b0, _, _), (d0, _, _) in zip(src_subset.ranges, dst_subset.ranges)]

    @staticmethod
    def forward(sdfg: SDFG, state: SDFGState, edge, offset: list, uses) -> None:
        """Retarget every read of the copy to the source at ``offset`` and remove the copy."""
        source, copy = edge.src.data, edge.dst.data

        def retarget(st: SDFGState, node: nodes.AccessNode) -> None:
            for out in st.out_edges(node):
                for tree_edge in st.memlet_tree(out):
                    memlet = tree_edge.data
                    if memlet.data == copy:
                        memlet.data = source
                        memlet.subset.offset(offset, False)
                    elif tree_edge is out and memlet.other_subset is not None:
                        memlet.other_subset.offset(offset, False)

        for st, node in uses:
            if node is edge.dst:
                continue
            retarget(st, node)
            node.data = source
        # In the copy's own state the readers hang off the source node itself
        retarget(state, edge.dst)
        for out in list(state.out_edges(edge.dst)):
            state.add_edge(edge.src, None, out.dst, out.dst_conn, out.data)
            state.remove_edge(out)
        state.remove_node(edge.dst)
        if state.degree(edge.src) == 0:
            state.remove_node(edge.src)
        if not any(n.data == copy for st in sdfg.states() for n in st.data_nodes()):
            sdfg.remove_data(copy, validate=False)


def _reach(region: ControlFlowRegion, start: ControlFlowBlock, step) -> set[ControlFlowBlock]:
    """The blocks of ``region`` reachable from ``start`` along ``step`` (successors or predecessors), ``start``
    excluded unless a cycle returns to it."""
    seen: set[ControlFlowBlock] = set()
    stack = list(step(start))
    while stack:
        block = stack.pop()
        if block not in seen:
            seen.add(block)
            stack.extend(step(block))
    return seen


def _writes_besides(state: SDFGState, name: str, edge) -> bool:
    """Whether ``state`` writes ``name`` anywhere (the copy only reads it)."""
    return any(state.in_degree(n) > 0 for n in state.data_nodes() if n.data == name)
