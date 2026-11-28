# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A transient that a map fills element by element from another array (``t[0, i] = pos[i, 2]``), and that the same
state then reads one element at a time, is read from that array at the mapped index."""

import ast
from dataclasses import dataclass
from typing import Any

from dace import SDFG, Memlet, data, dtypes, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.dataflow.redundant_array import same_extent
from dace.transformation.passes.canonicalize.fold_scalar_read_copies import control_flow_reads


@dataclass
class CopyMap:
    """``read`` and ``write``: the identity tasklet's edges; ``source``: the access node the map reads; ``index``: per
    source dimension, the map parameter or the fixed index; ``dims``: each parameter's dimension of the transient."""

    read: MultiConnectorEdge[Memlet]
    write: MultiConnectorEdge[Memlet]
    source: nodes.AccessNode
    index: list[Any]
    dims: dict[str, int]


def identity(tasklet: nodes.Tasklet) -> bool:
    """Whether ``tasklet`` is ``out = in``."""
    if tasklet.language != dtypes.Language.Python or len(tasklet.in_connectors) != 1:
        return False
    if len(tasklet.out_connectors) != 1 or len(tasklet.code.code) != 1:
        return False
    stmt = tasklet.code.code[0]
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return False
    target, value = stmt.targets[0], stmt.value
    return (
        isinstance(target, ast.Name)
        and isinstance(value, ast.Name)
        and target.id in tasklet.out_connectors
        and value.id in tasklet.in_connectors
    )


def transient_dims(entry: nodes.MapEntry, written: subsets.Range, desc: data.Data) -> dict[str, int] | None:
    """Each map parameter's dimension of the transient, when the map writes every element of it once."""
    params, dims = entry.map.params, {}
    for dim, (begin, end, step) in enumerate(written.ranges):
        if begin != end:
            return None
        if str(begin) not in params:
            if symbolic.equal(desc.shape[dim], 1) is not True or symbolic.equal(begin, 0) is not True:
                return None
            continue
        low, high, stride = entry.map.range[params.index(str(begin))]
        if symbolic.equal(low, 0) is not True or stride != 1 or symbolic.equal(high + 1, desc.shape[dim]) is not True:
            return None
        dims[str(begin)] = dim
    return dims if len(dims) == len(params) else None


def source_index(entry: nodes.MapEntry, read: subsets.Range) -> list[Any] | None:
    """Per source dimension, the map parameter it is, or an index free of the parameters."""
    params, index = entry.map.params, []
    for begin, end, step in read.ranges:
        if begin != end:
            return None
        if str(begin) in params:
            index.append(str(begin))
        elif {str(s) for s in symbolic.pystr_to_symbolic(begin).free_symbols} & set(params):
            return None
        else:
            index.append(begin)
    return index


def identity_write(state: SDFGState, node: nodes.AccessNode) -> MultiConnectorEdge[Memlet] | None:
    """The one ``out = in`` tasklet's edge into the map exit that writes ``node``, else ``None``."""
    incoming = state.in_edges(node)
    if len(incoming) != 1 or not isinstance(incoming[0].src, nodes.MapExit):
        return None
    leaves = list(state.memlet_tree(incoming[0]).leaves())
    if len(leaves) != 1 or not isinstance(leaves[0].src, nodes.Tasklet) or not identity(leaves[0].src):
        return None
    write = leaves[0]
    if write.data.data != node.data or state.in_degree(write.src) != 1 or state.out_degree(write.src) != 1:
        return None
    return write


def copied_from(
    sdfg: SDFG, state: SDFGState, read: MultiConnectorEdge[Memlet], node: nodes.AccessNode
) -> nodes.AccessNode | None:
    """The access node, not a view nor ``node``, that ``read`` takes its element from."""
    source = state.memlet_path(read)[0].src
    if not isinstance(source, nodes.AccessNode) or source.data != read.data.data or source.data == node.data:
        return None
    return None if isinstance(sdfg.arrays[source.data], data.View) else source


def copy_map(sdfg: SDFG, state: SDFGState, node: nodes.AccessNode) -> CopyMap | None:
    """The top-level map writing all of ``node`` through one identity tasklet, else ``None``."""
    write = identity_write(state, node)
    entry = None if write is None else state.entry_node(write.src)
    if entry is None or state.entry_node(entry) is not None or write.dst is not state.exit_node(entry):
        return None
    read = state.in_edges(write.src)[0]
    if read.src is not entry or any(
        m.wcr is not None or m.dynamic or m.other_subset is not None for m in (read.data, write.data)
    ):
        return None
    source = copied_from(sdfg, state, read, node)
    if source is None:
        return None
    dims = transient_dims(entry, write.data.subset, sdfg.arrays[node.data])
    index = source_index(entry, read.data.subset)
    return None if dims is None or index is None else CopyMap(read, write, source, index, dims)


def element_readers(state: SDFGState, node: nodes.AccessNode, source: str) -> list[MultiConnectorEdge] | None:
    """The edges out of ``node``, when each roots a tree of one-element tasklet reads whose scope does not write
    ``source``."""
    roots = state.out_edges(node)
    for root in roots:
        if root.data.is_empty() or root.data.data != node.data or root.data.other_subset is not None:
            return None
        for edge in state.memlet_tree(root):
            if not isinstance(edge.dst, (nodes.MapEntry, nodes.Tasklet)) or not isinstance(
                edge.data.subset, subsets.Range
            ):
                return None
        if any(symbolic.equal(e.data.subset.num_elements(), 1) is not True for e in state.memlet_tree(root).leaves()):
            return None
        last = state.exit_node(root.dst) if isinstance(root.dst, nodes.MapEntry) else root.dst
        if any(e.data.data == source for e in state.out_edges(last)):
            return None
    return roots


def reach(state: SDFGState, start: nodes.Node, forward: bool) -> set[nodes.Node]:
    """The nodes after (``forward``) or before ``start`` in ``state``, ``start`` included."""
    seen, stack = {start}, [start]
    while stack:
        current = stack.pop()
        nexts = state.successors(current) if forward else state.predecessors(current)
        for n in nexts:
            if n not in seen:
                seen.add(n)
                stack.append(n)
    return seen


def ordered(state: SDFGState, found: CopyMap, roots: list[MultiConnectorEdge]) -> bool:
    """Whether every write of the source in ``state`` lands before the copy reads it or after every reader."""
    before = reach(state, found.source, forward=False)
    afters = [
        reach(state, state.exit_node(r.dst) if isinstance(r.dst, nodes.MapEntry) else r.dst, forward=True)
        for r in roots
    ]
    for n in state.data_nodes():
        if n.data != found.source.data or state.in_degree(n) == 0 or n in before:
            continue
        if not all(n in after for after in afters):
            return False
    return True


def state_local(sdfg: SDFG, name: str) -> bool:
    """Whether every access node of ``name`` is the only one of its state and is written whole there, so no value of
    it flows from one state to another."""
    whole = subsets.Range.from_array(sdfg.arrays[name])
    for state in sdfg.states():
        held = [n for n in state.data_nodes() if n.data == name]
        if not held:
            continue
        if len(held) != 1:
            return False
        if not any(
            not e.data.is_empty() and not e.data.dynamic and same_extent(e.data.get_dst_subset(e, state), whole)
            for e in state.in_edges(held[0])
        ):
            return False
    return True


def mapped(found: CopyMap, inner: subsets.Range) -> subsets.Range:
    """``inner``, a range of the transient, restated in the source's index space."""
    ranges = []
    for index in found.index:
        if isinstance(index, str):
            ranges.append(inner.ranges[found.dims[index]])
        else:
            ranges.append((index, index, 1))
    return subsets.Range(ranges)


def bridge(state: SDFGState, before: list[nodes.Node], after: list[nodes.Node]) -> None:
    """Order every node of ``before`` ahead of every node of ``after``; drop those left without edges."""
    for src in before:
        for dst in after:
            state.add_nedge(src, dst, Memlet())
    for n in before + after:
        if n in state.nodes() and state.degree(n) == 0:
            state.remove_node(n)


def drop_empty_map(state: SDFGState, entry: nodes.MapEntry) -> None:
    """Remove ``entry``'s map when nothing is left inside it, keeping the order its ordering edges imposed."""
    # Removing the tasklet's paths drops an isolated map with them
    if entry not in state.nodes():
        return
    exit_ = next((n for n in state.nodes() if isinstance(n, nodes.MapExit) and n.map is entry.map), None)
    if exit_ is not None and state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes():
        return
    before = [e.src for e in state.in_edges(entry)]
    after = [] if exit_ is None else [e.dst for e in state.out_edges(exit_)]
    state.remove_nodes_from([n for n in (entry, exit_) if n is not None])
    bridge(state, before, after)


def forward(state: SDFGState, node: nodes.AccessNode, found: CopyMap, roots: list[MultiConnectorEdge]) -> None:
    """Read the source where the readers read ``node``, then drop the tasklet that filled it."""
    for root in roots:
        for edge in state.memlet_tree(root):
            edge.data.subset = mapped(found, edge.data.subset)
            edge.data.data = found.source.data
        state.add_edge(found.source, None, root.dst, root.dst_conn, root.data)
        state.remove_edge(root)
    entry, tasklet = found.read.src, found.read.dst
    sdutil.remove_edge_and_dangling_path(state, found.write)
    sdutil.remove_edge_and_dangling_path(state, found.read)
    for n in (tasklet, node):
        if n in state.nodes() and state.degree(n) == 0:
            state.remove_node(n)
    drop_empty_map(state, entry)


def forward_in(sdfg: SDFG, state: SDFGState, name: str) -> bool:
    """Forward the copy map of ``name`` in ``state`` when its readers allow it."""
    node = next((n for n in state.data_nodes() if n.data == name), None)
    found = None if node is None else copy_map(sdfg, state, node)
    roots = None if found is None else element_readers(state, node, found.source.data)
    if roots is None or not ordered(state, found, roots):
        return False
    forward(state, node, found, roots)
    return True


@properties.make_properties
@transformation.explicit_cf_compatible
class ForwardCopyMaps(ppl.Pass):
    """A transient a top-level map fills whole through ``out = in`` tasklets, ``T[.., i, ..] = S[.., i or c, ..]``, is
    read from ``S`` by the one-element readers of the same state. Each access node of ``T`` must be written whole in
    its state, and every write of ``S`` there must land before the copy or after every reader."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.AccessNodes)

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        forwarded = 0
        for sd in sdfg.all_sdfgs_recursive():
            forwarded += self.apply_to_sdfg(sd)
        return forwarded or None

    @staticmethod
    def apply_to_sdfg(sdfg: SDFG) -> int:
        named = control_flow_reads(sdfg)
        candidates = [
            name
            for name, desc in sdfg.arrays.items()
            if desc.transient and type(desc) is data.Array and name not in named and state_local(sdfg, name)
        ]
        forwarded = 0
        for name in candidates:
            forwarded += sum(forward_in(sdfg, state, name) for state in sdfg.states())
            if not any(n.data == name for st in sdfg.states() for n in st.data_nodes()):
                sdfg.remove_data(name, validate=False)
        return forwarded
