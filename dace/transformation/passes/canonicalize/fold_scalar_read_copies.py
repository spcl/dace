# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fold the frontend's scalar copies into direct tasklet accesses: reads (``A[i] -> A_index -> tasklet``)
and results (``tasklet -> result -> assign tasklet -> A[i]``).

The Python frontend reads an array element into a transient Scalar before a tasklet uses it. The copy is a
SNAPSHOT: the tasklet sees ``A[i]`` as it was when the copy ran. Reading ``A[i]`` directly instead is the
same value exactly when nothing can write ``A`` between the copy and the tasklet. Copy and consumers sit in
one state, so a write in another state runs wholly before or wholly after both and cannot interfere; only a
write in the same state that is ordered neither before the copy nor after every consumer can. A result
copy folds by the mirror rule: no other access to the array may fall between the producing tasklet and the
write.
"""
from typing import Any, Dict, Iterator, List, Optional, Set

from dace import SDFG, Memlet, data, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation


def reachable(state: SDFGState, start: nodes.Node, forward: bool = True) -> Set[nodes.Node]:
    """Every node a path of edges leads to from ``start`` (or, backward, leads from), ``start`` excluded."""
    seen: Set[nodes.Node] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for edge in (state.out_edges(node) if forward else state.in_edges(node)):
            neighbour = edge.dst if forward else edge.src
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


def writers(state: SDFGState, name: str) -> List[nodes.Node]:
    """The nodes through which ``state`` writes container ``name``: the access nodes it flows into."""
    return [node for node in state.data_nodes() if node.data == name and state.in_degree(node) > 0]


def foldable(sdfg: SDFG, state: SDFGState, copy: Any, counts: Dict[str, int]) -> Optional[Memlet]:
    """The direct read of the source element if the copy edge ``copy`` may be folded, else ``None``."""
    scalar = copy.dst
    desc = sdfg.arrays[scalar.data]
    if not (isinstance(desc, data.Scalar) and desc.transient and counts.get(scalar.data) == 1):
        return None
    if state.in_degree(scalar) != 1 or copy.data.wcr is not None or copy.data.is_empty():
        return None
    consumers = state.out_edges(scalar)
    if not consumers or not all(isinstance(edge.dst, nodes.Tasklet) for edge in consumers):
        return None
    root = state.memlet_path(copy)[0].src
    if not isinstance(root, nodes.AccessNode) or root.data == scalar.data:
        return None
    source = sdfg.arrays[root.data]
    if isinstance(source, (data.Scalar, data.Stream)) or source.dtype != desc.dtype:
        return None
    subset = copy.data.subset if copy.data.data == root.data else copy.data.other_subset
    if subset is None or symbolic.equal(subset.num_elements(), 1) is not True:
        return None
    before = reachable(state, copy.src, forward=False)
    after = set.intersection(*(reachable(state, edge.dst) for edge in consumers))
    if any(writer not in before and writer not in after for writer in writers(state, root.data)):
        return None
    return Memlet(data=root.data, subset=subset)


def fold(state: SDFGState, copy: Any, read: Memlet) -> None:
    """Connect every consumer of the copy's scalar to the copy's source, then drop the scalar."""
    scalar = copy.dst
    for edge in list(state.out_edges(scalar)):
        state.add_edge(copy.src, copy.src_conn, edge.dst, edge.dst_conn, Memlet.from_memlet(read))
        state.remove_edge(edge)
    state.remove_edge(copy)
    state.remove_node(scalar)


def assign_target(state: SDFGState, scalar: nodes.AccessNode) -> Optional[Any]:
    """The out-edge of the assign tasklet ``y = x`` that is ``scalar``'s only reader, if it is one."""
    readers = state.out_edges(scalar)
    if len(readers) != 1 or not isinstance(readers[0].dst, nodes.Tasklet):
        return None
    tasklet = readers[0].dst
    if len(tasklet.in_connectors) != 1 or len(tasklet.out_connectors) != 1 or state.out_degree(tasklet) != 1:
        return None
    out_conn, in_conn = next(iter(tasklet.out_connectors)), next(iter(tasklet.in_connectors))
    if tasklet.code.as_string.strip() != f'{out_conn} = {in_conn}':
        return None
    target = state.out_edges(tasklet)[0]
    return target if isinstance(target.dst, nodes.AccessNode) and target.data.wcr is None else None


def foldable_write(sdfg: SDFG, state: SDFGState, produce: Any, counts: Dict[str, int]) -> Optional[Any]:
    """The assign tasklet's edge into the array if the result scalar ``produce`` fills may be folded away."""
    scalar = produce.dst
    desc = sdfg.arrays[scalar.data]
    if not (isinstance(desc, data.Scalar) and desc.transient and counts.get(scalar.data) == 1):
        return None
    if not isinstance(produce.src, nodes.Tasklet) or state.in_degree(scalar) != 1 or produce.data.wcr is not None:
        return None
    target = assign_target(state, scalar)
    if target is None or sdfg.arrays[target.dst.data].dtype != desc.dtype:
        return None
    if symbolic.equal(target.data.subset.num_elements(), 1) is not True:
        return None
    before = reachable(state, produce.src, forward=False)
    after = reachable(state, target.dst)
    accessors = [node for node in state.data_nodes() if node.data == target.dst.data and node is not target.dst]
    if any(node not in before and node not in after for node in accessors):
        return None
    return target


def fold_write(state: SDFGState, produce: Any, target: Any) -> None:
    """Let the producing tasklet write the array element itself; drop the scalar and the assign tasklet."""
    assign = target.src
    state.add_edge(produce.src, produce.src_conn, target.dst, target.dst_conn,
                   Memlet(data=target.data.data, subset=target.data.subset))
    state.remove_node(assign)
    state.remove_node(produce.dst)


def copy_edges(sdfg: SDFG, state: SDFGState) -> Iterator[Any]:
    """The edges that fill a transient Scalar access node in ``state``."""
    for edge in state.edges():
        if isinstance(edge.dst, nodes.AccessNode) and isinstance(sdfg.arrays.get(edge.dst.data), data.Scalar):
            yield edge


@transformation.explicit_cf_compatible
class FoldScalarReadCopies(ppl.Pass):
    """Fold every scalar read and result copy the snapshot rule allows, in ``sdfg`` and its nested SDFGs."""
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Nodes | ppl.Modifies.Edges))

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """:returns: the number of folded copies, or ``None`` if none."""
        folded = 0
        for owner in sdfg.all_sdfgs_recursive():
            counts: Dict[str, int] = {}
            for state in owner.states():
                for node in state.data_nodes():
                    counts[node.data] = counts.get(node.data, 0) + 1
            for state in owner.states():
                for copy in list(copy_edges(owner, state)):
                    if copy.dst not in state.nodes():
                        continue
                    read = foldable(owner, state, copy, counts)
                    if read is not None:
                        fold(state, copy, read)
                        counts[copy.dst.data] = 0
                        folded += 1
                        continue
                    target = foldable_write(owner, state, copy, counts)
                    if target is not None:
                        fold_write(state, copy, target)
                        counts[copy.dst.data] = 0
                        folded += 1
        return folded or None
