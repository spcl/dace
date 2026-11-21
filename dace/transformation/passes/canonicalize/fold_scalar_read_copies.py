# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fold the frontend's scalar copies into direct tasklet accesses: reads (``A[i] -> A_index -> tasklet``) and results
(``tasklet -> result -> A[i]``, the copy ``TrivialTaskletElimination`` leaves of an assign tasklet)."""

from collections.abc import Iterator
from typing import Any

from dace import SDFG, Memlet, data, dtypes, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def view_keeps_a_data_binding(state: SDFGState, view: nodes.AccessNode, dropped: MultiConnectorEdge[Memlet]) -> bool:
    """Whether ``view`` binds data through some edge other than ``dropped`` (else the fold leaves it dangling)."""
    for e in state.in_edges(view):
        if e is not dropped and not e.data.is_empty() and isinstance(state.memlet_path(e)[0].src, nodes.AccessNode):
            return True
    for e in state.out_edges(view):
        if e is not dropped and not e.data.is_empty() and isinstance(state.memlet_path(e)[-1].dst, nodes.AccessNode):
            return True
    return False


def reachable(state: SDFGState, start: nodes.Node, forward: bool = True) -> set[nodes.Node]:
    """Every node a path of edges leads to from ``start`` (or, backward, leads from), ``start`` excluded."""
    seen: set[nodes.Node] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for edge in state.out_edges(node) if forward else state.in_edges(node):
            neighbour = edge.dst if forward else edge.src
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


def writers(state: SDFGState, name: str) -> list[nodes.Node]:
    """The nodes through which ``state`` writes container ``name``: the access nodes it flows into."""
    return [node for node in state.data_nodes() if node.data == name and state.in_degree(node) > 0]


def consumer_end(state: SDFGState, consumer: nodes.Node) -> nodes.Node:
    """The node after which ``consumer`` has read the scalar: a tasklet itself, a scope its exit."""
    return state.exit_node(consumer) if isinstance(consumer, nodes.EntryNode) else consumer


def read_by_tasklets(state: SDFGState, edge: Any) -> bool:
    """Whether the element ``edge`` hands on reaches tasklets only: a nested SDFG's connector descriptor must match
    the whole container it reads, so a scalar it reads cannot become the source's element."""
    return all(isinstance(leaf.dst, nodes.Tasklet) for leaf in state.memlet_tree(edge).leaves())


def reads_in_place(sdfg: SDFG, state: SDFGState, consumer: nodes.Node, storage: dtypes.StorageType) -> bool:
    """Whether ``consumer`` runs where a container of ``storage`` is addressable: a copy into a scalar is also the
    only transfer between host and device memory, so a kernel reads a host element (and host code a device element)
    only through it."""
    on_device = is_devicelevel_gpu(sdfg, state, consumer) or (
        isinstance(consumer, nodes.EntryNode) and consumer.schedule in dtypes.GPU_SCHEDULES
    )
    if storage in (dtypes.StorageType.Register, dtypes.StorageType.CPU_Pinned):
        return True
    return (storage in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES) == on_device


def foldable(sdfg: SDFG, state: SDFGState, copy: Any, counts: dict[str, int], scopes: bool = False) -> Memlet | None:
    """The direct read of the source element if the copy edge ``copy`` may be folded, else ``None``."""
    scalar = copy.dst
    desc = sdfg.arrays[scalar.data]
    if not (isinstance(desc, data.Scalar) and desc.transient and counts.get(scalar.data) == 1):
        return None
    if state.in_degree(scalar) != 1 or copy.data.wcr is not None or copy.data.is_empty():
        return None
    consumers = state.out_edges(scalar)
    kinds = (nodes.Tasklet, nodes.EntryNode) if scopes else nodes.Tasklet
    if not consumers or not all(isinstance(edge.dst, kinds) and read_by_tasklets(state, edge) for edge in consumers):
        return None
    # An empty edge out of the snapshot orders the read before a later overwrite of the source
    if any(edge.data.is_empty() for edge in consumers):
        return None
    root = state.memlet_path(copy)[0].src
    if not isinstance(root, nodes.AccessNode) or root.data == scalar.data:
        return None
    source = sdfg.arrays[root.data]
    if isinstance(source, (data.Scalar, data.Stream)) or source.dtype != desc.dtype:
        return None
    if not all(reads_in_place(sdfg, state, edge.dst, source.storage) for edge in consumers):
        return None
    # A view bound to the scalar alone would view nothing once the scalar is gone.
    if isinstance(source, data.View) and not view_keeps_a_data_binding(state, root, copy):
        return None
    subset = copy.data.subset if copy.data.data == root.data else copy.data.other_subset
    if subset is None or symbolic.equal(subset.num_elements(), 1) is not True:
        return None
    # The source node's own writes land before the copy reads it.
    before = reachable(state, copy.src, forward=False) | {copy.src}
    # A write inside a consuming scope is neither: its other iterations would read the overwritten element
    after = set.intersection(*(reachable(state, consumer_end(state, edge.dst)) for edge in consumers))
    viewed = sdutil.get_last_view_node(state, root).data if isinstance(source, data.View) else root.data
    if any(writer not in before and writer not in after for writer in writers(state, viewed)):
        return None
    # A scope writing the source overwrites the element its other iterations read
    ends = [consumer_end(state, edge.dst) for edge in consumers if isinstance(edge.dst, nodes.EntryNode)]
    if any(e.data.data in (root.data, viewed) for end in ends for e in state.out_edges(end)):
        return None
    return Memlet(data=root.data, subset=subset)


def fold(state: SDFGState, copy: Any, read: Memlet) -> None:
    """Connect every consumer of the copy's scalar to the copy's source, then drop the scalar."""
    scalar = copy.dst
    for edge in list(state.out_edges(scalar)):
        memlet = Memlet.from_memlet(read)
        if isinstance(edge.dst, nodes.EntryNode):
            memlet = edge.data
            for tree in state.memlet_tree(edge).traverse_children(include_self=True):
                tree.edge.data.data, tree.edge.data.subset = read.data, Memlet.from_memlet(read).subset
        state.remove_edge(edge)
        state.add_edge(copy.src, copy.src_conn, edge.dst, edge.dst_conn, memlet)
    state.remove_edge(copy)
    state.remove_node(scalar)


def foldable_write(sdfg: SDFG, state: SDFGState, produce: Any, counts: dict[str, int]) -> Any | None:
    """The copy edge out of the result scalar ``produce`` fills, if the scalar may be folded away."""
    scalar = produce.dst
    desc = sdfg.arrays[scalar.data]
    if not (isinstance(desc, data.Scalar) and desc.transient and counts.get(scalar.data) == 1):
        return None
    if not isinstance(produce.src, nodes.Tasklet) or state.in_degree(scalar) != 1 or produce.data.wcr is not None:
        return None
    if state.out_degree(scalar) != 1:
        return None
    target = state.out_edges(scalar)[0]
    if not isinstance(target.dst, nodes.AccessNode) or target.data.wcr is not None or target.data.is_empty():
        return None
    array = sdfg.arrays[target.dst.data]
    if isinstance(array, (data.Scalar, data.Stream)) or array.dtype != desc.dtype:
        return None
    if isinstance(array, data.View) and not view_keeps_a_data_binding(state, target.dst, target):
        return None
    subset = write_subset(target)
    if subset is None or symbolic.equal(subset.num_elements(), 1) is not True:
        return None
    before = reachable(state, produce.src, forward=False)
    after = reachable(state, target.dst)
    accessors = [node for node in state.data_nodes() if node.data == target.dst.data and node is not target.dst]
    if any(node not in before and node not in after for node in accessors):
        return None
    return target


def write_subset(copy: Any) -> Any | None:
    """The subset of the destination array a scalar-to-array copy edge writes."""
    if copy.data.data == copy.dst.data:
        return copy.data.subset
    return copy.data.other_subset


def fold_write(state: SDFGState, produce: Any, target: Any) -> None:
    """Let the producing tasklet write the array element itself; drop the result scalar."""
    state.add_edge(
        produce.src,
        produce.src_conn,
        target.dst,
        target.dst_conn,
        Memlet(data=target.dst.data, subset=write_subset(target)),
    )
    state.remove_node(produce.dst)


def control_flow_reads(sdfg: SDFG) -> set[str]:
    """The names the interstate edges and the control-flow regions' own conditions of ``sdfg`` read."""
    names: set[str] = set()
    for edge in sdfg.all_interstate_edges():
        names |= edge.data.free_symbols
    for region in sdfg.all_control_flow_regions():
        names |= region.used_symbols(all_symbols=True, with_contents=False)
    return names


def copy_edges(sdfg: SDFG, state: SDFGState) -> Iterator[Any]:
    """The edges that fill a transient Scalar access node in ``state``."""
    for edge in state.edges():
        if isinstance(edge.dst, nodes.AccessNode) and isinstance(sdfg.arrays.get(edge.dst.data), data.Scalar):
            yield edge


@properties.make_properties
@transformation.explicit_cf_compatible
class FoldScalarReadCopies(ppl.Pass):
    """Fold every scalar read and result copy the snapshot rule allows, in ``sdfg`` and its nested SDFGs."""

    CATEGORY: str = "Canonicalization"

    reads_only = properties.Property(dtype=bool, default=False, desc="Leave the result copies alone")
    scopes = properties.Property(
        dtype=bool, default=False, desc="Also fold a read copy into the scopes it feeds (a map broadcasting it)"
    )

    def __init__(self, reads_only: bool = False, scopes: bool = False):
        super().__init__()
        self.reads_only = reads_only
        self.scopes = scopes

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Nodes | ppl.Modifies.Edges))

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """:returns: the number of folded copies, or ``None`` if none."""
        folded = 0
        for owner in sdfg.all_sdfgs_recursive():
            counts: dict[str, int] = {}
            for state in owner.states():
                for node in state.data_nodes():
                    counts[node.data] = counts.get(node.data, 0) + 1
            # A scalar an interstate edge or a branch / loop condition reads has a use no access node shows
            for name in control_flow_reads(owner) & counts.keys():
                counts[name] += 1
            for state in owner.states():
                for copy in list(copy_edges(owner, state)):
                    if copy.dst not in state.nodes():
                        continue
                    read = foldable(owner, state, copy, counts, self.scopes)
                    if read is not None:
                        fold(state, copy, read)
                        counts[copy.dst.data] = 0
                        folded += 1
                        continue
                    if self.reads_only:
                        continue
                    target = foldable_write(owner, state, copy, counts)
                    if target is not None:
                        fold_write(state, copy, target)
                        counts[copy.dst.data] = 0
                        folded += 1
        return folded or None
