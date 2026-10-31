# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fewer, longer-lived transients: coalesce copies, share non-interfering buffers, allocate once."""

from dataclasses import dataclass, field

from dace import SDFG, SDFGState, data, dtypes, properties, subsets, symbolic
from dace.config import Config
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.transient_reuse import BufferSharing, inside_map
from dace.transformation.transformation import explicit_cf_compatible

#: Storages a coalesced pair may mix: all of them are plain host memory the other side's code can address.
HOST_STORAGES = (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap, dtypes.StorageType.Register)


@dataclass(slots=True)
class Placement:
    """Where the elements of a transient live inside the container it is coalesced into: ``target[subset]``, the
    transient's non-unit dimensions laid on the ``dims`` of ``target``, in order."""

    target: str
    subset: subsets.Range
    dims: list[int]
    unit: list[int]

    def place(self, inner: subsets.Range) -> subsets.Range:
        """``inner``, a subset of the transient, as a subset of ``target``."""
        kept = [inner.ranges[d] for d in range(inner.dims()) if d not in self.unit]
        ranges = []
        for d, (begin, end, step) in enumerate(self.subset.ranges):
            if d in self.dims:
                b, e, s = kept[self.dims.index(d)]
                ranges.append((begin + b, begin + e, s))
            else:
                ranges.append((begin, end, 1))
        return subsets.Range(ranges)


@dataclass(slots=True)
class Coalescing:
    """One SDFG's copy coalescing: a transient whose whole life is one state and that is copied whole to or from
    a region of another container is renamed onto that region, and the copy goes. Copy related, the two hold the
    same values wherever both are live; the rename is exact when no other access of the container falls between
    the transient's accesses and the copy."""

    sdfg: SDFG
    #: Per container, the states holding an access node of it and how many.
    occurrences: dict[str, dict[SDFGState, int]] = field(default_factory=dict)
    #: Containers interstate edges or control-flow code read.
    control: set[str] = field(default_factory=set)
    facts: symbolic.Facts | None = None

    def run(self) -> int:
        self.facts = self.sdfg.facts()
        for state in self.sdfg.states():
            for node in state.data_nodes():
                per_state = self.occurrences.setdefault(node.data, {})
                per_state[state] = per_state.get(state, 0) + 1
        for edge in self.sdfg.all_interstate_edges():
            self.control |= edge.data.free_symbols
        for region in self.sdfg.all_control_flow_regions():
            self.control |= region.used_symbols(all_symbols=True, with_contents=False)
        coalesced = 0
        for state in self.sdfg.states():
            if any(isinstance(n.desc(self.sdfg), (data.View, data.Reference)) for n in state.data_nodes()):
                continue
            changed = True
            while changed:
                changed = False
                for edge in state.edges():
                    if self.coalesce(state, edge):
                        coalesced += 1
                        changed = True
                        break
        return coalesced

    def coalesce(self, state: SDFGState, edge) -> bool:
        src, dst = edge.src, edge.dst
        if not (isinstance(src, nodes.AccessNode) and isinstance(dst, nodes.AccessNode)) or src.data == dst.data:
            return False
        if edge.data.is_empty() or edge.data.wcr is not None:
            return False
        for transient, other, into in ((dst, src, False), (src, dst, True)):
            placement = self.placement(state, edge, transient, other)
            if placement is None or not self.isolated(state, transient, other, into):
                continue
            self.rename(state, edge, transient, other, placement)
            return True
        return False

    def placement(self, state: SDFGState, edge, transient: nodes.AccessNode, other: nodes.AccessNode) -> Placement:
        """Where ``transient`` lands in ``other`` if the pair may be coalesced at all, else ``None``."""
        sd = self.sdfg
        desc, target = sd.arrays[transient.data], sd.arrays[other.data]
        if not desc.transient or type(desc) not in (data.Array, data.Scalar) or desc.may_alias:
            return None
        if desc.lifetime in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.External):
            return None
        if type(target) not in (data.Array, data.Scalar) or desc.dtype != target.dtype:
            return None
        if desc.storage != target.storage and not (desc.storage in HOST_STORAGES and target.storage in HOST_STORAGES):
            return None
        if transient.data in self.control or list(self.occurrences.get(transient.data, {})) != [state]:
            return None
        inward = transient is edge.dst
        mine = edge.data.get_dst_subset(edge, state) if inward else edge.data.get_src_subset(edge, state)
        theirs = edge.data.get_src_subset(edge, state) if inward else edge.data.get_dst_subset(edge, state)
        whole = subsets.Range.from_array(desc)
        if not isinstance(theirs, subsets.Range) or not (mine or whole).covers_precise(whole, self.facts):
            return None
        unit = [d for d, extent in enumerate(desc.shape) if symbolic.equal(extent, 1) is True]
        kept = [d for d in range(len(desc.shape)) if d not in unit]
        sizes = theirs.size()
        dims = [d for d, size in enumerate(sizes) if symbolic.equal(size, 1) is not True]
        if len(dims) != len(kept) or any(theirs.ranges[d][2] != 1 for d in dims):
            return None
        if any(symbolic.equal(desc.shape[t], sizes[d]) is not True for t, d in zip(kept, dims)):
            return None
        # Library nodes address the buffer through its strides; element memlets do not.
        laid_out = all(symbolic.equal(desc.strides[t], target.strides[d]) is True for t, d in zip(kept, dims))
        for node in state.data_nodes():
            if node.data != transient.data:
                continue
            for e in state.all_edges(node):
                for tree in state.memlet_tree(e).traverse_children(include_self=True):
                    for end in (tree.edge.src, tree.edge.dst):
                        if isinstance(end, nodes.NestedSDFG) or (isinstance(end, nodes.LibraryNode) and not laid_out):
                            return None
        return Placement(other.data, theirs, dims, unit)

    def isolated(self, state: SDFGState, transient: nodes.AccessNode, other: nodes.AccessNode, into: bool) -> bool:
        """Whether the transient's values are exactly the copied region's while it lives: copied out (``into``),
        every access of it precedes the copy; copied in, the copy is its only write. Every other access of the
        container precedes, or follows, all of the transient's accesses and the copy."""
        mine = [n for n in state.data_nodes() if n.data == transient.data]
        # An accumulation starts from the transient's own value, which the container does not hold.
        if any(e.data.wcr is not None for n in mine for e in state.in_edges(n)):
            return False
        scope = state.scope_dict()
        if scope[transient] is not scope[other]:
            return False
        before = state_upstream(state, transient)
        if into:
            if any(n is not transient and n not in before for n in mine):
                return False
            # Copying out a transient this state never writes copies what an earlier run or the allocation left.
            if not any(not e.data.is_empty() for n in mine for e in state.in_edges(n)):
                return False
            # The copy is the container's only write here: another one is ordered against the copy, not the producers.
            if any(n is not other and n.data == other.data and state.in_degree(n) > 0 for n in state.data_nodes()):
                return False
        else:
            if any(not e.data.is_empty() for n in mine for e in state.in_edges(n) if e.src is not other):
                return False
        # The transient's edges move onto the container's node: none may join the two or close a cycle through it.
        if len(state.edges_between(transient, other)) + len(state.edges_between(other, transient)) != 1:
            return False
        after_other = state_downstream(state, other)
        if any(e.src is not other and e.src in after_other for e in state.in_edges(transient)):
            return False
        if any(e.dst is not other and other in state_downstream(state, e.dst) for e in state.out_edges(transient)):
            return False
        group = [*mine, other]
        upstream = [state_upstream(state, n) for n in group]
        downstream = [state_downstream(state, n) for n in group]
        for node in state.data_nodes():
            if node.data != other.data or node is other:
                continue
            if not (all(node in up for up in upstream) or all(node in down for down in downstream)):
                return False
        return True

    def rename(self, state: SDFGState, edge, transient: nodes.AccessNode, other: nodes.AccessNode, at: Placement):
        old = transient.data
        mine = [n for n in state.data_nodes() if n.data == old]
        done = set()
        for node in mine:
            for e in state.all_edges(node):
                if e is edge:
                    continue
                for tree_edge in state.memlet_tree(e).traverse_children(include_self=True):
                    tree_edge = tree_edge.edge
                    if tree_edge in done:
                        continue
                    done.add(tree_edge)
                    memlet = tree_edge.data
                    if memlet.is_empty():
                        continue
                    if memlet.data == old:
                        copy_to = tree_edge.dst if tree_edge.src is node else tree_edge.src
                        if memlet.other_subset is None and isinstance(copy_to, nodes.AccessNode):
                            memlet.other_subset = (
                                memlet.get_dst_subset(tree_edge, state)
                                if tree_edge.src is node
                                else memlet.get_src_subset(tree_edge, state)
                            )
                        elif isinstance(copy_to, nodes.AccessNode) and copy_to.data == old:
                            memlet.other_subset = at.place(memlet.other_subset)
                        memlet.data = at.target
                        memlet.subset = at.place(memlet.subset)
                    elif tree_edge.dst is node or tree_edge.src is node:
                        # The other end names the memlet: the transient's side is the other subset.
                        side = (
                            tree_edge.data.get_dst_subset(tree_edge, state)
                            if tree_edge.dst is node
                            else (tree_edge.data.get_src_subset(tree_edge, state))
                        )
                        memlet.other_subset = at.place(side or subsets.Range.from_array(self.sdfg.arrays[old]))
        state.remove_edge(edge)
        for node in mine:
            node.data = at.target
        # The copy's two ends are one access node now.
        for e in list(state.in_edges(transient)):
            state.add_edge(e.src, e.src_conn, other, e.dst_conn, e.data)
            state.remove_edge(e)
        for e in list(state.out_edges(transient)):
            state.add_edge(other, e.src_conn, e.dst, e.dst_conn, e.data)
            state.remove_edge(e)
        state.remove_node(transient)
        self.sdfg.remove_data(old, validate=False)
        self.occurrences.pop(old, None)
        per_state = self.occurrences.setdefault(at.target, {})
        per_state[state] = sum(1 for n in state.data_nodes() if n.data == at.target)


def state_upstream(state: SDFGState, node: nodes.Node) -> set:
    """Every node with a path to ``node`` in ``state``."""
    return walk(state, node, forward=False)


def state_downstream(state: SDFGState, node: nodes.Node) -> set:
    """Every node a path from ``node`` reaches in ``state``."""
    return walk(state, node, forward=True)


def walk(state: SDFGState, start: nodes.Node, forward: bool) -> set:
    seen, stack = set(), [start]
    while stack:
        node = stack.pop()
        for e in state.out_edges(node) if forward else state.in_edges(node):
            nxt = e.dst if forward else e.src
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


@properties.make_properties
@explicit_cf_compatible
class MinimizeTransients(ppl.Pass):
    """Fewer, smaller and longer-lived transients.

    1. Coalesce, in every SDFG: a transient living in one state and copied whole to or from a region of another
       container is that region (:class:`Coalescing`).
    2. Share and lift, in SDFGs outside every map scope: transients whose uses never overlap share one buffer, and
       every buffer sized by symbols bound on entry is ``Persistent``
       (:class:`~dace.transformation.passes.transient_reuse.BufferSharing`).

    Contracting a scope-local transient to its footprint is :class:`ShrinkMapLocalTransients`, which storage
    finalization runs earlier; what remains is left to the allocator.
    """

    CATEGORY: str = "Memory Footprint Reduction"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Nodes | ppl.Modifies.Memlets))

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> int | None:
        """:returns: How many transients were coalesced or now share a buffer, or None if none."""
        storages = [dtypes.StorageType.CPU_Heap]
        if Config.get("compiler", "cuda", "max_concurrent_streams") == -1:
            storages.append(dtypes.StorageType.GPU_Global)
        changed = 0
        for sd in sdfg.all_sdfgs_recursive():
            changed += Coalescing(sd).run()
            # A buffer inside a map scope is per iteration: sharing it would share it between iterations.
            if not inside_map(sd):
                changed += len(BufferSharing(sd, tuple(storages)).run())
        return changed or None
