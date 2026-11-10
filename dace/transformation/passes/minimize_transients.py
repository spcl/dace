# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fewer, longer-lived transients: coalesce copies, share non-interfering buffers, allocate once."""

from dataclasses import dataclass, field
from typing import Any

from dace import SDFG, SDFGState, data, dtypes, properties, subsets, symbolic
from dace.config import Config
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.libraries.standard.nodes.fill import FillLibraryNode
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.transient_reuse import BufferSharing, StateOrder, sharing_storages
from dace.transformation.transformation import explicit_cf_compatible

#: Storages a coalesced pair may mix: all of them are plain host memory the other side's code can address.
HOST_STORAGES = (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap, dtypes.StorageType.Register)


@dataclass(slots=True)
class Contraction:
    """One SDFG's contraction: a transient of which only one box is ever read keeps that box alone. What is written
    outside it is never observed: a fill is cut to the box, and any other write must lie inside it. The box names
    only symbols that hold one value for a whole run of the SDFG, so it is the same box at every access."""

    sdfg: SDFG
    facts: symbolic.Facts | None = None

    def run(self) -> int:
        sd = self.sdfg
        skip: set[str] = set()
        varying: set[str] = set()
        for edge in sd.all_interstate_edges():
            skip |= edge.data.free_symbols
            varying |= edge.data.assignments.keys()
        for region in sd.all_control_flow_regions():
            skip |= region.used_symbols(all_symbols=True, with_contents=False)
            if isinstance(region, LoopRegion) and region.loop_variable:
                varying.add(region.loop_variable)
        uses: dict[str, list[tuple[SDFGState, nodes.AccessNode]]] = {}
        for state in sd.states():
            for node in state.data_nodes():
                desc = sd.arrays[node.data]
                if type(desc) is data.Array and desc.transient and not desc.may_alias and node.data not in skip:
                    uses.setdefault(node.data, []).append((state, node))
        contracted = 0
        for name, where in uses.items():
            box = self.read_box(name, where, varying)
            if box is None:
                continue
            # A copy: the box is the subset of one of the memlets moved below.
            box = subsets.Range(box.ranges)
            origin = [begin for begin, _, _ in box.ranges]
            for state, node in where:
                for edge in state.all_edges(node):
                    if edge.data.is_empty():
                        continue
                    if isinstance(edge.src, FillLibraryNode):
                        edge.data.subset = subsets.Range(box.ranges)
                    for tree in state.memlet_tree(edge).traverse_children(include_self=True):
                        memlet = tree.edge.data
                        if memlet.data == name:
                            memlet.subset.offset(origin, True)
                        elif tree.edge is edge:
                            memlet.other_subset.offset(origin, True)
            sd.arrays[name].set_shape(box.size())
            contracted += 1
        return contracted

    def read_box(
        self, name: str, where: list[tuple[SDFGState, nodes.AccessNode]], varying: set[str]
    ) -> subsets.Range | None:
        """The box of ``name`` that covers everything read of it, if the buffer can be cut to it; else ``None``."""
        sd = self.sdfg
        reads: list[subsets.Subset | None] = []
        for state, node in where:
            # Under a map the subsets name its parameters: no one box.
            if state.entry_node(node) is not None:
                return None
            for edge in state.all_edges(node):
                if edge.data.is_empty():
                    continue
                other = edge.dst if edge.src is node else edge.src
                if edge.data.wcr is not None or (
                    isinstance(other, nodes.AccessNode) and isinstance(other.desc(sd), (data.View, data.Reference))
                ):
                    return None
                # Whatever keeps a descriptor or a pointer of its own to the buffer sees its layout.
                for tree in state.memlet_tree(edge).traverse_children(include_self=True):
                    for end in (tree.edge.src, tree.edge.dst):
                        if isinstance(end, nodes.NestedSDFG):
                            return None
                        if isinstance(end, nodes.LibraryNode) and not isinstance(end, FillLibraryNode):
                            return None
                        if (
                            isinstance(end, nodes.Tasklet)
                            and symbolic.equal(tree.edge.data.num_elements(), 1) is not True
                        ):
                            return None
                if edge.src is node:
                    reads.append(edge.data.get_src_subset(edge, state))
        if not reads or not all(isinstance(read, subsets.Range) for read in reads):
            return None
        if self.facts is None:
            self.facts = sd.facts()
        box = next((r for r in reads if all(r.covers_precise(other, self.facts) for other in reads)), None)
        if box is None or any(step != 1 for _, _, step in box.ranges) or {str(s) for s in box.free_symbols} & varying:
            return None
        desc = sd.arrays[name]
        if all(symbolic.equal(size, extent) is True for size, extent in zip(box.size(), desc.shape)):
            return None
        for state, node in where:
            for edge in state.in_edges(node):
                if edge.data.is_empty():
                    continue
                written = edge.data.get_dst_subset(edge, state)
                if not isinstance(written, subsets.Range):
                    return None
                if isinstance(edge.src, FillLibraryNode) and written.covers_precise(box, self.facts):
                    continue
                if not box.covers_precise(written, self.facts):
                    return None
        return box


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
class Copy:
    """A copy between two access nodes: one edge, or the two edges of a copy node that has no other."""

    src: nodes.AccessNode
    dst: nodes.AccessNode
    #: The edge leaving ``src`` and the one entering ``dst``: the same edge unless a copy node sits between them.
    out: Any
    into: Any
    read: subsets.Subset | None
    written: subsets.Subset | None

    @staticmethod
    def at(state: SDFGState, edge) -> "Copy | None":
        src, dst = edge.src, edge.dst
        if not isinstance(src, nodes.AccessNode) or edge.data.is_empty() or edge.data.wcr is not None:
            return None
        if isinstance(dst, nodes.AccessNode):
            return Copy(
                src, dst, edge, edge, edge.data.get_src_subset(edge, state), edge.data.get_dst_subset(edge, state)
            )
        if not isinstance(dst, CopyLibraryNode) or state.in_degree(dst) != 1 or state.out_degree(dst) != 1:
            return None
        last = state.out_edges(dst)[0]
        if not isinstance(last.dst, nodes.AccessNode) or last.data.wcr is not None:
            return None
        if edge.data.data != src.data or last.data.data != last.dst.data:
            return None
        return Copy(src, last.dst, edge, last, edge.data.subset, last.data.subset)


@dataclass(slots=True)
class Coalescing:
    """One SDFG's copy coalescing: a transient whose whole life is one state and that is copied whole to or from
    a region of another container is renamed onto that region, and the copy goes. Copy related, the two hold the
    same values wherever both are live; the rename is exact when no access of the container that sees the
    difference runs while the transient lives: copied out, the region takes its values at the transient's first
    write instead of at the copy; copied in, it must hold them until the transient's last read."""

    sdfg: SDFG
    #: Per container, the states holding an access node of it and how many.
    occurrences: dict[str, dict[SDFGState, int]] = field(default_factory=dict)
    #: Containers interstate edges or control-flow code read.
    control: set[str] = field(default_factory=set)
    facts: symbolic.Facts | None = None
    #: In the state at hand: per view, the container it views, and the edges binding the views.
    viewed: dict[str, str] = field(default_factory=dict)
    bindings: set = field(default_factory=set)

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
            if not self.resolve_views(state):
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

    def resolve_views(self, state: SDFGState) -> bool:
        """Fill ``viewed`` and ``bindings`` for ``state``; False if a view's container is not known by name."""
        self.viewed, self.bindings = {}, set()
        for node in state.data_nodes():
            desc = node.desc(self.sdfg)
            if isinstance(desc, data.Reference):
                return False
            if not isinstance(desc, data.View):
                continue
            edge = sdutil.get_view_edge(state, node)
            if edge is None:
                return False
            path = state.memlet_path(edge)
            container = path[0].src if edge.dst is node else path[-1].dst
            if not isinstance(container, nodes.AccessNode):
                return False
            if self.viewed.setdefault(node.data, container.data) != container.data:
                return False
            self.bindings.update(path)
        for name in self.viewed:
            while self.viewed[name] in self.viewed:
                self.viewed[name] = self.viewed[self.viewed[name]]
        return True

    def coalesce(self, state: SDFGState, edge) -> bool:
        copy = Copy.at(state, edge)
        if copy is None or copy.src.data == copy.dst.data:
            return False
        for transient, other, into in ((copy.dst, copy.src, False), (copy.src, copy.dst, True)):
            placement = self.placement(state, copy, transient, other)
            if placement is None or not self.isolated(state, copy, transient, other, into, placement):
                continue
            self.rename(state, copy, transient, other, placement)
            return True
        return False

    def placement(
        self, state: SDFGState, copy: Copy, transient: nodes.AccessNode, other: nodes.AccessNode
    ) -> Placement:
        """Where ``transient`` lands in ``other`` if the pair may be coalesced at all, else ``None``."""
        sd = self.sdfg
        desc, target = sd.arrays[transient.data], sd.arrays[other.data]
        if not desc.transient or type(desc) not in (data.Array, data.Scalar) or desc.may_alias:
            return None
        if desc.lifetime in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.External):
            return None
        if type(target) not in (data.Array, data.Scalar, data.ArrayView) or desc.dtype != target.dtype:
            return None
        # Accesses through a view of the transient are not its own; it becomes a view only where it is copied from one
        # into its single access node, whose readers leave the view's binding the edge it was.
        if transient.data in self.viewed.values():
            return None
        if isinstance(target, data.View) and not (
            transient is copy.dst
            and self.occurrences[transient.data][state] == 1
            and [e for e in state.all_edges(other) if not e.data.is_empty() and e is not copy.out]
            == [e for e in state.in_edges(other) if e in self.bindings]
            and self.keeps_binding(state, transient)
        ):
            return None
        if desc.storage != target.storage and not (desc.storage in HOST_STORAGES and target.storage in HOST_STORAGES):
            return None
        if transient.data in self.control or list(self.occurrences.get(transient.data, {})) != [state]:
            return None
        mine, theirs = (copy.written, copy.read) if transient is copy.dst else (copy.read, copy.written)
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
        # A nested SDFG keeps its own descriptor of the buffer: right only if the container is laid out like it.
        twin = (
            laid_out
            and not unit
            and len(target.shape) == len(kept)
            and theirs.covers_precise(subsets.Range.from_array(target), self.facts)
            and all(symbolic.equal(x, y) is True for x, y in zip(desc.offset, target.offset))
            and symbolic.equal(desc.start_offset, target.start_offset) is True
        )
        for node in state.data_nodes():
            if node.data != transient.data:
                continue
            for e in state.all_edges(node):
                for tree in state.memlet_tree(e).traverse_children(include_self=True):
                    for end in (tree.edge.src, tree.edge.dst):
                        if isinstance(end, nodes.NestedSDFG) and not twin:
                            return None
                        # So do tasklets taking more than one element: a pointer they index linearly.
                        addressed = isinstance(end, nodes.LibraryNode) or (
                            isinstance(end, nodes.Tasklet)
                            and symbolic.equal(tree.edge.data.num_elements(), 1) is not True
                        )
                        if addressed and not laid_out:
                            return None
        return Placement(other.data, theirs, dims, unit)

    def keeps_binding(self, state: SDFGState, transient: nodes.AccessNode) -> bool:
        """Whether a view node with one binding in-edge still binds through it once it takes ``transient``'s readers
        (:func:`~dace.sdfg.utils.get_view_edge`)."""
        readers = [e for e in state.out_edges(transient) if not e.data.is_empty()]
        return len(readers) != 1 or not isinstance(state.memlet_path(readers[0])[-1].dst, nodes.AccessNode)

    def isolated(
        self,
        state: SDFGState,
        copy: Copy,
        transient: nodes.AccessNode,
        other: nodes.AccessNode,
        into: bool,
        at: Placement,
    ) -> bool:
        """Whether the transient's values are exactly the copied region's while it lives: copied out (``into``),
        every access of it precedes the copy; copied in, the copy is its only write. No access of the container
        that overlaps the region runs while the two differ, see :class:`Coalescing`."""
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
            # The copy is the container's only write here: another one is ordered against the copy, not the producers.
            if any(n is not other and n.data == other.data and state.in_degree(n) > 0 for n in state.data_nodes()):
                return False
        elif any(not e.data.is_empty() for n in mine for e in state.in_edges(n) if e is not copy.into):
            return False
        # The transient's edges move onto the container's node: none may join the two or close a cycle through it.
        direct = len(state.edges_between(transient, other)) + len(state.edges_between(other, transient))
        if direct != int(copy.out is copy.into):
            return False
        after_other = state_downstream(state, other)
        if any(e is not copy.into and e.src in after_other for e in state.in_edges(transient)):
            return False
        if any(e is not copy.out and other in state_downstream(state, e.dst) for e in state.out_edges(transient)):
            return False
        order = StateOrder.of(state, scope[transient])
        written = [n for n in mine if any(not e.data.is_empty() for e in state.in_edges(n))]
        # Copying a transient this state never writes copies what an earlier run or the allocation left, and a read
        # before the state's first write takes that value too: the container does not hold it.
        if not written:
            return False
        for node in mine:
            read = any(not e.data.is_empty() for e in state.out_edges(node))
            if read and node not in written and not any(node in order.descendants(w) for w in written):
                return False
        writers = [order.top(e.src) for n in written for e in state.in_edges(n) if not e.data.is_empty()]
        readers = [
            order.top(e.dst) for n in mine for e in state.out_edges(n) if e is not copy.out and not e.data.is_empty()
        ]
        for where, writes, subset in self.accesses(state, order, other, copy.into if into else copy.out, into):
            if subset is not None and subsets.intersects(subset, at.subset, self.facts) is symbolic.Truth.FALSE:
                continue
            if into:
                # Copied out: reads all finish before the region starts to change, or see it complete.
                if writes or not (
                    all(w in order.descendants(where) for w in writers) or where in order.descendants(other)
                ):
                    return False
            elif writes and not (
                other in order.descendants(where) or all(where in order.descendants(r) for r in readers)
            ):
                return False
        return True

    def accesses(
        self, state: SDFGState, order: StateOrder, other: nodes.AccessNode, copy_edge, into: bool
    ) -> list[tuple[nodes.Node, bool, subsets.Subset | None]]:
        """Who touches ``other``'s container in ``order``'s scope, apart from the copy: ``(node, writes, subset)``,
        ``node`` directly in that scope. ``other``'s readers run after the copy, so they count only copied in."""
        found = []
        name = self.viewed.get(other.data, other.data)
        for node in state.data_nodes():
            if self.viewed.get(node.data, node.data) != name:
                continue
            # Through another name of the container, any of it may be touched.
            exact = node.data == other.data
            place = order.top(node)
            # Ordered around the whole scope; through the scope's connectors its accesses are listed below.
            if place is None:
                if not (order.within in state_upstream(state, node) or order.within in state_downstream(state, node)):
                    found.append((order.within, True, None))
                continue
            found.extend(
                (order.top(e.src) or place, True, e.data.get_dst_subset(e, state) if exact else None)
                for e in state.in_edges(node)
                if e is not copy_edge and e not in self.bindings and not e.data.is_empty()
            )
            if node is other and into:
                continue
            found.extend(
                (order.top(e.dst) or place, False, e.data.get_src_subset(e, state) if exact else None)
                for e in state.out_edges(node)
                if e is not copy_edge and e not in self.bindings and not e.data.is_empty()
            )
        if order.within is not None:
            found.extend(
                (order.top(e.dst), False, e.data.subset if e.data.data == other.data else None)
                for e in state.out_edges(order.within)
                if self.viewed.get(e.data.data, e.data.data) == name
            )
        return found

    def rename(self, state: SDFGState, copy: Copy, transient: nodes.AccessNode, other: nodes.AccessNode, at: Placement):
        old = transient.data
        mine = [n for n in state.data_nodes() if n.data == old]
        done = set()
        for node in mine:
            for e in state.all_edges(node):
                if e is copy.out or e is copy.into:
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
        if copy.out is copy.into:
            state.remove_edge(copy.out)
        else:
            # Not ``remove_node``: the order the copy carried is the merged node itself.
            state.detach_node(copy.out.dst)
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
        # A transient nothing read leaves the source of its copy with no edge at all.
        if state.degree(other) == 0:
            state.remove_node(other)
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
class ContractTransients(ppl.Pass):
    """:class:`Contraction` alone, in every SDFG. Canonicalization runs it ahead of its last dead fill elimination: a
    fill of an array cut to what is stored and read is a fill the next whole-array write makes dead."""

    CATEGORY: str = "Memory Footprint Reduction"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Nodes | ppl.Modifies.Memlets))

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> int | None:
        """:returns: How many transients were contracted, or None if none."""
        return sum(Contraction(sd).run() for sd in sdfg.all_sdfgs_recursive()) or None


@properties.make_properties
@explicit_cf_compatible
class MinimizeTransients(ppl.Pass):
    """Fewer, smaller and longer-lived transients.

    1. Contract, in every SDFG: a transient of which one box only is ever read is that box (:class:`Contraction`).
    2. Coalesce: a transient living in one state and copied whole to or from a region of another container is that
       region (:class:`Coalescing`).
    3. Share and lift: transients whose uses never overlap share one buffer, and outside every parallel map scope every
       buffer sized by symbols bound on entry is ``Persistent``
       (:class:`~dace.transformation.passes.transient_reuse.BufferSharing`, :func:`sharing_storages`).

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
            changed += Contraction(sd).run()
            changed += Coalescing(sd).run()
            shared = sharing_storages(sd, tuple(storages))
            if shared:
                changed += len(BufferSharing(sd, shared).run())
        return changed or None
