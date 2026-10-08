# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Where each state wants its containers: on the host or on the device."""

from typing import NamedTuple

from ordered_set import OrderedSet

from dace import data, dtypes
from dace.sdfg import nodes, SDFG, SDFGState
from dace.sdfg.graph import MultiConnectorEdge

import dace.transformation.passes.offloading.offloading_helpers as helpers


class Wants(NamedTuple):
    """The containers a state needs on the device and on the host; they never overlap in a settled state."""

    gpu: OrderedSet[str]
    cpu: OrderedSet[str]


class Locations:
    """The wants of the states of one SDFG.

    A kernel wants what it accesses on the device, host code (a top-level tasklet, a callback) on the host. A map
    of ``host_maps`` launches rather than computes: its body decides where the data it reaches goes. A state that
    wants a container on both sides is *hybrid*: it is recorded in ``hybrid_states`` and reported as wanting all
    of them on the device, which is where wrapping its host code in size-1 maps puts them.
    """

    __slots__ = ("host_maps", "hybrid_states", "sdfg")

    def __init__(self, sdfg: SDFG, host_maps: OrderedSet[nodes.MapEntry]) -> None:
        self.sdfg = sdfg
        self.host_maps = host_maps
        self.hybrid_states: OrderedSet[SDFGState] = OrderedSet()

    def of_state(self, state: SDFGState) -> Wants:
        gpu: OrderedSet[str] = OrderedSet()
        cpu: OrderedSet[str] = OrderedSet()
        for node in state.scope_children()[None]:
            if isinstance(node, nodes.MapEntry):
                self.add_map(state, node, gpu, cpu, False)
            elif isinstance(node, (nodes.LibraryNode, nodes.NestedSDFG)):
                self.add_host_level_node(state, node, gpu, cpu)
            elif isinstance(node, nodes.Tasklet):  # outside every map, so on the host
                cpu |= self.used_by_node(state, node)
            elif not isinstance(node, (nodes.MapExit, nodes.AccessNode)):
                raise RuntimeError(f"Unknown node {node} of type {type(node).__name__} in state {state}.")

        if gpu & cpu:
            self.hybrid_states.add(state)
            return Wants(gpu | cpu, OrderedSet())
        return Wants(gpu, cpu)

    def of_sdfg(self) -> Wants:
        """What the whole SDFG wants: every state, and the host reads of its interstate edges and headers."""
        gpu: OrderedSet[str] = OrderedSet()
        cpu: OrderedSet[str] = OrderedSet()
        for state in self.sdfg.states():
            wants = self.of_state(state)
            gpu |= wants.gpu
            cpu |= wants.cpu
        for edge in self.sdfg.all_interstate_edges():
            cpu |= OrderedSet(edge.data.used_arrays(self.sdfg.arrays))
        for region in self.sdfg.all_control_flow_regions():
            cpu |= OrderedSet(memlet.data for memlet in region.get_meta_read_memlets())
        return Wants(gpu, cpu)

    def used_by_edge(self, state: SDFGState, edge: MultiConnectorEdge, is_out_edge: bool) -> OrderedSet[str]:
        sdfg = self.sdfg
        if edge.data.is_empty():
            return OrderedSet()
        name = edge.data.data
        desc = sdfg.arrays[name]
        if helpers.is_array(name, sdfg):
            return OrderedSet([name])
        anchor: nodes.Node | None
        if isinstance(desc, data.View):
            anchor = next((node for node in state.data_nodes() if node.data == name), None)
        elif isinstance(desc, data.Scalar):  # might be a scalar access of an array slice
            anchor = edge.dst if is_out_edge else edge.src
        elif helpers.is_unoffloadable(desc) or isinstance(desc, data.Stream) or name in sdfg.constants:
            # No single location to decide: a structure or container array, a stream (a queue the code generator
            # allocates where its pusher runs), a constant (declared on both sides).
            return OrderedSet()
        else:
            raise RuntimeError(f"Unknown data type (not array, scalar, view or stream) on edge {edge}: {edge.data}")
        if isinstance(anchor, nodes.AccessNode):
            return helpers.get_data_used_by_access_nodes(sdfg, state, anchor, downstream=is_out_edge)
        return OrderedSet()

    def used_by_node(self, state: SDFGState, node: nodes.Node) -> OrderedSet[str]:
        arrays: OrderedSet[str] = OrderedSet()
        for edge in state.in_edges(node):
            arrays |= self.used_by_edge(state, edge, False)
        for edge in state.out_edges(node):
            arrays |= self.used_by_edge(state, edge, True)
        arrays |= helpers.get_data_used_by_access_nodes(self.sdfg, state, node, downstream=False)
        arrays |= helpers.get_data_used_by_access_nodes(self.sdfg, state, node, downstream=True)
        return arrays

    def add_map(
        self, state: SDFGState, entry: nodes.MapEntry, gpu: OrderedSet[str], cpu: OrderedSet[str], on_device: bool
    ) -> None:
        """Add what ``entry``'s scope accesses: on the device if it or an enclosing map is a kernel.

        A host map is transparent: what its body leaves unclaimed goes to the device, the side of its kernels.
        """
        on_device = on_device or entry.map.schedule in dtypes.GPU_SCHEDULES
        launcher = (
            not on_device
            and entry in self.host_maps
            and any(
                helpers.is_device_work(node)
                for node in state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes()
            )
        )
        boundary = helpers.get_data_used_by_access_nodes(
            self.sdfg, state, entry, downstream=False
        ) | helpers.get_data_used_by_access_nodes(self.sdfg, state, state.exit_node(entry), downstream=True)
        if not launcher:
            (gpu if on_device else cpu).update(boundary)

        for node in state.scope_children()[entry]:
            if isinstance(node, nodes.MapEntry):
                self.add_map(state, node, gpu, cpu, on_device)
            elif launcher and isinstance(node, (nodes.LibraryNode, nodes.NestedSDFG)):
                self.add_host_level_node(state, node, gpu, cpu)
            else:
                self.add_scope_node(state, node, entry, gpu, cpu, on_device)
        if launcher:
            gpu |= boundary - cpu

    def add_scope_node(
        self,
        state: SDFGState,
        node: nodes.Node,
        entry: nodes.MapEntry,
        gpu: OrderedSet[str],
        cpu: OrderedSet[str],
        on_device: bool,
    ) -> None:
        """Add what ``node``, inside ``entry``'s scope, accesses beyond the scope boundary already counted."""
        if isinstance(node, nodes.AccessNode):
            used = helpers.get_data_used_by_access_nodes(self.sdfg, state, node, downstream=True)
        elif isinstance(node, nodes.Tasklet):
            used = self.used_by_node(state, node)
        elif isinstance(node, (nodes.NestedSDFG, nodes.MapExit, nodes.LibraryNode)):
            used = OrderedSet()
        else:
            raise RuntimeError(f"Unknown node {node} of type {type(node).__name__} inside map {entry}")
        for name in used:
            if name in gpu and not on_device:
                raise RuntimeError(f"{name} is used on the device and on the host inside map {entry}")
            if on_device or name not in gpu:
                (gpu if on_device else cpu).add(name)

    def add_host_level_node(
        self, state: SDFGState, node: nodes.Node, gpu: OrderedSet[str], cpu: OrderedSet[str]
    ) -> None:
        """A library node at a host level goes where its schedule runs; a nested SDFG where its body wants its
        bound arrays."""
        if isinstance(node, nodes.LibraryNode):
            (gpu if node.schedule in dtypes.GPU_SCHEDULES else cpu).update(self.used_by_node(state, node))
            return
        inner = Locations(node.sdfg, self.host_maps).of_sdfg()
        for edge in state.all_edges(node):
            connector = edge.dst_conn if edge.dst is node else edge.src_conn
            if edge.data.is_empty() or not helpers.is_array(edge.data.data, self.sdfg):
                continue
            if connector in inner.gpu:
                gpu.add(edge.data.data)
            elif connector in inner.cpu:
                cpu.add(edge.data.data)
