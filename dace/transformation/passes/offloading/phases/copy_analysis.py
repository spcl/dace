# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Build the offloading IR: per block, which arrays are wanted on the host and which on the device."""

from ordered_set import OrderedSet

import dace.transformation.passes.offloading.offloading_helpers as helpers
from dace import data, dtypes
from dace.sdfg import SDFG, nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.transformation.passes.offloading.offloading_ir_node import OffloadingIRNode

Locations = tuple[OrderedSet[str], OrderedSet[str]]


class CopyAnalysis:
    """One analysis of ``sdfg``; ``hybrid_states`` collects the states that touch an array from both sides.

    A map of ``host_maps`` launches rather than computes: its body decides where the data it reaches goes.
    """

    __slots__ = ("sdfg", "scopes", "host_maps", "hybrid_states")

    def __init__(
        self,
        sdfg: SDFG,
        scopes: dict[SDFGState, dict[nodes.Node, nodes.Node | None]],
        host_maps: OrderedSet[nodes.MapEntry],
    ) -> None:
        self.sdfg = sdfg
        self.scopes = scopes
        self.host_maps = host_maps
        self.hybrid_states: OrderedSet[SDFGState] = OrderedSet()

    def build_ir(self) -> OffloadingIRNode:
        sdfg = self.sdfg
        # A view is placed with the container it aliases; a structure or container array is not placed.
        non_transients = OrderedSet(
            name for name, desc in sdfg.arrays.items() if not desc.transient and helpers.is_array(name, sdfg)
        )
        initially_on_gpu = OrderedSet(name for name in non_transients if helpers.is_array_stored_on_GPU(sdfg, name))
        initially_on_cpu = non_transients - initially_on_gpu

        IR = OffloadingIRNode.new_open_node(sdfg)
        IR.gpu_set = initially_on_gpu.copy()
        IR.cpu_set = initially_on_cpu.copy()
        self.parse_region(sdfg, IR).append_node(IR.close)
        helpers.link_early_returns(IR)
        # Arrays end up where they started.
        IR.close.gpu_set = initially_on_gpu
        IR.close.cpu_set = initially_on_cpu

        # Snapshot before propagation: the names a node inherits are the ones the hoist may move.
        own_use: dict[OffloadingIRNode, OrderedSet[str]] = {}

        def record(node: OffloadingIRNode) -> None:
            own_use[node] = node.cpu_set | node.gpu_set

        helpers.traverse_IR(IR, record)
        self.propagate(IR)
        hoist_device_copies(IR, own_use)
        return IR

    def parse_region(self, cfr: ControlFlowRegion, curr_node: OffloadingIRNode) -> OffloadingIRNode:
        """Append the IR of ``cfr``'s blocks after ``curr_node``, as a line; return the last node."""
        for block in cfr.bfs_nodes():
            in_edge_arrays: OrderedSet[str] = OrderedSet()
            for edge in cfr.in_edges(block):
                in_edge_arrays |= OrderedSet(
                    name for name in edge.data.used_arrays(self.sdfg.arrays) if helpers.is_array(name, self.sdfg)
                )
            if in_edge_arrays:
                curr_node = append(curr_node, OffloadingIRNode.new_edge_node(block, in_edge_arrays))

            if isinstance(block, SDFGState):
                gpu_set, cpu_set = self.locations_of_state(block)
                curr_node = append(curr_node, OffloadingIRNode.new_state_node(block, cpu_set, gpu_set))
            elif not isinstance(block, (ReturnBlock, ContinueBlock, BreakBlock)):
                curr_node = self.parse_container(block, curr_node)
        return curr_node

    def parse_container(self, block: ControlFlowRegion, curr_node: OffloadingIRNode) -> OffloadingIRNode:
        outer_node = append(curr_node, OffloadingIRNode.new_open_node(block))
        curr_node = outer_node
        if isinstance(block, (ConditionalBlock, LoopRegion)):
            # A condition or a loop header reads on the host.
            meta_data = OrderedSet(
                memlet.data for memlet in block.get_meta_read_memlets() if memlet.data in self.sdfg.arrays
            )
            if meta_data:
                curr_node = append(curr_node, OffloadingIRNode.new_state_node(block, meta_data, OrderedSet()))
        if isinstance(block, ConditionalBlock):
            for _, branch in block.branches:
                self.parse_region(branch, curr_node).append_node(outer_node.close)
        elif isinstance(block, ControlFlowRegion):
            self.parse_region(block, curr_node).append_node(outer_node.close)
        else:
            raise RuntimeError(f"Unknown block type: {block} of type {type(block).__name__}")

        # The first location of each array in the section opens it, the last one closes it. With
        # several children or routes there is no single answer, and propagation fills the sets later.
        if len(outer_node.next) == 1:
            outer_node.gpu_set, outer_node.cpu_set = section_locations(outer_node, first=True)
        if outer_node.has_one_route():
            outer_node.close.gpu_set, outer_node.close.cpu_set = section_locations(outer_node, first=False)
        return outer_node.close

    def propagate(self, IR: OffloadingIRNode) -> None:
        """Hand every array a node does not use on to its successors where it last was."""

        def forward(node: OffloadingIRNode) -> None:
            if node.type == OffloadingIRNode.STATE and isinstance(node.block, SDFGState):
                self.place_copy_destinations(node)
            for next in node.next:
                next_arrays = next.cpu_set | next.gpu_set
                next.cpu_set |= OrderedSet(name for name in node.cpu_set if name not in next_arrays)
                next.gpu_set |= OrderedSet(name for name in node.gpu_set if name not in next_arrays)

        # A node forwards what it holds when visited, so a join must first hear from every arm.
        helpers.traverse_IR_after_predecessors(IR, forward)

    def place_copy_destinations(self, node: OffloadingIRNode) -> None:
        """A top-level container-to-container copy writes its destination on its source's side.

        The state analysis leaves such a copy unplaced, so the destination kept its earlier location
        and a later reader on the other side got no copy in.
        """
        state = node.block
        top = self.scopes[state]
        for edge in state.edges():
            src, dst = edge.src, edge.dst
            if not (isinstance(src, nodes.AccessNode) and isinstance(dst, nodes.AccessNode)) or edge.data.is_empty():
                continue
            if top[src] is not None or top[dst] is not None or dst.data in node.cpu_set | node.gpu_set:
                continue
            if not helpers.is_array(dst.data, self.sdfg):
                continue
            if src.data in node.gpu_set:
                node.gpu_set.add(dst.data)
            elif src.data in node.cpu_set:
                node.cpu_set.add(dst.data)

    def arrays_used_by_edge(self, state: SDFGState, edge: MultiConnectorEdge, is_out_edge: bool) -> OrderedSet[str]:
        sdfg = self.sdfg
        if edge.data.is_empty():
            return OrderedSet()
        name = edge.data.data
        if helpers.is_array(name, sdfg):
            return OrderedSet([name])
        if isinstance(sdfg.arrays[name], data.View):
            view = next((node for node in state.data_nodes() if node.data == name), None)
            if view is None:
                return OrderedSet()
            return helpers.get_data_used_by_access_nodes(sdfg, state, view, downstream=is_out_edge)
        if isinstance(sdfg.arrays[name], data.Scalar):  # might be a scalar access of an array slice
            neighbor = edge.dst if is_out_edge else edge.src
            if isinstance(neighbor, nodes.AccessNode):
                return helpers.get_data_used_by_access_nodes(sdfg, state, neighbor, downstream=is_out_edge)
            return OrderedSet()
        # A structure or container array has no single location to decide, a Stream is a queue with its own
        # device-side protocol that the code generator allocates where its pusher runs, and a constant is
        # declared on both sides.
        if helpers.is_unoffloadable(name, sdfg) or isinstance(sdfg.arrays[name], data.Stream) or name in sdfg.constants:
            return OrderedSet()
        raise RuntimeError(f"Unknown data type (not array, scalar, view or stream) on edge {edge}: {edge.data}")

    def arrays_used_by_node(self, state: SDFGState, node: nodes.Node) -> OrderedSet[str]:
        arrays: OrderedSet[str] = OrderedSet()
        for edge in state.in_edges(node):
            arrays |= self.arrays_used_by_edge(state, edge, False)
        for edge in state.out_edges(node):
            arrays |= self.arrays_used_by_edge(state, edge, True)
        arrays |= helpers.get_data_used_by_access_nodes(self.sdfg, state, node, downstream=False)
        arrays |= helpers.get_data_used_by_access_nodes(self.sdfg, state, node, downstream=True)
        return arrays

    def locations_of_map(
        self,
        state: SDFGState,
        map_entry: nodes.MapEntry,
        gpu_set: OrderedSet[str],
        cpu_set: OrderedSet[str],
        is_gpu: bool,
    ) -> None:
        """Add what ``map_entry``'s scope accesses: on the device if it or an enclosing map is a kernel.

        A host map is transparent: what its body leaves unclaimed goes to the device, the side of its kernels.
        """
        is_gpu = is_gpu or map_entry.map.schedule in dtypes.GPU_SCHEDULES
        launcher = (
            not is_gpu
            and map_entry in self.host_maps
            and any(helpers.is_device_work(node) for node in helpers.scope_nodes(state, map_entry))
        )
        boundary = helpers.get_data_used_by_access_nodes(
            self.sdfg, state, map_entry, downstream=False
        ) | helpers.get_data_used_by_access_nodes(self.sdfg, state, state.exit_node(map_entry), downstream=True)
        if not launcher:
            (gpu_set if is_gpu else cpu_set).update(boundary)

        for node in [n for n, parent in self.scopes[state].items() if parent is map_entry]:
            if isinstance(node, nodes.MapEntry):
                self.locations_of_map(state, node, gpu_set, cpu_set, is_gpu)
            elif launcher and isinstance(node, (nodes.LibraryNode, nodes.NestedSDFG)):
                self.add_host_level_node(state, node, gpu_set, cpu_set)
            else:
                self.add_scope_node(state, node, map_entry, gpu_set, cpu_set, is_gpu)
        if launcher:
            gpu_set |= boundary - cpu_set

    def add_scope_node(
        self,
        state: SDFGState,
        node: nodes.Node,
        map_entry: nodes.MapEntry,
        gpu_set: OrderedSet[str],
        cpu_set: OrderedSet[str],
        is_gpu: bool,
    ) -> None:
        for name in self.arrays_used_in_scope(state, node, map_entry):
            if name in gpu_set and not is_gpu:
                raise RuntimeError(f"{name} is used on the device and on the host inside map {map_entry}")
            if is_gpu or name not in gpu_set:
                (gpu_set if is_gpu else cpu_set).add(name)

    def arrays_used_in_scope(self, state: SDFGState, node: nodes.Node, map_entry: nodes.MapEntry) -> OrderedSet[str]:
        """What ``node``, inside ``map_entry``'s scope, accesses beyond the scope boundary already counted."""
        if isinstance(node, nodes.AccessNode):
            return helpers.get_data_used_by_access_nodes(self.sdfg, state, node, downstream=True)
        if isinstance(node, nodes.Tasklet):
            return self.arrays_used_by_node(state, node)
        if isinstance(node, (nodes.NestedSDFG, nodes.MapExit, nodes.LibraryNode)):
            return OrderedSet()
        raise RuntimeError(f"Unknown node {node} of type {type(node).__name__} inside map {map_entry}")

    def add_host_level_node(
        self, state: SDFGState, node: nodes.Node, gpu_set: OrderedSet[str], cpu_set: OrderedSet[str]
    ) -> None:
        """A library node at a host level goes where its schedule runs; a nested SDFG where its body wants its
        bound arrays."""
        if isinstance(node, nodes.LibraryNode):
            on_gpu = node.schedule in dtypes.GPU_SCHEDULES
            (gpu_set if on_gpu else cpu_set).update(self.arrays_used_by_node(state, node))
            return
        inner_gpu, inner_cpu = CopyAnalysis(
            node.sdfg, helpers.get_sdfg_scope_dict(node.sdfg), self.host_maps
        ).locations_of_sdfg()
        for edge in state.all_edges(node):
            connector = edge.dst_conn if edge.dst is node else edge.src_conn
            if edge.data.is_empty() or not helpers.is_array(edge.data.data, self.sdfg):
                continue
            if connector in inner_gpu:
                gpu_set.add(edge.data.data)
            elif connector in inner_cpu:
                cpu_set.add(edge.data.data)

    def locations_of_sdfg(self) -> Locations:
        """Where every state, interstate edge and loop or branch condition of the SDFG wants each array."""
        gpu_set: OrderedSet[str] = OrderedSet()
        cpu_set: OrderedSet[str] = OrderedSet()
        for state in self.sdfg.states():
            state_gpu, state_cpu = self.locations_of_state(state)
            gpu_set |= state_gpu
            cpu_set |= state_cpu
        for edge in self.sdfg.all_interstate_edges():
            cpu_set |= OrderedSet(edge.data.used_arrays(self.sdfg.arrays))
        for region in self.sdfg.all_control_flow_regions():
            cpu_set |= OrderedSet(memlet.data for memlet in region.get_meta_read_memlets())
        return gpu_set, cpu_set

    def locations_of_state(self, state: SDFGState) -> Locations:
        """Where the top-level nodes of ``state`` want each array; a hybrid state is recorded and put on the device."""
        gpu_set: OrderedSet[str] = OrderedSet()
        cpu_set: OrderedSet[str] = OrderedSet()
        for node in state.scope_children()[None]:
            if isinstance(node, nodes.MapEntry):
                self.locations_of_map(state, node, gpu_set, cpu_set, False)
            elif isinstance(node, (nodes.LibraryNode, nodes.NestedSDFG)):
                self.add_host_level_node(state, node, gpu_set, cpu_set)
            elif isinstance(node, nodes.Tasklet):  # outside every map, so on the host
                cpu_set |= self.arrays_used_by_node(state, node)
            elif not isinstance(node, (nodes.MapExit, nodes.AccessNode)):
                raise RuntimeError(f"Unknown node {node} of type {type(node).__name__} in state {state}.")

        if gpu_set & cpu_set:
            self.hybrid_states.add(state)
            return gpu_set | cpu_set, OrderedSet()
        return gpu_set, cpu_set


def append(curr_node: OffloadingIRNode, node: OffloadingIRNode) -> OffloadingIRNode:
    curr_node.append_node(node)
    return node


def section_locations(IR: OffloadingIRNode, first: bool) -> Locations:
    """The first (or last) location of each array along the section ``IR`` opens, on its own level."""
    location_on_gpu: dict[str, bool] = {}

    def gather(node: OffloadingIRNode) -> None:
        for on_gpu, names in ((True, node.gpu_set), (False, node.cpu_set)):
            for name in names:
                if not first or name not in location_on_gpu:
                    location_on_gpu[name] = on_gpu

    helpers.traverse_same_level(IR, gather)
    return (
        OrderedSet(name for name, on_gpu in location_on_gpu.items() if on_gpu),
        OrderedSet(name for name, on_gpu in location_on_gpu.items() if not on_gpu),
    )


def hoist_device_copies(IR: OffloadingIRNode, own_use: dict[OffloadingIRNode, OrderedSet[str]]) -> None:
    """Move a host-to-device copy above the states that do not touch the array.

    Propagation walks forwards, so an array first used on the device late stays on the host until
    exactly that point, and its copy lands in the middle of a run of device states -- a host state a
    caller fusing that run into one persistent kernel cannot swallow. A state that never touches the
    array does not care where it is, so the copy moves above it, but only when every successor wants
    the array on the device: a branch leaving it on the host must not pay for it.
    """
    nodes_in_order: list[OffloadingIRNode] = []
    helpers.traverse_IR(IR, nodes_in_order.append)

    # Hoisting can free the state above, so this runs to a fixpoint; names never move back.
    changed = True
    while changed:
        changed = False
        for node in reversed(nodes_in_order):
            if node.type != OffloadingIRNode.STATE or not node.next:
                continue
            for name in OrderedSet(name for name in node.cpu_set if name not in own_use[node]):
                if all(name in next.gpu_set for next in node.next):
                    node.cpu_set.remove(name)
                    node.gpu_set.add(name)
                    changed = True
