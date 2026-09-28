# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Materialize the offloading IR: rename each access to the side it runs on, and copy where a location changes."""
from typing import Dict, Optional, Tuple

from ordered_set import OrderedSet

from dace import dtypes, Memlet, subsets
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import AbstractControlFlowRegion, ControlFlowBlock, SDFGState
from dace.sdfg.utils import get_view_node

from dace.transformation.passes.offloading.offloading_ir_node import OffloadingIRNode
import dace.transformation.passes.offloading.offloading_helpers as helpers


class CopyInsertion:
    """One copy insertion into ``sdfg`` from its placement IR; run it with :meth:`apply`."""

    __slots__ = ('sdfg', 'scopes', 'no_copy_in_needed', 'written', 'placed_on_gpu', 'placed', 'entry_fills')

    def __init__(self, sdfg: SDFG, scopes: Dict[SDFGState, Dict[nodes.Node, Optional[nodes.Node]]]) -> None:
        self.sdfg = sdfg
        self.scopes = scopes
        # Read before renaming moves the accesses onto the twins.
        self.no_copy_in_needed = helpers.overwritten_before_any_read(sdfg)
        self.written: OrderedSet[str] = OrderedSet()
        #: Transients this insertion put in device memory.
        self.placed_on_gpu: OrderedSet[str] = OrderedSet()
        # Directions each container is already copied in at one program point, keyed (block, side).
        self.placed: Dict[Tuple[ControlFlowBlock, str], Dict[str, OrderedSet[bool]]] = {}
        # Fills of containers neither side writes, by direction: placed once, at the program's entry.
        self.entry_fills: Dict[bool, OrderedSet[str]] = {True: OrderedSet(), False: OrderedSet()}

    def apply(self, IR: OffloadingIRNode) -> None:
        self.place_transients(IR)
        self.place_views(keep_registers=False)
        helpers.traverse_IR(IR, self.rename_node)
        # After renaming: a write on the side a container does not live on now names its twin.
        self.written = helpers.containers_written(self.sdfg)
        helpers.traverse_IR(IR, self.insert_node_copies)
        for to_gpu, names in self.entry_fills.items():
            if names:
                self.create_interstate_copy(None, self.sdfg.start_block, names, to_gpu=to_gpu)
        # Only now does every container carry its final storage, so only now can a view follow it.
        self.place_views(keep_registers=True)

    def place_transients(self, IR: OffloadingIRNode) -> None:
        """A transient lives on the side of the first IR node that names it."""
        seen: OrderedSet[str] = OrderedSet()

        def place(node: OffloadingIRNode) -> None:
            for storage, names in ((dtypes.StorageType.GPU_Global, node.gpu_set), (dtypes.StorageType.Default,
                                                                                   node.cpu_set)):
                for name in names:
                    if self.sdfg.arrays[name].transient and name not in seen:
                        self.sdfg.arrays[name].storage = storage
                        seen.add(name)
                        if storage == dtypes.StorageType.GPU_Global:
                            self.placed_on_gpu.add(name)

        helpers.traverse_IR(IR, place)

    def place_views(self, keep_registers: bool) -> None:
        """A view lives where the container it aliases lives (npbench mlp, correlation), except in a kernel,
        where it is a register: codegen mishandles ``GPU_Global`` there. ``keep_registers`` keeps those."""
        sdfg = self.sdfg
        for state in sdfg.states():
            scope = self.scopes.get(state, {})
            for node in state.data_nodes():
                if not helpers.is_view(node.data, sdfg):
                    continue
                desc = sdfg.arrays[node.data]
                parent = scope.get(node)
                if isinstance(parent, nodes.MapEntry) and helpers.has_GPU_schedule(parent):
                    desc.storage = dtypes.StorageType.Register
                    continue
                if keep_registers and desc.storage == dtypes.StorageType.Register:
                    continue
                origin = helpers.view_origin(state, node)
                if origin is not None and origin in sdfg.arrays:
                    desc.storage = sdfg.arrays[origin].storage

    def rename_node(self, node: OffloadingIRNode) -> None:
        """Point ``node``'s accesses at the twin of every container used on the side it does not live on."""
        sdfg = self.sdfg
        rename_dict = {}
        for name in node.gpu_set:
            if not helpers.is_array_stored_on_GPU(sdfg, name):
                rename_dict[name] = helpers.gpu_name(name)
        for name in node.cpu_set:
            if helpers.is_array_stored_on_GPU(sdfg, name):
                rename_dict[name] = helpers.host_name(name)

        block = node.block
        if block is None:
            return
        # An interstate edge is host code, so it reads a device-resident container through its host twin;
        # whether a copy is needed is decided later.
        if isinstance(block.parent_graph, AbstractControlFlowRegion):
            for edge in block.parent_graph.in_edges(block):
                for name in edge.data.used_arrays(rename_dict):
                    if helpers.is_array_stored_on_GPU(sdfg, name):
                        edge.data.replace(name, helpers.host_name(name))
        # An EDGE node decides for its edges only, not for the block they reach.
        if node.type == OffloadingIRNode.EDGE:
            return
        if isinstance(block, SDFGState):
            self.rename_in_state(block, rename_dict)
        else:
            # Loop bounds and conditions; the blocks inside have IR nodes of their own.
            block.replace_meta_accesses(rename_dict)

    def rename_in_state(self, state: SDFGState, rename_dict: Dict[str, str]) -> None:
        self.stage_views_with_their_origin(state, rename_dict)
        self.rename_accesses(state, rename_dict)

    @staticmethod
    def rename_accesses(state: SDFGState, rename_dict: Dict[str, str]) -> None:
        """Point every access node and memlet of ``state`` naming a key of ``rename_dict`` at its value."""
        for access in state.data_nodes():
            if access.data in rename_dict:
                access.data = rename_dict[access.data]
        for edge in state.edges():
            if not edge.data.is_empty() and edge.data.data in rename_dict:
                edge.data.data = rename_dict[edge.data.data]

    def stage_views_with_their_origin(self, state: SDFGState, rename_dict: Dict[str, str]) -> None:
        """Give a view of a staged container a twin on the same side: ``C -> C_gpu`` makes ``C_0 -> C_0_gpu``,
        since one descriptor cannot serve a host state and a kernel (npbench mandelbrot2)."""
        sdfg = self.sdfg
        for access in state.data_nodes():
            name = access.data
            if name in rename_dict or name not in sdfg.arrays or not helpers.is_view(name, sdfg):
                continue
            origin = self.origin_of_a_staged_view(state, access)
            if origin is None or origin not in rename_dict:
                continue
            # Read off the name: the staged container is only registered by the copy insertion.
            to_gpu = rename_dict[origin] == helpers.gpu_name(origin)
            twin = helpers.gpu_name(name) if to_gpu else helpers.host_name(name)
            if twin not in sdfg.arrays:
                desc = sdfg.arrays[name]
                sdfg.add_view(twin,
                              desc.shape,
                              desc.dtype,
                              storage=dtypes.StorageType.GPU_Global if to_gpu else dtypes.StorageType.Default,
                              strides=desc.strides,
                              offset=desc.offset)
            rename_dict[name] = twin

    def origin_of_a_staged_view(self, state: SDFGState, access: nodes.AccessNode) -> Optional[str]:
        """The container ``access`` aliases, or None where the chain meets a non-access node (npbench trmm)
        or a twin not yet registered."""
        sdfg = self.sdfg
        node = access
        seen: OrderedSet[str] = OrderedSet()
        while isinstance(node, nodes.AccessNode) and node.data in sdfg.arrays and helpers.is_view(node.data, sdfg):
            if node.data in seen:  # a cycle is not a chain to a container
                return None
            seen.add(node.data)
            node = get_view_node(state, node)
        if not isinstance(node, nodes.AccessNode) or node.data not in sdfg.arrays:
            return None
        return node.data

    def insert_node_copies(self, node: OffloadingIRNode) -> None:
        """Copy wherever the location of an array changes between ``node`` and a successor."""
        if node.type == OffloadingIRNode.CLOSE and node.open.type == OffloadingIRNode.OPEN_LOOP:
            self.insert_loop_copies(node)
        for next in node.next:
            if node.cpu_set & node.gpu_set:
                raise NotImplementedError(f"This pass does not support copies within a single state. State "
                                          f"{node.debug_name} uses {node.cpu_set & node.gpu_set} on both sides.")
            # Before an interstate edge, or between two CLOSEs, there is no next block: copy after the node.
            if next.type == OffloadingIRNode.EDGE or (node.type == OffloadingIRNode.CLOSE
                                                      and next.type == OffloadingIRNode.CLOSE):
                self.insert_copies(node, next, block_after(node), None)
            else:
                self.insert_copies(node, next, node.block, next.block)

    def insert_loop_copies(self, close: OffloadingIRNode) -> None:
        """At the end of each iteration, bring back to where the next iteration starts what the body moved."""
        tails = close.open.get_all_tails()
        gpu_copies = close.cpu_set & close.open.gpu_set
        cpu_copies = close.gpu_set & close.open.cpu_set
        for names, to_gpu in ((gpu_copies, True), (cpu_copies, False)):
            if names:
                for tail in tails:
                    self.place_copy(block_after(tail), None, names, to_gpu=to_gpu)
        # The copies sit at the end of the body, so the loop leaves with them done.
        close.gpu_set = (close.gpu_set | gpu_copies) - cpu_copies
        close.cpu_set = (close.cpu_set | cpu_copies) - gpu_copies

    def insert_copies(self, node: OffloadingIRNode, next: OffloadingIRNode, node_block: Optional[ControlFlowBlock],
                      next_block: Optional[ControlFlowBlock]) -> None:
        gpu_copies = OrderedSet(name for name in node.cpu_set & next.gpu_set if name not in self.no_copy_in_needed)
        if gpu_copies:
            self.place_copy(node_block, next_block, gpu_copies, to_gpu=True)
        cpu_copies = node.gpu_set & next.cpu_set
        if cpu_copies:
            self.place_copy(node_block, next_block, cpu_copies, to_gpu=False)

    def place_copy(self, before: Optional[ControlFlowBlock], after: Optional[ControlFlowBlock],
                   array_names: OrderedSet[str], to_gpu: bool) -> None:
        """Copy ``array_names`` between ``before`` and ``after``, once per program point and direction.

        A copy toward a container's home only carries writes of its twin, so it is dropped when nothing
        writes the twin. A container neither side writes keeps its entry value, so its twin is filled
        once at the program's entry instead of wherever the IR moves it.
        """
        sdfg = self.sdfg
        array_names = OrderedSet(
            name for name in array_names
            if helpers.is_array_stored_on_GPU(sdfg, name) != to_gpu or helpers.twin_name(sdfg, name) in self.written)
        constant = OrderedSet(name for name in array_names
                              if name not in self.written and helpers.twin_name(sdfg, name) not in self.written)
        self.entry_fills[to_gpu] |= constant
        point = (after, 'before') if after is not None else (before, 'after')
        directions = self.placed.setdefault(point, {})
        fresh = OrderedSet(name for name in array_names
                           if name not in constant and directions.get(name) != OrderedSet([to_gpu]))
        for name in fresh:
            directions.setdefault(name, OrderedSet()).add(to_gpu)
        if fresh:
            self.create_interstate_copy(before, after, fresh, to_gpu=to_gpu)

    def create_interstate_copy(self, before: Optional[ControlFlowBlock], after: Optional[ControlFlowBlock],
                               array_names: OrderedSet[str], to_gpu: bool) -> None:
        """One new state, before ``after`` or else after ``before``, copying every name in ``array_names``."""
        assert before is not None or after is not None, "invalid: both states are None"
        sdfg = self.sdfg
        label = f"copy_{'_'.join(sorted(array_names))}_{'to_gpu' if to_gpu else 'to_host'}"
        if after is not None:
            region = after.parent_graph
            copy_state = region.add_state_before(after, label=label, is_start_block=after is region.start_block)
        else:
            region = before.parent_graph if before.parent_graph else before
            copy_state = region.add_state_after(before, label=label)

        for name in array_names:
            twin = helpers.twin_name(sdfg, name)
            # Leaving the home fills the twin; coming back restores the home.
            src, dst = (name, twin) if helpers.is_array_stored_on_GPU(sdfg, name) != to_gpu else (twin, name)
            if twin not in sdfg.arrays:
                register_twin(sdfg, twin, name)
            # A view has no storage: each side re-derives it from its container, whose copy is here.
            if helpers.is_view(name, sdfg):
                continue
            copy_state.add_edge(
                copy_state.add_access(src), None, copy_state.add_access(dst), None,
                Memlet(data=src,
                       subset=subsets.Range.from_array(sdfg.arrays[src]),
                       other_subset=subsets.Range.from_array(sdfg.arrays[dst])))


def block_after(node: OffloadingIRNode) -> ControlFlowBlock:
    """The block a copy after ``node`` follows: a CLOSE node has none, so the region it closes."""
    return node.open.block if node.type == OffloadingIRNode.CLOSE else node.block


def register_twin(sdfg: SDFG, twin: str, home: str) -> None:
    """Declare ``twin``, the copy of ``home`` on the other side."""
    desc = sdfg.arrays[home]
    on_gpu = helpers.is_array_stored_on_GPU(sdfg, home)
    storage = dtypes.StorageType.Default if on_gpu else dtypes.StorageType.GPU_Global
    if helpers.is_view(home, sdfg):
        sdfg.add_view(twin, desc.shape, desc.dtype, storage=storage)
    else:
        sdfg.add_array(twin, desc.shape, desc.dtype, storage=storage, transient=True)
