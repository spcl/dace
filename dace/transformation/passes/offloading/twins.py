# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Twins: a container used on the side it does not live on is accessed, and copied, through a twin on that side."""
from typing import NamedTuple

from ordered_set import OrderedSet

from dace import data, dtypes, Memlet, subsets
from dace.sdfg import nodes, SDFG, SDFGState
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion

import dace.transformation.passes.offloading.offloading_helpers as helpers


def rename_interstate_reads(sdfg: SDFG, edges: list[MultiConnectorEdge]) -> None:
    """An interstate edge is host code, so it reads a device-resident container through its host twin."""
    for edge in edges:
        for name in edge.data.used_arrays(sdfg.arrays):
            if helpers.is_array(name, sdfg) and helpers.is_array_stored_on_GPU(sdfg, name):
                edge.data.replace(name, helpers.host_name(name))


def rename_header_reads(sdfg: SDFG, block: ControlFlowBlock, names: OrderedSet[str]) -> None:
    """Loop bounds and conditions are host code too; the blocks inside have their own turn."""
    block.replace_meta_accesses(
        {name: helpers.host_name(name)
         for name in names if helpers.is_array_stored_on_GPU(sdfg, name)})


def rename_in_state(sdfg: SDFG, state: SDFGState, where: dict[str, bool]) -> None:
    """Point ``state``'s accesses at the twin of every container that is on the side it does not live on."""
    names = OrderedSet(node.data for node in state.data_nodes())
    names |= OrderedSet(edge.data.data for edge in state.edges() if not edge.data.is_empty())
    rename: dict[str, str] = {}
    for name in names:
        on_gpu = where.get(name)
        if on_gpu is not None and name in sdfg.arrays and helpers.is_array_stored_on_GPU(sdfg, name) != on_gpu:
            rename[name] = helpers.gpu_name(name) if on_gpu else helpers.host_name(name)
    stage_views_with_their_origin(sdfg, state, rename)
    for access in state.data_nodes():
        if access.data in rename:
            access.data = rename[access.data]
    for edge in state.edges():
        if not edge.data.is_empty() and edge.data.data in rename:
            edge.data.data = rename[edge.data.data]


def stage_views_with_their_origin(sdfg: SDFG, state: SDFGState, rename: dict[str, str]) -> None:
    """Give a view of a staged container a twin on the same side: ``C -> C_gpu`` makes ``C_0 -> C_0_gpu``, since one
    descriptor cannot serve a host state and a kernel (npbench mandelbrot2)."""
    for access in state.data_nodes():
        name = access.data
        if name in rename or name not in sdfg.arrays or not isinstance(sdfg.arrays[name], data.View):
            continue
        origin = helpers.view_origin(state, access)
        if origin is None or origin.data not in rename:
            continue
        # Read off the name: the staged container is only registered by the copies.
        to_gpu = rename[origin.data] == helpers.gpu_name(origin.data)
        twin = helpers.gpu_name(name) if to_gpu else helpers.host_name(name)
        if twin not in sdfg.arrays:
            desc = sdfg.arrays[name]
            sdfg.add_view(twin,
                          desc.shape,
                          desc.dtype,
                          storage=dtypes.StorageType.GPU_Global if to_gpu else dtypes.StorageType.Default,
                          strides=desc.strides,
                          offset=desc.offset)
        rename[name] = twin


def place_views(sdfg: SDFG) -> None:
    """A view lives where the container it aliases lives (npbench mlp, correlation), except in a kernel, where it is
    a register: codegen mishandles ``GPU_Global`` there."""
    for state in sdfg.states():
        scopes = state.scope_dict()
        for node in state.data_nodes():
            desc = sdfg.arrays[node.data]
            if not isinstance(desc, data.View):
                continue
            parent = scopes[node]
            if isinstance(parent, nodes.MapEntry) and parent.schedule in dtypes.GPU_SCHEDULES:
                desc.storage = dtypes.StorageType.Register
                continue
            origin = helpers.view_origin(state, node)
            if origin is not None:
                desc.storage = sdfg.arrays[origin.data].storage


def insert_copy_state(sdfg: SDFG, region: ControlFlowRegion, block: ControlFlowBlock, before: bool,
                      names: OrderedSet[str], to_gpu: bool) -> None:
    """Insert a state copying ``names`` before or after ``block`` of ``region``; ``to_gpu`` says which way."""
    label = f"copy_{'_'.join(sorted(names))}_{'to_gpu' if to_gpu else 'to_host'}"
    if before:
        state = region.add_state_before(block, label, is_start_block=block is region.start_block)
    else:
        state = region.add_state_after(block, label)

    for name in names:
        twin = helpers.twin_name(sdfg, name)
        # Leaving the home fills the twin; coming back restores the home.
        src, dst = (name, twin) if helpers.is_array_stored_on_GPU(sdfg, name) != to_gpu else (twin, name)
        if twin not in sdfg.arrays:
            register_twin(sdfg, twin, name)
        # A view has no storage: each side re-derives it from its container, whose copy is here.
        if isinstance(sdfg.arrays[name], data.View):
            continue
        state.add_edge(
            state.add_access(src), None, state.add_access(dst), None,
            Memlet(data=src,
                   subset=subsets.Range.from_array(sdfg.arrays[src]),
                   other_subset=subsets.Range.from_array(sdfg.arrays[dst])))


def register_twin(sdfg: SDFG, twin: str, home: str) -> None:
    """Declare ``twin``, the copy of ``home`` on the other side."""
    desc = sdfg.arrays[home]
    on_device = helpers.is_array_stored_on_GPU(sdfg, home)
    storage = dtypes.StorageType.Default if on_device else dtypes.StorageType.GPU_Global
    if isinstance(desc, data.View):
        sdfg.add_view(twin, desc.shape, desc.dtype, storage=storage)
    else:
        sdfg.add_array(twin, desc.shape, desc.dtype, storage=storage, transient=True)


class Copy(NamedTuple):
    """Copy ``names`` to the device or back, as a new state before or after ``block`` of ``region``."""
    region: ControlFlowRegion
    block: ControlFlowBlock
    before: bool
    names: OrderedSet[str]
    to_gpu: bool


class CopyPlan:
    """The copies of one SDFG, decided before any state exists.

    A copy toward a container's home only carries writes of its twin, so it is dropped when nothing writes the
    twin. A container neither side writes keeps its entry value, so its twin is filled once at the program's entry
    instead of wherever the copy was planned.
    """

    __slots__ = ('copies', 'sdfg', 'skip_copy_in')

    def __init__(self, sdfg: SDFG) -> None:
        self.sdfg = sdfg
        # Read before renaming moves the accesses onto the twins.
        self.skip_copy_in = helpers.overwritten_before_any_read(sdfg)
        self.copies: list[Copy] = []

    def add(self, region: ControlFlowRegion, block: ControlFlowBlock, before: bool, names: OrderedSet[str],
            to_gpu: bool) -> None:
        self.copies.append(Copy(region, block, before, names, to_gpu))

    def insert(self) -> None:
        """Turn the plan into states; call after every access is renamed."""
        sdfg = self.sdfg
        written = OrderedSet(node.data for state in sdfg.states() for node in state.data_nodes()
                             if state.in_degree(node) > 0)
        fills: dict[bool, OrderedSet[str]] = {True: OrderedSet(), False: OrderedSet()}
        for copy in self.copies:
            names = OrderedSet(name for name in copy.names if not (copy.to_gpu and name in self.skip_copy_in) and (
                helpers.is_array_stored_on_GPU(sdfg, name) != copy.to_gpu or helpers.twin_name(sdfg, name) in written))
            constant = OrderedSet(name for name in names
                                  if name not in written and helpers.twin_name(sdfg, name) not in written)
            fills[copy.to_gpu] |= constant
            if names - constant:
                insert_copy_state(sdfg, copy.region, copy.block, copy.before, names - constant, copy.to_gpu)
        for to_gpu, names in fills.items():
            if names:
                insert_copy_state(sdfg, sdfg, sdfg.start_block, True, names, to_gpu)
