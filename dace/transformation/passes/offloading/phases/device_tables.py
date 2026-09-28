# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fill a small table that kernels read on the device, instead of copying the host-filled one down mid-run."""
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from ordered_set import OrderedSet

from dace import data, dtypes
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import LoopRegion, SDFGState

import dace.transformation.passes.offloading.offloading_helpers as helpers
from dace.transformation.passes.offloading.phases.copy_insertion import CopyInsertion
from dace.transformation.passes.offloading.phases.single_iteration_maps import wrap_region_in_size1_map

#: Per table: the fill state, its fill tasklets, and the states reading a device twin (None: the fill moves).
Tables = Dict[str, Tuple[SDFGState, OrderedSet[nodes.Tasklet], Optional[List[SDFGState]]]]


def fill_device_tables(sdfg: SDFG) -> bool:
    """Fill every table of :func:`device_tables` on the device, one size-1 kernel per fill state.

    A table the host also reads keeps its host fill, and a device twin filled beside it serves the
    states that only kernels touch it in. Return whether anything changed.
    """
    regions: Dict[SDFGState, OrderedSet[nodes.Tasklet]] = {}
    for name, (state, tasklets, twin_states) in device_tables(sdfg).items():
        if twin_states is not None:
            tasklets = fill_device_twin(sdfg, state, tasklets, name, twin_states)
        regions.setdefault(state, OrderedSet()).update(tasklets)
    for state, region in regions.items():
        wrap_region_in_size1_map(state, region)
    return bool(regions)


def in_a_loop(block: SDFGState) -> bool:
    """A ``LoopRegion`` of the block's own SDFG encloses it."""
    region = block.parent_graph
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion):
            return True
        region = region.parent_graph
    return False


def fills_from_scalars(sdfg: SDFG, state: SDFGState, scopes: Dict, node: nodes.Node) -> bool:
    """A top-level tasklet with one output whose every input is a scalar, or that has none."""
    return (isinstance(node, nodes.Tasklet) and scopes[node] is None and state.out_degree(node) == 1 and all(
        edge.data.is_empty() or (isinstance(edge.src, nodes.AccessNode) and helpers.is_scalar(edge.src.data, sdfg))
        for edge in state.in_edges(node)))


def host_meta_reads(sdfg: SDFG) -> OrderedSet[str]:
    """Containers interstate edges or loop and branch conditions read."""
    names = OrderedSet(name for edge in sdfg.all_interstate_edges() for name in edge.data.used_arrays(sdfg.arrays))
    for region in sdfg.all_control_flow_regions():
        names |= OrderedSet(memlet.data for memlet in region.get_meta_read_memlets())
    return names


class TableCensus:
    """Per table: its fill tasklets per state, the sides touching it per state, and what disqualifies it."""

    __slots__ = ('sdfg', 'fills', 'sides', 'refused', 'host_read')

    def __init__(self, sdfg: SDFG) -> None:
        self.sdfg = sdfg
        self.fills: Dict[str, Dict[SDFGState, OrderedSet[nodes.Tasklet]]] = {}
        #: Per table and state: True for a device access, False for a host one.
        self.sides: Dict[str, Dict[SDFGState, OrderedSet[bool]]] = {}
        self.refused: OrderedSet[str] = OrderedSet()
        self.host_read = host_meta_reads(sdfg)

    def is_table(self, name: str) -> bool:
        desc = self.sdfg.arrays.get(name)
        return type(desc) is data.Array and desc.transient and not helpers.is_length1_array(name, self.sdfg)

    def visit(self, state: SDFGState, scopes: Dict, node: nodes.AccessNode) -> None:
        touched = self.sides.setdefault(node.data, {}).setdefault(state, OrderedSet())
        if helpers.enclosing_kernel(scopes, node):
            touched.add(True)
            # A kernel writing the table: the two copies could disagree.
            if any(not isinstance(e.src, nodes.EntryNode) and not e.data.is_empty() for e in state.in_edges(node)):
                self.refused.add(node.data)
            return
        for edge in state.out_edges(node):
            if isinstance(edge.dst, nodes.MapEntry) and helpers.has_GPU_schedule(edge.dst):
                touched.add(True)
            elif not edge.data.is_empty():
                touched.add(False)
                self.host_read.add(node.data)
        for edge in state.in_edges(node):
            if edge.data.is_empty():
                continue
            if not in_a_loop(state) and fills_from_scalars(self.sdfg, state, scopes, edge.src):
                self.fills.setdefault(node.data, {}).setdefault(state, OrderedSet()).add(edge.src)
                touched.add(False)
            else:
                self.refused.add(node.data)


def device_tables(sdfg: SDFG) -> Tables:
    """Tables to fill on the device: transient arrays longer than one element that some kernel reads.

    Every write is a top-level tasklet reading only scalars, all in one state outside every loop, and
    nothing else writes the table. A table the host also reads qualifies only if no state touches it
    from both sides, since a rename cannot split a state.
    """
    census = TableCensus(sdfg)
    for state in sdfg.states():
        scopes = state.scope_dict()
        for node in state.data_nodes():
            if census.is_table(node.data):
                census.visit(state, scopes, node)
    tables: Tables = {}
    for name, per_state in census.fills.items():
        sides = census.sides[name]
        if name in census.refused or len(per_state) != 1 or not any(True in on for on in sides.values()):
            continue
        (state, tasklets), = per_state.items()
        if name not in census.host_read:
            tables[name] = (state, tasklets, None)
        elif all(len(on) < 2 for on in sides.values()):
            tables[name] = (state, tasklets, [other for other, on in sides.items() if True in on])
    return tables


def fill_device_twin(sdfg: SDFG, state: SDFGState, tasklets: OrderedSet[nodes.Tasklet], name: str,
                     twin_states: List[SDFGState]) -> OrderedSet[nodes.Tasklet]:
    """Clone the fill of ``name`` onto its device twin, point ``twin_states`` at the twin, return the clones."""
    twin = helpers.gpu_name(name)
    desc = deepcopy(sdfg.arrays[name])
    desc.storage = dtypes.StorageType.GPU_Global
    sdfg.add_datadesc(twin, desc)
    for other in twin_states:
        CopyInsertion.rename_accesses(other, {name: twin})
    target = state.add_access(twin)
    clones: OrderedSet[nodes.Tasklet] = OrderedSet()
    for tasklet in tasklets:
        clone = deepcopy(tasklet)
        state.add_node(clone)
        for edge in state.in_edges(tasklet):
            state.add_edge(edge.src, edge.src_conn, clone, edge.dst_conn, deepcopy(edge.data))
        write = state.out_edges(tasklet)[0]
        memlet = deepcopy(write.data)
        memlet.data = twin
        state.add_edge(clone, write.src_conn, target, None, memlet)
        clones.add(clone)
    return clones
