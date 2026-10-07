# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-element containers: which form each takes across the host/device boundary, and where its copy happens."""

from copy import deepcopy

from ordered_set import OrderedSet

from dace import data, dtypes, Memlet
from dace.sdfg import nodes, SDFG, SDFGState
from dace.transformation.passes.length_one_array_scalar_conversion import (
    ConvertLengthOneArraysToScalars,
    ConvertScalarsToLengthOneArrays,
)

import dace.transformation.passes.offloading.offloading_helpers as helpers


def retype_single_elements(sdfg: SDFG, exceptions: OrderedSet[str]) -> OrderedSet[str]:
    """Device-written scalars become length-1 arrays (a kernel takes a scalar by value), the other length-1
    arrays scalars; ``exceptions`` are not asked again. Return the names asked for."""
    gpu_written = helpers.data_written_by_device_code(sdfg)
    to_arrays = OrderedSet(
        name
        for name, desc in sdfg.arrays.items()
        if isinstance(desc, data.Scalar) and name in gpu_written and name not in exceptions
    )
    # ``__return`` stays by reference: the caller reads the result back through it.
    to_scalars = OrderedSet(
        name
        for name in sdfg.arrays
        if helpers.is_length1_array(name, sdfg)
        and name not in gpu_written
        and name not in exceptions
        and not name.startswith("__return")
    )

    if to_arrays:
        ConvertScalarsToLengthOneArrays(recursive=True, preserve_abi=True, filter=to_arrays).apply_pass(sdfg, {})
        for name in to_arrays:  # allocated once, not in every iteration of a busy loop
            sdfg.arrays[name].lifetime = dtypes.AllocationLifetime.SDFG
    if to_scalars:
        before = OrderedSet(sdfg.arrays)
        ConvertLengthOneArraysToScalars(recursive=True, preserve_abi=True, filter=to_scalars).apply_pass(sdfg, {})
        # A staged scalar inherits its array's storage; no kernel writes it, so it lives on the host.
        for name in OrderedSet(sdfg.arrays) - before:
            if isinstance(sdfg.arrays[name], data.Scalar) and helpers.is_array_stored_on_GPU(sdfg, name):
                sdfg.arrays[name].storage = dtypes.StorageType.Default
    return to_scalars | to_arrays


def single_element_copies_into_map(sdfg: SDFG) -> None:
    """Move a single-element copy into the map that alone reads it: ``A -> s -> Map`` becomes ``A -> Map -> s``."""
    changes = [
        (state, access, map_entry)
        for state in sdfg.states()
        for map_entry in state.nodes()
        if isinstance(map_entry, nodes.MapEntry)
        for access in OrderedSet(state.predecessors(map_entry))
        if is_movable_copy(sdfg, state, access)
    ]
    for state, access, map_entry in changes:
        rewire_access_into_map(state, access, map_entry)


def is_movable_copy(sdfg: SDFG, state: SDFGState, access: nodes.Node) -> bool:
    """A single-element container copied from another container and read only through a pass-through connector of
    one map (another reader would be left inside the map's scope; a dynamic input is not passed-on data)."""
    return (
        isinstance(access, nodes.AccessNode)
        and state.in_degree(access) == 1
        and state.out_degree(access) == 1
        and isinstance(state.in_edges(access)[0].src, nodes.AccessNode)
        and (state.out_edges(access)[0].dst_conn or "").startswith("IN_")
        and (isinstance(sdfg.arrays[access.data], data.Scalar) or helpers.is_length1_array(access.data, sdfg))
    )


def rewire_access_into_map(state: SDFGState, access: nodes.AccessNode, map_entry: nodes.MapEntry) -> None:
    """Rewire ``B -> access -> map -> C`` into ``B -> map -> access -> C``."""
    source_edge = state.in_edges(access)[0]
    access_to_map = state.out_edges(access)[0]
    old_in = access_to_map.dst_conn
    old_out = "OUT_" + old_in[len("IN_") :]

    connector = map_entry.next_connector(access.data)
    in_conn, out_conn = f"IN_{connector}", f"OUT_{connector}"
    map_entry.add_in_connector(in_conn)
    map_entry.add_out_connector(out_conn)
    state.remove_edge(source_edge)
    outer, inner = deepcopy(source_edge.data), deepcopy(source_edge.data)
    clear_other_subset(outer, source_edge.src, map_entry)
    clear_other_subset(inner, map_entry, access)
    state.add_edge(source_edge.src, source_edge.src_conn, map_entry, in_conn, outer)
    state.add_edge(map_entry, out_conn, access, None, inner)

    # The access now feeds the map's readers directly.
    readers = [edge for edge in state.out_edges(map_entry) if edge.src_conn == old_out]
    map_entry.remove_in_connector(old_in)
    map_entry.remove_out_connector(old_out)
    state.remove_edge(access_to_map)
    for edge in readers:
        memlet = deepcopy(edge.data)
        clear_other_subset(memlet, access, edge.dst)
        state.add_edge(access, None, edge.dst, edge.dst_conn, memlet)
        state.remove_edge(edge)


def clear_other_subset(memlet: Memlet, src: nodes.Node, dst: nodes.Node) -> None:
    """Drop ``other_subset`` once the edge no longer runs between two containers."""
    if not isinstance(src, nodes.AccessNode) or not isinstance(dst, nodes.AccessNode):
        memlet.other_subset = None
