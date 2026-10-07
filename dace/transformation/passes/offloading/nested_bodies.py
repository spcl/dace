# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A nested SDFG at a host level is a level of its own: its body places its data after the level around it."""

from copy import deepcopy
from collections.abc import Iterator

from ordered_set import OrderedSet

from dace import data, dtypes, Memlet
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg import nodes, SDFG, SDFGState

import dace.transformation.passes.offloading.offloading_helpers as helpers


def host_level_nested_sdfgs(
    state: SDFGState, host_maps: OrderedSet[nodes.MapEntry], entry: nodes.MapEntry | None = None
) -> Iterator[nodes.NestedSDFG]:
    """Nested SDFGs at ``state``'s top level or under host maps only: one below a kernel is device code."""
    for node in state.scope_children()[entry]:
        if isinstance(node, nodes.NestedSDFG):
            yield node
        elif isinstance(node, nodes.MapEntry) and node in host_maps:
            yield from host_level_nested_sdfgs(state, host_maps, node)


def prepare_body(sdfg: SDFG, state: SDFGState, nsdfg_node: nodes.NestedSDFG) -> None:
    """Stage the device scalars the body reads on the host, then bind its arrays to their outer storage."""
    stage_device_scalar_bindings(sdfg, state, nsdfg_node)
    inherit_binding_storage(sdfg, state, nsdfg_node)


def stage_device_scalar_bindings(sdfg: SDFG, state: SDFGState, nsdfg_node: nodes.NestedSDFG) -> None:
    """Copy a device element bound to a scalar connector the body reads on the host into a host scalar: a scalar
    connector names one element by reference, so a host read through a device binding is invalid."""
    body = nsdfg_node.sdfg
    for edge in state.in_edges(nsdfg_node):
        if (
            edge.data.is_empty()
            or edge.dst_conn not in body.arrays
            or not isinstance(body.arrays[edge.dst_conn], data.Scalar)
        ):
            continue
        if sdfg.arrays[edge.data.data].storage not in GPU_RESIDENT_STORAGES:
            continue
        if not helpers.is_read(body, edge.dst_conn, outside_kernels=True):
            continue
        host_scalar, _ = sdfg.add_scalar(
            f"{edge.dst_conn}_host",
            body.arrays[edge.dst_conn].dtype,
            transient=True,
            storage=dtypes.StorageType.Default,
            find_new_name=True,
        )
        staged = state.add_access(host_scalar)
        # The edge's own source, so the old one is not left isolated once this edge goes.
        source = edge.src if isinstance(edge.src, nodes.AccessNode) else state.add_read(edge.data.data)
        state.remove_edge(edge)
        state.add_edge(source, edge.src_conn, staged, None, deepcopy(edge.data))
        state.add_edge(
            staged, None, nsdfg_node, edge.dst_conn, Memlet.from_array(host_scalar, sdfg.arrays[host_scalar])
        )


def inherit_binding_storage(sdfg: SDFG, state: SDFGState, nsdfg_node: nodes.NestedSDFG) -> None:
    """Give each inner array bound to a connector the storage of the outer array it binds; a scalar connector is
    left alone, since the outer storage says nothing about where the body reads it."""
    for edge in state.all_edges(nsdfg_node):
        connector = edge.dst_conn if edge.dst is nsdfg_node else edge.src_conn
        if edge.data.is_empty() or connector is None or connector not in nsdfg_node.sdfg.arrays:
            continue
        inner = nsdfg_node.sdfg.arrays[connector]
        if not isinstance(inner, data.Scalar):
            inner.storage = sdfg.arrays[edge.data.data].storage
