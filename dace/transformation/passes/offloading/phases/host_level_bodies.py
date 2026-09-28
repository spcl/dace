# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A nested SDFG at a host level is a level of its own: its body places its data after the level around it."""
from copy import deepcopy
from typing import Iterator, Optional

from ordered_set import OrderedSet

from dace import data, dtypes, Memlet
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg import nodes, SDFG
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.state import SDFGState


def host_level_nested_sdfgs(state: SDFGState,
                            host_maps: OrderedSet[nodes.MapEntry],
                            entry: Optional[nodes.MapEntry] = None) -> Iterator[nodes.NestedSDFG]:
    """Nested SDFGs at ``state``'s top level or under host maps only: one below a kernel is device code."""
    for node in state.scope_children().get(entry, ()):
        if isinstance(node, nodes.NestedSDFG):
            yield node
        elif isinstance(node, nodes.MapEntry) and node in host_maps:
            yield from host_level_nested_sdfgs(state, host_maps, node)


def prepare_body(sdfg: SDFG, state: SDFGState, nsdfg_node: nodes.NestedSDFG) -> None:
    """Stage the device scalars the body reads on the host, then bind its arrays to their outer storage."""
    stage_device_scalar_bindings(sdfg, state, nsdfg_node)
    inherit_binding_storage(sdfg, state, nsdfg_node)


def read_outside_a_kernel(sdfg: SDFG, name: str) -> bool:
    """``name`` is read somewhere in ``sdfg`` that no device schedule covers, or by an interstate edge."""
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.states():
            for node in state.data_nodes():
                if node.data == name and state.out_degree(node) > 0 and not is_devicelevel_gpu(nested, state, node):
                    return True
        if any(name in edge.data.used_arrays(nested.arrays) for edge in nested.all_interstate_edges()):
            return True
    return False


def stage_device_scalar_bindings(sdfg: SDFG, state: SDFGState, nsdfg_node: nodes.NestedSDFG) -> None:
    """Copy a device element bound to a scalar connector the body reads on the host into a host scalar.

    A scalar connector names one element by reference, so a host read of it through a device binding is
    invalid (npbench azimint_hist reads ``bin_edges[i]`` in a host subtraction).
    """
    body = nsdfg_node.sdfg
    for edge in state.in_edges(nsdfg_node):
        if edge.data.is_empty() or edge.dst_conn not in body.arrays or not isinstance(
                body.arrays[edge.dst_conn], data.Scalar):
            continue
        if sdfg.arrays[edge.data.data].storage not in GPU_RESIDENT_STORAGES:
            continue
        if not read_outside_a_kernel(body, edge.dst_conn):
            continue
        host_name, _ = sdfg.add_scalar(f"{edge.dst_conn}_host",
                                       body.arrays[edge.dst_conn].dtype,
                                       transient=True,
                                       storage=dtypes.StorageType.Default,
                                       find_new_name=True)
        staged = state.add_access(host_name)
        # The edge's own source, so the old one is not left isolated once this edge goes.
        source = edge.src if isinstance(edge.src, nodes.AccessNode) else state.add_read(edge.data.data)
        state.remove_edge(edge)
        state.add_edge(source, edge.src_conn, staged, None, deepcopy(edge.data))
        state.add_edge(staged, None, nsdfg_node, edge.dst_conn, Memlet.from_array(host_name, sdfg.arrays[host_name]))


def inherit_binding_storage(sdfg: SDFG, state: SDFGState, nsdfg_node: nodes.NestedSDFG) -> None:
    """Give each inner array bound to a connector the storage of the outer array it binds.

    Arrays only: a scalar connector names one element by reference, so the outer storage says nothing about
    where the body may read it.
    """
    for edge in state.all_edges(nsdfg_node):
        connector = edge.dst_conn if edge.dst is nsdfg_node else edge.src_conn
        if edge.data.is_empty() or connector is None or connector not in nsdfg_node.sdfg.arrays:
            continue
        inner = nsdfg_node.sdfg.arrays[connector]
        if not isinstance(inner, data.Scalar):
            inner.storage = sdfg.arrays[edge.data.data].storage
