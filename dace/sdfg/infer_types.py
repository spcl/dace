# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import data, dtypes
from dace.codegen.tools import type_inference
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import nodes
from dace.sdfg.utils import dfs_topological_sort
from typing import Dict, List

#############################################################################
# Connector type inference


def infer_connector_types(sdfg: SDFG):
    """ 
    Infers connector types throughout an SDFG and its nested SDFGs in-place.
    :param sdfg: The SDFG to infer.
    """
    # Loop over states, and in a topological sort over each state's nodes
    for state in sdfg.nodes():
        for node in dfs_topological_sort(state):
            # Try to infer input connector type from node type or previous edges
            for e in state.in_edges(node):
                cname = e.dst_conn
                if cname is None:
                    continue
                scalar = (e.data.subset and e.data.subset.num_elements() == 1)
                if e.data.data is not None:
                    allocated_as_scalar = (sdfg.arrays[e.data.data].storage is
                                           not dtypes.StorageType.GPU_Global)
                else:
                    allocated_as_scalar = True

                if node.in_connectors[cname].type is None:
                    # If nested SDFG, try to use internal array type
                    if isinstance(node, nodes.NestedSDFG):
                        scalar = (isinstance(node.sdfg.arrays[cname],
                                             data.Scalar)
                                  and allocated_as_scalar)
                        dtype = node.sdfg.arrays[cname].dtype
                        ctype = (dtype if scalar else dtypes.pointer(dtype))
                    elif e.data.data is not None:  # Obtain type from memlet
                        scalar |= isinstance(sdfg.arrays[e.data.data],
                                             data.Scalar)
                        if isinstance(node, nodes.LibraryNode):
                            scalar &= allocated_as_scalar
                        dtype = sdfg.arrays[e.data.data].dtype
                        ctype = (dtype if scalar else dtypes.pointer(dtype))
                    else:  # Code->Code
                        src_edge = state.memlet_path(e)[0]
                        sconn = src_edge.src.out_connectors[src_edge.src_conn]
                        if sconn.type is None:
                            raise TypeError('Ambiguous or uninferable type in'
                                            ' connector "%s" of node "%s"' %
                                            (sconn, src_edge.src))
                        ctype = sconn
                    node.in_connectors[cname] = ctype

            # Try to infer outputs from output edges
            for e in state.out_edges(node):
                cname = e.src_conn
                if cname is None:
                    continue
                scalar = (e.data.subset and e.data.subset.num_elements() == 1
                          and (not e.data.dynamic or
                               (e.data.dynamic and e.data.wcr is not None)))
                if e.data.data is not None:
                    allocated_as_scalar = (sdfg.arrays[e.data.data].storage is
                                           not dtypes.StorageType.GPU_Global)
                else:
                    allocated_as_scalar = True
                    
                if node.out_connectors[cname].type is None:
                    # If nested SDFG, try to use internal array type
                    if isinstance(node, nodes.NestedSDFG):
                        scalar = (isinstance(node.sdfg.arrays[cname],
                                             data.Scalar)
                                  and allocated_as_scalar)
                        dtype = node.sdfg.arrays[cname].dtype
                        ctype = (dtype if scalar else dtypes.pointer(dtype))
                    elif e.data.data is not None:  # Obtain type from memlet
                        scalar |= isinstance(sdfg.arrays[e.data.data],
                                             data.Scalar)
                        if isinstance(node, nodes.LibraryNode):
                            scalar &= allocated_as_scalar
                        dtype = sdfg.arrays[e.data.data].dtype
                        ctype = (dtype if scalar else dtypes.pointer(dtype))
                    else:
                        continue
                    node.out_connectors[cname] = ctype

            # Let the node infer other output types on its own
            node.infer_connector_types(sdfg, state)

            # If there are any remaining uninferable connectors, fail
            for e in state.out_edges(node):
                cname = e.src_conn
                if cname and node.out_connectors[cname].type is None:
                    raise TypeError('Ambiguous or uninferable type in'
                                    ' connector "%s" of node "%s"' %
                                    (cname, node))


#############################################################################
# Default schedule and storage type inference


def set_default_schedule_and_storage_types(
    sdfg: SDFG, toplevel_schedule: dtypes.ScheduleType):
    """ 
    Sets default storage and schedule types throughout SDFG in-place.
    Replaces `ScheduleType.Default` and `StorageType.Default`
    with the corresponding types according to the parent scope's schedule. 
    
    The defaults for storage types are determined by the
    ``dtypes.SCOPEDEFAULT_STORAGE`` dictionary (for example, a GPU device 
    schedule, by default, will allocate containers on the shared memory); and
    similarly for schedules by ``dtypes.SCOPEDEFAULT_SCHEDULE`` (e.g., a map
    nested in a CPU multi-core map will by default run within a single thread).

    :param sdfg: The SDFG to infer.
    :param toplevel_schedule: The default top-level schedule for "global" nodes
                              (without parent scope nodes).
    """
    _set_default_schedule_types(sdfg, toplevel_schedule)
    _set_default_storage_types(sdfg, toplevel_schedule)


def _scopes_with_tbmaps(state: SDFGState, scopes: List[nodes.EntryNode]):
    """ Returns a set of scopes where a thread-block (or dynamic thread-block)
        sub-scopes exist. Used, e.g., to modify storage defaults. """
    scopes_with_tbmaps = set()
    for scope_entry in scopes:
        subgraph = state.scope_subgraph(scope_entry)
        has_tb_map = False
        # Append thread-block maps from subgraph and nested SDFGs
        for node in subgraph.nodes():
            if isinstance(node, nodes.EntryNode) and node.schedule in (
                    dtypes.ScheduleType.GPU_ThreadBlock,
                    dtypes.ScheduleType.GPU_ThreadBlock_Dynamic):
                has_tb_map = True
                break
            elif isinstance(node, nodes.NestedSDFG):
                for n in node.sdfg.all_nodes_recursive():
                    if isinstance(node, nodes.EntryNode) and node.schedule in (
                            dtypes.ScheduleType.GPU_ThreadBlock,
                            dtypes.ScheduleType.GPU_ThreadBlock_Dynamic):
                        has_tb_map = True
                        break
                if has_tb_map:
                    break
        if has_tb_map:
            scopes_with_tbmaps.add(scope_entry)
    return scopes_with_tbmaps


def _set_default_schedule_in_scope(parent_node: nodes.Node,
                                   parent_schedule: dtypes.ScheduleType,
                                   reverse_scope_dict: Dict[nodes.Node,
                                                            List[nodes.Node]],
                                   use_parent_schedule: bool = False):
    for node in reverse_scope_dict[parent_node]:
        if use_parent_schedule:
            child_schedule = parent_schedule
            if parent_schedule in (dtypes.ScheduleType.Default,
                                   dtypes.ScheduleType.GPU_Default):
                child_schedule = dtypes.SCOPEDEFAULT_SCHEDULE[parent_schedule]
        else:
            child_schedule = dtypes.SCOPEDEFAULT_SCHEDULE[parent_schedule]
        # Set default schedule type
        if isinstance(node, nodes.MapEntry):
            if node.map.schedule is dtypes.ScheduleType.Default:
                node.map.schedule = child_schedule
            # Also traverse children (recursively)
            _set_default_schedule_in_scope(node, node.map.schedule,
                                           reverse_scope_dict)
        elif isinstance(node, nodes.ConsumeEntry):
            if node.consume.schedule is dtypes.ScheduleType.Default:
                node.consume.schedule = child_schedule

            # Also traverse children (recursively)
            _set_default_schedule_in_scope(node, node.consume.schedule,
                                           reverse_scope_dict)
        elif isinstance(node, nodes.NestedSDFG):
            # Nested SDFGs retain same schedule as their parent scope
            if node.schedule is dtypes.ScheduleType.Default:
                node.schedule = parent_schedule
            _set_default_schedule_types(node.sdfg, node.schedule)
        elif getattr(node, 'schedule', False):
            if node.schedule is dtypes.ScheduleType.Default:
                node.schedule = (
                    child_schedule if isinstance(node, nodes.EntryNode)
                    or parent_schedule is None else parent_schedule)


def _set_default_schedule_types(sdfg: SDFG,
                                toplevel_schedule: dtypes.ScheduleType,
                                use_parent_schedule: bool = False):
    for state in sdfg.nodes():
        reverse_scope_dict = state.scope_children()

        # Start with top-level nodes and call recursively
        _set_default_schedule_in_scope(None, toplevel_schedule,
                                       reverse_scope_dict, use_parent_schedule)


def _set_default_storage_types(sdfg: SDFG,
                               toplevel_schedule: dtypes.ScheduleType):
    for state in sdfg.nodes():
        scope_dict = state.scope_dict()
        scopes_with_tbmaps = _scopes_with_tbmaps(state, [
            n for n in state.nodes() if isinstance(n, nodes.MapEntry)
            and n.schedule in [dtypes.ScheduleType.GPU_Device]
        ])

        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode):
                continue
            desc = node.desc(sdfg)
            # Only set transients if nested
            if ((desc.transient or sdfg.parent_sdfg is None)
                    and desc.storage is dtypes.StorageType.Default):
                # Special cases
                parent_node = scope_dict[node]
                if parent_node is None:
                    parent_schedule = toplevel_schedule
                else:
                    parent_schedule = parent_node.map.schedule
                    # Skip sequential maps to determine storage
                    while parent_schedule is dtypes.ScheduleType.Sequential:
                        parent_node = scope_dict[parent_node]
                        if parent_node is None:
                            parent_schedule = toplevel_schedule
                            break
                        parent_schedule = parent_node.map.schedule
                # Determine default GPU schedule based on existence of
                # thread-block maps
                if parent_schedule is dtypes.ScheduleType.GPU_Device:
                    if parent_node not in scopes_with_tbmaps:
                        parent_schedule = dtypes.ScheduleType.GPU_ThreadBlock
                # End of special cases

                # Set default storage type
                desc.storage = dtypes.SCOPEDEFAULT_STORAGE[parent_schedule]

    # Take care of remaining arrays/scalars, e.g., code->code edges
    for desc in sdfg.arrays.values():
        if desc.storage is dtypes.StorageType.Default:
            desc.storage = dtypes.StorageType.Register

    for state in sdfg.nodes():
        # Loop again after all default storages have been set to set nested
        # SDFGs
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            for name, desc in node.sdfg.arrays.items():
                if (not desc.transient
                        and desc.storage is dtypes.StorageType.Default):
                    # Find connector and ensure storage types match
                    for e in state.in_edges(node):
                        if e.dst_conn == name:
                            desc.storage = sdfg.arrays[e.data.data].storage
                            break
                    for e in state.out_edges(node):
                        if e.src_conn == name:
                            desc.storage = sdfg.arrays[e.data.data].storage
                            break
            _set_default_storage_types(node.sdfg, node.schedule)
