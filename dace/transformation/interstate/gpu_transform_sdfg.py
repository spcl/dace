# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains inter-state transformations of an SDFG to run on the GPU. """

from dace import data, memlet, dtypes, registry, sdfg as sd, symbolic
from dace.sdfg import nodes, scope
from dace.sdfg import utils as sdutil
from dace.transformation import transformation, helpers as xfh
from dace.properties import Property, make_properties
from collections import defaultdict
from copy import deepcopy as dc
from typing import Dict

gpu_storage = [dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned]


def _recursive_out_check(node, state, gpu_scalars):
    """
    Recursively checks if the outputs of a node are scalars and if they are/should be stored in GPU memory.
    """
    scalset = set()
    scalout = True
    sdfg = state.parent
    for e in state.out_edges(node):
        last_edge = state.memlet_path(e)[-1]
        if isinstance(last_edge.dst, nodes.AccessNode):
            desc = sdfg.arrays[last_edge.dst.data]
            if isinstance(desc, data.Scalar):
                if desc.storage in gpu_storage or last_edge.dst.data in gpu_scalars:
                    scalout = False
                scalset.add(last_edge.dst.data)
                sset, ssout = _recursive_out_check(last_edge.dst, state, gpu_scalars)
                scalset = scalset.union(sset)
                scalout = scalout and ssout
                continue
            if desc.storage not in gpu_storage and last_edge.data.num_elements() == 1:
                sset, ssout = _recursive_out_check(last_edge.dst, state, gpu_scalars)
                scalset = scalset.union(sset)
                scalout = scalout and ssout
                continue
            scalout = False
    return scalset, scalout


def _recursive_in_check(node, state, gpu_scalars):
    """
    Recursively checks if the inputs of a node are scalars and if they are/should be stored in GPU memory.
    """
    scalset = set()
    scalout = True
    sdfg = state.parent
    for e in state.in_edges(node):
        last_edge = state.memlet_path(e)[0]
        if isinstance(last_edge.src, nodes.AccessNode):
            desc = sdfg.arrays[last_edge.src.data]
            if isinstance(desc, data.Scalar):
                if desc.storage in gpu_storage or last_edge.src.data in gpu_scalars:
                    scalout = False
                scalset.add(last_edge.src.data)
                sset, ssout = _recursive_in_check(last_edge.src, state, gpu_scalars)
                scalset = scalset.union(sset)
                scalout = scalout and ssout
                continue
            if desc.storage not in gpu_storage and last_edge.data.num_elements() == 1:
                sset, ssout = _recursive_in_check(last_edge.src, state, gpu_scalars)
                scalset = scalset.union(sset)
                scalout = scalout and ssout
                continue
            scalout = False
    return scalset, scalout


def _codenode_condition(node):
    return isinstance(node, (nodes.LibraryNode, nodes.NestedSDFG)) and node.schedule == dtypes.ScheduleType.GPU_Default


@make_properties
class GPUTransformSDFG(transformation.MultiStateTransformation):
    """ Implements the GPUTransformSDFG transformation.

        Transforms a whole SDFG to run on the GPU:

            1. Acquire metadata about SDFG and arrays
            2. Replace all non-transients with their GPU counterparts
            3. Copy-in state from host to GPU
            4. Copy-out state from GPU to host
            5. Re-store Default-top/CPU_Heap transients as GPU_Global
            6. Global tasklets are wrapped with a map of size 1
            7. Global Maps are re-scheduled to use the GPU
            8. Make data ready for interstate edges that use them
            9. Re-apply simplification to get rid of extra states and transients
    """

    toplevel_trans = Property(desc="Make all GPU transients top-level", dtype=bool, default=True)

    register_trans = Property(desc="Make all transients inside GPU maps registers", dtype=bool, default=True)

    sequential_innermaps = Property(desc="Make all internal maps Sequential", dtype=bool, default=True)

    skip_scalar_tasklets = Property(desc="If True, does not transform tasklets "
                                    "that manipulate (Default-stored) scalars",
                                    dtype=bool,
                                    default=True)

    simplify = Property(desc='Reapply simplification after modifying graph', dtype=bool, default=True)

    exclude_copyin = Property(desc="Exclude these arrays from being copied into the device "
                              "(comma-separated)",
                              dtype=str,
                              default='')

    exclude_tasklets = Property(desc="Exclude these tasklets from being processed as CPU tasklets "
                                "(comma-separated)",
                                dtype=str,
                                default='')

    exclude_copyout = Property(desc="Exclude these arrays from being copied out of the device "
                               "(comma-separated)",
                               dtype=str,
                               default='')

    @staticmethod
    def annotates_memlets():
        # Skip memlet propagation for now
        return True

    @classmethod
    def expressions(cls):
        # Matches anything
        return [sd.SDFG('_')]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        for node, _ in sdfg.all_nodes_recursive():
            # Consume scopes are currently unsupported
            if isinstance(node, (nodes.ConsumeEntry, nodes.ConsumeExit)):
                return False

        for state in sdfg.nodes():
            schildren = state.scope_children()
            for node in schildren[None]:
                # If two top-level tasklets are connected with a code->code
                # memlet, they will transform into an invalid SDFG
                if (isinstance(node, nodes.CodeNode)
                        and any(isinstance(e.dst, nodes.CodeNode) for e in state.out_edges(node))):
                    return False
        return True

    def apply(self, _, sdfg: sd.SDFG):

        #######################################################
        # Step 0: SDFG metadata

        # Find all input and output data descriptors
        input_nodes = []
        output_nodes = []
        global_code_nodes: Dict[sd.SDFGState, nodes.Tasklet] = defaultdict(list)

        for state in sdfg.nodes():
            sdict = state.scope_dict()
            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode) and node.desc(sdfg).transient == False):
                    if (state.out_degree(node) > 0 and node.data not in input_nodes):
                        # Special case: nodes that lead to top-level dynamic
                        # map ranges must stay on host
                        for e in state.out_edges(node):
                            last_edge = state.memlet_path(e)[-1]
                            if (isinstance(last_edge.dst, nodes.EntryNode) and last_edge.dst_conn
                                    and not last_edge.dst_conn.startswith('IN_') and sdict[last_edge.dst] is None):
                                break
                        else:
                            input_nodes.append((node.data, node.desc(sdfg)))
                    if (state.in_degree(node) > 0 and node.data not in output_nodes):
                        output_nodes.append((node.data, node.desc(sdfg)))

            # Input nodes may also be nodes with WCR memlets and no identity
            for e in state.edges():
                if e.data.wcr is not None:
                    if (e.data.data not in input_nodes and sdfg.arrays[e.data.data].transient == False):
                        input_nodes.append((e.data.data, sdfg.arrays[e.data.data]))

        start_state = sdfg.start_state
        end_states = sdfg.sink_nodes()

        #######################################################
        # Step 1: Create cloned GPU arrays and replace originals

        cloned_arrays = {}
        for inodename, inode in set(input_nodes):
            if isinstance(inode, data.Scalar):  # Scalars can remain on host
                continue
            if inode.storage == dtypes.StorageType.GPU_Global:
                continue
            newdesc = inode.clone()
            newdesc.storage = dtypes.StorageType.GPU_Global
            newdesc.transient = True
            name = sdfg.add_datadesc('gpu_' + inodename, newdesc, find_new_name=True)
            cloned_arrays[inodename] = name

        for onodename, onode in set(output_nodes):
            if onodename in cloned_arrays:
                continue
            if onode.storage == dtypes.StorageType.GPU_Global:
                continue
            newdesc = onode.clone()
            newdesc.storage = dtypes.StorageType.GPU_Global
            newdesc.transient = True
            name = sdfg.add_datadesc('gpu_' + onodename, newdesc, find_new_name=True)
            cloned_arrays[onodename] = name

        # Replace nodes
        for state in sdfg.nodes():
            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode) and node.data in cloned_arrays):
                    node.data = cloned_arrays[node.data]

        # Replace memlets
        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.data.data in cloned_arrays:
                    edge.data.data = cloned_arrays[edge.data.data]

        #######################################################
        # Step 2: Create copy-in state
        excluded_copyin = self.exclude_copyin.split(',')

        copyin_state = sdfg.add_state(sdfg.label + '_copyin')
        sdfg.add_edge(copyin_state, start_state, sd.InterstateEdge())

        for nname, desc in dtypes.deduplicate(input_nodes):
            if nname in excluded_copyin or nname not in cloned_arrays:
                continue
            src_array = nodes.AccessNode(nname, debuginfo=desc.debuginfo)
            dst_array = nodes.AccessNode(cloned_arrays[nname], debuginfo=desc.debuginfo)
            copyin_state.add_node(src_array)
            copyin_state.add_node(dst_array)
            copyin_state.add_nedge(src_array, dst_array, memlet.Memlet.from_array(src_array.data, src_array.desc(sdfg)))

        #######################################################
        # Step 3: Create copy-out state
        excluded_copyout = self.exclude_copyout.split(',')

        copyout_state = sdfg.add_state(sdfg.label + '_copyout')
        for state in end_states:
            sdfg.add_edge(state, copyout_state, sd.InterstateEdge())

        for nname, desc in dtypes.deduplicate(output_nodes):
            if nname in excluded_copyout or nname not in cloned_arrays:
                continue
            src_array = nodes.AccessNode(cloned_arrays[nname], debuginfo=desc.debuginfo)
            dst_array = nodes.AccessNode(nname, debuginfo=desc.debuginfo)
            copyout_state.add_node(src_array)
            copyout_state.add_node(dst_array)
            copyout_state.add_nedge(src_array, dst_array, memlet.Memlet.from_array(dst_array.data,
                                                                                   dst_array.desc(sdfg)))

        #######################################################
        # Step 4: Change all top-level maps and library nodes to GPU schedule

        gpu_nodes = set()
        for state in sdfg.nodes():
            sdict = state.scope_dict()
            for node in state.nodes():
                if sdict[node] is None:
                    if isinstance(node, (nodes.LibraryNode, nodes.NestedSDFG)):
                        node.schedule = dtypes.ScheduleType.GPU_Default
                        gpu_nodes.add((state, node))
                    elif isinstance(node, nodes.EntryNode):
                        node.schedule = dtypes.ScheduleType.GPU_Device
                        gpu_nodes.add((state, node))
                elif self.sequential_innermaps:
                    if isinstance(node, (nodes.EntryNode, nodes.LibraryNode)):
                        node.schedule = dtypes.ScheduleType.Sequential
                    elif isinstance(node, nodes.NestedSDFG):
                        for nnode, _ in node.sdfg.all_nodes_recursive():
                            if isinstance(nnode, (nodes.EntryNode, nodes.LibraryNode)):
                                nnode.schedule = dtypes.ScheduleType.Sequential

        # NOTE: The outputs of LibraryNodes, NestedSDFGs and Map that have GPU schedule must be moved to GPU memory.
        # TODO: Also use GPU-shared and GPU-register memory when appropriate.
        for state, node in gpu_nodes:
            if isinstance(node, (nodes.LibraryNode, nodes.NestedSDFG)):
                for e in state.out_edges(node):
                    dst = state.memlet_path(e)[-1].dst
                    if isinstance(dst, nodes.AccessNode):
                        desc = sdfg.arrays[dst.data]
                        desc.storage = dtypes.StorageType.GPU_Global
            if isinstance(node, nodes.EntryNode):
                for e in state.out_edges(state.exit_node(node)):
                    dst = state.memlet_path(e)[-1].dst
                    if isinstance(dst, nodes.AccessNode):
                        desc = sdfg.arrays[dst.data]
                        desc.storage = dtypes.StorageType.GPU_Global

        #######################################################
        # Step 5: Collect free tasklets and check for scalars that have to be moved to the GPU

        gpu_scalars = {}
        # Iterates over Tasklets that not inside a GPU kernel. Such Tasklets must be moved inside a GPU kernel only
        # if they write to GPU memory. The check takes into account the fact that GPU kernels can read host-based
        # Scalars, but cannot write to them.
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.Tasklet):
                if state.entry_node(node) is None and not scope.is_devicelevel_gpu_kernel(state.parent, state, node):
                    scalars, scalar_output = _recursive_out_check(node, state, gpu_scalars)
                    sset, ssout = _recursive_in_check(node, state, gpu_scalars)
                    scalars = scalars.union(sset)
                    scalar_output = scalar_output and ssout
                    csdfg = state.parent
                    # If the tasklet is not adjacent only to scalars or it is in a GPU scope.
                    # The latter includes NestedSDFGs that have a GPU-Device schedule but are not in a GPU kernel.
                    if (not scalar_output or (csdfg.parent is not None
                                              and csdfg.parent_nsdfg_node.schedule == dtypes.ScheduleType.GPU_Default)):
                        global_code_nodes[state].append(node)
                        gpu_scalars.update({k: None for k in scalars})

        # NOTE: We execute the above algorithm a second time to catch any potential new GPU-scalars.
        # TODO: Should we run this recursively until there are no new GPU-scalars?
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.Tasklet):
                if node in global_code_nodes[state]:
                    continue
                if state.entry_node(node) is None and not scope.is_devicelevel_gpu_kernel(state.parent, state, node):
                    scalars, scalar_output = _recursive_out_check(node, state, gpu_scalars)
                    sset, ssout = _recursive_in_check(node, state, gpu_scalars)
                    scalars = scalars.union(sset)
                    scalar_output = scalar_output and ssout
                    csdfg = state.parent
                    # If the tasklet is not adjacent only to scalars or it is in a GPU scope.
                    # The latter includes NestedSDFGs that have a GPU-Device schedule but are not in a GPU kernel.
                    if (not scalar_output or (csdfg.parent is not None
                                              and csdfg.parent_nsdfg_node.schedule == dtypes.ScheduleType.GPU_Default)):
                        global_code_nodes[state].append(node)
                        gpu_scalars.update({k: None for k in scalars})

        #######################################################
        # Step 6: Modify transient data storage

        const_syms = xfh.constant_symbols(sdfg)

        for state in sdfg.nodes():
            sdict = state.scope_dict()
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.desc(sdfg).transient:
                    nodedesc = node.desc(sdfg)

                    # Special case: nodes that lead to dynamic map ranges must stay on host
                    if any(isinstance(state.memlet_path(e)[-1].dst, nodes.EntryNode) for e in state.out_edges(node)):
                        continue

                    if sdict[node] is None and nodedesc.storage not in gpu_storage:

                        # Ensure that scalars not already GPU-marked are actually used in a GPU scope.
                        if isinstance(nodedesc, data.Scalar) and not node.data in gpu_scalars:
                            used_in_gpu_scope = False
                            for e in state.in_edges(node):
                                if _codenode_condition(state.memlet_path(e)[0].src):
                                    used_in_gpu_scope = True
                                    break
                            if not used_in_gpu_scope:
                                for e in state.out_edges(node):
                                    if _codenode_condition(state.memlet_path(e)[-1].dst):
                                        used_in_gpu_scope = True
                                        break
                            if not used_in_gpu_scope:
                                continue
                            for e in state.all_edges(node):
                                for node in (e.src, e.dst):
                                    if isinstance(node, nodes.Tasklet):
                                        if (state.entry_node(node) is None and not scope.is_devicelevel_gpu(
                                                state.parent, state, node, with_gpu_default=True)):
                                            global_code_nodes[state].append(node)

                        # NOTE: the cloned arrays match too but it's the same storage so we don't care
                        nodedesc.storage = dtypes.StorageType.GPU_Global

                        # Try to move allocation/deallocation out of loops
                        dsyms = set(map(str, nodedesc.free_symbols))
                        if (self.toplevel_trans and not isinstance(nodedesc, (data.Stream, data.View))
                                and len(dsyms - const_syms) == 0):
                            nodedesc.lifetime = dtypes.AllocationLifetime.SDFG
                    elif nodedesc.storage not in gpu_storage:
                        # Make internal transients registers
                        if self.register_trans:
                            nodedesc.storage = dtypes.StorageType.Register

        #######################################################
        # Step 7: Wrap free tasklets and nested SDFGs with a GPU map

        for state, gcodes in global_code_nodes.items():
            for gcode in gcodes:
                if gcode.label in self.exclude_tasklets.split(','):
                    continue
                # Create map and connectors
                me, mx = state.add_map(gcode.label + '_gmap', {gcode.label + '__gmapi': '0:1'},
                                       schedule=dtypes.ScheduleType.GPU_Device)
                # Store in/out edges in lists so that they don't get corrupted when they are removed from the graph.
                in_edges = list(state.in_edges(gcode))
                out_edges = list(state.out_edges(gcode))
                me.in_connectors = {('IN_' + e.dst_conn): None for e in in_edges}
                me.out_connectors = {('OUT_' + e.dst_conn): None for e in in_edges}
                mx.in_connectors = {('IN_' + e.src_conn): None for e in out_edges}
                mx.out_connectors = {('OUT_' + e.src_conn): None for e in out_edges}

                # Create memlets through map
                for e in in_edges:
                    state.remove_edge(e)
                    state.add_edge(e.src, e.src_conn, me, 'IN_' + e.dst_conn, e.data)
                    state.add_edge(me, 'OUT_' + e.dst_conn, e.dst, e.dst_conn, dc(e.data))
                for e in out_edges:
                    state.remove_edge(e)
                    state.add_edge(e.src, e.src_conn, mx, 'IN_' + e.src_conn, e.data)
                    state.add_edge(mx, 'OUT_' + e.src_conn, e.dst, e.dst_conn, dc(e.data))

                # Map without inputs
                if len(in_edges) == 0:
                    state.add_nedge(me, gcode, memlet.Memlet())

        #######################################################
        # Step 8: Introduce copy-out if data used in outgoing interstate edges

        cloned_data = set(cloned_arrays.keys()).union(gpu_scalars.keys())

        for state in list(sdfg.nodes()):
            arrays_used = set()
            for e in sdfg.out_edges(state):
                # Used arrays = intersection between symbols and cloned data
                arrays_used.update(set(e.data.free_symbols) & cloned_data)

            # Create a state and copy out used arrays
            if len(arrays_used) > 0:

                co_state = sdfg.add_state(state.label + '_icopyout')

                # Reconnect outgoing edges to after interim copyout state
                for e in sdfg.out_edges(state):
                    sdutil.change_edge_src(sdfg, state, co_state)
                # Add unconditional edge to interim state
                sdfg.add_edge(state, co_state, sd.InterstateEdge())

                # Add copy-out nodes
                for nname in arrays_used:

                    # Handle GPU scalars
                    if nname in gpu_scalars:
                        hostname = gpu_scalars[nname]
                        if not hostname:
                            desc = sdfg.arrays[nname].clone()
                            desc.storage = dtypes.StorageType.CPU_Heap
                            desc.transient = True
                            hostname = sdfg.add_datadesc('host_' + nname, desc, find_new_name=True)
                            gpu_scalars[nname] = hostname
                        else:
                            desc = sdfg.arrays[hostname]
                        devicename = nname
                    else:
                        desc = sdfg.arrays[nname]
                        hostname = nname
                        devicename = cloned_arrays[nname]

                    src_array = nodes.AccessNode(devicename, debuginfo=desc.debuginfo)
                    dst_array = nodes.AccessNode(hostname, debuginfo=desc.debuginfo)
                    co_state.add_node(src_array)
                    co_state.add_node(dst_array)
                    co_state.add_nedge(src_array, dst_array,
                                       memlet.Memlet.from_array(dst_array.data, dst_array.desc(sdfg)))
                    for e in sdfg.out_edges(co_state):
                        e.data.replace(devicename, hostname, False)

        # Step 9: Simplify
        if not self.simplify:
            return

        sdfg.simplify()
