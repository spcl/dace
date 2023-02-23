# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains inter-state transformations of an SDFG to run on the GPU. """

from dace import data, memlet, dtypes, registry, sdfg as sd, symbolic
from dace.sdfg import nodes, scope
from dace.sdfg import utils as sdutil
from dace.transformation import transformation, helpers as xfh
from dace.properties import Property, make_properties
from collections import defaultdict
from copy import deepcopy as dc
from typing import Dict, Optional

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
class GPUTransformSDFGCloudSC(transformation.MultiStateTransformation):
    """ Implements the GPUTransformSDFG transformation.

        Transforms a whole SDFG to run on the GPU:
        Steps of the full GPU transform
          0. Acquire metadata about SDFG and arrays
          1. Copy-in state from host to GPU
          2. Recursively schedule all maps for nested SDFGs and states.
          3. Copy-out state from GPU to host
          4. Re-apply simplification to get rid of extra states and
             transients

        What does not work currently:
          - tasklets are not touched
          - inter-state edges are not touched
          - transients are not touched yet
    """

    simplify = Property(desc='Reapply simplification after modifying graph', dtype=bool, default=True)

    exclude_copyin = Property(desc="Exclude these arrays from being copied into the device "
                              "(comma-separated)",
                              dtype=str,
                              default='')

    exclude_copyout = Property(desc="Exclude these arrays from being copied out of the device "
                               "(comma-separated)",
                               dtype=str,
                               default='')

    cloned_arrays = Property(desc="Exclude these arrays from being copied out of the device "
                               "(comma-separated)",
                               dtype=dict,
                               default={})

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

        cloned_arrays = {}

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

        for inodename, inode in set(input_nodes):
            if isinstance(inode, data.Scalar):  # Scalars can remain on host
                continue
            if inode.storage == dtypes.StorageType.GPU_Global:
                continue
            newdesc = inode.clone()
            newdesc.storage = dtypes.StorageType.GPU_Global
            newdesc.transient = True
            name = sdfg.add_datadesc('gpu_' + inodename, newdesc, find_new_name=True)
            cloned_arrays[inodename] = {'cpu': None, 'gpu': name}

        for onodename, onode in set(output_nodes):
            if onodename in cloned_arrays:
                continue
            if onode.storage == dtypes.StorageType.GPU_Global:
                continue
            newdesc = onode.clone()
            newdesc.storage = dtypes.StorageType.GPU_Global
            newdesc.transient = True
            name = sdfg.add_datadesc('gpu_' + onodename, newdesc, find_new_name=True)
            cloned_arrays[onodename] = {'cpu': None, 'gpu': name}

        # Replace nodes
        for state in sdfg.nodes():
            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode) and node.data in cloned_arrays):
                    node.data = cloned_arrays[node.data]['gpu']

        # Replace memlets
        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.data.data in cloned_arrays:
                    edge.data.data = cloned_arrays[edge.data.data]['gpu']

        #######################################################
        # Step 2: Create copy-in state
        excluded_copyin = self.exclude_copyin.split(',')

        copyin_state = sdfg.add_state(sdfg.label + '_copyin')
        sdfg.add_edge(copyin_state, start_state, sd.InterstateEdge())

        for nname, desc in dtypes.deduplicate(input_nodes):
            if nname in excluded_copyin or nname not in cloned_arrays:
                continue
            src_array = nodes.AccessNode(nname, debuginfo=desc.debuginfo)
            dst_array = nodes.AccessNode(cloned_arrays[nname]['gpu'], debuginfo=desc.debuginfo)
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
            src_array = nodes.AccessNode(cloned_arrays[nname]['gpu'], debuginfo=desc.debuginfo)
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

        gpu_scalars = {}
        nsdfgs = []
        changed = True
        # Iterates over Tasklets that not inside a GPU kernel. Such Tasklets must be moved inside a GPU kernel only
        # if they write to GPU memory. The check takes into account the fact that GPU kernels can read host-based
        # Scalars, but cannot write to them.
        while changed:
            changed = False
            for state in sdfg.states():
                for node in state.nodes():
                    # Handle NestedSDFGs later.
                    if isinstance(node, nodes.NestedSDFG):
                        if state.entry_node(node) is None and not scope.is_devicelevel_gpu_kernel(
                                state.parent, state, node):
                            nsdfgs.append((node, state))
                    elif isinstance(node, nodes.Tasklet):
                        if node in global_code_nodes[state]:
                            continue
                        if state.entry_node(node) is None and not scope.is_devicelevel_gpu_kernel(
                                state.parent, state, node):
                            scalars, scalar_output = _recursive_out_check(node, state, gpu_scalars)
                            sset, ssout = _recursive_in_check(node, state, gpu_scalars)
                            scalars = scalars.union(sset)
                            scalar_output = scalar_output and ssout
                            csdfg = state.parent
                            # If the tasklet is not adjacent only to scalars or it is in a GPU scope.
                            # The latter includes NestedSDFGs that have a GPU-Device schedule but are not in a GPU kernel.
                            if (not scalar_output
                                    or (csdfg.parent is not None
                                        and csdfg.parent_nsdfg_node.schedule == dtypes.ScheduleType.GPU_Default)):
                                global_code_nodes[state].append(node)
                                gpu_scalars.update({k: None for k in scalars})
                                changed = True

        # Apply GPUTransformSDFG recursively to NestedSDFGs.
        for node, state in nsdfgs:
            excl_copyin = set()
            for e in state.in_edges(node):
                src = state.memlet_path(e)[0].src
                if isinstance(src, nodes.AccessNode) and sdfg.arrays[src.data].storage in gpu_storage:
                    excl_copyin.add(e.dst_conn)
                    node.sdfg.arrays[e.dst_conn].storage = sdfg.arrays[src.data].storage
            excl_copyout = set()
            for e in state.out_edges(node):
                dst = state.memlet_path(e)[-1].dst
                if isinstance(dst, nodes.AccessNode) and sdfg.arrays[dst.data].storage in gpu_storage:
                    excl_copyout.add(e.src_conn)
                    node.sdfg.arrays[e.src_conn].storage = sdfg.arrays[dst.data].storage
            # TODO: Do we want to copy here the options from the top-level SDFG?
            node.sdfg.apply_transformations(
                GPUTransformSDFGCloudSC, {
                    'exclude_copyin': ','.join([str(n) for n in excl_copyin]),
                    'exclude_copyout': ','.join([str(n) for n in excl_copyout]),
                    'cloned_arrays': cloned_arrays | self.cloned_arrays
                })

        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.Tasklet):
                    # Ignore tasklets that are already in the GPU kernel
                    if state.entry_node(node) is None and not scope.is_devicelevel_gpu_kernel(state.parent, state, node):

                        # Find CPU tasklets that write to the GPU by checking all outgoing edges
                        for outgoing_conn in node.out_connectors:

                            for outgoing_edge in state.edges_by_connector(node, outgoing_conn):

                                data_desc = outgoing_edge.dst.desc(sdfg)
                                if data_desc.storage in gpu_storage:

                                    # Is there already a CPU array for this? If yes, we want to use it.
                                    array_name = outgoing_edge.dst.data
                                    # FIXME: avoid a redundant allocation - if the array is already allocated, we should use it
                                    # None indicates there was a clone before
                                    if array_name in self.cloned_arrays and self.cloned_arrays[array_name]['cpu'] is not None:
                                        new_data_name = self.cloned_arrays[array_name]['cpu']

                                        cpu_acc_node: Optional[nodes.AccessNode] = None
                                        for n in state.nodes():
                                            if isinstance(n, nodes.AccessNode) and n.data == new_data_name:
                                                cpu_acc_node = n
                                                break

                                        if cpu_acc_node is None:
                                            cpu_acc_node = nodes.AccessNode(new_data_name)
                                            state.add_node(cpu_acc_node)

                                    # There's only a transient GPU array, we need to create a CPU array
                                    else:
                                        new_data_desc = data_desc.clone()
                                        # We allocate it locally only for the purpose of touching some data on the CPU
                                        new_data_desc.transient = True
                                        new_data_desc.storage = dtypes.StorageType.CPU_Heap

                                        new_data_name = 'cpu_' + array_name
                                        cpu_acc_node = nodes.AccessNode(new_data_name)

                                        if new_data_name not in sdfg.arrays:
                                            sdfg.add_datadesc(new_data_name, new_data_desc)

                                            self.cloned_arrays[array_name] = {'cpu': new_data_name, 'gpu': array_name}

                                            #parent_nsdfg = sdfg.parent_nsdfg_node
                                            #if parent_nsdfg is not None and new_data_name not in parent_nsdfg.out_connectors:
                                            #    parent_nsdfg.add_out_connector(new_data_name)

                                            state.add_node(cpu_acc_node)

                                    gpu_acc_node = outgoing_edge.dst

                                    # create new edge from CPU access node to GPU access node to trigger a copy
                                    # We keep the shape of data access to be the same as the original one
                                    cpu_gpu_memlet = memlet.Memlet.simple(gpu_acc_node.data, outgoing_edge.data.subset)
                                    state.add_nedge(cpu_acc_node, gpu_acc_node, cpu_gpu_memlet)

                                    # now, replace the edge such that the CPU tasklet writes to the CPU array
                                    outgoing_edge._dst = cpu_acc_node
                                    outgoing_edge._data = memlet.Memlet.simple(new_data_name, outgoing_edge.data.subset)

        # Step 9: Simplify
        if not self.simplify:
            return

        sdfg.simplify()
