# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
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

        # Step 9: Simplify
        if not self.simplify:
            return

        sdfg.simplify()
