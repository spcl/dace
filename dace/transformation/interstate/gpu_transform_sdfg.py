# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains inter-state transformations of an SDFG to run on the GPU. """

from dace import data, memlet, dtypes, registry, sdfg as sd
from dace.sdfg import nodes, scope
from dace.sdfg import utils as sdutil
from dace.transformation import transformation, helpers as xfh
from dace.properties import Property, make_properties
from collections import defaultdict
from typing import Dict


@make_properties
class GPUTransformSDFG(transformation.MultiStateTransformation):
    """ Implements the GPUTransformSDFG transformation.

        Transforms a whole SDFG to run on the GPU:
        Steps of the full GPU transform
          0. Acquire metadata about SDFG and arrays
          1. Replace all non-transients with their GPU counterparts
          2. Copy-in state from host to GPU
          3. Copy-out state from GPU to host
          4. Re-store Default-top/CPU_Heap transients as GPU_Global
          5. Global tasklets are wrapped with a map of size 1
          6. Global Maps are re-scheduled to use the GPU
          7. Make data ready for interstate edges that use them
          8. Re-apply dataflow coarsening to get rid of extra states and
             transients
    """

    toplevel_trans = Property(desc="Make all GPU transients top-level", dtype=bool, default=True)

    register_trans = Property(desc="Make all transients inside GPU maps registers", dtype=bool, default=True)

    sequential_innermaps = Property(desc="Make all internal maps Sequential", dtype=bool, default=True)

    skip_scalar_tasklets = Property(desc="If True, does not transform tasklets "
                                    "that manipulate (Default-stored) scalars",
                                    dtype=bool,
                                    default=True)

    coarsen = Property(desc='Reapply dataflow coarsening after modifying graph', dtype=bool, default=True)

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

    @staticmethod
    def expressions():
        # Matches anything
        return [sd.SDFG('_')]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
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

    @staticmethod
    def match_to_str(graph, candidate):
        return graph.label

    def apply(self, sdfg: sd.SDFG):

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
        # Step 4: Modify transient data storage

        const_syms = xfh.constant_symbols(sdfg)

        for state in sdfg.nodes():
            sdict = state.scope_dict()
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.desc(sdfg).transient:
                    nodedesc = node.desc(sdfg)

                    # Special case: nodes that lead to dynamic map ranges must
                    # stay on host
                    if any(isinstance(state.memlet_path(e)[-1].dst, nodes.EntryNode) for e in state.out_edges(node)):
                        continue

                    gpu_storage = [
                        dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned
                    ]
                    if sdict[node] is None and nodedesc.storage not in gpu_storage:
                        # NOTE: the cloned arrays match too but it's the same
                        # storage so we don't care
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
        # Step 5: Change all top-level maps and library nodes to GPU schedule

        for state in sdfg.nodes():
            sdict = state.scope_dict()
            for node in state.nodes():
                if sdict[node] is None:
                    if isinstance(node, (nodes.LibraryNode, nodes.NestedSDFG)):
                        node.schedule = dtypes.ScheduleType.GPU_Default
                    elif isinstance(node, nodes.EntryNode):
                        node.schedule = dtypes.ScheduleType.GPU_Device
                elif self.sequential_innermaps:
                    if isinstance(node, (nodes.EntryNode, nodes.LibraryNode)):
                        node.schedule = dtypes.ScheduleType.Sequential
                    elif isinstance(node, nodes.NestedSDFG):
                        for nnode, _ in node.sdfg.all_nodes_recursive():
                            if isinstance(nnode, (nodes.EntryNode, nodes.LibraryNode)):
                                nnode.schedule = dtypes.ScheduleType.Sequential

        #######################################################
        # Step 6: Wrap free tasklets and nested SDFGs with a GPU map

        # Collect free tasklets
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.Tasklet):
                if (state.entry_node(node) is None
                        and not scope.is_devicelevel_gpu(state.parent, state, node, with_gpu_default=True)):
                    global_code_nodes[state].append(node)

        for state, gcodes in global_code_nodes.items():
            for gcode in gcodes:
                if gcode.label in self.exclude_tasklets.split(','):
                    continue
                # Create map and connectors
                me, mx = state.add_map(gcode.label + '_gmap', {gcode.label + '__gmapi': '0:1'},
                                       schedule=dtypes.ScheduleType.GPU_Device)
                # Store in/out edges in lists so that they don't get corrupted
                # when they are removed from the graph
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
                    state.add_edge(me, 'OUT_' + e.dst_conn, e.dst, e.dst_conn, e.data)
                for e in out_edges:
                    state.remove_edge(e)
                    state.add_edge(e.src, e.src_conn, mx, 'IN_' + e.src_conn, e.data)
                    state.add_edge(mx, 'OUT_' + e.src_conn, e.dst, e.dst_conn, e.data)

                # Map without inputs
                if len(in_edges) == 0:
                    state.add_nedge(me, gcode, memlet.Memlet())
        #######################################################
        # Step 7: Introduce copy-out if data used in outgoing interstate edges

        for state in list(sdfg.nodes()):
            arrays_used = set()
            for e in sdfg.out_edges(state):
                # Used arrays = intersection between symbols and cloned arrays
                arrays_used.update(set(e.data.free_symbols) & set(cloned_arrays.keys()))

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
                    desc = sdfg.arrays[nname]
                    src_array = nodes.AccessNode(cloned_arrays[nname], debuginfo=desc.debuginfo)
                    dst_array = nodes.AccessNode(nname, debuginfo=desc.debuginfo)
                    co_state.add_node(src_array)
                    co_state.add_node(dst_array)
                    co_state.add_nedge(src_array, dst_array,
                                       memlet.Memlet.from_array(dst_array.data, dst_array.desc(sdfg)))

        #######################################################
        # Step 8: Dataflow coarsening
        if not self.coarsen:
            return

        sdfg.coarsen_dataflow()
