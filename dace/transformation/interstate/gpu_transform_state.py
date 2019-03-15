""" Contains inter-state transformations of an SDFG to run on the GPU. """

import copy
import itertools

from dace import data, memlet, types, sdfg as sd, subsets as sbs, symbolic
from dace.config import Config
from dace.graph import nodes, nxutil, edges as ed
from dace.transformation import pattern_matching, optimizer
from dace.properties import Property, make_properties

from dace.transformation.dataflow import RedundantArray
from dace.transformation.interstate import StateFusion


@make_properties
class GPUTransformState(pattern_matching.Transformation):
    """ Implements the GPUTransformState transformation.

        Transforms a whole SDFG to run on the GPU:
        Steps of the full GPU transform
          0. Acquire metadata about SDFG and arrays
          1. Replace all non-transients with their GPU counterparts
          2. Copy-in state from host to GPU
          3. Copy-out state from GPU to host
          4. Re-store Default-top/CPU_Heap transients as GPU_Global
          5. Global tasklets are wrapped with a map of size 1
          6. Global Maps are re-scheduled to use the GPU
          7. Re-apply strict transformations to get rid of extra states and 
             transients
    """

    toplevel_trans = Property(
        desc="Make all GPU transients top-level", dtype=bool, default=True)
    register_trans = Property(
        desc="Make all transients inside GPU maps registers",
        dtype=bool,
        default=True)
    sequential_innermaps = Property(
        desc="Make all internal maps Sequential", dtype=bool, default=True)
    strict_transform = Property(
        desc='Reapply strict transformations after modifying graph',
        dtype=bool,
        default=True)

    @staticmethod
    def annotates_memlets():
        # Skip memlet propagation for now
        return True

    @staticmethod
    def expressions():
        # Matches anything
        return [sd.SDFG('_')]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        return graph.label

    def modifies_graph(self):
        return True

    def apply(self, sdfg: sd.SDFG):

        #######################################################
        # Step 0: SDFG metadata

        # Find all input and output data descriptors
        input_nodes = []
        output_nodes = []
        global_code_nodes = [[] for _ in sdfg.nodes()]

        for i, state in enumerate(sdfg.nodes()):
            sdict = state.scope_dict()
            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode)
                        and node.desc(sdfg).transient == False):
                    if (state.out_degree(node) > 0
                            and node.data not in input_nodes):
                        input_nodes.append((node.data, node.desc(sdfg)))
                    if (state.in_degree(node) > 0
                            and node.data not in output_nodes):
                        output_nodes.append((node.data, node.desc(sdfg)))
                elif isinstance(node, nodes.CodeNode) and sdict[node] is None:
                    if not isinstance(node, nodes.EmptyTasklet):
                        global_code_nodes[i].append(node)

            # Input nodes may also be nodes with WCR memlets and no identity
            for e in state.edges():
                if e.data.wcr is not None and e.data.wcr_identity is None:
                    if (e.data.data not in input_nodes
                            and sdfg.arrays[e.data.data].transient == False):
                        input_nodes.append(e.data.data)

        start_state = sdfg.start_state
        end_states = sdfg.sink_nodes()

        #######################################################
        # Step 1: Create cloned GPU arrays and replace originals

        cloned_arrays = {}
        for inodename, inode in input_nodes:
            newdesc = inode.clone()
            newdesc.storage = types.StorageType.GPU_Global
            newdesc.transient = True
            sdfg.add_datadesc('gpu_' + inodename, newdesc)
            cloned_arrays[inodename] = 'gpu_' + inodename

        for onodename, onode in output_nodes:
            if onodename in cloned_arrays:
                continue
            newdesc = onode.clone()
            newdesc.storage = types.StorageType.GPU_Global
            newdesc.transient = True
            sdfg.add_datadesc('gpu_' + onodename, newdesc)
            cloned_arrays[onodename] = 'gpu_' + onodename

        # Replace nodes
        for state in sdfg.nodes():
            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode)
                        and node.data in cloned_arrays):
                    node.data = cloned_arrays[node.data]

        # Replace memlets
        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.data.data in cloned_arrays:
                    edge.data.data = cloned_arrays[edge.data.data]

        #######################################################
        # Step 2: Create copy-in state

        copyin_state = sdfg.add_state(sdfg.label + '_copyin')
        sdfg.add_edge(copyin_state, start_state, ed.InterstateEdge())

        for nname, desc in input_nodes:
            src_array = nodes.AccessNode(nname, debuginfo=desc.debuginfo)
            dst_array = nodes.AccessNode(
                cloned_arrays[nname], debuginfo=desc.debuginfo)
            copyin_state.add_node(src_array)
            copyin_state.add_node(dst_array)
            copyin_state.add_nedge(
                src_array, dst_array,
                memlet.Memlet.from_array(src_array.data, src_array.desc(sdfg)))

        #######################################################
        # Step 3: Create copy-out state

        copyout_state = sdfg.add_state(sdfg.label + '_copyout')
        for state in end_states:
            sdfg.add_edge(state, copyout_state, ed.InterstateEdge())

        for nname, desc in output_nodes:
            src_array = nodes.AccessNode(
                cloned_arrays[nname], debuginfo=desc.debuginfo)
            dst_array = nodes.AccessNode(nname, debuginfo=desc.debuginfo)
            copyout_state.add_node(src_array)
            copyout_state.add_node(dst_array)
            copyout_state.add_nedge(
                src_array, dst_array,
                memlet.Memlet.from_array(dst_array.data, dst_array.desc(sdfg)))

        #######################################################
        # Step 4: Modify transient data storage

        for state in sdfg.nodes():
            sdict = state.scope_dict()
            for node in state.nodes():
                if isinstance(node,
                              nodes.AccessNode) and node.desc(sdfg).transient:
                    nodedesc = node.desc(sdfg)
                    if sdict[node] is None:
                        # NOTE: the cloned arrays match too but it's the same
                        # storage so we don't care
                        nodedesc.storage = types.StorageType.GPU_Global

                        # Try to move allocation/deallocation out of loops
                        if self.toplevel_trans:
                            nodedesc.toplevel = True
                    else:
                        # Make internal transients registers
                        if self.register_trans:
                            nodedesc.storage = types.StorageType.Register

        #######################################################
        # Step 5: Wrap free tasklets and nested SDFGs with a GPU map

        for state, gcodes in zip(sdfg.nodes(), global_code_nodes):
            for gcode in gcodes:
                # Create map and connectors
                me, mx = state.add_map(
                    gcode.label + '_gmap', {gcode.label + '__gmapi': '0:1'},
                    schedule=types.ScheduleType.GPU_Device)
                # Store in/out edges in lists so that they don't get corrupted
                # when they are removed from the graph
                in_edges = list(state.in_edges(gcode))
                out_edges = list(state.out_edges(gcode))
                me.in_connectors = set('IN_' + e.dst_conn for e in in_edges)
                me.out_connectors = set('OUT_' + e.dst_conn for e in in_edges)
                mx.in_connectors = set('IN_' + e.src_conn for e in out_edges)
                mx.out_connectors = set('OUT_' + e.src_conn for e in out_edges)

                # Create memlets through map
                for e in in_edges:
                    state.remove_edge(e)
                    state.add_edge(e.src, e.src_conn, me, 'IN_' + e.dst_conn,
                                   e.data)
                    state.add_edge(me, 'OUT_' + e.dst_conn, e.dst, e.dst_conn,
                                   e.data)
                for e in out_edges:
                    state.remove_edge(e)
                    state.add_edge(e.src, e.src_conn, mx, 'IN_' + e.src_conn,
                                   e.data)
                    state.add_edge(mx, 'OUT_' + e.src_conn, e.dst, e.dst_conn,
                                   e.data)

                # Map without inputs
                if len(in_edges) == 0:
                    state.add_nedge(me, gcode, memlet.EmptyMemlet())
        #######################################################
        # Step 6: Change all top-level maps to GPU maps

        for i, state in enumerate(sdfg.nodes()):
            sdict = state.scope_dict()
            for node in state.nodes():
                if isinstance(node, nodes.EntryNode):
                    if sdict[node] is None:
                        node.schedule = types.ScheduleType.GPU_Device
                    elif self.sequential_innermaps:
                        node.schedule = types.ScheduleType.Sequential

        #######################################################
        # Step 7: Strict transformations
        if not self.strict_transform:
            return

        # Apply strict state fusions greedily.
        opt = optimizer.SDFGOptimizer(sdfg, inplace=True)
        fusions = 0
        arrays = 0
        options = [
            match for match in opt.get_pattern_matches(strict=True)
            if isinstance(match, (StateFusion, RedundantArray))
        ]
        while options:
            ssdfg = sdfg.sdfg_list[options[0].sdfg_id]
            options[0].apply(ssdfg)
            ssdfg.validate()
            if isinstance(options[0], StateFusion):
                fusions += 1
            if isinstance(options[0], RedundantArray):
                arrays += 1

            options = [
                match for match in opt.get_pattern_matches(strict=True)
                if isinstance(match, (StateFusion, RedundantArray))
            ]

        if Config.get_bool('debugprint') and (fusions > 0 or arrays > 0):
            print('Automatically applied {} strict state fusions and removed'
                  ' {} redundant arrays.'.format(fusions, arrays))


pattern_matching.Transformation.register_stateflow_pattern(GPUTransformState)
