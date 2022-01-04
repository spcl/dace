# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains inter-state transformations of an SDFG to run on an FPGA. """

import copy
import dace
from dace import data, memlet, dtypes, registry, sdfg as sd, subsets
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation


def fpga_update(sdfg, state, depth):
    scope_dict = state.scope_dict()
    for node in state.nodes():
        if (isinstance(node, nodes.AccessNode) and node.desc(sdfg).storage == dtypes.StorageType.Default):
            nodedesc = node.desc(sdfg)
            if depth >= 2:
                nodedesc.storage = dtypes.StorageType.FPGA_Local
            else:
                if scope_dict[node]:
                    nodedesc.storage = dtypes.StorageType.FPGA_Local
                else:
                    nodedesc.storage = dtypes.StorageType.FPGA_Global
        if (hasattr(node, "schedule") and node.schedule == dace.dtypes.ScheduleType.Default):
            node.schedule = dace.dtypes.ScheduleType.FPGA_Device
        if isinstance(node, nodes.NestedSDFG):
            for s in node.sdfg.nodes():
                fpga_update(node.sdfg, s, depth + 1)


@registry.autoregister
class FPGATransformState(transformation.Transformation):
    """ Implements the FPGATransformState transformation. """

    _state = sd.SDFGState()

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(FPGATransformState._state)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        state = graph.nodes()[candidate[FPGATransformState._state]]

        for node, graph in state.all_nodes_recursive():
            # Consume scopes are currently unsupported
            if isinstance(node, (nodes.ConsumeEntry, nodes.ConsumeExit)):
                return False

            # Streams have strict conditions due to code generator limitations
            if (isinstance(node, nodes.AccessNode) and isinstance(graph.parent.arrays[node.data], data.Stream)):
                nodedesc = graph.parent.arrays[node.data]
                sdict = graph.scope_dict()
                if nodedesc.storage in [
                        dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_Pinned, dtypes.StorageType.CPU_ThreadLocal
                ]:
                    return False

                # Cannot allocate FIFO from CPU code
                if sdict[node] is None:
                    return False

                # Arrays of streams cannot have symbolic size on FPGA
                if dace.symbolic.issymbolic(nodedesc.total_size, graph.parent.constants):
                    return False

                # Streams cannot be unbounded on FPGA
                if nodedesc.buffer_size < 1:
                    return False

        for node in state.nodes():

            if (isinstance(node, nodes.AccessNode)
                    and node.desc(sdfg).storage not in (dtypes.StorageType.Default, dtypes.StorageType.Register)):
                return False

            if not isinstance(node, nodes.MapEntry):
                continue

            map_entry = node
            candidate_map = map_entry.map

            # Map schedules that are disallowed to transform to FPGAs
            if (candidate_map.schedule == dtypes.ScheduleType.MPI
                    or candidate_map.schedule == dtypes.ScheduleType.GPU_Device
                    or candidate_map.schedule == dtypes.ScheduleType.FPGA_Device
                    or candidate_map.schedule == dtypes.ScheduleType.GPU_ThreadBlock):
                return False

            # Recursively check parent for FPGA schedules
            sdict = state.scope_dict()
            current_node = map_entry
            while current_node is not None:
                if (current_node.map.schedule == dtypes.ScheduleType.GPU_Device
                        or current_node.map.schedule == dtypes.ScheduleType.FPGA_Device
                        or current_node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock):
                    return False
                current_node = sdict[current_node]

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        state = graph.nodes()[candidate[FPGATransformState._state]]

        return state.label

    def apply(self, sdfg):
        state = sdfg.nodes()[self.subgraph[FPGATransformState._state]]

        # Find source/sink (data) nodes that are relevant outside this FPGA
        # kernel
        shared_transients = set(sdfg.shared_transients())
        input_nodes = [
            n for n in sdutil.find_source_nodes(state)
            if isinstance(n, nodes.AccessNode) and (not sdfg.arrays[n.data].transient or n.data in shared_transients)
        ]
        output_nodes = [
            n for n in sdutil.find_sink_nodes(state)
            if isinstance(n, nodes.AccessNode) and (not sdfg.arrays[n.data].transient or n.data in shared_transients)
        ]

        fpga_data = {}

        # Input nodes may also be nodes with WCR memlets
        # We have to recur across nested SDFGs to find them
        wcr_input_nodes = set()
        stack = []

        parent_sdfg = {state: sdfg}  # Map states to their parent SDFG
        for node, graph in state.all_nodes_recursive():
            if isinstance(graph, dace.SDFG):
                parent_sdfg[node] = graph
            if isinstance(node, dace.sdfg.nodes.AccessNode):
                for e in graph.in_edges(node):
                    if e.data.wcr is not None:
                        trace = dace.sdfg.trace_nested_access(node, graph, parent_sdfg[graph])
                        for node_trace, memlet_trace, state_trace, sdfg_trace in trace:
                            # Find the name of the accessed node in our scope
                            if state_trace == state and sdfg_trace == sdfg:
                                _, outer_node = node_trace
                                if outer_node is not None:
                                    break
                        else:
                            # This does not trace back to the current state, so
                            # we don't care
                            continue
                        input_nodes.append(outer_node)
                        wcr_input_nodes.add(outer_node)
        if input_nodes:
            # create pre_state
            pre_state = sd.SDFGState('pre_' + state.label, sdfg)

            for node in input_nodes:

                if not isinstance(node, dace.sdfg.nodes.AccessNode):
                    continue
                desc = node.desc(sdfg)
                if not isinstance(desc, dace.data.Array):
                    # TODO: handle streams
                    continue

                if node.data in fpga_data:
                    fpga_array = fpga_data[node.data]
                elif node not in wcr_input_nodes:
                    fpga_array = sdfg.add_array('fpga_' + node.data,
                                                desc.shape,
                                                desc.dtype,
                                                transient=True,
                                                storage=dtypes.StorageType.FPGA_Global,
                                                allow_conflicts=desc.allow_conflicts,
                                                strides=desc.strides,
                                                offset=desc.offset)
                    fpga_array[1].location = copy.copy(desc.location)
                    desc.location.clear()
                    fpga_data[node.data] = fpga_array

                pre_node = pre_state.add_read(node.data)
                pre_fpga_node = pre_state.add_write('fpga_' + node.data)
                mem = memlet.Memlet(data=node.data, subset=subsets.Range.from_array(desc))
                pre_state.add_edge(pre_node, None, pre_fpga_node, None, mem)

                if node not in wcr_input_nodes:
                    fpga_node = state.add_read('fpga_' + node.data)
                    sdutil.change_edge_src(state, node, fpga_node)
                    state.remove_node(node)

            sdfg.add_node(pre_state)
            sdutil.change_edge_dest(sdfg, state, pre_state)
            sdfg.add_edge(pre_state, state, sd.InterstateEdge())

        if output_nodes:

            post_state = sd.SDFGState('post_' + state.label, sdfg)

            for node in output_nodes:

                if not isinstance(node, dace.sdfg.nodes.AccessNode):
                    continue
                desc = node.desc(sdfg)
                if not isinstance(desc, dace.data.Array):
                    # TODO: handle streams
                    continue

                if node.data in fpga_data:
                    fpga_array = fpga_data[node.data]
                else:
                    fpga_array = sdfg.add_array('fpga_' + node.data,
                                                desc.shape,
                                                desc.dtype,
                                                transient=True,
                                                storage=dtypes.StorageType.FPGA_Global,
                                                allow_conflicts=desc.allow_conflicts,
                                                strides=desc.strides,
                                                offset=desc.offset)
                    fpga_array[1].location = copy.copy(desc.location)
                    desc.location.clear()
                    fpga_data[node.data] = fpga_array
                # fpga_node = type(node)(fpga_array)

                post_node = post_state.add_write(node.data)
                post_fpga_node = post_state.add_read('fpga_' + node.data)
                mem = memlet.Memlet(f"fpga_{node.data}", None, subsets.Range.from_array(desc))
                post_state.add_edge(post_fpga_node, None, post_node, None, mem)

                fpga_node = state.add_write('fpga_' + node.data)
                sdutil.change_edge_dest(state, node, fpga_node)
                state.remove_node(node)

            sdfg.add_node(post_state)
            sdutil.change_edge_src(sdfg, state, post_state)
            sdfg.add_edge(state, post_state, sd.InterstateEdge())

        # propagate memlet info from a nested sdfg
        for src, src_conn, dst, dst_conn, mem in state.edges():
            if mem.data is not None and mem.data in fpga_data:
                mem.data = 'fpga_' + mem.data
        fpga_update(sdfg, state, 0)
