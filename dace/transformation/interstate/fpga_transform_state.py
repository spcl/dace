# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains inter-state transformations of an SDFG to run on an FPGA. """

import copy
import dace
from dace import memlet, dtypes, sdfg as sd, subsets
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ControlFlowRegion, SDFGState
from dace.transformation import transformation, helpers as xfh


def fpga_update(sdfg: SDFG, state: SDFGState, depth: int):
    scope_dict = state.scope_dict()
    for node in state.nodes():
        if (isinstance(node, nodes.AccessNode) and node.desc(sdfg).storage == dtypes.StorageType.Default):
            nodedesc = node.desc(sdfg)
            pmap = xfh.get_parent_map(state, node)
            if depth >= 2 or (pmap is not None and pmap[0].schedule == dtypes.ScheduleType.FPGA_Device):
                nodedesc.storage = dtypes.StorageType.FPGA_Local
            else:
                if scope_dict[node]:
                    nodedesc.storage = dtypes.StorageType.FPGA_Local
                else:
                    nodedesc.storage = dtypes.StorageType.FPGA_Global
        if (hasattr(node, "schedule") and node.schedule == dace.dtypes.ScheduleType.Default):
            node.schedule = dace.dtypes.ScheduleType.FPGA_Device
        if isinstance(node, nodes.NestedSDFG):
            for s in node.sdfg.states():
                fpga_update(node.sdfg, s, depth + 1)


@transformation.explicit_cf_compatible
class FPGATransformState(transformation.MultiStateTransformation):
    """ Implements the FPGATransformState transformation. """

    state = transformation.PatternNode(sd.SDFGState)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.state)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        state = self.state

        if not xfh.can_run_state_on_fpga(state):
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

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG):
        state = self.state

        # Find source/sink (data) nodes that are relevant outside this FPGA
        # kernel
        shared_transients = set(sdfg.shared_transients())
        input_nodes = [
            n for n in state.source_nodes()
            if isinstance(n, nodes.AccessNode) and (not sdfg.arrays[n.data].transient or n.data in shared_transients)
        ]
        output_nodes = [
            n for n in state.sink_nodes()
            if isinstance(n, nodes.AccessNode) and (not sdfg.arrays[n.data].transient or n.data in shared_transients)
        ]

        fpga_data = {}

        # Input nodes may also be nodes with WCR memlets
        # We have to recur across nested SDFGs to find them
        wcr_input_nodes = set()

        for node, node_parent_graph in state.all_nodes_recursive():
            if isinstance(node, dace.sdfg.nodes.AccessNode):
                for e in node_parent_graph.in_edges(node):
                    if e.data.wcr is not None:
                        trace = dace.sdfg.trace_nested_access(node, node_parent_graph, node_parent_graph.sdfg)
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
                        if any(outer_node.data == n.data for n in input_nodes):
                            continue  # Skip adding duplicates
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

            graph.add_node(pre_state)
            sdutil.change_edge_dest(graph, state, pre_state)
            graph.add_edge(pre_state, state, sd.InterstateEdge())

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

            graph.add_node(post_state)
            sdutil.change_edge_src(graph, state, post_state)
            graph.add_edge(state, post_state, sd.InterstateEdge())

        # propagate memlet info from a nested sdfg
        for src, src_conn, dst, dst_conn, mem in state.edges():
            if mem.data is not None and mem.data in fpga_data:
                mem.data = 'fpga_' + mem.data
        fpga_update(sdfg, state, 0)
