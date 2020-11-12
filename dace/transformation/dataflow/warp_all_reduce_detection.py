from dace import properties
from dace import registry
from dace.transformation import pattern_matching
from dace import nodes
from dace.sdfg import utils
from dace.sdfg import state
from dace.sdfg import sdfg
from dace.sdfg import graph as sdfg_graph
from dace.frontend import operations
from dace import dtypes

import textwrap

from typing import List


@registry.autoregister_params(singlestate=True)
@properties.make_properties
class WarpAllReduceDetection(pattern_matching.Transformation):
    _map_entry = nodes.MapEntry(nodes.Map("", [], []))
    _temp_node = nodes.AccessNode("_")
    _map_exit = nodes.MapExit(nodes.Map("", [], []))
    _output_node = nodes.AccessNode("_")

    @staticmethod
    def expressions():
        return [
            utils.node_path_graph(
                WarpAllReduceDetection._map_entry,
                WarpAllReduceDetection._temp_node,
                WarpAllReduceDetection._map_exit,
                WarpAllReduceDetection._output_node,
            )
        ]

    @staticmethod
    def can_be_applied(sdfg_state: state.SDFGState,
                       candidate,
                       expr_index,
                       sdfg: sdfg.SDFG,
                       strict=False):
        map_entry: nodes.MapEntry = sdfg_state.nodes()[candidate[
            WarpAllReduceDetection._map_entry]]
        temp_node: nodes.AccessNode = sdfg_state.nodes()[candidate[
            WarpAllReduceDetection._temp_node]]
        map_exit: nodes.MapExit = sdfg_state.nodes()[candidate[
            WarpAllReduceDetection._map_exit]]
        output_node: nodes.AccessNode = sdfg_state.nodes()[candidate[
            WarpAllReduceDetection._output_node]]

        if len(sdfg_state.in_edges(map_entry)) != 1 or len(
                sdfg_state.out_edges(map_entry)) != 1:
            return False

        if len(sdfg_state.in_edges(temp_node)) != 1 or len(
                sdfg_state.out_edges(temp_node)) != 1:
            return False

        if len(sdfg_state.in_edges(map_exit)) != 1 or len(
                sdfg_state.out_edges(map_exit)) != 1:
            return False

        if len(sdfg_state.in_edges(output_node)) != 1:
            return False

        if (not sdfg_state.in_edges(map_exit)[0].data.wcr) or (
                not sdfg_state.out_edges(map_exit)[0].data.wcr):
            return False

        # TODO we need to check that exactly 32 operations is done and those operations have warp schedule

        return True

    @staticmethod
    def match_to_str(sdfg_state: state.SDFGState, candidate):
        map_entry: nodes.MapEntry = sdfg_state.nodes()[candidate[
            WarpAllReduceDetection._map_entry]]
        temp_node: nodes.AccessNode = sdfg_state.nodes()[candidate[
            WarpAllReduceDetection._temp_node]]
        map_exit: nodes.MapExit = sdfg_state.nodes()[candidate[
            WarpAllReduceDetection._map_exit]]
        output_node: nodes.AccessNode = sdfg_state.nodes()[candidate[
            WarpAllReduceDetection._output_node]]

        return f'{map_entry.map.label} -> {temp_node.label} -> {map_exit.map.params} -> {output_node.data}'

    def apply(self, sdfg: sdfg.SDFG):
        sdfg_state = sdfg.nodes()[self.state_id]

        candidate = self.subgraph

        map_entry: nodes.MapEntry = sdfg_state.nodes()[candidate[
            WarpAllReduceDetection._map_entry]]
        temp_node: nodes.AccessNode = sdfg_state.nodes()[candidate[
            WarpAllReduceDetection._temp_node]]
        map_exit: nodes.MapExit = sdfg_state.nodes()[candidate[
            WarpAllReduceDetection._map_exit]]
        output_node: nodes.AccessNode = sdfg_state.nodes()[candidate[
            WarpAllReduceDetection._output_node]]

        # get edges
        outer_edge_in: sdfg_graph.MultiConnectorEdge = sdfg_state.in_edges(
            map_entry)[0]
        inner_edge_in: sdfg_graph.MultiConnectorEdge = sdfg_state.out_edges(
            map_entry)[0]
        inner_edge_out: sdfg_graph.MultiConnectorEdge = sdfg_state.in_edges(
            map_exit)[0]
        outer_edge_out: sdfg_graph.MultiConnectorEdge = sdfg_state.out_edges(
            map_exit)[0]

        input_name = 'in'
        output_name = 'out'

        outer_edge_in.dst_conn = input_name
        outer_edge_out.src_conn = output_name

        reduction_type = operations.detect_reduction_type(
            inner_edge_out.data.wcr)

        if reduction_type == dtypes.ReductionType.Max:
            reduction_op = 'if (new_val > out) { out = new_val; }'
        elif reduction_type == dtypes.ReductionType.Sum:
            reduction_op = 'out += new_val;'
        else:
            raise Exception("Unknown reduction type")

        code = textwrap.dedent("""
            out = in[threadIdx.x];
            # pragma unroll
            for (int i = 1; i < 32; i = i * 2) {{
                auto new_val = __shfl_xor_sync(0xffffffff, out, i);
                {reduction_op}
            }}
        """.format(reduction_op=reduction_op))

        warp_all_reduce_tasklet: nodes.Tasklet = sdfg_state.add_tasklet(
            name='warp_all_reduce',
            inputs={input_name},
            outputs={output_name},
            code=code,
            language=dtypes.Language.CPP)

        new_edge_in = sdfg_state.add_edge(outer_edge_in.src,
                                          outer_edge_in.src_conn,
                                          warp_all_reduce_tasklet, input_name,
                                          outer_edge_in.data)

        new_edge_out = sdfg_state.add_edge(warp_all_reduce_tasklet, output_name,
                                           outer_edge_out.dst,
                                           outer_edge_out.dst_conn,
                                           outer_edge_out.data)

        # remove WCR since it is not required anymore
        new_edge_in.data.wcr = None
        new_edge_out.data.wcr = None

        # TODO we should also remove initialization state for this WCR

        sdfg_state.remove_edge(outer_edge_in)
        sdfg_state.remove_edge(inner_edge_in)
        sdfg_state.remove_edge(inner_edge_out)
        sdfg_state.remove_edge(outer_edge_out)

        sdfg_state.remove_node(map_entry)
        sdfg_state.remove_node(temp_node)
        sdfg_state.remove_node(map_exit)
