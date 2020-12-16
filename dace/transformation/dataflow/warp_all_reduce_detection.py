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
from dace.transformation import transformation
import textwrap

from typing import List


@registry.autoregister_params(singlestate=True)
@properties.make_properties
class WarpAllReduceDetection(pattern_matching.Transformation):

    local_access = transformation.PatternNode(nodes.AccessNode)
    reduced_access = transformation.PatternNode(nodes.AccessNode)

    @staticmethod
    def expressions():
        return [
            utils.node_path_graph(
                WarpAllReduceDetection.local_access,
                WarpAllReduceDetection.reduced_access,
            )
        ]

    @staticmethod
    def can_be_applied(state: state.SDFGState,
                       candidate,
                       expr_index,
                       sdfg: sdfg.SDFG,
                       strict=False):
        local_access = state.nodes()[candidate[WarpAllReduceDetection.local_access]]
        reduced_access = state.nodes()[candidate[WarpAllReduceDetection.reduced_access]]

        # 1. access nodes should be inside gpu map over warp
        # 2. there should be reduction
        # 3. reduction should be done over warp index

        return True

    @staticmethod
    def match_to_str(state: state.SDFGState, candidate):
        local_access: nodes.AccessNode = state.nodes()[candidate[WarpAllReduceDetection.local_access]]
        reduced_access: nodes.AccessNode = state.nodes()[candidate[WarpAllReduceDetection.reduced_access]]

        return f'{local_access.label} -> {reduced_access.label}'

    def apply(self, sdfg: sdfg.SDFG):
        state = sdfg.nodes()[self.state_id]

        candidate = self.subgraph
        local_access: nodes.AccessNode = state.nodes()[candidate[WarpAllReduceDetection.local_access]]
        reduced_access: nodes.AccessNode = state.nodes()[candidate[WarpAllReduceDetection.reduced_access]]

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
