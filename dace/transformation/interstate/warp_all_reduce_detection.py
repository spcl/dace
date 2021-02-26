from dace import properties
from dace import registry
from dace.transformation import pattern_matching
from dace import nodes
from dace.sdfg import utils
from dace.sdfg import state as dace_state
from dace.sdfg import sdfg as dace_sdfg
from dace.sdfg import graph as sdfg_graph
from dace.frontend import operations
from dace import dtypes
from dace.transformation import transformation
import textwrap
from dace.memlet import Memlet
from typing import List, Tuple
from dace.libraries.standard.nodes import ParallelAllReduce


def add_all_reduce_node(state: dace_state.SDFGState, wcr: str):
    all_reduce_node = ParallelAllReduce(name='parallel_all_reduce', wcr=wcr)

    return all_reduce_node


def parse_accumulate_state(state: dace_state.SDFGState) -> Tuple[
        nodes.AccessNode, sdfg_graph.MultiConnectorEdge, nodes.Tasklet,
        sdfg_graph.MultiConnectorEdge, nodes.AccessNode]:
    id_tasklet: nodes.Tasklet
    for n in state.nodes():
        if isinstance(n, nodes.Tasklet):
            id_tasklet = n

    in_edge = state.in_edges(id_tasklet)[0]
    out_edge = state.out_edges(id_tasklet)[0]

    return in_edge.src, in_edge, id_tasklet, out_edge, out_edge.dst

def add_all_reduce_init_tasklet(state: dace_state.SDFGState, wcr: str):
    init_name = 'init'
    in_acc_name = 'in_acc'
    out_acc_name = 'out_acc'

    reduction_type = operations.detect_reduction_type(wcr)

    if reduction_type == dtypes.ReductionType.Max:
        reduction_op = 'out_acc = init > in_acc ? init : in_acc;'
    elif reduction_type == dtypes.ReductionType.Sum:
        reduction_op = 'out_acc = init + in_acc;'
    else:
        raise Exception("Unknown reduction type")

    warp_all_reduce_init_tasklet: nodes.Tasklet = state.add_tasklet(
            name='warp_all_reduce_init',
            inputs={init_name, in_acc_name},
            outputs={out_acc_name},
            code=reduction_op,
            language=dtypes.Language.CPP)

    return warp_all_reduce_init_tasklet


def replace_access_nodes(state: dace_sdfg, old_name: str, new_name: str):
    old_access_nodes = [n for n in state.nodes()
                        if isinstance(n, nodes.AccessNode) and n.data == old_name]

    for n in old_access_nodes:
        new_node = state.add_access(new_name)
        for e in state.in_edges(n):
            new_data = None if e.data.is_empty() else new_name
            state.add_edge(e.src, e.src_conn, new_node, None, Memlet(data=new_data, subset=e.data.subset))
        for e in state.out_edges(n):
            new_data = None if e.data.is_empty() else new_name
            state.add_edge(new_node, None, e.dst, e.dst_conn, Memlet(data=new_name, subset=e.data.subset))
        state.remove_node(n)


@registry.autoregister_params()
class WarpAllReduceDetection(transformation.Transformation):

    accumulate_state = transformation.PatternNode(dace_sdfg.SDFGState)

    @staticmethod
    def expressions():
        return [
            utils.node_path_graph(
                WarpAllReduceDetection.accumulate_state,
            )
        ]

    @staticmethod
    def can_be_applied(sdfg: dace_sdfg.SDFG,
                       candidate,
                       expr_index,
                       _sdfg: dace_sdfg.SDFG,
                       strict=False):
        accumulate_state: dace_state.SDFGState = sdfg.nodes()[candidate[WarpAllReduceDetection.accumulate_state]]

        # we expect exactly 3 nodes in accumulate state: access node -> identity tasklet --(wcr)-> access node
        if len(accumulate_state.nodes()) != 3:
            return False

        identity_tasklet: nodes.Tasklet = None
        for node in accumulate_state.nodes():
            if isinstance(node, nodes.Tasklet):
                identity_tasklet = node

        if identity_tasklet is None:
            return False

        oe = accumulate_state.out_edges(identity_tasklet)
        if len(oe) != 1:
            return False

        out_edge: sdfg_graph.MultiConnectorEdge = oe[0]
        memlet: Memlet =  out_edge.data
        if not memlet.wcr:
            return False

        return True

    def apply(self, sdfg: dace_sdfg.SDFG):
        candidate = self.subgraph
        accumulate_state: dace_state.SDFGState = sdfg.nodes()[candidate[WarpAllReduceDetection.accumulate_state]]
        in_access, in_edge, _, out_edge, out_access = parse_accumulate_state(accumulate_state)

        # detect exit states
        exit_states = [n for n in sdfg.nodes() if not sdfg.out_edges(n)]

        # create transient to store result of all reduce
        all_reduce_name = 'all_reduce_output'
        sdfg.add_transient(
            name=all_reduce_name,
            shape=sdfg.arrays[out_access.data].shape,
            dtype=sdfg.arrays[out_access.data].dtype)

        # create and fill all reduce state
        all_reduce_state: dace_state.SDFGState = sdfg.add_state('all_reduce_state')
        all_reduce_in = all_reduce_state.add_access(in_access.data)
        all_reduce_node = add_all_reduce_node(all_reduce_state, out_edge.data.wcr)
        all_reduce_out = all_reduce_state.add_access(all_reduce_name)

        all_reduce_state.add_edge(all_reduce_in, None, all_reduce_node, None, in_edge.data)
        all_reduce_state.add_edge(all_reduce_node, None, all_reduce_out, None, Memlet(data=all_reduce_name))

        # create state to add initial value of accumulate variable
        all_reduce_init_state: dace_state.SDFGState = sdfg.add_state('all_reduce_init_state')
        init_access = all_reduce_init_state.add_access(out_access.data)
        in_acc_access = all_reduce_init_state.add_access(all_reduce_name)
        out_acc_access = all_reduce_init_state.add_access(all_reduce_name)
        init_tasklet = add_all_reduce_init_tasklet(all_reduce_init_state, out_edge.data.wcr)
        all_reduce_init_state.add_edge(init_access, None, init_tasklet, 'init',
                                       Memlet(data=out_edge.data.data, subset=out_edge.data.subset))
        all_reduce_init_state.add_edge(in_acc_access, None, init_tasklet, 'in_acc', Memlet(data=all_reduce_name))
        all_reduce_init_state.add_edge(init_tasklet, 'out_acc', out_acc_access, None, Memlet(data=all_reduce_name))

        # create state to write result to accumulate variable
        all_reduce_write_state: dace_state.SDFGState = sdfg.add_state('all_reduce_write_state')
        in_reduced_access = all_reduce_write_state.add_access(all_reduce_name)
        out_reduced_access = all_reduce_write_state.add_access(out_access.data)
        # TODO only a single thread in a map should write it
        all_reduce_write_tasklet =  all_reduce_write_state.add_tasklet(
            name='tasklet',
            inputs={'in'},
            outputs={'out'},
            code='out = in;',
            language=dtypes.Language.CPP)
        all_reduce_write_state.add_edge(in_reduced_access, None, all_reduce_write_tasklet, 'in', Memlet(all_reduce_name))
        all_reduce_write_state.add_edge(all_reduce_write_tasklet, 'out', out_reduced_access, None,
                                        Memlet(data=out_edge.data.data, subset=out_edge.data.subset))

        # replace access nodes in the subsequent states that refer to the result of reduction
        for e in sdfg.bfs_edges(accumulate_state):
            state: dace_state.SDFGState = e.dst
            replace_access_nodes(state, out_access.data, all_reduce_name)

        # connect new states with edges
        for e in sdfg.in_edges(accumulate_state):
            e: dace_sdfg.Edge
            sdfg.add_edge(e.src, all_reduce_state, e.data)

        sdfg.add_edge(all_reduce_state, all_reduce_init_state, dace_sdfg.InterstateEdge())

        for e in sdfg.out_edges(accumulate_state):
            e: dace_sdfg.Edge
            sdfg.add_edge(all_reduce_init_state, e.dst, e.data)

        for n in exit_states:
            sdfg.add_edge(n, all_reduce_write_state, dace_sdfg.InterstateEdge())

        # remove old state
        sdfg.remove_node(accumulate_state)

# theoretically it should go to single state (dataflow) transformations
@registry.autoregister_params(singlestate=True)
class WarpAllReduceDetectionNoTasklet(transformation.Transformation):
    """
    Works for patterns AccessNode --(WCR)-> AccessNode
    """

    src_access = transformation.PatternNode(nodes.AccessNode)
    dst_access = transformation.PatternNode(nodes.AccessNode)

    @staticmethod
    def expressions():
        return [
            utils.node_path_graph(
                WarpAllReduceDetectionNoTasklet.src_access,
                WarpAllReduceDetectionNoTasklet.dst_access,
            )
        ]

    @staticmethod
    def can_be_applied(state: dace_sdfg.SDFGState,
                       candidate,
                       expr_index,
                       sdfg: dace_sdfg.SDFG,
                       strict=False):
        src_access: nodes.AccessNode = state.nodes()[candidate[WarpAllReduceDetectionNoTasklet.src_access]]
        dst_access: nodes.AccessNode = state.nodes()[candidate[WarpAllReduceDetectionNoTasklet.dst_access]]

        edges = state.edges_between(src_access, dst_access)
        if len(edges) != 1:
            return False

        if not edges[0].data.wcr:
            return False

        if state.in_degree(src_access) == 0:
            return False

        if state.out_degree(dst_access) > 0:
            return False

        return True

    def apply(self, sdfg: dace_sdfg.SDFG):
        state = sdfg.nodes()[self.state_id]

        candidate = self.subgraph
        src_access: nodes.AccessNode = state.nodes()[candidate[WarpAllReduceDetectionNoTasklet.src_access]]
        dst_access: nodes.AccessNode = state.nodes()[candidate[WarpAllReduceDetectionNoTasklet.dst_access]]

        wcr_edge = state.edges_between(src_access, dst_access)[0]

        # create new state for WCR
        new_state = sdfg.add_state()
        new_src = new_state.add_access(src_access.data)
        new_dst = new_state.add_access(dst_access.data)

        tasklet: nodes.Tasklet = new_state.add_tasklet(name='id_tasklet', inputs={'a'}, outputs={'b'}, code='b = a')
        new_state.add_edge(new_src, None, tasklet, 'a', Memlet(data=src_access.data, subset=wcr_edge.data.subset))
        new_state.add_edge(tasklet, 'b', new_dst, None, Memlet(data=dst_access.data, subset=wcr_edge.data.subset, wcr=wcr_edge.data.wcr))

        # remove nodes from existing state
        state.remove_node(dst_access)

        # connect new state to the graph
        old_interstate_edges = sdfg.out_edges(state)
        for edge in old_interstate_edges:
            sdfg.add_edge(new_state, edge.dst, dace_sdfg.InterstateEdge())
            sdfg.remove_edge(edge)

        sdfg.add_edge(state, new_state, dace_sdfg.InterstateEdge())

        # now apply transformation to new state
        WarpAllReduceDetection.apply_to(sdfg, accumulate_state=new_state)