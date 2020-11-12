from dace import data, dtypes, registry, symbolic, subsets
from dace.sdfg import nodes, SDFG, propagation, SDFGState
from dace.sdfg import utils as sdutil
from dace.sdfg.scope import ScopeSubgraphView
from dace.transformation import pattern_matching
from dace.transformation.helpers import replicate_scope
from dace.properties import Property, make_properties
import dace

from dace.sdfg import graph as sdfg_graph
from dace.sdfg import nodes as sdfg_nodes
from dace import subsets as dace_subsets

from typing import List, Tuple


def get_edges(
    sdfg_state: SDFGState, inner_node: sdfg_nodes.Node,
    map_exit: sdfg_nodes.MapExit, output_node: sdfg_nodes.AccessNode
) -> List[Tuple[sdfg_graph.MultiConnectorEdge, sdfg_graph.MultiConnectorEdge]]:
    internal_edges = sdfg_state.edges_between(inner_node, map_exit)
    external_edges = sdfg_state.edges_between(map_exit, output_node)
    edge_pairs = []

    if len(internal_edges) != len(external_edges):
        raise Exception("Number of inputs and exits of MapExit doesn't match")

    for int_edge in internal_edges:
        for ext_edge in external_edges:
            if int_edge.dst_conn[:3] != 'IN_' or ext_edge.src_conn[:4] != 'OUT_':
                raise Exception("Unexpected edges names")
            int_name = int_edge.dst_conn[3:]  # cut "IN_" prefix
            ext_name = ext_edge.src_conn[4:]  # cut "OUT_" prefix
            if int_name == ext_name:
                edge_pairs.append((int_edge, ext_edge))

    if len(edge_pairs) != len(internal_edges):
        raise Exception("Edges mismatch between MapExit inputs and outputs")

    return edge_pairs


def get_inner_node_and_edge(
    sdfg_state: SDFGState, map_exit: sdfg_nodes.MapExit,
    output_node: sdfg_nodes.AccessNode
) -> Tuple[sdfg_nodes.Node, sdfg_graph.MultiConnectorEdge]:
    outer_edges: List[sdfg_graph.MultiConnectorEdge] = sdfg_state.edges_between(
        map_exit, output_node)
    assert len(outer_edges) == 1
    outer_edge: sdfg_graph.MultiConnectorEdge = outer_edges[0]

    path_edges: List[sdfg_graph.MultiConnectorEdge] = sdfg_state.memlet_path(
        outer_edge)

    # find inner_edge - edge just before outer_edge
    idx = path_edges.index(outer_edge)
    assert idx > 0

    inner_edge: sdfg_graph.MultiConnectorEdge = path_edges[idx - 1]

    inner_node: sdfg_nodes.Node = inner_edge.src

    return inner_node, inner_edge


@registry.autoregister_params(singlestate=True)
@make_properties
class WCRExtraction(pattern_matching.Transformation):

    _map_exit = nodes.MapExit(nodes.Map("", [], []))
    _output_node = nodes.AccessNode("_")

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(WCRExtraction._map_exit,
                                   WCRExtraction._output_node)
        ]

    @staticmethod
    def can_be_applied(sdfg_state: SDFGState,
                       candidate,
                       expr_index,
                       sdfg: SDFG,
                       strict=False):
        map_exit: sdfg_nodes.MapExit = sdfg_state.nodes()[candidate[
            WCRExtraction._map_exit]]
        output_node: sdfg_nodes.AccessNode = sdfg_state.nodes()[candidate[
            WCRExtraction._output_node]]

        inner_node, inner_edge = get_inner_node_and_edge(
            sdfg_state, map_exit, output_node)

        edges = get_edges(sdfg_state, inner_node, map_exit, output_node)

        wcr_found = False

        for int_edge, ext_edge in edges:
            if int_edge.data.wcr is not None:
                if ext_edge.data.wcr is None:
                    raise Exception("Memlets WCR mismatch in MapExit")
                wcr_found = True

        return wcr_found

    @staticmethod
    def match_to_str(graph, candidate):
        map_exit = graph.nodes()[candidate[WCRExtraction._map_exit]]
        output_node = graph.nodes()[candidate[WCRExtraction._output_node]]

        return f'{map_exit.map.label} {map_exit.map.params} -> {output_node.data}'

    def apply(self, sdfg: SDFG):
        sdfg_state = sdfg.nodes()[self.state_id]

        map_exit: sdfg_nodes.MapExit = sdfg_state.nodes()[self.subgraph[
            WCRExtraction._map_exit]]
        output_node: sdfg_nodes.AccessNode = sdfg_state.nodes()[self.subgraph[
            WCRExtraction._output_node]]

        inner_node, inner_edge = get_inner_node_and_edge(
            sdfg_state, map_exit, output_node)

        edge_pairs = get_edges(sdfg_state, inner_node, map_exit, output_node)

        for int_edge, ext_edge in edge_pairs:
            if int_edge.data.wcr is None:  # if edge doesn't contain WCR, transformation will not extract it
                continue

            # make a copy of map

            map = map_exit.map
            param_str: str = ",".join(map.params)
            map_range: subsets.Range = map.range
            map_range_str: List[str] = [
                dace_subsets.Range.dim_to_string(dim) for dim in map_range
            ]
            param = symbolic.pystr_to_symbolic(param_str)

            dtype = sdfg.arrays[ext_edge.data.data].dtype

            me, mx = sdfg_state.add_map(
                'extracted_wcr',
                {str(k): v for k, v in zip(map.params, map_range_str)})
            id_array_name, id_array = sdfg.add_transient(
                'identity_array', [1], dtype)

            id_access = sdfg_state.add_access(id_array_name)

            me.add_in_connector('IN_ID_ENTRY', dtype)
            me.add_out_connector('OUT_ID_ENTRY', dtype)
            mx.add_in_connector('IN_ID_EXIT', dtype)
            mx.add_out_connector('OUT_ID_EXIT', dtype)

            transient_name, transient_array = sdfg.add_transient(
                'extracted_wcr_tmp', map_range.size(), dtype)
            transient_access = sdfg_state.add_access(transient_name)

            int_src_conn = int_edge.src_conn
            int_dst_conn = int_edge.dst_conn
            int_memlet = int_edge.data
            ext_src_conn = ext_edge.src_conn
            ext_memlet = ext_edge.data

            sdfg_state.remove_edge(int_edge)
            sdfg_state.remove_edge(ext_edge)

            transient_indexing = f'{transient_name}[{param_str}]'

            sdfg_state.add_edge(inner_node, int_src_conn, map_exit,
                                int_dst_conn, dace.Memlet(transient_indexing))
            sdfg_state.add_edge(map_exit, ext_src_conn, transient_access, None,
                                dace.Memlet(transient_name))
            sdfg_state.add_edge(transient_access, None, me, 'IN_ID_ENTRY',
                                dace.Memlet(transient_name))
            sdfg_state.add_edge(me, 'OUT_ID_ENTRY', id_access, None,
                                dace.Memlet(transient_indexing))
            sdfg_state.add_edge(id_access, None, mx, 'IN_ID_EXIT', int_memlet)
            sdfg_state.add_edge(mx, 'OUT_ID_EXIT', output_node, None,
                                ext_memlet)
