import networkx as nx

from networkx.algorithms import isomorphism as iso

from typing import Generator, Tuple, List

from dace import SDFG, SDFGState, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import SubgraphView

from dace.transformation.transformation import PatternNode
from dace.transformation.passes import pattern_matching
from dace.transformation.dataflow import OTFMapFusion


def map_fusion_enumerator(sdfg: SDFG) -> Generator[Tuple[SubgraphView, List], None, None]:
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.nodes():
            fusion_graph = nx.DiGraph()

            matches = []
            for match in _map_fusion_pattern_iterator(state):
                first_map_exit, access_node, second_map_entry = match
                first_map_entry = state.entry_node(first_map_exit)

                xform = OTFMapFusion()
                xform._sdfg = nsdfg
                xform.state_id = nsdfg.node_id(state)
                xform.first_map_exit = first_map_exit
                xform.array = access_node
                xform.second_map_entry = second_map_entry

                if not xform.can_be_applied(state, sdfg=nsdfg, expr_index=0):
                    continue

                f_node_id = state.node_id(first_map_entry)
                if not fusion_graph.has_node(f_node_id):
                    fusion_graph.add_node(f_node_id)

                s_node_id = state.node_id(second_map_entry)
                if not fusion_graph.has_node(s_node_id):
                    fusion_graph.add_node(s_node_id)

                fusion_graph.add_edge(f_node_id, s_node_id)

                matches.append((first_map_entry, access_node, second_map_entry))

            topo_sort = list(nx.topological_sort(fusion_graph))

            # Sort matches topologically
            matches = sorted(matches, key=lambda match: topo_sort.index(state.node_id(match[0])))

            components = []
            for component in nx.weakly_connected_components(fusion_graph):
                nodes = set()
                for map_entry_id in component:
                    map_entry = state.node(map_entry_id)
                    map_exit = state.exit_node(map_entry)

                    nodes.add(map_entry)
                    nodes.add(map_exit)
                    for node in state.all_nodes_between(map_entry, map_exit):
                        nodes.add(node)

                # Filter matches for subgraph
                subgraph_matches = []
                for first_map_entry, access_node, second_map_entry in matches:
                    f_node_id = state.node_id(first_map_entry)
                    s_node_id = state.node_id(second_map_entry)
                    if f_node_id in component and s_node_id in component:
                        subgraph_matches.append((first_map_entry, access_node, second_map_entry))

                components.append((SubgraphView(state, nodes), subgraph_matches))

            for component in components:
                yield component


def _map_fusion_pattern_iterator(state: SDFGState) -> Generator[List, None, None]:
    pattern = sdutil.node_path_graph(
        PatternNode(nodes.ExitNode),
        PatternNode(nodes.AccessNode),
        PatternNode(nodes.EntryNode),
    )
    pattern_digraph = pattern_matching.collapse_multigraph_to_nx(pattern)
    graph_matcher = iso.DiGraphMatcher(
        pattern_matching.collapse_multigraph_to_nx(state),
        pattern_digraph,
        node_match=pattern_matching.type_or_class_match,
        edge_match=None,
    )
    for subgraph in graph_matcher.subgraph_isomorphisms_iter():
        yield [state.node(i) for i in subgraph.keys()]
