""" General class for pattern replacement transformations. """
import abc
import dace
from dace import registry, nodes, data as dt
from dace.transformation import transformation, helpers as xfh
from typing import Any, Dict, List, Optional, Tuple, Union
from dace.sdfg import utils as sdutil
from dace.libraries.onnx import nodes as onnx_op
from dace.sdfg import graph as gr


def make_onnx_path(*path_nodes: nodes.Node) -> gr.OrderedDiGraph:
    result = gr.OrderedDiGraph()

    # First add the nodes in order, so that they can be accessed
    path_nodes = [transformation.PatternNode(n) for n in path_nodes]
    result.add_nodes_from(path_nodes)

    # Then make a path and add access nodes as necessary
    last_node = None
    for node in path_nodes:
        if last_node is not None:
            result.add_edge(last_node, node)
        last_node = node

    return result


def add_connecting_access_nodes(graph: gr.OrderedDiGraph):
    edges_to_remove = []
    outputs = {}
    for pnode in graph.nodes():
        if issubclass(pnode.node, (nodes.LibraryNode, nodes.NestedSDFG)):
            if any(issubclass(e.dst.node, (nodes.LibraryNode, nodes.NestedSDFG)) for e in graph.out_edges(pnode)):
                # Make new output node that everyone will link from
                new_node = transformation.PatternNode(nodes.AccessNode)
                graph.add_node(new_node)
                graph.add_edge(pnode, new_node)
                outputs[pnode] = new_node

    for e in graph.edges():
        if (issubclass(e.src.node, (nodes.LibraryNode, nodes.NestedSDFG))
                and issubclass(e.dst.node, (nodes.LibraryNode, nodes.NestedSDFG))):
            # Direct path between two library nodes means that there is at least
            # another access node in between
            if e.src in outputs:
                graph.add_edge(outputs[e.src], e.dst)
                edges_to_remove.append(e)
            else:
                raise ValueError('Found directly connected library nodes with source not designated as output')
    for e in edges_to_remove:
        graph.remove_edge(e)


def onnx_constant_or_none(sdfg: dace.SDFG, node_or_name: Union[nodes.AccessNode, str]) -> Optional[Any]:
    name = node_or_name if isinstance(node_or_name, str) else node_or_name.data
    if name not in sdfg._parent_onnx_model.clean_weights:
        return None
    cten = sdfg._parent_onnx_model.clean_weights[name]
    return cten.item() if cten.numel() == 1 else cten.tolist()


class ReplacementTransformation(transformation.SingleStateTransformation, abc.ABC):

    @classmethod
    @abc.abstractmethod
    def pattern(cls) -> gr.OrderedDiGraph[nodes.Node, dace.Memlet]:
        """ Returns a pattern to match as a directed graph. """
        raise NotImplementedError

    @abc.abstractmethod
    def replacement(self, subgraph: List[nodes.Node], sdfg: dace.SDFG,
                    state: dace.SDFGState) -> Tuple[nodes.Node, Dict[str, Tuple[nodes.Node, Union[str, dt.Data]]]]:
        """
        Defines replacement behavior for the transformation. This method returns
        a node (which could also be a nested SDFG if a subgraph should be
        returned), accompanied by instructions for reconnecting the surrounding
        nodes and creating new data (arrays).
        :param subgraph: The list of nodes in the matched state with the same
                         IDs as the pattern subgraph.
        :param sdfg: The SDFG in which to perform the replacement.
        :param state: The state in which the subgraph was found.
        :return: A 2-tuple of (new node, mapping), where the latter maps a
                 connector name on the new node to either a pair of
                 (old node, old connector) to redirect from, or
                 (None, data descriptor) if a new one shall be created.
        """
        raise NotImplementedError

    @classmethod
    def expressions(cls):
        if hasattr(cls, '_pattern'):
            return [cls._pattern]
        result = cls.pattern()
        add_connecting_access_nodes(result)

        # Set subgraph as class property
        cls._pattern = result
        # Set pattern nodes as class properties
        for i, node in enumerate(result.nodes()):
            setattr(cls, f'_pnode{i}', node)
        return [result]

    def can_be_applied(self, graph: Union[dace.SDFG, dace.SDFGState], candidate: Dict[transformation.PatternNode, int],
                       expr_index: int, sdfg: dace.SDFG, simplify: bool) -> bool:
        # All internal nodes must not be global (non-transient) or reused
        # anywhere else
        subgraph = gr.SubgraphView(graph, [graph.node(id) for id in candidate.values()])
        for node in subgraph.nodes():
            # Check for internal nodes
            if node in subgraph.source_nodes() or node in subgraph.sink_nodes():
                continue
            if not isinstance(node, nodes.AccessNode):
                continue
            if not node.desc(sdfg).transient:
                return False
            other_data_nodes_with_same_name = [
                n for s in sdfg.nodes() for n in s.nodes()
                if isinstance(n, nodes.AccessNode) and n.data == node.data and n not in subgraph.nodes()
            ]
            if len(other_data_nodes_with_same_name) > 0:
                return False
        return True

    def apply(self, sdfg: dace.SDFG) -> nodes.Node:
        state: dace.SDFGState = sdfg.node(self.state_id)
        matcher = self.expressions()[0]
        subgraph = [state.node(self.subgraph[n]) for n in matcher.nodes()]
        new_node, reconnection = self.replacement(subgraph, sdfg, state)

        # Remap edges and add new arrays
        for new_conn, (node, old_conn) in reconnection.items():
            # Make new array
            if node is None:
                desc = old_conn
                name = sdfg.add_datadesc('_' + new_conn, desc, find_new_name=True)
                node = state.add_access(name)
                if new_conn in new_node.in_connectors:
                    state.add_edge(node, None, new_node, new_conn, dace.Memlet(name))
                elif new_conn in new_node.out_connectors:
                    state.add_edge(new_node, new_conn, node, None, dace.Memlet(name))
                continue
            # END of new array

            if new_conn in new_node.in_connectors:
                e = next(state.in_edges_by_connector(node, old_conn))
                xfh.redirect_edge(state, e, new_dst=new_node, new_dst_conn=new_conn)
            elif new_conn in new_node.out_connectors:
                e = next(state.out_edges_by_connector(node, old_conn))
                xfh.redirect_edge(state, e, new_src=new_node, new_src_conn=new_conn)

        # Remove subgraph nodes that are not connected from outside
        sgview = gr.SubgraphView(state, subgraph)
        state.remove_nodes_from(
            [n for n in subgraph if isinstance(n, nodes.CodeNode) or state.degree(n) == sgview.degree(n)])
        # Remove orphan nodes
        state.remove_nodes_from([n for n in state.nodes() if isinstance(n, nodes.AccessNode) and state.degree(n) == 0])
        return new_node
