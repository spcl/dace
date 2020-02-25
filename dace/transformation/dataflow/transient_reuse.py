from dace import data, memlet, dtypes, registry, sdfg as sd
from dace.graph import nodes, nxutil, edges as ed
from dace.transformation import pattern_matching
from dace.properties import Property, make_properties, SubsetProperty
from dace.memlet import Memlet

@registry.autoregister
@make_properties
class TransientReuse(pattern_matching.Transformation):

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        return graph.label

    @staticmethod
    def expressions():
        return [sd.SDFG('_')]

    def expansion(node):
        pass

    def apply(self, sdfg):
        list = []
        for i, state in enumerate(sdfg.nodes()):
            for j, node in enumerate(state.nodes()):
                if isinstance(node,
                              nodes.AccessNode) and node.desc(sdfg).transient:
                    print(node, type(node))
                    list.append(node)

        temp1 = list[0]
        temp2 = list[2]
        for i, state in enumerate(sdfg.nodes()):
            for e in state.edges():
                if e._dst == temp2:
                    print(e.data.subset)
                    state.add_edge(e._src, e.src_conn, temp1, None, Memlet.from_array(temp1, temp2.desc(sdfg)))
                    state.remove_edge(e)
                if e._src == temp2:
                    state.add_edge(temp1, None, e._dst, e.dst_conn, Memlet.from_array(temp1, temp2.desc(sdfg)))
                    state.remove_edge(e)
        sdfg.states()[0].remove_node(temp2)

        #sdfg.states()[0].remove_edge(sdfg.states()[0].edges()[3])