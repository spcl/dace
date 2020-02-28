from dace import data, memlet, dtypes, registry, sdfg as sd
from dace.graph import nodes, nxutil, edges as ed
from dace.transformation import pattern_matching
from dace.properties import Property, make_properties, SubsetProperty
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState

@registry.autoregister
@make_properties
class TransientReuse(pattern_matching.Transformation):

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
        state: SDFGState = sdfg.nodes()[self.state_id]
        # 1 Find all transients.
        transients = []

        for a in sdfg.arrays:
            if sdfg.arrays[a].transient == True:
                transients.append(a)

        # TODO 2 Determine all possible reuses and decide on a mapping.
        print(transients)
        mapping = []
        mapping.append((transients[2],transients[0]))
        print(state.nodes())
        # TODO 3 For each mapping redirect edges and rename memlets in the whole tree.
        for (old, new) in mapping:
            new_node = state.find_node(new)
            for e in state.all_edges(state.find_node(old)):
                print('edge: ', e.data, e.src, e.dst)
                for edge in state.memlet_tree(e):
                    if edge is not e:
                        edge.data.data = new
                        if edge.dst.label == old:
                            state.add_edge(edge.src, edge.src_conn, new_node, None, edge.data)

                            state.remove_edge(edge)
                        if edge._src.label == old:
                            state.add_edge(new_node, None, edge.dst, edge.dst_conn, edge.data)
                            state.remove_edge(edge)

                e.data.data = new
                if e.dst.label == old:
                    state.add_edge(e.src, e.src_conn, new_node, None, e.data)
                    state.remove_edge(e)
                if e._src.label == old:
                    state.add_edge(new_node, None, e.dst, e.dst_conn, e.data)

                    state.remove_edge(e)

            state.remove_node(state.find_node(old))
        print('yes')
        '''list = []
        for i, state in enumerate(sdfg.states()):
            for j, node in enumerate(state.nodes()):
                if isinstance(node,
                              nodes.AccessNode) and node.desc(sdfg).transient:
                    list.append(node)

        for a in sdfg.arrays:
            print(a)

        temp1 = list[0]
        temp2 = list[2]
        for i, state in enumerate(sdfg.nodes()):
            for e in state.edges():
                print(sdfg.arrays[e.data.data].transient)
                if e._dst == temp2:
                    print(e.data.data)
                    memlet = e.data
                    memlet.data = temp1.label
                    state.add_edge(e._src, e.src_conn, temp1, None, memlet )
                    state.remove_edge(e)
                if e._src == temp2:
                    state.add_edge(temp1, None, e._dst, e.dst_conn, Memlet.from_array(temp1, temp2.desc(sdfg)))
                    state.remove_edge(e)
        sdfg.states()[0].remove_node(temp2)

        #sdfg.states()[0].remove_edge(sdfg.states()[0].edges()[3])'''