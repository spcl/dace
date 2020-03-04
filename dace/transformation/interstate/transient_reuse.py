from dace import data, memlet, dtypes, registry, sdfg as sd
from dace.graph import nodes, nxutil, edges as ed
from dace.transformation import pattern_matching
from dace.properties import Property, make_properties, SubsetProperty
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from sympy import Symbol

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

        # Step 1: Find all transients.
        transients = []
        self.memory_before = 0
        for a in sdfg.arrays:
            self.memory_before += sdfg.arrays[a].total_size
            if sdfg.arrays[a].transient == True:
                transients.append(a)
        print(transients)
        print(sdfg.arrays)

        live = {}
        # Step 2: Determine all possible reuses and decide on a mapping.
        for t in transients:
            live[t] = set()
            for n in state.nodes():
                if isinstance(n, nodes.AccessNode) and n.data == t:
                    live[t].add(n)
                    for e in state.all_edges(n):
                        for edge in state.memlet_tree(e):
                            if edge.src == n:
                                live[t].add(edge.dst)
                            if edge.dst == n:
                                live[t].add(edge.src)
        print(live)
        mapping = set()
        for t in live:
            for s in live:
                if not s is t and live[t].intersection(live[s]) == set():
                    mapping.add(frozenset([s,t]))
        print(mapping)


        '''mapping = []
        for n in state.nodes():
            if isinstance(n, nodes.AccessNode) and n.data in transients:
                predecessors = [state.predecessors(x) for x in state.predecessors(n)]
                predecessors = [item.data for sublist in predecessors for item in sublist]
                for t in transients:
                    if t not in predecessors and not n.data == t:
                        mapping.append((n.data, t))
                    else:
                        for m in mapping:
                            if m[0] == t:
                                mapping.remove(m)
                print(n, n.data)
                print(state.predecessors(n))
                print([state.predecessors(x) for x in state.predecessors(n)])
        print(mapping)

        for (old1, new1) in mapping:
            for (old2, new2) in mapping:
                if (old1, new1) == (old2,new2) or (new1, old1) == (old2, new2):
                    mapping.remove((old2, new2))
        print(mapping)'''
        # Step 3 For each mapping reshape transients to fit data

        # Step 4: For each mapping redirect edges and rename memlets in the whole tree.
        for (old, new) in mapping:
            old_node = state.find_node(old)
            old_node.data = new
            for e in state.all_edges(old_node):
                for edge in state.memlet_tree(e):
                    if edge.data.data == old:
                        edge.data.data = new
            sdfg.remove_data(old)
            for m in mapping:
                if old in m:
                    mapping.remove(m)

        # Analyze memory savings
        self.memory_after = 0
        for a in sdfg.arrays:
            self.memory_after += sdfg.arrays[a].total_size
        print('memory before: ', self.memory_before, 'B')
        print('memory after: ', self.memory_after, 'B')
        print('memory savings: ', self.memory_before- self.memory_after, 'B')
