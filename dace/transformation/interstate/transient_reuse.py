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
                    mapping.add(tuple(sorted([s,t],key=transients.index)))
        print(mapping)

        #remove conflicting mappings, simplify mappings
        temp = set()
        for m in mapping:
            add = True
            for t in list(temp):
                if m is t: # there should be no duplicates!! shouldn't be necessary
                    add = False
                (tnew, told) = t
                (mnew, mold) = m
                if told == mold:
                    add = False
                elif told == mnew:
                    add = False
                    temp.add((tnew, mold))
                elif tnew == mold:
                    add = False
                    temp.add((mnew, told))
            if add == True:
                temp.add(m)
        mapping = temp

        print(mapping)
        # Step 3 For each mapping reshape transients to fit data

        # Step 4: For each mapping redirect edges and rename memlets in the whole tree.
        for (new, old) in list(mapping):
            for n in state.nodes():
                if isinstance(n, nodes.AccessNode) and n.data == old:
                    n.data = new
                    for e in state.all_edges(n):
                        for edge in state.memlet_tree(e):
                            if edge.data.data == old:
                                edge.data.data = new
            sdfg.remove_data(old)
        '''for (old, new) in list(mapping):
            print(old, new)
            try:
                old_node = state.find_node(old)
            except:
                continue
            old_node.data = new
            for e in state.all_edges(old_node):
                for edge in state.memlet_tree(e):
                    if edge.data.data == old:
                        edge.data.data = new
            remove_data = sdfg.remove_data(old)
            for m in list(mapping):
                if old in m:
                    mapping.remove(m)'''

        # Analyze memory savings
        self.memory_after = 0
        for a in sdfg.arrays:
            self.memory_after += sdfg.arrays[a].total_size
        print('memory before: ', self.memory_before, 'B')
        print('memory after: ', self.memory_after, 'B')
        print('memory savings: ', self.memory_before- self.memory_after, 'B')
