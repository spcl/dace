from dace import data, memlet, dtypes, registry, sdfg as sd
from dace.graph import nodes, nxutil, edges as ed
from dace.transformation import pattern_matching
from dace.properties import Property, make_properties, SubsetProperty
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from sympy import Symbol, N
import numpy as np
import networkx as nx

def _atomic_counter_generator():
    ctr = 0
    while True:
        ctr += 1
        yield ctr
_atomic_count = _atomic_counter_generator()

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
        for state in sdfg.nodes():
            transients = []
            memory_before = 0
            for a in sdfg.arrays:
                memory_before += sdfg.arrays[a].total_size
                if sdfg.arrays[a].transient == True:
                    transients.append(a)

            # Step 1: Define ancestors for all nodes:
            ancestors = {}
            successors = {}
            for n in state.nodes():
                ancestors[n] = set()
                if isinstance(n, nodes.AccessNode):
                    successors[n] = set()
                    
            G = nx.DiGraph()
            for n in state.nodes():
                G.add_node(n)
            for n in state.nodes():
                for e in state.all_edges(n):
                    G.add_edge(e.src, e.dst)

            for n in state.nodes():
                if not isinstance(n, nodes.AccessNode):
                    for p in G.predecessors(n):
                        for c in G.successors(n):
                            G.add_edge(p,c)
                    G.remove_node(n)

            for n in G.nodes():
                successors[n] = set(G.successors(n))
                ancestors[n] = set(nx.ancestors(G,n))

            import matplotlib.pyplot as plt
            nx.draw_networkx(G, with_labels=True)
            plt.show()

            mappings = {}
            for t in transients:
                mappings[t] = set()
            # Step 3: Find valid mappings
            for n in successors:
                for m in ancestors:
                    if isinstance(m, nodes.AccessNode) and successors[n].issubset(ancestors[m]) and \
                            sdfg.arrays[n.data].transient and sdfg.arrays[m.data].transient:
                        mappings[n.data].add(m.data)

            print("mappings: \n", mappings)

            # Step 4: find a final mapping
            buckets = []
            for i in range(len(transients)):
                buckets.append([])

            for n in transients:
                for i in range(len(mappings)):
                    if buckets[i] == []:
                        buckets[i].append(n)
                        break
                    temp = True
                    for j in range(len(buckets[i])):
                         temp = temp and n in mappings[buckets[i][j]] and sdfg.arrays[n].shape == sdfg.arrays[buckets[i][j]].shape
                    if temp:
                        buckets[i].append(n)
                        break
                    temp2 = True
                    for j in range(len(buckets[i])):
                        temp2 = temp2 and buckets[i][j] in mappings[n] and sdfg.arrays[n].shape == sdfg.arrays[buckets[i][j]].shape
                    if temp2:
                        buckets[i].insert(0, n)
                        break
            print(buckets)

            for i in range(len(transients)):
                if len(buckets[i]) > 1:
                    local_ctr = str(next(_atomic_count))
                    temp = sdfg.add_transient(name="transient_reuse_" + local_ctr, shape=sdfg.arrays[buckets[i][0]].shape,
                                              dtype=sdfg.arrays[buckets[i][0]].dtype)
                    buckets[i].insert(0, "transient_reuse_" + local_ctr)

            mapping = set()
            for i in range(len(buckets)):
                for j in range(1, len(buckets[i])):
                    mapping.add((buckets[i][0], buckets[i][j]))
            print(mapping)

            # Step 5: For each mapping redirect edges and rename memlets in the whole tree.
            for (new, old) in list(mapping):
                sdfg.arrays[new].shape = max(sdfg.arrays[new].shape, sdfg.arrays[old].shape)
                for n in state.nodes():
                    if isinstance(n, nodes.AccessNode) and n.data == old:
                        n.data = new
                        for e in state.all_edges(n):
                            for edge in state.memlet_tree(e):
                                if edge.data.data == old:
                                    edge.data.data = new
                sdfg.remove_data(old)

            # Analyze memory savings
            memory_after = 0
            for a in sdfg.arrays:
                memory_after += sdfg.arrays[a].total_size

            print('memory before: ', memory_before, 'B')
            print('memory after: ', memory_after, 'B')
            print('memory savings: ', memory_before - memory_after, 'B ,',
                  100 - N((100 / memory_before) * memory_after, 2), "%")