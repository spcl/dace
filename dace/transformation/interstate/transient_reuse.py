from dace import data, memlet, dtypes, registry, sdfg as sd
from dace.graph import nodes, nxutil, edges as ed
from dace.transformation import pattern_matching
from dace.properties import Property, make_properties, SubsetProperty
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from sympy import Symbol, N
import numpy as np

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

            # Step 1: Find all transients.
            transients = []
            memory_before = 0
            for a in sdfg.arrays:
                memory_before += sdfg.arrays[a].total_size
                if sdfg.arrays[a].transient == True:
                    transients.append(a)
            print(len(sdfg.arrays))

            # Step 2: Determine all possible reuses and decide on a mapping.
            #   a) for each transient construct set of nodes that use that transient.
            live = {}
            for t in transients:
                live[t] = set()
                for n in state.nodes():
                    if isinstance(n, nodes.AccessNode) and n.data == t:
                        live[t].add(n)
                        for e in state.all_edges(n):
                            for edge in state.memlet_tree(e):
                                live[t].add(edge.dst)
                                live[t].add(edge.src)

            #   b) Find all possible mappings. Intersect all sets with each other.
            #      Only empty intersections are valid mappings. (Interference graph)
            mappings = {}
            for t in transients:
                mappings[t] = set()
            for t in live:
                for s in live:
                    if not s is t and live[t].intersection(live[s]) == set():
                        mappings[t].add(s)
                        mappings[s].add(t)

            #   c) Construct valid mapping combination. Each bucket contains transients that can be merged. (graph coloring)
            buckets = []
            for i in range(len(transients)):
                buckets.append(set())

            for t in mappings:
                for i in range(len(buckets)):
                    if buckets[i] == set():
                        buckets[i].add(t)
                        break
                    elif buckets[i].issubset(mappings[t]) and len(sdfg.arrays[t].shape) == len(sdfg.arrays[list(buckets[i])[0]].shape):  # only same shape
                        buckets[i].add(t)
                        break

            # Step 3 For each mapping reshape transients to fit data
            bucket_list = []
            for i in range(len(buckets)):
                bucket_list.append(sorted(list(buckets[i])))
                if len(buckets[i]) > 1:
                    local_ctr = str(next(_atomic_count))
                    shape = []
                    shape_list = [list(sdfg.arrays[e].shape) for e in bucket_list[i]]
                    for el in shape_list:
                        for j in range(len(el)):
                            if len(shape) == j:
                                shape.append(el[j])
                            else:
                                shape[j] = max(shape[j],el[j])
                    print("transient_reuse_" + local_ctr, shape)
                    temp = sdfg.add_transient(name="transient_reuse_" + local_ctr, shape=shape, dtype=sdfg.arrays[bucket_list[i][0]].dtype)
                    bucket_list[i].insert(0, "transient_reuse_" + local_ctr)

            mapping = set()
            for i in range(len(bucket_list)):
                for j in range(1, len(bucket_list[i])):
                    mapping.add((bucket_list[i][0], bucket_list[i][j]))


            # Step 4: For each mapping redirect edges and rename memlets in the whole tree.
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

            print(len(sdfg.arrays))
            print('memory before: ', memory_before, 'B')
            print('memory after: ', memory_after, 'B')
            print('memory savings: ', memory_before- memory_after, 'B ,', 100-N((100/memory_before)*memory_after, 2), "%")

    def payoff(shapes):
        numpy = np.array(shapes)
        prod = np.prod(numpy)
        return np.prod(numpy) < np.sum(numpy)
