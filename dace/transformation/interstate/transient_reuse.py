from dace import registry, sdfg as sd
from dace.graph import nodes
from dace.transformation import pattern_matching
from dace.properties import make_properties
from sympy import N
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
    """ Implements the TransientReuse transformation.
        Finds all possible reuses of arrays,
        decides for a valid combination and changes sdfg accordingly.
    """

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

        memory_before = 0
        for a in sdfg.arrays:
            memory_before += sdfg.arrays[a].total_size

        for state in sdfg.nodes():

            # Copy the whole graph
            G = nx.DiGraph()
            for n in state.nodes():
                G.add_node(n)
            for n in state.nodes():
                for e in state.all_edges(n):
                    G.add_edge(e.src, e.dst)

            # Collapse all mappings and their scopes into one node
            scope_dict = state.scope_dict(node_to_children=True)
            for n in state.nodes():
                if isinstance(n, nodes.EntryNode):
                    for m in scope_dict[n]:
                        if isinstance(m, nodes.ExitNode):
                            G.add_edges_from([(n, x) for (y, x) in G.out_edges(m)])
                        G.remove_node(m)

            # Remove all nodes that are not AccessNodes or have incoming wcr edges
            # and connect their predecessors and successors
            for n in state.nodes():
                if n in G.nodes():
                    if not isinstance(n, nodes.AccessNode):
                        for p in G.predecessors(n):
                            for c in G.successors(n):
                                G.add_edge(p,c)
                        G.remove_node(n)
                    else:
                        for e in state.all_edges(n):
                            if e.data.wcr is not None:
                                for p in G.predecessors(n):
                                    for s in G.successors(n):
                                        G.add_edge(p,s)
                                G.remove_node(n)
                                break

            # Setup the ancestors and successors arrays
            ancestors = {}
            successors = {}
            for n in G.nodes():
                successors[n] = set(G.successors(n))
                ancestors[n] = set(nx.ancestors(G, n))

            # Find all transients and set up the mappings dict
            transients = set()
            mappings = {}
            for n in G.nodes():
                if sdfg.arrays[n.data].transient:
                    transients.add(n.data)
                    mappings[n.data] = set()

            # Find valid mappings. A mapping (n, m) is only valid if the successors of n
            # are a subset of the ancestors of m and n is also an ancestor of m.
            # Further the arrays have to be equivalent.
            # Assumption: Arrays are never used in two AccessNodes
            for n in G.nodes():
                for m in G.nodes():
                    n_array = sdfg.arrays[n.data]
                    m_array = sdfg.arrays[m.data]
                    if (n_array.transient and m_array.transient and n_array.is_equivalent(m_array)
                            and n in ancestors[m] and successors[n].issubset(ancestors[m])):
                        mappings[n.data].add(m.data)

            # Find a final mapping, greedy coloring algorithm to find a mapping.
            # Only add a transient to a bucket if either there is a mapping from it to
            # all other elements of that bucket or there is a mapping from each element in the bucket to it.
            buckets = []
            for i in range(len(transients)):
                buckets.append([])

            for n in transients:
                for i in range(len(transients)):
                    if buckets[i] == []:
                        buckets[i].append(n)
                        break

                    temp = True
                    for j in range(len(buckets[i])):
                        temp = (temp and n in mappings[buckets[i][j]])
                    if temp:
                        buckets[i].append(n)
                        break

                    temp2 = True
                    for j in range(len(buckets[i])):
                        temp2 = (temp2 and buckets[i][j] in mappings[n])
                    if temp2:
                        buckets[i].insert(0, n)
                        break

            # Build new custom transient to replace the other transients
            for i in range(len(buckets)):
                if len(buckets[i]) > 1:
                    local_ctr = str(next(_atomic_count))
                    array = sdfg.arrays[buckets[i][0]]
                    name = "transient_reuse_" + local_ctr
                    datadesc = array.clone()
                    sdfg.add_datadesc(name, datadesc)
                    buckets[i].insert(0, name)

            # Construct final mapping (transient_reuse_i, some_transient)
            mapping = set()
            for i in range(len(buckets)):
                for j in range(1, len(buckets[i])):
                    mapping.add((buckets[i][0], buckets[i][j]))

            # For each mapping redirect edges and rename memlets in the state
            for (new, old) in sorted(list(mapping)):
                for n in state.nodes():
                    if isinstance(n, nodes.AccessNode) and n.data == old:
                        n.data = new
                        for e in state.all_edges(n):
                            for edge in state.memlet_tree(e):
                                if edge.data.data == old:
                                    edge.data.data = new

        # clean up the arrays
        for a in list(sdfg.arrays):
            used = False
            for s in sdfg.states():
                for n in s.nodes():
                    if isinstance(n, nodes.AccessNode) and n.data == a:
                        used = True
                        break
            if not used:
                sdfg.remove_data(a)

        # Analyze memory savings and output them
        memory_after = 0
        for a in sdfg.arrays:
            memory_after += sdfg.arrays[a].total_size

        print('memory before: ', memory_before, 'B')
        print('memory after: ', memory_after, 'B')
        print('memory savings: ', memory_before - memory_after, 'B -',
              100 - N((100 / memory_before) * memory_after, 2), "%")
