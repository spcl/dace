# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict, Iterator, Optional, Set, Tuple

import networkx as nx

from dace import SDFG, SDFGState, data
from dace import sdfg as sd
from dace.properties import make_properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


class TransientReuse(ppl.Pass):
    """
    Reduces memory consumption by reusing allocated transient array memory. Only modifies arrays that can safely be
    reused.
    """
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If states changed
        return modified & (ppl.Modifies.Nodes | ppl.Modifies.Memlets)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[str]]:
        result: Set[str] = set()

        memory_before = 0
        arrays = {}
        for a in sdfg.arrays:
            memory_before += sdfg.arrays[a].total_size * sdfg.arrays[a].dtype.bytes
            if sdfg.arrays[a].transient:
                arrays[a] = 0

        # only consider transients appearing in one single state
        for state in sdfg.states():
            for a in state.all_transients():
                arrays[a] += 1

        transients = set()
        for a in arrays:
            if arrays[a] == 1:
                transients.add(a)

        for state in sdfg.nodes():
            # Copy the whole graph
            G = nx.MultiDiGraph()
            for n in state.nodes():
                G.add_node(n)
            for n in state.nodes():
                for e in state.all_edges(n):
                    G.add_edge(e.src, e.dst)

            # Collapse all mappings and their scopes into one node
            scope_children = state.scope_children()
            for n in scope_children[None]:
                if isinstance(n, nodes.EntryNode):
                    G.add_edges_from([(n, x) for (y, x) in G.out_edges(state.exit_node(n))])
                    G.remove_nodes_from(scope_children[n])

            # Remove all nodes that are not AccessNodes or have incoming wcr edges
            # and connect their predecessors and successors
            for n in state.nodes():
                if n in G.nodes():
                    if not isinstance(n, nodes.AccessNode):
                        for p in G.predecessors(n):
                            for c in G.successors(n):
                                G.add_edge(p, c)
                        G.remove_node(n)
                    else:
                        for e in state.all_edges(n):
                            if e.data.wcr is not None:
                                for p in G.predecessors(n):
                                    for s in G.successors(n):
                                        G.add_edge(p, s)
                                G.remove_node(n)
                                break

            # Setup the ancestors and successors arrays as well as the mappings dict
            ancestors = {}
            successors = {}
            for n in G.nodes():
                successors[n] = set(G.successors(n))
                ancestors[n] = set(nx.ancestors(G, n))
            mappings = {}
            for n in transients:
                mappings[n] = set()

            # Find valid mappings. A mapping (n, m) is only valid if the successors of n
            # are a subset of the ancestors of m and n is also an ancestor of m.
            # Further the arrays have to be equivalent.
            for n in G.nodes():
                for m in G.nodes():
                    if n is not m and n.data in transients and m.data in transients:
                        if n.data == m.data:
                            transients.remove(n.data)
                        if (sdfg.arrays[n.data].is_equivalent(sdfg.arrays[m.data]) and n in ancestors[m]
                                and successors[n].issubset(ancestors[m])):
                            mappings[n.data].add(m.data)

            # Find a final mapping, greedy coloring algorithm to find a mapping.
            # Only add a transient to a bucket if either there is a mapping from it to
            # all other elements of that bucket or there is a mapping from each element in the bucket to it.
            buckets = []
            for i in range(len(transients)):
                buckets.append([])

            for n in transients:
                for i in range(len(transients)):
                    if not buckets[i]:
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
                    array = sdfg.arrays[buckets[i][0]]
                    name = sdfg.add_datadesc("transient_reuse", array.clone(), find_new_name=True)
                    buckets[i].insert(0, name)

            # Construct final mapping (transient_reuse_i, some_transient)
            mapping = set()
            for i in range(len(buckets)):
                for j in range(1, len(buckets[i])):
                    mapping.add((buckets[i][0], buckets[i][j]))

            # For each mapping redirect edges and rename memlets in the state
            for (new, old) in sorted(list(mapping)):
                result.add(old)
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
                sdfg.remove_data(a, validate=False)

        # Analyze memory savings and output them
        memory_after = 0
        for a in sdfg.arrays:
            memory_after += sdfg.arrays[a].total_size * sdfg.arrays[a].dtype.bytes

        print('memory before: ', memory_before, 'B')
        print('memory after: ', memory_after, 'B')
        print('memory savings: ', memory_before - memory_after, 'B')
        return result or None
