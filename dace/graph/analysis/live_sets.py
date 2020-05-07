import networkx as nx
from dace import nodes

def live_sets(sdfg):
    """

    :param sdfg:
    :return: A dict holding for each array a tuple of lists.
    The first list of the tuple holds the nodes where allocation should happen.
    The second list holds the (name of?) nodes where deallocation can happen.
    If array appears in multiple states dict holds Empty lists.
    Further a number denoting the max. size of memory needed for allocating transients in bytes.
    """

    #initialize arrays and determine shared transients
    alloc_dealloc = {}

    for a in sdfg.transients():
        alloc_dealloc[a] = ([], [])

    memory_before = 0
    arrays = {}
    for a in sdfg.transients():
        arrays[a] = 0
        memory_before += sdfg.arrays[a].total_size

    for state in sdfg.states():
        for a in state.all_transients():
            arrays[a] += 1

    transients = set()
    for a in arrays:
        if arrays[a] == 1:
            transients.add(a)

    #for each state
    for state in sdfg.states():
        # build graph
        G = nx.DiGraph()
        for n in state.nodes():
            G.add_node(n)
        for n in state.nodes():
            for e in state.all_edges(n):
                G.add_edge(e.src, e.dst)

        # Collapse Maps
        scope_dict = state.scope_dict(node_to_children=True)
        for n in scope_dict:
            if n is not None:
                G.add_edges_from([(n, x) for (y, x) in G.out_edges(state.exit_node(n))])
                G.remove_nodes_from(scope_dict[n])

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

        # Setup the ancestors and successors arrays
        ancestors = {}
        successors = {}
        for n in G.nodes():
            successors[n] = set(G.successors(n))
            ancestors[n] = set(nx.ancestors(G, n))
            ancestors[n].add(n)

        #### KILL NODES
        # Find the kill nodes for each array
        for n in G.nodes():
            if n.data in transients:
                alloc_dealloc[n.data][0].append(n)
                for m in G.nodes():
                    if n in ancestors[m] and successors[n].issubset(ancestors[m]):
                        alloc_dealloc[n.data][1].append(m)

        # maybe remove descendant kill nodes
        for a in alloc_dealloc:
            kill = alloc_dealloc[a][1].copy()
            for i in range(len(kill)):
                for j in range(len(kill)):
                    if kill[i] in ancestors[kill[j]] and kill[i] != kill[j]:
                        try:
                            alloc_dealloc[a][1].remove(kill[j])
                        except:
                            continue

        #### MAX LIVE SET

        import matplotlib.pyplot as plt
        nx.draw_networkx(G, with_labels=True)
        plt.show()

        # Get longest path
        longest_path = nx.dag_longest_path(G)

        # Generate levels (maybe remove non-transients)
        levels = {}
        node_level = {}

        for i in range(len(longest_path)):
            levels[i] = set()
        for n in G.nodes():
            node_level[n] = float("inf")
        for s in [node for node in G.nodes() if G.in_degree(node) == 0]:
            levels[0].add(s)
            node_level[s] = 0



        for l in range(0, len(longest_path)):
            for n in levels[l]:
                for s in G.successors(n):
                    if all([node_level[x] < l+1 for x in G.predecessors(s)]):
                        levels[l+1].add(s)
                        node_level[s] = l
        print('levels: ', levels)

        # Find live set for each level
        
        # Take maximum of those as maximum live set