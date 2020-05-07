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
                G.add_edges_from([(n, x) for (y, x) in G.out_edges(state.exit_nodes(n)[0])])
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

        #maybe remove descendant kill nodes
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

        #generate garanteed live set for each node in G: itself and arrays of parents.
        live = {}
        for n in G:
            live[n] = set()
            if n.data in transients:
                live[n].add(n.data)
            live[n].update([x.data for x in G.predecessors(n) if x.data in transients])

        #import matplotlib.pyplot as plt
        #nx.draw_networkx(G, with_labels=True)
        #plt.show()

        #Do recursively from sink nodes: copmute max set for n
        def maxset(n):
            live_sum = sum([sdfg.arrays[x].total_size for x in live[n]])
            parent_set = set().union(*([maxset(p) for p in G.predecessors(n)]))
            parent_set_sum = sum([sdfg.arrays[x].total_size for x in parent_set])
            if live_sum >= parent_set_sum:
                maxset_n = live[n]
            else:
                maxset_n = parent_set
            return maxset_n

        for n in [node for node in G.nodes() if G.out_degree(node) == 0]:
            m = maxset(n)
            print(n, sum([sdfg.arrays[x].total_size for x in m]), memory_before, 100.0/memory_before*(memory_before - sum([sdfg.arrays[x].total_size for x in m])), m)

        '''arrays = list(transients)

        # find live variables for each node
        ## Initialize dead dict
        dead = {}
        for n in G.nodes():
            dead[n] = set()
        ## for each array
        for a in transients:
            gen, kill = alloc_dealloc[a]
            # for each ancestor of gen node add array to dead set
            for i in range(len(gen)):
                for anc in ancestors[gen[i]]:
                    dead[anc].add(a)
                dead[gen[i]].remove(a)
            # for each descendant of kill node add array to dead set.
            for i in range(len(kill)):
                for desc in nx.descendants(G, kill[i]):
                    dead[desc].add(a)

        for d in dead:
            dead[d] = transients - dead[d]

        # from live sets generate graph where two arrays are connected if never live together.
        Gp = nx.Graph()
        for t in transients:
            Gp.add_node(t)
        for t in transients:
            Gp.add_edges_from([(t,x) for x in Gp.nodes() if t != x])

        for d in dead:
            if d.data in transients:
                for b in dead[d]:
                    if d.data != b:
                        if Gp.has_edge(d.data,b):
                            Gp.remove_edge(d.data,b)

        import matplotlib.pyplot as plt
        nx.draw_networkx(Gp, with_labels=True)
        plt.show()

        # Now reduce arrays by selecting an edge and merging these arrays
        # (maybe reduce graph to be more simple)
        digraph = nx.DiGraph()
        for (a,b) in Gp.edges():
            if sdfg.arrays[a].total_size > sdfg.arrays[b].total_size:
                digraph.add_edge(a,b)

        import matplotlib.pyplot as plt
        nx.draw_networkx(digraph, with_labels=True)
        plt.show()
        # check for consitency and repeat.'''
        '''# Initialize dead dict
        dead = {}
        for n in G.nodes():
            dead[n] = set()
        # for each array
        for a in transients:
            gen, kill = alloc_dealloc[a]
            # for each ancestor of gen node add array to dead set
            for i in range(len(gen)):
                for anc in ancestors[gen[i]]:
                    dead[anc].add(a)
                dead[gen[i]].remove(a)
            # for each descendant of kill node add array to dead set.
            for i in range(len(kill)):
                for desc in nx.descendants(G, kill[i]):
                    dead[desc].add(a)

        print('dead: ', dead)
        for d in dead:
            dead[d] = transients - dead[d]

        print('live: ', dead)'''
