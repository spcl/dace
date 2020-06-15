import networkx as nx
from dace import nodes


def live_sets(sdfg):
    """ Finds the maximum live set of a sdfg and allocation/deallocation points for all its
        transients.
    :param sdfg: A SDFG.
    :return: A dict holding the allocation/deallocation nodes for each transient.
             A tuple holding the maximum live set and its size in bytes
             A dict holding the maximum live set for each state, without shared transients,
             and its size.
             A set holding all static transients.
    """

    # Initialize arrays.
    memory_before = 0
    arrays = {}
    shared_transients = set()
    transients = set()
    alloc_dealloc_states = {}
    maximum_live_set_states = {}

    # Determine static transients.
    for a in sdfg.transients():
        arrays[a] = 0
        memory_before += sdfg.arrays[a].total_size
    for state in sdfg.states():
        for a in state.all_transients():
            arrays[a] += 1
    for a in arrays:
        if arrays[a] == 1:
            transients.add(a)
        if arrays[a] != 1:
            shared_transients.add(a)

    # Determine the maximum live set and the allocation/deallocation table for each state.
    for state in sdfg.states():

        # Copy the state graph to build the proxy graph.
        G = nx.DiGraph()
        for n in state.nodes():
            G.add_node(n)
        for n in state.nodes():
            for e in state.all_edges(n):
                G.add_edge(e.src, e.dst)

        # Collapse all maps into one node.
        scope_dict = state.scope_dict(node_to_children=True)
        for n in scope_dict:
            if n is not None:
                G.add_edges_from([(n, x) for (y, x) in G.out_edges(state.exit_node(n))])
                for m in scope_dict[n]:
                    if isinstance(m, nodes.AccessNode) and m.data in transients:
                        shared_transients.add(m.data)
                        transients.remove(m.data)
                    G.remove_node(m)

        G_collapsed = G.copy()

        # Remove all nodes that are not AccessNodes and connect their predecessors and successors.
        for n in state.nodes():
            if n in G.nodes():
                if not isinstance(n, nodes.AccessNode):
                    for p in G.predecessors(n):
                        for c in G.successors(n):
                            G.add_edge(p, c)
                    G.remove_node(n)

        # Setup the ancestors and successors arrays.
        ancestors = {}
        successors = {}
        for n in G.nodes():
            successors[n] = set(G.successors(n))
            ancestors[n] = set(nx.ancestors(G, n))
            ancestors[n].add(n)

        ############
        # KILL NODES

        alloc_dealloc_nodes = {}
        for n in G:
            alloc_dealloc_nodes[n] = ([], [])

        # Find the kill nodes for each array.
        alloc_dealloc = {}
        for a in transients:
            alloc_dealloc[a] = ([], [])
        for n in G.nodes():
            if n.data in transients:
                alloc_dealloc[n.data][0].append(n)
                alloc_dealloc_nodes[n][0].append(n.data)
                for op in G.predecessors(n):
                    alloc_dealloc_nodes[n][0].remove(n.data)
                    alloc_dealloc_nodes[op][0].append(n.data)
                    break
                for m in G.nodes():
                    if n in ancestors[m] and successors[n].issubset(ancestors[m]):
                        alloc_dealloc[n.data][1].append(m)

        # Remove kill nodes which are descendants of other kill nodes.
        for a in alloc_dealloc:
            kill = alloc_dealloc[a][1].copy()
            for i in range(len(kill)):
                for j in range(len(kill)):
                    if (kill[i] in ancestors[kill[j]]
                            and kill[i] in ancestors[kill[j]]
                            and kill[j] in alloc_dealloc[a][1]
                            and kill[i] != kill[j]):
                        alloc_dealloc[a][1].remove(kill[j])
            alloc_dealloc_nodes[alloc_dealloc[a][1][0]][1].append(a)
        alloc_dealloc_states[state] = alloc_dealloc_nodes

        ##############
        # MAX LIVE SET

        # Get longest path in the DAG. And continue if it is zero.
        longest_path = nx.dag_longest_path(G)
        if len(longest_path) == 0:
            maximum_live_set_states[state] = ([], 0)
            continue

        # Generate node levels.
        levels = []
        node_level = {}
        for i in range(len(longest_path)):
            levels.append(set())
        for n in G.nodes():
            node_level[n] = float("inf")
        for s in [node for node in G.nodes() if G.in_degree(node) == 0]:
            levels[0].add(s)
            node_level[s] = 0
        for l in range(0, len(levels)):
            for n in levels[l]:
                for s in G.successors(n):
                    if all([node_level[x] < l+1 for x in G.predecessors(s)]):
                        levels[l+1].add(s)
                        node_level[s] = l+1

        # Generate transient levels.
        # Each transient is once in the level of its allocation node and of its deallocation node.
        # If alloc is empty it is added to level 0.
        # If dealloc is empty it is added to the last level.
        transient_levels = []
        for i in range(len(levels)):
            transient_levels.append(set())
        for t in transients:
            alloc, dealloc = alloc_dealloc[t]  # TODO: maybe change to alloc_dealloc_states or alloc_dealloc_nodes
            if not alloc:
                transient_levels[0].add(t)
            else:
                transient_levels[node_level[alloc[0]]].add(t)
            if not dealloc:
                transient_levels[len(levels)-1].add(t)
            else:
                transient_levels[node_level[dealloc[0]]].add(t)

        # Generate the maximum live set by one pass through the levels.
        liveSet = set()
        size = 0
        maxLiveSet = set()
        maxSize = 0
        for l in range(len(transient_levels)):
            new = transient_levels[l] - liveSet
            old = liveSet.intersection(transient_levels[l])
            # Add new occuring arrays to LiveSet of this level.
            for t in new:
                liveSet.add(t)
                size += sdfg.arrays[t].total_size
            # Compare LiveSet of this level to maxLiveSet
            if size > maxSize:
                maxSize = size
                maxLiveSet = liveSet.copy()
            # Remove arrays that appeared a second time, marking end of their liveness.
            for t in old:
                liveSet.remove(t)
                size -= sdfg.arrays[t].total_size
        maximum_live_set_states[state] = (maxLiveSet, maxSize)

    # Generate maximum live set over all states.  Add static transients
    # and union all maximum live set of the states and add all sizes together.
    maximum_live_set = [set(), 0]
    for t in shared_transients:
        maximum_live_set[0].add(t)
        maximum_live_set[1] += sdfg.arrays[t].total_size
    for state in maximum_live_set_states:
        liveSet = maximum_live_set_states[state][0]
        size = maximum_live_set_states[state][1]
        maximum_live_set[0] = maximum_live_set[0].union(liveSet)
        maximum_live_set[1] += size

    return alloc_dealloc_states, maximum_live_set, maximum_live_set_states, shared_transients
