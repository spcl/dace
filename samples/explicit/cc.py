# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Sample showing the Shiloach-Viskin pointer-chasing connected components graph algorithm in the explicit DaCe syntax.
It showcases write-conflicting accesses, location constraints, and explicit data movement volume.
"""
import argparse
import dace
import numpy as np
import networkx as nx

E = dace.symbol('E')
V = dace.symbol('V')


@dace.program
def shiloach_vishkin(EL: dace.uint64[2 * E, 2], comp: dace.uint64[V]):
    """
    The Shiloach-Vishkin algorithm [1] has two steps that run consecutively: Hook and Compress.
    Hook connects two components together, whereas Compress condenses ``comp`` to become a tree of depth-1 ("star").
    The process runs until there are no more components to connect, and ends with ``comp`` pointing each vertex to
    its component ID, designated by the smallest vertex index contained within it.

    [1] Y. Shiloach and U. Vishkin. An O(log N) parallel connectivity algorithm. Journal of Algorithms, 3:57-67, 1982.

    :param EL: The graph represented as an edge list.
    :param comp: The component parent-pointing tree.
    """
    flag_hook = np.ones([1], np.int32)

    for v in dace.map[0:V]:
        with dace.tasklet:
            out >> comp[v]
            out = v

    while flag_hook[0]:
        with dace.tasklet:
            out >> flag_hook
            out = 0

        # Hook in parallel. Notice that writing to `comp` may have conflicts, but the last access will "win" and set
        # the flag.
        for e in dace.map[0:2 * E]:
            with dace.tasklet:
                u << EL[e, 0]
                v << EL[e, 1]
                parents << comp(3)[:]  # The data movement volume is known (3), but not the location
                out >> comp(1)[:]
                f >> flag_hook(-1)

                pu = parents[u]
                pv = parents[v]
                ppv = parents[pv]

                if pu < pv and pv == ppv:
                    out[ppv] = pu
                    f = 1

        # Compress uses the "multi-jump" version, where the tasklet keeps accessing `comp` in a loop
        # until reaching the top of the parent-pointing auxiliary data structure. This will always point
        # backwards, so we introduce this as a location constraint "hint" in the memlet.
        for v in dace.map[0:V]:
            with dace.tasklet:
                inp << comp(-1)[0:v + 1]  # The volume is unknown, but the location constraints are known
                out >> comp(-1)[v]

                p = inp[v]
                pp = inp[p]
                while p != pp:
                    out = pp
                    p = pp
                    pp = inp[p]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("edges", type=int, nargs="?", default=17)
    parser.add_argument("vertices", type=int, nargs="?", default=16)
    parser.add_argument("-seed", type=int, nargs="?", default=None)
    args = parser.parse_args()

    E = args.edges
    V = args.vertices

    print(f'Connected Components (Shiloach-Vishkin) E={E}, V={V}')

    # Create a random graph and use it to create an edge list (EL)
    graph = nx.gnm_random_graph(V, E, seed=args.seed)
    EL = np.ndarray([2 * E, 2], np.uint64)
    EL[:E] = np.array([[u, v] for u, v, d in nx.to_edgelist(graph)], dtype=np.uint64)
    EL[E:] = np.array([[v, u] for u, v, d in nx.to_edgelist(graph)], dtype=np.uint64)

    # Initialize list of components with every node having its own component
    comp = np.arange(0, V, dtype=np.uint64)

    # Call program
    shiloach_vishkin(EL, comp, E=E, V=V)

    # Verify correctness
    cc = nx.number_connected_components(graph)
    diff = abs(cc - len(np.unique(comp)))
    print("Difference:", diff, '(dace-sv:', len(np.unique(comp)), ', networkx:', cc, ')')
    exit(0 if diff == 0 else 1)
