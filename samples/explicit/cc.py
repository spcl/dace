# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Sample showing the Shiloach-Viskin pointer-chasing connected components algorithm in the
explicit DaCe syntax.
"""
import argparse
import dace
import numpy as np
import networkx as nx

E = dace.symbol('E')
V = dace.symbol('V')

@dace.program
def shiloach_vishkin(EL, comp):
    flag_hook = dace.define_local_scalar(dace.int32)

    with dace.tasklet:
        out >> flag_hook
        out = 1

    for v in dace.map[0:V]:
        with dace.tasklet:
            out >> comp[v]
            out = v

    while flag_hook:
        with dace.tasklet:
            out >> flag_hook
            out = 0

        for e in dace.map[0:2 * E]:
            with dace.tasklet:
                u << EL[e, 0]
                v << EL[e, 1]
                parents << comp(3)[:]
                out >> comp(1)[:]
                f >> flag_hook(-1)

                pu = parents[u]
                pv = parents[v]
                ppv = parents[pv]

                if pu < pv and pv == ppv:
                    out[ppv] = pu
                    f = 1

        # Multi-jump version
        for v in dace.map[0:V]:
            with dace.tasklet:
                inp << comp(-1)[0:v + 1]
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
