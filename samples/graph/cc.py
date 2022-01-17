# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

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
    args = vars(parser.parse_args())

    E.set(args['edges'])
    V.set(args['vertices'])

    print('Connected Components (Shiloach-Vishkin) E=%d, V=%d' % (E.get(), V.get()))

    graph = nx.gnm_random_graph(V.get(), E.get(), seed=args['seed'])

    comp = np.arange(0, V.get(), dtype=np.uint64)
    EL = dace.ndarray([2 * E, 2], dace.uint64)
    EL[:E.get()] = np.array([[u, v] for u, v, d in nx.to_edgelist(graph)], dtype=np.uint64)
    EL[E.get():] = np.array([[v, u] for u, v, d in nx.to_edgelist(graph)], dtype=np.uint64)

    shiloach_vishkin(EL, comp, E=E, V=V)

    cc = nx.number_connected_components(graph)
    diff = abs(cc - len(np.unique(comp)))
    print("Difference:", diff, '(SV:', len(np.unique(comp)), ', NX:', cc, ')')
    print("==== Program end ====")
    exit(0 if diff == 0 else 1)
