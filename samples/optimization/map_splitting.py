# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
from dace import optimization as optim
from dace.sdfg.graph import SubgraphView
import numpy as np

from dace.transformation.subgraph.split_maps import SplitMaps

N = 256


@dace.program
def sample(A, B, C):
    for j, i in dace.map[0:N, 0:N]:
        with dace.tasklet:
            a << A[i,j]
            b >> B[i, j]

            c = math.exp(a)

    for i, j in dace.map[0:N / 2, 0:N / 2]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]

            c = math.log(a)


if __name__ == '__main__':
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.random.rand(N, N)

    sdfg = sample.to_sdfg(A, B, C)

    state = sdfg.start_state
    subgraph = SubgraphView(state._graph, subgraph_nodes=state.nodes())
    split = SplitMaps(subgraph, sdfg_id=sdfg.sdfg_id, state_id=sdfg.node_id(state))

    if split.can_be_applied(sdfg, subgraph):
        split.apply(sdfg, subgraph)
