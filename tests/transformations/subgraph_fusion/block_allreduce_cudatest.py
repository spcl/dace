# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from dace.transformation.dataflow import ReduceExpansion

from dace.libraries.standard.nodes.reduce import Reduce

N = dace.symbol('N')
M = dace.symbol('M')

@dace.program
def program(A: dace.float32[M, N]):
    return dace.reduce(lambda a, b: max(a, b), A, axis=1, identity=0)


@pytest.mark.gpu
def test_blockallreduce():
    A = np.random.rand(30, 30).astype(np.float32)
    sdfg = program.to_sdfg()
    sdfg.apply_gpu_transformations()

    graph = sdfg.nodes()[0]
    for node in graph.nodes():
        if isinstance(node, Reduce):
            reduce_node = node
    reduce_node.implementation = 'CUDA (device)'

    csdfg = sdfg.compile()
    result1 = csdfg(A=A, M=30, N=30)
    del csdfg

    cfg_id = 0
    state_id = 0
    subgraph = {ReduceExpansion.reduce: graph.node_id(reduce_node)}
    # expand first
    transform = ReduceExpansion()
    transform.setup_match(sdfg, cfg_id, state_id, subgraph, 0)
    transform.reduce_implementation = 'CUDA (block allreduce)'
    transform.apply(sdfg.node(0), sdfg)
    csdfg = sdfg.compile()
    result2 = csdfg(A=A, M=30, N=30)
    del csdfg

    print(np.linalg.norm(result1))
    print(np.linalg.norm(result2))
    assert np.allclose(result1, result2)


if __name__ == '__main__':
    test_blockallreduce()
