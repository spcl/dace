import dace
import numpy as np

from dace.transformation.subgraph import ReduceExpansion

from dace.libraries.standard.nodes.reduce import Reduce

N = dace.symbol('N')
M = dace.symbol('M')
N.set(300); M.set(300)

@dace.program
def TEST(A: dace.float32[M,N]):
    return dace.reduce(lambda a, b: max(a,b), A, axis=1, identity = 0)


def test():
    A = np.random.rand(M.get(), N.get()).astype(np.float32)
    sdfg = TEST.to_sdfg()
    sdfg.apply_gpu_transformations()

    graph = sdfg.nodes()[0]
    for node in graph.nodes():
        if isinstance(node, Reduce):
            reduce_node = node
    reduce_node.implementation = 'CUDA (device)'

    csdfg = sdfg.compile()
    result1 = csdfg(A=A,M=M,N=N)

    sdfg_id = 0
    state_id = 0
    subgraph = {ReduceExpansion._reduce: graph.nodes().index(reduce_node)}
    # expand first
    transform = ReduceExpansion(sdfg_id, state_id, subgraph, 0)
    transform.reduce_implementation = 'CUDA (block allreduce)'
    transform.apply(sdfg)
    csdfg = sdfg.compile()
    result2 = csdfg(A=A,M=M,N=N)

    assert np.allclose(result1, result2)

    print("PASS")


if __name__ == '__main__':
    test()
