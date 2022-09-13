# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import os
import dace
from dace.transformation.dataflow import MapFusion


@dace.program
def fusion(A: dace.float32[10, 20], B: dace.float32[10, 20], out: dace.float32[1]):
    tmp = dace.define_local([10, 20], dtype=A.dtype)
    tmp_2 = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]

            b = a * a

    for i, j in dace.map[0:20, 0:10]:
        with dace.tasklet:
            a << tmp[j, i]
            b << B[j, i]
            c >> tmp_2[j, i]

            c = a + b

    for m, n in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << tmp_2[m, n]
            b >> out(1, lambda a, b: a + b)[0]

            b = a


@dace.program
def multiple_fusions(A: dace.float32[10, 20], B: dace.float32[10, 20], C: dace.float32[10, 20], out: dace.float32[1]):
    A_prime = dace.define_local([10, 20], dtype=A.dtype)
    A_prime_copy = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            inp << A[i, j]
            out1 >> out(1, lambda a, b: a + b)[0]
            out2 >> A_prime[i, j]
            out3 >> A_prime_copy[i, j]
            out1 = inp
            out2 = inp * inp
            out3 = inp * inp

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            inp << A_prime[i, j]
            out1 >> B[i, j]
            out1 = inp + 1

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            inp << A_prime_copy[i, j]
            out2 >> C[i, j]
            out2 = inp + 2


@dace.program
def fusion_chain(A: dace.float32[10, 20], B: dace.float32[10, 20]):
    tmp1 = A * 2
    tmp2 = tmp1 * 4
    B[:] = tmp2 + 5


def test_fusion_simple():
    sdfg = fusion.to_sdfg()
    sdfg.save(os.path.join('_dacegraphs', 'before1.sdfg'))
    sdfg.simplify()
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.save(os.path.join('_dacegraphs', 'after1.sdfg'))

    A = np.random.rand(10, 20).astype(np.float32)
    B = np.random.rand(10, 20).astype(np.float32)
    out = np.zeros(shape=1, dtype=np.float32)
    sdfg(A=A, B=B, out=out)

    diff = abs(np.sum(A * A + B) - out)
    print('Difference:', diff)
    assert diff <= 1e-3


def test_multiple_fusions():
    sdfg = multiple_fusions.to_sdfg()
    num_nodes_before = len([node for state in sdfg.nodes() for node in state.nodes()])

    sdfg.save(os.path.join('_dacegraphs', 'before2.sdfg'))
    sdfg.simplify()
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.save(os.path.join('_dacegraphs', 'after2.sdfg'))

    num_nodes_after = len([node for state in sdfg.nodes() for node in state.nodes()])
    # Ensure that the number of nodes was reduced after transformation
    if num_nodes_after >= num_nodes_before:
        raise RuntimeError('SDFG was not properly transformed '
                           '(nodes before: %d, after: %d)' % (num_nodes_before, num_nodes_after))

    A = np.random.rand(10, 20).astype(np.float32)
    B = np.zeros_like(A)
    C = np.zeros_like(A)
    out = np.zeros(shape=1, dtype=np.float32)
    sdfg(A=A, B=B, C=C, out=out)
    diff1 = np.linalg.norm(A * A + 1 - B)
    diff2 = np.linalg.norm(A * A + 2 - C)
    print('Difference1:', diff1)
    assert diff1 <= 1e-4

    print('Difference2:', diff2)
    assert diff2 <= 1e-4


def test_fusion_chain():
    sdfg = fusion_chain.to_sdfg()
    sdfg.save(os.path.join('_dacegraphs', 'before3.sdfg'))
    sdfg.simplify()
    sdfg.apply_transformations(MapFusion)
    num_nodes_before = len([node for state in sdfg.nodes() for node in state.nodes()])
    sdfg.apply_transformations(MapFusion)
    sdfg.apply_transformations(MapFusion)
    sdfg.save(os.path.join('_dacegraphs', 'after3.sdfg'))

    num_nodes_after = len([node for state in sdfg.nodes() for node in state.nodes()])
    # Ensure that the number of nodes was reduced after transformation
    if num_nodes_after >= num_nodes_before:
        raise RuntimeError('SDFG was not properly transformed '
                           '(nodes before: %d, after: %d)' % (num_nodes_before, num_nodes_after))

    A = np.random.rand(10, 20).astype(np.float32)
    B = np.zeros_like(A)
    sdfg(A=A, B=B)
    diff = np.linalg.norm(A * 8 + 5 - B)
    print('Difference:', diff)
    assert diff <= 1e-4


@dace.program
def fusion_with_transient(A: dace.float64[2, 20]):
    res = np.ndarray([2, 20], dace.float64)
    for i in dace.map[0:20]:
        for j in dace.map[0:2]:
            with dace.tasklet:
                a << A[j, i]
                t >> res[j, i]
                t = a * a
    for i in dace.map[0:20]:
        for j in dace.map[0:2]:
            with dace.tasklet:
                t << res[j, i]
                o >> A[j, i]
                o = t * 2


def test_fusion_with_transient():
    A = np.random.rand(2, 20)
    expected = A * A * 2
    sdfg = fusion_with_transient.to_sdfg()
    sdfg.simplify()
    sdfg.apply_transformations(MapFusion)
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_fusion_with_inverted_indices():

    @dace.program
    def inverted_maps(A: dace.int32[10]):
        B = np.empty_like(A)
        for i in dace.map[0:10]:
            B[i] = i
        for i in dace.map[0:10]:
            A[9-i] = B[9-i] + 5
    
    ref = np.arange(5, 15, dtype=np.int32)

    sdfg = inverted_maps.to_sdfg(simplify=True)
    val0 = np.ndarray((10,), dtype=np.int32)
    sdfg(A=val0)
    assert(np.array_equal(val0, ref))

    sdfg.apply_transformations(MapFusion)
    val1 = np.ndarray((10,), dtype=np.int32)
    sdfg(A=val1)
    assert(np.array_equal(val1, ref))


def test_fusion_with_empty_memlet():

    N = dace.symbol('N', positive=True)

    @dace.program
    def inner_product(A: dace.float32[N], B: dace.float32[N], out: dace.float32[1]):
        tmp = np.empty_like(A)
        for i in dace.map[0:N:128]:
            for j in dace.map[0:128]:
                tmp[i+j] = A[i+j] * B[i+j]
        for i in dace.map[0:N:128]:
            lsum = dace.float32(0)
            for j in dace.map[0:128]:
                lsum = lsum + tmp[i+j]
            out[0] += lsum
    
    sdfg = inner_product.to_sdfg(simplify=True)
    count = sdfg.apply_transformations_repeated(MapFusion)
    assert(count == 2)

    A = np.arange(1024, dtype=np.float32)
    B = np.arange(1024, dtype=np.float32)
    val = np.zeros((1,), dtype=np.float32)
    sdfg(A=A, B=B, out=val, N=1024)
    ref = A @ B
    assert(np.allclose(val[0], ref))


if __name__ == '__main__':
    test_fusion_simple()
    test_multiple_fusions()
    test_fusion_chain()
    test_fusion_with_transient()
    test_fusion_with_inverted_indices()
    test_fusion_with_empty_memlet()
