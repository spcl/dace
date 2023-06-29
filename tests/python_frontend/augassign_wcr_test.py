# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def augassign_wcr(A: dace.int32[10, 10, 10], B: dace.int32[10], W: dace.bool_[10]):
    count = 0
    for i in dace.map[0:10]:
        B[i] = 0
        if W[i] is not False:
            count += 1
            for j in range(10):
                for k in range(10):
                    B[i] += A[i, j, k]

    return count


@dace.program
def augassign_wcr2(A: dace.int32[10, 10, 10], B: dace.int32[10], W: dace.bool_[10, 10, 10]):
    count = 0
    B[:] = 0
    for i, j, k in dace.map[0:10, 0:10, 0:10]:
        if W[i, j, k]:
            count += 1
            B[i] += A[i, j, k]

    return count


@dace.program
def augassign_wcr3(A: dace.int32[10, 10, 10], B: dace.int32[10], W: dace.bool_[10, 10, 10], ind: dace.int32[10]):
    count = 0
    B[:] = 0
    for i, j, k in dace.map[0:10, 0:10, 0:10]:
        if W[i, j, k]:
            count += 1
            B[ind[i]] += A[i, j, k]

    return count


@dace.program
def augassign_wcr4():
    a = np.zeros((10,))
    for i in dace.map[1:9]:
        a[i-1] += 1
        a[i] += 2
        a[i+1] += 3
    return a


def test_augassign_wcr():
    A = np.random.randint(1, 10, size=(10, 10, 10), dtype=np.int32)
    B = np.empty((10, ), dtype=np.int32)
    W = np.random.randint(2, size=(10, ), dtype=np.bool_)

    with dace.config.set_temporary('frontend', 'avoid_wcr', value=True):
        test_sdfg = augassign_wcr.to_sdfg(simplify=False)
    wcr_count = 0
    for sdfg in test_sdfg.sdfg_list:
        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.data.wcr:
                    wcr_count += 1
    assert (wcr_count == 1)

    count = test_sdfg(A=A, B=B, W=W)
    assert (count[0] == np.count_nonzero(W))
    assert (np.array_equal(np.add.reduce(A, axis=(1, 2))[W], B[W]))


def test_augassign_wcr2():
    A = np.random.randint(1, 10, size=(10, 10, 10), dtype=np.int32)
    B = np.empty((10, ), dtype=np.int32)
    C = np.zeros((10, ), dtype=np.int32)
    W = np.random.randint(2, size=(10, 10, 10), dtype=np.bool_)

    with dace.config.set_temporary('frontend', 'avoid_wcr', value=True):
        test_sdfg = augassign_wcr2.to_sdfg(simplify=False)
    wcr_count = 0
    for sdfg in test_sdfg.sdfg_list:
        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.data.wcr:
                    wcr_count += 1
    assert (wcr_count == 2)

    count = test_sdfg(A=A, B=B, W=W)
    C = np.add.reduce(A, axis=(1, 2), where=W)
    assert (count[0] == np.count_nonzero(W))
    assert (np.array_equal(B, C))


def test_augassign_wcr3():
    A = np.random.randint(1, 10, size=(10, 10, 10), dtype=np.int32)
    B = np.empty((10, ), dtype=np.int32)
    C = np.zeros((10, ), dtype=np.int32)
    D = np.zeros((10, ), dtype=np.int32)
    ind = np.random.randint(0, 10, size=(10, ), dtype=np.int32)
    W = np.random.randint(2, size=(10, 10, 10), dtype=np.bool_)

    with dace.config.set_temporary('frontend', 'avoid_wcr', value=True):
        test_sdfg = augassign_wcr3.to_sdfg(simplify=False)
    wcr_count = 0
    for sdfg in test_sdfg.sdfg_list:
        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.data.wcr:
                    wcr_count += 1
    assert (wcr_count == 2)

    count = test_sdfg(A=A, B=B, W=W, ind=ind)
    C = np.add.reduce(A, axis=(1, 2), where=W)
    for i in range(10):
        D[ind[i]] += C[i]
    assert (count[0] == np.count_nonzero(W))
    assert (np.array_equal(B, D))


def test_augassign_no_wcr():
    @dace.program
    def no_wcr(A: dace.int32[5, 5, 5]):
        A[2, 3, :] += A[3, 2, :]

    with dace.config.set_temporary('frontend', 'avoid_wcr', value=True):
        sdfg = no_wcr.to_sdfg(simplify=False)
    for e, _ in sdfg.all_edges_recursive():
        if hasattr(e.data, 'wcr'):
            assert (not e.data.wcr)

    ref = np.reshape(np.arange(125, dtype=np.int32), (5, 5, 5))
    A = ref.copy()
    sdfg(A)
    no_wcr.f(ref)
    assert (np.allclose(A, ref))


def test_augassign_no_wcr2():
    @dace.program
    def no_wcr(A: dace.int32[5, 5, 5]):
        A[2, 3, 1:4] += A[2:5, 1, 4]

    with dace.config.set_temporary('frontend', 'avoid_wcr', value=True):
        sdfg = no_wcr.to_sdfg(simplify=False)
    for e, _ in sdfg.all_edges_recursive():
        if hasattr(e.data, 'wcr'):
            assert (not e.data.wcr)

    ref = np.reshape(np.arange(125, dtype=np.int32), (5, 5, 5))
    A = ref.copy()
    sdfg(A)
    no_wcr.f(ref)
    assert (np.allclose(A, ref))


def test_augassign_wcr4():
    
    with dace.config.set_temporary('frontend', 'avoid_wcr', value=False):
        val = augassign_wcr4()
        ref = augassign_wcr4.f()
        assert np.allclose(val, ref)


def test_augassign_scalar_in_map():

    @dace.program
    def tester(a: dace.float64[20], b: dace.float64[20, 2], c: dace.float64[20, 2]):
        for i in dace.map[0:20]:
            tmp: dace.float64 = 0
            if i % 2 == 0:
                tmp += b[i, 0] * c[i, 0]
            else:
                tmp += b[i, 1] * c[i, 1]
            a[i] = tmp

    a = np.random.rand(20)
    b = np.random.rand(20, 2)
    c = np.random.rand(20, 2)
    ref = np.zeros(20)
    ref[::2] = (b * c)[::2, 0]
    ref[1::2] = (b * c)[1::2, 1]

    tester(a, b, c)

    assert np.allclose(a, ref)


if __name__ == "__main__":
    test_augassign_wcr()
    test_augassign_wcr2()
    test_augassign_wcr3()
    test_augassign_wcr4()
    test_augassign_no_wcr()
    test_augassign_no_wcr2()
    test_augassign_scalar_in_map()
