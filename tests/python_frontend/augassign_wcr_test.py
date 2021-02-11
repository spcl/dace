# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def augassign_wcr(A: dace.int32[10, 10, 10],
                  B: dace.int32[10],
                  W: dace.bool_[10]):
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
def augassign_wcr2(A: dace.int32[10, 10, 10],
                   B: dace.int32[10],
                   W: dace.bool_[10, 10, 10]):
    count = 0
    B[:] = 0
    for i, j, k in dace.map[0:10, 0:10, 0:10]:
        if W[i, j, k]:
            count += 1
            B[i] += A[i, j, k]
    
    return count


@dace.program
def augassign_wcr3(A: dace.int32[10, 10, 10],
                   B: dace.int32[10],
                   W: dace.bool_[10, 10, 10],
                   ind: dace.int32[10]):
    count = 0
    B[:] = 0
    for i, j, k in dace.map[0:10, 0:10, 0:10]:
        if W[i, j, k]:
            count += 1
            B[ind[i]] += A[i, j, k]
    
    return count


def test_augassign_wcr():
    A = np.random.randint(1, 10, size=(10, 10, 10), dtype=np.int32)
    B = np.empty((10,), dtype=np.int32)
    W = np.random.randint(2, size=(10,), dtype=np.bool_)

    test_sdfg = augassign_wcr.to_sdfg(strict=False)
    wcr_count = 0
    for sdfg in test_sdfg.sdfg_list:
        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.data.wcr:
                    wcr_count += 1     
    assert(wcr_count == 1)

    count = test_sdfg(A=A, B=B, W=W)
    assert(count[0] == np.count_nonzero(W))
    assert(np.array_equal(np.add.reduce(A, axis=(1, 2))[W], B[W]))


def test_augassign_wcr2():
    A = np.random.randint(1, 10, size=(10, 10, 10), dtype=np.int32)
    B = np.empty((10,), dtype=np.int32)
    C = np.zeros((10,), dtype=np.int32)
    W = np.random.randint(2, size=(10, 10, 10), dtype=np.bool_)

    test_sdfg = augassign_wcr2.to_sdfg(strict=False)
    wcr_count = 0
    for sdfg in test_sdfg.sdfg_list:
        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.data.wcr:
                    wcr_count += 1
    assert(wcr_count == 2)

    count = test_sdfg(A=A, B=B, W=W)
    C = np.add.reduce(A, axis=(1, 2), where=W)
    assert(count[0] == np.count_nonzero(W))
    assert(np.array_equal(B, C))


def test_augassign_wcr3():
    A = np.random.randint(1, 10, size=(10, 10, 10), dtype=np.int32)
    B = np.empty((10,), dtype=np.int32)
    C = np.zeros((10,), dtype=np.int32)
    D = np.zeros((10,), dtype=np.int32)
    ind = np.random.randint(0, 10, size=(10,), dtype=np.int32)
    W = np.random.randint(2, size=(10, 10, 10), dtype=np.bool_)

    test_sdfg = augassign_wcr3.to_sdfg(strict=False)
    wcr_count = 0
    for sdfg in test_sdfg.sdfg_list:
        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.data.wcr:
                    wcr_count += 1
    assert(wcr_count == 2)

    count = test_sdfg(A=A, B=B, W=W, ind=ind)
    C = np.add.reduce(A, axis=(1, 2), where=W)
    for i in range(10):
        D[ind[i]] += C[i]
    assert(count[0] == np.count_nonzero(W))
    assert(np.array_equal(B, D))


if __name__ == "__main__":
    test_augassign_wcr()
    test_augassign_wcr2()
    test_augassign_wcr3()
