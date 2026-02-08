# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
import pytest

M, N, K = (dace.symbol(name) for name in ['M', 'N', 'K'])


@dace.program
def copy3d(A: dace.float32[M, N, K], B: dace.float32[M, N, K]):
    for i in parrange(M):
        for j, k in dace.map[0:N, 0:K]:
            with dace.tasklet:
                a << A[i, j, k]
                b >> B[i, j, k]
                b = a


def test_copy3d():
    N = M = K = 24
    A = np.random.rand(M, N, K).astype(np.float32)
    B = np.random.rand(M, N, K).astype(np.float32)
    copy3d(A, B)

    diff = np.linalg.norm(B - A) / (M * N)
    print('Difference:', diff)
    assert diff < 1e-5


def test_map_python():
    A = np.random.rand(20, 20)
    B = np.random.rand(20, 20)
    for i, j in dace.map[0:20, 1:20]:
        B[i, j] = A[i, j]

    assert np.allclose(A[:, 1:], B[:, 1:])


def test_nested_map_with_indirection():
    N = dace.symbol('N')

    @dace.program
    def indirect_to_indirect(arr1: dace.float64[N], ind: dace.int32[10], arr2: dace.float64[N]):
        for i in dace.map[0:9]:
            begin, end, stride = ind[i], ind[i + 1], 1
            for _ in dace.map[0:1]:
                for j in dace.map[begin:end:stride]:
                    arr2[j] = arr1[j] + i

    a = np.random.rand(50)
    b = np.zeros(50)
    ind = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45], dtype=np.int32)
    sdfg = indirect_to_indirect.to_sdfg(simplify=False)
    sdfg(a, ind, b)

    ref = np.zeros(50)
    for i in range(9):
        begin, end = ind[i], ind[i + 1]
        ref[begin:end] = a[begin:end] + i

    assert np.allclose(b, ref)


@pytest.mark.parametrize(["scope_type"], [["map"], ["for"]])
@pytest.mark.parametrize(["scope_use_type"], [["scalar"], ["view"]])
def test_nested_map_with_indirection_view(scope_type, scope_use_type):
    N = dace.symbol('N')

    @dace.program
    def indirect_to_indirectv(arr1: dace.float64[N], ind: dace.int32[10], arr2: dace.float64[N]):
        for i in dace.map[0:9]:
            begin, end, end2, stride = ind[i], ind[i + 1:i + 2], ind[i + 1], 1
            for _ in dace.map[0:1]:
                if scope_use_type == "scalar":
                    if scope_type == "map":
                        for j in dace.map[begin:end2:stride]:
                            arr2[j] = arr1[j] + end
                    else:
                        for j in range(begin, end2, stride):
                            arr2[j] = arr1[j] + end
                else:
                    if scope_type == "map":
                        for j in dace.map[begin:end:stride]:
                            arr2[j] = arr1[j] + end
                    else:
                        for j in range(begin, end, stride):
                            arr2[j] = arr1[j] + end

    a = np.random.rand(50)
    b = np.zeros(50)
    ind = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45], dtype=np.int32)
    sdfg = indirect_to_indirectv.to_sdfg(simplify=False)
    sdfg(a, ind, b)

    ref = np.zeros(50)
    for i in range(9):
        begin, end = ind[i], ind[i + 1]
        ref[begin:end] = a[begin:end] + end

    assert np.allclose(b, ref)


def test_dynamic_map_range_scalar():
    """
    From issue #650.
    """

    @dace.program
    def test(A: dace.float64[20], B: dace.float64[20]):
        N = dace.define_local_scalar(dace.int32)
        N = 5
        for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
            for j in dace.map[0:N] @ dace.ScheduleType.CPU_Multicore:
                with dace.tasklet:
                    a << A[i]
                    b >> B[j]
                    b = a + 1

    A = np.random.rand(20)
    B = np.zeros(20)
    test(A, B)
    assert np.allclose(B[:5], A[4] + 1)


if __name__ == '__main__':
    test_copy3d()
    test_map_python()
    test_nested_map_with_indirection()
    test_nested_map_with_indirection_view("map", "scalar")
    test_nested_map_with_indirection_view("for", "scalar")
    test_nested_map_with_indirection_view("map", "view")
    test_nested_map_with_indirection_view("for", "view")
    test_dynamic_map_range_scalar()
