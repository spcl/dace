# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.dataflow import GPUTransformMap
import numpy as np
import pytest

# Symbols
N = dace.symbol('N')
M = dace.symbol('M')
K = dace.symbol('K')
L = dace.symbol('L')

X = dace.symbol('X')
Y = dace.symbol('Y')
Z = dace.symbol('Z')
W = dace.symbol('W')
U = dace.symbol('U')


@dace.program
def highdim(A: dace.uint64[N, M, K, L, X, Y, Z, W, U], B: dace.uint64[N, M, K, L]):
    @dace.mapscope
    def kernel(i: _[5:N - 5], j: _[0:M], k: _[7:K - 1], l: _[0:L]):
        @dace.map
        def block(a: _[0:X], b: _[0:Y], c: _[1:Z], d: _[2:W - 2], e: _[0:U]):
            input << A[i, j, k, l, a, b, c, d, e]
            output >> B(1, lambda a, b: a + b)[i, j, k, l]
            output = input


def makendrange(*args):
    result = []
    for i in range(0, len(args), 2):
        result.append(slice(args[i], args[i + 1] - 1, 1))
    return result


def _test(sdfg):
    # 4D kernel with 5D block
    N = 12
    M = 3
    K = 14
    L = 15
    X = 1
    Y = 2
    Z = 3
    W = 4
    U = 5
    dims = tuple(s for s in (N, M, K, L, X, Y, Z, W, U))
    outdims = tuple(s for s in (N, M, K, L))
    print('High-dimensional GPU kernel test', dims)

    A = dace.ndarray((N, M, K, L, X, Y, Z, W, U), dtype=dace.uint64)
    B = dace.ndarray((N, M, K, L), dtype=dace.uint64)
    A[:] = np.random.randint(10, size=dims).astype(np.uint64)
    B[:] = np.zeros(outdims, dtype=np.uint64)
    B_regression = np.zeros(outdims, dtype=np.uint64)

    # Equivalent python code
    for i, j, k, l in dace.ndrange(makendrange(5, N - 5, 0, M, 7, K - 1, 0, L)):
        for a, b, c, d, e in dace.ndrange(makendrange(0, X, 0, Y, 1, Z, 2, W - 2, 0, U)):
            B_regression[i, j, k, l] += A[i, j, k, l, a, b, c, d, e]

    sdfg(A=A, B=B, N=N, M=M, K=K, L=L, X=X, Y=Y, Z=Z, W=W, U=U)

    diff = np.linalg.norm(B_regression - B) / (N * M * K * L)
    print('Difference:', diff)
    assert diff <= 1e-5


def test_cpu():
    _test(highdim.to_sdfg())


@pytest.mark.gpu
def test_gpu():
    sdfg = highdim.to_sdfg()
    assert sdfg.apply_transformations(GPUTransformMap, options=dict(fullcopy=True)) == 1
    _test(sdfg)


@pytest.mark.gpu
def test_highdim_implicit_block():
    @dace.program
    def tester(x: dace.float64[32, 90, 80, 70]):
        for i, j, k, l in dace.map[0:32, 0:90, 0:80, 0:70]:
            x[i, j, k, l] = 2.0

    # Create GPU SDFG
    sdfg = tester.to_sdfg()
    sdfg.apply_gpu_transformations()

    # Change map implicit block size
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            node.map.gpu_block_size = [8, 2, 4]

    a = np.random.rand(32, 90, 80, 70)
    sdfg(a)
    assert np.allclose(a, 2)


@pytest.mark.gpu
def test_highdim_implicit_block_threadsplit():
    @dace.program
    def tester(x: dace.float64[2, 2, 80, 70]):
        for i, j, k, l in dace.map[0:2, 0:2, 0:80, 0:70]:
            x[i, j, k, l] = 2.0

    # Create GPU SDFG
    sdfg = tester.to_sdfg()
    sdfg.apply_gpu_transformations()

    # Change map implicit block size
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            node.map.gpu_block_size = [8, 2, 3]

    a = np.random.rand(2, 2, 80, 70)
    sdfg(a)
    assert np.allclose(a, 2)


def test_highdim_default_block_size():
    @dace.program
    def tester(a: dace.float64[1024, 1024] @ dace.StorageType.GPU_Global):
        for i, j in dace.map[0:1024, 0:1024] @ dace.ScheduleType.GPU_Device:
            a[i, j] = 1

    with dace.config.set_temporary('compiler', 'cuda', 'default_block_size', value='32, 8, 2'):
        with pytest.warns(UserWarning, match='has more dimensions'):
            sdfg = tester.to_sdfg()
            gpu_code = sdfg.generate_code()[1]
            assert 'dim3(32, 16, 1)' in gpu_code.code


def test_block_size_mismatch_warning():
    @dace.program
    def tester(a: dace.float64[1024, 1024] @ dace.StorageType.GPU_Global):
        for i, j in dace.map[0:512:2, 0:512:2] @ dace.ScheduleType.GPU_Device:
            for bi, bj in dace.map[0:2, 0:2] @ dace.ScheduleType.GPU_ThreadBlock:
                a[i + bi, j + bj] = 1
            for bi, bj in dace.map[0:2, 0:1] @ dace.ScheduleType.GPU_ThreadBlock:
                a[i + bi, j + bj] = 1

    sdfg = tester.to_sdfg()
    with pytest.warns(UserWarning, match='Multiple thread-block maps'):
        sdfg.generate_code()


def test_block_size_mismatch_error():
    @dace.program
    def tester(a: dace.float64[1024, 1024] @ dace.StorageType.GPU_Global):
        for i, j in dace.map[0:512:2, 0:512:2] @ dace.ScheduleType.GPU_Device:
            for bi, bj in dace.map[0:2, 0:2] @ dace.ScheduleType.GPU_ThreadBlock:
                a[i + bi, j + bj] = 1

    sdfg = tester.to_sdfg()
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry) and n.schedule == dace.ScheduleType.GPU_Device:
            n.gpu_block_size = [4, 2, 1]

    with pytest.raises(ValueError):
        sdfg.generate_code()


def test_block_size_too_large():
    @dace.program
    def tester(a: dace.float64[1024, 1024] @ dace.StorageType.GPU_Global):
        for i, j in dace.map[0:1024, 0:1024] @ dace.ScheduleType.GPU_Device:
            a[i, j] = 1

    sdfg = tester.to_sdfg()
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry) and n.schedule == dace.ScheduleType.GPU_Device:
            n.gpu_block_size = [64, 32, 1]

    with pytest.raises(ValueError):
        sdfg.generate_code()


def test_highdim_block_size_too_large():
    BX, BY, BZ, BW = 64, 2, 2, 2

    @dace.program
    def tester(a: dace.float64[1024, 2, 2, 20] @ dace.StorageType.GPU_Global):
        for i, j, k, l in dace.map[0:16, 0:1, 0:1, 0:10:2] @ dace.ScheduleType.GPU_Device:
            for bi, bj, bk, bl in dace.map[0:BX, 0:BY, 0:BZ, 0:BW] @ dace.ScheduleType.GPU_ThreadBlock:
                a[i + bi, j + bj, k + bk, l + bl] = 1

    sdfg = tester.to_sdfg()
    with pytest.raises(ValueError):
        sdfg.generate_code()


if __name__ == "__main__":
    test_cpu()
    test_gpu()
    test_highdim_implicit_block()
    test_highdim_implicit_block_threadsplit()
    test_highdim_default_block_size()
    test_block_size_mismatch_warning()
    test_block_size_mismatch_error()
    test_block_size_too_large()
    test_highdim_block_size_too_large()
