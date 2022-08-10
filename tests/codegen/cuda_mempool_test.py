# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pytest

CudaArray = dace.data.Array(dace.float64, [20], storage=dace.StorageType.GPU_Global)

@pytest.mark.gpu
def test_memory_pool():

    @dace.program
    def tester(A: CudaArray, B: CudaArray):
        # Things that can be in the same state
        tmp = A + 1
        tmp += B
        # Things that must be in a different state
        B[:] = tmp
        tmp2 = tmp + 2
        B[:] = tmp2 + 5

    sdfg = tester.to_sdfg()
    for arr in sdfg.arrays.values():
        arr.storage = dace.StorageType.GPU_Global
        arr.pool = True
    for me, _ in sdfg.all_nodes_recursive():
        if isinstance(me, dace.nodes.MapEntry):
            me.schedule = dace.ScheduleType.GPU_Device

    assert sdfg.number_of_nodes() >= 2

    code = sdfg.generate_code()[0].clean_code
    assert code.count('cudaMallocAsync') == 2
    assert code.count('cudaFreeAsync') == 2

    # Test code
    import cupy as cp
    a = cp.random.rand(20)
    b = cp.random.rand(20)
    a_expected = cp.copy(a)
    b_expected = cp.copy(b)
    tester.f(a_expected, b_expected)

    sdfg(a, b)
    assert cp.allclose(a, a_expected)
    assert cp.allclose(b, b_expected)


@pytest.mark.gpu
def test_memory_pool_state():

    @dace.program
    def tester(A: CudaArray, B: CudaArray, C: CudaArray):
        # Things that can be in the same state
        tmp = A + 1
        B[:] = tmp
        C[:] = tmp + 1

    sdfg = tester.to_sdfg()
    for arr in sdfg.arrays.values():
        arr.storage = dace.StorageType.GPU_Global
        arr.pool = True
    for me, _ in sdfg.all_nodes_recursive():
        if isinstance(me, dace.nodes.MapEntry):
            me.schedule = dace.ScheduleType.GPU_Device

    code = sdfg.generate_code()[0].clean_code
    assert code.count('cudaMallocAsync') == 1
    assert code.count('cudaFree') == 1

    # Test code
    import cupy as cp
    a = cp.random.rand(20)
    b = cp.random.rand(20)
    c = cp.random.rand(20)

    sdfg(a, b, c)
    assert cp.allclose(b, a + 1)
    assert cp.allclose(c, a + 2)


@pytest.mark.gpu
def test_memory_pool_tasklet():

    @dace.program
    def tester(A: CudaArray, B: CudaArray):
        # Things that can be in the same state
        tmp = A + 1
        with dace.tasklet(dace.Language.CPP):
            t << tmp
            b >> B
            """
            // Do nothing
            """
        A[:] = B

    sdfg = tester.to_sdfg()
    for arr in sdfg.arrays.values():
        arr.storage = dace.StorageType.GPU_Global
        arr.pool = True
    for me, _ in sdfg.all_nodes_recursive():
        if isinstance(me, dace.nodes.MapEntry):
            me.schedule = dace.ScheduleType.GPU_Device

    code = sdfg.generate_code()[0].clean_code
    assert code.count('cudaMallocAsync') == 1
    assert code.count('cudaFreeAsync') == 1

    # Test code
    import cupy as cp
    a = cp.random.rand(20)
    b = cp.random.rand(20)
    b_expected = cp.copy(b)
    sdfg(a, b)
    assert cp.allclose(a, b_expected)
    assert cp.allclose(b, b_expected)


if __name__ == '__main__':
    test_memory_pool()
    test_memory_pool_state()
    test_memory_pool_tasklet()
