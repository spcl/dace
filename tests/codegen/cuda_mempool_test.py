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
            '''
            // Do nothing
            '''
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


@pytest.mark.gpu
def test_memory_pool_multistate():

    @dace.program
    def tester(A: CudaArray, B: CudaArray):
        # Things that can be in the same state
        pooled = dace.define_local(A.shape, A.dtype)

        for i in range(5):
            pooled << A

            if i == 1:
                B += 1

            B[:] = pooled

        return B

    sdfg = tester.to_sdfg(simplify=False)
    for aname, arr in sdfg.arrays.items():
        if aname == 'pooled':
            arr.storage = dace.StorageType.GPU_Global
            arr.pool = True
    for me, _ in sdfg.all_nodes_recursive():
        if isinstance(me, dace.nodes.MapEntry):
            me.schedule = dace.ScheduleType.GPU_Device

    code = sdfg.generate_code()[0].clean_code
    assert code.count('cudaMallocAsync') == 1
    assert code.count('cudaFreeAsync(pooled, __state->gpu_context->streams[0]') == 1 or code.count(
        'cudaFreeAsync(pooled, gpu_stream0') == 1

    # Test code
    import cupy as cp
    a = cp.random.rand(20)
    b = cp.random.rand(20)
    b_expected = cp.copy(a)
    sdfg(a, b)
    assert cp.allclose(a, b_expected)
    assert cp.allclose(b, b_expected)


@pytest.mark.gpu
@pytest.mark.parametrize('cnd', (0, 1))
def test_memory_pool_if_states(cnd):
    N = 20
    sdfg = dace.SDFG('test_memory_pool_if_states')
    sdfg.add_symbol('cnd', stype=dace.int32)

    A, A_desc = sdfg.add_array('A', shape=[N], dtype=dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    tmp, tmp_desc = sdfg.add_temp_transient_like(A_desc)
    tmp_desc.pool = True

    entry_state = sdfg.add_state('entry', is_start_block=True)
    exit_state = sdfg.add_state('exit')

    tstate = sdfg.add_state('true_branch')
    sdfg.add_edge(entry_state, tstate, dace.InterstateEdge(condition='Eq(cnd, 0)'))
    sdfg.add_edge(tstate, exit_state, dace.InterstateEdge())

    fstate = sdfg.add_state('false_branch')
    sdfg.add_edge(entry_state, fstate, dace.InterstateEdge(condition='Ne(cnd, 0)'))
    sdfg.add_edge(fstate, exit_state, dace.InterstateEdge())

    tmp_node = tstate.add_access(tmp)
    tstate.add_mapped_tasklet('write_zero',
                              map_ranges=dict(i=f'0:{N}'),
                              inputs={},
                              outputs={'_val': dace.Memlet(data=tmp, subset='i')},
                              output_nodes={tmp: tmp_node},
                              code='_val = 0.0',
                              external_edges=True)
    tstate.add_nedge(tmp_node, tstate.add_access(A), sdfg.make_array_memlet(A))

    fstate.add_mapped_tasklet('write_cond',
                              map_ranges=dict(i=f'0:{N}'),
                              inputs={},
                              outputs={'_val': dace.Memlet(data=A, subset='i')},
                              code='_val = dace.float64(cnd)',
                              external_edges=True)

    sdfg.validate()
    code = sdfg.generate_code()[0].clean_code
    assert code.count('cudaMallocAsync') == 1
    assert code.count(f'cudaFreeAsync({tmp}, __state->gpu_context->streams[0]') == 1 or code.count(
        f'cudaFreeAsync({tmp}, gpu_stream0') == 1

    # Test code
    import cupy as cp
    a = cp.random.rand(N)
    a_expected = cp.full(N, cnd, dtype=cp.float64)
    sdfg(A=a, cnd=cnd)
    assert cp.allclose(a, a_expected)


if __name__ == '__main__':
    test_memory_pool()
    test_memory_pool_state()
    test_memory_pool_tasklet()
    test_memory_pool_multistate()
    test_memory_pool_if_states()
