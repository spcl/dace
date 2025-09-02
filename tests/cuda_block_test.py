# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.dataflow import GPUTransformMap
from dace.transformation.interstate import GPUTransformSDFG
import numpy as np
import pytest

N = dace.symbol('N')


@dace.program(dace.float64[N], dace.float64[N])
def cudahello(V, Vout):

    @dace.mapscope(_[0:N:32])
    def multiplication(i):

        @dace.map(_[0:32])
        def mult_block(bi):
            in_V << V[i + bi]
            out >> Vout[i + bi]
            out = in_V * 2

        @dace.map(_[0:32])
        def mult_block_2(bi):
            in_V << V[i + bi]
            out >> Vout[i + bi]
            out = in_V * 2


def _test(sdfg):
    N = 128

    print('Vector double CUDA (block) %d' % (N))

    V = dace.ndarray([N], dace.float64)
    Vout = dace.ndarray([N], dace.float64)
    V[:] = np.random.rand(N).astype(dace.float64.type)
    Vout[:] = dace.float64(0)

    cudahello(V=V, Vout=Vout, N=N)

    diff = np.linalg.norm(2 * V - Vout) / N
    print("Difference:", diff)
    assert diff <= 1e-5


def test_cpu():
    _test(cudahello.to_sdfg())


@pytest.mark.gpu
def test_gpu():
    sdfg = cudahello.to_sdfg()
    assert sdfg.apply_transformations(GPUTransformMap) == 1
    _test(sdfg)


@pytest.mark.gpu
def test_different_block_sizes_nesting():

    @dace.program
    def nested(V: dace.float64[34], v1: dace.float64[1]):
        with dace.tasklet:
            o >> v1(-1)
            # Tasklet that does nothing
            pass

        for i in dace.map[0:34]:
            with dace.tasklet:
                inp << V[i]
                out >> v1(1, lambda a, b: a + b)[0]
                out = inp + inp

    @dace.program
    def nested2(V: dace.float64[34], v1: dace.float64[1]):
        with dace.tasklet:
            o >> v1(-1)
            # Tasklet that does nothing
            pass

        nested(V, v1)

    @dace.program
    def diffblocks(V: dace.float64[130], v1: dace.float64[4], v2: dace.float64[128]):
        for bi in dace.map[1:129:32]:
            for i in dace.map[0:32]:
                with dace.tasklet:
                    in_V << V[i + bi]
                    out >> v2[i + bi - 1]
                    out = in_V * 3

            nested2(V[bi - 1:bi + 33], v1[bi // 32:bi // 32 + 1])

    sdfg = diffblocks.to_sdfg()
    assert sdfg.apply_transformations(GPUTransformSDFG, dict(sequential_innermaps=False)) == 1
    V = np.random.rand(130)
    v1 = np.zeros([4], np.float64)
    v2 = np.random.rand(128)
    expected_v2 = V[1:129] * 3
    expected_v1 = np.zeros([4], np.float64)
    for i in range(4):
        expected_v1[i] = np.sum(V[i * 32:(i + 1) * 32 + 2]) * 2

    sdfg(V, v1, v2)
    assert np.linalg.norm(v1 - expected_v1) <= 1e-6
    assert np.allclose(v2, expected_v2)


@pytest.mark.gpu
def test_custom_block_size_onemap():

    @dace.program
    def tester(A: dace.float64[400, 300]):
        for i, j in dace.map[0:400, 0:300]:
            with dace.tasklet:
                a >> A[i, j]
                a = 1

    sdfg = tester.to_sdfg()
    sdfg.apply_gpu_transformations()
    mapentry: dace.nodes.MapEntry = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))

    # Test 1: too many dimensions
    mapentry.map.gpu_block_size = (13, 5, 3, 4)
    code = sdfg.generate_code()[1].clean_code  # Get GPU code (second file)
    assert 'dim3(13, 5, 12)' in code

    # Test 2: too few dimensions
    mapentry.map.gpu_block_size = (127, 5)
    code = sdfg.generate_code()[1].clean_code  # Get GPU code (second file)
    assert 'dim3(127, 5, 1)' in code

    # Test 3: compilation
    sdfg.compile()


@pytest.mark.gpu
def test_custom_block_size_twomaps():

    @dace.program
    def tester(A: dace.float64[400, 300, 2, 32]):
        for i, j in dace.map[0:400, 0:300]:
            for bi, bj in dace.map[0:2, 0:32]:
                with dace.tasklet:
                    a >> A[i, j, bi, bj]
                    a = 1

    sdfg = tester.to_sdfg()
    sdfg.apply_gpu_transformations(sequential_innermaps=True)
    mapentry: dace.nodes.MapEntry = next(
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_Device)

    mapentry.map.gpu_block_size = (127, 5)
    code = sdfg.generate_code()[1].clean_code  # Get GPU code (second file)
    assert 'dim3(127, 5, 1)' in code

    # Test 3: compilation
    sdfg.compile()


@pytest.mark.gpu
def test_block_thread_specialization():

    @dace.program
    def tester(A: dace.float64[200]):
        for i in dace.map[0:200:32]:
            for bi in dace.map[0:32]:
                with dace.tasklet:
                    a >> A[i + bi]
                    a = 1
                with dace.tasklet:  # Tasklet to be specialized
                    a >> A[i + bi]
                    a = 2

    sdfg = tester.to_sdfg()
    sdfg.apply_gpu_transformations(sequential_innermaps=False)
    tasklet = next(n for n, _ in sdfg.all_nodes_recursive()
                   if isinstance(n, dace.nodes.Tasklet) and '2' in n.code.as_string)
    tasklet.location['gpu_thread'] = dace.subsets.Range.from_string('2:9:3')
    tasklet.location['gpu_block'] = 1

    code = sdfg.generate_code()[1].clean_code  # Get GPU code (second file)
    assert '>= 2' in code and '<= 8' in code
    assert ' == 1' in code

    a = np.random.rand(200)
    ref = np.ones_like(a)
    ref[32:64][2:9:3] = 2
    sdfg(a)
    assert np.allclose(a, ref)


if __name__ == "__main__":
    test_cpu()
    test_gpu()
    test_different_block_sizes_nesting()
    test_custom_block_size_onemap()
    test_custom_block_size_twomaps()
    test_block_thread_specialization()
