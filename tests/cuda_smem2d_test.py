# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace import nodes
from dace.transformation.dataflow import GPUTransformMap, InLocalStorage
import pytest

H = dace.symbol('H')
W = dace.symbol('W')


@dace.program(dace.float64[H, W], dace.float64[H, W])
def cudahello(V, Vout):
    @dace.mapscope(_[0:H:8, 0:W:32])
    def multiplication(i, j):
        @dace.map(_[0:8, 0:32])
        def mult_block(bi, bj):
            in_V << V[i + bi, j + bj]
            out >> Vout[i + bi, j + bj]
            out = in_V * 2.0


def _test(sdfg):
    W.set(128)
    H.set(64)

    print('Vector double CUDA (shared memory 2D) %dx%d' % (W.get(), H.get()))

    V = dace.ndarray([H, W], dace.float64)
    Vout = dace.ndarray([H, W], dace.float64)
    V[:] = np.random.rand(H.get(), W.get()).astype(dace.float64.type)
    Vout[:] = dace.float64(0)

    sdfg(V=V, Vout=Vout, H=H, W=W)

    diff = np.linalg.norm(2 * V - Vout) / (H.get() * W.get())
    print("Difference:", diff)
    assert diff <= 1e-5


def test_cpu():
    sdfg = cudahello.to_sdfg()
    sdfg.name = "cuda_smem2d_cpu"
    _test(sdfg)


@pytest.mark.gpu
def test_gpu():
    sdfg = cudahello.to_sdfg()
    sdfg.name = "cuda_smem2d_gpu"
    _test(sdfg)


@pytest.mark.gpu
def test_gpu_localstorage():
    sdfg = cudahello.to_sdfg()
    sdfg.name = "cuda_smem2d_gpu_localstorage"
    assert sdfg.apply_transformations([GPUTransformMap, InLocalStorage], options=[{}, {'array': 'gpu_V'}]) == 2
    _test(sdfg)


@pytest.mark.gpu
def test_gpu_2localstorage():
    @dace.program
    def addtwoandmult(A: dace.float64[H, W], B: dace.float64[H, W], Vout: dace.float64[H, W]):
        for i, j in dace.map[0:H:8, 0:W:32]:
            for bi, bj in dace.map[0:8, 0:32]:
                with dace.tasklet:
                    a << A[i + bi, j + bj]
                    b << B[i + bi, j + bj]
                    out = (a + b) * 2.0
                    out >> Vout[i + bi, j + bj]

    sdfg = addtwoandmult.to_sdfg()
    sdfg.name = "cuda_2_smem2d_gpu_localstorage"
    assert sdfg.apply_transformations([GPUTransformMap, InLocalStorage, InLocalStorage],
                                      options=[{}, {
                                          'array': 'gpu_A'
                                      }, {
                                          'array': 'gpu_B'
                                      }]) == 3

    A = np.random.rand(128, 64)
    B = np.random.rand(128, 64)
    out = np.random.rand(128, 64)
    refout = (A + B) * 2
    sdfg(A, B, out, H=128, W=64)
    assert np.allclose(refout, out)


@pytest.mark.gpu
def test_gpu_2shared_for():
    @dace.program
    def addtwoandmult(A: dace.float64[H, W], B: dace.float64[H, W], Vout: dace.float64[H, W]):
        for i, j in dace.map[0:H:8, 0:W:32]:
            for _ in range(1):
                local_a = dace.ndarray([8, 32], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
                local_b = dace.ndarray([8, 32], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
                local_a << A[i:i + 8, j:j + 32]
                local_b << B[i:i + 8, j:j + 32]
                for bi, bj in dace.map[0:8, 0:32]:
                    with dace.tasklet:
                        a << local_a[bi, bj]
                        b << local_b[bi, bj]
                        out = (a + b) * 2.0
                        out >> Vout[i + bi, j + bj]

    sdfg = addtwoandmult.to_sdfg()
    sdfg.name = "cuda_2_shared_for"
    state = sdfg.nodes()[0]
    map_entry = -1
    for node in state.nodes():
        if isinstance(node, nodes.MapEntry) and 'i' in node.map.params:
            map_entry = state.node_id(node)
            break
    transformation = GPUTransformMap()
    transformation.setup_match(sdfg, 0, 0, {GPUTransformMap.map_entry: map_entry}, 0)
    transformation.apply(state, sdfg)

    A = np.random.rand(128, 64)
    B = np.random.rand(128, 64)
    out = np.random.rand(128, 64)
    refout = (A + B) * 2
    sdfg(A, B, out, H=128, W=64)
    assert np.allclose(refout, out)


def _find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


@pytest.mark.gpu
def test_gpu_2shared_map():
    K = dace.symbol('K')

    @dace.program
    def addtwoandmult(A: dace.float64[H, W], B: dace.float64[H, W], Vout: dace.float64[H, W]):
        for i, j in dace.map[0:H:8, 0:W:32]:
            for _ in dace.map[0:K]:
                local_a = dace.ndarray([8, 32], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
                local_b = dace.ndarray([8, 32], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
                local_a << A[i:i + 8, j:j + 32]
                local_b << B[i:i + 8, j:j + 32]
                for bi, bj in dace.map[0:8, 0:32]:
                    with dace.tasklet:
                        a << local_a[bi, bj]
                        b << local_b[bi, bj]
                        out = (a + b) * 2.0
                        out >> Vout[i + bi, j + bj]

    sdfg = addtwoandmult.to_sdfg()
    sdfg.name = "cuda_2_shared_map"

    me = _find_map_by_param(sdfg, '_')
    me.schedule = dace.ScheduleType.Sequential
    sdfg.apply_gpu_transformations()
    me = _find_map_by_param(sdfg, 'bi')
    me.schedule = dace.ScheduleType.GPU_ThreadBlock

    A = np.random.rand(128, 64)
    B = np.random.rand(128, 64)
    out = np.random.rand(128, 64)
    refout = (A + B) * 2
    sdfg(A, B, out, H=128, W=64, K=1)
    assert np.allclose(refout, out)


if __name__ == "__main__":
    test_cpu()
    test_gpu_2localstorage()
    test_gpu_2shared_for()
    test_gpu_2shared_map()
