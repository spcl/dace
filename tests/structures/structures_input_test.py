# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import cupy as cp
import numpy as np
import pytest


def test_input_0a_host_host():
    ''' Non-transient structure with scalar fields. Using views for scalar fields. Host storage/computation. '''

    struct = dace.data.Structure(dict(x=dace.data.Scalar(dace.float64), y=dace.data.Scalar(dace.float64)), 'Coords')
    x_view = dace.data.View.view(struct.members['x'])
    y_view = dace.data.View.view(struct.members['y'])

    sdfg = dace.SDFG('compute_distance')

    sdfg.add_datadesc('A', struct)
    sdfg.add_datadesc('vx', x_view)
    sdfg.add_datadesc('vy', y_view)
    sdfg.add_array('B', [1], dace.float64)

    state = sdfg.add_state()

    A = state.add_access('A')
    vx = state.add_access('vx')
    vy = state.add_access('vy')
    t = state.add_tasklet('compute_distance', {'__x', '__y'}, {'__out'}, '__out = sqrt(__x * __x + __y * __y)')
    B = state.add_access('B')

    state.add_edge(A, None, vx, 'views', dace.Memlet.from_array('A.x', struct.members['x']))
    state.add_edge(A, None, vy, 'views', dace.Memlet.from_array('A.y', struct.members['y']))
    state.add_edge(vx, None, t, '__x', dace.Memlet.from_array('vx', x_view))
    state.add_edge(vy, None, t, '__y', dace.Memlet.from_array('vy', y_view))
    state.add_edge(t, '__out', B, None, dace.Memlet.from_array('B', sdfg.arrays['B']))

    rng = np.random.default_rng(42)
    x = rng.random()
    y = rng.random()
    A = struct.dtype._typeclass.as_ctypes()(x, y)
    B = np.empty([1], dtype=np.float64)

    sdfg(A=A, B=B)
    assert np.allclose(B, np.sqrt(x * x + y * y))


@pytest.mark.skip
def test_input_0a_gpu_gpu():
    ''' Non-transient structure with scalar fields. Using views for scalar fields. GPU storage/computation. '''
    ''' Test cannot work because we cannot store scalas in GPU_Global storage. '''

    struct = dace.data.Structure(dict(x=dace.data.Scalar(dace.float64), y=dace.data.Scalar(dace.float64)), 'Coords',
                                 storage=dace.StorageType.GPU_Global)
    x_view = dace.data.View.view(struct.members['x'])
    y_view = dace.data.View.view(struct.members['y'])

    sdfg = dace.SDFG('compute_distance')

    sdfg.add_datadesc('A', struct)
    sdfg.add_datadesc('vx', x_view)
    sdfg.add_datadesc('vy', y_view)
    sdfg.add_array('B', [1], dace.float64, storage=dace.StorageType.GPU_Global)

    state = sdfg.add_state()

    A = state.add_access('A')
    vx = state.add_access('vx')
    vy = state.add_access('vy')
    t = state.add_tasklet('compute_distance', {'__x', '__y'}, {'__out'}, '__out = sqrt(__x * __x + __y * __y)')
    B = state.add_access('B')

    state.add_edge(A, None, vx, 'views', dace.Memlet.from_array('A.x', struct.members['x']))
    state.add_edge(A, None, vy, 'views', dace.Memlet.from_array('A.y', struct.members['y']))
    state.add_edge(vx, None, t, '__x', dace.Memlet.from_array('vx', x_view))
    state.add_edge(vy, None, t, '__y', dace.Memlet.from_array('vy', y_view))
    state.add_edge(t, '__out', B, None, dace.Memlet.from_array('B', sdfg.arrays['B']))

    sdfg.apply_gpu_transformations()

    rng = np.random.default_rng(42)
    x = rng.random()
    y = rng.random()
    A_dev = struct.dtype._typeclass.as_ctypes()(cp.float64(x), cp.float64(y))
    B_dev = cp.empty([1], dtype=np.float64)

    sdfg(A=A_dev, B=B_dev)
    B = cp.asnumpy(B_dev)
    assert np.allclose(B, np.sqrt(x * x + y * y))


if __name__ == '__main__':
    test_input_0a_host_host()
    test_input_0a_gpu_gpu()
