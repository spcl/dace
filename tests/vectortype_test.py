# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

# Define vector types
float2 = dace.vector(dace.float32, 2)
float4 = dace.vector(dace.float32, 4)


def test_vector_type():
    sdfg = dace.SDFG('vectortypes')
    sdfg.add_array('A', [2], float2)
    sdfg.add_array('B', [2], float2)
    state = sdfg.add_state()
    r = state.add_read('A')
    # With type inference
    t1 = state.add_tasklet('something', {'a'}, {'b'}, 'b = a * 2')
    # Without type inference
    t2 = state.add_tasklet('something', dict(a=float2), dict(b=float2), 'b = a * 2')
    w = state.add_write('B')
    state.add_edge(r, None, t1, 'a', dace.Memlet('A[0]'))
    state.add_edge(t1, 'b', w, None, dace.Memlet('B[0]'))
    state.add_edge(r, None, t2, 'a', dace.Memlet('A[1]'))
    state.add_edge(t2, 'b', w, None, dace.Memlet('B[1]'))

    A = np.random.rand(4).astype(np.float32)
    B = np.random.rand(4).astype(np.float32)
    sdfg(A=A, B=B)
    assert np.allclose(B, 2 * A)


def test_vector_type_inference():
    sdfg = dace.SDFG('vectortypes')
    sdfg.add_array('A', [1], float2)
    sdfg.add_array('B', [1], float2)
    state = sdfg.add_state()
    r = state.add_read('A')
    t1 = state.add_tasklet('something', {'a'}, {'b'}, 'b = a * 2')
    t2 = state.add_tasklet('something', {'a'}, {'b'}, 'b = a * 2')
    w = state.add_write('B')
    state.add_edge(r, None, t1, 'a', dace.Memlet('A[0]'))
    state.add_edge(t1, 'b', t2, 'a', dace.Memlet())
    state.add_edge(t2, 'b', w, None, dace.Memlet('B[0]'))

    A = np.random.rand(2).astype(np.float32)
    B = np.random.rand(2).astype(np.float32)
    sdfg(A=A, B=B)
    assert np.allclose(B, 4 * A)


def test_vector_type_cast():
    """ Test vector types with casting from float* to float2. """
    sdfg = dace.SDFG('vectortypes')
    sdfg.add_array('A', [4], dace.float32)
    sdfg.add_array('B', [4], dace.float32)
    state = sdfg.add_state()
    r = state.add_read('A')
    t1 = state.add_tasklet('something', dict(a=float2), dict(b=float2), 'b = a * 2')
    t2 = state.add_tasklet('something', dict(a=float2), dict(b=float2), 'b = a * 2')
    w = state.add_write('B')
    state.add_edge(r, None, t1, 'a', dace.Memlet('A[0:2]'))
    state.add_edge(t1, 'b', w, None, dace.Memlet('B[0:2]'))
    state.add_edge(r, None, t2, 'a', dace.Memlet('A[2:4]'))
    state.add_edge(t2, 'b', w, None, dace.Memlet('B[2:4]'))

    A = np.random.rand(4).astype(np.float32)
    B = np.random.rand(4).astype(np.float32)
    sdfg(A=A, B=B)
    assert np.allclose(B, 2 * A)


def test_vector_reduction():
    """ 
    Tests "horizontal" summation (hadd) of vector types using 
    write-conflicted memlets.
    """
    sdfg = dace.SDFG('vectorhadd')
    sdfg.add_array('A', [2], float2)
    sdfg.add_array('B', [2], dace.float32)
    state = sdfg.add_state()
    r = state.add_read('A')
    t1 = state.add_tasklet('something', {'a'}, {'b': float2}, 'b = a')
    t2 = state.add_tasklet('something', {'a'}, {'b': float2}, 'b = a')
    w = state.add_write('B')
    state.add_edge(r, None, t1, 'a', dace.Memlet('A[0]'))
    state.add_edge(t1, 'b', w, None, dace.Memlet('B[0]', wcr='lambda x, y: x + y'))
    state.add_edge(r, None, t2, 'a', dace.Memlet('A[1]'))
    state.add_edge(t2, 'b', w, None, dace.Memlet('B[1]', wcr='lambda x, y: x + y'))

    assert '_atomic' not in sdfg.generate_code()[0].clean_code

    A = np.random.rand(4).astype(np.float32)
    B = np.zeros([2], dtype=np.float32)
    sdfg(A=A, B=B)
    assert np.allclose(B, np.array([A[0] + A[1], A[2] + A[3]]))


def test_vector_to_vector_wcr():
    """ 
    Tests write-conflicted memlets from vectors on vectors.
    """
    sdfg = dace.SDFG('vectoradd')
    sdfg.add_array('A', [2], float2)
    sdfg.add_array('B', [2], float2)
    state = sdfg.add_state()
    r = state.add_read('A')
    t1 = state.add_tasklet('something', {'a'}, {'b'}, 'b = a')
    t2 = state.add_tasklet('something', {'a'}, {'b'}, 'b = a')
    w = state.add_write('B')
    state.add_edge(r, None, t1, 'a', dace.Memlet('A[0]'))
    state.add_edge(t1, 'b', w, None, dace.Memlet('B[0]', wcr='lambda x, y: x + y'))
    state.add_edge(r, None, t2, 'a', dace.Memlet('A[1]'))
    state.add_edge(t2, 'b', w, None, dace.Memlet('B[1]', wcr='lambda x, y: x + y'))

    assert '_atomic' not in sdfg.generate_code()[0].clean_code

    A = np.random.rand(4).astype(np.float32)
    B = np.random.rand(4).astype(np.float32)
    B_reg = B + A
    sdfg(A=A, B=B)
    assert np.allclose(B, B_reg)


def test_vector_reduction_atomic():
    """ 
    Tests "horizontal" summation (hadd) of vector types using 
    write-conflicted memlets, with atomics.
    """
    sdfg = dace.SDFG('vectorhadd_atomic')
    sdfg.add_array('A', [2], float2)
    sdfg.add_array('B', [1], dace.float32)
    state = sdfg.add_state()
    r = state.add_read('A')
    t1 = state.add_tasklet('something', {'a'}, {'b': float2}, 'b = a')
    t2 = state.add_tasklet('something', {'a'}, {'b': float2}, 'b = a')
    w = state.add_write('B')
    state.add_edge(r, None, t1, 'a', dace.Memlet('A[0]'))
    state.add_edge(t1, 'b', w, None, dace.Memlet('B[0]', wcr='lambda x, y: x + y'))
    state.add_edge(r, None, t2, 'a', dace.Memlet('A[1]'))
    state.add_edge(t2, 'b', w, None, dace.Memlet('B[0]', wcr='lambda x, y: x + y'))

    A = np.random.rand(4).astype(np.float32)
    B = np.zeros([1], dtype=np.float32)
    sdfg(A=A, B=B)
    assert np.allclose(B, np.sum(A))


@pytest.mark.gpu
def test_vector_reduction_gpu():
    """ 
    Tests "horizontal" summation (hadd) of vector types using 
    write-conflicted memlets.
    """
    sdfg = dace.SDFG('vectorhadd_gpu')
    sdfg.add_array('A', [1], float4)
    sdfg.add_transient('gA', [1], float4, storage=dace.StorageType.GPU_Global)
    sdfg.add_transient('gB', [1], dace.float32, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B', [1], dace.float32)
    state = sdfg.add_state()
    r = state.add_access('gA')
    me, mx = state.add_map('kernel', dict(i='0:1'), dace.ScheduleType.GPU_Device)
    t1 = state.add_tasklet('something', {'a'}, {'b': float4}, 'b = a')
    w = state.add_access('gB')
    state.add_memlet_path(r, me, t1, dst_conn='a', memlet=dace.Memlet('gA[0]'))
    state.add_memlet_path(t1, mx, w, src_conn='b', memlet=dace.Memlet('gB[0]', wcr='lambda x, y: x + y'))
    hr = state.add_read('A')
    hw = state.add_write('B')
    state.add_nedge(hr, r, dace.Memlet('gA'))
    state.add_nedge(w, hw, dace.Memlet('gB'))

    assert '_atomic' not in sdfg.generate_code()[0].clean_code

    A = np.random.rand(4).astype(np.float32)
    B = np.zeros([1], dtype=np.float32)
    sdfg(A=A, B=B)
    assert np.allclose(B, np.sum(A))


if __name__ == '__main__':
    test_vector_type()
    test_vector_type_inference()
    test_vector_type_cast()
    test_vector_reduction()
    test_vector_to_vector_wcr()
    test_vector_reduction_atomic()
