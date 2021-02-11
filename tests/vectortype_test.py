# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

# Define vector type
float2 = dace.vector(dace.float32, 2)


def test_vector_type():
    sdfg = dace.SDFG('vectortypes')
    sdfg.add_array('A', [2], float2)
    sdfg.add_array('B', [2], float2)
    state = sdfg.add_state()
    r = state.add_read('A')
    # With type inference
    t1 = state.add_tasklet('something', {'a'}, {'b'}, 'b = a * 2')
    # Without type inference
    t2 = state.add_tasklet('something', dict(a=float2), dict(b=float2),
                           'b = a * 2')
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
    t1 = state.add_tasklet('something', dict(a=float2), dict(b=float2),
                           'b = a * 2')
    t2 = state.add_tasklet('something', dict(a=float2), dict(b=float2),
                           'b = a * 2')
    w = state.add_write('B')
    state.add_edge(r, None, t1, 'a', dace.Memlet('A[0:2]'))
    state.add_edge(t1, 'b', w, None, dace.Memlet('B[0:2]'))
    state.add_edge(r, None, t2, 'a', dace.Memlet('A[2:4]'))
    state.add_edge(t2, 'b', w, None, dace.Memlet('B[2:4]'))

    A = np.random.rand(4).astype(np.float32)
    B = np.random.rand(4).astype(np.float32)
    sdfg(A=A, B=B)
    assert np.allclose(B, 2 * A)


if __name__ == '__main__':
    test_vector_type()
    test_vector_type_inference()
    test_vector_type_cast()
