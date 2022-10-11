import dace
import numpy as np
import warnings
warnings.filterwarnings("error")

def memlet_range_overlap():
    sdfg = dace.SDFG('memlet_range_overlap')
    sdfg.add_array('A', [5], dace.int32)
    sdfg.add_array('B', [5], dace.int32)

    state = sdfg.add_state()
    t1 = state.add_tasklet('first_tasklet', {'a'}, {'b'}, 'b[0] = a[0] + 10')
    t2 = state.add_tasklet('second_tasklet', {'a'}, {'b'}, 'b[1] = a[1] - 20')

    r = state.add_read('A')
    w = state.add_write('B')

    state.add_edge(r, None, t1, 'a', dace.Memlet('A[0:2]'))
    state.add_edge(r, None, t2, 'a', dace.Memlet('A[0:2]'))

    state.add_edge(t1, 'b', w, None, dace.Memlet('B[0:2]'))
    state.add_edge(t2, 'b', w, None, dace.Memlet('B[1:4]'))

    A = np.ones([5], dtype=np.int32)
    B = np.zeros([5], dtype=np.int32)

    try:
        sdfg(A=A, B=B)
        raise AssertionError('No warning generated, test failed.')
    except UserWarning:
        print("Warning successfully caught, test passed.")

if __name__ == '__main__':
    memlet_range_overlap()
