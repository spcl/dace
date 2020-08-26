import dace
import numpy as np


def copy_with_strides(src_numeric, dst_numeric):
    symvals = {}

    if src_numeric:
        E, F, G, H = 120, 10, 30, 2
    else:
        E, F, G, H = (dace.symbol(s) for s in 'EFGH')
        symvals.update({'E': 120, 'F': 10, 'G': 30, 'H': 2})

    if dst_numeric:
        X, Y, Z, W = 60, 20, 5, 1
    else:
        X, Y, Z, W = (dace.symbol(s) for s in 'XYZW')
        symvals.update({'X': 60, 'Y': 20, 'Z': 5, 'W': 1})

    sdfg = dace.SDFG('cws')
    sdfg.add_array('A',
                   shape=[2, 3, 4, 5],
                   dtype=dace.float64,
                   strides=[E, F, G, H])
    sdfg.add_array('B',
                   shape=[2, 3, 4, 5],
                   dtype=dace.float64,
                   strides=[X, Y, Z, W])
    state = sdfg.add_state()

    r = state.add_read('A')
    w = state.add_write('B')
    state.add_nedge(r, w, dace.Memlet.from_array('A', sdfg.arrays['A']))

    A = np.random.rand(2, 4, 3, 5, 2)
    B = np.random.rand(2, 3, 4, 5)
    sdfg(A=A, B=B, **symvals)

    expected = A[:, :, :, :, 0].transpose(0, 2, 1, 3)
    assert np.allclose(B, expected)


def test_copy_with_numeric_strides():
    copy_with_strides(True, True)


def test_copy_with_symbolic_strides():
    copy_with_strides(False, False)


def test_copy_with_src_symbolic_stride():
    copy_with_strides(True, False)


def test_copy_with_dst_symbolic_stride():
    copy_with_strides(False, True)


if __name__ == '__main__':
    test_copy_with_numeric_strides()
    test_copy_with_symbolic_strides()
    test_copy_with_src_symbolic_stride()
    test_copy_with_dst_symbolic_stride()
