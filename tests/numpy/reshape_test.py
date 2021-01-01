# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for reshaping and reinterpretation of existing arrays. """
import dace
import numpy as np
import pytest

N = dace.symbol('N')


def test_reshape():
    """ Array->View->Tasklet """
    @dace.program
    def reshp(A: dace.float64[2, 3, 4], B: dace.float64[8, 3]):
        C = np.reshape(A, [8, 3])
        B[:] += C

    A = np.random.rand(2, 3, 4)
    B = np.random.rand(8, 3)
    expected = np.reshape(A, [8, 3]) + B

    reshp(A, B)
    assert np.allclose(expected, B)



@pytest.mark.parametrize('memlet_dst', (False, True))
def test_reshape_copy(memlet_dst):
    """ 
    Symmetric case of Array->View->Array. Should be translated to a reference
    and a copy.
    """
    sdfg = dace.SDFG('reshpcpy')
    sdfg.add_array('A', [2, 3], dace.float64)
    sdfg.add_array('B', [6], dace.float64)
    sdfg.add_view('Av', [6], dace.float64)
    state = sdfg.add_state()
    r = state.add_read('A')
    v = state.add_access('Av')
    w = state.add_write('B')
    state.add_nedge(r, v, dace.Memlet(data='A'))
    state.add_nedge(v, w, dace.Memlet(data='B' if memlet_dst else 'Av'))
    sdfg.validate()

    A = np.random.rand(2, 3)
    B = np.random.rand(6)
    sdfg(A=A, B=B)
    assert np.allclose(A.reshape([6]), B)


def test_reinterpret():
    @dace.program
    def reint(A: dace.int32[N]):
        C = A.view(dace.int16)
        C[:] += 1

    A = np.random.randint(0, 262144, size=[10], dtype=np.int32)
    expected = np.copy(A)
    B = expected.view(np.int16)
    B[:] += 1

    reint(A)
    assert np.allclose(expected, A)


def test_reinterpret_invalid():
    @dace.program
    def reint_invalid(A: dace.float32[5]):
        C = A.view(dace.float64)
        C[:] += 1

    A = np.random.rand(5).astype(np.float32)
    try:
        reint_invalid(A)
        raise AssertionError('Program should not be compilable')
    except ValueError:
        pass


if __name__ == "__main__":
    test_reshape()
    test_reshape_copy(False)
    test_reshape_copy(True)
    test_reinterpret()
    test_reinterpret_invalid()
