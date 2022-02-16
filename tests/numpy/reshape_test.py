# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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


def test_reshape_dst():
    """ Tasklet->View->Array """
    @dace.program
    def reshpdst(A: dace.float64[2, 3, 4], B: dace.float64[8, 3]):
        C = np.reshape(B, [2, 3, 4])
        C[:] = A

    A = np.random.rand(2, 3, 4)
    B = np.random.rand(8, 3)

    reshpdst(A, B)
    assert np.allclose(A, np.reshape(B, [2, 3, 4]))


def test_reshape_dst_explicit():
    """ Tasklet->View->Array """
    sdfg = dace.SDFG('reshapedst')
    sdfg.add_array('A', [2, 3, 4], dace.float64)
    sdfg.add_view('Bv', [2, 3, 4], dace.float64)
    sdfg.add_array('B', [8, 3], dace.float64)
    state = sdfg.add_state()

    me, mx = state.add_map('compute', dict(i='0:2', j='0:3', k='0:4'))
    t = state.add_tasklet('add', {'a'}, {'b'}, 'b = a + 1')
    state.add_memlet_path(state.add_read('A'), me, t, dst_conn='a', memlet=dace.Memlet('A[i,j,k]'))
    v = state.add_access('Bv')
    state.add_memlet_path(t, mx, v, src_conn='b', memlet=dace.Memlet('Bv[i,j,k]'))
    state.add_nedge(v, state.add_write('B'), dace.Memlet('B'))
    sdfg.validate()

    A = np.random.rand(2, 3, 4)
    B = np.random.rand(8, 3)
    sdfg(A=A, B=B)
    assert np.allclose(A + 1, np.reshape(B, [2, 3, 4]))


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
    state.add_edge(r, None, v, 'views', dace.Memlet(data='A'))
    state.add_nedge(v, w, dace.Memlet(data='B' if memlet_dst else 'Av'))
    sdfg.validate()

    A = np.random.rand(2, 3)
    B = np.random.rand(6)
    sdfg(A=A, B=B)
    assert np.allclose(A.reshape([6]), B)


def test_reshape_copy_scoped():
    """ Array->View->Array where one array is located within a map scope. """
    sdfg = dace.SDFG('reshpcpy')
    sdfg.add_array('A', [2, 3], dace.float64)
    sdfg.add_array('B', [6], dace.float64)
    sdfg.add_view('Av', [6], dace.float64)
    sdfg.add_transient('tmp', [1], dace.float64)
    state = sdfg.add_state()
    r = state.add_read('A')
    me, mx = state.add_map('reverse', dict(i='0:6'))
    v = state.add_access('Av')
    t = state.add_access('tmp')
    w = state.add_write('B')
    state.add_edge_pair(me, v, r, dace.Memlet('A[0:2, 0:3]'), dace.Memlet('A[0:2, 0:3]'))
    state.add_nedge(v, t, dace.Memlet('Av[i]'))
    state.add_memlet_path(t, mx, w, memlet=dace.Memlet('B[6 - i - 1]'))
    sdfg.validate()

    A = np.random.rand(2, 3)
    B = np.random.rand(6)
    sdfg(A=A, B=B)
    assert np.allclose(A.reshape([6])[::-1], B)


def test_reshape_subset():
    """ Tests reshapes on subsets of arrays. """
    @dace.program
    def reshp(A: dace.float64[2, 3, 4], B: dace.float64[12]):
        C = np.reshape(A[1, :, :], [12])
        B[:] += C

    A = np.random.rand(2, 3, 4)
    B = np.random.rand(12)
    expected = np.reshape(A[1, :, :], [12]) + B

    reshp(A, B)
    assert np.allclose(expected, B)


def test_reshape_subset_explicit():
    """ Tests reshapes on subsets of arrays. """
    sdfg = dace.SDFG('reshp')
    sdfg.add_array('A', [2, 3, 4], dace.float64)
    sdfg.add_array('B', [12], dace.float64)
    sdfg.add_view('Av', [12], dace.float64)
    state = sdfg.add_state()

    state.add_mapped_tasklet('compute',
                             dict(i='0:12'),
                             dict(a=dace.Memlet('Av[i]'), b=dace.Memlet('B[i]')),
                             'out = a + b',
                             dict(out=dace.Memlet('B[i]')),
                             external_edges=True)
    v = next(n for n in state.source_nodes() if n.data == 'Av')
    state.add_nedge(state.add_read('A'), v, dace.Memlet('A[1, 0:3, 0:4]'))

    A = np.random.rand(2, 3, 4)
    B = np.random.rand(12)
    expected = np.reshape(A[1, :, :], [12]) + B

    sdfg(A=A, B=B)
    assert np.allclose(expected, B)


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
    # test_reshape()
    # test_reshape_dst()
    # test_reshape_dst_explicit()
    # test_reshape_copy(False)
    test_reshape_copy(True)
    test_reshape_copy_scoped()
    test_reshape_subset()
    test_reshape_subset_explicit()
    test_reinterpret()
    test_reinterpret_invalid()
