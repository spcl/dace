# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import warnings

N = dace.symbol('N')
N.set(12345)


@dace.program
def nested(A: dace.float64[N], B: dace.float64[N], factor: dace.float64):
    B[:] = A * factor


@dace.program
def nested_symbol(A: dace.float64[N], B: dace.float64[N]):
    nested(A[0:5], B[0:5], 0.5)
    nested(A=A[5:N], B=B[5:N], factor=2.0)


@dace.program
def nested_symbol_dynamic(A: dace.float64[N]):
    for i in range(5):
        nested(A[0:i], A[0:i], i)


def test_nested_symbol():
    A = np.random.rand(20)
    B = np.random.rand(20)
    nested_symbol(A, B)
    assert np.allclose(B[0:5], A[0:5] / 2) and np.allclose(B[5:20], A[5:20] * 2)


def test_nested_symbol_dynamic():
    if not dace.Config.get_bool('optimizer', 'automatic_dataflow_coarsening'):
        warnings.warn("Test disabled (missing allocation lifetime support)")
        return

    A = np.random.rand(5)
    expected = A.copy()
    for i in range(5):
        expected[0:i] *= i
    nested_symbol_dynamic(A)
    assert np.allclose(A, expected)


def test_scal2sym():
    N = dace.symbol('N', dace.float64)

    @dace.program
    def symarg(A: dace.float64[20]):
        A[:] = N

    @dace.program
    def scalarg(A: dace.float64[20], scal: dace.float64):
        s2 = scal + 1
        symarg(A, N=s2)

    sdfg = scalarg.to_sdfg(strict=False)
    A = np.random.rand(20)
    sc = 5.0

    sdfg(A, sc)
    assert np.allclose(A, sc + 1)


def test_arr2sym():
    N = dace.symbol('N', dace.float64)

    @dace.program
    def symarg(A: dace.float64[20]):
        A[:] = N

    @dace.program
    def scalarg(A: dace.float64[20], arr: dace.float64[2]):
        symarg(A, N=arr[1])

    sdfg = scalarg.to_sdfg(strict=False)
    A = np.random.rand(20)
    sc = np.array([2.0, 3.0])

    sdfg(A, sc)
    assert np.allclose(A, sc[1])


def test_nested_symbol_in_args():
    inner = dace.SDFG('inner')
    state = inner.add_state('inner_state')
    inner.add_symbol('rdt', stype=float)
    inner.add_datadesc('field', dace.float64[10])
    state.add_mapped_tasklet(
        'tasklet',
        map_ranges={'i': "0:10"},
        inputs={},
        outputs={'field_out': dace.Memlet.simple('field', subset_str="i")},
        code="field_out = rdt",
        external_edges=True)
    inner.arg_names = ['field', 'rdt']

    @dace.program
    def funct(field, dt):
        rdt = 1.0 / dt
        inner(field, rdt)

    sdfg = funct.to_sdfg(np.random.randn(10, ), 1.0, strict=False)
    sdfg(np.random.randn(10, ), 1.0)


def test_nested_symbol_as_constant():
    inner = dace.SDFG('inner')
    state = inner.add_state('inner_state')
    inner.add_symbol('rdt', stype=float)
    inner.add_datadesc('field', dace.float64[10])
    tasklet, map_entry, map_exit = state.add_mapped_tasklet(
        'tasklet',
        map_ranges={'i': "0:10"},
        inputs={},
        outputs={'field_out': dace.Memlet.simple('field', subset_str="i")},
        code="field_out = rdt",
        external_edges=True)
    inner.arg_names = ['field', 'rdt']
    rdt = 1e30

    @dace.program
    def funct(field):
        inner(field, rdt)

    funct(np.random.randn(10, ))


if __name__ == '__main__':
    test_nested_symbol()
    test_nested_symbol_dynamic()
    test_scal2sym()
    test_arr2sym()
    test_nested_symbol_in_args()
    test_nested_symbol_as_constant()
