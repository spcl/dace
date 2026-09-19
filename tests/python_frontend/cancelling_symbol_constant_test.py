# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Scalar expressions whose symbols cancel out. """
import dace
import numpy as np

N = dace.symbol('N', dtype=dace.int64, positive=True)


def test_cancelling_symbols_fold_to_a_constant():
    """A float expression whose symbols cancel is a constant, not a refusal.

    The frontend's own constant folding returns a symbol-free ``sympy.Float`` here, which is not
    symbolic (``free_symbols`` is empty) and is not a key ``dtype_to_typeclass`` holds -- so
    ``result_type`` took the ``numbers.Number`` branch, since SymPy registers its numeric atoms in
    that ABC tower, and raised ``KeyError: <class 'sympy.core.numbers.Float'>``.
    """

    @dace.program
    def cancels(a: dace.float64[N], out: dace.float64[N]):
        scale = 1.0 / (2.0 / N * N)
        out[:] = a * scale

    a = np.random.rand(16)
    out = np.zeros(16)
    cancels(a=a, out=out, N=16)
    assert np.allclose(out, a * 0.5)


def test_surviving_symbol_still_reaches_the_kernel():
    """The control: a symbol that does NOT cancel stays symbolic and still binds per call."""

    @dace.program
    def survives(a: dace.float64[N], out: dace.float64[N]):
        scale = 1.0 / (2.0 / N)
        out[:] = a * scale

    a = np.random.rand(16)
    out = np.zeros(16)
    survives(a=a, out=out, N=16)
    assert np.allclose(out, a * 8.0)


def test_cancelling_integer_symbols_stay_integer():
    """A cancelling INTEGER expression must not come back as a float."""

    @dace.program
    def int_cancels(a: dace.int64[N], out: dace.int64[N]):
        scale = (3 * N) // N
        out[:] = a * scale

    a = np.arange(16, dtype=np.int64)
    out = np.zeros(16, dtype=np.int64)
    int_cancels(a=a, out=out, N=16)
    assert np.array_equal(out, a * 3)


if __name__ == '__main__':
    test_cancelling_symbols_fold_to_a_constant()
    test_surviving_symbol_still_reaches_the_kernel()
    test_cancelling_integer_symbols_stay_integer()
