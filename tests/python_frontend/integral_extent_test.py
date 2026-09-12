# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Array extents that are not whole numbers. """
import dace
import numpy as np
import pytest

N = dace.symbol('N', dtype=dace.int64, positive=True)
FN = dace.symbol('FN', dtype=dace.float64)


def test_a_float_valued_extent_is_refused_not_truncated():
    """``np.zeros(4.0)`` sized a four-element array, which is what the value happens to truncate to.

    numpy refuses a float shape outright, and so must this: the same path takes ``4.7`` to four
    elements with nothing said, so a kernel whose extent arithmetic drifted off the integers would
    allocate and iterate a silently different array.
    """

    @dace.program
    def literal_float_extent(a: dace.float64[N], out: dace.float64[N]):
        tmp = np.zeros(4.0, dtype=np.float64)
        tmp[0] = a[0]
        out[:] = tmp[0]

    with pytest.raises(TypeError, match='integral'):
        literal_float_extent.to_sdfg(simplify=False)


def test_an_extent_with_a_floating_point_coefficient_is_refused():
    """A fold that cancels a symbol leaves the coefficient behind: two grid spacings that are both
    ``extent / N`` reduce to ``0.5 * N``, which reached the descriptor as the shape it stands for and
    sized the allocation by a value no integer division produced."""

    @dace.program
    def folded_float_extent(a: dace.float64[N], out: dace.float64[N]):
        tmp = np.zeros(N / 2.0, dtype=np.float64)
        tmp[0] = a[0]
        out[:] = tmp[0]

    with pytest.raises(TypeError, match='integral'):
        folded_float_extent.to_sdfg(simplify=False)


def test_a_float_typed_symbol_is_refused_as_an_extent():
    """The expression carries no float atom of its own, so only the symbol's dtype says it cannot
    count elements. Refused while the annotation is read, which is before any parse."""

    with pytest.raises(TypeError, match='integral'):

        @dace.program
        def float_symbol_extent(a: dace.float64[FN], out: dace.float64[FN]):
            out[:] = a


def test_a_float_valued_scalar_promoted_into_a_shape_is_refused():
    """A local scalar reaching a shape is promoted to a symbol, and the promotion is where a float
    value would otherwise lose its dtype and pass for an extent."""

    @dace.program
    def promoted_float_extent(a: dace.float64[N], out: dace.float64[N]):
        n = N * 1.5
        tmp = np.zeros(n, dtype=np.float64)
        tmp[0] = a[0]
        out[:] = tmp[0]

    with pytest.raises(TypeError, match='integral'):
        promoted_float_extent.to_sdfg(simplify=False)


def test_an_integer_division_extent_still_sizes_and_computes():
    """The control: an extent that divides exactly is integral and must keep working, or the refusal
    above has taken every symbolic extent with it."""

    @dace.program
    def halves(a: dace.float64[N], out: dace.float64[N]):
        half = N // 2
        tmp = np.zeros(half, dtype=np.float64)
        tmp[:] = a[0:half] * 2.0
        out[0:half] = tmp

    a = np.arange(16, dtype=np.float64)
    out = np.zeros(16, dtype=np.float64)
    halves(a=a, out=out, N=16)
    np.testing.assert_allclose(out[0:8], a[0:8] * 2.0, rtol=0.0, atol=0.0)


if __name__ == '__main__':
    test_a_float_valued_extent_is_refused_not_truncated()
    test_an_extent_with_a_floating_point_coefficient_is_refused()
    test_a_float_typed_symbol_is_refused_as_an_extent()
    test_a_float_valued_scalar_promoted_into_a_shape_is_refused()
    test_an_integer_division_extent_still_sizes_and_computes()


def test_integer_only_operations_on_symbols_are_accepted_as_extents():
    """A shift, a modulo and a bitwise and of integer symbols are integers, so refusing them would reject real
    kernels such as a wavelet transform sized by ``N >> level``."""

    @dace.program
    def integer_operation_extents(a: dace.float64[N], out: dace.float64[3]):
        halved = np.zeros(N >> 1, dtype=np.float64)
        remainder = np.zeros(N % 3 + 1, dtype=np.float64)
        masked = np.zeros((N & 7) + 1, dtype=np.float64)
        halved[:] = a[0]
        remainder[:] = a[1]
        masked[:] = a[2]
        out[0] = halved.sum()
        out[1] = remainder.sum()
        out[2] = masked.sum()

    a = np.arange(1.0, 13.0)
    out = np.zeros(3)
    integer_operation_extents(a, out)
    size = a.size
    assert out.tolist() == [a[0] * (size >> 1), a[1] * (size % 3 + 1), a[2] * ((size & 7) + 1)], out
