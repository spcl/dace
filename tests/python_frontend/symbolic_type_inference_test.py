# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""``sym_type`` reads an expression's type by evaluating it at ONE representative point, and both
halves of that were wrong for an ordinary spacing expression.

The integer representative is 1, so ``N - 1`` is zero there: sympy folds ``1.0 / (N - 1)`` to
``zoo``, ``astutils.unparse`` renders that as ``math.nan``, and the eval that read the type was
given no globals at all -- so ``h = 1.0 / (N - 1)`` refused with "name 'math' is not defined",
naming a module the program never mentions and the frontend has no business requiring. The integer
spelling ``N // (N - 1)`` did not even reach the eval; it raised ZeroDivisionError out of sympy's
own substitution.
"""
import numpy as np
import pytest

import dace
from dace import symbolic
from dace.frontend.python.replacements.utils import sym_type


def test_a_float_over_a_symbolic_sum_infers_a_float():
    N = symbolic.symbol('N', dtype=dace.int64, positive=True)
    assert sym_type(1.0 / (N - 1)) == dace.float64
    assert sym_type(1.0 / ((N - 1) * (N - 1))) == dace.float64


def test_an_integer_expression_stays_integral_at_the_degenerate_point():
    """``N // (N - 1)`` divides by zero at the first representative point; the answer is still int."""
    N = symbolic.symbol('N', dtype=dace.int64, positive=True)
    assert sym_type(symbolic.int_floor(N, N - 1)) == dace.int64
    assert sym_type(N - 1) == dace.int64


def test_the_frontend_parses_a_grid_spacing():
    """The shape this arrived as: a spacing computed from the grid extent, at program scope."""
    N = dace.symbol('N', dtype=dace.int64, positive=True)

    @dace.program
    def spacing(u: dace.float64[N, N]):
        h = 1.0 / (N - 1)
        u[0, 0] = h * h

    sdfg = spacing.to_sdfg(simplify=False)
    sdfg.validate()

    u = np.ones((8, 8), dtype=np.float64)
    sdfg(u=u, N=8)
    assert u[0, 0] == pytest.approx((1.0 / 7)**2)


if __name__ == '__main__':
    test_a_float_over_a_symbolic_sum_infers_a_float()
    test_an_integer_expression_stays_integral_at_the_degenerate_point()
    test_the_frontend_parses_a_grid_spacing()
