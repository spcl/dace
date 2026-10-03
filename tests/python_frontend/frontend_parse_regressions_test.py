# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Regression tests for crashes in the Python frontend parse path. """
import numpy as np
import pytest

import dace
from dace.frontend.python.common import DaceSyntaxError

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def symbolic_or(x: dace.float64[10]):
    if N == 0 or N == 1:
        x[:] = 1.0
    else:
        x[:] = 2.0


@dace.program
def symbolic_and(x: dace.float64[10]):
    if N == 0 and N != 1:
        x[:] = 1.0
    else:
        x[:] = 2.0


@dace.program
def symbolic_ifexp(x: dace.float64[10]):
    if (N == 0) if N > 5 else (N == 1):
        x[:] = 1.0
    else:
        x[:] = 2.0


def test_symbolic_equality_in_boolop():
    """ A boolean operation over symbolic equalities used to raise ``TypeError: 'Equality' object is not
        iterable`` while preprocessing, because the equality rewriter returned a sympy object where the AST
        transformer contract requires an AST node. """
    for n, expected in ((0, 1.0), (1, 1.0), (5, 2.0)):
        x = np.zeros(10, dtype=np.float64)
        symbolic_or(x, N=n)
        assert np.allclose(x, expected)

    for n, expected in ((0, 1.0), (1, 2.0), (5, 2.0)):
        x = np.zeros(10, dtype=np.float64)
        symbolic_and(x, N=n)
        assert np.allclose(x, expected)

    # The same rewrite used to plant the sympy object in the tree on single-valued fields, which the
    # unparser then tripped over with ``'ExtUnparser' object has no attribute '_Equality'``.
    for n, expected in ((0, 2.0), (1, 1.0), (6, 2.0)):
        x = np.zeros(10, dtype=np.float64)
        symbolic_ifexp(x, N=n)
        assert np.allclose(x, expected)


@dace.program
def computed_index_gather(p: dace.float64[10], cols: dace.int64[5], out: dace.float64[5]):
    for i in range(5):
        out[i] = p[int(cols[i])]


@dace.program
def literal_index_list(x: dace.float64[10], out: dace.float64[3]):
    out[:] = x[[0, 2, 4]]


@dace.program
def symbolic_index_list(x: dace.float64[10], out: dace.float64[2]):
    out[:] = x[[0, N]]


def test_computed_index():
    """ Indexing with a computed index used to raise ``AttributeError: 'str' object has no attribute
        '_fields'``: the name returned by the replacement was mistaken for a list literal, i.e., for
        advanced indexing. """
    p = np.arange(10, dtype=np.float64) * 1.5
    cols = np.array([4, 0, 9, 2, 7], dtype=np.int64)
    out = np.zeros(5, dtype=np.float64)

    computed_index_gather(p, cols, out)

    assert np.allclose(out, p[cols])


def test_literal_index_list():
    """ A list literal index with plain numbers keeps working as advanced indexing. """
    x = np.arange(10, dtype=np.float64)
    out = np.zeros(3, dtype=np.float64)

    literal_index_list(x, out)

    assert np.allclose(out, x[[0, 2, 4]])


def test_symbolic_index_list():
    """ A list literal index with a symbolic element cannot be turned into a constant index array. It used
        to raise ``AttributeError: 'symbol' object has no attribute '_fields'`` from CPython's ast module,
        and now names the offending element and the source line. """
    with pytest.raises(DaceSyntaxError, match='is not an integer known at parse time'):
        symbolic_index_list.to_sdfg(simplify=False)


if __name__ == '__main__':
    test_symbolic_equality_in_boolop()
    test_computed_index()
    test_literal_index_list()
    test_symbolic_index_list()
