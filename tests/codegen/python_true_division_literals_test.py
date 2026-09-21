# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python's ``/`` between integer literals is true division, and C's is not.

Canonicalize lands a scalar square root on an interstate edge as ``beta ** (1 / 2)``. Printed verbatim
that is ``pow(beta, 0)`` in C and C++: bdf_newton_krylov's GMRES norms all came out 1.0, so every call ran
its full restart and the CPF drop-in took over 50 s where the reference takes under one.
"""
import numpy as np
import pytest

import dace
from dace import cpf_lowering
from dace.codegen.common import unparse_interstate_edge


@pytest.mark.parametrize('code, cpp, c', [
    ('beta ** (1 / 2)', 'dace::math::sqrt(beta)', 'sqrt(beta)'),
    ('beta ** (1 / 3)', 'dace::math::pow(beta, (1.0 / 3))', 'pow(beta, (1.0 / 3))'),
    ('(7 / 2)', '(7.0 / 2)', '(7.0 / 2)'),
])
def test_an_integer_literal_quotient_keeps_python_true_division(code, cpp, c):
    sdfg = dace.SDFG('literal_quotient')
    sdfg.add_symbol('beta', dace.float64)
    assert unparse_interstate_edge(code, sdfg) == cpp
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE_C):
        assert unparse_interstate_edge(code, sdfg) == c


def test_a_square_root_on_an_interstate_edge_computes_the_root():
    sdfg = dace.SDFG('interstate_sqrt')
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_symbol('x', dace.float64)
    sdfg.add_symbol('r', dace.float64)
    first = sdfg.add_state('first', is_start_block=True)
    second = sdfg.add_state('second')
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={'r': 'x ** (1 / 2)'}))
    tasklet = second.add_tasklet('store', {}, {'o'}, 'o = r')
    second.add_edge(tasklet, 'o', second.add_write('out'), None, dace.Memlet('out[0]'))
    out = np.zeros(1)
    sdfg(out=out, x=6.25)
    np.testing.assert_array_equal(out, [2.5])
