# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Quotients through ``symstr``, which reprints every interstate-edge assignment on save and on load.

bdf_newton_krylov's ``np.sqrt(s / (2.0 * N * N))`` came back from one save as ``0.5*s``: the power in the
denominator printed as the bare product ``(N) * (N)``, so the division took only its first factor. Its
``sqrt`` came back as ``x ** (1/2)``, and text divides by its operands' types, so C read ``pow(x, 0)``.
"""
import numpy as np
import pytest
import sympy

import dace
from dace import symbolic
from dace.symbolic import symstr

X = symbolic.symbol('x', dace.float64)
N = symbolic.symbol('N', dace.int64)


@pytest.mark.parametrize('power', [2, 3, -2])
@pytest.mark.parametrize('cpp_mode', [False, True])
def test_an_integer_power_in_a_denominator_stays_one_operand(power, cpp_mode):
    printed = symstr(X / N**power, cpp_mode=cpp_mode)
    scope = {'x': 5.0, 'N': 3, 'reciprocal': lambda value: 1.0 / value}
    assert eval(printed, scope) == pytest.approx(5.0 / 3**power), printed


@pytest.mark.parametrize('expr, python, cpp', [
    (N + sympy.Rational(1, 2), '(N + (1.0 / 2))', '(N + (1.0 / 2))'),
    (N - sympy.Rational(3, 4), '(N + (-3.0 / 4))', '(N + (-3.0 / 4))'),
    (X + sympy.Rational(1, 2), '(x + (1.0 / 2))', '(x + (double(1) / double(2)))'),
])
def test_a_rational_prints_as_a_floating_quotient(expr, python, cpp):
    assert symstr(expr) == python
    assert symstr(expr, cpp_mode=True) == cpp


@pytest.mark.parametrize('expr, python', [
    (N / 2, '(int_floor(N, 2))'),
    (X / 2, '(x/2)'),
    (X / (2 * N), '(x/(2*N))'),
])
def test_only_an_integer_quotient_becomes_a_floor_division(expr, python):
    assert symstr(expr) == python


def test_bdf_norms_survive_a_save_and_load(tmp_path):
    """The assignments bdf's Newton loop carries, with values chosen so each dropped factor shows."""
    assignments = {
        'r_norm': 's / (2.0 * N * N)',
        'r_cube': 's / N ** 3',
        'r_neg': 's / (-N ** 2)',
        'r_root': 'sqrt(s) + 0.0',
    }
    expected = {'r_norm': 6.25 / 32.0, 'r_cube': 6.25 / 64.0, 'r_neg': -6.25 / 16.0, 'r_root': 2.5}
    sdfg = dace.SDFG('bdf_norms')
    sdfg.add_array('out', [len(assignments)], dace.float64)
    sdfg.add_symbol('s', dace.float64)
    sdfg.add_symbol('N', dace.int64)
    for name in assignments:
        sdfg.add_symbol(name, dace.float64)
    first = sdfg.add_state('first', is_start_block=True)
    second = sdfg.add_state('second')
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments=assignments))
    for i, name in enumerate(assignments):
        tasklet = second.add_tasklet(f'store_{name}', {}, {'o'}, f'o = {name}')
        second.add_edge(tasklet, 'o', second.add_write('out'), None, dace.Memlet(f'out[{i}]'))
    path = tmp_path / 'bdf_norms.sdfgz'
    sdfg.save(str(path), compress=True)
    out = np.zeros(len(assignments))
    dace.SDFG.from_file(str(path))(out=out, s=6.25, N=4)
    np.testing.assert_allclose(out, list(expected.values()), rtol=1e-15)
