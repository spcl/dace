# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop with a data-dependent ``break`` keeps its exit through canonicalization and CPF rendering.

Splitting such a loop at an ``if k == 1`` guard, or unrolling it, leaves the ``break`` outside every
loop: the unit does not compile, and a break kept inside one split segment would still run the
segments after it. Both loops here compute the numbers Python computes, whichever iteration exits.
"""
import numpy as np
import pytest

import dace
from dace.codegen.cpf import render
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.codegen.cpf.conftest import assert_matches, build_standalone, call_standalone

N = dace.symbol('N')


@dace.program
def early_exit_long(x: dace.float64[N], r: dace.float64[N], tol: dace.float64):
    p = np.zeros_like(x)
    for k in range(1, 26):
        if np.sqrt(np.dot(r, r)) <= tol:
            break
        if k == 1:
            p[:] = r
        else:
            p[:] = 0.5 * p + r
        x[:] = x + 0.25 * p
        r[:] = 0.5 * r


@dace.program
def early_exit_short(x: dace.float64[N], r: dace.float64[N], tol: dace.float64):
    p = np.zeros_like(x)
    for k in range(1, 5):
        if np.sqrt(np.dot(r, r)) <= tol:
            break
        if k == 1:
            p[:] = r
        else:
            p[:] = 0.5 * p + r
        x[:] = x + 0.25 * p
        r[:] = 0.5 * r


@pytest.mark.parametrize('language', ['c++', 'c'])
@pytest.mark.parametrize('program', [early_exit_long, early_exit_short], ids=['split_candidate', 'unroll_candidate'])
def test_a_data_dependent_break_exits_the_loop_it_was_written_in(program, language):
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = f'{program.name}_{"cpp" if language == "c++" else "c"}'
    canonicalize(sdfg, validate=True, validate_all=False, target='cpu')
    finalize_for_target(sdfg, 'cpu', validate=True)
    result = render(sdfg, language=language)
    library = build_standalone(result.code, sdfg.name, language=language)
    n = 8
    # The residual norm starts near 4.2 and halves per iteration: exit on the first check, on the
    # fourth, or never.
    for tol in (100.0, 1.0, 0.0):
        expected = {'x': np.linspace(0.0, 1.0, n), 'r': np.linspace(1.0, 2.0, n)}
        got = {name: value.copy() for name, value in expected.items()}
        program.f(expected['x'], expected['r'], tol)
        call_standalone(library, result.sdfg, {'x': got['x'], 'r': got['r'], 'tol': tol, 'N': n})
        assert_matches(expected, got, f'{sdfg.name}(tol={tol})')
