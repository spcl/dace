# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the shared map-loop style knobs ``compiler.cpu.codegen_params.loop_index_type`` and
    ``loop_bound_cmp``. Both live in the shared cpu.py emitter, so they affect the legacy generator
    too -- their defaults must therefore reproduce today's loop verbatim. """
import numpy
import pytest

import dace
from dace.config import set_temporary

N = dace.symbol('N')


@dace.program
def double_it(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] * 2.0


@dace.program
def strided(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N:2]:  # non-unit stride: `!=` is unsound here and must fall back
        B[i] = A[i] * 2.0


def generate(program, implementation='legacy', loop_index_type='auto', loop_bound_cmp='lt'):
    sdfg = program.to_sdfg(simplify=True)
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'loop_index_type', value=loop_index_type), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'loop_bound_cmp', value=loop_bound_cmp):
        return '\n'.join(obj.code for obj in sdfg.generate_code() if obj.language == 'cpp')


def loop_lines(code):
    return [line.strip() for line in code.splitlines() if line.strip().startswith('for (')]


@pytest.mark.parametrize('implementation', ['legacy', 'experimental_readable'])
def test_defaults_emit_the_historical_loop(implementation):
    """The default spelling must be byte-identical to the pre-knob emitter: `for (auto i = ...; i <
    end + 1; i += ...)`. This is what keeps legacy unaffected."""
    lines = loop_lines(generate(double_it, implementation))
    assert lines, 'no loop emitted'
    assert any(line.startswith('for (auto ') and ' < ' in line for line in lines)
    assert not any('<=' in line or '!=' in line for line in lines)


@pytest.mark.parametrize('loop_index_type, expected', [('auto', 'for (auto '), ('int64', 'for (long long '),
                                                       ('int32', 'for (int ')])
def test_loop_index_type(loop_index_type, expected):
    lines = loop_lines(generate(double_it, loop_index_type=loop_index_type))
    assert any(line.startswith(expected) for line in lines), lines


def test_loop_bound_cmp_le_drops_the_plus_one():
    lines = loop_lines(generate(double_it, loop_bound_cmp='le'))
    assert any('<=' in line for line in lines), lines
    # `i <= end` covers the same range without the `+ 1` add.
    assert not any('+ 1;' in line for line in lines), lines


def test_loop_bound_cmp_ne_on_unit_stride():
    lines = loop_lines(generate(double_it, loop_bound_cmp='ne'))
    assert any('!=' in line for line in lines), lines


def test_loop_bound_cmp_ne_supports_non_unit_stride():
    """`ne` handles any stride by normalising the bound to a value the counter actually lands on --
    a naive `i != end + 1` would be stepped over and never terminate."""
    lines = loop_lines(generate(strided, loop_bound_cmp='ne'))
    assert lines, 'no loop emitted'
    assert any('!=' in line for line in lines), lines
    assert any('int_ceil' in line for line in lines), lines


@pytest.mark.parametrize('loop_index_type', ['auto', 'int64', 'int32'])
@pytest.mark.parametrize('loop_bound_cmp', ['lt', 'le', 'ne'])
def test_every_combination_runs_correctly(loop_index_type, loop_bound_cmp):
    with set_temporary('compiler', 'cpu', 'codegen_params', 'loop_index_type', value=loop_index_type), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'loop_bound_cmp', value=loop_bound_cmp):
        A = numpy.random.default_rng(0).random(32)
        B = numpy.zeros(32)
        double_it(A=A, B=B, N=32)
        assert numpy.allclose(B, A * 2.0)


@pytest.mark.parametrize('n', [31, 32])
@pytest.mark.parametrize('loop_bound_cmp', ['lt', 'le', 'ne'])
def test_strided_terminates_and_is_correct(loop_bound_cmp, n):
    """The normalised `ne` bound is a correctness guard, not cosmetics: a naive `i != end + 1` steps
    over the bound and hangs. Both an odd and an even extent are covered, so the stride divides the
    range in one case and not the other."""
    with set_temporary('compiler', 'cpu', 'codegen_params', 'loop_bound_cmp', value=loop_bound_cmp):
        A = numpy.random.default_rng(0).random(n)
        B = numpy.zeros(n)
        strided(A=A, B=B, N=n)
        assert numpy.allclose(B[::2], A[::2] * 2.0)


if __name__ == '__main__':
    test_defaults_emit_the_historical_loop('legacy')
    test_loop_bound_cmp_ne_supports_non_unit_stride()
    test_strided_terminates_and_is_correct('ne', 31)
