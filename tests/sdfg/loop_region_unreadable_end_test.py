# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop whose condition is not ``i <op> X`` still has to type its own iterator.

``loop_analysis.get_loop_end`` reads an end bound out of the condition and returns ``None`` when the
condition is not that shape -- a compound test like ``i < N and i * i < N`` is not. Two places then
inferred the iterator's type from ``(start, step, end)`` unconditionally and handed that ``None`` to
``infer_expr_type``, which raised ``Cannot convert type <class 'NoneType'> to a Python AST`` from
inside codegen, for a loop that is otherwise perfectly well formed.

The iterator's type comes from the statements that DEFINE it -- its init and its update. The end
bound only ever participates through a comparison, so leaving it out when it cannot be read narrows
nothing. These tests pin that at both sites, and on the whole path through to a run.
"""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis

N = dace.symbol('N', dtype=dace.int64)


def compound_condition_sdfg(condition: str = 'i < N and i * i < N') -> dace.SDFG:
    sdfg = dace.SDFG('compound_cond')
    sdfg.add_array('a', [N], dace.float64)
    loop = LoopRegion('walk', condition, 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state('body', is_start_block=True)
    tasklet = state.add_tasklet('t', {'__in'}, {'__out'}, '__out = __in + 1.0')
    state.add_edge(state.add_read('a'), None, tasklet, '__in', dace.Memlet('a[i]'))
    state.add_edge(tasklet, '__out', state.add_write('a'), None, dace.Memlet('a[i]'))
    sdfg.validate()
    return sdfg


def loop_of(sdfg: dace.SDFG) -> LoopRegion:
    return next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion))


def test_the_end_bound_is_genuinely_unreadable():
    """Non-vacuity: if this ever starts returning a bound, the tests below stop testing anything."""
    assert loop_analysis.get_loop_end(loop_of(compound_condition_sdfg())) is None


def test_the_iterator_is_typed_from_init_and_step_alone():
    symbols = loop_of(compound_condition_sdfg()).new_symbols({})
    assert symbols == {'i': dace.int64}, symbols


def test_codegen_survives_an_unreadable_end():
    """The second site: the frame codegen infers the same three expressions independently."""
    code = '\n'.join(obj.clean_code for obj in compound_condition_sdfg().generate_code())
    assert 'i < N' in code.replace('(', '').replace(')', '')


@pytest.mark.parametrize('n', [1, 4, 32])
def test_the_compiled_loop_runs_the_right_iterations(n):
    sdfg = compound_condition_sdfg()
    got = np.zeros(n)
    sdfg(a=got, N=n)
    want = np.zeros(n)
    i = 0
    while i < n and i * i < n:
        want[i] += 1.0
        i += 1
    assert np.allclose(got, want), f'{got} != {want}'


if __name__ == '__main__':
    test_the_end_bound_is_genuinely_unreadable()
    test_the_iterator_is_typed_from_init_and_step_alone()
    test_codegen_survives_an_unreadable_end()
    test_the_compiled_loop_runs_the_right_iterations(32)
