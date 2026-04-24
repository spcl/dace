# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for induction-variable detection in loop_analysis."""

import dace
import sympy
from dace import symbolic
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis


def _make_loop(loop_var='i', init='i = 0', cond='i < N', update='i = i + 1', inverted=False):
    sdfg = dace.SDFG(f'test_{loop_var}')
    loop = LoopRegion('L',
                      condition_expr=cond,
                      loop_var=loop_var,
                      initialize_expr=init,
                      update_expr=update,
                      inverted=inverted,
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state('body', is_start_block=True)
    return sdfg, loop


def test_basic_iv_simple_range():
    _, loop = _make_loop()
    ivs = loop_analysis.detect_induction_variables(loop)
    assert list(ivs) == ['i']
    assert ivs['i'].kind == 'basic'
    assert ivs['i'].start == 0
    assert ivs['i'].step == 1


def test_basic_iv_symbolic_bounds_and_step():
    _, loop = _make_loop(init='i = A', cond='i < B', update='i = i + 3')
    ivs = loop_analysis.detect_induction_variables(loop)
    assert ivs['i'].start == symbolic.pystr_to_symbolic('A')
    assert ivs['i'].step == 3


def test_basic_iv_reversed_loop():
    _, loop = _make_loop(init='i = N', cond='i >= 0', update='i = i - 1')
    ivs = loop_analysis.detect_induction_variables(loop)
    assert ivs['i'].start == symbolic.pystr_to_symbolic('N')
    assert ivs['i'].step == -1


def test_basic_iv_inverted_loop():
    _, loop = _make_loop(inverted=True)
    ivs = loop_analysis.detect_induction_variables(loop)
    assert 'i' in ivs
    assert ivs['i'].kind == 'basic'


def test_derived_iv_interstate_edge():
    _, loop = _make_loop()
    b2 = loop.add_state('b2')
    loop.add_edge(loop.start_block, b2, dace.InterstateEdge(assignments={'j': '2*i + 5'}))
    ivs = loop_analysis.detect_induction_variables(loop)
    assert set(ivs) == {'i', 'j'}
    assert ivs['j'].kind == 'derived'
    assert ivs['j'].start == 5  # 2*0 + 5
    assert ivs['j'].step == 2   # 2*1


def test_derived_iv_chained():
    _, loop = _make_loop()
    b2 = loop.add_state('b2')
    b3 = loop.add_state('b3')
    loop.add_edge(loop.start_block, b2, dace.InterstateEdge(assignments={'j': '2*i + 1'}))
    loop.add_edge(b2, b3, dace.InterstateEdge(assignments={'k': '3*j + 2'}))
    ivs = loop_analysis.detect_induction_variables(loop)
    assert set(ivs) == {'i', 'j', 'k'}
    # j = 2*i + 1 => start=1, step=2
    # k = 3*j + 2 => start=3*1 + 2 = 5, step=3*2 = 6
    assert ivs['k'].start == 5
    assert ivs['k'].step == 6


def test_derived_iv_tasklet_scalar():
    sdfg = dace.SDFG('t')
    loop = LoopRegion('L', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    sdfg.add_scalar('j', dace.int64, transient=True)
    body = loop.add_state('body', is_start_block=True)
    t = body.add_tasklet('derive', {}, {'j'}, 'j = 2 * i + 7')
    an = body.add_access('j')
    body.add_edge(t, 'j', an, None, dace.Memlet('j[0]'))
    ivs = loop_analysis.detect_induction_variables(loop)
    assert 'j' in ivs
    assert ivs['j'].kind == 'derived'
    assert ivs['j'].start == 7
    assert ivs['j'].step == 2


def test_non_affine_rhs_not_classified():
    _, loop = _make_loop()
    b2 = loop.add_state('b2')
    loop.add_edge(loop.start_block, b2, dace.InterstateEdge(assignments={'j': 'i*i'}))
    ivs = loop_analysis.detect_induction_variables(loop)
    assert 'j' not in ivs
    assert set(ivs) == {'i'}


def test_self_referential_assignment_not_classified():
    _, loop = _make_loop()
    b2 = loop.add_state('b2')
    loop.add_edge(loop.start_block, b2, dace.InterstateEdge(assignments={'j': 'j + i'}))
    ivs = loop_analysis.detect_induction_variables(loop)
    assert 'j' not in ivs


def test_self_referential_step_rejects_loop():
    _, loop = _make_loop(init='i = 1', update='i = i * 2')
    ivs = loop_analysis.detect_induction_variables(loop)
    assert ivs == {}


def test_missing_loop_variable_returns_empty():
    sdfg = dace.SDFG('t')
    loop = LoopRegion('L', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    assert loop_analysis.detect_induction_variables(loop) == {}


def test_missing_init_returns_empty():
    sdfg = dace.SDFG('t')
    loop = LoopRegion('L', condition_expr='i < N', loop_var='i', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    assert loop_analysis.detect_induction_variables(loop) == {}


def test_nested_loops_outer_iv_invariant_to_inner():
    sdfg = dace.SDFG('t')
    outer = LoopRegion('outer', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion('inner', condition_expr='j < M', loop_var='j', initialize_expr='j = 0', update_expr='j = j + 1')
    outer.add_node(inner, is_start_block=True)
    inner.add_state('body', is_start_block=True)
    # Inner has a derived IV k = i + 2*j — i is constant from inner's viewpoint.
    b2 = inner.add_state('b2')
    inner.add_edge(inner.start_block, b2, dace.InterstateEdge(assignments={'k': 'i + 2*j'}))
    inner_ivs = loop_analysis.detect_induction_variables(inner)
    assert set(inner_ivs) == {'j', 'k'}  # outer i is NOT the inner's IV
    assert inner_ivs['k'].step == 2
    # Start = i + 2*0 = i (symbolic)
    assert inner_ivs['k'].start == symbolic.pystr_to_symbolic('i')


def test_affine_in_iv_pure_constant():
    _, loop = _make_loop()
    ivs = loop_analysis.detect_induction_variables(loop)
    result = loop_analysis.affine_in_iv(symbolic.pystr_to_symbolic('42'), ivs)
    assert result == (None, sympy.Integer(0), symbolic.pystr_to_symbolic('42'))


def test_affine_in_iv_pure_invariant():
    _, loop = _make_loop()
    ivs = loop_analysis.detect_induction_variables(loop)
    iv_name, scale, offset = loop_analysis.affine_in_iv(symbolic.pystr_to_symbolic('N + 3'), ivs)
    assert iv_name is None
    assert scale == 0
    assert offset == symbolic.pystr_to_symbolic('N + 3')


def test_affine_in_iv_rejects_when_invariant_syms_mismatch():
    _, loop = _make_loop()
    ivs = loop_analysis.detect_induction_variables(loop)
    # M is not in the permitted invariant set => reject.
    assert loop_analysis.affine_in_iv(symbolic.pystr_to_symbolic('M + 3'), ivs, invariant_syms={'N'}) is None
    # N alone is allowed.
    result = loop_analysis.affine_in_iv(symbolic.pystr_to_symbolic('N + 3'), ivs, invariant_syms={'N'})
    assert result is not None and result[0] is None


def test_affine_in_iv_rejects_two_ivs():
    _, loop = _make_loop()
    b2 = loop.add_state('b2')
    loop.add_edge(loop.start_block, b2, dace.InterstateEdge(assignments={'j': '2*i + 1'}))
    ivs = loop_analysis.detect_induction_variables(loop)
    # i + j mixes two IVs — no single-IV affine form.
    assert loop_analysis.affine_in_iv(symbolic.pystr_to_symbolic('i + j'), ivs) is None


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
