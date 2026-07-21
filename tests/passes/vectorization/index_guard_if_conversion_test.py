# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An index guard only blocks if-conversion when the arms actually NEED it for their range.

If-conversion makes both arms' accesses unconditional, so it is unsound when the guard is what was
keeping them in bounds -- ``if i < N - 1: s += a[i + 1] * a[i + 1]`` would let lane ``i = N-1`` read
``a[N]``. But a guard that merely SELECTS a value is a different thing: TSVC s276's
``if i + 1 < N/2: a[i] = b[i] + c[i]*d[i] else: a[i] = b[i] + e[i]*d[i]`` indexes ``[i]`` in both
arms, in range for every ``i`` the map runs, so evaluating both and blending is exact.

Refusing on "the condition mentions the iteration symbol" conflates the two and leaves the second
kind scalar. :func:`arm_accesses_are_in_range_unguarded` separates them by proving the arms in
range over the FULL (unguarded) iteration range, so s276 lowers to a per-lane ``TileITE`` blend.
"""
import copy

import dace
import pytest
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
    arm_accesses_are_in_range_unguarded, condition_guards_iteration_symbol, provably_nonnegative)

N = dace.symbol('N')


def test_nonnegativity_uses_the_positive_symbol_assumption():
    """A shape symbol is a positive integer, so the top index of an ``N``-long array is in range --
    but ``N - 2`` stays indeterminate, so the proof still fails closed."""
    assert provably_nonnegative(dace.symbolic.pystr_to_symbolic('N - 1')) is True
    assert provably_nonnegative(dace.symbolic.pystr_to_symbolic('0')) is True
    assert provably_nonnegative(dace.symbolic.pystr_to_symbolic('-1')) is False
    assert provably_nonnegative(dace.symbolic.pystr_to_symbolic('N - 2')) is False


def _guarded_map(then_index: str, else_index: str):
    """``for i: if i + 1 < N: a[i] = b[<then_index>] else: a[i] = b[<else_index>]``."""
    sdfg = dace.SDFG('guarded')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)

    inner = dace.SDFG('body')
    inner.add_symbol('i', dace.int64)
    inner.add_array('a', [N], dace.float64)
    inner.add_array('b', [N], dace.float64)
    cond = ConditionalBlock('guard', sdfg=inner)
    inner.add_node(cond, is_start_block=True)
    for label, index in (('then', then_index), ('else', else_index)):
        from dace.sdfg.state import ControlFlowRegion
        branch = ControlFlowRegion(label, sdfg=inner)
        cond.add_branch(dace.properties.CodeBlock('i + 1 < N') if label == 'then' else None, branch)
        st = branch.add_state(f'{label}_body', is_start_block=True)
        t = st.add_tasklet(f'{label}_t', {'v'}, {'o'}, 'o = v * 2')
        st.add_edge(st.add_access('b'), None, t, 'v', dace.Memlet(f'b[{index}]'))
        st.add_edge(t, 'o', st.add_access('a'), None, dace.Memlet('a[i]'))

    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('m', dict(i='0:N'))
    ns = state.add_nested_sdfg(inner, {'b'}, {'a'}, symbol_mapping={'i': 'i', 'N': 'N'})
    state.add_memlet_path(state.add_access('b'), me, ns, dst_conn='b', memlet=dace.Memlet('b[0:N]'))
    state.add_memlet_path(ns, mx, state.add_access('a'), src_conn='a', memlet=dace.Memlet('a[0:N]'))
    return cond


def test_value_selecting_guard_is_if_convertible():
    """Both arms index ``[i]`` -- always in range, so the guard only picks a value."""
    cond = _guarded_map('i', 'i')
    assert condition_guards_iteration_symbol(cond) is True, 'the guard does mention the iteration symbol'
    assert arm_accesses_are_in_range_unguarded(cond) is True, 'but the arms never needed it for range'


def test_range_protecting_guard_is_refused():
    """An arm reading ``b[i + 1]`` runs off the end at ``i = N-1`` without the guard."""
    cond = _guarded_map('i + 1', 'i')
    assert arm_accesses_are_in_range_unguarded(cond) is False, 'the guard IS the bounds check here'


def test_s276_lowers_to_a_per_lane_blend():
    """End-to-end: s276 vectorizes (a ``TileITE`` blend) and stays value-correct.

    Skipped if the corpus harness is unavailable in this environment.
    """
    try:
        import tests.corpus.measure_parallelization as mp
    except Exception:
        pytest.skip('corpus harness unavailable')
    from dace.libraries.tileops.nodes import TileITE
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    base, checker = mp.CORPORA['tsvc'][1]('s276_d_single')
    sd = copy.deepcopy(base)
    mp.apply_config(sd, 'canon+vec', mp.cpu_params(4))

    blends = [n for n, _ in sd.all_nodes_recursive() if isinstance(n, TileITE)]
    assert blends, 's276 must lower its value-selecting guard to a per-lane blend, not stay scalar'

    fin = finalize_for_target(copy.deepcopy(sd), 'cpu')
    fin.name = 's276_index_guard_test'
    assert bool(checker(fin)), 's276 must be value-correct after if-conversion'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
