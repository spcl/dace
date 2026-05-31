# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :meth:`dace.sdfg.state.SDFGState.symbols_defined_at`.

Locks down which symbols are reported as "defined" at a given node. Memlet
propagation across ``NestedSDFG`` boundaries reads this set to decide which
symbols may appear in the propagated outer memlet -- any inner subset symbol
that is NOT in this set falls back to the array's full extent (see
``propagate_memlets_nested_sdfg`` in ``dace/sdfg/propagation.py``). Missing
symbols here therefore widen propagation and hide loop-carried dependencies
downstream (the cloudsc ``for_1133`` / ``for_430`` shape).

The test file exercises both contributors that ``symbols_defined_at`` walks:

* enclosing ``LoopRegion`` loop variables (the for-loop side)
* enclosing dataflow scope nodes' new symbols (the map side, including
  ``MapEntry``'s ``Map`` parameters AND any non-pass-through input connector
  that supplies a dynamic Map range or a scope-local parameter)
"""
import dace
import pytest

from dace.sdfg.state import LoopRegion

N = dace.symbol('N')


def test_global_sdfg_symbol_visible():
    """Global ``SDFG.symbols`` entries reach every node."""
    sdfg = dace.SDFG('g')
    sdfg.add_symbol('K', dace.int32)
    sdfg.add_array('A', [10], dace.float64)
    s = sdfg.add_state('s')
    t = s.add_tasklet('t', {}, {'o'}, 'o = K')
    w = s.add_write('A')
    s.add_edge(t, 'o', w, None, dace.Memlet('A[0]'))

    syms = s.symbols_defined_at(t)
    assert 'K' in syms


def test_enclosing_loop_region_var_visible_at_state_node():
    """The loop variable of an enclosing ``LoopRegion`` is reported as defined."""
    sdfg = dace.SDFG('one_loop')
    sdfg.add_array('A', [N], dace.float64)
    loop = LoopRegion('L', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    s = loop.add_state('s', is_start_block=True)
    t = s.add_tasklet('t', {}, {'o'}, 'o = i')
    w = s.add_write('A')
    s.add_edge(t, 'o', w, None, dace.Memlet('A[i]'))

    syms = s.symbols_defined_at(t)
    assert 'i' in syms, 'enclosing LoopRegion loop variable must be visible'


def test_triple_nested_loop_regions_all_visible():
    """Every enclosing loop variable up the parent-region chain is visible."""
    sdfg = dace.SDFG('triple')
    sdfg.add_array('A', [10], dace.float64)
    L1 = LoopRegion('L1', 'a < 10', 'a', 'a = 0', 'a = a + 1')
    sdfg.add_node(L1)
    L2 = LoopRegion('L2', 'b < 10', 'b', 'b = 0', 'b = b + 1')
    L1.add_node(L2)
    L3 = LoopRegion('L3', 'c < 10', 'c', 'c = 0', 'c = c + 1')
    L2.add_node(L3)
    s = L3.add_state('s', is_start_block=True)
    t = s.add_tasklet('t', {}, {'o'}, 'o = a + b + c')
    w = s.add_write('A')
    s.add_edge(t, 'o', w, None, dace.Memlet('A[a]'))

    syms = s.symbols_defined_at(t)
    assert {'a', 'b', 'c'} <= set(syms), f'all enclosing loop vars must be visible; got {sorted(syms)}'


def test_map_iteration_var_visible_inside_scope():
    """The ``MapEntry``'s map parameter is visible at every node inside the scope."""
    sdfg = dace.SDFG('m')
    sdfg.add_array('A', [10], dace.float64)
    s = sdfg.add_state('s')
    me, mx = s.add_map('m', {'i': '0:10'})
    me.add_in_connector('IN_A')
    me.add_out_connector('OUT_A')
    A_r = s.add_read('A')
    A_w = s.add_write('A')
    t = s.add_tasklet('t', {'a'}, {'o'}, 'o = a')
    s.add_edge(A_r, None, me, 'IN_A', dace.Memlet('A[0:10]'))
    s.add_edge(me, 'OUT_A', t, 'a', dace.Memlet('A[i]'))
    mx.add_in_connector('IN_A')
    mx.add_out_connector('OUT_A')
    s.add_edge(t, 'o', mx, 'IN_A', dace.Memlet('A[i]'))
    s.add_edge(mx, 'OUT_A', A_w, None, dace.Memlet('A[0:10]'))

    syms = s.symbols_defined_at(t)
    assert 'i' in syms, 'Map iteration variable must be visible inside the scope'


def test_nested_map_iteration_vars_both_visible():
    """Both outer and inner ``Map`` parameters are visible at the innermost node."""
    sdfg = dace.SDFG('nested_m')
    sdfg.add_array('A', [10, 10], dace.float64)
    s = sdfg.add_state('s')

    me_o, mx_o = s.add_map('mo', {'i': '0:10'})
    me_i, mx_i = s.add_map('mi', {'j': '0:10'})
    me_o.add_in_connector('IN_A')
    me_o.add_out_connector('OUT_A')
    me_i.add_in_connector('IN_A')
    me_i.add_out_connector('OUT_A')
    A_r = s.add_read('A')
    A_w = s.add_write('A')
    t = s.add_tasklet('t', {'a'}, {'o'}, 'o = a')

    s.add_edge(A_r, None, me_o, 'IN_A', dace.Memlet('A[0:10, 0:10]'))
    s.add_edge(me_o, 'OUT_A', me_i, 'IN_A', dace.Memlet('A[i, 0:10]'))
    s.add_edge(me_i, 'OUT_A', t, 'a', dace.Memlet('A[i, j]'))
    mx_o.add_in_connector('IN_A')
    mx_o.add_out_connector('OUT_A')
    mx_i.add_in_connector('IN_A')
    mx_i.add_out_connector('OUT_A')
    s.add_edge(t, 'o', mx_i, 'IN_A', dace.Memlet('A[i, j]'))
    s.add_edge(mx_i, 'OUT_A', mx_o, 'IN_A', dace.Memlet('A[i, 0:10]'))
    s.add_edge(mx_o, 'OUT_A', A_w, None, dace.Memlet('A[0:10, 0:10]'))

    syms = s.symbols_defined_at(t)
    assert {'i', 'j'} <= set(syms), f'both enclosing Map iter vars must be visible; got {sorted(syms)}'


def test_dynamic_non_passthrough_map_connector_visible():
    """A ``MapEntry`` input connector that does NOT start with ``IN_`` supplies
    a dynamic Map-range / scope-local parameter (e.g. a runtime upper bound).
    Its name must be reported as defined for nodes inside the scope so memlets
    written in terms of that parameter survive ``NSDFG``-boundary propagation
    without being widened to the array extent."""
    sdfg = dace.SDFG('dyn')
    sdfg.add_array('A', [100], dace.int32)
    sdfg.add_array('N_arr', [1], dace.int32)
    sdfg.add_array('out', [100], dace.int32)
    s = sdfg.add_state('s')

    # Map range parameterized by a dynamic input connector ``N_arr_val``.
    me, mx = s.add_map('m', {'i': '0:N_arr_val'})
    me.add_in_connector('N_arr_val')
    me.add_in_connector('IN_A')
    me.add_out_connector('OUT_A')
    N_read = s.add_read('N_arr')
    s.add_edge(N_read, None, me, 'N_arr_val', dace.Memlet('N_arr[0]'))
    A_read = s.add_read('A')
    s.add_edge(A_read, None, me, 'IN_A', dace.Memlet('A[0:100]'))

    t = s.add_tasklet('t', {'a'}, {'o'}, 'o = a')
    s.add_edge(me, 'OUT_A', t, 'a', dace.Memlet('A[i]'))
    mx.add_in_connector('IN_o')
    mx.add_out_connector('OUT_o')
    s.add_edge(t, 'o', mx, 'IN_o', dace.Memlet('out[i]'))
    out_w = s.add_write('out')
    s.add_edge(mx, 'OUT_o', out_w, None, dace.Memlet('out[0:100]'))

    syms = s.symbols_defined_at(t)
    assert 'i' in syms, 'Map iter var must be visible'
    assert 'N_arr_val' in syms, ('non-pass-through Map in-connector must be reported as defined inside the scope; '
                                 'without this, memlets that reference the dynamic-range parameter would widen '
                                 'to the array extent when propagated outward.')


def test_loop_region_and_map_combined_visible():
    """Stacked enclosing scopes (``LoopRegion`` outside, ``Map`` inside): a
    node inside the Map sees BOTH the LoopRegion's loop variable and the
    Map's iter variable. This is the cloudsc-style stack -- outer scan loop
    over levels, inner Map over columns."""
    sdfg = dace.SDFG('stack')
    sdfg.add_array('A', [10, 20], dace.float64)
    loop = LoopRegion('lk', 'jk < 10', 'jk', 'jk = 0', 'jk = jk + 1')
    sdfg.add_node(loop)
    s = loop.add_state('s', is_start_block=True)
    me, mx = s.add_map('m', {'jl': '0:20'})
    me.add_in_connector('IN_A')
    me.add_out_connector('OUT_A')
    t = s.add_tasklet('t', {'a'}, {'o'}, 'o = a')
    A_r = s.add_read('A')
    A_w = s.add_write('A')
    s.add_edge(A_r, None, me, 'IN_A', dace.Memlet('A[0:10, 0:20]'))
    s.add_edge(me, 'OUT_A', t, 'a', dace.Memlet('A[jk, jl]'))
    mx.add_in_connector('IN_A')
    mx.add_out_connector('OUT_A')
    s.add_edge(t, 'o', mx, 'IN_A', dace.Memlet('A[jk, jl]'))
    s.add_edge(mx, 'OUT_A', A_w, None, dace.Memlet('A[0:10, 0:20]'))

    syms = s.symbols_defined_at(t)
    assert 'jk' in syms, 'enclosing LoopRegion loop variable visible across Map scope'
    assert 'jl' in syms, 'Map iter variable visible inside its own scope'


def test_nsdfg_inside_map_inside_loop_region_propagation_endpoint():
    """The exact contract that ``propagate_memlets_nested_sdfg`` relies on:
    the symbols available at the ``NestedSDFG`` node include the enclosing
    ``LoopRegion``'s loop variable. Without ``jk`` reported here, any subset
    of the form ``arr[jk, ...]`` inside the nested SDFG widens to the array
    extent on propagation out -- the cloudsc ``for_1133`` failure mode.
    """
    sdfg = dace.SDFG('cloudsc_shape')
    sdfg.add_symbol('K', dace.int32)
    sdfg.add_array('A', [10, 20], dace.float64)

    loop = LoopRegion('lk', 'jk < K', 'jk', 'jk = 0', 'jk = jk + 1')
    sdfg.add_node(loop)
    s = loop.add_state('s', is_start_block=True)
    me, mx = s.add_map('m', {'jl': '0:20'})
    me.add_in_connector('IN_A')
    me.add_out_connector('OUT_A')
    mx.add_in_connector('IN_A')
    mx.add_out_connector('OUT_A')

    inner = dace.SDFG('inner')
    inner.add_symbol('jk', dace.int32)
    inner.add_symbol('jl', dace.int32)
    inner.add_array('A', [10, 20], dace.float64)
    si = inner.add_state('si')
    t_inner = si.add_tasklet('t', {'a'}, {'o'}, 'o = a')
    A_ri = si.add_read('A')
    A_wi = si.add_write('A')
    si.add_edge(A_ri, None, t_inner, 'a', dace.Memlet('A[jk, jl]'))
    si.add_edge(t_inner, 'o', A_wi, None, dace.Memlet('A[jk, jl]'))

    nsdfg = s.add_nested_sdfg(inner, inputs={'A'}, outputs={'A'}, symbol_mapping={'jk': 'jk', 'jl': 'jl'})
    A_r = s.add_read('A')
    A_w = s.add_write('A')
    s.add_edge(A_r, None, me, 'IN_A', dace.Memlet('A[0:10, 0:20]'))
    s.add_edge(me, 'OUT_A', nsdfg, 'A', dace.Memlet('A[0:10, 0:20]'))
    s.add_edge(nsdfg, 'A', mx, 'IN_A', dace.Memlet('A[0:10, 0:20]'))
    s.add_edge(mx, 'OUT_A', A_w, None, dace.Memlet('A[0:10, 0:20]'))

    syms = s.symbols_defined_at(nsdfg)
    assert 'jk' in syms, 'enclosing LoopRegion var jk must be defined at the NSDFG node'
    assert 'jl' in syms, 'enclosing Map iter var jl must be defined at the NSDFG node'
    assert 'K' in syms, 'SDFG-global symbol K must be defined'


def test_outside_any_scope_is_empty_of_local_scope_symbols():
    """At a state node that sits OUTSIDE any Map scope and outside any
    LoopRegion, no scope-local symbols are reported (only SDFG-global ones)."""
    sdfg = dace.SDFG('flat')
    sdfg.add_symbol('K', dace.int32)
    sdfg.add_array('A', [10], dace.float64)
    s = sdfg.add_state('s')
    t = s.add_tasklet('t', {}, {'o'}, 'o = K')
    w = s.add_write('A')
    s.add_edge(t, 'o', w, None, dace.Memlet('A[0]'))

    syms = s.symbols_defined_at(t)
    assert 'K' in syms
    # No scope-local names from a hypothetical outer scope.
    assert 'i' not in syms and 'j' not in syms


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
