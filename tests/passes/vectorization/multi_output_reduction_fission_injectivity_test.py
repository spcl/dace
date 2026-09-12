# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Guard test for :func:`_wcr_output_is_injective_rmw`.

:class:`_MultiOutputReductionMapFission` must fission a fused map only when it separates two
GENUINE cross-iteration reductions / contractions (gesummv ``tmp[i] += A[i,j]*x[j]``; ``j``
reduced), never when the map's WCR outputs are INJECTIVE per-element read-modify-writes
(s212 ``a[i] *= c[i]``; ``b[i] += a_snap[i+1]*d[i]``). Fissioning the injective case is both
unnecessary (the tiler widens each write as an ordinary ``TileStore``) and unsound: it detaches
the in-place ``a`` write from the anti-dependence snapshot ``a_split_snap = a`` that ordered it,
so codegen overwrites ``a`` before the snapshot reads it and miscompiles ``b``.

The classifier decides this per WCR output by whether the inner write index carries every map
param (injective) or omits one (reduced over). These tests pin that distinction directly.
"""
import dace
from dace.transformation.passes.vectorization.vectorize_multi_dim import _wcr_output_is_injective_rmw

N = dace.symbol('N')


def _map_with_wcr_writes(map_params, writes):
    """Build a one-state SDFG with a single map whose exit carries a ``+=`` WCR write per
    ``(array, subset)`` in ``writes``; return ``(state, map_exit, params)``."""
    sdfg = dace.SDFG('t')
    for arr, _ in writes:
        if arr not in sdfg.arrays:
            sdfg.add_array(arr, [N, N], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', {p: '0:N' for p in map_params})
    for arr, sub in writes:
        tasklet = state.add_tasklet(f't_{arr}', {}, {'o'}, 'o = 1.0')
        access = state.add_access(arr)
        state.add_edge(entry, None, tasklet, None, dace.Memlet())
        state.add_memlet_path(tasklet,
                              exit_node,
                              access,
                              src_conn='o',
                              memlet=dace.Memlet(data=arr, subset=sub, wcr='lambda a, b: a + b'))
    return state, exit_node, map_params


def test_injective_per_element_rmw_is_not_a_reduction():
    """s212 shape: a ``(i)`` map writing ``a[i]`` / ``b[i]`` -- every element written once,
    the sole map param present in each write -> injective, must NOT be counted for fission."""
    state, map_exit, params = _map_with_wcr_writes(['i'], [('a', 'i'), ('b', 'i')])
    assert _wcr_output_is_injective_rmw(state, map_exit, 'a', params) is True
    assert _wcr_output_is_injective_rmw(state, map_exit, 'b', params) is True


def test_contraction_reduces_over_inner_param():
    """gesummv shape: a collapsed ``(i, j)`` map writing ``tmp[i]`` / ``y[i]`` -- ``j`` is
    absent from the write index (reduced over) -> genuine contraction, IS fission-eligible."""
    state, map_exit, params = _map_with_wcr_writes(['i', 'j'], [('tmp', 'i'), ('y', 'i')])
    assert _wcr_output_is_injective_rmw(state, map_exit, 'tmp', params) is False
    assert _wcr_output_is_injective_rmw(state, map_exit, 'y', params) is False


def test_scalar_reduction_is_not_injective():
    """A scalar accumulator ``s[0]`` under an ``(i)`` map carries no map param in its write
    index -> reduced over ``i`` -> not injective (handled by the reduction paths, not fission)."""
    sdfg = dace.SDFG('t_scalar')
    sdfg.add_array('s', [1], dace.float64)
    state = sdfg.add_state()
    entry, map_exit = state.add_map('m', {'i': '0:N'})
    tasklet = state.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    access = state.add_access('s')
    state.add_edge(entry, None, tasklet, None, dace.Memlet())
    state.add_memlet_path(tasklet,
                          map_exit,
                          access,
                          src_conn='o',
                          memlet=dace.Memlet(data='s', subset='0', wcr='lambda a, b: a + b'))
    assert _wcr_output_is_injective_rmw(state, map_exit, 's', ['i']) is False


def test_partial_param_coverage_counts_as_reduction():
    """A ``(i, j)`` map whose write ``a[i]`` covers only ``i`` (``j`` reduced) is a reduction;
    a write ``a[i, j]`` covering both params is injective. Pins the all-params-present rule."""
    state, map_exit, params = _map_with_wcr_writes(['i', 'j'], [('a', 'i, j')])
    assert _wcr_output_is_injective_rmw(state, map_exit, 'a', params) is True
    state2, map_exit2, params2 = _map_with_wcr_writes(['i', 'j'], [('a', 'i')])
    assert _wcr_output_is_injective_rmw(state2, map_exit2, 'a', params2) is False


if __name__ == '__main__':
    test_injective_per_element_rmw_is_not_a_reduction()
    test_contraction_reduces_over_inner_param()
    test_scalar_reduction_is_not_injective()
    test_partial_param_coverage_counts_as_reduction()
    print('ok')
