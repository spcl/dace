# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Structural tests for automatic ``#pragma omp simd`` / ``#pragma omp parallel for simd``
emission on innermost (leaf-body) maps.

``MarkSIMDMaps`` (run from code generation, gated by ``compiler.cpu.simd_maps``) owns the
decision and records it in ``Map.omp_simd``; the CPU targets only render the clause. These cases
pin what comes out the other end.

Runs every case against BOTH CPU code generators (``legacy`` and ``experimental_readable``,
via ``compiler.cpu.implementation``): the two are separate classes (``ExperimentalCPUCodeGen``
subclasses ``CPUCodeGen`` but the map-loop emitter lives in the shared base method), and
``experimental_readable`` also runs a pre-codegen sweep that inlines trivial (loop-free) nested
SDFGs -- so a NestedSDFG that has no internal control flow is gone by the time the leaf check
runs under ``experimental_readable`` while it is still present under ``legacy``. A NestedSDFG
that genuinely carries a loop is not inlinable by that sweep and is refused identically by both
(see ``test_map_with_uninlinable_nested_loop_gets_no_simd``).
"""
import re

import pytest

import dace
from dace import dtypes
from dace.config import Config, temporary_config
from dace.sdfg.state import LoopRegion

IMPLS = ['legacy', 'experimental_readable']

PRAGMA_RE = re.compile(r'#pragma omp[^\n]*')


def _pragmas(sdfg: dace.SDFG) -> list:
    code = sdfg.generate_code()[0].clean_code
    return PRAGMA_RE.findall(code)


def _leaf_map_sdfg(name: str, schedule, wcr=None, wcr_target_is_array=False, wcr_index_dependent=False):
    """A single map ``B[i] = A[i] + 1`` (or a WCR write into ``acc``/``hist`` instead of ``B``)."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [10], dace.float64)
    if wcr is None:
        sdfg.add_array('B', [10], dace.float64)
    elif wcr_target_is_array:
        sdfg.add_array('hist', [10], dace.float64)
    else:
        sdfg.add_scalar('acc', dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i='0:10'), schedule=schedule)
    t = state.add_tasklet('t', {'a'}, {'o'}, 'o = a + 1.0')
    a_acc = state.add_access('A')
    state.add_memlet_path(a_acc, me, t, dst_conn='a', memlet=dace.Memlet('A[i]'))
    if wcr is None:
        out_acc = state.add_access('B')
        out_mem = dace.Memlet('B[i]')
    elif wcr_target_is_array:
        out_acc = state.add_access('hist')
        idx = 'i' if wcr_index_dependent else '0'
        out_mem = dace.Memlet(f'hist[{idx}]', wcr=wcr)
    else:
        out_acc = state.add_access('acc')
        out_mem = dace.Memlet('acc[0]', wcr=wcr)
    state.add_memlet_path(t, mx, out_acc, src_conn='o', memlet=out_mem)
    sdfg.validate()
    return sdfg


def _nested_map_sdfg(name: str, outer_schedule, inner_schedule):
    """Outer map directly wrapping an inner leaf map, ``B[i,j] = A[i,j] + 1``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [4, 4], dace.float64)
    sdfg.add_array('B', [4, 4], dace.float64)
    state = sdfg.add_state()
    ome, omx = state.add_map('outer', dict(i='0:4'), schedule=outer_schedule)
    ime, imx = state.add_map('inner', dict(j='0:4'), schedule=inner_schedule)
    t = state.add_tasklet('t', {'a'}, {'o'}, 'o = a + 1.0')
    a_acc = state.add_access('A')
    b_acc = state.add_access('B')
    state.add_memlet_path(a_acc, ome, ime, t, dst_conn='a', memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(t, imx, omx, b_acc, src_conn='o', memlet=dace.Memlet('B[i, j]'))
    sdfg.validate()
    return sdfg


def _multidim_leaf_map_sdfg(name: str, schedule, collapse: int = 1):
    """One map over two dimensions -- a single MapEntry emitting a two-deep loop nest,
    ``B[i,j] = A[i,j] + 1``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [4, 4], dace.float64)
    sdfg.add_array('B', [4, 4], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i='0:4', j='0:4'), schedule=schedule)
    me.map.collapse = collapse
    t = state.add_tasklet('t', {'a'}, {'o'}, 'o = a + 1.0')
    a_acc = state.add_access('A')
    b_acc = state.add_access('B')
    state.add_memlet_path(a_acc, me, t, dst_conn='a', memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(t, mx, b_acc, src_conn='o', memlet=dace.Memlet('B[i, j]'))
    sdfg.validate()
    return sdfg


def _map_with_uninlinable_nested_sdfg(name: str, schedule):
    """A map whose body is a NestedSDFG carrying a real ``LoopRegion`` -- not flattenable by the
    experimental generator's pre-codegen inlining sweep, so both generators must see it as-is."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i='0:10'), schedule=schedule)

    inner = dace.SDFG('inner_' + name)
    inner.add_array('ai', [1], dace.float64)
    inner.add_array('bo', [1], dace.float64)
    inner.add_scalar('acc', dace.float64, transient=True)

    zero = inner.add_state('zero', is_start_block=True)
    t0 = zero.add_tasklet('z', {}, {'o'}, 'o = 0.0')
    zero.add_edge(t0, 'o', zero.add_access('acc'), None, dace.Memlet('acc[0]'))

    loop = LoopRegion('accloop', condition_expr='k < 4', loop_var='k', initialize_expr='k = 0', update_expr='k = k + 1')
    inner.add_node(loop)
    inner.add_edge(zero, loop, dace.InterstateEdge())
    body = loop.add_state('body', is_start_block=True)
    t1 = body.add_tasklet('accum', {'inacc'}, {'outacc'}, 'outacc = inacc + 1.0')
    body.add_edge(body.add_access('acc'), None, t1, 'inacc', dace.Memlet('acc[0]'))
    body.add_edge(t1, 'outacc', body.add_access('acc'), None, dace.Memlet('acc[0]'))

    final = inner.add_state('final')
    tf = final.add_tasklet('write', {'inacc'}, {'o'}, 'o = inacc')
    final.add_edge(final.add_access('acc'), None, tf, 'inacc', dace.Memlet('acc[0]'))
    final.add_edge(tf, 'o', final.add_access('bo'), None, dace.Memlet('bo[0]'))
    inner.add_edge(loop, final, dace.InterstateEdge())

    nsdfg = state.add_nested_sdfg(inner, {'ai'}, {'bo'})
    a_acc = state.add_access('A')
    b_acc = state.add_access('B')
    state.add_memlet_path(a_acc, me, nsdfg, dst_conn='ai', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(nsdfg, mx, b_acc, src_conn='bo', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('impl', IMPLS)
def test_sequential_leaf_gets_simd(impl):
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        sdfg = _leaf_map_sdfg('seq_leaf_' + impl, dtypes.ScheduleType.Sequential)
        pragmas = _pragmas(sdfg)
    assert any(p == '#pragma omp simd' for p in pragmas), pragmas
    assert not any('parallel' in p for p in pragmas), pragmas


@pytest.mark.parametrize('impl', IMPLS)
def test_multicore_leaf_gets_parallel_for_simd(impl):
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        sdfg = _leaf_map_sdfg('mc_leaf_' + impl, dtypes.ScheduleType.CPU_Multicore)
        pragmas = _pragmas(sdfg)
    assert any(p.startswith('#pragma omp parallel for') and 'simd' in p for p in pragmas), pragmas
    # The clause is on the SAME directive, not a second pragma.
    assert not any(p == '#pragma omp simd' for p in pragmas), pragmas


@pytest.mark.parametrize('impl', IMPLS)
def test_multicore_multidim_leaf_gets_no_simd(impl):
    """The pragma binds to the loop right after it, so on a two-dimensional map ``simd`` would
    ask for outer-loop vectorization of a nest instead of the element-wise vectorization it is
    written for."""
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        sdfg = _multidim_leaf_map_sdfg('mc_2d_' + impl, dtypes.ScheduleType.CPU_Multicore)
        pragmas = _pragmas(sdfg)
    outer = [p for p in pragmas if 'parallel' in p]
    assert len(outer) == 1, pragmas
    assert 'simd' not in outer[0], pragmas


@pytest.mark.parametrize('impl', IMPLS)
def test_a_collapse_hint_is_honoured_instead_of_vectorized(impl):
    """``collapse(k)`` and ``simd`` on the inner loop are contradictory: fusing k dimensions into
    one iteration space leaves no inner loop to vectorize. ``MarkSIMDMaps`` reads the hint and
    HONOURS it -- the nest keeps its collapsed form and takes no clause -- because taking it apart
    to vectorize answers a different question than the one asked and costs the combined trip count
    as thread parallelism (a ``[0:2, 0:1000000]`` nest would drop from 2,000,000-way to 2-way)."""
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        sdfg = _multidim_leaf_map_sdfg('mc_2d_coll_' + impl, dtypes.ScheduleType.CPU_Multicore, collapse=2)
        pragmas = _pragmas(sdfg)
    outer = [p for p in pragmas if 'parallel' in p]
    assert len(outer) == 1, pragmas
    assert 'collapse(2)' in outer[0], pragmas
    # Never both: the property says collapse, so no pragma anywhere may claim a vectorized loop.
    assert not any('simd' in p for p in pragmas), pragmas


@pytest.mark.parametrize('impl', IMPLS)
def test_multicore_outer_wrapping_inner_map_gets_no_simd(impl):
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        sdfg = _nested_map_sdfg('mc_outer_' + impl, dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.Sequential)
        pragmas = _pragmas(sdfg)
    outer = [p for p in pragmas if 'parallel' in p]
    assert len(outer) == 1, pragmas
    assert 'simd' not in outer[0], pragmas
    # The (leaf) inner Sequential map still gets its own pragma.
    assert any(p == '#pragma omp simd' for p in pragmas), pragmas


@pytest.mark.parametrize('impl', IMPLS)
def test_outer_sequential_wrapping_inner_map_gets_no_simd(impl):
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        sdfg = _nested_map_sdfg('seq_outer_' + impl, dtypes.ScheduleType.Sequential, dtypes.ScheduleType.Sequential)
        pragmas = _pragmas(sdfg)
    # Exactly one ``#pragma omp simd`` -- on the inner (leaf) loop, never the outer one.
    assert pragmas.count('#pragma omp simd') == 1, pragmas


@pytest.mark.parametrize('impl', IMPLS)
@pytest.mark.parametrize('schedule', [dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.Sequential])
def test_map_with_uninlinable_nested_loop_gets_no_simd(impl, schedule):
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        sdfg = _map_with_uninlinable_nested_sdfg('nsdfg_' + impl + '_' + schedule.name, schedule)
        code = sdfg.generate_code()[0].clean_code
        pragmas = _pragmas(sdfg)
    # The inner loop's own control flow (``k < 4``) must have survived codegen -- otherwise this
    # case would not be testing what it claims to.
    assert 'k = 0' in code, code
    assert not any('simd' in p for p in pragmas), pragmas


@pytest.mark.parametrize('impl', IMPLS)
def test_sequential_fixed_sum_wcr_gets_no_simd(impl):
    """A Sequential map lowers ANY WCR write to a plain, non-atomic ``wcr_fixed::reduce`` in the
    loop body -- an accumulation into a fixed location carries across iterations, which is exactly
    what ``simd`` asserts does not happen. The mark is withheld for every WCR here, fixed-target
    reduction included; the reduction-clause path belongs to the CPU_Multicore pragma."""
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        sdfg = _leaf_map_sdfg('seq_sum_' + impl, dtypes.ScheduleType.Sequential, wcr='lambda a, b: a + b')
        pragmas = _pragmas(sdfg)
    assert not any('simd' in p for p in pragmas), pragmas


@pytest.mark.parametrize('impl', IMPLS)
def test_sequential_minmax_wcr_gets_no_simd(impl):
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        sdfg = _leaf_map_sdfg('seq_max_' + impl, dtypes.ScheduleType.Sequential, wcr='lambda a, b: max(a, b)')
        pragmas = _pragmas(sdfg)
    assert not any('simd' in p for p in pragmas), pragmas


@pytest.mark.parametrize('impl', IMPLS)
def test_sequential_scatter_wcr_gets_no_simd(impl):
    """A WCR whose target subset depends on the map's own iteration variable (scatter, e.g. a
    histogram) is NOT a fixed-target reduction: ``_collect_omp_reductions`` does not cover it, and
    a Sequential map's WCR write lowers to a plain non-atomic ``wcr_fixed::reduce`` -- so a
    colliding target address across loop-carried SIMD lanes is a genuine hazard. The rule must
    withhold ``simd`` entirely rather than emit it bare."""
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        sdfg = _leaf_map_sdfg('seq_scatter_' + impl,
                              dtypes.ScheduleType.Sequential,
                              wcr='lambda a, b: a + b',
                              wcr_target_is_array=True,
                              wcr_index_dependent=True)
        pragmas = _pragmas(sdfg)
    assert not any('simd' in p for p in pragmas), pragmas


@pytest.mark.parametrize('impl', IMPLS)
def test_multicore_scatter_wcr_still_gets_simd(impl):
    """Unlike Sequential, CPU_Multicore always lowers an uncovered (scatter) WCR write through an
    atomic (``wcr_fixed::reduce_atomic``), which composes correctly with ``simd`` -- so the leaf
    check alone gates the pragma here, not WCR coverage."""
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        sdfg = _leaf_map_sdfg('mc_scatter_' + impl,
                              dtypes.ScheduleType.CPU_Multicore,
                              wcr='lambda a, b: a + b',
                              wcr_target_is_array=True,
                              wcr_index_dependent=True)
        pragmas = _pragmas(sdfg)
    assert any(p.startswith('#pragma omp parallel for') and 'simd' in p for p in pragmas), pragmas


def _map_with_nested_wcr_sdfg(name: str, schedule, wcr: str):
    """A map whose body is a NestedSDFG holding the WCR write -- the frontend-outlined scatter
    shape ``hist[bin[i]] (op)= w[i]``. The nested-SDFG-to-MapExit edge is a PLAIN write, so the
    only WCR is one level down; the scatter index is a symbol the body binds from its own read.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('idx', [10], dace.int64)
    sdfg.add_array('w', [10], dace.float64)
    sdfg.add_array('hist', [10], dace.float64)
    state = sdfg.add_state()

    body = dace.SDFG('body_' + name)
    body.add_scalar('b_in', dace.int64)
    body.add_scalar('w_in', dace.float64)
    body.add_array('oc', [10], dace.float64)
    body.add_symbol('bsym', dace.int64)
    body.add_scalar('b_scal', dace.int64, transient=True)
    read = body.add_state('read', is_start_block=True)
    rd = read.add_tasklet('rd', {'__b'}, {'__o'}, '__o = __b')
    read.add_edge(read.add_read('b_in'), None, rd, '__b', dace.Memlet('b_in[0]'))
    read.add_edge(rd, '__o', read.add_access('b_scal'), None, dace.Memlet('b_scal[0]'))
    accum = body.add_state('accum')
    body.add_edge(read, accum, dace.InterstateEdge(assignments={'bsym': 'b_scal'}))
    ac = accum.add_tasklet('acc', {'__w'}, {'__o'}, '__o = __w')
    accum.add_edge(accum.add_read('w_in'), None, ac, '__w', dace.Memlet('w_in[0]'))
    accum.add_edge(ac, '__o', accum.add_write('oc'), None, dace.Memlet(data='oc', subset='bsym', wcr=wcr))

    me, mx = state.add_map('m', dict(i='0:10'), schedule=schedule)
    nsdfg = state.add_nested_sdfg(body, {'b_in', 'w_in'}, {'oc'})
    state.add_memlet_path(state.add_read('idx'), me, nsdfg, dst_conn='b_in', memlet=dace.Memlet('idx[i]'))
    state.add_memlet_path(state.add_read('w'), me, nsdfg, dst_conn='w_in', memlet=dace.Memlet('w[i]'))
    mx.add_in_connector('IN_hist')
    mx.add_out_connector('OUT_hist')
    state.add_edge(nsdfg, 'oc', mx, 'IN_hist', dace.Memlet('hist[0:10]'))
    state.add_edge(mx, 'OUT_hist', state.add_write('hist'), None, dace.Memlet('hist[0:10]'))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('impl', IMPLS)
def test_sequential_wcr_inside_nested_body_gets_no_simd(impl):
    """The WCR hazard does not have to sit on the map exit. An outlined body carries the
    accumulate INSIDE its NestedSDFG, and a Sequential map still lowers it to a plain non-atomic
    ``wcr_fixed::reduce``; vectorizing colliding scatter addresses across lanes DROPS updates
    (measured: ~20% of a 4000-element histogram). The scan must look through the body."""
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        sdfg = _map_with_nested_wcr_sdfg('seq_nested_wcr_' + impl,
                                         dtypes.ScheduleType.Sequential,
                                         wcr='lambda a, b: a + b')
        pragmas = _pragmas(sdfg)
    assert not any('simd' in p for p in pragmas), pragmas


@pytest.mark.parametrize('impl', IMPLS)
def test_multicore_minmax_wcr_inside_nested_body_gets_no_simd(impl):
    """Same blind spot on the CPU_Multicore side: ``min``/``max`` withholds the clause there, and
    a nested-body WCR must trigger that refusal exactly as an exit-edge one does."""
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        sdfg = _map_with_nested_wcr_sdfg('mc_nested_minmax_' + impl,
                                         dtypes.ScheduleType.CPU_Multicore,
                                         wcr='lambda a, b: max(a, b)')
        pragmas = _pragmas(sdfg)
    assert not any('simd' in p for p in pragmas), pragmas
    assert any(p == '#pragma omp parallel for' for p in pragmas), pragmas


@pytest.mark.parametrize('impl', IMPLS)
def test_config_off_switch_disables_both_rules(impl):
    """``simd_maps`` keeps ``MarkSIMDMaps`` out of code generation, so no map is marked and
    neither schedule renders a clause."""
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        Config.set('compiler', 'cpu', 'simd_maps', value=False)
        seq_pragmas = _pragmas(_leaf_map_sdfg('seq_off_' + impl, dtypes.ScheduleType.Sequential))
        mc_pragmas = _pragmas(_leaf_map_sdfg('mc_off_' + impl, dtypes.ScheduleType.CPU_Multicore))
    assert not any('simd' in p for p in seq_pragmas), seq_pragmas
    assert not any('simd' in p for p in mc_pragmas), mc_pragmas
    assert any(p == '#pragma omp parallel for' for p in mc_pragmas), mc_pragmas


@pytest.mark.parametrize('impl', IMPLS)
def test_defaults_are_on(impl):
    with temporary_config():
        Config.set('compiler', 'cpu', 'implementation', value=impl)
        assert Config.get_bool('compiler', 'cpu', 'simd_maps') is True


if __name__ == '__main__':
    test_sequential_leaf_gets_simd('legacy')
    test_sequential_leaf_gets_simd('experimental_readable')
    test_multicore_leaf_gets_parallel_for_simd('legacy')
    test_multicore_leaf_gets_parallel_for_simd('experimental_readable')
