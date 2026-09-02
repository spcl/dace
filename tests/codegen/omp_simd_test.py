# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for ``MarkSIMDMaps`` and the OpenMP ``simd`` clause it drives. """

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.sdfg import infer_types
from dace.sdfg import propagation
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.mark_simd_maps import MarkSIMDMaps


def entries(sdfg: dace.SDFG):
    return [(state, n) for state in sdfg.all_states() for n in state.nodes() if isinstance(n, nodes.MapEntry)]


def mark(sdfg: dace.SDFG):
    """ Runs the pass the way code generation does: after schedules are resolved. """
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    return MarkSIMDMaps().apply_pass(sdfg, {})


def test_leaf_map_is_marked_and_vectorized():
    """A one-dimensional leaf map takes the clause on its own ``parallel for``."""

    @dace.program
    def axpy(a: dace.float64[256], b: dace.float64[256]):
        for i in dace.map[0:256]:
            b[i] = a[i] * 2.0 + 1.0

    sdfg = axpy.to_sdfg(simplify=True)
    assert mark(sdfg)
    assert all(n.map.omp_simd for _, n in entries(sdfg))
    assert '#pragma omp parallel for simd' in sdfg.generate_code()[0].clean_code

    a = np.random.rand(256)
    b = np.zeros(256)
    sdfg(a=a, b=b)
    assert np.allclose(b, a * 2.0 + 1.0)


def test_multidimensional_map_vectorizes_its_innermost_dimension():
    """The clause vectorizes the loop it precedes, so the innermost dimension gets a map of its
    own (``MapExpansion``) and only that one is marked."""

    @dace.program
    def scale(a: dace.float64[64, 64], b: dace.float64[64, 64]):
        for i, j in dace.map[0:64, 0:64]:
            b[i, j] = a[i, j] * 2.0

    sdfg = scale.to_sdfg(simplify=True)
    assert len(entries(sdfg)) == 1
    assert mark(sdfg)

    all_entries = entries(sdfg)
    assert len(all_entries) == 2, "the innermost dimension must become a map of its own"
    marked = [n for _, n in all_entries if n.map.omp_simd]
    assert len(marked) == 1 and marked[0].map.params == ['j']

    code = sdfg.generate_code()[0].clean_code
    assert '#pragma omp parallel for' in code
    assert '#pragma omp simd' in code

    a = np.random.rand(64, 64)
    b = np.zeros((64, 64))
    sdfg(a=a, b=b)
    assert np.allclose(b, a * 2.0)


@pytest.mark.parametrize('extents', [(8, 12), (4, 6, 10)])
def test_the_marked_multidimensional_map_expands_in_order(extents):
    """The clause needs the innermost dimension in a loop of its own, so the pass runs
    ``MapExpansion`` on the maps it marks. What that leaves behind is asserted here: one
    single-dimension scope per dimension, in the original order and with the original ranges,
    nested and re-joined in mirror order, each scope handing the next its slice, and only the
    innermost one carrying the clause."""
    params = ['i', 'j', 'k'][:len(extents)]
    point = ', '.join(params)
    sdfg = dace.SDFG(f'expand_{len(extents)}d')
    sdfg.add_array('a', extents, dace.float64)
    sdfg.add_array('b', extents, dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet('nd', {
        p: f'0:{ext}'
        for p, ext in zip(params, extents)
    }, {'inp': dace.Memlet(f'a[{point}]')},
                             'o = inp * 2.0', {'o': dace.Memlet(f'b[{point}]')},
                             schedule=dtypes.ScheduleType.CPU_Multicore,
                             external_edges=True)
    sdfg.validate()
    original = next(n for _, n in entries(sdfg))
    assert original.map.params == params
    original_ranges = list(original.map.range.ranges)

    assert mark(sdfg)
    sdfg.validate()

    # One scope per dimension, and they form a single nest rather than siblings.
    scope = state.scope_dict()
    map_entries = [n for _, n in entries(sdfg)]
    assert len(map_entries) == len(extents)
    chain = [next(n for n in map_entries if scope[n] is None)]
    while True:
        inner = [n for n in map_entries if scope[n] is chain[-1]]
        if not inner:
            break
        assert len(inner) == 1
        chain.append(inner[0])
    assert len(chain) == len(extents)

    # Outermost first, one parameter each, ranges untouched -- a permuted nest computes garbage.
    assert [e.map.params for e in chain] == [[p] for p in params]
    assert [e.map.range.ranges[0] for e in chain] == original_ranges

    # Only the innermost loop is vectorized, and only it was made Sequential to be one.
    assert chain[0].map.schedule == dtypes.ScheduleType.CPU_Multicore
    assert all(e.map.schedule == dtypes.ScheduleType.Sequential for e in chain[1:])
    assert chain[-1].map.omp_simd
    assert not any(e.map.omp_simd for e in chain[:-1])

    # The exits mirror the entries: one per entry, re-joining innermost first.
    exits = [state.exit_node(e) for e in chain]
    assert len(set(id(x) for x in exits)) == len(chain)
    assert [state.entry_node(x) for x in exits] == chain
    assert all(state.edges_between(exits[i], exits[i - 1]) for i in range(len(chain) - 1, 0, -1))

    # Each entry hands the next its slice, and the tasklet still sees a single point.
    for i in range(len(chain) - 1):
        edge = next(e for e in state.edges_between(chain[i], chain[i + 1]) if e.data.data == 'a')
        assert str(edge.data.subset) == ', '.join(params[:i + 1] + [f'0:{ext}' for ext in extents[i + 1:]])
    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert str(state.in_edges(tasklet)[0].data.subset) == point
    assert str(state.out_edges(tasklet)[0].data.subset) == point
    written = next(e for e in state.out_edges(exits[0]) if e.data.data == 'b')
    assert str(written.data.subset) == ', '.join(f'0:{ext}' for ext in extents)

    # One vectorized loop in the nest, not one per level.
    assert sdfg.generate_code()[0].clean_code.count('#pragma omp simd') == 1

    a = np.random.rand(*extents)
    b = np.zeros(extents)
    sdfg(a=a, b=b)
    assert np.allclose(b, a * 2.0)


def test_outer_map_of_a_nest_is_not_marked():
    """A map whose body holds another map is not the loop the clause belongs on."""

    @dace.program
    def nested(a: dace.float64[32, 32], b: dace.float64[32, 32]):
        for i in dace.map[0:32]:
            for j in dace.map[0:32]:
                b[i, j] = a[i, j] + 1.0

    sdfg = nested.to_sdfg(simplify=True)
    mark(sdfg)
    marked = [n for _, n in entries(sdfg) if n.map.omp_simd]
    assert len(marked) == 1 and marked[0].map.params == ['j']


def test_minmax_reduction_withholds_the_clause():
    """``min``/``max`` combine with a NaN-preserving compare vectorizers do not reliably fold."""
    sdfg = dace.SDFG('minmax_wcr_map')
    sdfg.add_array('a', (128, ), dace.float64)
    sdfg.add_array('out', (1, ), dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet('reduce_max', {'i': '0:128'}, {'inp': dace.Memlet('a[i]')},
                             'o = inp', {'o': dace.Memlet('out[0]', wcr='lambda x, y: max(x, y)')},
                             schedule=dtypes.ScheduleType.CPU_Multicore,
                             external_edges=True)
    sdfg.validate()

    mark(sdfg)
    assert not any(n.map.omp_simd for _, n in entries(sdfg))
    assert ' simd' not in sdfg.generate_code()[0].clean_code


@pytest.mark.parametrize('target', ('hist[i]', 'hist[0]'))
def test_sequential_wcr_withholds_the_clause(target):
    """A Sequential map lowers WCR to a non-atomic read-modify-write in the loop body: a scatter can
    alias across lanes, and an accumulation into a fixed location carries across iterations. Neither
    is what ``simd`` asserts."""
    sdfg = dace.SDFG(f"seq_wcr_{target.replace('[', '_').replace(']', '')}")
    sdfg.add_array('a', (64, ), dace.float64)
    sdfg.add_array('hist', (64, ), dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet('accumulate', {'i': '0:64'}, {'inp': dace.Memlet('a[i]')},
                             'o = inp', {'o': dace.Memlet(target, wcr='lambda x, y: x + y')},
                             schedule=dtypes.ScheduleType.Sequential,
                             external_edges=True)
    sdfg.validate()

    mark(sdfg)
    assert not any(n.map.omp_simd for _, n in entries(sdfg))
    assert ' simd' not in sdfg.generate_code()[0].clean_code


def test_multicore_reduction_keeps_the_clause_and_the_value():
    """CPU_Multicore lowers a conflicted WCR through ``reduce_atomic``, which composes with the
    clause, so the sum still lands."""
    sdfg = dace.SDFG('multicore_wcr_map')
    sdfg.add_array('a', (1024, ), dace.float64)
    sdfg.add_array('out', (1, ), dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet('reduce_sum', {'i': '0:1024'}, {'inp': dace.Memlet('a[i]')},
                             'o = inp', {'o': dace.Memlet('out[0]', wcr='lambda x, y: x + y')},
                             schedule=dtypes.ScheduleType.CPU_Multicore,
                             external_edges=True)
    sdfg.validate()

    mark(sdfg)
    assert all(n.map.omp_simd for _, n in entries(sdfg))
    assert '#pragma omp parallel for simd' in sdfg.generate_code()[0].clean_code

    a = np.random.rand(1024)
    out = np.zeros(1)
    sdfg(a=a, out=out)
    assert np.allclose(out[0], a.sum())


def test_unexpanded_library_node_withholds_the_clause():
    """A library node in the body expands to a Map after this pass has run, and an
    ``omp parallel for`` lexically inside a simd region is not valid OpenMP."""
    sdfg = dace.SDFG('library_node_body')
    sdfg.add_array('a', (8, 16), dace.float64)
    sdfg.add_array('out', (8, ), dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    entry, exit_node = state.add_map('outer', {'i': '0:8'}, schedule=dtypes.ScheduleType.CPU_Multicore)
    red = state.add_reduce('lambda x, y: x + y', None, 0)
    state.add_memlet_path(state.add_read('a'), entry, red, memlet=dace.Memlet('a[i, 0:16]'), dst_conn='_in')
    state.add_memlet_path(red, exit_node, state.add_write('out'), memlet=dace.Memlet('out[i]'), src_conn='_out')
    sdfg.validate()

    mark(sdfg)
    assert not any(n.map.omp_simd for _, n in entries(sdfg))

    a = np.random.rand(8, 16)
    out = np.zeros(8)
    sdfg(a=a, out=out)
    assert np.allclose(out, a.sum(axis=1))


def test_tasklet_carrying_a_directive_withholds_the_clause():
    """The OpenMP Reduce expansion writes ``#pragma omp parallel for`` into the tasklet body, and
    that construct is illegal inside a simd region."""
    sdfg = dace.SDFG('directive_in_tasklet')
    sdfg.add_array('a', (8, ), dace.float64)
    sdfg.add_array('b', (8, ), dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet('carries_a_pragma', {'i': '0:8'}, {'inp': dace.Memlet('a[i]')},
                             '#pragma omp parallel for\nfor (int k = 0; k < 1; ++k) o = inp;',
                             {'o': dace.Memlet('b[i]')},
                             schedule=dtypes.ScheduleType.CPU_Multicore,
                             language=dace.Language.CPP,
                             external_edges=True)
    sdfg.validate()

    mark(sdfg)
    assert not any(n.map.omp_simd for _, n in entries(sdfg))
    assert ' simd' not in sdfg.generate_code()[0].clean_code


def double_tasklet(state) -> None:
    """``b_out[0] = a_in[0] * 2`` as plain dataflow."""
    t = state.add_tasklet('nt', {'a': None}, {'b': None}, 'b = a * 2.0')
    state.add_edge(state.add_read('a_in'), None, t, 'a', dace.Memlet('a_in[0]'))
    state.add_edge(t, 'b', state.add_write('b_out'), None, dace.Memlet('b_out[0]'))


def inner_sdfg(payload: str) -> dace.SDFG:
    nsdfg = dace.SDFG(f'inner_{payload}')
    nsdfg.add_array('a_in', [1], dace.float64)
    nsdfg.add_array('b_out', [1], dace.float64)
    if payload == 'loopregion':
        loop = LoopRegion('nloop', 'k < 1', 'k', 'k = 0', 'k = k + 1')
        nsdfg.add_node(loop, is_start_block=True)
        double_tasklet(loop.add_state(is_start_block=True))
    elif payload == 'backedge':
        s0 = nsdfg.add_state(is_start_block=True)
        s1 = nsdfg.add_state()
        double_tasklet(s1)
        # The cycle is the point; the conditions keep it from ever being taken.
        nsdfg.add_edge(s0, s1, dace.InterstateEdge(condition='1 > 0'))
        nsdfg.add_edge(s1, s0, dace.InterstateEdge(condition='0 > 1'))
    elif payload == 'map':
        nsdfg.add_state().add_mapped_tasklet('innermap',
                                             dict(k='0:1'),
                                             dict(a=dace.Memlet('a_in[0]')),
                                             'b = a * 2.0',
                                             dict(b=dace.Memlet('b_out[0]')),
                                             external_edges=True)
    elif payload == 'nested':
        nstate = nsdfg.add_state()
        deeper = nstate.add_nested_sdfg(inner_sdfg('straight'), {'a_in': None}, {'b_out': None})
        nstate.add_edge(nstate.add_read('a_in'), None, deeper, 'a_in', dace.Memlet('a_in[0]'))
        nstate.add_edge(deeper, 'b_out', nstate.add_write('b_out'), None, dace.Memlet('b_out[0]'))
    else:
        double_tasklet(nsdfg.add_state())
    return nsdfg


def outlined_body_sdfg(payload: str) -> dace.SDFG:
    sdfg = dace.SDFG(f'simd_outlined_{payload}')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('outermap', dict(i='0:10'), schedule=dtypes.ScheduleType.CPU_Multicore)
    body = state.add_nested_sdfg(inner_sdfg(payload), {'a_in': None}, {'b_out': None})
    state.add_memlet_path(state.add_read('A'), entry, body, dst_conn='a_in', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(body, exit_node, state.add_write('B'), src_conn='b_out', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('payload,leaf', [('straight', True), ('nested', True), ('loopregion', False),
                                          ('backedge', False), ('map', False)])
def test_outlined_body_is_opened_not_refused(payload, leaf):
    """An outlined body (what the Fortran frontend emits for every map) is opened rather than
    refused on sight; a loop -- ``LoopRegion`` or back edge -- or a map inside withholds the
    clause."""
    sdfg = outlined_body_sdfg(payload)
    mark(sdfg)
    outer = next(n for _, n in entries(sdfg) if n.map.params == ['i'])
    assert outer.map.omp_simd is leaf


def test_the_mark_survives_a_round_trip():
    """The pass marks Sequential maps too, so the property has to serialize for them -- otherwise
    a save/load between the pass and code generation silently drops the clause."""
    sdfg = dace.SDFG('roundtrip_seq_simd')
    sdfg.add_array('a', (8, ), dace.float64)
    sdfg.add_array('b', (8, ), dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet('m', {'i': '0:8'}, {'inp': dace.Memlet('a[i]')},
                             'o = inp * 2.0', {'o': dace.Memlet('b[i]')},
                             schedule=dtypes.ScheduleType.Sequential,
                             external_edges=True)
    next(n for _, n in entries(sdfg)).map.omp_simd = True

    restored = dace.SDFG.from_json(sdfg.to_json())
    assert all(n.map.omp_simd for _, n in entries(restored))


def test_expansion_does_not_trigger_whole_sdfg_propagation(monkeypatch):
    """Performance regression guard. ``apply_to`` propagates memlets over the whole SDFG once per
    application unless told not to, which is quadratic in the map count and hung code generation on
    a model-sized graph for hours. MapExpansion propagates the scope it rewrote by itself, and that
    scope is all this pass disturbs."""
    sdfg = dace.SDFG('propagation_scope')
    previous = None
    for k in range(4):
        sdfg.add_array(f'a{k}', (64, 64), dace.float64)
        sdfg.add_array(f'b{k}', (64, 64), dace.float64)
        state = sdfg.add_state(f's{k}', is_start_block=(k == 0))
        if previous is not None:
            sdfg.add_edge(previous, state, dace.InterstateEdge())
        previous = state
        state.add_mapped_tasklet(f'm{k}', {
            'i': '0:64',
            'j': '0:64'
        }, {'inp': dace.Memlet(f'a{k}[i, j]')},
                                 'o = inp * 2.0', {'o': dace.Memlet(f'b{k}[i, j]')},
                                 external_edges=True)
    sdfg.validate()

    whole_sdfg_calls = []
    monkeypatch.setattr(propagation, 'propagate_memlets_sdfg', lambda *args, **kwargs: whole_sdfg_calls.append(args))
    marked = mark(sdfg)

    assert len(marked) == 4
    assert not whole_sdfg_calls, 'expansion must not drag a whole-SDFG memlet propagation behind it'

    # What the expansion did touch is still propagated: the outer map hands the inner one a row.
    for state in sdfg.all_states():
        outer = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry) and n.map.params == ['i'])
        inner = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry) and n.map.params == ['j'])
        edge = next(e for e in state.edges_between(outer, inner) if e.data.data is not None)
        assert str(edge.data.subset) == 'i, 0:64'


def test_config_switch_suppresses_the_clause():
    """The escape hatch keeps the pass out of code generation entirely."""

    @dace.program
    def axpy2(a: dace.float64[128], b: dace.float64[128]):
        for i in dace.map[0:128]:
            b[i] = a[i] + 1.0

    sdfg = axpy2.to_sdfg(simplify=True)
    with dace.config.set_temporary('compiler', 'cpu', 'simd_maps', value=False):
        assert ' simd' not in sdfg.generate_code()[0].clean_code
    assert ' simd' in sdfg.generate_code()[0].clean_code


if __name__ == '__main__':
    test_leaf_map_is_marked_and_vectorized()
    test_multidimensional_map_vectorizes_its_innermost_dimension()
    for map_extents in ((8, 12), (4, 6, 10)):
        test_the_marked_multidimensional_map_expands_in_order(map_extents)
    test_outer_map_of_a_nest_is_not_marked()
    test_minmax_reduction_withholds_the_clause()
    for wcr_target in ('hist[i]', 'hist[0]'):
        test_sequential_wcr_withholds_the_clause(wcr_target)
    test_multicore_reduction_keeps_the_clause_and_the_value()
    test_unexpanded_library_node_withholds_the_clause()
    test_tasklet_carrying_a_directive_withholds_the_clause()
    for name, is_leaf in (('straight', True), ('nested', True), ('loopregion', False), ('backedge', False), ('map',
                                                                                                             False)):
        test_outlined_body_is_opened_not_refused(name, is_leaf)
    test_the_mark_survives_a_round_trip()
    with pytest.MonkeyPatch.context() as mp:
        test_expansion_does_not_trigger_whole_sdfg_propagation(mp)
    test_config_switch_suppresses_the_clause()
