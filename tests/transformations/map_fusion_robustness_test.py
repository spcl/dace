# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Robustness contract of ``MapFusionVertical`` and ``MapFusionHorizontal``.

Driven through ``SDFG.apply_transformations_repeated()`` neither transformation may ever
raise: a candidate it cannot handle has to be refused by ``can_be_applied()`` returning
``False``, and a candidate it accepts has to rewrite the SDFG without an exception and
leave it valid. A refusal must moreover be CLEAN -- the ``can_be_applied()`` safety nets
warn about what they swallow, and a warning means a precondition is missing rather than
checked, so the tests below assert on the warnings too.

``optimizer.match_exception`` is forced on for every case: without it
``dace.transformation.passes.pattern_matching`` catches everything ``can_be_applied()``
raises and merely prints it, which would make a crash look like a refusal here.
"""
from typing import Any, Dict, List, Tuple

import pytest

import dace
from dace import subsets
from dace.sdfg import SDFG, SDFGState, nodes
from dace.transformation.dataflow import MapFusionHorizontal, MapFusionVertical
from dace.transformation.dataflow import map_fusion_helper as mfhelper

N = 10


@pytest.fixture(autouse=True)
def propagate_match_exceptions():
    """Let a ``can_be_applied()`` crash escape instead of being printed and swallowed."""
    old = dace.Config.get_bool('optimizer', 'match_exception')
    dace.Config.set('optimizer', 'match_exception', value=True)
    yield
    dace.Config.set('optimizer', 'match_exception', value=old)


def apply_and_collect(sdfg: SDFG, xform: Any, recwarn: Any) -> Tuple[int, List[str]]:
    """Run ``xform`` to fixpoint; return the application count and the swallowed refusals."""
    applied = sdfg.apply_transformations_repeated([xform], validate=False, validate_all=False)
    swallowed = [str(w.message) for w in recwarn.list if 'refused a malformed match' in str(w.message)]
    return applied, swallowed


def assert_clean_refusal(sdfg: SDFG, xform: Any, recwarn: Any) -> None:
    """``xform`` must refuse every candidate without raising and without a swallow warning."""
    applied, swallowed = apply_and_collect(sdfg, xform, recwarn)
    assert applied == 0, f'{xform.__name__} applied {applied} time(s) to a shape it cannot handle.'
    assert not swallowed, f'{xform.__name__} refused through its safety net instead of a guard: {swallowed}'


def serial_maps_sdfg(name: str, intermediate_shape: Tuple[int, ...] = (N, )) -> Tuple[SDFG, SDFGState, Dict[str, Any]]:
    """``A -> map1 -> T -> map2 -> B``, the shape ``MapFusionVertical`` matches."""
    sdfg = SDFG(name)
    for array in ('A', 'B'):
        sdfg.add_array(array, (N, ), dace.float64)
    sdfg.add_transient('T', intermediate_shape, dace.float64)
    state = sdfg.add_state(is_start_block=True)
    nodes_by_name: Dict[str, Any] = {
        'A': state.add_access('A'),
        'T': state.add_access('T'),
        'B': state.add_access('B'),
    }
    inner_write = 'T[i]' if len(intermediate_shape) == 1 else 'T[i, 0]'
    inner_read = 'T[j]' if len(intermediate_shape) == 1 else 'T[j, 0]'
    outer = 'T[0:%d]' % N if len(intermediate_shape) == 1 else 'T[0:%d, 0:%d]' % intermediate_shape

    entry1, exit1 = state.add_map('first', {'i': f'0:{N}'})
    tasklet1 = state.add_tasklet('t1', {'__in'}, {'__out'}, '__out = __in + 1.0')
    entry1.add_in_connector('IN_A')
    entry1.add_out_connector('OUT_A')
    exit1.add_in_connector('IN_T')
    exit1.add_out_connector('OUT_T')
    state.add_edge(nodes_by_name['A'], None, entry1, 'IN_A', dace.Memlet(f'A[0:{N}]'))
    state.add_edge(entry1, 'OUT_A', tasklet1, '__in', dace.Memlet('A[i]'))
    state.add_edge(tasklet1, '__out', exit1, 'IN_T', dace.Memlet(inner_write))
    state.add_edge(exit1, 'OUT_T', nodes_by_name['T'], None, dace.Memlet(outer))

    entry2, exit2 = state.add_map('second', {'j': f'0:{N}'})
    tasklet2 = state.add_tasklet('t2', {'__in'}, {'__out'}, '__out = __in * 2.0')
    entry2.add_in_connector('IN_T')
    entry2.add_out_connector('OUT_T')
    exit2.add_in_connector('IN_B')
    exit2.add_out_connector('OUT_B')
    state.add_edge(nodes_by_name['T'], None, entry2, 'IN_T', dace.Memlet(outer))
    state.add_edge(entry2, 'OUT_T', tasklet2, '__in', dace.Memlet(inner_read))
    state.add_edge(tasklet2, '__out', exit2, 'IN_B', dace.Memlet('B[j]'))
    state.add_edge(exit2, 'OUT_B', nodes_by_name['B'], None, dace.Memlet(f'B[0:{N}]'))

    nodes_by_name.update(entry1=entry1, exit1=exit1, entry2=entry2, exit2=exit2)
    return sdfg, state, nodes_by_name


def parallel_maps_sdfg(name: str) -> Tuple[SDFG, SDFGState, Dict[str, Any]]:
    """``A -> map1 -> B`` beside ``A -> map2 -> C``, the shape ``MapFusionHorizontal`` matches."""
    sdfg = SDFG(name)
    for array in ('A', 'B', 'C'):
        sdfg.add_array(array, (N, ), dace.float64)
    state = sdfg.add_state(is_start_block=True)
    source = state.add_access('A')
    nodes_by_name: Dict[str, Any] = {'A': source}
    for index, (output, param) in enumerate((('B', 'i'), ('C', 'j'))):
        entry, exit_node = state.add_map(f'map{index}', {param: f'0:{N}'})
        tasklet = state.add_tasklet(f't{index}', {'__in'}, {'__out'}, f'__out = __in + {index}.0')
        entry.add_in_connector('IN_A')
        entry.add_out_connector('OUT_A')
        exit_node.add_in_connector(f'IN_{output}')
        exit_node.add_out_connector(f'OUT_{output}')
        sink = state.add_access(output)
        state.add_edge(source, None, entry, 'IN_A', dace.Memlet(f'A[0:{N}]'))
        state.add_edge(entry, 'OUT_A', tasklet, '__in', dace.Memlet(f'A[{param}]'))
        state.add_edge(tasklet, '__out', exit_node, f'IN_{output}', dace.Memlet(f'{output}[{param}]'))
        state.add_edge(exit_node, f'OUT_{output}', sink, None, dace.Memlet(f'{output}[0:{N}]'))
        nodes_by_name.update({f'entry{index}': entry, f'exit{index}': exit_node, output: sink})
    return sdfg, state, nodes_by_name


def make_unresolvable_view_sdfg() -> SDFG:
    """A View feeding the first Map that no incoming edge defines and two edges consume.

    ``sdutil.get_view_edge()`` has no rule for "no incoming data edge and more than one
    outgoing edge" and indexes ``in_edges[0]`` unguarded there, so resolving this View
    raises ``IndexError`` -- which is not in either ``can_be_applied()`` safety net's
    exception list and therefore escaped the transformation entirely.
    """
    sdfg, state, built = serial_maps_sdfg('unresolvable_view')
    sdfg.add_view('AV', (N, ), dace.float64)
    view = built['A']
    view.data = 'AV'
    for edge in state.all_edges(view):
        edge.data.data = 'AV'
    # A second consumer, so the View has two out-edges and no in-edge at all.
    state.add_nedge(view, state.add_access('AV'), dace.Memlet(f'AV[0:{N}]'))
    return sdfg


def make_subset_union_producer_sdfg() -> SDFG:
    """The producer Memlet carries a ``SubsetUnion``, which has no ``size()``.

    ``compute_reduced_intermediate()`` sizes the new intermediate with
    ``producer_subset.size()`` and ``compute_offset_subset()`` walks it per dimension, so
    a subset that is not a ``Range`` raised ``AttributeError`` from inside the matcher.
    """
    sdfg, state, built = serial_maps_sdfg('subset_union_producer')
    for edge in state.in_edges(built['exit1']):
        edge.data.subset = subsets.SubsetUnion([subsets.Range.from_string('i'), subsets.Range.from_string('0')])
    return sdfg


def make_subset_union_consumer_sdfg() -> SDFG:
    """The consumer Memlet inside the second Map carries a ``SubsetUnion``."""
    sdfg, state, built = serial_maps_sdfg('subset_union_consumer')
    for edge in state.out_edges(built['entry2']):
        edge.data.subset = subsets.SubsetUnion([subsets.Range.from_string('j'), subsets.Range.from_string('0')])
    return sdfg


def make_happens_before_view_sdfg() -> SDFG:
    """Two Maps ordered by an empty Memlet, one of them holding an unresolvable View.

    This is the shape that reaches ``MapFusionHorizontal``'s own View resolution:
    ``analyze_happens_before_fusion()`` de-aliases every AccessNode inside the two Map
    bodies, and it only runs for Maps that are ordered without data flowing between them.
    """
    sdfg, state, built = parallel_maps_sdfg('happens_before_view')
    sdfg.add_view('PV', (N, ), dace.float64)
    state.add_nedge(built['exit0'], built['entry1'], dace.Memlet())
    view = state.add_access('PV')
    state.add_nedge(built['entry0'], view, dace.Memlet())
    for _ in range(2):
        state.add_nedge(view, state.add_access('PV'), dace.Memlet(f'PV[0:{N}]'))
    return sdfg


def test_vertical_refuses_unresolvable_view(recwarn):
    """The View chain cannot be resolved, so the fusion must be refused, not crash."""
    assert_clean_refusal(make_unresolvable_view_sdfg(), MapFusionVertical, recwarn)


def test_horizontal_refuses_unresolvable_view(recwarn):
    assert_clean_refusal(make_unresolvable_view_sdfg(), MapFusionHorizontal, recwarn)


def test_horizontal_refuses_unresolvable_view_in_scope(recwarn):
    """The de-aliasing inside ``analyze_happens_before_fusion()`` must refuse, not raise."""
    assert_clean_refusal(make_happens_before_view_sdfg(), MapFusionHorizontal, recwarn)


def test_view_source_resolution_is_decidable():
    """``resolve_view_source()`` reports the undecidable View shape instead of raising."""
    sdfg = make_unresolvable_view_sdfg()
    state = sdfg.states()[0]
    views = [n for n in state.data_nodes() if n.data == 'AV' and state.out_degree(n) > 1]
    assert len(views) == 1
    assert mfhelper.resolve_view_source(state, views[0]) is None


def test_view_source_still_resolves_a_normal_view():
    """The guard must not refuse a View that ``get_view_edge()`` can perfectly well resolve."""
    sdfg, state, built = serial_maps_sdfg('resolvable_view')
    sdfg.add_view('AV', (N, ), dace.float64)
    view = state.add_access('AV')
    for edge in list(state.out_edges(built['A'])):
        state.add_edge(view, None, edge.dst, edge.dst_conn, dace.Memlet(f'AV[0:{N}]'))
        state.remove_edge(edge)
    state.add_edge(built['A'], None, view, 'views', dace.Memlet(f'A[0:{N}]'))
    resolved = mfhelper.resolve_view_source(state, view)
    assert resolved is not None and resolved.data == 'A'


def test_vertical_refuses_subset_union_producer(recwarn):
    assert_clean_refusal(make_subset_union_producer_sdfg(), MapFusionVertical, recwarn)


def test_vertical_refuses_subset_union_consumer(recwarn):
    assert_clean_refusal(make_subset_union_consumer_sdfg(), MapFusionVertical, recwarn)


def test_horizontal_refuses_subset_union_producer(recwarn):
    assert_clean_refusal(make_subset_union_producer_sdfg(), MapFusionHorizontal, recwarn)


def test_indices_subsets_are_still_fused(recwarn):
    """``Indices`` IS a ``Range``; the non-``Range`` guard must not refuse it."""
    sdfg, state, built = serial_maps_sdfg('indices_subsets')
    for edge in state.in_edges(built['exit1']):
        edge.data.subset = subsets.Indices([dace.symbolic.pystr_to_symbolic('i')])
    for edge in state.out_edges(built['entry2']):
        edge.data.subset = subsets.Indices([dace.symbolic.pystr_to_symbolic('j')])
    sdfg.validate()
    applied, swallowed = apply_and_collect(sdfg, MapFusionVertical, recwarn)
    assert applied == 1
    assert not swallowed
    sdfg.validate()


def test_plain_serial_maps_still_fuse(recwarn):
    """The baseline the guards above must not have cost."""
    sdfg = serial_maps_sdfg('plain_serial')[0]
    sdfg.validate()
    applied, swallowed = apply_and_collect(sdfg, MapFusionVertical, recwarn)
    assert applied == 1
    assert not swallowed
    sdfg.validate()


def test_plain_parallel_maps_still_fuse(recwarn):
    sdfg = parallel_maps_sdfg('plain_parallel')[0]
    sdfg.validate()
    applied, swallowed = apply_and_collect(sdfg, MapFusionHorizontal, recwarn)
    assert applied == 1
    assert not swallowed
    sdfg.validate()


@pytest.mark.parametrize('xform', [MapFusionVertical, MapFusionHorizontal])
def test_stale_descriptor_is_refused(xform, recwarn):
    """A Memlet naming an array whose descriptor is gone must refuse, not raise."""
    sdfg, state, built = serial_maps_sdfg('stale_descriptor')
    sdfg.add_array('C', (N, ), dace.float64)
    source = state.add_access('C')
    tasklet = [n for n in state.scope_subgraph(built['entry2'], False, False).nodes()
               if isinstance(n, nodes.Tasklet)][0]
    built['entry2'].add_in_connector('IN_C')
    built['entry2'].add_out_connector('OUT_C')
    tasklet.add_in_connector('__in2')
    state.add_edge(source, None, built['entry2'], 'IN_C', dace.Memlet(f'C[0:{N}]'))
    state.add_edge(built['entry2'], 'OUT_C', tasklet, '__in2', dace.Memlet('C[j]'))
    del sdfg.arrays['C']
    assert_clean_refusal(sdfg, xform, recwarn)


@pytest.mark.parametrize('xform', [MapFusionVertical, MapFusionHorizontal])
def test_dangling_scope_connector_is_refused(xform, recwarn):
    """A declared connector with no edge breaks ``relocate_nodes()`` half way through."""
    sdfg, state, built = serial_maps_sdfg('dangling_connector')
    built['exit1'].add_in_connector('IN_dead')
    built['exit1'].add_out_connector('OUT_dead')
    assert_clean_refusal(sdfg, xform, recwarn)


@pytest.mark.parametrize('xform', [MapFusionVertical, MapFusionHorizontal])
def test_stream_intermediate_is_refused(xform, recwarn):
    """A Stream has no reducible layout, so ``compute_offset_subset()`` must never see it."""
    sdfg, state, built = serial_maps_sdfg('stream_intermediate')
    del sdfg.arrays['T']
    sdfg.add_stream('T', dace.float64, shape=(N, ), transient=True)
    assert_clean_refusal(sdfg, xform, recwarn)


@pytest.mark.parametrize('xform', [MapFusionVertical, MapFusionHorizontal])
def test_duplicated_map_parameter_is_refused(xform, recwarn):
    """A repeated parameter name collapses the ``param -> range`` dicts of the remapping."""
    sdfg, state, built = serial_maps_sdfg('duplicated_parameter')
    built['entry2'].map.params = ['j', 'j']
    built['entry2'].map.range = subsets.Range.from_string(f'0:{N}, 0:{N}')
    assert_clean_refusal(sdfg, xform, recwarn)


@pytest.mark.parametrize('xform', [MapFusionVertical, MapFusionHorizontal])
def test_params_range_length_mismatch_is_refused(xform, recwarn):
    """More parameters than range dimensions makes the remapping ``zip()`` drop entries."""
    sdfg, state, built = serial_maps_sdfg('params_range_mismatch')
    built['entry2'].map.params = ['j', 'k']
    assert_clean_refusal(sdfg, xform, recwarn)


@pytest.mark.parametrize('xform', [MapFusionVertical, MapFusionHorizontal])
def test_cycle_elsewhere_in_state_does_not_crash(xform, recwarn):
    """``scope_dict()`` raises for a cycle anywhere in the state, even a disjoint one."""
    sdfg, state, built = serial_maps_sdfg('cycle_elsewhere')
    for array in ('C', 'D'):
        sdfg.add_array(array, (N, ), dace.float64)
    left, right = state.add_access('C'), state.add_access('D')
    state.add_nedge(left, right, dace.Memlet(f'C[0:{N}]'))
    state.add_nedge(right, left, dace.Memlet(f'D[0:{N}]'))
    applied = sdfg.apply_transformations_repeated([xform], validate=False, validate_all=False)
    assert applied == 0


@pytest.mark.parametrize('xform', [MapFusionVertical, MapFusionHorizontal])
def test_ordering_edges_survive_a_fusion(xform, recwarn):
    """An empty Memlet is an ordering edge: fusing must never silently drop one."""
    sdfg, state, built = serial_maps_sdfg('ordering_edges_survive')
    sdfg.add_array('C', (N, ), dace.float64)
    guard = state.add_access('C')
    state.add_nedge(guard, built['entry1'], dace.Memlet())
    before = sum(1 for e in state.edges() if e.data.is_empty())
    sdfg.validate()
    sdfg.apply_transformations_repeated([xform], validate=False, validate_all=False)
    state = sdfg.states()[0]
    after = sum(1 for e in state.edges() if e.data.is_empty())
    assert after >= before, 'An ordering edge was dropped by the fusion.'
    sdfg.validate()


@pytest.mark.parametrize('xform', [MapFusionVertical, MapFusionHorizontal])
def test_wcr_producer_is_refused_or_valid(xform, recwarn):
    """A WCR write into the intermediate is an accumulate, never a plain write."""
    sdfg, state, built = serial_maps_sdfg('wcr_producer')
    for edge in list(state.in_edges(built['exit1'])) + list(state.out_edges(built['exit1'])):
        edge.data.wcr = 'lambda a, b: a + b'
    sdfg.validate()
    applied, swallowed = apply_and_collect(sdfg, xform, recwarn)
    assert not swallowed
    if applied:
        sdfg.validate()


@pytest.mark.parametrize('xform', [MapFusionVertical, MapFusionHorizontal])
def test_dynamic_map_range_from_the_same_source(xform, recwarn):
    """Both Maps bind the same dynamic-map-range symbol to the same value."""
    sdfg, state, built = serial_maps_sdfg('dynamic_map_range')
    sdfg.add_array('S', (1, ), dace.int64)
    source = state.add_access('S')
    for entry in (built['entry1'], built['entry2']):
        entry.add_in_connector('dsym')
        state.add_edge(source, None, entry, 'dsym', dace.Memlet('S[0]'))
    sdfg.validate()
    applied, swallowed = apply_and_collect(sdfg, xform, recwarn)
    assert not swallowed
    if applied:
        sdfg.validate()


@pytest.mark.parametrize('xform', [MapFusionVertical, MapFusionHorizontal])
def test_nested_sdfg_body_does_not_crash(xform, recwarn):
    """A NestedSDFG consumer of the intermediate reaches the stride-rewriting path."""
    sdfg, state, built = serial_maps_sdfg('nested_sdfg_body')
    inner = SDFG('inner')
    inner.add_array('ti', (1, ), dace.float64)
    inner.add_array('bo', (1, ), dace.float64)
    inner_state = inner.add_state(is_start_block=True)
    inner_tasklet = inner_state.add_tasklet('it', {'__in'}, {'__out'}, '__out = __in * 3.0')
    inner_state.add_edge(inner_state.add_access('ti'), None, inner_tasklet, '__in', dace.Memlet('ti[0]'))
    inner_state.add_edge(inner_tasklet, '__out', inner_state.add_access('bo'), None, dace.Memlet('bo[0]'))
    old = [n for n in state.scope_subgraph(built['entry2'], False, False).nodes() if isinstance(n, nodes.Tasklet)][0]
    nested = state.add_nested_sdfg(inner, {'ti'}, {'bo'})
    for edge in list(state.in_edges(old)):
        state.add_edge(edge.src, edge.src_conn, nested, 'ti', dace.Memlet('T[j]'))
        state.remove_edge(edge)
    for edge in list(state.out_edges(old)):
        state.add_edge(nested, 'bo', edge.dst, edge.dst_conn, dace.Memlet('B[j]'))
        state.remove_edge(edge)
    state.remove_node(old)
    sdfg.validate()
    applied, swallowed = apply_and_collect(sdfg, xform, recwarn)
    assert not swallowed
    if applied:
        sdfg.validate()


@pytest.mark.parametrize('xform', [MapFusionVertical, MapFusionHorizontal])
def test_three_map_chain_reaches_a_fixpoint(xform, recwarn):
    """Repeated application must terminate and leave a valid SDFG."""
    sdfg = SDFG('three_map_chain')
    sdfg.add_array('A', (N, ), dace.float64)
    sdfg.add_array('D', (N, ), dace.float64)
    for transient in ('T1', 'T2'):
        sdfg.add_transient(transient, (N, ), dace.float64)
    state = sdfg.add_state(is_start_block=True)
    source = state.add_access('A')
    for index, (inp, out, param) in enumerate((('A', 'T1', 'i'), ('T1', 'T2', 'j'), ('T2', 'D', 'k'))):
        entry, exit_node = state.add_map(f'map{index}', {param: f'0:{N}'})
        tasklet = state.add_tasklet(f't{index}', {'__in'}, {'__out'}, '__out = __in + 1.0')
        entry.add_in_connector(f'IN_{inp}')
        entry.add_out_connector(f'OUT_{inp}')
        exit_node.add_in_connector(f'IN_{out}')
        exit_node.add_out_connector(f'OUT_{out}')
        sink = state.add_access(out)
        state.add_edge(source, None, entry, f'IN_{inp}', dace.Memlet(f'{inp}[0:{N}]'))
        state.add_edge(entry, f'OUT_{inp}', tasklet, '__in', dace.Memlet(f'{inp}[{param}]'))
        state.add_edge(tasklet, '__out', exit_node, f'IN_{out}', dace.Memlet(f'{out}[{param}]'))
        state.add_edge(exit_node, f'OUT_{out}', sink, None, dace.Memlet(f'{out}[0:{N}]'))
        source = sink
    sdfg.validate()
    applied, swallowed = apply_and_collect(sdfg, xform, recwarn)
    assert not swallowed
    sdfg.validate()
    if xform is MapFusionVertical:
        assert applied == 2


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
