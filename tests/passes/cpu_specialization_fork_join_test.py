# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CPU fork/join cost model and the transfer specialization that follows it.

Pins the decisions of
:class:`~dace.transformation.passes.cpu_specialization.sequentialize_parallel_scopes.SequentializeParallelScopes`
and
:class:`~dace.transformation.passes.cpu_specialization.specialize_cpu_transfers.SpecializeCpuTransfers`
-- the single home of "make it sequential again" on CPU. The canonical form these run on is the
maximally parallel one, so every assertion here is about what the SPECIALIZATION takes away, never
about what canonicalization produced.

Kernel shapes referenced by name below (all measured on a 72-thread node, LEN_2D=768):
``s115``   ``for j: map i in j+1:LEN_2D``  -- 768 regions per call, ~100% team startup.
``s119``   ``for i: map j in 1:LEN_2D``    -- same shape, 767 regions.
jacobi/heat ``for t: map[i, j] over N x N`` -- must KEEP its map: the work per entry outgrows the
number of entries, which is exactly what separates it from the two above.
"""
import dace
import pytest
from dace import dtypes
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode
from dace.libraries.standard.nodes.reduce import Reduce
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.cpu_specialization import SequentializeParallelScopes, SpecializeCpuTransfers

N = dace.symbol('N')
M = dace.symbol('M')


def map_state(container, sdfg, label, ranges, schedule=dtypes.ScheduleType.Default):
    """A state whose only content is one mapped ``a[...] = 1.0`` tasklet over ``ranges``."""
    state = container.add_state(label, is_start_block=len(container.nodes()) == 0)
    index = ', '.join(ranges)
    state.add_mapped_tasklet(label, {
        f'__i{i}': r
        for i, r in enumerate(ranges)
    }, {},
                             'out = 1.0', {'out': dace.Memlet(f"a[{','.join(f'__i{i}' for i in range(len(ranges)))}]")},
                             schedule=schedule,
                             external_edges=True)
    assert index  # ranges must be non-empty
    return state


def loop(container, label, end, var='it'):
    """A ``for var in 0:end`` LoopRegion added to ``container``."""
    region = LoopRegion(label,
                        initialize_expr=f'{var} = 0',
                        condition_expr=f'{var} < {end}',
                        update_expr=f'{var} = {var} + 1',
                        loop_var=var)
    container.add_node(region, is_start_block=len(container.nodes()) == 0)
    return region


def one_map_sdfg(name, shape, ranges, loop_end=None, schedule=dtypes.ScheduleType.Default):
    """An SDFG holding one map, optionally wrapped in a ``for it in 0:loop_end`` loop."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', shape, dace.float64)
    container = sdfg if loop_end is None else loop(sdfg, 'outer', loop_end)
    map_state(container, sdfg, 'body', ranges, schedule)
    sdfg.validate()
    return sdfg


def schedules(sdfg):
    """Every map label -> schedule in ``sdfg``."""
    return {n.map.label: n.map.schedule for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)}


def test_map_below_break_even_goes_sequential():
    """A constant trip count under the threshold cannot pay for a fork/join, at any nesting."""
    sdfg = one_map_sdfg('small_top_level', [64], ['0:64'])
    SequentializeParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)['body_map'] == dtypes.ScheduleType.Sequential


def test_map_above_break_even_stays_parallel():
    """A constant trip count above the threshold keeps its region."""
    sdfg = one_map_sdfg('large_top_level', [4096], ['0:4096'])
    SequentializeParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)['body_map'] == dtypes.ScheduleType.Default


def test_symbolic_extent_stays_parallel():
    """Every symbol is assumed big enough: a symbolic extent is never sequentialized on size.

    This is the ``s4116`` case (a top-level reduction over ``LEN_2D - 1``): it is a measured loss
    at LEN_2D=768, but nothing in the SDFG says so, and the ruling is to prefer the parallel form
    over a runtime guard.
    """
    sdfg = one_map_sdfg('symbolic_top_level', [N], ['0:N-1'])
    SequentializeParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)['body_map'] == dtypes.ScheduleType.Default


def test_map_re_entered_by_symbolic_loop_goes_sequential():
    """s115 / s119: a symbolic loop is LONG, and a map whose work does not outgrow the number of
    entries pays more fork/join than it saves."""
    sdfg = one_map_sdfg('map_in_long_loop', [N], ['0:N'], loop_end='N')
    SequentializeParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)['body_map'] == dtypes.ScheduleType.Sequential


def test_triangular_map_in_loop_goes_sequential():
    """s115 exactly: ``for j in 0:N { map i in j+1:N }``."""
    sdfg = dace.SDFG('triangular')
    sdfg.add_array('a', [N], dace.float64)
    outer = loop(sdfg, 'outer', 'N', var='j')
    map_state(outer, sdfg, 'body', ['j+1:N'])
    sdfg.validate()
    SequentializeParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)['body_map'] == dtypes.ScheduleType.Sequential


def test_wide_map_in_time_loop_stays_parallel():
    """jacobi / heat: ``for t: map[i, j]`` keeps its map -- ``N*M`` work per entry outgrows the
    ``N`` entries, which is the whole difference from s115."""
    sdfg = one_map_sdfg('wide_map_in_loop', [N, M], ['0:N', '0:M'], loop_end='N')
    SequentializeParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)['body_map'] == dtypes.ScheduleType.Default


def test_provably_short_loop_is_not_re_entry():
    """A provably short loop around real work keeps the map parallel -- the trip-count awareness
    that ``libnode_is_sequential`` lacks (it counts ANY enclosing loop as re-entry)."""
    sdfg = one_map_sdfg('map_in_short_loop', [N], ['0:N'], loop_end='4')
    SequentializeParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)['body_map'] == dtypes.ScheduleType.Default


def test_nested_parallel_map_is_always_sequentialized():
    """The rule inherited from the canonicalize finalize tail: an explicit ``CPU_Multicore`` map
    inside a parallel map forks a team per outer iteration, at any size."""
    sdfg = dace.SDFG('nested_parallel')
    sdfg.add_array('a', [N, N], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map('outer', dict(i='0:N'), schedule=dtypes.ScheduleType.CPU_Multicore)
    inner_entry, inner_exit = state.add_map('inner', dict(j='0:N'), schedule=dtypes.ScheduleType.CPU_Multicore)
    tasklet = state.add_tasklet('t', {}, {'out'}, 'out = 1.0')
    access = state.add_access('a')
    state.add_edge(outer_entry, None, inner_entry, None, dace.Memlet())
    state.add_edge(inner_entry, None, tasklet, None, dace.Memlet())
    state.add_memlet_path(tasklet, inner_exit, outer_exit, access, src_conn='out', memlet=dace.Memlet('a[i, j]'))
    sdfg.validate()

    SequentializeParallelScopes().apply_pass(sdfg, {})

    assert schedules(sdfg)['outer'] == dtypes.ScheduleType.CPU_Multicore
    assert schedules(sdfg)['inner'] == dtypes.ScheduleType.Sequential


def test_nested_parallel_canonical_form_is_valid():
    """Schedules are labels: canonical form may leave parallel maps nested and still validate, so
    resolving the nesting is a specialization and not a correctness repair."""
    sdfg = dace.SDFG('nested_parallel_valid')
    sdfg.add_array('a', [N, N], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map('outer', dict(i='0:N'), schedule=dtypes.ScheduleType.CPU_Multicore)
    inner_entry, inner_exit = state.add_map('inner', dict(j='0:N'), schedule=dtypes.ScheduleType.CPU_Multicore)
    tasklet = state.add_tasklet('t', {}, {'out'}, 'out = 1.0')
    access = state.add_access('a')
    state.add_edge(outer_entry, None, inner_entry, None, dace.Memlet())
    state.add_edge(inner_entry, None, tasklet, None, dace.Memlet())
    state.add_memlet_path(tasklet, inner_exit, outer_exit, access, src_conn='out', memlet=dace.Memlet('a[i, j]'))

    sdfg.validate()


def test_threshold_zero_disables_the_cost_model():
    """``parallel_min_work_per_region = 0`` is the A/B lever: only the nested-parallelism rule
    survives, so a re-entered map keeps its region."""
    sdfg = one_map_sdfg('ab_off', [N], ['0:N'], loop_end='N')
    with dace.config.set_temporary('compiler', 'cpu', 'parallel_min_work_per_region', value=0):
        SequentializeParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)['body_map'] == dtypes.ScheduleType.Default


def reduce_in_loop(name, loop_end):
    """An SDFG with one ``Reduce`` library node inside a ``for it in 0:loop_end`` loop."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('row', [N], dace.float64)
    sdfg.add_array('acc', [1], dace.float64)
    outer = loop(sdfg, 'outer', loop_end)
    state = outer.add_state('body', is_start_block=True)
    node = Reduce('reduce_sum', wcr='lambda a, b: a + b', axes=None, identity=0.0)
    node.schedule = dtypes.ScheduleType.CPU_Multicore
    state.add_node(node)
    state.add_edge(state.add_access('row'), None, node, None, dace.Memlet('row[0:N]'))
    state.add_edge(node, None, state.add_access('acc'), None, dace.Memlet('acc[0]'))
    sdfg.validate()
    return sdfg, node


def test_library_node_in_long_loop_goes_sequential():
    """A library node opens its own region per entry, so a long loop around it is a hazard."""
    sdfg, node = reduce_in_loop('reduce_long_loop', 'N')
    SequentializeParallelScopes().apply_pass(sdfg, {})
    assert node.schedule == dtypes.ScheduleType.Sequential


def test_library_node_in_short_loop_stays_parallel():
    """...but a PROVABLY short loop is not re-entry -- the trip-count-aware half of the rule."""
    sdfg, node = reduce_in_loop('reduce_short_loop', '4')
    SequentializeParallelScopes().apply_pass(sdfg, {})
    assert node.schedule == dtypes.ScheduleType.CPU_Multicore


def transfer_sdfg(name, loop_end, kind, subset='0:N'):
    """An SDFG with one copy / memset library node inside a ``for it in 0:loop_end`` loop."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('src', [N], dace.float64)
    sdfg.add_array('dst', [N], dace.float64)
    container = sdfg if loop_end is None else loop(sdfg, 'outer', loop_end)
    state = container.add_state('body', is_start_block=True)
    if kind == 'copy':
        node = CopyLibraryNode(name='cpy')
        state.add_edge(state.add_access('src'), None, node, CopyLibraryNode.INPUT_CONNECTOR_NAME,
                       dace.Memlet(f'src[{subset}]'))
        state.add_edge(node, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access('dst'), None,
                       dace.Memlet(f'dst[{subset}]'))
    else:
        node = MemsetLibraryNode(name='mset')
        state.add_edge(node, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access('dst'), None,
                       dace.Memlet(f'dst[{subset}]'))
    sdfg.validate()
    return sdfg, node


@pytest.mark.parametrize('kind,implementation', [('copy', 'MemcpyCPU'), ('memset', 'CPU')])
def test_re_entered_transfer_collapses_to_libc(kind, implementation):
    """stockham_fft's inner copy: re-entered -> Sequential, and a contiguous sequential transfer
    is one libc call, not a serial element loop."""
    sdfg, node = transfer_sdfg(f'{kind}_in_loop', 'N', kind)
    SpecializeCpuTransfers().apply_pass(sdfg, {})
    assert node.schedule == dtypes.ScheduleType.Sequential
    assert node.implementation == implementation


@pytest.mark.parametrize('kind', ['copy', 'memset'])
def test_top_level_transfer_keeps_the_parallel_element_map(kind):
    """A bulk transfer nobody re-enters keeps the canonical parallel element map, and no
    implementation is forced on it -- ``memcpy`` is not the default."""
    sdfg, node = transfer_sdfg(f'{kind}_top_level', None, kind)
    SpecializeCpuTransfers().apply_pass(sdfg, {})
    assert node.schedule == dtypes.ScheduleType.Default
    assert node.implementation in (None, 'Auto')


def test_passes_are_idempotent():
    """Both passes run in the canonicalize band AND in the ``finalize_for_target`` tail, so a
    second application must find nothing left to do."""
    sdfg = one_map_sdfg('idempotent_map', [N], ['0:N'], loop_end='N')
    assert SequentializeParallelScopes().apply_pass(sdfg, {}) == 1
    assert SequentializeParallelScopes().apply_pass(sdfg, {}) is None

    transfers, _node = transfer_sdfg('idempotent_copy', 'N', 'copy')
    assert SpecializeCpuTransfers().apply_pass(transfers, {}) == 2
    assert SpecializeCpuTransfers().apply_pass(transfers, {}) is None


def test_canonicalize_wires_the_cpu_band():
    """End to end: ``canonicalize(target='cpu')`` must leave TSVC s115's shape with no parallel
    region inside the sweep loop -- the band is wired, not just importable."""
    from dace.transformation.passes.canonicalize.pipeline import canonicalize

    @dace.program
    def triangular_sweep(a: dace.float64[N]):
        for j in range(N):
            for i in range(j + 1, N):
                a[i] = a[i] - a[j]

    sdfg = triangular_sweep.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)

    parallel = [
        n.map.label for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule != dtypes.ScheduleType.Sequential
    ]
    assert not parallel, f"s115's inner map must not open a region per sweep iteration, got {parallel}"


def test_strided_re_entered_copy_is_sequential_but_not_memcpy():
    """A non-contiguous copy has no single-``memcpy`` form: it is still sequentialized, but the
    implementation is left to the library node's own selector."""
    sdfg, node = transfer_sdfg('strided_copy_in_loop', 'N', 'copy', subset='0:N:2')
    SpecializeCpuTransfers().apply_pass(sdfg, {})
    assert node.schedule == dtypes.ScheduleType.Sequential
    assert node.implementation in (None, 'Auto')


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
