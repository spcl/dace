# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One OpenMP team for a whole sequential loop, asserted on the pragmas that reach the compiler.

:class:`~dace.transformation.passes.cpu_specialization.hoist_parallel_region.HoistParallelRegion`
is a rewrite whose entire effect is in the emitted form: the computation, the iteration-to-thread
assignment and every barrier stay exactly as they were, and only the fork/join per trip of the
enclosing loop goes away. So the assertions here are on the emitted C++ -- ONE ``#pragma omp
parallel`` where there used to be one ``#pragma omp parallel for`` per trip -- paired with a
numeric check, because "it still computes the right answer" passes just as happily on the
un-hoisted form and is not the property at stake.

The refusal tests are the other half, and the more important one. Every statement inside a parallel
region that is not inside a worksharing construct runs once per THREAD, so a loop body carrying one
is a wrong answer waiting to happen; the pass has to leave those graphs untouched, byte for byte.
"""
import os

os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import re

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import nodes as nd
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.cpu_specialization.hoist_parallel_region import HoistParallelRegion

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES

N = dace.symbol('N')

#: The three TSVC kernels whose canonical form is exactly "sequential loop around one parallel
#: map", i.e. the shape this pass exists for. ``s115`` additionally carries an anti-dependence
#: snapshot at the state's top level, which is the branch that has to be worksharing-wrapped.
HOISTED_KERNELS = ['s115_d_single', 's119_d_single', 's233_d_single']


def finalized(name, tag):
    """``(kernel, sdfg)`` for one TSVC kernel put through canonicalize and the CPU perf tail."""
    kernel = tsvc.collect(name=name)[0]
    sdfg = tsvc.to_sdfg(kernel, tag, simplify=True)
    canonicalize(sdfg, validate=True)
    finalize_for_target(sdfg, 'cpu')
    return kernel, sdfg


def pragmas(sdfg):
    """``(teams, per_trip_regions, worksharing_loops)`` counted in ``sdfg``'s emitted C++.

    A bare ``#pragma omp parallel`` opens a team; ``#pragma omp parallel for`` opens one AND
    distributes, which is the per-trip form this pass replaces; ``#pragma omp for`` distributes
    inside a team already open.
    """
    code = sdfg.generate_code()[0].clean_code
    return (len(re.findall(r'#pragma omp parallel(?! for)',
                           code)), len(re.findall(r'#pragma omp parallel for',
                                                  code)), len(re.findall(r'#pragma omp for', code)))


def assert_matches_reference(kernel, sdfg):
    """The finalized kernel must reproduce the numpy reference element for element."""
    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[kernel.name](**ref, **call_kwargs)
    got = {n: a.copy() for n, a in arrays.items()}
    sdfg.compile()(**got, **call_kwargs)
    for n, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue
        assert np.allclose(ref[n], got[n], equal_nan=True), f'{kernel.name}: value mismatch on {n}'


def loop_region(container, label, end, var='it'):
    """A ``for var in 0:end`` LoopRegion added to ``container``."""
    region = LoopRegion(label,
                        initialize_expr=f'{var} = 0',
                        condition_expr=f'{var} < {end}',
                        update_expr=f'{var} = {var} + 1',
                        loop_var=var)
    container.add_node(region, is_start_block=len(container.nodes()) == 0)
    return region


def mapped_state(container, label, target, expr, ranges, schedule=dtypes.ScheduleType.CPU_Multicore, inputs=None):
    """A state whose only content is one mapped ``target[i] = expr`` tasklet over ``ranges``."""
    state = container.add_state(label, is_start_block=len(container.nodes()) == 0)
    params = {f'__i{i}': r for i, r in enumerate(ranges)}
    index = ','.join(params)
    state.add_mapped_tasklet(label,
                             params, {f'in_{name}': dace.Memlet(f'{name}[{index}]')
                                      for name in (inputs or ())},
                             f'out = {expr}', {'out': dace.Memlet(f'{target}[{index}]')},
                             schedule=schedule,
                             external_edges=True)
    return state


def loop_over_map_sdfg(name):
    """``for it in 0:N { map i in 0:N: a[i] = 1.0 }`` -- the minimal hoistable shape."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [N], dace.float64)
    mapped_state(loop_region(sdfg, 'outer', 'N'), 'body', 'a', '1.0', ['0:N'])
    sdfg.validate()
    return sdfg


def teams(sdfg):
    """How many ``CPU_Persistent`` map scopes the pass left in ``sdfg``."""
    return sum(1 for n, _ in sdfg.all_nodes_recursive()
               if isinstance(n, nd.MapEntry) and n.map.schedule == dtypes.ScheduleType.CPU_Persistent)


def assert_declined(sdfg, why):
    """The pass must refuse ``sdfg`` AND leave it bit-identical."""
    before = sdfg.to_json()
    assert HoistParallelRegion().apply_pass(sdfg, {}) is None, why
    assert sdfg.to_json() == before, f'declining must not mutate the SDFG ({why})'


@pytest.mark.parametrize('name', HOISTED_KERNELS)
def test_one_team_replaces_the_per_trip_region(name):
    """The whole point: one ``#pragma omp parallel``, no ``#pragma omp parallel for`` left."""
    _kernel, sdfg = finalized(name, 'hoist_' + name)
    team_count, per_trip, worksharing = pragmas(sdfg)
    assert team_count == 1, f'{name} must open exactly one team, got {team_count}'
    assert per_trip == 0, f'{name} must not open a region per trip, got {per_trip}'
    assert worksharing >= 1, f'{name} must still distribute its map, got {worksharing} omp-for'


@pytest.mark.parametrize('name', HOISTED_KERNELS)
def test_hoisted_kernel_matches_the_numpy_reference(name):
    """The rewrite reorders nothing, so the values must be the reference's."""
    kernel, sdfg = finalized(name, 'num_' + name)
    assert teams(sdfg) == 1, f'{name} was expected to hoist'
    assert_matches_reference(kernel, sdfg)


def test_s115_snapshot_is_worksharing_rather_than_replicated():
    """``s115``'s anti-dependence snapshot sits at the state's TOP level, one scalar store per trip.

    Left as it is, the hoisted team would run that store on every thread. The pass wraps it in a
    one-iteration map, so it becomes a second ``omp for`` -- run once, by one thread, with the
    barrier that the reader on the other side of it needs.
    """
    _kernel, sdfg = finalized('s115_d_single', 'snapshot_s115')
    team_count, per_trip, worksharing = pragmas(sdfg)
    assert (team_count, per_trip) == (1, 0)
    assert worksharing == 2, 'the snapshot and the sweep must each be their own omp-for'


def test_wavefront_reaching_into_the_neighbouring_band_is_still_correct():
    """``a[i, j] = a[i, j] + a[i-1, j] + a[i-1, j+1]``: the case a ``nowait`` would silently break.

    Every band reads one element of the band to its right, written on the previous trip. The team
    hoist keeps the barrier that makes that read safe, so this must agree with numpy exactly -- it
    is the kernel that catches a barrier removed one stage too early.
    """

    @dace.program
    def wf_diff_skew(a: dace.float64[N, N]):
        for i in range(1, N):
            for j in range(0, N - 1):
                a[i, j] = a[i, j] + a[i - 1, j] + a[i - 1, j + 1]

    sdfg = wf_diff_skew.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)
    finalize_for_target(sdfg, 'cpu')
    assert teams(sdfg) == 1, 'the wavefront must hoist its team'
    assert pragmas(sdfg)[1] == 0, 'no per-trip region may survive'

    rng = np.random.default_rng(1234)
    a = rng.random((37, 37))
    ref = a.copy()
    for i in range(1, 37):
        for j in range(0, 36):
            ref[i, j] = ref[i, j] + ref[i - 1, j] + ref[i - 1, j + 1]
    got = a.copy()
    sdfg.compile()(a=got, N=37)
    assert np.allclose(ref, got)


def test_minimal_loop_over_map_hoists():
    """The predicate's positive control: the refusal tests below differ from this by one node."""
    sdfg = loop_over_map_sdfg('hoistable')
    assert HoistParallelRegion().apply_pass(sdfg, {}) == 1
    assert teams(sdfg) == 1
    sdfg.validate()


def test_second_run_adds_no_second_team():
    """Idempotent: the hoisted loop now sits inside a map scope, which the walk never descends into."""
    sdfg = loop_over_map_sdfg('idempotent')
    HoistParallelRegion().apply_pass(sdfg, {})
    assert HoistParallelRegion().apply_pass(sdfg, {}) is None
    assert teams(sdfg) == 1


def test_nested_sdfg_inside_the_map_keeps_its_parent_pointers():
    """Outlining moves states into a NEW SDFG, and a nested SDFG that rode along still names the one
    it left. Validation reads that pointer, so the pass has to repair it -- the wavefront kernels,
    whose skewed body is a nested SDFG under the map, are the ones that found this."""
    sdfg = dace.SDFG('nested_in_map')
    sdfg.add_array('a', [N], dace.float64)
    body = loop_region(sdfg, 'outer', 'N').add_state('body', is_start_block=True)
    inner = dace.SDFG('inner')
    inner.add_array('o', [1], dace.float64)
    inner_state = inner.add_state('set', is_start_block=True)
    tasklet = inner_state.add_tasklet('set', {}, {'out'}, 'out = 1.0')
    inner_state.add_edge(tasklet, 'out', inner_state.add_access('o'), None, dace.Memlet('o[0]'))
    nsdfg = body.add_nested_sdfg(inner, {}, {'o'})
    entry, exit_node = body.add_map('m', {'i': '0:N'}, schedule=dtypes.ScheduleType.CPU_Multicore)
    body.add_nedge(entry, nsdfg, dace.Memlet())
    body.add_memlet_path(nsdfg, exit_node, body.add_access('a'), src_conn='o', memlet=dace.Memlet('a[i]'))
    sdfg.validate()

    assert HoistParallelRegion().apply_pass(sdfg, {}) == 1
    sdfg.validate()


def test_top_level_sequential_map_in_the_loop_is_refused():
    """A ``Sequential`` map beside the parallel one is not worksharing: the team would run it P times."""
    sdfg = dace.SDFG('sequential_neighbour')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    outer = loop_region(sdfg, 'outer', 'N')
    mapped_state(outer, 'par', 'a', '1.0', ['0:N'])
    mapped_state(outer, 'seq', 'b', '2.0', ['0:N'], schedule=dtypes.ScheduleType.Sequential)
    outer.add_edge(outer.nodes()[0], outer.nodes()[1], dace.InterstateEdge())
    sdfg.validate()
    assert_declined(sdfg, 'a Sequential map at the top level breaks replication-freedom (H)')


def test_top_level_library_node_in_the_loop_is_refused():
    """A library node expands to whatever it likes, including its own parallel region. Refuse."""
    from dace.libraries.standard.nodes.fill import FillLibraryNode
    sdfg = dace.SDFG('library_neighbour')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    outer = loop_region(sdfg, 'outer', 'N')
    mapped_state(outer, 'par', 'a', '1.0', ['0:N'])
    fill_state = outer.add_state('fill')
    node = FillLibraryNode('fill_b', value=0.0)
    fill_state.add_node(node)
    fill_state.add_edge(node, '_fill_out', fill_state.add_access('b'), None, dace.Memlet('b[0:N]'))
    outer.add_edge(outer.nodes()[0], fill_state, dace.InterstateEdge())
    sdfg.validate()
    assert_declined(sdfg, 'a top-level library node breaks replication-freedom (H)')


def test_loop_local_transient_crossing_two_map_scopes_is_refused():
    """A scope-lifetime transient moves INTO the outlined nest, i.e. one copy per thread.

    Here the first map fills ``t`` and the second reads it, so a private copy would hand the second
    ``omp for`` whatever its own thread happened to write -- condition (T).
    """
    sdfg = dace.SDFG('privatized_transient')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_transient('t', [N], dace.float64)
    outer = loop_region(sdfg, 'outer', 'N')
    producer = mapped_state(outer, 'produce', 't', '1.0', ['0:N'])
    consumer = mapped_state(outer, 'consume', 'a', 'in_t + 1.0', ['0:N'], inputs=['t'])
    outer.add_edge(producer, consumer, dace.InterstateEdge())
    sdfg.arrays['t'].lifetime = dtypes.AllocationLifetime.Scope
    sdfg.validate()
    assert_declined(sdfg, 'a scope-lifetime transient crossing two map scopes breaks (T)')


def test_bulk_copy_between_access_nodes_in_the_loop_is_refused():
    """``jacobi_2d``'s ``A[1:N-1, 1:N-1] = B[1:N-1, 1:N-1]``: two access nodes and a memlet.

    It carries no code node, so it reads like a node that emits nothing -- and it emits a full
    array copy. Replicated across the team it is both a data race and, worse, unsynchronized: no
    barrier stands between it and the next trip's ``omp for`` reading what it wrote.
    """
    sdfg = dace.SDFG('bulk_copy_neighbour')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    outer = loop_region(sdfg, 'outer', 'N')
    body = mapped_state(outer, 'par', 'b', '1.0', ['0:N'])
    body.add_nedge(body.add_access('b'), body.add_access('a'), dace.Memlet('b[0:N] -> [0:N]'))
    sdfg.validate()
    assert_declined(sdfg, 'an access-node-to-access-node copy at the top level breaks (H)')


def test_loop_without_a_parallel_map_is_refused():
    """Nothing to distribute, so a team would only replicate work."""
    sdfg = dace.SDFG('all_sequential')
    sdfg.add_array('a', [N], dace.float64)
    mapped_state(loop_region(sdfg, 'outer', 'N'), 'body', 'a', '1.0', ['0:N'], schedule=dtypes.ScheduleType.Sequential)
    sdfg.validate()
    assert_declined(sdfg, 'a loop with no CPU_Multicore map has nothing to hoist')


def test_break_in_the_loop_is_refused():
    """A team must encounter the same worksharing constructs on every thread; a break is not worth
    proving that about, so the pass declines the shape outright."""
    from dace.sdfg.state import BreakBlock
    sdfg = dace.SDFG('with_break')
    sdfg.add_array('a', [N], dace.float64)
    outer = loop_region(sdfg, 'outer', 'N')
    body = mapped_state(outer, 'body', 'a', '1.0', ['0:N'])
    stop = BreakBlock('stop')
    outer.add_node(stop)
    outer.add_edge(body, stop, dace.InterstateEdge(condition='it > 3'))
    sdfg.validate()
    assert_declined(sdfg, 'a BreakBlock inside the loop is refused')


@pytest.mark.parametrize('name', HOISTED_KERNELS)
def test_the_canonical_form_keeps_every_barrier(name):
    """No ``omp for`` may carry ``nowait``, and this is policy rather than an unfinished feature.

    Dropping the exit barrier is legal only when every thread's reads at trip ``k+1`` fall inside
    the band it wrote at trip ``k`` -- formally, for a partition ``B_1..B_P`` of the map's index
    space, ``R_p(k+1) & (U_q W_q(k)) subset W_p(k)``. ``s233``, ``s231`` and ``s235`` satisfy it;
    ``s119`` and ``wf_diff_skew`` miss it by ONE element at the band boundary, and ``s115`` reads a
    scalar that only one band writes. Measured out of tree by patching the pragma into the emitted
    source: ``nowait`` leaves ``s233`` bit-identical and 2.4x faster, and turns ``s119`` into a
    wrong answer (max relative error 4.8e7). A verdict that sharp is a specialization decision --
    it belongs to a stage that can afford the dependence analysis, not to the canonical starting
    point, which has to be right for every shape it is handed.
    """
    _kernel, sdfg = finalized(name, 'barrier_' + name)
    assert not re.search(r'#pragma omp for[^\n]*nowait', sdfg.generate_code()[0].clean_code), \
        'canonicalization must not remove a barrier: that verdict belongs to a later stage'


if __name__ == '__main__':
    for kernel_name in HOISTED_KERNELS:
        test_one_team_replaces_the_per_trip_region(kernel_name)
        test_hoisted_kernel_matches_the_numpy_reference(kernel_name)
    test_s115_snapshot_is_worksharing_rather_than_replicated()
    test_wavefront_reaching_into_the_neighbouring_band_is_still_correct()
    test_minimal_loop_over_map_hoists()
    test_second_run_adds_no_second_team()
    test_nested_sdfg_inside_the_map_keeps_its_parent_pointers()
    test_top_level_sequential_map_in_the_loop_is_refused()
    test_top_level_library_node_in_the_loop_is_refused()
    test_loop_local_transient_crossing_two_map_scopes_is_refused()
    test_bulk_copy_between_access_nodes_in_the_loop_is_refused()
    test_loop_without_a_parallel_map_is_refused()
    test_break_in_the_loop_is_refused()
    for kernel_name in HOISTED_KERNELS:
        test_the_canonical_form_keeps_every_barrier(kernel_name)
