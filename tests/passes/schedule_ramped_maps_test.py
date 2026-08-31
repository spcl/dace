# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The balancing schedule must reach a ramped map and nothing else.

The cost of over-firing is not a missed win, it is a large regression: on the balanced elementwise
``fuse_diamond`` shape, ``schedule(dynamic, 1)`` measured 0.03x -- a 33x slowdown, because
contiguous blocks are what let each thread prefetch linearly and keep its own pages. So most of
what follows checks that the pass stays SILENT.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import finalize, pipeline as canon
from dace.transformation.passes.cpu_specialization.schedule_ramped_maps import ScheduleRampedMaps

N = dace.symbol('N', dtype=dace.int64)


def prepared(prog):
    sdfg = prog.to_sdfg(simplify=False)
    canon.canonicalize(sdfg, target='cpu')
    finalize.finalize_for_target(sdfg, 'cpu')
    return sdfg


def schedules(sdfg):
    return [(n.map.omp_schedule, n.map.omp_chunk_size) for g in sdfg.all_sdfgs_recursive() for st in g.all_states()
            for n in st.nodes()
            if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.CPU_Multicore]


@dace.program
def triangular(a: dace.float64[N, N], out: dace.float64[N]):
    for i in range(N):
        acc = 0.0
        for j in range(i, N):
            acc += a[i, j]
        out[i] = acc


@dace.program
def rectangular(a: dace.float64[N], out: dace.float64[N]):
    for i in range(N):
        out[i] = a[i] * a[i]


def test_a_ramped_map_gets_a_balancing_schedule():
    sdfg = prepared(triangular)
    assert ScheduleRampedMaps().apply_pass(sdfg, {}), 'the triangular nest was not recognised'
    assert (dtypes.OMPScheduleType.Dynamic, 1) in schedules(sdfg)


def test_a_rectangular_map_is_left_alone():
    """The 33x case: any chunk clause on a balanced elementwise stream is a large regression."""
    sdfg = prepared(rectangular)
    assert ScheduleRampedMaps().apply_pass(sdfg, {}) is None
    assert all(s == dtypes.OMPScheduleType.Default for s, _ in schedules(sdfg))


def test_a_map_inside_a_sequential_loop_is_left_alone():
    """A re-forking region pays fork/join per outer iteration and wants its cache back each time.

    This is the ``wf_triangular`` shape, where a dynamic schedule measured 0.68x.
    """
    from dace.sdfg.state import LoopRegion
    sdfg = dace.SDFG('reforking')
    sdfg.add_array('a', [N, N], dace.float64)
    outer = LoopRegion('t', 't < N', 't', 't = 0', 't = t + 1')
    sdfg.add_node(outer, is_start_block=True)
    state = outer.add_state('body', is_start_block=True)
    state.add_mapped_tasklet('inner', {'p': '0:N'}, {'__in': dace.Memlet('a[t, p]')},
                             '__out = __in * 2.0', {'__out': dace.Memlet('a[t, p]')},
                             external_edges=True)
    for node in state.nodes():
        if isinstance(node, nodes.MapEntry):
            node.map.schedule = dtypes.ScheduleType.CPU_Multicore
    sdfg.validate()
    assert ScheduleRampedMaps().apply_pass(sdfg, {}) is None, 'a re-forking map must not be chunked'


def test_an_explicit_schedule_is_never_overruled():
    sdfg = prepared(triangular)
    for g in sdfg.all_sdfgs_recursive():
        for st in g.all_states():
            for n in st.nodes():
                if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.CPU_Multicore:
                    n.map.omp_schedule = dtypes.OMPScheduleType.Guided
                    n.map.omp_chunk_size = 8
    assert ScheduleRampedMaps().apply_pass(sdfg, {}) is None
    assert all(s == dtypes.OMPScheduleType.Guided for s, _ in schedules(sdfg))


def test_a_gpu_map_is_never_touched():
    """``omp_schedule`` is meaningless on a device; the pass must not reach one."""
    sdfg = prepared(triangular)
    for g in sdfg.all_sdfgs_recursive():
        for st in g.all_states():
            for n in st.nodes():
                if isinstance(n, nodes.MapEntry):
                    n.map.schedule = dtypes.ScheduleType.GPU_Device
    assert ScheduleRampedMaps().apply_pass(sdfg, {}) is None


@pytest.mark.parametrize('n', [1, 2, 33, 64])
def test_the_scheduled_graph_still_computes(n):
    sdfg = prepared(triangular)
    ScheduleRampedMaps().apply_pass(sdfg, {})
    rng = np.random.default_rng(3 + n)
    a = rng.random((n, n))
    want = np.array([a[i, i:].sum() for i in range(n)])
    got = np.zeros(n)
    sdfg(a=a, out=got, N=n)
    assert np.allclose(got, want), f'{got} != {want}'


def test_the_emitted_pragma_carries_the_clause():
    """The point of the pass is a clause in the generated code, not a property in the graph."""
    sdfg = prepared(triangular)
    assert ScheduleRampedMaps().apply_pass(sdfg, {})
    code = '\n'.join(obj.clean_code for obj in sdfg.generate_code())
    assert 'schedule(dynamic, 1)' in code, 'the schedule never reached the pragma'


if __name__ == '__main__':
    test_a_ramped_map_gets_a_balancing_schedule()
    test_a_rectangular_map_is_left_alone()
    test_a_map_inside_a_sequential_loop_is_left_alone()
    test_an_explicit_schedule_is_never_overruled()
    test_a_gpu_map_is_never_touched()
    test_the_scheduled_graph_still_computes(33)
    test_the_emitted_pragma_carries_the_clause()
