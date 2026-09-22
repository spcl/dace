# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MoveLoopIntoMapGated(target='gpu')`` inside ``canonicalize(target='gpu')``: a loop over maps becomes one kernel that
runs the loop per thread, when the fork/join cost model says the launches it saves pay and the loop axis is not the
contiguous one."""
import contextlib
import os

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize import finalize
from dace.transformation.passes.canonicalize.move_loop_into_map_gated import MoveLoopIntoMapGated, launches_saved
from dace.transformation.passes.gpu_block_size_selection import select_gpu_device_block_size

K = dace.symbol('K')
N = dace.symbol('N')


@dace.program
def rows(a: dace.float64[K, N], b: dace.float64[N], w: dace.float64[K], c: dace.float64[K]):
    for k in range(1, K):
        for i in dace.map[0:N]:
            a[k, i] = a[k - 1, i] + b[i] * w[k]
        if c[k] > 0.5:
            for i in dace.map[0:N]:
                b[i] = b[i] * 0.5 + a[k, i]


# A single map whose loop axis ``k`` is the contiguous one.
@dace.program
def single_map_columns(a: dace.float64[N, K], b: dace.float64[N]):
    for k in range(1, K):
        for i in dace.map[0:N]:
            a[i, k] = a[i, k - 1] + b[i]


# The same recurrence stored transposed: the loop axis ``k`` is now the contiguous one.
@dace.program
def columns(a: dace.float64[N, K], b: dace.float64[N], w: dace.float64[K], c: dace.float64[K]):
    for k in range(1, K):
        for i in dace.map[0:N]:
            a[i, k] = a[i, k - 1] + b[i] * w[k]
        if c[k] > 0.5:
            for i in dace.map[0:N]:
                b[i] = b[i] * 0.5 + a[i, k]


def canonical(prog) -> dace.SDFG:
    sdfg = prog.to_sdfg(simplify=False)
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        canonicalize(sdfg, target='gpu')
    return sdfg


def top_level_loops(sdfg: dace.SDFG) -> list:
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]


def test_canonicalize_moves_a_loop_over_branching_maps_into_one_map():
    sdfg = canonical(rows)
    assert top_level_loops(sdfg) == []
    outer = [n for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.MapEntry)]
    assert len(outer) == 1 and str(outer[0].map.range) == '0:N', outer


@pytest.mark.parametrize('prog', [columns, single_map_columns], ids=['branching', 'single_map'])
def test_the_loop_stays_outside_when_its_own_axis_is_contiguous(prog):
    """Threads over ``i`` would stride by ``K``; one kernel per trip keeps them coalesced. The single-map shape follows
    the same rule, no longer interchanged unconditionally."""
    sdfg = canonical(prog)
    assert len(top_level_loops(sdfg)) == 1


def loop_over_maps(name: str, trips: str, maps: int) -> tuple:
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', (4, 16), dace.float64)
    loop = LoopRegion('kloop', f'k < {trips}', 'k', 'k = 0', 'k = k + 1')
    sdfg.add_node(loop, is_start_block=True)
    previous = None
    for m in range(maps):
        state = loop.add_state(f'step{m}', is_start_block=previous is None)
        state.add_mapped_tasklet(f'step{m}', {'i': '0:16'}, {'__in': dace.Memlet('a[k, i]')},
                                 '__out = __in + 1.0', {'__out': dace.Memlet('a[k, i]')},
                                 external_edges=True)
        if previous is not None:
            loop.add_edge(previous, state, dace.InterstateEdge())
        previous = state
    return sdfg, loop


@pytest.mark.parametrize('trips, maps, saved', [('1', 1, 0), ('1', 2, 1), ('3', 2, 5), ('K', 1, float('inf'))])
def test_the_cost_model_counts_the_kernel_launches_the_interchange_saves(trips, maps, saved):
    """One launch per map per trip before, one after; a symbolic trip count is as large as any other extent."""
    loop = loop_over_maps(f'launches_{trips}_{maps}', trips, maps)[1]
    assert launches_saved(loop) == saved


def test_a_single_launch_loop_is_not_worth_a_kernel_of_its_own():
    sdfg = loop_over_maps('single_launch', '1', 1)[0]
    assert MoveLoopIntoMapGated(target='gpu').apply_pass(sdfg, {}) is None
    assert len(top_level_loops(sdfg)) == 1


@pytest.mark.gpu
def test_the_column_form_runs_as_one_kernel_and_matches_numpy():
    import cupy  # Only present on GPU runners.
    sdfg = canonical(rows)
    finalize.offload_to_gpu(sdfg)
    finalize.finalize_for_target(sdfg, 'gpu')
    kernels = [
        n for n, parent in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_Device
    ]
    assert len(kernels) == 1, [k.map.label for k in kernels]

    rng = np.random.default_rng(0)
    a, b, w, c = rng.random((7, 33)), rng.random(33), rng.random(7), rng.random(7)
    want_a, want_b = a.copy(), b.copy()
    for k in range(1, 7):
        want_a[k] = want_a[k - 1] + want_b * w[k]
        if c[k] > 0.5:
            want_b = want_b * 0.5 + want_a[k]
    args = {name: cupy.asarray(v) for name, v in dict(a=a, b=b, w=w, c=c).items()}
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        sdfg(**args, K=7, N=33)
    np.testing.assert_allclose(cupy.asnumpy(args['a']), want_a, rtol=1e-13, atol=0)
    np.testing.assert_allclose(cupy.asnumpy(args['b']), want_b, rtol=1e-13, atol=0)


def carry_through_single_iteration_kernel(name: str, lanes: str = '0:1') -> dace.SDFG:
    """``for j: a[j] += a[j - 1]`` whose body is one ``GPU_Device`` map over ``lanes``, the shape the offload's hybrid
    resolution leaves around a sequential carry (npbench ``seidel_2d``'s ``j`` loop): one launch per trip."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', (16, ), dace.float64, storage=dace.StorageType.GPU_Global)
    loop = LoopRegion('jloop', 'j < 16', 'j', 'j = 1', 'j = j + 1')
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state('body', is_start_block=True)
    state.add_mapped_tasklet('carry', {'w': lanes}, {
        'prev': dace.Memlet('a[j - 1]'),
        'cur': dace.Memlet('a[j]')
    },
                             'out = cur + prev', {'out': dace.Memlet('a[j]')},
                             schedule=dace.ScheduleType.GPU_Device,
                             external_edges=True)
    return sdfg


def device_kernels(sdfg: dace.SDFG) -> list:
    return [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_Device
    ]


def test_a_loop_around_a_single_iteration_kernel_moves_into_it():
    """The GPU specialization of an offloaded graph runs the loop inside the one-thread kernel: one launch instead of
    one per trip. A single-iteration map has no thread axis to coalesce, so the loop axis being the contiguous one
    does not veto it, and with one lane the interchange cannot reorder anything."""
    sdfg = carry_through_single_iteration_kernel('size1_carry')
    finalize.gpu_specialize_offloaded(sdfg)
    assert top_level_loops(sdfg) == []
    kernels = device_kernels(sdfg)
    assert len(kernels) == 1 and str(kernels[0].map.range) == '0', kernels
    nested = [n for n in sdfg.start_block.nodes() if isinstance(n, dace.nodes.NestedSDFG)]
    assert len(nested) == 1
    assert [r.loop_variable for r in nested[0].sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)] == ['j']
    sdfg.validate()


def test_the_offloaded_graph_stage_leaves_a_loop_around_a_wide_kernel_alone():
    """Only single-iteration kernels are taken after the offload: a loop around a map with threads was already weighed
    by the canonicalize-time interchange, and its coalescing veto is not revisited here."""
    sdfg = carry_through_single_iteration_kernel('wide_carry', lanes='0:4')
    assert MoveLoopIntoMapGated(target='gpu', single_iteration_only=True).apply_pass(sdfg, {}) is None
    assert len(top_level_loops(sdfg)) == 1


@pytest.mark.gpu
def test_the_carry_in_a_single_iteration_kernel_matches_numpy():
    import cupy  # Only present on GPU runners.
    sdfg = carry_through_single_iteration_kernel('size1_carry_run')
    select_gpu_device_block_size(sdfg)  # What ``offload_to_gpu`` leaves on every kernel.
    finalize.gpu_specialize_offloaded(sdfg)
    assert len(device_kernels(sdfg)) == 1
    a = np.random.default_rng(0).random(16)
    want = a.copy()
    for j in range(1, 16):
        want[j] += want[j - 1]
    dev = cupy.asarray(a)
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        sdfg(a=dev)
    np.testing.assert_allclose(cupy.asnumpy(dev), want, rtol=1e-13, atol=0)
