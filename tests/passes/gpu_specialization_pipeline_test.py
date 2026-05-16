# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``GPUSpecializationPipeline`` idempotency and ``is_inside_gpu_device_kernel`` across nesting shapes."""
import dace
from dace import SDFG, dtypes
from dace.memlet import Memlet
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUSpecializationPipeline
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (
    get_gpu_stream_array_name,
    is_gpu_lowering_applied,
    is_inside_gpu_device_kernel,
)


def _build_simple_gpu_copy_sdfg() -> SDFG:
    """Tiny CPU->GPU->CPU pipeline: a host array staged into a GPU_Global transient and copied back,
    enough to trigger the full gpu_specialization pipeline."""
    sdfg = SDFG('idem_pipeline')
    sdfg.add_array('A', [16], dace.float32)
    sdfg.add_array('B', [16], dace.float32)
    sdfg.add_array('G', [16], dace.float32, storage=dtypes.StorageType.GPU_Global, transient=True)

    state = sdfg.add_state('s0')
    a = state.add_access('A')
    g1 = state.add_access('G')
    g2 = state.add_access('G')
    b = state.add_access('B')
    state.add_edge(a, None, g1, None, Memlet('G[0:16]'))
    state.add_edge(g1, None, g2, None, Memlet('G[0:16]'))
    state.add_edge(g2, None, b, None, Memlet('B[0:16]'))
    return sdfg


def _topology_signature(sdfg: SDFG):
    """A coarse but stable signature: array names + per-state node count."""
    arrays = tuple(sorted(sdfg.arrays.keys()))
    state_sizes = tuple((s.label, len(s.nodes()), len(list(s.edges()))) for s in sdfg.states())
    return arrays, state_sizes


def test_pipeline_idempotent_on_simple_sdfg():
    """Re-applying the pipeline is a no-op (returns ``{}``, topology untouched)."""
    sdfg = _build_simple_gpu_copy_sdfg()

    pipeline = GPUSpecializationPipeline()

    pipeline.apply_pass(sdfg, {})
    assert is_gpu_lowering_applied(sdfg), 'first pass must mark lowering as applied'
    assert get_gpu_stream_array_name() in sdfg.arrays
    sig_after_first = _topology_signature(sdfg)

    second = pipeline.apply_pass(sdfg, {})

    assert second == {}, 'a re-applied pipeline must be a no-op (return {})'
    assert _topology_signature(sdfg) == sig_after_first, 're-application must not mutate topology'

    # Defensive: still exactly one ``gpu_streams`` array.
    assert sum(1 for k in sdfg.arrays if k == get_gpu_stream_array_name()) == 1


def _trivial_inner_sdfg(name: str) -> SDFG:
    """Empty NestedSDFG with one state."""
    inner = SDFG(name)
    inner.add_state('s0')
    return inner


def _wrap_with_outer_map(inner: SDFG, schedule: dtypes.ScheduleType) -> SDFG:
    """Wrap ``inner`` inside an outer SDFG with a single map of the given schedule."""
    outer = SDFG(f'outer_{schedule.name}')
    state = outer.add_state('s0')
    nsdfg_node = state.add_nested_sdfg(inner, set(), set())
    me, mx = state.add_map('m', dict(i='0:1'), schedule=schedule)
    state.add_edge(me, None, nsdfg_node, None, Memlet())
    state.add_edge(nsdfg_node, None, mx, None, Memlet())
    return outer


def test_is_inside_gpu_device_kernel_true_for_inside_gpu_device_map():
    inner = _trivial_inner_sdfg('inner_gpu')
    _wrap_with_outer_map(inner, dtypes.ScheduleType.GPU_Device)
    assert is_inside_gpu_device_kernel(inner) is True


def test_is_inside_gpu_device_kernel_false_for_inside_sequential_map():
    inner = _trivial_inner_sdfg('inner_seq')
    _wrap_with_outer_map(inner, dtypes.ScheduleType.Sequential)
    assert is_inside_gpu_device_kernel(inner) is False


def test_is_inside_gpu_device_kernel_false_for_sibling_consumer():
    """Sibling-scope NSDFG consuming a kernel's output is not nested in the GPU_Device scope, so the
    answer is ``False`` (a naive data-flow predecessor walk would get this wrong)."""
    outer = SDFG('sibling')
    outer.add_array('G', [16], dace.float32, storage=dtypes.StorageType.GPU_Global, transient=True)
    state = outer.add_state('s0')

    # Kernel scope writing into G.
    g_in = state.add_access('G')
    me, mx = state.add_map('k', dict(i='0:16'), schedule=dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet('w', set(), {'g'}, 'g = 1.0f;', language=dtypes.Language.CPP)
    state.add_edge(me, None, tasklet, None, Memlet())
    mx.add_in_connector('IN_G')
    mx.add_out_connector('OUT_G')
    state.add_edge(tasklet, 'g', mx, 'IN_G', Memlet('G[i]'))
    state.add_edge(mx, 'OUT_G', g_in, None, Memlet('G[0:16]'))

    # Sibling NSDFG that reads G.
    inner = _trivial_inner_sdfg('sibling_inner')
    inner.add_array('g_in', [16], dace.float32, storage=dtypes.StorageType.GPU_Global)
    nsdfg_node = state.add_nested_sdfg(inner, {'g_in'}, set())
    state.add_edge(g_in, None, nsdfg_node, 'g_in', Memlet('G[0:16]'))

    assert is_inside_gpu_device_kernel(inner) is False


if __name__ == '__main__':
    test_pipeline_idempotent_on_simple_sdfg()
    test_is_inside_gpu_device_kernel_true_for_inside_gpu_device_map()
    test_is_inside_gpu_device_kernel_false_for_inside_sequential_map()
    test_is_inside_gpu_device_kernel_false_for_sibling_consumer()
