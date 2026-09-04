# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU tile-vectorizer applicability rule: only tile an innermost map running
inside a GPU kernel.

half2 (FP16x2) tile ops lower to ``__device__`` intrinsics -> compile only inside a
GPU kernel. Rule (``is_gpu_resident_map`` / ``MarkTileDims(require_gpu_resident=True)``):
tile an innermost map iff either

* ``GPU_Device``-scheduled (map is itself the kernel), or
* non-kernel (e.g. ``Sequential``) map with a ``GPU_Device`` PARENT map.

Host-side map -> skipped. No nested SDFG needed: a parent GPU_Device map in the same
state's scope tree already carries a sequential inner map onto the device.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import dace
from dace.dtypes import ScheduleType
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.utils.map_predicates import is_gpu_resident_map, is_innermost_map


def _single_map_sdfg(sched):
    """A single innermost map ``j -> B[j] = A[j]`` scheduled ``sched``."""
    sdfg = dace.SDFG("single_map")
    sdfg.add_array("A", [8], dace.float16)
    sdfg.add_array("B", [8], dace.float16)
    state = sdfg.add_state()
    a, b = state.add_read("A"), state.add_write("B")
    me, mx = state.add_map("m", dict(j="0:8"), schedule=sched)
    t = state.add_tasklet("c", {"inp"}, {"out"}, "out = inp")
    state.add_memlet_path(a, me, t, dst_conn="inp", memlet=dace.Memlet("A[j]"))
    state.add_memlet_path(t, mx, b, src_conn="out", memlet=dace.Memlet("B[j]"))
    return sdfg, state, me


def _nested_maps_sdfg(outer_sched, inner_sched):
    """A perfectly nested ``i`` (outer) / ``j`` (inner) map pair, each schedulable."""
    sdfg = dace.SDFG("nested_maps")
    sdfg.add_array("A", [8, 8], dace.float16)
    sdfg.add_array("B", [8, 8], dace.float16)
    state = sdfg.add_state()
    a, b = state.add_read("A"), state.add_write("B")
    oe, ox = state.add_map("outer", dict(i="0:8"), schedule=outer_sched)
    ie, ix = state.add_map("inner", dict(j="0:8"), schedule=inner_sched)
    t = state.add_tasklet("c", {"inp"}, {"out"}, "out = inp")
    state.add_memlet_path(a, oe, ie, t, dst_conn="inp", memlet=dace.Memlet("A[i, j]"))
    state.add_memlet_path(t, ix, ox, b, src_conn="out", memlet=dace.Memlet("B[i, j]"))
    return sdfg, state, oe, ie


def test_gpu_device_map_is_resident():
    """A GPU_Device-scheduled map is itself the kernel -> resident."""
    _, state, me = _single_map_sdfg(ScheduleType.GPU_Device)
    assert is_gpu_resident_map(state, me)


def test_host_map_is_not_resident():
    """A top-level host map (no GPU parent) is not resident."""
    for sched in (ScheduleType.Sequential, ScheduleType.CPU_Multicore):
        _, state, me = _single_map_sdfg(sched)
        assert not is_gpu_resident_map(state, me), sched


def test_sequential_inner_with_gpu_parent_is_resident():
    """A Sequential inner map under a GPU_Device parent map runs on the device."""
    _, state, oe, ie = _nested_maps_sdfg(ScheduleType.GPU_Device, ScheduleType.Sequential)
    assert is_innermost_map(state, ie)
    assert is_gpu_resident_map(state, ie)  # sequential inner, GPU_Device parent
    assert is_gpu_resident_map(state, oe)  # the parent kernel itself


def test_sequential_inner_without_gpu_parent_is_not_resident():
    """The same nest with a host parent map stays host-side."""
    _, state, _oe, ie = _nested_maps_sdfg(ScheduleType.CPU_Multicore, ScheduleType.Sequential)
    assert not is_gpu_resident_map(state, ie)


def test_mark_tile_dims_skips_host_map_under_gpu_gate():
    """With ``require_gpu_resident``, a host innermost map produces no tile spec;
    without the gate, the same map tiles normally."""
    sdfg, _state, me = _single_map_sdfg(ScheduleType.Sequential)
    assert MarkTileDims(widths=(2, ), require_gpu_resident=True).apply_pass(sdfg, {}) is None
    specs = MarkTileDims(widths=(2, ), require_gpu_resident=False).apply_pass(sdfg, {})
    assert specs is not None and me in specs


def test_mark_tile_dims_tiles_gpu_resident_maps():
    """The gate keeps a GPU_Device map and a Sequential-under-GPU inner map."""
    sdfg, _s, me = _single_map_sdfg(ScheduleType.GPU_Device)
    specs = MarkTileDims(widths=(2, ), require_gpu_resident=True).apply_pass(sdfg, {})
    assert specs is not None and me in specs

    sdfg2, _s2, _oe, ie = _nested_maps_sdfg(ScheduleType.GPU_Device, ScheduleType.Sequential)
    specs2 = MarkTileDims(widths=(2, ), require_gpu_resident=True).apply_pass(sdfg2, {})
    assert specs2 is not None and ie in specs2  # inner sequential map, tiled (device-resident)


if __name__ == "__main__":
    test_gpu_device_map_is_resident()
    test_host_map_is_not_resident()
    test_sequential_inner_with_gpu_parent_is_resident()
    test_sequential_inner_without_gpu_parent_is_not_resident()
    test_mark_tile_dims_skips_host_map_under_gpu_gate()
    test_mark_tile_dims_tiles_gpu_resident_maps()
    print("ok")
