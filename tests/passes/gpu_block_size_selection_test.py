# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Structural tests for domain-matched GPU device-map block-size selection.

None of these need a GPU: they build SDFGs, run the selection pass, and inspect
``map.gpu_block_size`` -- pure SDFG manipulation.
"""

import pytest

import dace
from dace import dtypes, nodes, subsets
from dace.transformation.passes.gpu_block_size_selection import (
    domain_matched_block,
    pick_gpu_block_size,
    select_gpu_device_block_size,
)

N = dace.symbol("N")
M = dace.symbol("M")

#: The expected shapes: a line of 512, 256 threads with a 64-lane wavefront on the coalesced axis.
BLOCK_1D = [512, 1, 1]
BLOCK_2D = [64, 4, 1]
#: Without the lane floor (``compiler.cuda.block_min_lanes`` 1): a square of 256, rectangles following a skewed domain.
BLOCK_2D_SQUARE = [16, 16, 1]
#: ``i`` over 256, ``j`` (contiguous, ``x``) over 1024: four times as wide on ``x``.
BLOCK_2D_WIDE_X = [32, 8, 1]
BLOCK_2D_WIDE_Y = [8, 32, 1]


def without_lane_floor():
    """The plain domain-matched shape: no axis starts wider than one thread."""
    return dace.config.set_temporary("compiler", "cuda", "block_min_lanes", value=1)


def make_device_map(params, ranges):
    """A bare ``GPU_Device`` ``Map`` with the given params and (b, e, s) ranges."""
    return nodes.Map("kernel", list(params), subsets.Range(list(ranges)), schedule=dtypes.ScheduleType.GPU_Device)


def test_pick_block_size_1d():
    m = make_device_map(["i"], [(0, N - 1, 1)])
    assert pick_gpu_block_size(m) == BLOCK_1D


def test_pick_block_size_2d_square_symbolic():
    # Symbolic extents are assumed large and roughly equal -> square block.
    m = make_device_map(["i", "j"], [(0, N - 1, 1), (0, M - 1, 1)])
    with without_lane_floor():
        assert pick_gpu_block_size(m) == BLOCK_2D_SQUARE


def test_pick_block_size_2d_square_constant():
    m = make_device_map(["i", "j"], [(0, 1023, 1), (0, 1023, 1)])
    with without_lane_floor():
        assert pick_gpu_block_size(m) == BLOCK_2D_SQUARE


def test_pick_block_size_2d_skewed_last_dim_wide():
    # Last (contiguous, threadIdx.x) dimension is 4x larger -> 32 on x.
    m = make_device_map(["i", "j"], [(0, 255, 1), (0, 1023, 1)])
    with without_lane_floor():
        assert pick_gpu_block_size(m) == BLOCK_2D_WIDE_X


def test_pick_block_size_2d_skewed_outer_dim_wide():
    # Outer (threadIdx.y) dimension is 4x larger -> 32 on y.
    m = make_device_map(["i", "j"], [(0, 1023, 1), (0, 255, 1)])
    with without_lane_floor():
        assert pick_gpu_block_size(m) == BLOCK_2D_WIDE_Y


def make_wcr_reduction_sdfg():
    """A one-state SDFG with a 1-D ``GPU_Device`` map that accumulates into ``s`` via a WCR
    out of the map exit -- the tree-reduction shape."""
    sdfg = dace.SDFG("wcrred")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("s", [1], dace.float64)
    state = sdfg.add_state()
    a = state.add_access("a")
    s = state.add_access("s")
    me, mx = state.add_map("kernel", {"i": "0:N"}, schedule=dtypes.ScheduleType.GPU_Device)
    t = state.add_tasklet("acc", {"inp"}, {"out"}, "out = inp")
    state.add_memlet_path(a, me, t, dst_conn="inp", memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(t, mx, s, src_conn="out", memlet=dace.Memlet("s[0]", wcr="lambda x, y: x + y"))
    return sdfg, me.map


def test_tree_reduction_wcr_map_gets_deep_block():
    # A WCR-reduction device map (tree_reduction on) takes the deep 512-thread block, not the
    # 1-D 128 default, so the block-reduce folds more of the reduction per block.
    sdfg, m = make_wcr_reduction_sdfg()
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="legacy"):
        with dace.config.set_temporary("compiler", "emit_tree_reductions", value=True):
            select_gpu_device_block_size(sdfg)
    assert m.gpu_block_size == [512, 1, 1]


def test_wcr_map_uses_default_block_when_tree_reduction_off():
    # Legacy codegen with the flag off: the WCR write is a plain atomic, not a block tree-reduce,
    # so the map keeps the ordinary 1-D default block.
    sdfg, m = make_wcr_reduction_sdfg()
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="legacy"):
        with dace.config.set_temporary("compiler", "emit_tree_reductions", value=False):
            select_gpu_device_block_size(sdfg)
    assert m.gpu_block_size == BLOCK_1D


def test_experimental_codegen_gets_deep_block_with_the_flag_off():
    # ``compiler.emit_tree_reductions`` is legacy-only: the experimental CUDA target always folds
    # the WCR through gpucub::BlockReduce, so the map still wants the deep block with the flag off.
    sdfg, m = make_wcr_reduction_sdfg()
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        with dace.config.set_temporary("compiler", "emit_tree_reductions", value=False):
            select_gpu_device_block_size(sdfg)
    assert m.gpu_block_size == [512, 1, 1]


def test_pick_block_size_2d_mild_ratio_leans_toward_the_wider_axis():
    with without_lane_floor():
        m = make_device_map(["i", "j"], [(0, 599, 1), (0, 1023, 1)])
        assert pick_gpu_block_size(m) == [16, 16, 1]
        m = make_device_map(["i", "j"], [(0, 299, 1), (0, 1023, 1)])
        assert pick_gpu_block_size(m) == [32, 8, 1]


def test_pick_block_size_3d_is_a_cube():
    m = make_device_map(["i", "j", "k"], [(0, N - 1, 1), (0, N - 1, 1), (0, N - 1, 1)])
    with without_lane_floor():
        assert pick_gpu_block_size(m) == [8, 8, 8]


def test_the_coalesced_axis_starts_at_a_wavefront():
    """One CDNA wavefront (two NVIDIA warps) of contiguous elements per load, wherever the domain is that wide; the
    rest of the block still follows the domain."""
    assert pick_gpu_block_size(make_device_map(["i", "j"], [(0, N - 1, 1), (0, M - 1, 1)])) == BLOCK_2D
    assert pick_gpu_block_size(make_device_map(["i", "j"], [(0, 1023, 1), (0, 255, 1)])) == [64, 4, 1]
    assert pick_gpu_block_size(make_device_map(["i", "j", "k"], [(0, N - 1, 1)] * 3)) == [64, 4, 2]
    # The unit-stride axis takes the lanes, not ``x``; an axis narrower than the floor keeps its extent.
    assert domain_matched_block([N, N], [0, 1]) == [4, 64, 1]
    assert domain_matched_block([8, N]) == [8, 32, 1]
    with dace.config.set_temporary("compiler", "cuda", "block_min_lanes", value=32):
        assert domain_matched_block([N, N]) == [32, 8, 1]


def test_pick_block_size_4d_leaves_default():
    m = make_device_map(["i", "j", "k", "l"], [(0, N - 1, 1)] * 4)
    assert pick_gpu_block_size(m) is None


def test_a_small_axis_hands_its_threads_to_the_unit_stride_axis():
    """An axis of four takes no thread past a symbolic one; of the two symbolic axes the one indexed at unit stride
    more often takes the extra doubling."""
    with without_lane_floor():
        assert domain_matched_block([N, 4, N], [2, 0, 1]) == [32, 1, 16]
        assert domain_matched_block([N, 4, N], [1, 0, 2]) == [16, 1, 32]


def test_a_domain_smaller_than_the_block_keeps_only_the_threads_it_uses():
    assert domain_matched_block([100]) == [128, 1, 1]
    assert domain_matched_block([3, 5]) == [4, 8, 1]


def test_every_block_stays_in_the_thread_band():
    """256 to 512 threads wherever the domain can fill them, at most three axes."""
    for params, ranges in (
        (["i"], [(0, N - 1, 1)]),
        (["i", "j"], [(0, N - 1, 1), (0, M - 1, 1)]),
        (["i", "j"], [(0, 1023, 1), (0, 7, 1)]),
        (["i", "j"], [(0, 7, 1), (0, 1023, 1)]),
        (["i", "j", "k"], [(0, N - 1, 1), (0, 63, 1), (0, 1, 1)]),
    ):
        block = pick_gpu_block_size(make_device_map(params, ranges))
        assert 256 <= block[0] * block[1] * block[2] <= 512, block


def build_2d_map_sdfg(schedule):
    """Minimal SDFG: one 64x64 map writing a 2-D array on the given schedule."""
    sdfg = dace.SDFG("single_map_build_2d_map_sdfg")
    sdfg.add_array("A", [64, 64], dace.float64)
    state = sdfg.add_state()
    _, me, _ = state.add_mapped_tasklet(
        "kernel",
        {"i": "0:64", "j": "0:64"},
        {},
        "o = 1.0",
        {"o": dace.Memlet("A[i, j]")},
        schedule=schedule,
        external_edges=True,
    )
    return sdfg, me


def device_maps(sdfg):
    return [
        n.map
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_Device
    ]


def test_pass_assigns_2d_square():
    sdfg, _ = build_2d_map_sdfg(dtypes.ScheduleType.GPU_Device)
    assigned = select_gpu_device_block_size(sdfg)
    assert assigned == {"kernel_map": BLOCK_2D}
    assert device_maps(sdfg)[0].gpu_block_size == BLOCK_2D


def test_pass_does_not_override_user_block_size():
    sdfg, me = build_2d_map_sdfg(dtypes.ScheduleType.GPU_Device)
    me.map.gpu_block_size = [8, 8, 1]
    assert select_gpu_device_block_size(sdfg) == {}
    assert device_maps(sdfg)[0].gpu_block_size == [8, 8, 1]


def two_device_maps_sdfg() -> dace.SDFG:
    """Two 1-D ``GPU_Device`` maps, one per state, in that order."""
    sdfg = dace.SDFG("two_kernels")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    first = sdfg.add_state("first")
    second = sdfg.add_state_after(first, "second")
    for state, label in ((first, "first"), (second, "second")):
        state.add_mapped_tasklet(
            label,
            {"i": "0:N"},
            {"a": dace.Memlet("A[i]")},
            "b = a + 1",
            {"b": dace.Memlet("B[i]")},
            schedule=dtypes.ScheduleType.GPU_Device,
            external_edges=True,
        )
    return sdfg


def test_an_override_gives_the_map_of_its_ordinal_that_block_and_leaves_the_others_to_the_pick():
    sdfg = two_device_maps_sdfg()

    with dace.config.set_temporary("compiler", "cuda", "block_size_overrides", value="1:32,4,1"):
        assigned = select_gpu_device_block_size(sdfg)

    assert assigned == {"first_map": BLOCK_1D, "second_map": [32, 4, 1]}
    assert [m.gpu_block_size for m in device_maps(sdfg)] == [BLOCK_1D, [32, 4, 1]]


def test_a_malformed_override_is_refused_by_its_entry():
    with dace.config.set_temporary("compiler", "cuda", "block_size_overrides", value="0:32,4"):
        with pytest.raises(ValueError, match="'0:32,4' is not ordinal:x,y,z"):
            select_gpu_device_block_size(two_device_maps_sdfg())


def test_pass_ignores_cpu_map():
    sdfg, _ = build_2d_map_sdfg(dtypes.ScheduleType.CPU_Multicore)
    assert select_gpu_device_block_size(sdfg) == {}


def test_gpu_transform_end_to_end():
    # Build a plain 2-D map, GPU-transform the SDFG, then select block sizes.
    @dace.program
    def prog2d(a: dace.float64[N, N]):
        for i, j in dace.map[0:N, 0:N]:
            a[i, j] = 1.0

    sdfg = prog2d.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    dmaps = device_maps(sdfg)
    assert len(dmaps) == 1
    assert dmaps[0].gpu_block_size is None  # transform leaves it unset
    select_gpu_device_block_size(sdfg)
    assert dmaps[0].gpu_block_size == BLOCK_2D


def test_gpu_transform_1d_end_to_end():

    @dace.program
    def prog1d(a: dace.float64[N]):
        for i in dace.map[0:N]:
            a[i] = 1.0

    sdfg = prog1d.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    dmaps = device_maps(sdfg)
    assert len(dmaps) == 1
    select_gpu_device_block_size(sdfg)
    assert dmaps[0].gpu_block_size == BLOCK_1D


def test_config_default_block_size_is_128():
    assert dace.Config.get("compiler", "cuda", "default_block_size") == "128,1,1"


def build_kernel_over_inner_map(inner_schedule, warp_tile=False):
    """A ``GPU_Device`` map over ``i`` wrapping a second map over ``t`` -- the shape a kernel has
    once something below it owns, or is about to own, the thread-block level."""
    sdfg = dace.SDFG("kernel_over_tile")
    sdfg.add_array("A", [64, 64], dace.float64)
    state = sdfg.add_state()
    a = state.add_access("A")
    outer_entry, outer_exit = state.add_map("kernel", {"i": "0:64"}, schedule=dtypes.ScheduleType.GPU_Device)
    inner_entry, inner_exit = state.add_map("tile", {"t": "0:64"}, schedule=inner_schedule)
    inner_entry.map.is_warp_tile = warp_tile
    tasklet = state.add_tasklet("write", {}, {"o"}, "o = 1.0")
    # Empty memlets down the entry chain: the tasklet reads nothing, so these edges carry no data
    # and only place it inside both scopes.
    state.add_nedge(outer_entry, inner_entry, dace.Memlet())
    state.add_nedge(inner_entry, tasklet, dace.Memlet())
    state.add_memlet_path(tasklet, inner_exit, outer_exit, a, src_conn="o", memlet=dace.Memlet("A[i, t]"))
    sdfg.validate()
    return sdfg, outer_entry.map, inner_entry.map


def test_pass_skips_a_kernel_that_already_has_a_thread_block_map():
    # The kernel takes its block from the inner map. A declared size on top of that is the pair
    # codegen refuses.
    sdfg, kernel, _ = build_kernel_over_inner_map(dtypes.ScheduleType.GPU_ThreadBlock)
    assert select_gpu_device_block_size(sdfg) == {}
    assert kernel.gpu_block_size is None


def test_pass_skips_a_kernel_whose_warp_tile_is_not_promoted_yet():
    # ``is_warp_tile`` on a sequentialized map is a thread-block map ``PromoteWarpTiles`` has not
    # created yet -- it runs in codegen preprocessing, after this pass. Sizing the kernel now is the
    # same conflict, deferred until the promotion happens.
    sdfg, kernel, tile = build_kernel_over_inner_map(dtypes.ScheduleType.Sequential, warp_tile=True)
    assert select_gpu_device_block_size(sdfg) == {}
    assert kernel.gpu_block_size is None
    # The skip is all this pass does about the tag; promoting is not its job.
    assert tile.schedule == dtypes.ScheduleType.Sequential


def test_pass_still_sizes_a_kernel_over_an_untagged_sequential_map():
    # No tag and no thread-block schedule: nothing under the kernel will ever own the block, so the
    # 1-D domain takes the ordinary default.
    sdfg, kernel, _ = build_kernel_over_inner_map(dtypes.ScheduleType.Sequential)
    # A 64-iteration kernel keeps the 64 threads it can use.
    assert select_gpu_device_block_size(sdfg) == {"kernel": [64, 1, 1]}
    assert kernel.gpu_block_size == [64, 1, 1]


def test_a_pending_warp_tile_reaches_codegen_without_a_block_size_conflict():
    """End to end: size the kernel, then let codegen preprocessing promote the warp tile.

    ``PromoteWarpTiles`` runs inside ``GPUCodegenPreprocessPipeline``, so the thread-block map
    appears after this pass has already run. While the pass also claimed such a kernel, the two
    sizes disagreed and ``InferGPUGridAndBlockSize`` refused the pair -- rightly, because the kernel
    would have launched 128 threads and indexed 64.
    """
    sdfg, kernel, _ = build_kernel_over_inner_map(dtypes.ScheduleType.Sequential, warp_tile=True)
    for desc in sdfg.arrays.values():
        desc.storage = dtypes.StorageType.GPU_Global
    select_gpu_device_block_size(sdfg)
    assert kernel.gpu_block_size is None
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        code = "\n".join(obj.clean_code for obj in sdfg.generate_code())
    # The tile is what sizes the block: 64 threads on x, and the launch dimensions say so.
    assert "dim3(64, 1, 1)" in code, code


def test_the_z_axis_stays_within_the_last_dimension_limit():
    """A 3-D map whose only wide axis is the outermost puts that axis on z, which holds at most
    ``compiler.cuda.block_size_lastdim_limit`` threads; past it the kernel cannot launch (seissol_tensor_contraction
    asked for 512)."""
    limit = dace.Config.get("compiler", "cuda", "block_size_lastdim_limit")
    assert domain_matched_block([1, 1, dace.symbol("N")]) == [1, 1, limit]


def test_pass_is_idempotent():
    sdfg, _ = build_2d_map_sdfg(dtypes.ScheduleType.GPU_Device)
    select_gpu_device_block_size(sdfg)
    # Re-running assigns nothing (already set) and preserves the value.
    assert select_gpu_device_block_size(sdfg) == {}
    assert device_maps(sdfg)[0].gpu_block_size == BLOCK_2D


if __name__ == "__main__":
    test_pick_block_size_1d()
    test_pick_block_size_2d_square_symbolic()
    test_pick_block_size_2d_square_constant()
    test_pick_block_size_2d_skewed_last_dim_wide()
    test_pick_block_size_2d_skewed_outer_dim_wide()
    test_pick_block_size_2d_mild_ratio_leans_toward_the_wider_axis()
    test_pick_block_size_3d_is_a_cube()
    test_the_coalesced_axis_starts_at_a_wavefront()
    test_pick_block_size_4d_leaves_default()
    test_a_small_axis_hands_its_threads_to_the_unit_stride_axis()
    test_a_domain_smaller_than_the_block_keeps_only_the_threads_it_uses()
    test_every_block_stays_in_the_thread_band()
    test_pass_assigns_2d_square()
    test_pass_does_not_override_user_block_size()
    test_an_override_gives_the_map_of_its_ordinal_that_block_and_leaves_the_others_to_the_pick()
    test_a_malformed_override_is_refused_by_its_entry()
    test_pass_ignores_cpu_map()
    test_gpu_transform_end_to_end()
    test_gpu_transform_1d_end_to_end()
    test_config_default_block_size_is_128()
    test_the_z_axis_stays_within_the_last_dimension_limit()
    test_pass_is_idempotent()
    test_pass_skips_a_kernel_that_already_has_a_thread_block_map()
    test_pass_skips_a_kernel_whose_warp_tile_is_not_promoted_yet()
    test_pass_still_sizes_a_kernel_over_an_untagged_sequential_map()
    test_a_pending_warp_tile_reaches_codegen_without_a_block_size_conflict()
    test_tree_reduction_wcr_map_gets_deep_block()
    test_wcr_map_uses_default_block_when_tree_reduction_off()
    test_experimental_codegen_gets_deep_block_with_the_flag_off()
    print("OK")
