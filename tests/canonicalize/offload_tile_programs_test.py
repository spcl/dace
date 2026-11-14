# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``offload_to_gpu(tile_programs=True)`` leaves the CUDA thread specialization out of a kernel a tile DSL writes.

A GPU wavefront tile skews its interior into a parallel map tagged ``is_warp_tile``; the default offload promotes it
to a thread block with a ``__syncthreads`` barrier after it. A tile program runs that map as tiles of its own, which
the vectorizer makes from a sequential map, so the tile-program offload leaves the tag unredeemed.
"""

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.finalize import offload_to_gpu
from dace.transformation.passes.parallelization.wavefront.skew import WavefrontSkew

N = dace.symbol("N")


@dace.program
def wavefront(a: dace.float64[N, N]):
    for i in range(1, N):
        for j in range(1, N):
            a[i, j] = (a[i, j] + a[i - 1, j] + a[i, j - 1]) / 3.0


def offloaded(tile_programs: bool) -> dace.SDFG:
    sdfg = wavefront.to_sdfg(simplify=True)
    skew = WavefrontSkew(target="gpu")
    skew.tile_i, skew.tile_j = 32, 32
    assert skew.apply_pass(sdfg, {}) == 1
    offload_to_gpu(sdfg, tile_programs=tile_programs)
    return sdfg


def schedules(sdfg: dace.SDFG) -> list[dace.ScheduleType]:
    return [n.map.schedule for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def cpp_tasklets(sdfg: dace.SDFG) -> list[str]:
    return [
        n.label
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.Tasklet) and n.language is dace.Language.CPP
    ]


def test_a_tile_program_offload_leaves_the_warp_tile_a_sequential_map():
    sdfg = offloaded(tile_programs=True)

    assert dace.ScheduleType.GPU_ThreadBlock not in schedules(sdfg)
    assert any(
        n.map.is_warp_tile and n.map.schedule is dace.ScheduleType.Sequential
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry)
    )
    assert not cpp_tasklets(sdfg)


def test_the_default_offload_promotes_the_warp_tile_to_a_thread_block():
    sdfg = offloaded(tile_programs=False)

    assert dace.ScheduleType.GPU_ThreadBlock in schedules(sdfg)
    assert "sync_threads" in cpp_tasklets(sdfg)


if __name__ == "__main__":
    test_a_tile_program_offload_leaves_the_warp_tile_a_sequential_map()
    test_the_default_offload_promotes_the_warp_tile_to_a_thread_block()
