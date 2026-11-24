# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map over a small block axis (ICON's ``jb`` over ``nblks``, or a declared-small extent) stays a sequential host map
around the kernels of its body."""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen import cpf
from dace.sdfg.utils import specialize_symbols
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.cpu_specialization.demote_level_maps import small_names

N = dace.symbol("N")


def blocked_columns(blocks: str) -> dace.SDFG:
    """Rows of ``N`` cells in blocks, the block count named ``blocks``: each block fills a row buffer with a map, scans
    it with a loop and writes it out with a map, so the block map holds maps (ICON's ``jb`` around its ``jc`` maps)."""
    NB = dace.symbol(blocks)

    @dace.program
    def program(A: dace.float64[NB, N], S: dace.float64[NB, N]):
        for jb in dace.map[0:NB]:
            row = np.empty((N,), dace.float64)
            for jc in dace.map[0:N]:
                row[jc] = 2.0 * A[jb, jc]
            for jc in range(1, N):
                row[jc] = row[jc] + row[jc - 1]
            for jc in dace.map[0:N]:
                S[jb, jc] = row[jc] + A[jb, jc]

    return program.to_sdfg(simplify=True)


def offloaded(blocks: str, small: frozenset[str] = frozenset(), constants: dict[str, int] | None = None) -> dace.SDFG:
    sdfg = blocked_columns(blocks)
    if constants:
        # Configuration constants, as dace-fortran's VT pipeline bakes ICON's block counts
        specialize_symbols(sdfg, constants)
    canonicalize(sdfg, target="gpu")
    offload_to_gpu(sdfg, small=small)
    finalize_for_target(sdfg, "gpu")
    return sdfg


def block_maps(sdfg: dace.SDFG, blocks: str) -> list[dace.nodes.MapEntry]:
    """The maps over the blocks alone, whatever canonicalization named their parameter."""
    return [
        node
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry) and [str(size) for size in node.map.range.size()] == [blocks]
    ]


def kernels(sdfg: dace.SDFG) -> list[dace.nodes.MapEntry]:
    return [
        node
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device
    ]


def device_maps(sdfg: dace.SDFG) -> list[dace.nodes.MapEntry]:
    """The kernels and every map scheduled inside one: grid, thread-block and lane maps."""
    return [
        node
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule in dtypes.GPU_SCHEDULES
    ]


def assert_sequential_host_map_around_the_kernels(sdfg: dace.SDFG, blocks: str) -> None:
    outer = block_maps(sdfg, blocks)
    assert outer and all(entry.map.schedule == dtypes.ScheduleType.Sequential for entry in outer)
    assert all(f"small dimension ({blocks})" in entry.specialization_hint for entry in outer)
    assert kernels(sdfg)
    # No kernel over the blocks: not as its grid, and not lane-strided inside one (a size-1 wrap kernel's lanes)
    block_params = {param for entry in outer for param in entry.map.params}
    for entry in device_maps(sdfg):
        assert blocks not in map(str, entry.map.range.free_symbols) and block_params.isdisjoint(entry.map.params)


def test_a_map_over_blocks_of_unknown_size_stays_the_kernel():
    outer = block_maps(offloaded("NB"), "NB")
    assert outer and all(entry.map.schedule == dtypes.ScheduleType.GPU_Device for entry in outer)


def test_a_map_over_nblks_is_a_sequential_host_map_around_the_kernels():
    assert_sequential_host_map_around_the_kernels(offloaded("nblks"), "nblks")


def test_a_map_over_declared_small_blocks_is_a_sequential_host_map_around_the_kernels():
    assert_sequential_host_map_around_the_kernels(offloaded("NB", frozenset({"NB"})), "NB")


def test_a_single_block_baked_before_canonicalization_leaves_kernels_over_the_columns_only():
    sdfg = offloaded("nblks", constants={"nblks": 1})
    assert kernels(sdfg)
    assert not any("nblks" in map(str, entry.map.range.free_symbols) for entry in device_maps(sdfg))


@pytest.mark.gpu
def test_a_single_baked_block_matches_numpy():
    cupy = pytest.importorskip("cupy")
    A = np.random.default_rng(6).random((1, 700))
    sdfg = offloaded("nblks", constants={"nblks": 1})
    S = cupy.zeros_like(cupy.asarray(A))
    sdfg(A=cupy.asarray(A), S=S, N=700)
    np.testing.assert_allclose(S.get(), np.cumsum(2.0 * A, axis=1) + A, rtol=1e-13)


@pytest.mark.gpu
def test_the_rendering_says_why_the_block_map_is_sequential():
    rendering = cpf.render(offloaded("nblks"), language="cuda")
    assert "parallel -- generated as sequential: small dimension (nblks)" in rendering.code


def test_a_symbol_read_from_a_small_table_is_small():
    sdfg = dace.SDFG("block_bounds")
    sdfg.add_array("end_block", [3], dace.int32)
    sdfg.add_symbol("nb", dace.int32)
    sdfg.add_symbol("other", dace.int32)
    sdfg.add_symbol("N", dace.int32)
    first = sdfg.add_state("first", is_start_block=True)
    sdfg.add_state_after(first, "second", assignments={"nb": "end_block[2 - 1]", "other": "nb + N"})
    assert small_names(sdfg, {"end_block"}) >= {"end_block", "nb"}
    assert "other" not in small_names(sdfg, {"end_block"})


def test_icon_block_tables_are_small_by_default():
    sdfg = dace.SDFG("icon_block_bounds")
    sdfg.add_array("p_patch_cells_end_block", [14], dace.int32)
    sdfg.add_symbol("i_endblk", dace.int32)
    first = sdfg.add_state("first", is_start_block=True)
    sdfg.add_state_after(first, "second", assignments={"i_endblk": "p_patch_cells_end_block[4]"})
    assert "i_endblk" in small_names(sdfg)


@pytest.mark.gpu
def test_a_map_over_nblks_matches_numpy():
    cupy = pytest.importorskip("cupy")
    rng = np.random.default_rng(5)
    A = rng.random((3, 700))
    sdfg = offloaded("nblks")
    S = cupy.zeros_like(cupy.asarray(A))
    sdfg(A=cupy.asarray(A), S=S, nblks=3, N=700)
    np.testing.assert_allclose(S.get(), np.cumsum(2.0 * A, axis=1) + A, rtol=1e-13)


if __name__ == "__main__":
    test_a_map_over_blocks_of_unknown_size_stays_the_kernel()
    test_a_map_over_nblks_is_a_sequential_host_map_around_the_kernels()
    test_a_map_over_declared_small_blocks_is_a_sequential_host_map_around_the_kernels()
    test_a_single_block_baked_before_canonicalization_leaves_kernels_over_the_columns_only()
    test_a_single_baked_block_matches_numpy()
    test_the_rendering_says_why_the_block_map_is_sequential()
    test_a_symbol_read_from_a_small_table_is_small()
    test_icon_block_tables_are_small_by_default()
    test_a_map_over_nblks_matches_numpy()
