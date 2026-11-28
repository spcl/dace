# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A block map over a small extent (a literal, or symbols declared ``SMALL``), or one the caller made sequential, stays
a host map around the kernels of its body; a block map over a symbolic count stays the grid."""

import numpy as np
import pytest

import dace
from dace import dtypes, symbolic
from dace.codegen import cpf
from dace.sdfg.utils import specialize_symbols
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.cpu_specialization.demote_level_maps import block_params, rejoin_block_maps

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


NB, K, lo, hi = (dace.symbol(name) for name in ("NB", "K", "lo", "hi"))


@dace.program
def blocked_levels(A: dace.float64[NB, K], B: dace.float64[NB, K]):
    # ICON's jb x jk: the block range comes from bounds (lo, hi) of default scale, the arrays are sized by NB
    for jb, jk in dace.map[lo:hi, 0:K]:
        B[jb, jk] = 2.0 * A[jb, jk]


def levels_offloaded(small: bool) -> dace.SDFG:
    sdfg = blocked_levels.to_sdfg(simplify=True)
    if small:
        sdfg.set_symbol_scale("NB", symbolic.Scale.SMALL)
    canonicalize(sdfg, target="gpu")
    return offloaded(sdfg)


def test_a_block_dimension_of_a_map_without_inner_maps_stays_in_the_one_kernel():
    """ICON's ``jb`` x ``jk`` with no map inside is all parallelism: one kernel over both, not a host loop over the
    blocks launching a kernel each (velocity_tendencies left a MapFusion buffer in host memory between them)."""
    from dace.sdfg.scope import is_devicelevel_gpu

    sdfg = levels_offloaded(small=True)
    hosts = [
        entry
        for entry, state in sdfg.all_nodes_recursive()
        if isinstance(entry, dace.nodes.MapEntry)
        and entry.map.schedule == dtypes.ScheduleType.Sequential
        and not is_devicelevel_gpu(state.sdfg, state, entry)
    ]
    assert not hosts, "a host map around the kernel"
    assert len(kernels(sdfg)) == 1
    assert {"lo", "hi"} <= {str(free) for free in kernels(sdfg)[0].map.range.free_symbols}


def test_a_block_dimension_of_undeclared_scale_stays_in_the_grid():
    sdfg = levels_offloaded(small=False)
    assert any({"lo", "hi"} & {str(free) for free in entry.map.range.free_symbols} for entry in kernels(sdfg))


@pytest.mark.gpu
def test_a_block_dimension_in_the_kernel_matches_numpy():
    cupy = pytest.importorskip("cupy")
    A = np.random.default_rng(8).random((4, 300))
    B = cupy.zeros_like(cupy.asarray(A))
    levels_offloaded(small=True)(A=cupy.asarray(A), B=B, NB=4, K=300, lo=1, hi=3)
    expected = np.zeros_like(A)
    expected[1:3] = 2.0 * A[1:3]
    np.testing.assert_array_equal(B.get(), expected)


def canonical(blocks: str, constants: dict[str, int] | None = None, small: bool = False) -> dace.SDFG:
    sdfg = blocked_columns(blocks)
    if constants:
        # Configuration constants, as dace-fortran's VT pipeline bakes ICON's block counts
        specialize_symbols(sdfg, constants)
    if small:
        sdfg.set_symbol_scale(blocks, symbolic.Scale.SMALL)
    canonicalize(sdfg, target="gpu")
    return sdfg


def offloaded(sdfg: dace.SDFG) -> dace.SDFG:
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu")
    return sdfg


def block_maps(sdfg: dace.SDFG, extent: str) -> list[dace.nodes.MapEntry]:
    """The maps over the blocks alone, whatever canonicalization named their parameter."""
    return [
        node
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry) and [str(size) for size in node.map.range.size()] == [extent]
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


def assert_sequential_host_map_around_the_kernels(sdfg: dace.SDFG, extent: str, hint: str | None) -> None:
    outer = block_maps(sdfg, extent)
    assert outer and all(entry.map.schedule == dtypes.ScheduleType.Sequential for entry in outer)
    if hint is not None:
        assert all(hint in entry.specialization_hint for entry in outer)
    assert kernels(sdfg)
    # No kernel over the blocks: not as its grid, and not lane-strided inside one (a size-1 wrap kernel's lanes)
    block_params = {param for entry in outer for param in entry.map.params}
    for entry in device_maps(sdfg):
        assert extent not in map(str, entry.map.range.free_symbols) and block_params.isdisjoint(entry.map.params)


@pytest.mark.parametrize("blocks", ["NB", "nblks"])
def test_a_map_over_a_symbolic_block_count_stays_the_grid(blocks: str):
    # CloudSC's NBLOCKS is large: a block map is small by a constant or a SMALL declaration, never by its name
    outer = block_maps(offloaded(canonical(blocks)), blocks)
    assert outer and all(entry.map.schedule == dtypes.ScheduleType.GPU_Device for entry in outer)


def test_a_map_over_blocks_declared_small_is_a_sequential_host_map_around_the_kernels():
    sdfg = offloaded(canonical("NB", small=True))
    assert_sequential_host_map_around_the_kernels(sdfg, "NB", "small dimension (NB)")


def test_a_map_over_sixteen_baked_blocks_is_no_kernel():
    # Below a schedule's share-out size (64): a host map, unrolled once sequential, around the column kernels
    sdfg = offloaded(canonical("nblks", constants={"nblks": 16}))
    assert kernels(sdfg)
    assert not any(entry.map.schedule in dtypes.GPU_SCHEDULES for entry in block_maps(sdfg, "16"))


def test_a_block_map_the_caller_made_sequential_stays_on_the_host():
    # ICON's multiblock VT: the block count stays symbolic, and the driver keeps jb on the host
    sdfg = canonical("nblks")
    for entry in block_maps(sdfg, "nblks"):
        entry.map.schedule = dtypes.ScheduleType.Sequential
    assert_sequential_host_map_around_the_kernels(offloaded(sdfg), "nblks", None)


def test_a_single_block_baked_before_canonicalization_leaves_kernels_over_the_columns_only():
    sdfg = offloaded(canonical("nblks", constants={"nblks": 1}))
    assert kernels(sdfg)
    assert not any("nblks" in map(str, entry.map.range.free_symbols) for entry in device_maps(sdfg))


@pytest.mark.gpu
def test_the_rendering_says_why_the_block_map_is_sequential():
    rendering = cpf.render(offloaded(canonical("NB", small=True)), language="cuda")
    assert "parallel -- generated as sequential: small dimension (NB), so the maps inside run as the GPU kernels" in (
        rendering.code
    )


def run_on_gpu(sdfg: dace.SDFG, blocks: int, **symbols: int) -> None:
    cupy = pytest.importorskip("cupy")
    A = np.random.default_rng(blocks).random((blocks, 700))
    S = cupy.zeros_like(cupy.asarray(A))
    sdfg(A=cupy.asarray(A), S=S, N=700, **symbols)
    np.testing.assert_allclose(S.get(), np.cumsum(2.0 * A, axis=1) + A, rtol=1e-13)


@pytest.mark.gpu
def test_a_single_baked_block_matches_numpy():
    run_on_gpu(offloaded(canonical("nblks", constants={"nblks": 1})), 1)


@pytest.mark.gpu
def test_sixteen_baked_blocks_match_numpy():
    run_on_gpu(offloaded(canonical("nblks", constants={"nblks": 16})), 16)


@pytest.mark.gpu
def test_a_block_map_the_caller_made_sequential_matches_numpy():
    sdfg = canonical("nblks")
    for entry in block_maps(sdfg, "nblks"):
        entry.map.schedule = dtypes.ScheduleType.Sequential
    run_on_gpu(offloaded(sdfg), 3, nblks=3)


def test_a_block_index_read_through_an_approximated_bound_is_found():
    # A device map's memlets carry SymExpr bounds (exact value, approximation) once offloaded
    sdfg = dace.SDFG("approximated_bound")
    sdfg.add_array("A", [NB, K], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("blocks", {"jb": "0:NB"})
    tasklet = state.add_tasklet("t", {"a"}, set(), "pass")
    bound = symbolic.SymExpr("jb", "0")
    state.add_memlet_path(
        state.add_read("A"),
        entry,
        tasklet,
        dst_conn="a",
        memlet=dace.Memlet(data="A", subset=dace.subsets.Range([(bound, bound, 1), (0, K - 1, 1)])),
    )
    state.add_nedge(tasklet, exit_, dace.Memlet())
    assert block_params(state, entry, {"NB"}) == {"jb": {"NB"}}


def test_a_host_block_map_around_one_kernel_map_becomes_one_kernel():
    """A host block map left around one perfectly nested kernel map (ICON's ``jb`` around its ``jc`` lanes after the
    offload) collapses into one kernel over both, so the host launches once, not once per block."""
    sdfg = dace.SDFG("rejoin")
    sdfg.add_array("A", (NB, N), dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map("blocks", {"jb": "0:NB"}, schedule=dtypes.ScheduleType.Sequential)
    inner_entry, inner_exit = state.add_map("lanes", {"jc": "0:N"}, schedule=dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet("set", {}, {"out"}, "out = 1.0")
    write = state.add_write("A")
    state.add_nedge(outer_entry, inner_entry, dace.Memlet())
    state.add_nedge(inner_entry, tasklet, dace.Memlet())
    state.add_edge(tasklet, "out", inner_exit, "IN_A", dace.Memlet("A[jb, jc]"))
    inner_exit.add_in_connector("IN_A")
    inner_exit.add_out_connector("OUT_A")
    outer_exit.add_in_connector("IN_A")
    outer_exit.add_out_connector("OUT_A")
    state.add_edge(inner_exit, "OUT_A", outer_exit, "IN_A", dace.Memlet("A[jb, 0:N]"))
    state.add_edge(outer_exit, "OUT_A", write, None, dace.Memlet("A[0:NB, 0:N]"))
    sdfg.validate()

    assert rejoin_block_maps(sdfg) == 1
    entries = [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]
    assert len(entries) == 1
    assert entries[0].map.schedule == dtypes.ScheduleType.GPU_Device
    assert entries[0].map.params == ["jb", "jc"]
    sdfg.validate()


if __name__ == "__main__":
    test_a_block_index_read_through_an_approximated_bound_is_found()
    test_a_block_dimension_of_a_map_without_inner_maps_stays_in_the_one_kernel()
    test_a_block_dimension_of_undeclared_scale_stays_in_the_grid()
    test_a_block_dimension_in_the_kernel_matches_numpy()
    test_a_host_block_map_around_one_kernel_map_becomes_one_kernel()
    test_a_map_over_a_symbolic_block_count_stays_the_grid("NB")
    test_a_map_over_a_symbolic_block_count_stays_the_grid("nblks")
    test_a_map_over_blocks_declared_small_is_a_sequential_host_map_around_the_kernels()
    test_a_map_over_sixteen_baked_blocks_is_no_kernel()
    test_a_block_map_the_caller_made_sequential_stays_on_the_host()
    test_a_single_block_baked_before_canonicalization_leaves_kernels_over_the_columns_only()
    test_the_rendering_says_why_the_block_map_is_sequential()
    test_a_single_baked_block_matches_numpy()
    test_sixteen_baked_blocks_match_numpy()
    test_a_block_map_the_caller_made_sequential_matches_numpy()
