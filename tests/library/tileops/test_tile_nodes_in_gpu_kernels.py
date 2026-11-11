# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileIota``, ``TileScan``, ``TileMMA`` and ``TileScatter`` inside a GPU kernel, in both groups a kernel runs: a
``THREAD`` node is one thread's register tile and lowers ``pure``, a ``BLOCK`` node spreads over the thread block."""

import contextlib
from collections.abc import Callable, Iterator

import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileIota, TileMMA, TileScan, TileScatter
from dace.libraries.tileops.dispatch import ISA, TileGroup, select_tile_implementation
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.transformation.passes.offloading.taskloop import is_device_wide_libnode, is_kernel_refused_libnode

L = dace.symbol("L")
M, N, K = (dace.symbol(name) for name in "MNK")
W = 64
BM, BN, BK = 8, 8, 4
#: The lane value the fill tests give the iota, so each lane of a tile differs
IOTA = "double(3 * __l0 + 1)"


@contextlib.contextmanager
def experimental_gpu_code() -> Iterator[None]:
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        yield


@dace.program
def filled(F: dace.float64[L, W]):
    for r in dace.map[0:L]:
        t = dace.define_local([W], dace.float64, storage=dace.StorageType.Register)
        dace.tile.fill(t, 0.0)
        dace.tile.store(F[r, :], t)


@dace.program
def scanned(A: dace.float64[L, W], C: dace.float64[L, W], X: dace.float64[L, W]):
    for r in dace.map[0:L]:
        dace.tile.store(C[r, :], dace.tile.cumsum(A[r, :]))
        dace.tile.store(X[r, :], dace.tile.cummax(A[r, :]))


@dace.program
def multiplied(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
    for i, j in dace.map[0:M:BM, 0:N:BN]:
        acc = dace.define_local([BM, BN], dace.float64, storage=dace.StorageType.Register)
        dace.tile.fill(acc, 0.0)
        for k in range(0, K, BK):
            dace.tile.mma(A[i : i + BM, k : k + BK], B[k : k + BK, j : j + BN], acc)
        dace.tile.store(C[i : i + BM, j : j + BN], acc)


def scattered() -> dace.SDFG:
    """``O[r, Idx[r, l]] = A[r, l]`` for every lane ``l`` of row ``r``."""
    sdfg = dace.SDFG("tile_scatter_in_gpu_kernel")
    sdfg.add_array("A", (L, W), dace.float64)
    sdfg.add_array("Idx", (L, W), dace.int64)
    sdfg.add_array("O", (L, W), dace.float64)
    sdfg.add_array("src", (W,), dace.float64, transient=True, storage=dace.StorageType.Register)
    sdfg.add_array("idx", (W,), dace.int64, transient=True, storage=dace.StorageType.Register)
    state = sdfg.add_state(is_start_block=True)
    entry, exit_node = state.add_map("rows", {"r": "0:L"})
    src, idx = state.add_access("src"), state.add_access("idx")
    state.add_memlet_path(state.add_read("A"), entry, src, memlet=dace.Memlet(f"A[r, 0:{W}]"))
    state.add_memlet_path(state.add_read("Idx"), entry, idx, memlet=dace.Memlet(f"Idx[r, 0:{W}]"))
    node = TileScatter("scatter", widths=(W,), gather_dims=(1,))
    state.add_node(node)
    state.add_edge(src, None, node, "_src", dace.Memlet(f"src[0:{W}]"))
    state.add_edge(idx, None, node, "_idx_1", dace.Memlet(f"idx[0:{W}]"))
    state.add_memlet_path(node, exit_node, state.add_write("O"), src_conn="_dst", memlet=dace.Memlet(f"O[r, 0:{W}]"))
    return sdfg


def gpu_sdfg(sdfg: dace.SDFG, group: TileGroup, name: str) -> dace.SDFG:
    """``sdfg`` in GPU kernels, its tile nodes in ``group``; a ``THREAD`` node targets the CUDA tile backend."""
    sdfg.name = name
    sdfg.apply_gpu_transformations()
    if group is TileGroup.THREAD:
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
                node.map.gpu_block_size = [32, 1, 1]
    for node, state in tile_nodes(sdfg):
        node.group = group
        node.target_isa = ISA.CUDA
    return sdfg


def tile_nodes(sdfg: dace.SDFG) -> list[tuple[TileOp, dace.SDFGState]]:
    return [(node, state) for node, state in sdfg.all_nodes_recursive() if isinstance(node, TileOp)]


def run_on_gpu(sdfg: dace.SDFG, kind: type[TileOp], group: TileGroup, **arguments) -> str:
    """Run ``sdfg`` and give the kernel code, after checking the lowering every ``kind`` node selects."""
    expected = "pure" if group is TileGroup.THREAD else "block"
    selected = {select_tile_implementation(node, state) for node, state in tile_nodes(sdfg) if isinstance(node, kind)}
    assert selected == {expected}
    with experimental_gpu_code():
        sdfg(**arguments)
        return "\n".join(
            obj.clean_code for obj in sdfg.generate_code() if obj.target.target_name in ("cuda", "experimental_cuda")
        )


def check_group_code(kernel: str, group: TileGroup) -> None:
    """A ``THREAD`` tile stays in its thread; a ``BLOCK`` one is shared over a thread-block map."""
    if group is TileGroup.THREAD:
        assert "tile_threads_" not in kernel and "__shared__" not in kernel
    else:
        assert "tile_threads_" in kernel and "__shared__ double" in kernel


GROUPS = pytest.mark.parametrize("group", [TileGroup.THREAD, TileGroup.BLOCK], ids=lambda group: group.name)


@pytest.mark.gpu
@GROUPS
def test_iota_in_a_gpu_kernel(group: TileGroup):
    sdfg = gpu_sdfg(filled.to_sdfg(), group, f"tile_iota_gpu_{group.name.lower()}")
    for node, state in tile_nodes(sdfg):
        if isinstance(node, TileIota):
            node.expr = IOTA
    f = np.zeros((4, W))
    kernel = run_on_gpu(sdfg, TileIota, group, F=f, L=4)
    check_group_code(kernel, group)
    np.testing.assert_array_equal(f, np.broadcast_to(3.0 * np.arange(W) + 1, (4, W)))


@pytest.mark.gpu
@GROUPS
def test_scan_in_a_gpu_kernel(group: TileGroup):
    sdfg = gpu_sdfg(scanned.to_sdfg(), group, f"tile_scan_gpu_{group.name.lower()}")
    a = np.random.default_rng(0).random((4, W))
    c, x = np.zeros((4, W)), np.zeros((4, W))
    kernel = run_on_gpu(sdfg, TileScan, group, A=a, C=c, X=x, L=4)
    check_group_code(kernel, group)
    # The lanes depend on each other: a block of threads scans them together
    assert ("dace::tileops::BlockScan<double, 128>" in kernel) == (group is TileGroup.BLOCK)
    np.testing.assert_allclose(c, np.cumsum(a, axis=1), rtol=1e-12)
    np.testing.assert_array_equal(x, np.maximum.accumulate(a, axis=1))


@pytest.mark.gpu
@GROUPS
def test_mma_in_a_gpu_kernel(group: TileGroup):
    sdfg = gpu_sdfg(multiplied.to_sdfg(), group, f"tile_mma_gpu_{group.name.lower()}")
    rng = np.random.default_rng(1)
    a, b, c = rng.random((32, 16)), rng.random((16, 24)), np.zeros((32, 24))
    kernel = run_on_gpu(sdfg, TileMMA, group, A=a, B=b, C=c, M=32, N=24, K=16)
    check_group_code(kernel, group)
    np.testing.assert_allclose(c, a @ b, rtol=1e-12)


def scatter_reference(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The lanes store in order, so the last lane to a destination wins, as the pure lowering does."""
    out = np.zeros_like(a)
    for row in range(a.shape[0]):
        for lane in range(a.shape[1]):
            out[row, idx[row, lane]] = a[row, lane]
    return out


@pytest.mark.gpu
@GROUPS
def test_scatter_with_conflicts_in_a_gpu_kernel(group: TileGroup):
    sdfg = gpu_sdfg(scattered(), group, f"tile_scatter_gpu_{group.name.lower()}")
    rng = np.random.default_rng(2)
    a = rng.random((4, W))
    # Half the destinations for all the lanes: most are written by several
    idx = rng.integers(0, W // 2, size=(4, W), dtype=np.int64)
    o = np.zeros((4, W))
    kernel = run_on_gpu(sdfg, TileScatter, group, A=a, Idx=idx, O=o, L=4)
    check_group_code(kernel, group)
    np.testing.assert_array_equal(o, scatter_reference(a, idx))


def test_offloading_keeps_every_tile_node_in_its_kernel():
    """Both lowerings of a tile node emit device code, so no kernel is refused for holding one."""
    programs = (filled.to_sdfg(), scanned.to_sdfg(), multiplied.to_sdfg(), scattered())
    sdfgs = [gpu_sdfg(sdfg, TileGroup.BLOCK, f"tile_offload_{index}") for index, sdfg in enumerate(programs)]
    found = {type(node) for sdfg in sdfgs for node, state in tile_nodes(sdfg)}
    assert {TileIota, TileScan, TileMMA, TileScatter} <= found
    for node, state in (pair for sdfg in sdfgs for pair in tile_nodes(sdfg)):
        assert not is_device_wide_libnode(node, state) and not is_kernel_refused_libnode(node, state)


def all_groups(test: Callable[[TileGroup], None]) -> None:
    for group in (TileGroup.THREAD, TileGroup.BLOCK):
        test(group)


if __name__ == "__main__":
    all_groups(test_iota_in_a_gpu_kernel)
    all_groups(test_scan_in_a_gpu_kernel)
    all_groups(test_mma_in_a_gpu_kernel)
    all_groups(test_scatter_with_conflicts_in_a_gpu_kernel)
    test_offloading_keeps_every_tile_node_in_its_kernel()
