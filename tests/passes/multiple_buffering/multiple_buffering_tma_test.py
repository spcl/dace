# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Multiple buffering of shared-memory tiles that are filled by TMA (``MemoryEngineBulk``) copies.

By default the copies are asynchronous: ``InsertAsyncWaits`` pipelines them on one mbarrier per slot (parity mode).
With ``asynchronous=False`` they stay synchronous ``tma_load`` calls.

The generation-only tests need no GPU: they pin ``compiler.cuda.cuda_arch`` to 90 so the TMA expansion is accepted.
The numerical tests need a GPU with compute capability 9.0 or newer.
"""

import contextlib

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.multiple_buffering import LowerMultipleBuffering, find_staged_buffers
from tests.library.copy_node_test import _require_memory_engine

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")

#: Elements per 1-D tile; 32 elements are 256 bytes for doubles and 128 bytes for floats, so every slot of the ring
#: keeps the 128-byte alignment of the shared buffer.
TILE = 32
#: Elements of ``A`` a thread block handles.
BLOCK = 128
#: Edges of the tiles of the matrix product.
BM, BN, BK = 16, 16, 16

IMPLEMENTATIONS = ["legacy", "experimental"]
ASYNCHRONOUS = [True, False]


@contextlib.contextmanager
def hopper(implementation: str = "legacy"):
    """Accept the TMA expansion without a Hopper GPU and select the CUDA backend."""
    with dace.config.set_temporary("compiler", "cuda", "cuda_arch", value="90"):
        with dace.config.set_temporary("compiler", "cuda", "implementation", value=implementation):
            yield


def _tma_copy(
    state,
    src,
    dst,
    src_subset: str,
    dst_subset: str,
    name: str = "tma_copy",
    implementation="MemoryEngineBulk",
    asynchronous: bool = True,
):
    """Wire ``src -> CopyLibraryNode -> dst``; ``implementation=None`` leaves the choice to the passes."""
    libnode = CopyLibraryNode(name=name, asynchronous=asynchronous)
    libnode.implementation = implementation
    state.add_edge(src, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet(f"{src.data}[{src_subset}]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst, None, dace.Memlet(f"{dst.data}[{dst_subset}]"))
    return libnode


def tma_tiled_add(
    name: str,
    depth: int,
    dtype=dace.float64,
    tile: int = TILE,
    block: int = BLOCK,
    stride: int = 0,
    implementation="MemoryEngineBulk",
    asynchronous: bool = True,
    repeat: int = 0,
) -> dace.SDFG:
    """``B = 2 * A + 1`` where every thread block streams ``block`` elements in tiles of ``tile``, each staged in
    shared memory by one TMA copy. The loop over the tiles lives in a NestedSDFG of the kernel and has
    ``pipeline_depth=depth``; ``stride`` (default ``tile``) is the distance between the tiles it visits.

    A whole array is passed into the NestedSDFG and the tile is selected by a symbol, so the tensor map does not
    depend on the kernel's map parameter and can be encoded on the host.

    With ``repeat > 0``, the tile loop runs ``repeat`` times in a sequential loop of the kernel and accumulates into
    ``B`` (``B += 2 * A + 1`` each time), so the pipeline restarts with barriers that were used before.
    """
    stride = stride or tile
    sdfg = dace.SDFG(name)
    for arr in ("A", "B"):
        sdfg.add_array(arr, [N], dtype, dace.StorageType.GPU_Global)
    state = sdfg.add_state()

    inner = dace.SDFG(name + "_kernel")
    for arr in ("A", "B"):
        inner.add_array(arr, [N], dtype, dace.StorageType.GPU_Global)
    inner.add_array("buf", [tile], dtype, dace.StorageType.GPU_Shared, transient=True)
    loop = LoopRegion("tiles", f"t < {block}", "t", "t = 0", f"t = t + {stride}")
    loop.pipeline_depth = depth
    if repeat:
        outer = LoopRegion("repeat", f"r < {repeat}", "r", "r = 0", "r = r + 1")
        inner.add_node(outer, is_start_block=True)
        outer.add_node(loop, is_start_block=True)
    else:
        inner.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    a, buf, b = body.add_access("A"), body.add_access("buf"), body.add_access("B")
    _tma_copy(
        body, a, buf, f"blk + t:blk + t + {tile}", f"0:{tile}", implementation=implementation, asynchronous=asynchronous
    )
    lanes_entry, lanes_exit = body.add_map("lanes", {"j": f"0:{tile}"}, schedule=dace.ScheduleType.GPU_ThreadBlock)
    tasklet = body.add_tasklet("axpb", {"x"}, {"y"}, "y = 2 * x + 1")
    body.add_memlet_path(buf, lanes_entry, tasklet, dst_conn="x", memlet=dace.Memlet("buf[j]"))
    out = dace.Memlet("B[blk + t + j]", wcr="lambda p, q: p + q") if repeat else dace.Memlet("B[blk + t + j]")
    body.add_memlet_path(tasklet, lanes_exit, b, src_conn="y", memlet=out)

    entry, exit_ = state.add_map("device", {"blk": f"0:N:{block}"}, schedule=dace.ScheduleType.GPU_Device)
    nested = state.add_nested_sdfg(inner, {"A"}, {"B"}, symbol_mapping={"blk": "blk", "N": "N"})
    state.add_memlet_path(state.add_access("A"), entry, nested, dst_conn="A", memlet=dace.Memlet("A[0:N]"))
    state.add_memlet_path(nested, exit_, state.add_access("B"), src_conn="B", memlet=dace.Memlet("B[0:N]"))
    return sdfg


def tma_tiled_gemm(
    name: str, depth: int, dtype=dace.float64, asynchronous: bool = True, asynchronous_b: bool | None = None
) -> dace.SDFG:
    """``C += A @ B`` with a thread block per ``BM x BN`` tile of ``C``. A loop over K tiles with
    ``pipeline_depth=depth`` stages one tile of ``A`` and one of ``B`` per iteration, both with 2-D TMA copies.
    ``asynchronous_b`` (default: ``asynchronous``) sets the copy of the ``B`` tile apart."""
    asynchronous_b = asynchronous if asynchronous_b is None else asynchronous_b
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [M, K], dtype, dace.StorageType.GPU_Global)
    sdfg.add_array("B", [K, N], dtype, dace.StorageType.GPU_Global)
    sdfg.add_array("C", [M, N], dtype, dace.StorageType.GPU_Global)
    state = sdfg.add_state()

    inner = dace.SDFG(name + "_kernel")
    inner.add_array("A", [M, K], dtype, dace.StorageType.GPU_Global)
    inner.add_array("B", [K, N], dtype, dace.StorageType.GPU_Global)
    inner.add_array("C", [M, N], dtype, dace.StorageType.GPU_Global)
    inner.add_array("a_tile", [BM, BK], dtype, dace.StorageType.GPU_Shared, transient=True)
    inner.add_array("b_tile", [BK, BN], dtype, dace.StorageType.GPU_Shared, transient=True)
    loop = LoopRegion("ktiles", "kk < K", "kk", "kk = 0", f"kk = kk + {BK}")
    loop.pipeline_depth = depth
    inner.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    a, b, c = body.add_access("A"), body.add_access("B"), body.add_access("C")
    a_tile, b_tile = body.add_access("a_tile"), body.add_access("b_tile")
    _tma_copy(
        body, a, a_tile, f"bi:bi + {BM}, kk:kk + {BK}", f"0:{BM}, 0:{BK}", name="tma_a", asynchronous=asynchronous
    )
    _tma_copy(
        body, b, b_tile, f"kk:kk + {BK}, bj:bj + {BN}", f"0:{BK}, 0:{BN}", name="tma_b", asynchronous=asynchronous_b
    )
    tile_entry, tile_exit = body.add_map(
        "tile", {"i": f"0:{BM}", "j": f"0:{BN}"}, schedule=dace.ScheduleType.GPU_ThreadBlock
    )
    red_entry, red_exit = body.add_map("reduce", {"l": f"0:{BK}"}, schedule=dace.ScheduleType.Sequential)
    tasklet = body.add_tasklet("fma", {"x", "y"}, {"z"}, "z = x * y")
    body.add_memlet_path(a_tile, tile_entry, red_entry, tasklet, dst_conn="x", memlet=dace.Memlet("a_tile[i, l]"))
    body.add_memlet_path(b_tile, tile_entry, red_entry, tasklet, dst_conn="y", memlet=dace.Memlet("b_tile[l, j]"))
    body.add_memlet_path(
        tasklet,
        red_exit,
        tile_exit,
        c,
        src_conn="z",
        memlet=dace.Memlet("C[bi + i, bj + j]", wcr="lambda p, q: p + q"),
    )

    entry, exit_ = state.add_map(
        "device", {"bi": f"0:M:{BM}", "bj": f"0:N:{BN}"}, schedule=dace.ScheduleType.GPU_Device
    )
    nested = state.add_nested_sdfg(
        inner, {"A", "B"}, {"C"}, symbol_mapping={"bi": "bi", "bj": "bj", "M": "M", "N": "N", "K": "K"}
    )
    state.add_memlet_path(state.add_access("A"), entry, nested, dst_conn="A", memlet=dace.Memlet("A[0:M, 0:K]"))
    state.add_memlet_path(state.add_access("B"), entry, nested, dst_conn="B", memlet=dace.Memlet("B[0:K, 0:N]"))
    state.add_memlet_path(nested, exit_, state.add_access("C"), src_conn="C", memlet=dace.Memlet("C[0:M, 0:N]"))
    return sdfg


def pipelined_loops(sdfg: dace.SDFG) -> list:
    """The ``(loop, sdfg)`` pairs of every loop of ``sdfg`` with ``pipeline_depth >= 2``."""
    return [
        (region, sd)
        for sd in sdfg.all_sdfgs_recursive()
        for region in sd.all_control_flow_regions()
        if isinstance(region, LoopRegion) and region.pipeline_depth >= 2
    ]


def kernel_code(sdfg: dace.SDFG) -> dict:
    """The generated code by object title (``CUDA`` for the kernels, ``Frame`` for the host)."""
    return {obj.title: obj.clean_code for obj in sdfg.generate_code()}


def load_call(rank: int, asynchronous: bool) -> str:
    """The device call of a pipelined TMA load: issued by the leader only when asynchronous (parity mode)."""
    return f"dace::cuda::tma_load_async_leader<{rank}>" if asynchronous else f"dace::cuda::tma_load<{rank}>"


def check_async_pipeline(cuda: str, depth: int, label: str, per_stage: int, rank: int):
    """The kernel runs the parity-mode schedule of ``InsertAsyncWaits``: barrier init, prologue issues, a main loop
    that waits for every thread, issues, waits for its tile and consumes, an epilogue that waits and consumes, and the
    barrier teardown, with no other block barrier."""
    issue, wait = f"dace::cuda::tma_load_async_leader<{rank}>", "dace::cuda::mbarrier_wait_parity"
    assert f"__shared__ __align__(128) uint64_t {label}_full_bar[{depth}];" in cuda
    assert cuda.count(f"dace::cuda::mbarrier_init_array(_tma_bar_out, {depth}, {per_stage});") == 1
    assert cuda.count(f"dace::cuda::mbarrier_inval_array(_tma_bar_out, {depth});") == 1
    assert cuda.count(issue) == 2 * per_stage  # prologue and main loop
    assert cuda.count(wait) == 2  # main loop and epilogue
    assert "dace::cuda::tma_load<" not in cuda and "tma_barrier_init" not in cuda
    # One block barrier in the main loop (slot reuse) and one before the teardown
    assert cuda.count("__syncthreads()") == 2
    # Order: init, prologue issue(s), main loop barrier, issue(s), wait, epilogue wait, teardown barrier, inval
    positions = [
        cuda.index("mbarrier_init_array"),
        cuda.index(issue),
        cuda.index("__syncthreads()"),
        cuda.rindex(issue),
        cuda.index(wait),
        cuda.rindex(wait),
        cuda.rindex("__syncthreads()"),
        cuda.index("mbarrier_inval_array"),
    ]
    assert positions == sorted(positions), positions


def count_copy_nodes(sdfg: dace.SDFG) -> int:
    return sum(
        isinstance(node, CopyLibraryNode)
        for sd in sdfg.all_sdfgs_recursive()
        for state in sd.states()
        for node in state.nodes()
    )


# --------------------------------------------------------------------------------------------------------------------
# Generation only


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_tma_copy_is_a_staged_copy(depth: int):
    """The analysis accepts a ``MemoryEngineBulk`` copy node as the single writer of the staged buffer."""
    sdfg = tma_tiled_add(f"mb_tma_analysis_{depth}", depth)
    ((loop, sd),) = pipelined_loops(sdfg)
    staged, rejected = find_staged_buffers(loop, sd, None)
    assert [s.name for s in staged] == ["buf"]
    (copy,) = staged[0].copies
    assert isinstance(copy.copy_node, CopyLibraryNode)
    assert copy.copy_node.implementation == "MemoryEngineBulk"
    assert "buf" not in rejected


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_lowering_structure(depth: int):
    sdfg = tma_tiled_add(f"mb_tma_lower_{depth}", depth)
    ((loop, sd),) = pipelined_loops(sdfg)
    (lowered,) = LowerMultipleBuffering().apply_pass(sdfg, {})
    assert (lowered.label, lowered.depth) == ("tiles", depth)
    # The buffer is shared by the block, so the main loop waits for every thread before it refills a slot
    assert lowered.war_barrier is not None
    assert lowered.main.start_block.start_block is lowered.war_barrier
    sdfg.validate()
    assert sd.arrays["buf"].shape == (depth, TILE)
    labels = {r.label for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)}
    assert {"tiles", "tiles_prologue", "tiles_epilogue"} <= labels
    # One copy node per producer: the prologue's and the main loop's
    assert count_copy_nodes(sdfg) == 2
    assert not pipelined_loops(sdfg) or all(lp.pipeline_depth < 2 for lp, _ in pipelined_loops(sdfg))


@pytest.mark.parametrize("asynchronous", ASYNCHRONOUS)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_code_structure(depth: int, implementation: str, asynchronous: bool):
    """The ring of tiles is shared static memory, and every producer loads through its own tensor map."""
    sdfg = tma_tiled_add(f"mb_tma_code_{depth}_{implementation}_{asynchronous}", depth, asynchronous=asynchronous)
    with hopper(implementation):
        code = kernel_code(sdfg)
    cuda = code["CUDA"]
    assert f"__shared__ __align__(128) double buf[{depth * TILE}];" in cuda
    assert cuda.count("const __grid_constant__ CUtensorMap") >= 2  # prologue and main loop
    assert cuda.count(load_call(1, asynchronous)) == 2
    if asynchronous:
        check_async_pipeline(cuda, depth, "tiles", per_stage=1, rank=1)
    else:
        assert "mbarrier_wait_parity" not in cuda and "full_bar" not in cuda
    assert f"% {depth})" in cuda
    assert "dace::cuda::make_tiled_tensor_map(__state, CU_TENSOR_MAP_DATA_TYPE_FLOAT64" in code["Frame"]
    assert code["Frame"].count("make_tiled_tensor_map") == 2
    # Consuming reads the slot of the tile, not slot 0
    assert f"buf[(j + ({TILE} * ((t / {TILE}) % {depth})))]" in cuda


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_code_structure_float32(implementation: str):
    sdfg = tma_tiled_add(f"mb_tma_code_f32_{implementation}", 3, dtype=dace.float32)
    with hopper(implementation):
        code = kernel_code(sdfg)
    assert f"__shared__ __align__(128) float buf[{3 * TILE}];" in code["CUDA"]
    assert "CU_TENSOR_MAP_DATA_TYPE_FLOAT32" in code["Frame"]


@pytest.mark.parametrize("asynchronous", ASYNCHRONOUS)
@pytest.mark.parametrize("block", [TILE, 2 * TILE, 3 * TILE, 5 * TILE])
@pytest.mark.parametrize("depth", [2, 4])
def test_code_structure_trip_counts(depth: int, block: int, asynchronous: bool):
    """Fewer tiles than the ring is deep (so the prologue is partial), one tile, and many tiles all generate."""
    sdfg = tma_tiled_add(f"mb_tma_trip_{depth}_{block}_{asynchronous}", depth, block=block, asynchronous=asynchronous)
    with hopper():
        cuda = kernel_code(sdfg)["CUDA"]
    assert cuda.count(load_call(1, asynchronous)) == 2


def test_stride_larger_than_tile_code_structure():
    sdfg = tma_tiled_add("mb_tma_stride", 3, stride=2 * TILE)
    with hopper():
        cuda = kernel_code(sdfg)["CUDA"]
    assert f"__shared__ __align__(128) double buf[{3 * TILE}];" in cuda


@pytest.mark.parametrize("asynchronous", ASYNCHRONOUS)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("depth", [2, 3])
def test_gemm_code_structure(depth: int, implementation: str, asynchronous: bool):
    """Two staged buffers, each filled by a 2-D TMA copy (a prologue and a main loop copy each). Asynchronously, both
    copies of a stage complete on the same barrier, which therefore counts two arrivals."""
    sdfg = tma_tiled_gemm(f"mb_tma_gemm_code_{depth}_{implementation}_{asynchronous}", depth, asynchronous=asynchronous)
    with hopper(implementation):
        code = kernel_code(sdfg)
    cuda = code["CUDA"]
    assert f"__shared__ __align__(128) double a_tile[{depth * BM * BK}];" in cuda
    assert f"__shared__ __align__(128) double b_tile[{depth * BK * BN}];" in cuda
    assert cuda.count(load_call(2, asynchronous)) == 4
    if asynchronous:
        check_async_pipeline(cuda, depth, "ktiles", per_stage=2, rank=2)
    assert code["Frame"].count("make_tiled_tensor_map") == 4


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_unpipelined_copy_waits_by_token(implementation: str):
    """Without pipelining, the copy issues asynchronously and its wait follows in the same iteration (token mode)."""
    sdfg = tma_tiled_add(f"mb_tma_token_{implementation}", 0)
    with hopper(implementation):
        cuda = kernel_code(sdfg)["CUDA"]
    assert cuda.count("dace::cuda::tma_load_async<1>") == 1
    assert cuda.count("dace::cuda::mbarrier_wait_token") == 1
    assert cuda.index("tma_load_async<1>") < cuda.index("mbarrier_wait_token") < cuda.index("2 * x")
    assert "mbarrier_wait_parity" not in cuda and "full_bar" not in cuda


def test_mixed_stage_falls_back_to_token_mode():
    """A stage with a synchronous copy cannot be pipelined on the slot barriers: the asynchronous copy of the stage
    waits by token instead, and the synchronous one stays a ``tma_load``."""
    sdfg = tma_tiled_gemm("mb_tma_mixed", 2, asynchronous_b=False)
    with hopper(), pytest.warns(UserWarning, match="not pipelined asynchronously"):
        cuda = kernel_code(sdfg)["CUDA"]
    assert cuda.count("dace::cuda::tma_load_async<2>") == 2
    assert cuda.count("dace::cuda::mbarrier_wait_token") == 2
    assert cuda.count("dace::cuda::tma_load<2>") == 2
    assert "mbarrier_wait_parity" not in cuda


def test_unaligned_slot_falls_back_to_token_mode():
    """A slot of 4 floats (16 bytes) cannot start every slot on a 128-byte boundary, so the loop is not pipelined on
    slot barriers."""
    from dace.transformation import pass_pipeline as ppl
    from dace.transformation.passes.insert_async_waits import InsertAsyncWaits

    sdfg = tma_tiled_add("mb_tma_unaligned_slot", 2, dtype=dace.float32, tile=4, block=16)
    with hopper(), pytest.warns(UserWarning, match="not a multiple of 128"):
        results = ppl.Pipeline([LowerMultipleBuffering(), InsertAsyncWaits()]).apply_pass(sdfg, {})
    assert results["InsertAsyncWaits"] == {"token": 2}


def test_gemm_buffers_are_both_staged():
    sdfg = tma_tiled_gemm("mb_tma_gemm_analysis", 2)
    ((loop, sd),) = pipelined_loops(sdfg)
    staged, _ = find_staged_buffers(loop, sd, None)
    assert sorted(s.name for s in staged) == ["a_tile", "b_tile"]


def test_explicit_implementation_is_kept():
    """``InsertMemoryEngineBulkCopies`` does not touch a node whose implementation is already ``MemoryEngineBulk``."""
    from dace.transformation.passes.insert_memory_engine_bulk_copies import InsertMemoryEngineBulkCopies

    sdfg = tma_tiled_add("mb_tma_idempotent", 3)
    LowerMultipleBuffering().apply_pass(sdfg, {})
    with hopper():
        assert not InsertMemoryEngineBulkCopies().apply_pass(sdfg, {})
    impls = {
        node.implementation
        for sd in sdfg.all_sdfgs_recursive()
        for state in sd.states()
        for node in state.nodes()
        if isinstance(node, CopyLibraryNode)
    }
    assert impls == {"MemoryEngineBulk"}


def test_unaligned_tile_is_rejected():
    """A tile of 3 floats (12 bytes) violates the 16-byte rule; the error comes from the TMA expansion, and the
    multiple-buffering lowering does not hide it."""
    sdfg = tma_tiled_add("mb_tma_unaligned", 3, dtype=dace.float32, tile=3, block=12)
    with hopper(), pytest.raises(ValueError, match="multiple of 16 bytes"):
        sdfg.generate_code()


# --------------------------------------------------------------------------------------------------------------------
# On hardware


def _check_add(sdfg: dace.SDFG, cp, dtype, n: int):
    A = (cp.arange(n) % 97).astype(dtype)
    B = cp.zeros(n, dtype=dtype)
    sdfg(A=A, B=B, N=n)
    cp.testing.assert_array_equal(B, 2 * A + 1)


@pytest.mark.gpu
@pytest.mark.parametrize("asynchronous", ASYNCHRONOUS)
@pytest.mark.parametrize("blocks", [1, 2, 7])
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_tiled_add(depth: int, blocks: int, asynchronous: bool):
    cp = _require_memory_engine()
    # With BLOCK/TILE = 4 tiles per block, depth 4 and 5 hit the partial-prologue and single-tile cases below
    sdfg = tma_tiled_add(f"mb_tma_add_{depth}_{blocks}_{asynchronous}", depth, asynchronous=asynchronous)
    _check_add(sdfg, cp, np.float64, blocks * BLOCK)


@pytest.mark.gpu
@pytest.mark.parametrize("asynchronous", ASYNCHRONOUS)
@pytest.mark.parametrize("tiles", [1, 2, 3, 4, 9])
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_tiled_add_trip_counts(depth: int, tiles: int, asynchronous: bool):
    """From a single tile (shorter than the prologue) up to many times the depth of the ring."""
    cp = _require_memory_engine()
    sdfg = tma_tiled_add(
        f"mb_tma_trips_{depth}_{tiles}_{asynchronous}", depth, block=tiles * TILE, asynchronous=asynchronous
    )
    _check_add(sdfg, cp, np.float64, 3 * tiles * TILE)


@pytest.mark.gpu
@pytest.mark.parametrize("tiles", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_pipeline_restarts_in_kernel(depth: int, tiles: int):
    """The pipelined loop runs three times in a kernel: the slot barriers are invalidated and initialized again, so
    the phase parities of every run start over (a trip count that is not a multiple of the depth leaves the
    barriers in different phases at the end of a run)."""
    cp = _require_memory_engine()
    repeat = 3
    sdfg = tma_tiled_add(f"mb_tma_restart_{depth}_{tiles}", depth, block=tiles * TILE, repeat=repeat)
    n = 2 * tiles * TILE
    A = (cp.arange(n) % 97).astype(np.float64)
    B = cp.zeros(n)
    sdfg(A=A, B=B, N=n)
    cp.testing.assert_array_equal(B, repeat * (2 * A + 1))


@pytest.mark.gpu
@pytest.mark.parametrize("repeat", [1, 3])
def test_token_mode_in_loop(repeat: int):
    """Without pipelining, every iteration waits on the arrival token of its own copy, reusing one barrier."""
    cp = _require_memory_engine()
    sdfg = tma_tiled_add(f"mb_tma_token_loop_{repeat}", 0, repeat=repeat)
    n = 3 * BLOCK
    A = (cp.arange(n) % 97).astype(np.float64)
    B = cp.zeros(n)
    sdfg(A=A, B=B, N=n)
    cp.testing.assert_array_equal(B, repeat * (2 * A + 1))


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [2, 3])
def test_tiled_add_float32(depth: int):
    cp = _require_memory_engine()
    sdfg = tma_tiled_add(f"mb_tma_add_f32_{depth}", depth, dtype=dace.float32)
    _check_add(sdfg, cp, np.float32, 4 * BLOCK)


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [2, 3])
def test_tiled_add_stride_larger_than_tile(depth: int):
    """Only every other tile is visited, so half of the elements keep their initial value."""
    cp = _require_memory_engine()
    sdfg = tma_tiled_add(f"mb_tma_add_stride_{depth}", depth, stride=2 * TILE)
    n = 4 * BLOCK
    A = (cp.arange(n) % 97).astype(np.float64)
    B = cp.zeros(n)
    sdfg(A=A, B=B, N=n)
    expected = cp.zeros(n)
    for start in range(0, n, BLOCK):
        for t in range(0, BLOCK, 2 * TILE):
            expected[start + t : start + t + TILE] = 2 * A[start + t : start + t + TILE] + 1
    cp.testing.assert_array_equal(B, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_matches_single_buffering(depth: int):
    """A pipelined kernel computes exactly what the same kernel computes with ``pipeline_depth=0``."""
    cp = _require_memory_engine()
    n = 5 * BLOCK
    A = cp.random.rand(n)
    single, multi = (cp.zeros(n), cp.zeros(n))
    tma_tiled_add(f"mb_tma_ref_{depth}", 0)(A=A, B=single, N=n)
    tma_tiled_add(f"mb_tma_multi_{depth}", depth)(A=A, B=multi, N=n)
    cp.testing.assert_array_equal(single, multi)


@pytest.mark.gpu
@pytest.mark.parametrize("asynchronous", ASYNCHRONOUS)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_tiled_add_both_backends(implementation: str, asynchronous: bool):
    cp = _require_memory_engine()
    sdfg = tma_tiled_add(f"mb_tma_backend_{implementation}_{asynchronous}", 3, asynchronous=asynchronous)
    with dace.config.set_temporary("compiler", "cuda", "implementation", value=implementation):
        _check_add(sdfg, cp, np.float64, 4 * BLOCK)


@pytest.mark.gpu
@pytest.mark.parametrize("asynchronous", ASYNCHRONOUS)
@pytest.mark.parametrize("k_tiles", [1, 2, 3, 8])
@pytest.mark.parametrize("depth", [0, 2, 3])
def test_tiled_gemm(depth: int, k_tiles: int, asynchronous: bool):
    cp = _require_memory_engine()
    # Powers of two keep every product and sum exact, so the comparison needs no tolerance
    m, n, k = 2 * BM, 3 * BN, k_tiles * BK
    A = cp.random.randint(0, 4, (m, k)).astype(np.float64)
    B = cp.random.randint(0, 4, (k, n)).astype(np.float64)
    C = cp.zeros((m, n))
    sdfg = tma_tiled_gemm(f"mb_tma_gemm_{depth}_{k_tiles}_{asynchronous}", depth, asynchronous=asynchronous)
    sdfg(A=A, B=B, C=C, M=m, N=n, K=k)
    cp.testing.assert_array_equal(C, A @ B)


@pytest.mark.gpu
def test_tiled_gemm_mixed_stage():
    """One asynchronous and one synchronous copy per stage (token-mode fallback) still compute correctly."""
    cp = _require_memory_engine()
    m, n, k = 2 * BM, 2 * BN, 5 * BK
    A = cp.random.randint(0, 4, (m, k)).astype(np.float64)
    B = cp.random.randint(0, 4, (k, n)).astype(np.float64)
    C = cp.zeros((m, n))
    with pytest.warns(UserWarning, match="not pipelined asynchronously"):
        tma_tiled_gemm("mb_tma_gemm_mixed", 2, asynchronous_b=False)(A=A, B=B, C=C, M=m, N=n, K=k)
    cp.testing.assert_array_equal(C, A @ B)


@pytest.mark.gpu
@pytest.mark.parametrize("asynchronous", ASYNCHRONOUS)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_tiled_gemm_both_backends(implementation: str, asynchronous: bool):
    cp = _require_memory_engine()
    m, n, k = 2 * BM, 2 * BN, 4 * BK
    A = cp.random.randint(0, 4, (m, k)).astype(np.float64)
    B = cp.random.randint(0, 4, (k, n)).astype(np.float64)
    C = cp.zeros((m, n))
    sdfg = tma_tiled_gemm(f"mb_tma_gemm_backend_{implementation}_{asynchronous}", 3, asynchronous=asynchronous)
    with dace.config.set_temporary("compiler", "cuda", "implementation", value=implementation):
        sdfg(A=A, B=B, C=C, M=m, N=n, K=k)
    cp.testing.assert_array_equal(C, A @ B)


if __name__ == "__main__":
    for depth in (2, 3, 4):
        test_tma_copy_is_a_staged_copy(depth)
        test_lowering_structure(depth)
        for implementation in IMPLEMENTATIONS:
            for asynchronous in ASYNCHRONOUS:
                test_code_structure(depth, implementation, asynchronous)
    for implementation in IMPLEMENTATIONS:
        test_code_structure_float32(implementation)
        test_unpipelined_copy_waits_by_token(implementation)
    for depth in (2, 4):
        for block in (TILE, 2 * TILE, 3 * TILE, 5 * TILE):
            for asynchronous in ASYNCHRONOUS:
                test_code_structure_trip_counts(depth, block, asynchronous)
    test_stride_larger_than_tile_code_structure()
    for depth in (2, 3):
        for implementation in IMPLEMENTATIONS:
            for asynchronous in ASYNCHRONOUS:
                test_gemm_code_structure(depth, implementation, asynchronous)
    test_mixed_stage_falls_back_to_token_mode()
    test_unaligned_slot_falls_back_to_token_mode()
    test_gemm_buffers_are_both_staged()
    test_explicit_implementation_is_kept()
    test_unaligned_tile_is_rejected()
    # Hardware tests
    for depth in (2, 3, 4):
        for asynchronous in ASYNCHRONOUS:
            for blocks in (1, 2, 7):
                test_tiled_add(depth, blocks, asynchronous)
            for tiles in (1, 2, 3, 4, 9):
                test_tiled_add_trip_counts(depth, tiles, asynchronous)
        for tiles in (1, 2, 3, 4, 7):
            test_pipeline_restarts_in_kernel(depth, tiles)
        test_matches_single_buffering(depth)
    for repeat in (1, 3):
        test_token_mode_in_loop(repeat)
    for depth in (2, 3):
        test_tiled_add_float32(depth)
        test_tiled_add_stride_larger_than_tile(depth)
    for depth in (0, 2, 3):
        for k_tiles in (1, 2, 3, 8):
            for asynchronous in ASYNCHRONOUS:
                test_tiled_gemm(depth, k_tiles, asynchronous)
    test_tiled_gemm_mixed_stage()
    for implementation in IMPLEMENTATIONS:
        for asynchronous in ASYNCHRONOUS:
            test_tiled_add_both_backends(implementation, asynchronous)
            test_tiled_gemm_both_backends(implementation, asynchronous)
