# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lowering of ``LoopRegion.pipeline_depth`` into a prologue, a main and an epilogue loop (CPU)."""

import copy

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation.passes.multiple_buffering import LowerMultipleBuffering

TILE = 4


def tiled_add(
    name: str,
    depth: int,
    n: int | str = 16,
    start: int = 0,
    stride: int | str = TILE,
    copy_node: bool = False,
    nested: bool = False,
    two_states: bool = False,
) -> dace.SDFG:
    """``B[t:t+TILE] = 2 * A[t:t+TILE] + 1`` for every tile, through the staged buffer ``buf``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [n], dace.float64)
    sdfg.add_array("B", [n], dace.float64)
    sdfg.add_transient("buf", [TILE], dace.float64)
    loop = LoopRegion("tiles", f"t < {n} - {TILE - 1}", "t", f"t = {start}", f"t = t + {stride}", pipeline_depth=depth)
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state("load", is_start_block=True)
    a = state.add_access("A")
    b = state.add_access("buf")
    if copy_node:
        node = CopyLibraryNode(name="load_tile")
        state.add_edge(a, None, node, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet(f"A[t:t+{TILE}]"))
        state.add_edge(node, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, b, None, dace.Memlet(f"buf[0:{TILE}]"))
    else:
        state.add_nedge(a, b, dace.Memlet(f"A[t:t+{TILE}]", other_subset=f"0:{TILE}"))
    if two_states:
        state = loop.add_state_after(state, "compute")
        b = state.add_read("buf")
    if nested:
        inner = dace.SDFG("compute")
        inner.add_array("x", [TILE], dace.float64)
        inner.add_array("y", [n], dace.float64)
        inner.add_symbol("t", dace.int64)
        inner_state = inner.add_state()
        inner_state.add_mapped_tasklet(
            "c",
            {"j": f"0:{TILE}"},
            {"v": dace.Memlet("x[j]")},
            "w = 2 * v + 1",
            {"w": dace.Memlet("y[t + j]")},
            external_edges=True,
        )
        nsdfg = state.add_nested_sdfg(inner, {"x"}, {"y"}, symbol_mapping={"t": "t"})
        state.add_edge(b, None, nsdfg, "x", dace.Memlet(f"buf[0:{TILE}]"))
        state.add_edge(nsdfg, "y", state.add_write("B"), None, dace.Memlet(f"B[0:{n}]"))
    else:
        state.add_mapped_tasklet(
            "compute",
            {"j": f"0:{TILE}"},
            {"x": dace.Memlet("buf[j]")},
            "y = 2 * x + 1",
            {"y": dace.Memlet("B[t + j]")},
            input_nodes={"buf": b},
            external_edges=True,
        )
    return sdfg


def tiled_gemm(name: str, depth: int, m: int = 5, n: int = 6, k: int = 20, bk: int = 4) -> dace.SDFG:
    """``C = A @ B``, looping over K in tiles of ``bk`` staged in ``a_tile`` and ``b_tile``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [m, k], dace.float64)
    sdfg.add_array("B", [k, n], dace.float64)
    sdfg.add_array("C", [m, n], dace.float64)
    sdfg.add_transient("a_tile", [m, bk], dace.float64)
    sdfg.add_transient("b_tile", [bk, n], dace.float64)
    loop = LoopRegion("ktiles", f"kk < {k}", "kk", "kk = 0", f"kk = kk + {bk}", pipeline_depth=depth)
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state("body", is_start_block=True)
    a_tile = state.add_access("a_tile")
    b_tile = state.add_access("b_tile")
    state.add_nedge(state.add_read("A"), a_tile, dace.Memlet(f"A[0:{m}, kk:kk+{bk}]", other_subset=f"0:{m}, 0:{bk}"))
    state.add_nedge(state.add_read("B"), b_tile, dace.Memlet(f"B[kk:kk+{bk}, 0:{n}]", other_subset=f"0:{bk}, 0:{n}"))
    state.add_mapped_tasklet(
        "mac",
        {"i": f"0:{m}", "j": f"0:{n}", "l": f"0:{bk}"},
        {"x": dace.Memlet("a_tile[i, l]"), "y": dace.Memlet("b_tile[l, j]")},
        "z = x * y",
        {"z": dace.Memlet("C[i, j]", wcr="lambda a, b: a + b")},
        input_nodes={"a_tile": a_tile, "b_tile": b_tile},
        external_edges=True,
    )
    return sdfg


def blocked_gemm(
    name: str, depth: int, m: int, n: int, k: int, bm: int, bn: int, bk: int, n_depth: int = 0
) -> dace.SDFG:
    """``C = A @ B`` blocked in M, N and K: loops over M and N tiles around a pipelined loop over K tiles.

    If ``n_depth`` is set, the loop over N tiles is pipelined too: it stages a tile of an extra input ``D`` in ``d_tile``
    and adds it after the K loop, computing ``C = A @ B + D``.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [m, k], dace.float64)
    sdfg.add_array("B", [k, n], dace.float64)
    sdfg.add_array("C", [m, n], dace.float64)
    sdfg.add_transient("a_tile", [bm, bk], dace.float64)
    sdfg.add_transient("b_tile", [bk, bn], dace.float64)
    if n_depth:
        sdfg.add_array("D", [m, n], dace.float64)
        sdfg.add_transient("d_tile", [bm, bn], dace.float64)
    mloop = LoopRegion("mtiles", f"mm < {m}", "mm", "mm = 0", f"mm = mm + {bm}")
    sdfg.add_node(mloop, is_start_block=True)
    nloop = LoopRegion("ntiles", f"nn < {n}", "nn", "nn = 0", f"nn = nn + {bn}", pipeline_depth=n_depth)
    mloop.add_node(nloop, is_start_block=True)
    kloop = LoopRegion("ktiles", f"kk < {k}", "kk", "kk = 0", f"kk = kk + {bk}", pipeline_depth=depth)
    if n_depth:
        load = nloop.add_state("load_d", is_start_block=True)
        load.add_nedge(
            load.add_read("D"),
            load.add_access("d_tile"),
            dace.Memlet(f"D[mm:mm+{bm}, nn:nn+{bn}]", other_subset=f"0:{bm}, 0:{bn}"),
        )
        nloop.add_node(kloop)
        nloop.add_edge(load, kloop, dace.InterstateEdge())
    else:
        nloop.add_node(kloop, is_start_block=True)
    state = kloop.add_state("body", is_start_block=True)
    a_tile = state.add_access("a_tile")
    b_tile = state.add_access("b_tile")
    state.add_nedge(
        state.add_read("A"), a_tile, dace.Memlet(f"A[mm:mm+{bm}, kk:kk+{bk}]", other_subset=f"0:{bm}, 0:{bk}")
    )
    state.add_nedge(
        state.add_read("B"), b_tile, dace.Memlet(f"B[kk:kk+{bk}, nn:nn+{bn}]", other_subset=f"0:{bk}, 0:{bn}")
    )
    state.add_mapped_tasklet(
        "mac",
        {"i": f"0:{bm}", "j": f"0:{bn}", "l": f"0:{bk}"},
        {"x": dace.Memlet("a_tile[i, l]"), "y": dace.Memlet("b_tile[l, j]")},
        "z = x * y",
        {"z": dace.Memlet("C[mm + i, nn + j]", wcr="lambda a, b: a + b")},
        input_nodes={"a_tile": a_tile, "b_tile": b_tile},
        external_edges=True,
    )
    if n_depth:
        add = nloop.add_state_after(kloop, "add_d")
        add.add_mapped_tasklet(
            "add_d",
            {"i": f"0:{bm}", "j": f"0:{bn}"},
            {"d": dace.Memlet("d_tile[i, j]")},
            "z = d",
            {"z": dace.Memlet("C[mm + i, nn + j]", wcr="lambda a, b: a + b")},
            external_edges=True,
        )
    return sdfg


def loops(sdfg: dace.SDFG) -> list[LoopRegion]:
    return [region for region in sdfg.all_control_flow_regions() if isinstance(region, LoopRegion)]


def expected_add(A: np.ndarray, start: int, stride: int) -> np.ndarray:
    B = np.zeros_like(A)
    for t in range(start, len(A) - (TILE - 1), stride):
        B[t : t + TILE] = 2 * A[t : t + TILE] + 1
    return B


@pytest.mark.parametrize("depth", [2, 3, 4])
@pytest.mark.parametrize("tiles", [1, 2, 3, 4, 7])
def test_tiled_add(depth: int, tiles: int):
    n = tiles * TILE
    sdfg = tiled_add(f"mb_add_{depth}_{tiles}", depth, n)
    assert LowerMultipleBuffering().apply_pass(sdfg, {}) == {"tiles": depth}
    sdfg.validate()
    A = np.random.rand(n)
    B = np.zeros(n)
    sdfg(A=A, B=B)
    assert np.allclose(B, expected_add(A, 0, TILE))


@pytest.mark.parametrize("depth", [2, 3])
def test_symbolic_size_start_and_stride(depth: int):
    """A symbolic trip count, a non-zero start and a stride larger than the tile, with gaps between tiles."""
    sdfg = tiled_add(f"mb_add_sym_{depth}", depth, "N", start=2, stride=6)
    assert LowerMultipleBuffering().apply_pass(sdfg, {}) == {"tiles": depth}
    sdfg.validate()
    for n in (3, 5, 6, 8, 12, 14, 20, 33):
        A = np.random.rand(n)
        B = np.zeros(n)
        sdfg(A=A, B=B, N=n)
        assert np.allclose(B, expected_add(A, 2, 6)), n


def test_structure():
    depth = 3
    sdfg = tiled_add("mb_structure", depth)
    LowerMultipleBuffering().apply_pass(sdfg, {})

    assert sdfg.arrays["buf"].shape == (depth, TILE)
    assert sdfg.arrays["buf"].total_size == depth * TILE
    prologue = sdfg.start_block
    (main,) = [e.dst for e in sdfg.out_edges(prologue)]
    (epilogue,) = [e.dst for e in sdfg.out_edges(main)]
    assert len(sdfg.nodes()) == 3
    assert all(isinstance(block, LoopRegion) for block in sdfg.nodes())
    assert (prologue.label, main.label, epilogue.label) == ("tiles_prologue", "tiles", "tiles_epilogue")
    assert all(loop.pipeline_depth == 0 for loop in loops(sdfg))

    # The main loop consumes tile n, then produces tile n + 2
    consume = main.start_block
    (produce,) = [e.dst for e in main.out_edges(consume)]
    assert len(main.nodes()) == 2
    assert isinstance(produce, ControlFlowRegion) and produce.label == "tiles_produce"
    assert isinstance(consume, ControlFlowRegion) and consume.label == "tiles_consume"

    # The main loop loads tile n + 2 into slot (n + 2) % 3, and the body reads tile n from slot n % 3
    (copy_edge,) = [e for state in produce.states() for e in state.edges()]
    assert str(copy_edge.data.subset) == "t + 8:t + 12"
    assert str(copy_edge.data.other_subset.ranges[0][0]) == "Mod(int_floor(t, 4) + 2, 3)"
    reads = [e.data for state in consume.states() for e in state.edges() if e.data.data == "buf"]
    assert reads and all(str(memlet.subset.ranges[0][0]) == "Mod(int_floor(t, 4), 3)" for memlet in reads)

    # No placeholder slot symbol is left behind
    assert not any("__mb_slot" in str(sym) for sym in sdfg.used_symbols(all_symbols=True))


def test_two_staged_buffers():
    depth = 3
    sdfg = tiled_gemm("mb_gemm", depth)
    assert LowerMultipleBuffering().apply_pass(sdfg, {}) == {"ktiles": depth}
    assert sdfg.arrays["a_tile"].shape == (depth, 5, 4)
    assert sdfg.arrays["b_tile"].shape == (depth, 4, 6)
    A = np.random.rand(5, 20)
    B = np.random.rand(20, 6)
    C = np.zeros((5, 6))
    sdfg(A=A, B=B, C=C)
    assert np.allclose(C, A @ B)


@pytest.mark.parametrize("variant", ["copy_node", "nested", "two_states"])
def test_body_variants(variant: str):
    """A copy library node, a buffer consumed in a nested SDFG, and a buffer read in a later state."""
    sdfg = tiled_add(f"mb_{variant}", 3, 20, **{variant: True})
    assert LowerMultipleBuffering().apply_pass(sdfg, {}) == {"tiles": 3}
    sdfg.validate()
    A = np.random.rand(20)
    B = np.zeros(20)
    sdfg(A=A, B=B)
    assert np.allclose(B, expected_add(A, 0, TILE))


def test_serialization_round_trip():
    sdfg = tiled_add("mb_serialize", 3, 28)
    LowerMultipleBuffering().apply_pass(sdfg, {})
    sdfg = dace.SDFG.from_json(sdfg.to_json())
    A = np.random.rand(28)
    B = np.zeros(28)
    sdfg(A=A, B=B)
    assert np.allclose(B, expected_add(A, 0, TILE))


@pytest.mark.parametrize("depth", [0, 1])
def test_depth_below_two_is_noop(depth: int):
    sdfg = tiled_add(f"mb_noop_{depth}", depth)
    before = copy.deepcopy(sdfg.to_json())
    assert LowerMultipleBuffering().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before


def _assert_skipped(sdfg: dace.SDFG):
    with pytest.warns(UserWarning, match="Multiple buffering of loop tiles skipped"):
        assert LowerMultipleBuffering().apply_pass(sdfg, {}) is None
    assert sdfg.arrays["buf"].shape == (TILE,)
    assert len(loops(sdfg)) == 1
    assert loops(sdfg)[0].pipeline_depth == 0
    sdfg.validate()


def test_reject_break():
    sdfg = tiled_add("mb_break", 2)
    (loop,) = loops(sdfg)
    loop.add_edge(loop.start_block, loop.add_break("stop"), dace.InterstateEdge("t > 100"))
    _assert_skipped(sdfg)


def test_reject_symbolic_stride():
    _assert_skipped(tiled_add("mb_symstride", 2, stride="S"))


def test_reject_inverted():
    sdfg = tiled_add("mb_inverted", 2)
    loops(sdfg)[0].inverted = True
    _assert_skipped(sdfg)


def test_reject_buffer_used_after_loop():
    sdfg = tiled_add("mb_live_out", 2)
    after = sdfg.add_state_after(loops(sdfg)[0], "after")
    after.add_nedge(after.add_read("buf"), after.add_write("B"), dace.Memlet(f"buf[0:{TILE}]", other_subset="0:4"))
    _assert_skipped(sdfg)


def test_reject_buffer_read_before_copy():
    sdfg = tiled_add("mb_read_first", 2)
    (loop,) = loops(sdfg)
    sdfg.add_array("C", [1], dace.float64)
    first = loop.add_state_before(loop.start_block, "first", is_start_block=True)
    first.add_nedge(first.add_read("buf"), first.add_write("C"), dace.Memlet("buf[0]", other_subset="0"))
    _assert_skipped(sdfg)


def test_reject_source_written_in_loop():
    sdfg = tiled_add("mb_src_written", 2)
    (loop,) = loops(sdfg)
    last = loop.add_state_after(loop.start_block, "write_a")
    tasklet = last.add_tasklet("w", {}, {"o"}, "o = 0")
    last.add_edge(tasklet, "o", last.add_write("A"), None, dace.Memlet("A[t]"))
    _assert_skipped(sdfg)


def test_buffers_override():
    sdfg = tiled_gemm("mb_override", 2)
    LowerMultipleBuffering(buffers=["a_tile"]).apply_pass(sdfg, {})
    assert sdfg.arrays["a_tile"].shape == (2, 5, 4)
    assert sdfg.arrays["b_tile"].shape == (4, 6)
    A = np.random.rand(5, 20)
    B = np.random.rand(20, 6)
    C = np.zeros((5, 6))
    sdfg(A=A, B=B, C=C)
    assert np.allclose(C, A @ B)


def test_buffers_override_warns_on_illegal_buffer():
    sdfg = tiled_add("mb_override_bad", 2)
    with pytest.warns(UserWarning, match="Buffer B of loop tiles cannot be staged"):
        LowerMultipleBuffering(buffers=["B"]).apply_pass(sdfg, {})


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 20])
def test_blocked_gemm(depth: int, k: int):
    """GEMM tiled in M, N and K; only the innermost K loop is pipelined."""
    m, n, bm, bn, bk = 6, 8, 3, 4, 4
    sdfg = blocked_gemm(f"mb_blocked_gemm_{depth}_{k}", depth, m, n, k, bm, bn, bk)
    assert LowerMultipleBuffering().apply_pass(sdfg, {}) == {"ktiles": depth}
    sdfg.validate()
    assert sdfg.arrays["a_tile"].shape == (depth, bm, bk)
    assert sdfg.arrays["b_tile"].shape == (depth, bk, bn)
    A = np.random.rand(m, k)
    B = np.random.rand(k, n)
    C = np.zeros((m, n))
    sdfg(A=A, B=B, C=C)
    assert np.allclose(C, A @ B)


@pytest.mark.parametrize("n_depth", [2, 3])
@pytest.mark.parametrize("n", [4, 8, 12])
def test_blocked_gemm_pipelined_n_tiles(n_depth: int, n: int):
    """Both the loop over K tiles and the loop over N tiles are pipelined, the latter staging a tile of ``D``."""
    m, k, bm, bn, bk = 6, 8, 3, 4, 4
    sdfg = blocked_gemm(f"mb_blocked_gemm_n_{n_depth}_{n}", 2, m, n, k, bm, bn, bk, n_depth)
    assert LowerMultipleBuffering().apply_pass(sdfg, {}) == {"ktiles": 2, "ntiles": n_depth}
    sdfg.validate()
    assert sdfg.arrays["d_tile"].shape == (n_depth, bm, bn)
    A = np.random.rand(m, k)
    B = np.random.rand(k, n)
    D = np.random.rand(m, n)
    C = np.zeros((m, n))
    sdfg(A=A, B=B, C=C, D=D)
    assert np.allclose(C, A @ B + D)


if __name__ == "__main__":
    for d in (2, 3, 4):
        for t in (1, 2, 3, 4, 7):
            test_tiled_add(d, t)
        test_symbolic_size_start_and_stride(d) if d < 4 else None
    test_structure()
    test_two_staged_buffers()
    for v in ("copy_node", "nested", "two_states"):
        test_body_variants(v)
    test_serialization_round_trip()
    test_depth_below_two_is_noop(0)
    test_depth_below_two_is_noop(1)
    test_reject_break()
    test_reject_symbolic_stride()
    test_reject_inverted()
    test_reject_buffer_used_after_loop()
    test_reject_buffer_read_before_copy()
    test_reject_source_written_in_loop()
    test_buffers_override()
    test_buffers_override_warns_on_illegal_buffer()
    for d in (2, 3):
        for kk in (4, 8, 20):
            test_blocked_gemm(d, kk)
    for d in (2, 3):
        for nn in (4, 8, 12):
            test_blocked_gemm_pipelined_n_tiles(d, nn)
