# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A memset whose array the next map overwrites element for element, unread, is dropped."""

import numpy as np

import dace
from dace.libraries.standard.nodes.fill import FillLibraryNode
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.dead_fill_elimination import DeadFillElimination

N = dace.symbol("N")


def zeroed_then_written(index: str = "i", map_range: str = "0:N", read_back: bool = False) -> dace.SDFG:
    """``tmp = 0; tmp[index] = b[i] (+ tmp[i]) for i in map_range; out = tmp`` in one state."""
    sdfg = dace.SDFG(f"zeroed_then_written_{abs(hash((index, map_range, read_back)))}")
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_array("idx", [N], dace.int64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_transient("tmp", [N], dace.float64)
    state = sdfg.add_state()
    zeroed = state.add_access("tmp")
    fill = FillLibraryNode("memset_tmp", value=0)
    state.add_node(fill)
    state.add_edge(fill, FillLibraryNode.OUTPUT_CONNECTOR_NAME, zeroed, None, dace.Memlet("tmp[0:N]"))
    b = state.add_read("b")
    state.add_nedge(zeroed, b, dace.Memlet())
    written = state.add_access("tmp")
    inputs = {"__b": dace.Memlet("b[i]")}
    code = "__t = __b"
    if read_back:
        inputs["__old"] = dace.Memlet("tmp[i]")
        code = "__t = __b + __old"
    if index != "i":
        inputs["__k"] = dace.Memlet("idx[i]")
    state.add_mapped_tasklet(
        "write",
        {"i": map_range},
        inputs,
        code,
        {"__t": dace.Memlet(f"tmp[{index}]")},
        input_nodes={"b": b, **({"tmp": zeroed} if read_back else {})},
        output_nodes={"tmp": written},
        external_edges=True,
    )
    state.add_nedge(written, state.add_write("out"), dace.Memlet("tmp[0:N]"))
    return sdfg


def fills(sdfg: dace.SDFG) -> int:
    return sum(isinstance(n, FillLibraryNode) for n, _ in sdfg.all_nodes_recursive())


def test_overwritten_memset_is_dropped():
    sdfg = zeroed_then_written()
    assert DeadFillElimination().apply_pass(sdfg, {}) == 1
    assert fills(sdfg) == 0
    sdfg.validate()
    b = np.random.rand(17)
    out = np.zeros(17)
    sdfg(b=b, idx=np.arange(17), out=out, N=17)
    assert np.allclose(out, b)


def test_memset_kept_when_the_map_misses_elements():
    sdfg = zeroed_then_written(map_range="1:N")
    assert DeadFillElimination().apply_pass(sdfg, {}) is None
    assert fills(sdfg) == 1


def test_memset_kept_under_an_indirect_write():
    sdfg = zeroed_then_written(index="__k")
    assert DeadFillElimination().apply_pass(sdfg, {}) is None
    assert fills(sdfg) == 1


def test_memset_kept_when_the_map_reads_it():
    sdfg = zeroed_then_written(read_back=True)
    assert DeadFillElimination().apply_pass(sdfg, {}) is None
    assert fills(sdfg) == 1


M = dace.symbol("M")


def zeroed_then_swept(
    start: int = 0,
    read_between: bool = False,
    guard: str | None = None,
    with_else: bool = False,
    fill_guarded: bool = False,
) -> dace.SDFG:
    """``tmp = 0`` in its own state, then ``for k in range(start, M): tmp[k, :] = b[k, :]`` (copy), then ``out = tmp``.

    ``guard`` puts the loop under ``if guard`` (``with_else``: an ``else`` sweeping too); ``fill_guarded`` puts the
    memset under ``if flag`` instead; ``read_between`` copies ``tmp`` out before the loop.
    """
    sdfg = dace.SDFG(f"zeroed_then_swept_{start}_{read_between}_{guard is not None}_{with_else}_{fill_guarded}")
    sdfg.add_array("b", [M, N], dace.float64)
    sdfg.add_array("out", [M, N], dace.float64)
    sdfg.add_array("early", [M, N], dace.float64)
    sdfg.add_scalar("flag", dace.int32)
    sdfg.add_transient("tmp", [M, N], dace.float64)

    def memset(state: dace.SDFGState) -> None:
        fill = FillLibraryNode("memset_tmp", value=0)
        state.add_node(fill)
        state.add_edge(
            fill, FillLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access("tmp"), None, dace.Memlet("tmp[0:M, 0:N]")
        )

    def sweep(region: ControlFlowRegion, label: str) -> LoopRegion:
        loop = LoopRegion(label, "k < M", "k", f"k = {start}", "k = k + 1")
        region.add_node(loop)
        body = loop.add_state(f"{label}_body", is_start_block=True)
        body.add_nedge(body.add_read("b"), body.add_write("tmp"), dace.Memlet("b[k, 0:N] -> [k, 0:N]"))
        return loop

    if fill_guarded:
        cond = ConditionalBlock("fill_if")
        sdfg.add_node(cond, is_start_block=True)
        branch = ControlFlowRegion("fill_g", sdfg=sdfg)
        cond.add_branch(CodeBlock("flag > 0"), branch)
        memset(branch.add_state("zero", is_start_block=True))
        last = cond
    else:
        last = sdfg.add_state("zero", is_start_block=True)
        memset(last)
    if read_between:
        peek = sdfg.add_state_after(last, "peek")
        peek.add_nedge(peek.add_read("tmp"), peek.add_write("early"), dace.Memlet("tmp[0:M, 0:N]"))
        last = peek
    if guard is None:
        middle = sweep(sdfg, "sweep")
    else:
        middle = ConditionalBlock("sweep_if")
        sdfg.add_node(middle)
        then = ControlFlowRegion("sweep_then", sdfg=sdfg)
        middle.add_branch(CodeBlock(guard), then)
        sweep(then, "sweep_a")
        if with_else:
            other = ControlFlowRegion("sweep_else", sdfg=sdfg)
            middle.add_branch(None, other)
            sweep(other, "sweep_b")
    sdfg.add_edge(last, middle, dace.InterstateEdge())
    done = sdfg.add_state_after(middle, "done")
    done.add_nedge(done.add_read("tmp"), done.add_write("out"), dace.Memlet("tmp[0:M, 0:N]"))
    return sdfg


def run_swept(sdfg: dace.SDFG, flag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    b = np.random.rand(5, 7)
    out = np.full((5, 7), -1.0)
    sdfg(b=b, out=out, early=np.zeros((5, 7)), flag=flag, M=5, N=7)
    return b, out


def test_memset_a_later_loop_overwrites_slice_by_slice_is_dropped():
    sdfg = zeroed_then_swept()
    assert DeadFillElimination().apply_pass(sdfg, {}) == 1
    assert fills(sdfg) == 0
    sdfg.validate()
    b, out = run_swept(sdfg)
    assert np.allclose(out, b)


def test_memset_kept_when_the_loop_skips_a_slice():
    sdfg = zeroed_then_swept(start=1)
    assert DeadFillElimination().apply_pass(sdfg, {}) is None
    b, out = run_swept(sdfg)
    assert np.allclose(out[0], 0) and np.allclose(out[1:], b[1:])


def test_memset_kept_when_read_before_the_loop():
    sdfg = zeroed_then_swept(read_between=True)
    assert DeadFillElimination().apply_pass(sdfg, {}) is None


def test_memset_kept_when_the_loop_is_guarded_without_else():
    sdfg = zeroed_then_swept(guard="flag > 0")
    assert DeadFillElimination().apply_pass(sdfg, {}) is None
    _, out = run_swept(sdfg, flag=0)
    assert np.allclose(out, 0)


def test_memset_every_branch_overwrites_is_dropped():
    sdfg = zeroed_then_swept(guard="flag > 0", with_else=True)
    assert DeadFillElimination().apply_pass(sdfg, {}) == 1
    for flag in (0, 1):
        b, out = run_swept(sdfg, flag=flag)
        assert np.allclose(out, b)


def test_memset_in_a_branch_overwritten_after_it_is_dropped():
    sdfg = zeroed_then_swept(fill_guarded=True)
    assert DeadFillElimination().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    b, out = run_swept(sdfg)
    assert np.allclose(out, b)


if __name__ == "__main__":
    test_overwritten_memset_is_dropped()
    test_memset_kept_when_the_map_misses_elements()
    test_memset_kept_under_an_indirect_write()
    test_memset_kept_when_the_map_reads_it()
    test_memset_a_later_loop_overwrites_slice_by_slice_is_dropped()
    test_memset_kept_when_the_loop_skips_a_slice()
    test_memset_kept_when_read_before_the_loop()
    test_memset_kept_when_the_loop_is_guarded_without_else()
    test_memset_every_branch_overwrites_is_dropped()
    test_memset_in_a_branch_overwritten_after_it_is_dropped()


def zeros_then_prefix_copy(prefix: str) -> dace.SDFG:
    """``sink = zeros(N + 1, 4); sink[prefix] = c; out = sink`` in one state, the zeros a map."""
    sdfg = dace.SDFG(f"zeros_then_prefix_copy_{abs(hash(prefix))}")
    sdfg.add_array("c", [N, 4], dace.float64)
    sdfg.add_array("out", [N + 1, 4], dace.float64)
    sdfg.add_transient("sink", [N + 1, 4], dace.float64)
    state = sdfg.add_state()
    zeroed = state.add_access("sink")
    state.add_mapped_tasklet(
        "zeros", {"i": "0:N + 1", "j": "0:4"}, {}, "__o = 0.0", {"__o": dace.Memlet("sink[i, j]")},
        output_nodes={"sink": zeroed}, external_edges=True,
    )
    c = state.add_read("c")
    state.add_nedge(zeroed, c, dace.Memlet())
    written = state.add_access("sink")
    state.add_nedge(c, written, dace.Memlet(data="c", subset="0:N, 0:4", other_subset=f"{prefix}, 0:4"))
    state.add_nedge(written, state.add_write("out"), dace.Memlet("sink[0:N + 1, 0:4]"))
    return sdfg


def zeroing_map_ranges(sdfg: dace.SDFG) -> list[str]:
    return [str(n.map.range) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]


def test_zeros_a_copy_overwrites_but_its_last_row_shrink_to_that_row():
    sdfg = zeros_then_prefix_copy("0:N")
    assert DeadFillElimination().apply_pass(sdfg, {}) == 1
    assert zeroing_map_ranges(sdfg) == ["N, 0:4"]
    sdfg.validate()
    c = np.random.rand(9, 4)
    out = np.full((10, 4), np.nan)
    sdfg(c=c, out=out, N=9)
    assert np.array_equal(out, np.concatenate([c, np.zeros((1, 4))]))


def test_zeros_a_copy_overwrites_but_its_first_row_shrink_to_that_row():
    sdfg = zeros_then_prefix_copy("1:N + 1")
    assert DeadFillElimination().apply_pass(sdfg, {}) == 1
    assert zeroing_map_ranges(sdfg) == ["0, 0:4"]
    c = np.random.rand(9, 4)
    out = np.full((10, 4), np.nan)
    sdfg(c=c, out=out, N=9)
    assert np.array_equal(out, np.concatenate([np.zeros((1, 4)), c]))


def test_zeros_a_copy_leaves_a_middle_row_of_stay_whole():
    sdfg = zeros_then_prefix_copy("1:N")
    assert DeadFillElimination().apply_pass(sdfg, {}) is None
    assert zeroing_map_ranges(sdfg) == ["0:N + 1, 0:4"]
