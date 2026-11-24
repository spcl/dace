# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ForwardOffsetCopies``: a shifted copy read in later states is read from its source at the shifted index, and
the copy goes; a write to the source between the copy and a read keeps it."""

import copy as copy_module

import numpy as np

import dace
from dace.libraries.blas.nodes.gemm import Gemm
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.canonicalize.forward_offset_copies import ForwardOffsetCopies

N = dace.symbol("N")
K = dace.symbol("K")


def shifted_copy_sdfg(write_between: bool) -> dace.SDFG:
    """``t = q[1:N+1]`` in one state, ``out[i] = t[i] * 2`` in the next; with ``write_between`` a state between them
    sets ``q[:] = 0``."""
    sdfg = dace.SDFG("shifted_copy_" + ("written" if write_between else "clean"))
    sdfg.add_array("q", [N + 1], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_transient("t", [N], dace.float64)
    copy = sdfg.add_state("copy")
    copy.add_nedge(copy.add_read("q"), copy.add_write("t"), dace.Memlet("q[1:N + 1] -> [0:N]"))
    last = copy
    if write_between:
        clobber = sdfg.add_state_after(copy, "clobber")
        clobber.add_mapped_tasklet(
            "zero", {"i": "0:N + 1"}, {}, "o = 0.0", {"o": dace.Memlet("q[i]")}, external_edges=True
        )
        last = clobber
    use = sdfg.add_state_after(last, "use")
    use.add_mapped_tasklet(
        "double",
        {"i": "0:N"},
        {"x": dace.Memlet("t[i]")},
        "o = x * 2.0",
        {"o": dace.Memlet("out[i]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def test_a_shifted_copy_read_in_a_later_state_reads_its_source():
    sdfg = shifted_copy_sdfg(write_between=False)
    assert ForwardOffsetCopies().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert "t" not in sdfg.arrays
    reads = [str(e.data) for st in sdfg.states() for e in st.edges() if isinstance(e.dst, nodes.Tasklet)]
    assert reads == ["q[i + 1]"], reads
    q, out = np.random.default_rng(0).random(9), np.zeros(8)
    sdfg(q=q, out=out, N=8)
    assert np.allclose(out, 2.0 * q[1:])


def test_a_write_to_the_source_between_keeps_the_copy():
    sdfg = shifted_copy_sdfg(write_between=True)
    assert ForwardOffsetCopies().apply_pass(sdfg, {}) is None
    q, out = np.random.default_rng(1).random(9), np.zeros(8)
    expected = 2.0 * q[1:]
    sdfg(q=q, out=out, N=8)
    assert np.allclose(out, expected)


def test_a_device_to_host_transfer_is_not_forwarded():
    """warpx_esirkepov on GPU: a host map reading the host twin of a device array keeps reading the twin."""
    sdfg = shifted_copy_sdfg(write_between=False)
    sdfg.arrays["q"].storage = dace.StorageType.GPU_Global
    sdfg.arrays["t"].storage = dace.StorageType.CPU_Heap
    assert ForwardOffsetCopies().apply_pass(sdfg, {}) is None
    assert "t" in sdfg.arrays


def test_rebound_offset_symbol_keeps_the_copy():
    """``t = q[k:k + N]; k = k + 1; out = t``: reading ``q`` at ``k`` afterwards would read the next window."""
    sdfg = dace.SDFG("shifted_copy_rebound")
    sdfg.add_array("q", [N + 4], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_transient("t", [N], dace.float64)
    sdfg.add_symbol("k", dace.int64)
    copy = sdfg.add_state("copy")
    copy.add_nedge(copy.add_read("q"), copy.add_write("t"), dace.Memlet("q[k:k + N] -> [0:N]"))
    use = sdfg.add_state("use")
    sdfg.add_edge(copy, use, dace.InterstateEdge(assignments={"k": "k + 1"}))
    use.add_mapped_tasklet(
        "read", {"i": "0:N"}, {"x": dace.Memlet("t[i]")}, "o = x", {"o": dace.Memlet("out[i]")}, external_edges=True
    )
    sdfg.validate()
    assert ForwardOffsetCopies().apply_pass(sdfg, {}) is None
    assert "t" in sdfg.arrays


def test_a_reader_naming_its_destination_reads_at_the_offset():
    sdfg = dace.SDFG("forward_offset_implicit_source")
    sdfg.add_array("A", [6], dace.float64)
    sdfg.add_array("C", [3], dace.float64)
    sdfg.add_transient("B", [3], dace.float64)
    first = sdfg.add_state("copy", is_start_block=True)
    second = sdfg.add_state_after(first, "read")
    first.add_nedge(first.add_read("A"), first.add_write("B"), dace.Memlet("A[2:5] -> [0:3]"))
    second.add_nedge(second.add_read("B"), second.add_write("C"), dace.Memlet("C[0:3]"))
    assert ForwardOffsetCopies().apply_pass(sdfg, {}) == 1
    A, C = np.arange(1.0, 7.0), np.zeros(3)
    sdfg(A=A, C=C)
    assert np.array_equal(C, A[2:5])


def test_a_flat_view_of_the_copy_keeps_the_copy():
    """``t = q[1:4, 1:4]`` read through a flat view of ``t``: ``q``'s rows are longer, so the view cannot take ``q``."""
    sdfg = dace.SDFG("forward_offset_flat_view")
    sdfg.add_array("q", [4, 4], dace.float64)
    sdfg.add_array("out", [9], dace.float64)
    sdfg.add_transient("t", [3, 3], dace.float64)
    sdfg.add_view("v", [9], dace.float64)
    first = sdfg.add_state("copy", is_start_block=True)
    second = sdfg.add_state_after(first, "read")
    first.add_nedge(first.add_read("q"), first.add_write("t"), dace.Memlet("q[1:4, 1:4] -> [0:3, 0:3]"))
    view = second.add_access("v")
    second.add_nedge(second.add_read("t"), view, dace.Memlet("t[0:3, 0:3]"))
    second.add_nedge(view, second.add_write("out"), dace.Memlet("v[0:9] -> [0:9]"))
    sdfg.validate()
    ForwardOffsetCopies().apply_pass(sdfg, {})
    q, out = np.arange(16.0).reshape(4, 4), np.zeros(9)
    sdfg(q=q, out=out)
    assert np.array_equal(out, q[1:, 1:].ravel())


def copy_then_loop_sdfg(name: str, write_in_loop: bool) -> dace.SDFG:
    """``t = q[1:N+1]``, then a loop over ``k`` whose body sets ``out[i] = t[i] * 2``; with ``write_in_loop`` a second
    body state sets ``q[:] = 0`` after the read, so the next trip would read the zeroed source."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("q", [N + 1], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_transient("t", [N], dace.float64)
    copy = sdfg.add_state("copy", is_start_block=True)
    copy.add_nedge(copy.add_read("q"), copy.add_write("t"), dace.Memlet("q[1:N + 1] -> [0:N]"))
    loop = LoopRegion("trips", "k < 3", "k", "k = 0", "k = k + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(copy, loop, dace.InterstateEdge())
    use = loop.add_state("use", is_start_block=True)
    use.add_mapped_tasklet(
        "double",
        {"i": "0:N"},
        {"x": dace.Memlet("t[i]")},
        "o = x * 2.0",
        {"o": dace.Memlet("out[i]")},
        external_edges=True,
    )
    if write_in_loop:
        clobber = loop.add_state_after(use, "clobber")
        clobber.add_mapped_tasklet(
            "zero", {"i": "0:N + 1"}, {}, "o = 0.0", {"o": dace.Memlet("q[i]")}, external_edges=True
        )
    sdfg.reset_cfg_list()
    sdfg.validate()
    return sdfg


def test_a_shifted_copy_read_in_a_later_loop_reads_its_source():
    sdfg = copy_then_loop_sdfg("shifted_copy_loop_reader", write_in_loop=False)
    assert ForwardOffsetCopies().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert "t" not in sdfg.arrays
    q, out = np.random.default_rng(2).random(9), np.zeros(8)
    sdfg(q=q, out=out, N=8)
    assert np.allclose(out, 2.0 * q[1:])


def test_a_loop_writing_the_source_keeps_the_copy():
    sdfg = copy_then_loop_sdfg("shifted_copy_loop_writer", write_in_loop=True)
    assert ForwardOffsetCopies().apply_pass(sdfg, {}) is None
    assert "t" in sdfg.arrays
    q, out = np.random.default_rng(3).random(9), np.zeros(8)
    expected = 2.0 * q[1:]
    sdfg(q=q, out=out, N=8)
    assert np.allclose(out, expected)


def test_a_copy_whose_extent_is_spelled_floor_is_forwarded():
    """``t``'s extent is Python's ``//`` (sympy ``floor``) and the copy's range the parser's (``int_floor``): one
    value, so the copy covers ``t`` and is forwarded."""
    extent = (N - 7) // 2 + 1
    sdfg = dace.SDFG("shifted_copy_floor_extent")
    sdfg.add_array("q", [extent + 1], dace.float64)
    sdfg.add_array("out", [extent], dace.float64)
    sdfg.add_transient("t", [extent], dace.float64)
    copy = sdfg.add_state("copy")
    copy.add_nedge(
        copy.add_read("q"), copy.add_write("t"), dace.Memlet("q[1:(N - 7) // 2 + 2] -> [0:(N - 7) // 2 + 1]")
    )
    use = sdfg.add_state_after(copy, "use")
    use.add_mapped_tasklet(
        "double",
        {"i": "0:(N - 7) // 2 + 1"},
        {"x": dace.Memlet("t[i]")},
        "o = x * 2.0",
        {"o": dace.Memlet("out[i]")},
        external_edges=True,
    )
    sdfg.validate()
    reference = copy_module.deepcopy(sdfg)
    reference.name = "shifted_copy_floor_extent_reference"
    assert ForwardOffsetCopies().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert "t" not in sdfg.arrays
    size = 21
    q = np.random.default_rng(4).random((size - 7) // 2 + 2)
    out, expected = np.zeros(q.size - 1), np.zeros(q.size - 1)
    reference(q=q, out=expected, N=size)
    sdfg(q=q, out=out, N=size)
    assert np.allclose(out, expected)
    assert np.allclose(out, 2.0 * q[1:])


def test_an_offset_symbol_rebound_inside_a_branch_keeps_the_copy():
    """``t = q[k:k + N]``, then a branch whose arm sets ``k = k + 1`` before it reads ``t``: ``q`` at ``k`` there is the
    next window."""
    sdfg = dace.SDFG("shifted_copy_rebound_in_branch")
    sdfg.add_array("q", [N + 4], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_transient("t", [N], dace.float64)
    sdfg.add_symbol("k", dace.int64)
    copy = sdfg.add_state("copy", is_start_block=True)
    copy.add_nedge(copy.add_read("q"), copy.add_write("t"), dace.Memlet("q[k:k + N] -> [0:N]"))
    branch = ConditionalBlock("branch")
    sdfg.add_node(branch)
    sdfg.add_edge(copy, branch, dace.InterstateEdge())
    arm = ControlFlowRegion("arm", sdfg=sdfg)
    branch.add_branch(None, arm)
    head = arm.add_state("head", is_start_block=True)
    use = arm.add_state("use")
    arm.add_edge(head, use, dace.InterstateEdge(assignments={"k": "k + 1"}))
    use.add_mapped_tasklet(
        "read", {"i": "0:N"}, {"x": dace.Memlet("t[i]")}, "o = x", {"o": dace.Memlet("out[i]")}, external_edges=True
    )
    sdfg.reset_cfg_list()
    sdfg.validate()
    assert ForwardOffsetCopies().apply_pass(sdfg, {}) is None
    assert "t" in sdfg.arrays


def snapshot_shaped_copy_sdfg(name: str, rebind: bool) -> dace.SDFG:
    """``s = n``, then a branch copying ``t = a[0:n, 0:K]`` into ``t`` of shape ``(s, K)`` that feeds ``c = t @ b``
    and setting ``n = n - 1`` after, as the frontend sizes ``np.copy(a[:n, :k])`` by a copy of ``n``; with
    ``rebind``, ``n = n - 1`` also runs between ``s = n`` and the copy."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N + 2, K + 3], dace.float64)
    sdfg.add_array("b", [K, 4], dace.float64)
    sdfg.add_array("c", [N, 4], dace.float64)
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_symbol("s", dace.int64)
    sdfg.add_transient("t", ["s", K], dace.float64)
    head = sdfg.add_state("head", is_start_block=True)
    bind = sdfg.add_state_after(head, "bind", assignments={"n": "N"})
    branch = ConditionalBlock("branch")
    sdfg.add_node(branch)
    if rebind:
        middle = sdfg.add_state_after(bind, "middle", assignments={"s": "n"})
        sdfg.add_edge(middle, branch, dace.InterstateEdge(assignments={"n": "n - 1"}))
    else:
        sdfg.add_edge(bind, branch, dace.InterstateEdge(assignments={"s": "n"}))
    arm = ControlFlowRegion("arm", sdfg=sdfg)
    branch.add_branch("N > 0", arm)
    copy = arm.add_state("copy", is_start_block=True)
    arm.add_edge(copy, arm.add_state("after"), dace.InterstateEdge(assignments={"n": "n - 1"}))
    gemm = Gemm("gemm")
    gemm.implementation = "pure"
    copy.add_node(gemm)
    t = copy.add_access("t")
    copy.add_nedge(copy.add_read("a"), t, dace.Memlet("a[0:n, 0:K] -> [0:n, 0:K]"))
    copy.add_edge(t, None, gemm, "_a", dace.Memlet("t[0:n, 0:K]"))
    copy.add_edge(copy.add_read("b"), None, gemm, "_b", dace.Memlet("b[0:K, 0:4]"))
    copy.add_edge(gemm, "_c", copy.add_write("c"), None, dace.Memlet("c[0:n, 0:4]"))
    sdfg.reset_cfg_list()
    sdfg.validate()
    return sdfg


def test_a_copy_into_a_buffer_sized_by_a_symbol_copy_feeds_the_gemm_from_its_source():
    sdfg = snapshot_shaped_copy_sdfg("snapshot_shaped_copy", rebind=False)
    assert ForwardOffsetCopies().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert "t" not in sdfg.arrays
    (operand,) = [e for st in sdfg.states() for e in st.edges() if e.dst_conn == "_a"]
    assert operand.data.data == "a"
    rng = np.random.default_rng(5)
    a, b, c = rng.random((8, 6)), rng.random((3, 4)), np.zeros((6, 4))
    sdfg(a=a, b=b, c=c, N=6, K=3)
    assert np.allclose(c, a[:6, :3] @ b)


def test_a_symbol_rebound_after_its_copy_keeps_the_copy():
    sdfg = snapshot_shaped_copy_sdfg("snapshot_shaped_copy_rebound", rebind=True)
    assert ForwardOffsetCopies().apply_pass(sdfg, {}) is None
    assert "t" in sdfg.arrays
    rng = np.random.default_rng(6)
    a, b, c = rng.random((8, 6)), rng.random((3, 4)), np.zeros((6, 4))
    sdfg(a=a, b=b, c=c, N=6, K=3)
    assert np.allclose(c[:5], a[:5, :3] @ b)


if __name__ == "__main__":
    test_a_shifted_copy_read_in_a_later_state_reads_its_source()
    test_a_write_to_the_source_between_keeps_the_copy()
    test_a_device_to_host_transfer_is_not_forwarded()
    test_rebound_offset_symbol_keeps_the_copy()
    test_a_reader_naming_its_destination_reads_at_the_offset()
    test_a_flat_view_of_the_copy_keeps_the_copy()
    test_a_shifted_copy_read_in_a_later_loop_reads_its_source()
    test_a_loop_writing_the_source_keeps_the_copy()
    test_a_copy_whose_extent_is_spelled_floor_is_forwarded()
    test_an_offset_symbol_rebound_inside_a_branch_keeps_the_copy()
    test_a_copy_into_a_buffer_sized_by_a_symbol_copy_feeds_the_gemm_from_its_source()
    test_a_symbol_rebound_after_its_copy_keeps_the_copy()
