# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ForwardOffsetCopies``: a shifted copy read in later states is read from its source at the shifted index, and
the copy goes; a write to the source between the copy and a read keeps it."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.forward_offset_copies import ForwardOffsetCopies

N = dace.symbol("N")


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


if __name__ == "__main__":
    test_a_shifted_copy_read_in_a_later_state_reads_its_source()
    test_a_write_to_the_source_between_keeps_the_copy()
    test_a_device_to_host_transfer_is_not_forwarded()
    test_rebound_offset_symbol_keeps_the_copy()
    test_a_reader_naming_its_destination_reads_at_the_offset()
    test_a_flat_view_of_the_copy_keeps_the_copy()
