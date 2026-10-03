# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Offline structural tests for MpiPackUnpack.

A non-contiguous (strided) MPI buffer gets a contiguous ``packed_*`` transient plus a pack/unpack map, so
MPI sees a contiguous buffer in the new layout; a contiguous buffer is left untouched; a shuffled buffer
is refused. Runtime correctness is covered by the 2-rank tests under tests/library/mpi.
"""
import pytest
import dace
from dace.memlet import Memlet
from dace.libraries.mpi.nodes import Send, Recv
from dace.libraries.mpi.utils import is_access_contiguous
from dace.transformation.layout.mpi_pack_unpack import MpiPackUnpack


def _send_sdfg(bufname, subset, shape=(8, 8)):
    sdfg = dace.SDFG("mpu_send")
    sdfg.add_array(bufname, list(shape), dace.float64)
    sdfg.add_array("dest", [1], dace.dtypes.int32)
    sdfg.add_array("tag", [1], dace.dtypes.int32)
    st = sdfg.add_state("s", is_start_block=True)
    snd = Send("send")
    st.add_node(snd)
    st.add_edge(st.add_access(bufname), None, snd, "_buffer", Memlet.simple(bufname, subset))
    st.add_edge(st.add_access("dest"), None, snd, "_dest", Memlet.simple("dest", "0:1"))
    st.add_edge(st.add_access("tag"), None, snd, "_tag", Memlet.simple("tag", "0:1"))
    return sdfg, snd


def _packed_arrays(sdfg):
    return [n for n in sdfg.arrays if n.startswith("packed_")]


def _buffer_memlet(state, node, conn, out=False):
    edges = state.out_edges(node) if out else state.in_edges(node)
    key = "src_conn" if out else "dst_conn"
    return next(e.data for e in edges if getattr(e, key) == conn)


def test_pack_noncontiguous_send():
    """A strided (column) send buffer gets a packed transient + pack map; the Send buffer becomes contiguous."""
    sdfg, snd = _send_sdfg("A", "0:8, 2:3")  # a column: non-contiguous in C layout
    st = sdfg.start_state
    assert not is_access_contiguous(_buffer_memlet(st, snd, "_buffer"), sdfg.arrays["A"])

    assert MpiPackUnpack().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    packed = _packed_arrays(sdfg)
    assert len(packed) == 1
    buf = _buffer_memlet(st, snd, "_buffer")
    assert buf.data == packed[0]
    assert is_access_contiguous(buf, sdfg.arrays[packed[0]])  # MPI now sees a contiguous buffer
    # a pack map was inserted
    assert any(isinstance(n, dace.nodes.MapEntry) and n.label.startswith("pack_") for n in st.nodes())


def test_contiguous_send_untouched():
    """A contiguous (row) send buffer is left as-is (identity fast-path)."""
    sdfg, snd = _send_sdfg("A", "2:3, 0:8")  # a row: contiguous
    assert MpiPackUnpack().apply_pass(sdfg, {}) == 0
    assert _packed_arrays(sdfg) == []


def test_shuffled_buffer_refused():
    """MPI on a shuffled array is refused (Shuffle and MPI are mutually exclusive)."""
    sdfg, _ = _send_sdfg("shuffled_A", "0:8, 2:3")
    with pytest.raises(NotImplementedError):
        MpiPackUnpack().apply_pass(sdfg, {})


def test_map_produced_buffer_refused():
    """A send buffer produced directly by a map (MapExit), not an AccessNode, is refused (MPI-in-map YAGNI)."""
    sdfg = dace.SDFG("mpu_map_send")
    sdfg.add_array("A", [8, 8], dace.float64)
    sdfg.add_array("dest", [1], dace.dtypes.int32)
    sdfg.add_array("tag", [1], dace.dtypes.int32)
    st = sdfg.add_state("s", is_start_block=True)
    snd = Send("send")
    st.add_node(snd)
    me, mx = st.add_map("m", {"i": "0:8"})
    tsk = st.add_tasklet("t", {}, {"__o"}, "__o = 0.0")
    st.add_edge(me, None, tsk, None, Memlet())
    # MapExit feeds the Send buffer directly (a strided column) -> src is not an AccessNode.
    st.add_memlet_path(tsk, mx, snd, src_conn="__o", dst_conn="_buffer", memlet=Memlet.simple("A", "0:8, 2:3"))
    with pytest.raises(NotImplementedError):
        MpiPackUnpack().apply_pass(sdfg, {})


def test_unpack_noncontiguous_recv():
    """A strided recv buffer gets a packed transient + an unpack map; the Recv writes the packed transient."""
    sdfg = dace.SDFG("mpu_recv")
    sdfg.add_array("A", [8, 8], dace.float64)
    sdfg.add_array("src", [1], dace.dtypes.int32)
    sdfg.add_array("tag", [1], dace.dtypes.int32)
    st = sdfg.add_state("s", is_start_block=True)
    rcv = Recv("recv")
    st.add_node(rcv)
    st.add_edge(st.add_access("src"), None, rcv, "_src", Memlet.simple("src", "0:1"))
    st.add_edge(st.add_access("tag"), None, rcv, "_tag", Memlet.simple("tag", "0:1"))
    st.add_edge(rcv, "_buffer", st.add_access("A"), None, Memlet.simple("A", "0:8, 2:3"))

    assert MpiPackUnpack().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert len(_packed_arrays(sdfg)) == 1
    assert _buffer_memlet(st, rcv, "_buffer", out=True).data.startswith("packed_")
    assert any(isinstance(n, dace.nodes.MapEntry) and n.label.startswith("unpack_") for n in st.nodes())


if __name__ == "__main__":
    test_pack_noncontiguous_send()
    test_contiguous_send_untouched()
    test_shuffled_buffer_refused()
    test_map_produced_buffer_refused()
    test_unpack_noncontiguous_recv()
    print("mpi_pack_unpack offline tests PASS")
