# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Async (Irecv/Isend + Wait/Waitall) coverage for MpiPackUnpack.

A non-blocking recv into a strided (column) buffer is made MPI-legal by packing the recv into a
contiguous transient at the Irecv and unpacking it back into the column *after* the matching Wait --
the receive is only complete there. Two structural checks run offline (no MPI): one flat, one with the
Irecv/Wait inside a ``LoopRegion`` (the halo-in-a-time-loop shape), which is where the async unpack must
be spliced into the Wait's own graph, not the top SDFG. A 2-rank ring exercises it at runtime.
"""
import dace
from dace.memlet import Memlet
import dace.libraries.mpi as mpi
from dace.sdfg.state import LoopRegion
from dace.transformation.layout.mpi_pack_unpack import MpiPackUnpack
import numpy as np
import pytest

COL = 2

###############################################################################


def _wire_irecv_wait(state, y_name, col):
    """Irecv into column ``col`` of ``y_name`` (strided) + a Wait on the shared recv request."""
    recv = mpi.nodes.irecv.Irecv("irecv")
    wait = mpi.nodes.wait.Wait("wait")
    y = state.add_access(y_name)
    recv_req = state.add_access("recv_req")
    state.add_memlet_path(recv, y, src_conn="_buffer", memlet=Memlet.simple(y_name, f"0:n, {col}:{col + 1}"))
    state.add_memlet_path(recv, recv_req, src_conn="_request", memlet=Memlet.simple("recv_req", "0:1"))
    state.add_memlet_path(state.add_access("src"), recv, dst_conn="_src", memlet=Memlet.simple("src", "0:1"))
    state.add_memlet_path(state.add_access("tag"), recv, dst_conn="_tag", memlet=Memlet.simple("tag", "0:1"))
    state.add_memlet_path(recv_req, wait, dst_conn="_request", memlet=Memlet.simple("recv_req", "0:1"))
    state.add_memlet_path(wait,
                          state.add_access("stat_tag"),
                          src_conn="_stat_tag",
                          memlet=Memlet.simple("stat_tag", "0:1"))
    state.add_memlet_path(wait,
                          state.add_access("stat_source"),
                          src_conn="_stat_source",
                          memlet=Memlet.simple("stat_source", "0:1"))
    return wait


def _irecv_wait_sdfg(in_loop: bool):
    n = dace.symbol("n")
    sdfg = dace.SDFG("mpi_irecv_async_loop" if in_loop else "mpi_irecv_async")
    sdfg.add_array("Y", [n, n], dace.float64, transient=False)
    for nm in ("src", "tag"):
        sdfg.add_array(nm, [1], dace.dtypes.int32, transient=False)
    sdfg.add_array("recv_req", [1], dace.dtypes.opaque("MPI_Request"), transient=True)
    sdfg.add_array("stat_tag", [1], dace.dtypes.int32, transient=True)
    sdfg.add_array("stat_source", [1], dace.dtypes.int32, transient=True)

    if in_loop:
        loop = LoopRegion("halo",
                          loop_var="t",
                          initialize_expr="t = 0",
                          condition_expr="t < 4",
                          update_expr="t = t + 1")
        sdfg.add_node(loop, is_start_block=True)
        sdfg.add_symbol("t", dace.int64)
        state = loop.add_state("exchange", is_start_block=True)
        host = loop
    else:
        state = sdfg.add_state("exchange", is_start_block=True)
        host = sdfg

    wait = _wire_irecv_wait(state, "Y", COL)
    return sdfg, state, wait, host


def _packed_arrays(sdfg):
    return [a for a in sdfg.arrays if a.startswith("packed_")]


def _unpack_state(sdfg):
    for st in sdfg.all_states():
        if any(isinstance(nd, dace.nodes.MapEntry) and nd.label.startswith("unpack_") for nd in st.nodes()):
            return st
    return None


###############################################################################


def test_irecv_unpack_after_wait():
    """Flat: the strided Irecv gets a packed recv buffer + an unpack in a NEW state after the Wait."""
    sdfg, state, wait, _ = _irecv_wait_sdfg(in_loop=False)
    assert MpiPackUnpack().apply_pass(sdfg, {}) == 1

    packed = _packed_arrays(sdfg)
    assert len(packed) == 1
    # Fix: the async buffer must outlive its state (Irecv -> Wait), so SDFG lifetime, not Scope.
    assert sdfg.arrays[packed[0]].lifetime == dace.dtypes.AllocationLifetime.SDFG
    # The Irecv now writes the contiguous packed buffer.
    buf = next(e.data for e in state.out_edges_by_connector(_irecv(state), "_buffer"))
    assert buf.data == packed[0]
    # The unpack landed in a fresh state wired after the Wait's state.
    post = _unpack_state(sdfg)
    assert post is not None and post is not state
    assert any(e.dst is post for e in sdfg.out_edges(state))
    sdfg.validate()


def test_irecv_unpack_inside_loop():
    """Wait inside a LoopRegion: the unpack state is spliced into the loop, not the top SDFG (region-safe)."""
    sdfg, state, wait, loop = _irecv_wait_sdfg(in_loop=True)
    assert MpiPackUnpack().apply_pass(sdfg, {}) == 1

    post = _unpack_state(sdfg)
    assert post is not None
    # The new unpack state lives in the loop's graph, never leaked to the top SDFG.
    assert post in set(loop.nodes())
    assert post not in set(sdfg.nodes())
    sdfg.validate()  # a cross-region edge (the pre-fix bug) would fail here


def _irecv(state):
    return next(n for n in state.nodes() if isinstance(n, mpi.nodes.irecv.Irecv))


###############################################################################
#  Runtime: 2-rank async ring exchanging a strided column (Isend + Irecv + Waitall)
###############################################################################


def _async_ring_sdfg(dtype):
    n = dace.symbol("n")
    sdfg = dace.SDFG("mpi_pack_async_ring")
    state = sdfg.add_state("exchange", is_start_block=True)
    sdfg.add_array("X", [n, n], dtype, transient=False)  # send column COL
    sdfg.add_array("Y", [n, n], dtype, transient=False)  # recv column COL
    for nm in ("src", "dest", "tag"):
        sdfg.add_array(nm, [1], dace.dtypes.int32, transient=False)
    sdfg.add_array("req", [2], dace.dtypes.opaque("MPI_Request"), transient=True)

    col = f"0:n, {COL}:{COL + 1}"
    isend = mpi.nodes.isend.Isend("isend")
    irecv = mpi.nodes.irecv.Irecv("irecv")
    waitall = mpi.nodes.wait.Waitall("waitall")
    req_s = state.add_access("req")
    req_r = state.add_access("req")

    state.add_memlet_path(state.add_access("X"), isend, dst_conn="_buffer", memlet=Memlet.simple("X", col))
    state.add_memlet_path(state.add_access("dest"), isend, dst_conn="_dest", memlet=Memlet.simple("dest", "0:1"))
    state.add_memlet_path(state.add_access("tag"), isend, dst_conn="_tag", memlet=Memlet.simple("tag", "0:1"))
    state.add_memlet_path(isend, req_s, src_conn="_request", memlet=Memlet.simple("req", "0:1"))

    state.add_memlet_path(irecv, state.add_access("Y"), src_conn="_buffer", memlet=Memlet.simple("Y", col))
    state.add_memlet_path(state.add_access("src"), irecv, dst_conn="_src", memlet=Memlet.simple("src", "0:1"))
    state.add_memlet_path(state.add_access("tag"), irecv, dst_conn="_tag", memlet=Memlet.simple("tag", "0:1"))
    state.add_memlet_path(irecv, req_r, src_conn="_request", memlet=Memlet.simple("req", "1:2"))

    state.add_memlet_path(req_s, waitall, dst_conn="_request", memlet=Memlet.simple("req", "0:2"))

    # send column and recv column are strided -> pack both; recv unpacks after the Waitall.
    assert MpiPackUnpack().apply_pass(sdfg, {}) == 2
    return sdfg


def _test_mpi(sdfg, dtype):
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    drank, srank = (rank + 1) % size, (rank - 1 + size) % size
    mpi_sdfg = None
    for r in range(size):
        if r == rank:
            mpi_sdfg = sdfg.compile()
        comm.Barrier()

    N = 16
    X = np.full((N, N), rank, dtype=dtype)
    Y = np.zeros((N, N), dtype=dtype)
    mpi_sdfg(X=X,
             Y=Y,
             src=np.array([srank], dtype=np.int32),
             dest=np.array([drank], dtype=np.int32),
             tag=np.array([7], dtype=np.int32),
             n=N)
    if not np.allclose(Y[:, COL], srank):
        raise ValueError(f"rank {rank}: Y[:,{COL}] = {Y[:, COL]} (expected {srank})")


@pytest.mark.mpi
def test_mpi_pack_unpack_async_ring():
    _test_mpi(_async_ring_sdfg(np.float64), np.float64)


if __name__ == "__main__":
    test_irecv_unpack_after_wait()
    test_irecv_unpack_inside_loop()
    print("async offline structural tests PASS")
