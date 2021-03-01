# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
from dace.codegen.exceptions import CompilerConfigurationError, CompilationError
import dace.libraries.mpi as mpi
import sys
import warnings
import numpy as np

###############################################################################


def make_sdfg(dtype):

    n = dace.symbol("n")

    sdfg = dace.SDFG("mpi_send_recv")
    state = sdfg.add_state("dataflow")

    sdfg.add_array("x", [n], dtype, transient=False)
    sdfg.add_array("y", [n], dtype, transient=False)
    sdfg.add_array("src", [1], dace.dtypes.int32, transient=False)
    sdfg.add_array("dest", [1], dace.dtypes.int32, transient=False)
    sdfg.add_array("tag", [1], dace.dtypes.int32, transient=False)
    x = state.add_access("x")
    y = state.add_access("y")
    src = state.add_access("src")
    dest = state.add_access("dest")
    tag = state.add_access("tag")

    send_node = mpi.nodes.send.Send("send")
    recv_node = mpi.nodes.recv.Recv("recv")

    state.add_memlet_path(x,
                          send_node,
                          dst_conn="_buffer",
                          memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state.add_memlet_path(dest,
                          send_node,
                          dst_conn="_dest",
                          memlet=Memlet.simple(dest, "0:1", num_accesses=1))
    state.add_memlet_path(tag,
                          send_node,
                          dst_conn="_tag",
                          memlet=Memlet.simple(tag, "0:1", num_accesses=1))
    state.add_memlet_path(recv_node,
                          y,
                          src_conn="_buffer",
                          memlet=Memlet.simple(y, "0:n", num_accesses=n))
    state.add_memlet_path(src,
                          recv_node,
                          dst_conn="_src",
                          memlet=Memlet.simple(src, "0:1", num_accesses=1))
    state.add_memlet_path(tag,
                          recv_node,
                          dst_conn="_tag",
                          memlet=Memlet.simple(tag, "0:1", num_accesses=1))
    return sdfg


###############################################################################


def _test_mpi(info, sdfg, dtype):
    mpi_sdfg = sdfg.compile()
  
    size = 128
    A = np.random.randn(size)
    B = np.zeros(size)
    src = np.array([0], dtype=np.int32)
    dest = np.array([1], dtype=np.int32)
    tag = np.array([23], dtype=np.int32)
    mpi_sdfg(x=A, y=B, src=src, dest=dest, tag=tag, n=size)


def test_mpi():
    _test_mpi("MPI Send/Recv", make_sdfg(np.float), np.float)

###############################################################################

if __name__ == "__main__":
    test_mpi()
###############################################################################
