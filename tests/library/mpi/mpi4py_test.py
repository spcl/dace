# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg import utils
import dace.dtypes as dtypes
import numpy as np
import pytest



@pytest.mark.mpi
def test_process_grid_bcast():

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    @dace.program
    def pgrid_bcast(A: dace.int32[10]):
        pgrid = MPI.COMM_WORLD.Create_cart([1, size])
        if pgrid != MPI.COMM_NULL:
            pgrid.Bcast(A)

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    sdfg = None
    if rank == 0:
        sdfg = pgrid_bcast.to_sdfg()
    func = utils.distributed_compile(sdfg, commworld)

    if rank == 0:
        A = np.arange(10, dtype=np.int32)
        A_ref = A.copy()
    else:
        A = np.zeros((10, ), dtype=np.int32)
        A_ref = A.copy()

    func(A=A)
    pgrid_bcast.f(A_ref)

    assert(np.array_equal(A, A_ref))


@pytest.mark.mpi
def test_sub_grid_bcast():

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    @dace.program
    def subgrid_bcast(A: dace.int32[10], rank: dace.int32):
        pgrid = commworld.Create_cart([2, size // 2])
        if pgrid != MPI.COMM_NULL:
            sgrid = pgrid.Sub([False, True])
            pgrid.Bcast(A)
        B = np.empty_like(A)
        B[:] = rank % 10
        if pgrid != MPI.COMM_NULL:
            sgrid.Bcast(B)
        A[:] = B

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    sdfg = None
    if rank == 0:
        sdfg = subgrid_bcast.to_sdfg()
    func = utils.distributed_compile(sdfg, commworld)

    if rank == 0:
        A = np.arange(10, dtype=np.int32)
    else:
        A = np.ones((10, ), dtype=np.int32)
    A_ref = A.copy()

    func(A=A, rank=rank)
    subgrid_bcast.f(A_ref, rank)

    assert(np.array_equal(A, A_ref))


def initialize_3mm(b_NI: int, b_NJ: int, b_NK: int, b_NL: int, b_NM: int,
                   ts_NI: int, ts_NJ: int, ts_NK, ts_NL: int, ts_NM: int,
                   NI: int, NJ: int, NK: int, NL: int, NM: int,
                   datatype: type = np.float64):

    A = np.fromfunction(lambda i, k: b_NK + k + 1, (ts_NI, ts_NK), dtype=datatype)
    B = np.eye(ts_NK, ts_NJ, b_NK - b_NJ)
    C = np.fromfunction(lambda j, m: b_NJ + j + 1, (ts_NJ, ts_NM), dtype=datatype)
    D = np.eye(ts_NM, ts_NL, b_NM - b_NL)

    if b_NI + ts_NI > NI:
        A[NI - b_NI:] = 0
    if b_NJ + ts_NJ > NJ:
        B[:, NJ - b_NJ:] = 0
        C[NJ - b_NJ:] = 0
    if b_NK + ts_NJ > NK:
        A[:, NK - b_NK:] = 0
        B[NK - b_NK:] = 0
    if b_NL + ts_NL > NL:
        D[:NL - b_NL] = 0
    if b_NM + ts_NM > NM:
        C[:NM - b_NM] = 0
        D[NM - b_NM:] = 0

    return A, B, C, D


@pytest.mark.mpi
def test_3mm():

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    @dace.program
    def k3mm(A, B, C, D):
        cart_comm = commworld.Create_cart([1, size, 1])
        if cart_comm != MPI.COMM_NULL:

            ab_reduce_comm = cart_comm.Sub([False, False, True])
            cd_reduce_comm = cart_comm.Sub([True, False, False])
            abcd_reduce_comm = cart_comm.Sub([False, True, False])

            ab = A @ B
            ab_reduce_comm.Allreduce(MPI.IN_PLACE, ab, op=MPI.SUM)
            cd = C @ D
            cd_reduce_comm.Allreduce(MPI.IN_PLACE, cd, op=MPI.SUM)
            E = ab @ cd
            abcd_reduce_comm.Allreduce(MPI.IN_PLACE, E, op=MPI.SUM)

            return E

    N = 128
    assert(size <= 128)
    
    NI, NJ, NK, NL, NM = (N,) * 5
    PNI, PNJ, PNK, PNL, PNM = 1, 2, 1, 1, 1

    cart_comm = commworld.Create_cart([1, size, 1])
    cart_rank = cart_comm.Get_rank()
    cart_size = cart_comm.Get_size()
    cart_coords = cart_comm.Get_coords(cart_rank)
    
    ts_NI = int(np.ceil(NI / PNI))
    ts_NJ = int(np.ceil(NJ / PNJ))
    ts_NK = int(np.ceil(NJ / PNK))
    ts_NL = int(np.ceil(NL / PNL))
    ts_NM = int(np.ceil(NM / PNM))

    b_NI = cart_coords[0] * ts_NI
    b_NJ = cart_coords[1] * ts_NJ
    b_NK = cart_coords[2] * ts_NK
    b_NL = cart_coords[2] * ts_NL
    b_NM = cart_coords[0] * ts_NM
    A, B, C, D = initialize_3mm(b_NI, b_NJ, b_NK, b_NL, b_NM, ts_NI, ts_NJ, ts_NK, ts_NL, ts_NM, NI, NJ, NK, NL, NM)

    sdfg = None
    if rank == 0:
        sdfg = k3mm.to_sdfg(A=A, B=B, C=C, D=D)
    func = utils.distributed_compile(sdfg, commworld)

    E = func(A=A, B=B, C=C, D=D)
    commworld.Barrier()
    E_ref = k3mm.f(A, B, C, D)
    commworld.Barrier()

    if E_ref is not None:
        assert(np.array_equal(E, E_ref))


@pytest.mark.mpi
def test_isend_irecv():

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    @dace.program
    def chain(rank: dace.int32, size: dace.int32):
        src = (rank - 1) % size
        dst = (rank + 1) % size
        req = np.empty((2, ), dtype=MPI.Request)
        sbuf = np.full((1,), rank, dtype=np.int32)
        req[0] = commworld.Isend(sbuf, dst, tag=0)
        rbuf = np.empty((1, ), dtype=np.int32)
        req[1] = commworld.Irecv(rbuf, src, tag=0)
        MPI.Request.Waitall(req)
        return rbuf
    
    sdfg = None
    if rank == 0:
        sdfg = chain.to_sdfg(simplify=True)
    func = utils.distributed_compile(sdfg, commworld)

    val = func(rank=rank, size=size)
    ref = chain.f(rank, size)

    assert(val[0] == ref[0])


if __name__ == "__main__":

    # test_process_grid_bcast()
    # test_sub_grid_bcast()
    # test_3mm()
    test_isend_irecv()
