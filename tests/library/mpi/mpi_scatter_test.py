# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
import dace.libraries.mpi as mpi
import numpy as np
import pytest
import time

from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL

###############################################################################


def make_sdfg(dtype):

    n = dace.symbol("n")
    p = dace.symbol("p")

    sdfg = dace.SDFG("mpi_scatter")
    state = sdfg.add_state("dataflow")

    sdfg.add_array("inbuf", [n * p], dtype, transient=False)
    sdfg.add_array("outbuf", [n], dtype, transient=False)
    sdfg.add_array("root", [1], dace.dtypes.int32, transient=False)
    inbuf = state.add_access("inbuf")
    outbuf = state.add_access("outbuf")
    root = state.add_access("root")
    scatter_node = mpi.nodes.scatter.Scatter("scatter")

    state.add_memlet_path(inbuf,
                          scatter_node,
                          dst_conn="_inbuffer",
                          memlet=Memlet.simple(inbuf, "0:n*p", num_accesses=n))
    state.add_memlet_path(root,
                          scatter_node,
                          dst_conn="_root",
                          memlet=Memlet.simple(root, "0:1", num_accesses=1))
    state.add_memlet_path(scatter_node,
                          outbuf,
                          src_conn="_outbuffer",
                          memlet=Memlet.simple(outbuf, "0:n", num_accesses=1))

    return sdfg


###############################################################################


@pytest.mark.parametrize("implementation, dtype", [
    pytest.param("MPI", dace.float32, marks=pytest.mark.mpi),
    pytest.param("MPI", dace.float64, marks=pytest.mark.mpi)
])
def test_mpi(implementation, dtype):
    from mpi4py import MPI as MPI4PY
    np_dtype = getattr(np, dtype.to_string())
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    if commsize < 2:
        raise ValueError(
            "This test is supposed to be run with at least two processes!")
    for r in range(0, commsize):
        if r == rank:
            sdfg = make_sdfg(dtype)
            mpi_sdfg = sdfg.compile()
        comm.Barrier()

    size = 8
    A = np.full(size * commsize, 7, dtype=np_dtype)
    B = np.full(size, 42, dtype=np_dtype)
    root = np.array([0], dtype=np.int32)
    mpi_sdfg(inbuf=A, outbuf=B, root=root, n=size, p=commsize)
    # now B should be an array of size, containing 0
    if not np.allclose(B, np.full(size, 7, dtype=np_dtype)):
        raise (ValueError("The received values are not what I expected."))


###############################################################################

N = dace.symbol('N', dtype=dace.int64)
P = dace.symbol('P', dtype=dace.int64)


@dace.program
def dace_scatter_gather(A: dace.float32[N * P]):
    tmp = np.empty_like(A, shape=[N])
    dace.comm.Scatter(A, tmp, root=0)
    tmp[:] = np.pi
    dace.comm.Gather(tmp, A, root=0)


@pytest.mark.mpi
def test_dace_scatter_gather():
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    if commsize < 2:
        raise ValueError(
            "This test is supposed to be run with at least two processes!")
    for r in range(0, commsize):
        if r == rank:
            mpi_sdfg = dace_scatter_gather.compile()
        comm.Barrier()

    length = 128
    if rank == 0:
        A = np.full([length * commsize], np.pi, dtype=np.float32)
    else:
        A = np.random.randn(length * commsize).astype(np.float32)

    mpi_sdfg(A=A, N=length, P=commsize)

    if rank == 0:
        assert (np.allclose(
            A, np.full([length * commsize], np.pi, dtype=np.float32)))
    else:
        assert (True)

B = dace.symbol('B')

# @dace.program
# def dace_block_scatter(A: dace.int32[N, N]):
#     lA = np.empty_like(A, shape=[B, B])
#     dace.comm.BlockScatter(A, lA, [B, B], [2, 2, 2], [0, 2], [1, 0, 1])
#     return lA

@dace.program
def dace_block_scatter(A: dace.int32[N, N, N]):
    # lA = np.empty_like(A, shape=[B, 1, 2*B])
    # dace.comm.BlockScatter(A, lA, [B, 1, 2*B], [2, 4, 1, 2], [0, 1, 2], [1, 1, 1, 0])
    lA = np.empty_like(A, shape=[1, 2*B, 2])
    dace.comm.BlockScatter(A, lA, [1, 4, 2], [4, 1, 2, 2], [0, 1, 3], [1, 1, 0, 1])
    dace.comm.BlockGather(lA, A, [1, 4, 2], [4, 1, 2, 2], [0, 1, 3], [1, 1, 0, 1])
    return lA


@pytest.mark.mpi
def test_dace_block_scatter():
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    if commsize != 16:
        raise ValueError(
            "This test is supposed to be run with eight processes!")
    for r in range(commsize):
        if r == rank:
            mpi_sdfg = dace_block_scatter.compile()
        comm.Barrier()

    if rank == 0:
        A = np.arange(64, dtype=np.int32).reshape(4, 4, 4).copy()
    else:
        A = np.zeros((4, 4, 4), dtype=np.int32)

    lA = mpi_sdfg(A=A, N=4, B=2, P=commsize)

    for r in range(commsize):
        if r == rank:
            print(f"Rank {r}:")
            print(lA, flush=True)
        comm.Barrier()
    if rank == 0:
        print(f"Rank 0:")
        print(A, flush=True)


def debug(arr):
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    for i in range(size):
        if i == rank:
            print(f"Rank {rank}:")
            print(arr, flush=True)
        comm.barrier()


BS = dace.symbol('BS')
@dace.program
def soap_mm(A: dace.float32[N, N], B: dace.float32[N, N]):
    C = np.ndarray((N, N), dtype=np.float32)
    lA = np.empty_like(A, shape=[BS, BS])
    lB = np.empty_like(B, shape=[BS, BS])
    dace.comm.BlockScatter(A, lA, [BS, BS], [2, 2, 2], [0, 1], [1, 1, 0])
    debug(lA)
    dace.comm.BlockScatter(B, lB, [BS, BS], [2, 2, 2], [2, 1], [0, 1, 1])
    debug(lB)
    lC = lA @ lB
    dace.comm.BlockGather(lC, C, [BS, BS], [2, 2, 2], [0, 2], [1, 0, 1])
    return C


@pytest.mark.mpi
def test_soap_mm():
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    if commsize != 8:
        raise ValueError(
            "This test is supposed to be run with eight processes!")
    for r in range(commsize):
        if r == rank:
            mpi_sdfg = soap_mm.compile()
        comm.Barrier()

    if rank == 0:
        A = np.arange(64, dtype=np.float32).reshape(8, 8).copy()
        B = np.arange(64, dtype=np.float32).reshape(8, 8).copy()
    else:
        A = np.zeros((1, ), dtype=np.float32)
        B = np.zeros((1, ), dtype=np.float32)

    C = mpi_sdfg(A=A, B=B, N=8, BS=4, P=commsize, debug=debug, debug_0=debug)

    # for r in range(commsize):
    #     if r == rank:
    #         print(f"Rank {r}:")
    #         print(lA, flush=True)
    #     comm.Barrier()
    if rank == 0:
        print(f"Rank 0:")
        print(C, flush=True)
        print("Ref:")
        print(A @ B, flush=True)


P0, P1, P2 = (dace.symbol(s) for s in ('P0', 'P1', 'P2'))
B0, B1 = (dace.symbol(s) for s in ('B0', 'B1'))

def debug2(arr, subarr):
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        arr[:] = 0
    for i in range(size):
        if i == rank:
            print(f"Rank {rank}:")
            print(subarr, flush=True)
        comm.barrier()
    comm.barrier()

@dace.program
def pgrid(A: dace.float32[N, N]):
    mygrid = dace.comm.Cart_create([P0, P1, P2])
    scatter_grid = dace.comm.Cart_sub(mygrid, [True, True, False], 0)
    bcast_grid = dace.comm.Cart_sub(mygrid, [False, False, True])
    lA = np.empty_like(A, shape=[B0, B1])
    dace.comm.BlockScatter(A, lA, scatter_grid, bcast_grid, [0, 1])
    debug2(A, lA)
    dace.comm.BlockGather(lA, A, scatter_grid, bcast_grid, [0, 1])


def test_cart():
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    # if commsize != 8:
    #     raise ValueError(
    #         "This test is supposed to be run with eight processes!")
    for r in range(commsize):
        if r == rank:
            mpi_sdfg = pgrid.compile()
        comm.Barrier()
    if rank == 0:
        A = np.arange(64, dtype=np.float32).reshape(8, 8).copy()
    else:
        A = np.zeros((1, ), dtype=np.float32)
    mpi_sdfg(P0=2, P1=4, P2=2, N=8, B0=4, B1=2, debug2=debug2, A=A)
    if rank == 0:
        print(f"Full array = {A}")


def debug3(arra, arrb):
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    for i in range(size):
        if i == rank:
            print(f"Rank {rank}:")
            print(arra)
            print(arrb, flush=True)
        comm.barrier()
    comm.barrier()


P0a, P0b, P1a, P1b = (dace.symbol(s) for s in ('P0a', 'P0b', 'P1a', 'P1b'))
B0a, B0b, B1a, B1b = (dace.symbol(s) for s in ('B0a', 'B0b', 'B1a', 'B1b'))

@dace.program
def redistribute(A: dace.float32[N, N]):
    mygrid = dace.comm.Cart_create([P0a, P1a, P2])
    mygrid2 = dace.comm.Cart_create([P0b, P1b])
    scatter_grid = dace.comm.Cart_sub(mygrid, [True, True, False], 0)
    bcast_grid = dace.comm.Cart_sub(mygrid, [False, False, True])
    lAa = np.empty_like(A, shape=[B0a, B1a])
    sa = dace.comm.BlockScatter(A, lAa, scatter_grid, bcast_grid, [0, 1])
    lAb = np.empty_like(A, shape=[B0b, B1b])
    sb = dace.comm.Redistribute(lAa, lAb, sa, mygrid2, [0, 1])
    debug(lAa, lAb)


def test_redistribute():
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    # if commsize != 8:
    #     raise ValueError(
    #         "This test is supposed to be run with eight processes!")
    for r in range(commsize):
        if r == rank:
            mpi_sdfg = redistribute.compile()
        comm.Barrier()
    if rank == 0:
        A = np.arange(64, dtype=np.float32).reshape(8, 8).copy()
    else:
        A = np.zeros((1, ), dtype=np.float32)
    mpi_sdfg(P0a=2, P1a=2, P2=2, P0b=2, P1b=4, N=8, B0a=4, B1a=4, B0b=4, B1b=2, debug=debug3, A=A)
    # if rank == 0:
    #     print(f"Full array = {A}")


@dace.program
def redistribute2(A: dace.float32[N, N]):
    mygrid = dace.comm.Cart_create([P0a, P1a, P2])
    mygrid2 = dace.comm.Cart_create([P0b, P1b])
    scatter_grid = dace.comm.Cart_sub(mygrid, [True, True, False], 0)
    bcast_grid = dace.comm.Cart_sub(mygrid, [False, False, True])
    lAa = np.empty_like(A, shape=[B0b, B1b])
    sa = dace.comm.BlockScatter(A, lAa, mygrid2, None, [0, 1])
    lAb = np.empty_like(A, shape=[B0a, B1a])
    sb = dace.comm.Redistribute(lAa, lAb, sa, scatter_grid, [0, 1])
    debug(lAa, lAb)


def test_redistribute2():
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    # if commsize != 8:
    #     raise ValueError(
    #         "This test is supposed to be run with eight processes!")
    for r in range(commsize):
        if r == rank:
            mpi_sdfg = redistribute2.compile()
        comm.Barrier()
    if rank == 0:
        A = np.arange(64, dtype=np.float32).reshape(8, 8).copy()
    else:
        A = np.zeros((1, ), dtype=np.float32)
    mpi_sdfg(P0a=2, P1a=2, P2=2, P0b=2, P1b=4, N=8, B0a=4, B1a=4, B0b=4, B1b=2, debug=debug3, A=A)
    # if rank == 0:
    #     print(f"Full array = {A}")


@dace.program
def redistribute3(A: dace.float32[N, N]):
    mygrid = dace.comm.Cart_create([P0a, P1a, P2])
    scatter_grid = dace.comm.Cart_sub(mygrid, [True, True, False], 0)
    scatter_grid2 = dace.comm.Cart_sub(mygrid, [False, True, True], 0)
    lAa = np.empty_like(A, shape=[B0a, B1a])
    sa = dace.comm.BlockScatter(A, lAa, scatter_grid, None, [0, 1])
    lAb = np.empty_like(A, shape=[B0a, B1a])
    sb = dace.comm.Redistribute(lAa, lAb, sa, scatter_grid2, [0, 1])
    debug(lAa, lAb)


def test_redistribute3():
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    # if commsize != 8:
    #     raise ValueError(
    #         "This test is supposed to be run with eight processes!")
    for r in range(commsize):
        if r == rank:
            mpi_sdfg = redistribute3.compile()
        comm.Barrier()
    if rank == 0:
        A = np.arange(64, dtype=np.float32).reshape(8, 8).copy()
    else:
        A = np.zeros((1, ), dtype=np.float32)
    mpi_sdfg(P0a=2, P1a=2, P2=2, N=8, B0a=4, B1a=4, debug=debug3, A=A)
    # if rank == 0:
    #     print(f"Full array = {A}")


@dace.program
def redistribute4(A: dace.float32[N, N]):
    mygrid = dace.comm.Cart_create([2, 2, 2])
    mygrid2 = dace.comm.Cart_create([2, 1, 4])
    scatter_grid = dace.comm.Cart_sub(mygrid, [True, True, False], 0)
    scatter_grid2 = dace.comm.Cart_sub(mygrid2, [False, True, True], 0)
    lAa = np.empty_like(A, shape=[4, 4])
    sa = dace.comm.BlockScatter(A, lAa, scatter_grid, None, [0, 1])
    lAb = np.empty_like(A, shape=[8, 2])
    sb = dace.comm.Redistribute(lAa, lAb, sa, scatter_grid2, [0, 1])
    debug(lAa, lAb)


def test_redistribute4():
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    # if commsize != 8:
    #     raise ValueError(
    #         "This test is supposed to be run with eight processes!")
    for r in range(commsize):
        if r == rank:
            mpi_sdfg = redistribute4.compile()
        comm.Barrier()
    if rank == 0:
        A = np.arange(64, dtype=np.float32).reshape(8, 8).copy()
    else:
        A = np.zeros((1, ), dtype=np.float32)
    mpi_sdfg(N=8, debug=debug3, A=A)
    # if rank == 0:
    #     print(f"Full array = {A}")


NIa, NJa, NKa, NJb = (dace.symbol(s) for s in ('NIa', 'NJa', 'NKa', 'NJb'))
lNIa, lNJa, lNKa, lNIb, lNJb, lNKb = (dace.symbol(s) for s in ('lNIa', 'lNJa', 'lNKa', 'lNIb', 'lNJb', 'lNKb'))
PIa, PJa, PKa = (dace.symbol(s) for s in ('PIa', 'PJa', 'PKa'))
PIb, PJb, PKb = (dace.symbol(s) for s in ('PIb', 'PJb', 'PKb'))


@dace.program
def one_mm(lA: dace.float64[lNIa, lNKa], lB: dace.float64[lNKa, lNJa]):
    parent_grid = dace.comm.Cart_create([PIa, PJa, PKa])
    a_grid = dace.comm.Cart_sub(parent_grid, [False, True, False])
    b_grid = dace.comm.Cart_sub(parent_grid, [True, False, False])
    c_grid = dace.comm.Cart_sub(parent_grid, [False, False, True])
    dace.comm.Bcast(lA, 0, a_grid)
    dace.comm.Bcast(lB, 0, b_grid)
    lC = lA @ lB
    dace.comm.Reduce(lC, 'MPI_SUM', 0, c_grid)
    return lC


def test_one_mm():

    from mpi4py import MPI as MPI4PY

    world_comm = MPI4PY.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    p = int(np.cbrt(world_size))
    comm = world_comm.Create_cart(
        dims = [p, p, p],
        periods = [False,False,False],
        reorder = False)
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    mpi_sdfg = None
    if world_rank == 0:
        mpi_sdfg = one_mm.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_sdfg.apply_strict_transformations()
        mpi_func= mpi_sdfg.compile()
    world_comm.Barrier()
    if world_rank > 0:
        mpi_sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=one_mm.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=one_mm.name),
            one_mm.name))
    world_comm.Barrier()

    size =  1000
    tmp = np.arange(size * size, dtype=np.float64).reshape(size, size)
    lA = tmp * (world_rank + 1)
    lB = tmp * (world_rank + 1)
    lC = mpi_func(lNIa=size, lNJa=size, lNKa=size, PIa=p, PJa=p, PKa=p, lA=lA, lB=lB)

    if comm_size == 8:
        a_mult = np.array([[1, 2], [5, 6]])
        b_mult = np.array([[1, 3], [2, 4]])
        c_mult = a_mult @ b_mult

        idx = 0
        for r in range(comm_size):
            if r in (0, 2, 4, 6):
                if r == comm_rank and r == world_rank:
                    if not np.allclose(lC, tmp @ tmp * c_mult.flatten()[idx]):
                        print(f"""
                            Error!
                            Rank: {comm_rank}
                            multiplier = {c_mult.flatten()[idx]}
                            result = {lC[:4, :4]}
                            ref = {(tmp @ tmp * c_mult.flatten()[idx])[:4, :4]}
                        """, flush=True)
                idx += 1
            world_comm.Barrier()


@dace.program
def two_mm(lA: dace.float64[lNIa, lNKa], lB: dace.float64[lNKa, lNJa],
           lC: dace.float64[lNJa, lNJb]):
    parent_grid = dace.comm.Cart_create([PIa, PJa, PKa])
    a_grid = dace.comm.Cart_sub(parent_grid, [False, True, False])
    b_grid = dace.comm.Cart_sub(parent_grid, [True, False, False])
    tmp_grid = dace.comm.Cart_sub(parent_grid, [False, False, True])
    c_grid = b_grid  # but transposed
    d_grid = a_grid
    dace.comm.Bcast(lA, 0, a_grid)
    dace.comm.Bcast(lB, 0, b_grid)
    dace.comm.Bcast(lC, 0, c_grid)
    tmp = lA @ lB
    dace.comm.Allreduce(tmp, 'MPI_SUM', tmp_grid)
    lD = tmp @ lC
    dace.comm.Reduce(lD, 'MPI_SUM', 0, d_grid)
    return lD


@dace.program
def two_mm_simple(lA: dace.float64[lNIa, lNKa], lB: dace.float64[lNKa, lNJa],
                  lC: dace.float64[lNJa, lNJb]):
    tmp = lA @ lB
    lD = tmp @ lC
    return lD


def test_two_mm(size = 1000, p = None):

    from mpi4py import MPI as MPI4PY
    
    world_comm = MPI4PY.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    if not p:
        p = [int(np.cbrt(world_size))] * 3
    comm = world_comm.Create_cart(
        dims = p,
        periods = [False,False,False],
        reorder = False)
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    mpi_sdfg = None
    if world_rank == 0:
        mpi_sdfg = two_mm.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_sdfg.apply_strict_transformations()
        mpi_func= mpi_sdfg.compile()
    world_comm.Barrier()
    if world_rank > 0:
        mpi_sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=two_mm.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=two_mm.name),
            two_mm.name))
    world_comm.Barrier()

    if isinstance(size, list):
        SI, SJ, SK, SJb = size
        from numpy.random import default_rng
        rng = default_rng(42)
        lA = rng.random((SI, SJ))
        lB = rng.random((SK, SJ))
        lC = rng.random((SJ, SJb))
    else:
        SI, SJ, SK, SJb = [size] * 4
        tmp = np.arange(size * size, dtype=np.float64).reshape(size, size)
        lA = tmp * (world_rank + 1)
        lB = tmp * (world_rank + 1)
        # lC = tmp * (world_rank + 1)
        lC = tmp.copy()
    
    if world_rank == 0:
        print(f"Full version: {size}", flush=True)
    world_comm.Barrier()
    
    runtime = []
    for i in range(10):
        world_comm.Barrier()
        tic = time.perf_counter()
        lD = mpi_func(lNIa=SI, lNJa=SJ, lNKa=SK, lNJb=SJb, PIa=p[0], PJa=p[1], PKa=[2], lA=lA, lB=lB, lC=lC)
        world_comm.Barrier()
        toc = time.perf_counter()
        if world_rank == 0:
            duration = toc-tic
            runtime.append(duration)
            print(f"{i}-th iteration: {(toc-tic) * 1000} ms", flush=True)
    
    if world_rank == 0:
        print(f"Median: {np.median(runtime) * 1000} ms", flush=True)
    world_comm.Barrier()


    if comm_size == 8 and not isinstance(size, list):
        a_mult = np.array([[1, 2], [5, 6]])
        b_mult = np.array([[1, 3], [2, 4]])
        tmp_mult = a_mult @ b_mult
        # c_mult = np.transpose(b_mult) # difficult to test with this
        c_mult = np.array([[1, 1], [1, 1]])
        d_mult = tmp_mult @ c_mult

        idx = 0
        for r in range(comm_size):
            if r in (0, 1, 4, 5):
                if r == comm_rank and r == world_rank:
                    if not np.allclose(lD, tmp @ tmp @ tmp * d_mult.flatten()[idx]):
                        print(f"""
                            Error!
                            Rank: {comm_rank}
                            multiplier = {d_mult.flatten()[idx]}
                            result = {lD[:4, :4]}
                            ref = {(tmp @ tmp @ tmp * d_mult.flatten()[idx])[:4, :4]}
                        """, flush=True)
                idx += 1
            world_comm.Barrier()


def test_two_mm_simple(size = 1000):

    from mpi4py import MPI as MPI4PY
    
    world_comm = MPI4PY.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    mpi_sdfg = None
    if world_rank == 0:
        mpi_sdfg = two_mm_simple.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_sdfg.apply_strict_transformations()
        mpi_func= mpi_sdfg.compile()
    world_comm.Barrier()
    if world_rank > 0:
        mpi_sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=two_mm_simple.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=two_mm_simple.name),
            two_mm_simple.name))
    world_comm.Barrier()

    if isinstance(size, list):
        SI, SJ, SK, SJb = size
        from numpy.random import default_rng
        rng = default_rng(42)
        lA = rng.random((SI, SJ))
        lB = rng.random((SK, SJ))
        lC = rng.random((SJ, SJb))
    else:
        SI, SJ, SK, SJb = [size] * 4
        tmp = np.arange(size * size, dtype=np.float64).reshape(size, size)
        lA = tmp * (world_rank + 1)
        lB = tmp * (world_rank + 1)
        # lC = tmp * (world_rank + 1)
        lC = tmp.copy()

    if world_rank == 0:
        print(f"Just compute version: {size}", flush=True)
    world_comm.Barrier()
    
    runtime = []
    for i in range(10):
        world_comm.Barrier()
        tic = time.perf_counter()
        lD = mpi_func(lNIa=SI, lNJa=SJ, lNKa=SK, lNJb=SJb, lA=lA, lB=lB, lC=lC)
        world_comm.Barrier()
        toc = time.perf_counter()
        if world_rank == 0:
            duration = toc-tic
            runtime.append(duration)
            print(f"{i}-th iteration: {(toc-tic) * 1000} ms", flush=True)
    
    if world_rank == 0:
        print(f"Median: {np.median(runtime) * 1000} ms", flush=True)
    world_comm.Barrier()


@dace.program
def two_mm_redistr(lA: dace.float64[lNIa, lNKa], lB: dace.float64[lNKa, lNJa],
                   lC: dace.float64[lNKb, lNJb]):
    first_grid = dace.comm.Cart_create([PIa, PJa, PKa])
    second_grid = dace.comm.Cart_create([PIb, PJb, PKb])
    a_grid = dace.comm.Cart_sub(first_grid, [False, True, False])
    b_grid = dace.comm.Cart_sub(first_grid, [True, False, False])
    tmp1_scatter_grid = dace.comm.Cart_sub(first_grid, [True, True, False], 0)
    tmp1_grid = dace.comm.Cart_sub(first_grid, [False, False, True])
    tmp2_scatter_grid = dace.comm.Cart_sub(second_grid, [True, False, True], 0)
    tmp2_grid = dace.comm.Cart_sub(second_grid, [False, True, False])
    c_grid = dace.comm.Cart_sub(second_grid, [True, False, False])
    d_grid = dace.comm.Cart_sub(second_grid, [False, False, True])
    dace.comm.Bcast(lA, 0, a_grid)
    dace.comm.Bcast(lB, 0, b_grid)
    dace.comm.Bcast(lC, 0, c_grid)
    tmp1 = lA @ lB
    dace.comm.Reduce(tmp1, 'MPI_SUM', 0, tmp1_grid)
    s1 = dace.comm.Subarray(tmp1, tmp1_scatter_grid, [0, 1], shape=[NIa, NJa])
    tmp2 = np.ndarray([lNIb, lNKb], np.float64)
    dace.comm.Redistribute(tmp1, tmp2, s1, tmp2_scatter_grid, [0, 1])
    dace.comm.Bcast(tmp2, 0, tmp2_grid)
    lD = tmp2 @ lC
    dace.comm.Reduce(lD, 'MPI_SUM', 0, d_grid)
    return lD


def test_two_mm_redistr(size = [100, 100, 100, 2500, 200]):

    from mpi4py import MPI as MPI4PY
    
    world_comm = MPI4PY.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    p = int(np.cbrt(world_size))
    comm = world_comm.Create_cart(
        dims = [p, p, p],
        periods = [False,False,False],
        reorder = False)
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    mpi_sdfg = None
    if world_rank == 0:
        mpi_sdfg = two_mm_redistr.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_sdfg.apply_strict_transformations()
        mpi_func= mpi_sdfg.compile()
    world_comm.Barrier()
    if world_rank > 0:
        mpi_sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=two_mm_redistr.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=two_mm_redistr.name),
            two_mm_redistr.name))
    world_comm.Barrier()

    SI, SJ, SK, SJb, SKb = size
    # if size != [100, 100, 100, 2500, 200]:
    from numpy.random import default_rng
    rng = default_rng(42)
    lA = rng.random((SI, SJ))
    lB = rng.random((SK, SJ))
    lC = rng.random((SKb, SJb))
    # else:
    #     tmp = np.arange(size * size, dtype=np.float64).reshape(size, size)
    #     lA = tmp * (world_rank + 1)
    #     lB = tmp * (world_rank + 1)
    #     # lC = tmp * (world_rank + 1)
    #     lC = tmp.copy()
    
    if world_rank == 0:
        print(f"Redistributing version: {size}", flush=True)
    world_comm.Barrier()
    
    runtime = []
    for i in range(10):
        world_comm.Barrier()
        tic = time.perf_counter()
        lD = mpi_func(
            NIa=SI*2, NJa=SJ*2,
            lNIa=SI, lNJa=SJ, lNKa=SK,
            lNIb=SI*2, lNJb=SJb, lNKb=SI*2,
            PIa=p, PJa=p, PKa=p,
            PIb=1, PJb=comm_size, PKb=1,
            lA=lA, lB=lB, lC=lC)
        world_comm.Barrier()
        toc = time.perf_counter()
        if world_rank == 0:
            duration = toc-tic
            runtime.append(duration)
            print(f"{i}-th iteration: {(toc-tic) * 1000} ms", flush=True)
    
    if world_rank == 0:
        print(f"Median: {np.median(runtime) * 1000} ms", flush=True)
    world_comm.Barrier()


    # if comm_size == 8 and not isinstance(size, list):
    #     a_mult = np.array([[1, 2], [5, 6]])
    #     b_mult = np.array([[1, 3], [2, 4]])
    #     tmp_mult = a_mult @ b_mult
    #     # c_mult = np.transpose(b_mult) # difficult to test with this
    #     c_mult = np.array([[1, 1], [1, 1]])
    #     d_mult = tmp_mult @ c_mult

    #     idx = 0
    #     for r in range(comm_size):
    #         if r in (0, 1, 4, 5):
    #             if r == comm_rank and r == world_rank:
    #                 if not np.allclose(lD, tmp @ tmp @ tmp * d_mult.flatten()[idx]):
    #                     print(f"""
    #                         Error!
    #                         Rank: {comm_rank}
    #                         multiplier = {d_mult.flatten()[idx]}
    #                         result = {lD[:4, :4]}
    #                         ref = {(tmp @ tmp @ tmp * d_mult.flatten()[idx])[:4, :4]}
    #                     """, flush=True)
    #             idx += 1
    #         world_comm.Barrier()


###############################################################################

if __name__ == "__main__":
    # test_mpi("MPI", dace.float32)
    # test_mpi("MPI", dace.float64)
    # test_dace_scatter_gather()
    # test_dace_block_scatter()
    # test_soap_mm()
    # test_cart()
    # test_redistribute4()
    # test_one_mm()
    # test_two_mm(size = [6400, 7200, 4400, 4800])

    # for i in (1, 2, 4): #, 'special'):
    #     if i == 'special':
    #         size = [6400, 7200, 4400, 4800]
    #     else:
    #         size = [1000 * i] * 4
    #     test_two_mm_simple(size = size)
    #     test_two_mm(size = size)

    test_two_mm_redistr()
###############################################################################
