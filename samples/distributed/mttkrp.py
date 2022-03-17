# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicit distributed MTTKRP sample programs. """
import dace
import numpy as np
# import opt_einsum as oe
import timeit

from dace.sdfg import utils


# Tensor mode lengths and rank
S0, S1, S2, S3, S4, R = (dace.symbol(s) for s in ('S0', 'S1', 'S2', 'S3', 'S4', 'R'))
# Process grid lengths
P0, P1, P2, P3, P4, PR = (dace.symbol(s) for s in ('P0', 'P1', 'P2', 'P3', 'P4', 'PR'))


# Einsums
# Tensor modes = 5
# mode-0 MTTKRP
einsum_0 = 'ijklm, ja, ka, la, ma -> ia'
# mode-2 MTTKRP
einsum_2 = 'ijklm, ia, ja, la, ma -> ka'
# mode-4 MTTKRP
einsum_4 = 'ijklm, ia, ja, ka, la -> ma'


dctype = dace.float64
nptype = np.float64

weak_scaling = {
    1: 60,
    2: 68,
    4: 80,
    6: 90,
    8: 92,
    16: 104,
    30: 120,
    32: 120,
    64: 136
}

grid = {
    1: [1, 1, 1, 1, 1, 1],
    2: [1, 1, 2, 1, 1, 1],
    4: [1, 2, 2, 1, 1, 1],
    6: [1, 2, 3, 1, 1, 1],
    8: [1, 2, 4, 1, 1, 1],
    16: [1, 4, 4, 1, 1, 1],
    30: [1, 5, 6, 1, 1, 1],  # Suggested by SOAP instead of 32
    32: [1, 4, 8, 1, 1, 1],  # I made myself
    64: [1, 8, 8, 1, 1, 1]
}

grid4 = {
    1: [1, 1, 1, 1, 1, 1],
    2: [1, 1, 1, 1, 2, 1],
    4: [1, 1, 1, 2, 2, 1],
    6: [1, 1, 1, 2, 3, 1],
    8: [1, 1, 1, 2, 4, 1],
    16: [1, 1, 1, 4, 4, 1],
    30: [1, 1, 1, 5, 6, 1],  # Suggested by SOAP instead of 32
    32: [1, 1, 1, 4, 8, 1],  # I made myself
    64: [1, 1, 1, 8, 8, 1]
}


@dace.program
def mode_0(X: dctype[S0, S1, S2, S3, S4],
           JM: dctype[S1, R],
           KM: dctype[S2, R],
           LM: dctype[S3, R],
           MM: dctype[S4, R],
           out: dctype[S0, R]):

    parent_grid = dace.comm.Cart_create([P0, P1, P2, P3, P4, PR])
    reduce_grid = dace.comm.Cart_sub(parent_grid, [False, True, True, True, True, False])

    # 'la, ma -> lma'
    tmp = np.ndarray((S3, S4, R), dtype=nptype)
    for l, m, a in dace.map[0:S3, 0:S4, 0:R]:
        tmp[l , m, a] = LM[l, a] * MM[m, a]
    
    # 'ijklm, lma -> ijka'
    tmp2 = np.tensordot(X, tmp, axes=([3, 4], [0, 1]))

    # 'ja, ka -> jka'
    tmp3 = np.ndarray((S1, S2, R), dtype=nptype)
    for j, k, a in dace.map[0:S1, 0:S2, 0:R]:
        tmp3[j, k, a] = JM[j, a] * KM[k, a]

    # 'ijka, jka -> ia'
    for i, a in dace.map[0:S0, 0:R]:
        for j in range(S1):
            for k in range(S2):
                out[i, a] += tmp2[i, j, k, a] * tmp3[j, k, a]
    
    # Reduce
    dace.comm.Allreduce(out, 'MPI_SUM', grid=reduce_grid)


@dace.program
def mode_2(X: dctype[S0, S1, S2, S3, S4],
           IM: dctype[S0, R],
           JM: dctype[S1, R],
           LM: dctype[S3, R],
           MM: dctype[S4, R],
           out: dctype[S2, R]):

    parent_grid = dace.comm.Cart_create([P0, P1, P2, P3, P4, PR])
    reduce_grid = dace.comm.Cart_sub(parent_grid, [True, True, False, True, True, False])

    # 'la, ma -> lma'
    tmp = np.ndarray((S3, S4, R), dtype=nptype)
    for l, m, a in dace.map[0:S3, 0:S4, 0:R]:
        tmp[l , m, a] = LM[l, a] * MM[m, a]
    
    # 'ijklm, lma -> ijka'
    tmp2 = np.tensordot(X, tmp, axes=([3, 4], [0, 1]))

    # 'ia, ja -> ija'
    tmp3 = np.ndarray((S0, S1, R), dtype=nptype)
    for i, j, a in dace.map[0:S0, 0:S1, 0:R]:
        tmp3[i, j, a] = IM[i, a] * JM[j, a]

    # 'ijka, ija -> ka'
    for k, a in dace.map[0:S2, 0:R]:
        for i in range(S0):
            for j in range(S1):
                out[k, a] += tmp2[i, j, k, a] * tmp3[i, j, a]
    
    # Reduce
    dace.comm.Allreduce(out, 'MPI_SUM', grid=reduce_grid)


@dace.program
def mode_4(X: dctype[S0, S1, S2, S3, S4],
           IM: dctype[S0, R],
           JM: dctype[S1, R],
           KM: dctype[S2, R],
           LM: dctype[S3, R],
           out: dctype[S4, R]):

    parent_grid = dace.comm.Cart_create([P0, P1, P2, P3, P4, PR])
    reduce_grid = dace.comm.Cart_sub(parent_grid, [True, True, True, True, False, False])

    # 'ia, ja -> ija'
    tmp = np.ndarray((S0, S1, R), dtype=nptype)
    for i, j, a in dace.map[0:S0, 0:S1, 0:R]:
        tmp[i, j, a] = IM[i, a] * JM[j, a]
    
    # 'ijklm, ija -> klma'
    tmp2 = np.tensordot(X, tmp, axes=([0, 1], [0, 1]))

    # 'ka, la -> klma'
    tmp3 = np.ndarray((S2, S3, R), dtype=nptype)
    for k, l, a in dace.map[0:S2, 0:S3, 0:R]:
        tmp3[k, l, a] = KM[k, a] * LM[l, a]

    # 'klma, kla -> ma'
    for m, a in dace.map[0:S4, 0:R]:
        for k in range(S2):
            for l in range(S3):
                out[m, a] += tmp2[k, l, m, a] * tmp3[k, l, a]
    
    # Reduce
    dace.comm.Allreduce(out, 'MPI_SUM', grid=reduce_grid)


if __name__ == "__main__":

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in grid:
        raise ValueError("Selected number of MPI processes is not supported.")
    
    sdfg0, sdfg2, sdfg4 = None, None, None
    if rank == 0:
        sdfg0 = mode_0.to_sdfg(simplify=True)
        sdfg2 = mode_2.to_sdfg(simplify=True)
        sdfg4 = mode_4.to_sdfg(simplify=True)
    func0 = utils.distributed_compile(sdfg0, commworld)
    func2 = utils.distributed_compile(sdfg2, commworld)
    func4 = utils.distributed_compile(sdfg4, commworld)

    LS = [weak_scaling[size] // p for p in grid[size][:-1]]
    LR = 25
    pgrid = grid[size]
    if rank == 0:
        print(LS, LR, pgrid, flush=True)

    rng = np.random.default_rng(42)
    X = rng.random((LS[0], LS[1], LS[2], LS[3], LS[4]))
    IM = rng.random((LS[0], LR))
    JM = rng.random((LS[1], LR))
    KM = rng.random((LS[2], LR))
    LM = rng.random((LS[3], LR))
    MM = rng.random((LS[4], LR))

    out0 = np.zeros((LS[0], LR))
    out2 = np.zeros((LS[2], LR))
    out4 = np.zeros((LS[4], LR))

    # validate = True
    # if validate:

    #     func0(X=X, JM=JM, KM=KM, LM=LM, MM=MM, out=out0,
    #         S0=LS[0], S1=LS[1], S2=LS[2], S3=LS[3], S4=LS[4], R=LR,
    #         P0=pgrid[0], P1=pgrid[1], P2=pgrid[2], P3=pgrid[3], P4=pgrid[4], PR=pgrid[5])
        
    #     func2(X=X, JM=JM, KM=KM, LM=LM, MM=MM, out=out2,
    #         S0=LS[0], S1=LS[1], S2=LS[2], S3=LS[3], S4=LS[4], R=LR,
    #         P0=pgrid[0], P1=pgrid[1], P2=pgrid[2], P3=pgrid[3], P4=pgrid[4], PR=pgrid[5])

    #     func4(X=X, JM=JM, KM=KM, LM=LM, MM=MM, out=out4,
    #         S0=LS[0], S1=LS[1], S2=LS[2], S3=LS[3], S4=LS[4], R=LR,
    #         P0=pgrid[0], P1=pgrid[1], P2=pgrid[2], P3=pgrid[3], P4=pgrid[4], PR=pgrid[5])
    

    runtimes = timeit.repeat(
        """func0(X=X, JM=JM, KM=KM, LM=LM, MM=MM, out=out0,
                 S0=LS[0], S1=LS[1], S2=LS[2], S3=LS[3], S4=LS[4], R=LR,
                 P0=pgrid[0], P1=pgrid[1], P2=pgrid[2], P3=pgrid[3], P4=pgrid[4], PR=pgrid[5]); commworld.Barrier()
        """,
        setup="commworld.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )

    if rank == 0:
        print(f"Mode-0 median runtime: {np.median(runtimes)} seconds")

    runtimes = timeit.repeat(
        """func2(X=X, IM=IM, JM=JM, LM=LM, MM=MM, out=out2,
                 S0=LS[0], S1=LS[1], S2=LS[2], S3=LS[3], S4=LS[4], R=LR,
                 P0=pgrid[0], P1=pgrid[1], P2=pgrid[2], P3=pgrid[3], P4=pgrid[4], PR=pgrid[5]); commworld.Barrier()
        """,
        setup="commworld.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )

    if rank == 0:
        print(f"Mode-2 median runtime: {np.median(runtimes)} seconds")

    LS = [weak_scaling[size] // p for p in grid4[size][:-1]]
    LR = 25
    pgrid = grid4[size]
    if rank == 0:
        print(LS, LR, pgrid, flush=True)

    runtimes = timeit.repeat(
        """func4(X=X, IM=IM, JM=JM, KM=KM, LM=LM, out=out4,
                 S0=LS[0], S1=LS[1], S2=LS[2], S3=LS[3], S4=LS[4], R=LR,
                 P0=pgrid[0], P1=pgrid[1], P2=pgrid[2], P3=pgrid[3], P4=pgrid[4], PR=pgrid[5]); commworld.Barrier()
        """,
        setup="commworld.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )

    if rank == 0:
        print(f"Mode-4 median runtime: {np.median(runtimes)} seconds")


