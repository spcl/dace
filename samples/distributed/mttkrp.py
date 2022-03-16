# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicit distributed MTTKRP sample programs. """
import dace
import numpy as np
import opt_einsum as oe
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
    8: 90,
    16: 104,
    30: 120,
    32: 120,
    64: 136
}

grid = {
    1: [1, 1, 1, 1, 1, 1],
    2: [1, 1, 2, 1, 1, 1],
    4: [1, 2, 2, 1, 1, 1],
    8: [1, 2, 3, 1, 1, 1],
    16: [1, 1, 4, 4, 1, 1],
    30: [1, 5, 6, 1, 1, 1],  # Suggested by SOAP instead of 32
    32: [1, 4, 8, 1, 1, 1],  # I made myself
    64: [1, 8, 8, 1, 1, 1]
}


subgrid_remain = {k: [True if p == 1 else False for p in v] for k, v in grid.items()}


@dace.program
def mode_0(X: dctype[S0, S1, S2, S3, S4],
           JM: dctype[S1, R],
           KM: dctype[S2, R],
           LM: dctype[S3, R],
           MM: dctype[S4, R],
           out: dctype[S0, R],
           procs: dace.constant):

    parent_grid = dace.comm.Cart_create([P0, P1, P2, P3, P4, PR])
    reduce_grid = dace.comm.Cart_sub(parent_grid, subgrid_remain[procs])

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


if __name__ == "__main__":

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in grid:
        raise ValueError("Selected number of MPI processes is not supported.")
    
    sdfg = None
    if rank == 0:
        sdfg = mode_0.to_sdfg(simplify=True, procs=size)
    func = utils.distributed_compile(sdfg, commworld)

    LS = [weak_scaling[size] // p for p in grid[size][:-1]]
    LR = 25
    pgrid = grid[size]
    remain = subgrid_remain[size]
    if rank == 0:
        print(LS, LR, pgrid, remain, flush=True)

    rng = np.random.default_rng(42)
    X = rng.random((LS[0], LS[1], LS[2], LS[3], LS[4]))
    # IM = rng.random((LS[0], LR))
    JM = rng.random((LS[1], LR))
    KM = rng.random((LS[2], LR))
    LM = rng.random((LS[3], LR))
    MM = rng.random((LS[4], LR))
    out = np.zeros((LS[0], LR))

    func(X=X, JM=JM, KM=KM, LM=LM, MM=MM, out=out, procs=size,
         S0=LS[0], S1=LS[1], S2=LS[2], S3=LS[3], S4=LS[4], R=LR,
         P0=pgrid[0], P1=pgrid[1], P2=pgrid[2], P3=pgrid[3], P4=pgrid[4], PR=pgrid[5])


