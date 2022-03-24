# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicit distributed TTMc sample programs. """
import dace
import numpy as np
import timeit

from dace.sdfg import utils


# Tensor mode lengths
S0, S1, S2, S3, S4 = (dace.symbol(s) for s in ('S0', 'S1', 'S2', 'S3', 'S4'))
S0G1, S1G1, S2G1, S3G1, S4G1 = (dace.symbol(s) for s in ('S0G1', 'S1G1', 'S2G1', 'S3G1', 'S4G1'))
S0G2, S1G2, S2G2, S3G2, S4G2 = (dace.symbol(s) for s in ('S0G2', 'S1G2', 'S2G2', 'S3G2', 'S4G2'))
S0G3, S1G3, S2G3, S3G3, S4G3 = (dace.symbol(s) for s in ('S0G3', 'S1G3', 'S2G3', 'S3G3', 'S4G3'))
S0G4, S1G4, S2G4, S3G4, S4G4 = (dace.symbol(s) for s in ('S0G4', 'S1G4', 'S2G4', 'S3G4', 'S4G4'))
# Number(s) of tensor eigenvectors
R0, R1, R2, R3, R4 = (dace.symbol(s) for s in ('R0', 'R1', 'R2', 'R3', 'R4'))
R0G1, R1G1, R2G1, R3G1, R4G1 = (dace.symbol(s) for s in ('R0G1', 'R1G1', 'R2G1', 'R3G1', 'R4G1'))
R0G2, R1G2, R2G2, R3G2, R4G2 = (dace.symbol(s) for s in ('R0G2', 'R1G2', 'R2G2', 'R3G2', 'R4G2'))
R0G3, R1G3, R2G3, R3G3, R4G3 = (dace.symbol(s) for s in ('R0G3', 'R1G3', 'R2G3', 'R3G3', 'R4G3'))
R0G4, R1G4, R2G4, R3G4, R4G4 = (dace.symbol(s) for s in ('R0G4', 'R1G4', 'R2G4', 'R3G4', 'R4G4'))

# Process grid lengths for tensor modes
P0, P1, P2, P3, P4 = (dace.symbol(s) for s in ('P0', 'P1', 'P2', 'P3', 'P4'))
# Process grid lengths for tensor eigenvectors
PR0, PR1, PR2, PR3, PR4 = (dace.symbol(s) for s in ('PR0', 'PR1', 'PR2', 'PR3', 'PR4'))


# Einsums
# Tensor modes = 5
# mode-0 TTMc
einsum_0 = 'ijklm, jb, kc, ld, me -> ibcde'
# mode-2 MTTKRP
einsum_2 = 'ijklm, ia, jb, ld, me -> abkde'
# mode-4 MTTKRP
einsum_4 = 'ijklm, ia, jb, kc, ld -> abcdm'


dctype = dace.float64
nptype = np.float64


weak_scaling = {
    1: 60,
    2: 70,
    4: 82,
    6: 90,
    8: 96,
    # 12: 108,
    16: 112,
    # 27: 123,
    32: 128,
    64: 136,
    # 125: 175,
    128: 176,
    256: 200,
    512: 232
}

grid_ijklmbcde = {
    #     [i, j, k, l, m, b, c, d, e]
    1:    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    2:    [1, 1, 1, 1, 2, 1, 1, 1, 1],
    4:    [1, 1, 1, 2, 2, 1, 1, 1, 1],
    8:    [1, 1, 2, 2, 2, 1, 1, 1, 1],
    16:   [1, 1, 2, 2, 2, 1, 1, 1, 2],
    32:   [1, 1, 2, 2, 2, 1, 1, 2, 2],
    64:   [1, 1, 2, 2, 2, 1, 2, 2, 2],
    128:  [1, 1, 2, 2, 4, 1, 2, 2, 2],
    256:  [1, 1, 2, 4, 4, 1, 2, 2, 2],
    512:  [1, 1, 4, 4, 4, 1, 2, 2, 2]
}

grid_ijklme = {
    #     [i, j, k, l, m, e]
    1:    [1, 1, 1, 1, 1, 1],
    2:    [1, 1, 1, 2, 1, 1],
    4:    [1, 1, 2, 2, 1, 1],
    8:    [1, 2, 2, 2, 1, 1],
    12:   [1, 2, 2, 3, 1, 1],
    16:   [1, 2, 2, 4, 1, 1],
    27:   [1, 3, 3, 3, 1, 1],
    32:   [1, 2, 4, 4, 1, 1],
    64:   [1, 4, 4, 4, 1, 1],
    125:  [1, 5, 5, 5, 1, 1],
    128:  [1, 4, 4, 8, 1, 1],
    256:  [1, 4, 8, 8, 1, 1],
    512:  [1, 8, 8, 8, 1, 1]
}

grid_ijklde = {
    #     [i, j, k, l, d, e]
    1:    [1, 1, 1, 1, 1, 1],
    2:    [1, 1, 2, 1, 1, 1],
    4:    [1, 2, 2, 1, 1, 1],
    8:    [2, 2, 2, 1, 1, 1],
    12:   [2, 2, 3, 1, 1, 1],
    16:   [2, 2, 4, 1, 1, 1],
    27:   [3, 3, 3, 1, 1, 1],
    32:   [2, 4, 4, 1, 1, 1],
    64:   [4, 4, 4, 1, 1, 1],
    125:  [5, 5, 5, 1, 1, 1],
    128:  [4, 4, 8, 1, 1, 1],
    256:  [4, 8, 8, 1, 1, 1],
    512:  [8, 8, 8, 1, 1, 1]
}

grid_ijkcde = {
    #     [i, j, k , c, d, e]
    1:    [1, 1, 1, 1, 1, 1],
    2:    [1, 2, 1, 1, 1, 1],
    4:    [2, 2, 1, 1, 1, 1],
    8:    [2, 2, 1, 1, 1, 2],
    12:   [2, 3, 1, 1, 1, 2],
    16:   [2, 4, 1, 1, 1, 2],
    27:   [3, 3, 1, 1, 1, 3],
    32:   [4, 4, 1, 1, 1, 2],
    64:   [4, 4, 1, 1, 1, 4],
    125:  [5, 5, 1, 1, 1, 5],
    128:  [4, 8, 1, 1, 1, 4],
    256:  [8, 8, 1, 1, 1, 4],
    512:  [8, 8, 1, 1, 1, 8]
}

grid_ijbcde = {
    #     [i, j, b, c, d, e]
    1:    [1, 1, 1, 1, 1, 1],
    2:    [2, 1, 1, 1, 1, 1],
    4:    [2, 1, 1, 1, 1, 2],
    8:    [2, 1, 1, 1, 2, 2],
    12:   [3, 1, 1, 1, 2, 2],
    16:   [4, 1, 1, 1, 2, 2],
    27:   [3, 1, 1, 1, 3, 3],
    32:   [4, 1, 1, 1, 2, 4],
    64:   [4, 1, 1, 1, 4, 4],
    125:  [5, 1, 1, 1, 5, 5],
    128:  [8, 1, 1, 1, 4, 4],
    256:  [8, 1, 1, 1, 4, 8],
    512:  [8, 1, 1, 1, 8, 8]
}


@dace.program
def mode_0_shared(X: dace.float64[S0, S1, S2, S3, S4],
                  JM: dace.float64[S1, R1],
                  KM: dace.float64[S2, R2],
                  LM: dace.float64[S3, R3],
                  MM: dace.float64[S4, R4]) -> dace.float64[S0, R1, R2, R3, R4]:
    tmp = np.tensordot(X, MM, axes=([4], [0]))
    tmp2 = np.transpose(tmp, axes=[4, 0, 1, 2, 3])
    tmp3 = np.tensordot(tmp2, LM, axes=([4], [0]))
    tmp4 = np.transpose(tmp3, axes=[4, 0, 1, 2, 3])
    tmp5 = np.tensordot(tmp4, KM, axes=([4], [0]))
    tmp6 = np.transpose(tmp5, axes=[4, 0, 1, 2, 3])
    tmp7 = np.tensordot(tmp6, JM, axes=([4], [0]))
    return np.transpose(tmp7, axes=[3, 4, 0, 1, 2])


@dace.program
def mode_0_four_grids(X: dace.float64[S0G1, S1G1, S2G1, S3G1, S4G1],
                      JM: dace.float64[S1G4, R1G4],
                      KM: dace.float64[S2G3, R2G3],
                      LM: dace.float64[S3G2, R3G2],
                      MM: dace.float64[S4G1, R4G1],
                      procs: dace.constant) -> dace.float64[S0G4, R1G4, R2G4, R3G4, R4G4]:

    # grid: ijklme
    grid1 = dace.comm.Cart_create(grid_ijklme[procs])
    # out: ijkle
    grid1_out_gather = dace.comm.Cart_sub(grid1, [True, True, True, True, False, True], exact_grid=0)
    grid1_out_reduce = dace.comm.Cart_sub(grid1, [False, False, False, False, True, False])
    grid1_out_subarray = dace.comm.Subarray((S0, S1, S2, S3, R4), (S0G1, S1G1, S2G1, S3G1, R4G1), dace.float64, process_grid=grid1_out_gather)

    # grid: ijklde
    grid2 = dace.comm.Cart_create(grid_ijklde[procs])
    # in: ijkle
    grid2_in_scatter = dace.comm.Cart_sub(grid2, [True, True, True, True, False, True], exact_grid=0)
    grid2_in_bcast = dace.comm.Cart_sub(grid2, [False, False, False, False, True, False])
    grid2_in_subarray = dace.comm.Subarray((S0, S1, S2, S3, R4), (S0G2, S1G2, S2G2, S3G2, R4G2), dace.float64, process_grid=grid2_in_scatter)
    # out: eijkd (ijkde)
    grid2_out_gather = dace.comm.Cart_sub(grid2, [True, True, True, False, True, True], exact_grid=0)
    grid2_out_reduce = dace.comm.Cart_sub(grid2, [False, False, False, True, False, False])
    grid2_out_subarray = dace.comm.Subarray((R4, S0, S1, S2, R3), (R4G2, S0G2, S1G2, S2G2, R3G2), dace.float64, process_grid=grid2_out_gather, correspondence=(4, 0, 1, 2, 3))

    # grid: ijkcde
    grid3 = dace.comm.Cart_create(grid_ijkcde[procs])
    # in: eijkd (ijkde)
    grid3_in_scatter = dace.comm.Cart_sub(grid3, [True, True, True, False, True, True], exact_grid=0)
    grid3_in_bcast = dace.comm.Cart_sub(grid3, [False, False, False, True, False, False])
    grid3_in_subarray = dace.comm.Subarray((R4, S0, S1, S2, R3), (R4G3, S0G3, S1G3, S2G3, R3G3), dace.float64, process_grid=grid3_in_scatter, correspondence=(4, 0, 1, 2, 3))
    # out: deijc (ijcde)
    grid3_out_gather = dace.comm.Cart_sub(grid3, [True, True, False, True, True, True], exact_grid=0)
    grid3_out_reduce = dace.comm.Cart_sub(grid3, [False, False, True, False, False, False])
    grid3_out_subarray = dace.comm.Subarray((R3, R4, S0, S1, R2), (R3G3, R4G3, S0G3, S1G3, R2G3), dace.float64, process_grid=grid3_out_gather, correspondence=(3, 4, 0, 1, 2))

    # grid: ijbcde
    grid4 = dace.comm.Cart_create(grid_ijbcde[procs])
    # in: deijc (ijcde)
    grid4_in_scatter = dace.comm.Cart_sub(grid4, [True, True, False, True, True, True], exact_grid=0)
    grid4_in_bcast = dace.comm.Cart_sub(grid4, [False, False, True, False, False, False])
    grid4_in_subarray = dace.comm.Subarray((R3, R4, S0, S1, R2), (R3G4, R4G4, S0G4, S1G4, R2G4), dace.float64, process_grid=grid4_in_scatter, correspondence=(3, 4, 0, 1, 2))
    # out: cdeib (ibcde)
    grid4_out_gather = dace.comm.Cart_sub(grid4, [True, False, True, True, True, True], exact_grid=0)
    grid4_out_reduce = dace.comm.Cart_sub(grid4, [False, True, False, False, False, False])
    # grid4_out_subarray = dace.comm.Subarray((R2, R3, R4, S0, R1), (R2G4, R3G4, R4G4, S0G4, R1G4), dace.float64, process_grid=grid4_out_gather, correspondence=(2, 3, 4, 0, 1))
    
    grid1_out = np.tensordot(X, MM, axes=([4], [0]))     #ijkle
    dace.comm.Reduce(grid1_out, 'MPI_SUM', grid=grid1_out_reduce)
    grid2_in = np.empty_like(grid1_out, shape=(S0G2, S1G2, S2G2, S3G2, R4G2))  # Need a nice way to infer the shape here
    dace.comm.Redistribute(grid1_out, grid1_out_subarray, grid2_in, grid2_in_subarray)
    dace.comm.Bcast(grid2_in, grid=grid2_in_bcast)

    tmp = np.transpose(grid2_in, axes=[4, 0, 1, 2, 3])   # eijkl
    grid2_out = np.tensordot(tmp, LM, axes=([4], [0]))   # eijkd
    dace.comm.Reduce(grid2_out, 'MPI_SUM', grid=grid2_out_reduce)
    grid3_in = np.empty_like(grid2_out, shape=(R4G3, S0G3, S1G3, S2G3, R3G3))  # Need a nice way to infer the shape here
    dace.comm.Redistribute(grid2_out, grid2_out_subarray, grid3_in, grid3_in_subarray)
    dace.comm.Bcast(grid3_in, grid=grid3_in_bcast)

    tmp2 = np.transpose(grid3_in, axes=[4, 0, 1, 2, 3])  # deijk
    grid3_out = np.tensordot(tmp2, KM, axes=([4], [0]))  # deijc
    dace.comm.Reduce(grid3_out, 'MPI_SUM', grid=grid3_out_reduce)
    grid4_in = np.empty_like(grid3_out, shape=(R3G4, R4G4, S0G4, S1G4, R2G4))  # Need a nice way to infer the shape here
    dace.comm.Redistribute(grid3_out, grid3_out_subarray, grid4_in, grid4_in_subarray)
    dace.comm.Bcast(grid4_in, grid=grid4_in_bcast)

    tmp3 = np.transpose(grid4_in, axes=[4, 0, 1, 2, 3])  # cdeij
    grid4_out = np.tensordot(tmp3, JM, axes=([4], [0]))  # cdeib
    dace.comm.Allreduce(grid4_out, 'MPI_SUM', grid=grid4_out_reduce)

    return np.transpose(grid4_out, axes=[3, 4, 0, 1, 2])  # ibcde


@dace.program
def mode_0_four_grids_compute(X: dace.float64[S0G1, S1G1, S2G1, S3G1, S4G1],
                              JM: dace.float64[S1G4, R1G4],
                              KM: dace.float64[S2G3, R2G3],
                              LM: dace.float64[S3G2, R3G2],
                              MM: dace.float64[S4G1, R4G1],
                              procs: dace.constant) -> dace.float64[S0G4, R1G4, R2G4, R3G4, R4G4]:

    grid1_out = np.tensordot(X, MM, axes=([4], [0]))     #ijkle
    grid2_in = np.empty_like(grid1_out, shape=(S0G2, S1G2, S2G2, S3G2, R4G2))  # Need a nice way to infer the shape here

    tmp = np.transpose(grid2_in, axes=[4, 0, 1, 2, 3])   # eijkl
    grid2_out = np.tensordot(tmp, LM, axes=([4], [0]))   # eijkd
    grid3_in = np.empty_like(grid2_out, shape=(R4G3, S0G3, S1G3, S2G3, R3G3))  # Need a nice way to infer the shape here

    tmp2 = np.transpose(grid3_in, axes=[4, 0, 1, 2, 3])  # deijk
    grid3_out = np.tensordot(tmp2, KM, axes=([4], [0]))  # deijc
    grid4_in = np.empty_like(grid3_out, shape=(R3G4, R4G4, S0G4, S1G4, R2G4))  # Need a nice way to infer the shape here

    tmp3 = np.transpose(grid4_in, axes=[4, 0, 1, 2, 3])  # cdeij
    grid4_out = np.tensordot(tmp3, JM, axes=([4], [0]))  # cdeib

    return np.transpose(grid4_out, axes=[3, 4, 0, 1, 2])  # ibcde


@dace.program
def mode_0_ijkcde_grid(X: dace.float64[S0, S1, S2, S3, S4],
                       JM: dace.float64[S1, R1],
                       KM: dace.float64[S2, R2],
                       LM: dace.float64[S3, R3],
                       MM: dace.float64[S4, R4],
                       procs: dace.constant) -> dace.float64[S0, R1, R2, R3, R4]:

    grid = dace.comm.Cart_create(grid_ijkcde[procs])

    out1_reduce = dace.comm.Cart_sub(grid, [False, False, False, False, True, False])
    out2_reduce = dace.comm.Cart_sub(grid, [False, False, False, True, False, False])
    out3_reduce = dace.comm.Cart_sub(grid, [False, False, True, False, False, False])
    out4_reduce = dace.comm.Cart_sub(grid, [False, True, False, False, False, False])
    
    out1 = np.tensordot(X, MM, axes=([4], [0]))      # ijkle
    dace.comm.Allreduce(out1, 'MPI_SUM', grid=out1_reduce)

    tmp = np.transpose(out1, axes=[4, 0, 1, 2, 3])   # eijkl
    out2 = np.tensordot(tmp, LM, axes=([4], [0]))    # eijkd
    dace.comm.Allreduce(out2, 'MPI_SUM', grid=out2_reduce)

    tmp2 = np.transpose(out2, axes=[4, 0, 1, 2, 3])  # deijk
    out3 = np.tensordot(tmp2, KM, axes=([4], [0]))   # deijc
    dace.comm.Allreduce(out3, 'MPI_SUM', grid=out3_reduce)

    tmp3 = np.transpose(out3, axes=[4, 0, 1, 2, 3])  # cdeij
    out4 = np.tensordot(tmp3, JM, axes=([4], [0]))   # cdeib
    dace.comm.Allreduce(out4, 'MPI_SUM', grid=out4_reduce)

    return np.transpose(out4, axes=[3, 4, 0, 1, 2])  # ibcde


@dace.program
def mode_0_ijkcde_grid_compute(X: dace.float64[S0, S1, S2, S3, S4],
                               JM: dace.float64[S1, R1],
                               KM: dace.float64[S2, R2],
                               LM: dace.float64[S3, R3],
                               MM: dace.float64[S4, R4],
                               procs: dace.constant) -> dace.float64[S0, R1, R2, R3, R4]:
    
    out1 = np.tensordot(X, MM, axes=([4], [0]))      # ijkle

    tmp = np.transpose(out1, axes=[4, 0, 1, 2, 3])   # eijkl
    out2 = np.tensordot(tmp, LM, axes=([4], [0]))    # eijkd

    tmp2 = np.transpose(out2, axes=[4, 0, 1, 2, 3])  # deijk
    out3 = np.tensordot(tmp2, KM, axes=([4], [0]))   # deijc

    tmp3 = np.transpose(out3, axes=[4, 0, 1, 2, 3])  # cdeij
    out4 = np.tensordot(tmp3, JM, axes=([4], [0]))   # cdeib

    return np.transpose(out4, axes=[3, 4, 0, 1, 2])  # ibcde


@dace.program
def mode_0_unified_grid(X: dace.float64[S0, S1, S2, S3, S4],
                        JM: dace.float64[S1, R1],
                        KM: dace.float64[S2, R2],
                        LM: dace.float64[S3, R3],
                        MM: dace.float64[S4, R4],
                        procs: dace.constant) -> dace.float64[S0, R1, R2, R3, R4]:

    grid = dace.comm.Cart_create(grid_ijklmbcde[procs])
    out_reduce = dace.comm.Cart_sub(grid, [False, True, True, True, True, False, False, False, False])
    
    out1 = np.tensordot(X, MM, axes=([4], [0]))      # ijkle

    tmp = np.transpose(out1, axes=[4, 0, 1, 2, 3])   # eijkl
    out2 = np.tensordot(tmp, LM, axes=([4], [0]))    # eijkd

    tmp2 = np.transpose(out2, axes=[4, 0, 1, 2, 3])  # deijk
    out3 = np.tensordot(tmp2, KM, axes=([4], [0]))   # deijc

    tmp3 = np.transpose(out3, axes=[4, 0, 1, 2, 3])  # cdeij
    out4 = np.tensordot(tmp3, JM, axes=([4], [0]))   # cdeib
    dace.comm.Allreduce(out4, 'MPI_SUM', grid=out_reduce)

    return np.transpose(out4, axes=[3, 4, 0, 1, 2])  # ibcde


@dace.program
def mode_0_unified_grid_compute(X: dace.float64[S0, S1, S2, S3, S4],
                                JM: dace.float64[S1, R1],
                                KM: dace.float64[S2, R2],
                                LM: dace.float64[S3, R3],
                                MM: dace.float64[S4, R4],
                                procs: dace.constant) -> dace.float64[S0, R1, R2, R3, R4]:

    out1 = np.tensordot(X, MM, axes=([4], [0]))      # ijkle

    tmp = np.transpose(out1, axes=[4, 0, 1, 2, 3])   # eijkl
    out2 = np.tensordot(tmp, LM, axes=([4], [0]))    # eijkd

    tmp2 = np.transpose(out2, axes=[4, 0, 1, 2, 3])  # deijk
    out3 = np.tensordot(tmp2, KM, axes=([4], [0]))   # deijc

    tmp3 = np.transpose(out3, axes=[4, 0, 1, 2, 3])  # cdeij
    out4 = np.tensordot(tmp3, JM, axes=([4], [0]))   # cdeib

    return np.transpose(out4, axes=[3, 4, 0, 1, 2])  # ibcde


if __name__ == "__main__":

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in weak_scaling:
        raise ValueError("Selected number of MPI processes is not supported.")
    
    sdfg0, sdfg1, sdfg2, sdfg3, sdfg4, sdfg5, sdfg6 = None, None, None, None, None, None, None
    if rank == 0:
        sdfg0 = mode_0_four_grids.to_sdfg(simplify=True, procs=size)
        sdfg1 = mode_0_ijkcde_grid.to_sdfg(simplify=True, procs=size)
        sdfg2 = mode_0_unified_grid.to_sdfg(simplify=True, procs=size)
        sdfg3 = mode_0_four_grids_compute.to_sdfg(simplify=True, procs=size)
        sdfg4 = mode_0_ijkcde_grid_compute.to_sdfg(simplify=True, procs=size)
        sdfg5 = mode_0_unified_grid_compute.to_sdfg(simplify=True, procs=size)
        if size == 1:
            sdfg6 = mode_0_shared.to_sdfg(simplify=True)
            func6 = sdfg6.compile()
    func0 = utils.distributed_compile(sdfg0, commworld)
    func1 = utils.distributed_compile(sdfg1, commworld)
    func2 = utils.distributed_compile(sdfg2, commworld)
    func3 = utils.distributed_compile(sdfg3, commworld)
    func4 = utils.distributed_compile(sdfg4, commworld)
    func5 = utils.distributed_compile(sdfg5, commworld)

    # S = np.int32(weak_scaling[size])
    S = np.int32(24)
    R = np.int32(24)

    rng = np.random.default_rng(42)

    ##### Shared Memory #####

    if size == 1:

        print(f"##### Shared Memory Execution #####\nSizes: {[S]*5}, {[R]*5}""", flush=True)

        X = rng.random((S, S, S, S, S))
        JM = rng.random((S, R))
        KM = rng.random((S, R))
        LM = rng.random((S, R))
        MM = rng.random((S, R))
        IM = rng.random((S, R))

        runtimes = timeit.repeat(
            """func6(X=X, JM=JM, KM=KM, LM=LM, MM=MM,
                    S0=S, S1=S, S2=S, S3=S, S4=S, R0=R, R1=R, R2=R, R3=R, R4=R)
            """,
            setup="",
            repeat=10,
            number=1,
            globals=locals()
        )

        print(f"Median runtime: {np.median(runtimes)} seconds")
    
    #######################

    SGU = [S // np.int32(p) for p in grid_ijklmbcde[size][:5]]
    RGU = [R] + [R // np.int32(p) for p in grid_ijklmbcde[size][5:]]

    SG1 = [S // np.int32(p) for p in grid_ijklme[size][:-1]] + [S]
    RG1 = [R, R, R, R] + [R // np.int32(p) for p in grid_ijklme[size][-1:]]
    SG2 = [S // np.int32(p) for p in grid_ijklde[size][:-2]] + [S, S]
    RG2 = [R, R, R] + [R // np.int32(p) for p in grid_ijklde[size][-2:]]
    SG3 = [S // np.int32(p) for p in grid_ijkcde[size][:-3]] + [S, S, S]
    RG3 = [R, R] + [R // np.int32(p) for p in grid_ijkcde[size][-3:]]
    SG4 = [S // np.int32(p) for p in grid_ijbcde[size][:-4]]  + [S, S, S, S]
    RG4 = [R] + [R // np.int32(p) for p in grid_ijbcde[size][-4:]]

    ##### One Grid per TensorDot #####

    if rank == 0:
        print(
            f"""
##### One Grid per TensorDot #####
ijklm, me -> ijkle: local sizes {SG1}, {RG1}, grid {grid_ijklme[size]}
ijkle, ld -> ijkde: local sizes {SG2}, {RG2}, grid {grid_ijklde[size]}
ijkde, kc -> ijcde: local sizes {SG3}, {RG3}, grid {grid_ijkcde[size]}
ijcde, jb -> ibcde: local sizes {SG4}, {RG4}, grid {grid_ijbcde[size]}""", flush=True
        )

    X = rng.random((SG1[0], SG1[1], SG1[2], SG1[3], SG1[4]))
    JM = rng.random((SG4[1], RG4[1]))
    KM = rng.random((SG3[2], RG3[2]))
    LM = rng.random((SG2[3], RG2[3]))
    MM = rng.random((SG1[4], RG1[4]))

    runtimes = timeit.repeat(
        """func0(X=X, JM=JM, KM=KM, LM=LM, MM=MM, procs=size,
                S0=S, S1=S, S2=S, S3=S, S4=S, R0=R, R1=R, R2=R, R3=R, R4=R,
                S0G1=SG1[0], S1G1=SG1[1], S2G1=SG1[2], S3G1=SG1[3], S4G1=SG1[4],
                S0G2=SG2[0], S1G2=SG2[1], S2G2=SG2[2], S3G2=SG2[3], S4G2=SG2[4],
                S0G3=SG3[0], S1G3=SG3[1], S2G3=SG3[2], S3G3=SG3[3], S4G3=SG3[4],
                S0G4=SG4[0], S1G4=SG4[1], S2G4=SG4[2], S3G4=SG4[3], S4G4=SG4[4],
                R0G1=RG1[0], R1G1=RG1[1], R2G1=RG1[2], R3G1=RG1[3], R4G1=RG1[4],
                R0G2=RG2[0], R1G2=RG2[1], R2G2=RG2[2], R3G2=RG2[3], R4G2=RG2[4],
                R0G3=RG3[0], R1G3=RG3[1], R2G3=RG3[2], R3G3=RG3[3], R4G3=RG3[4],
                R0G4=RG4[0], R1G4=RG4[1], R2G4=RG4[2], R3G4=RG4[3], R4G4=RG4[4]); commworld.Barrier()
        """,
        setup="commworld.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )

    if rank == 0:
        print(f"Median total runtime: {np.median(runtimes)} seconds", flush=True)

        runtimes = timeit.repeat(
            """func3(X=X, JM=JM, KM=KM, LM=LM, MM=MM, procs=size,
                     S0=S, S1=S, S2=S, S3=S, S4=S, R0=R, R1=R, R2=R, R3=R, R4=R,
                     S0G1=SG1[0], S1G1=SG1[1], S2G1=SG1[2], S3G1=SG1[3], S4G1=SG1[4],
                     S0G2=SG2[0], S1G2=SG2[1], S2G2=SG2[2], S3G2=SG2[3], S4G2=SG2[4],
                     S0G3=SG3[0], S1G3=SG3[1], S2G3=SG3[2], S3G3=SG3[3], S4G3=SG3[4],
                     S0G4=SG4[0], S1G4=SG4[1], S2G4=SG4[2], S3G4=SG4[3], S4G4=SG4[4],
                     R0G1=RG1[0], R1G1=RG1[1], R2G1=RG1[2], R3G1=RG1[3], R4G1=RG1[4],
                     R0G2=RG2[0], R1G2=RG2[1], R2G2=RG2[2], R3G2=RG2[3], R4G2=RG2[4],
                     R0G3=RG3[0], R1G3=RG3[1], R2G3=RG3[2], R3G3=RG3[3], R4G3=RG3[4],
                     R0G4=RG4[0], R1G4=RG4[1], R2G4=RG4[2], R3G4=RG4[3], R4G4=RG4[4])
            """,
            setup="",
            repeat=10,
            number=1,
            globals=locals()
        )

        print(f"Median compute runtime: {np.median(runtimes)} seconds\n", flush=True)

     ###### Intersection Grid #####

    if rank == 0:
        print(f"##### Intersection Grid #####\nLocal Sizes: {SG3}, {RG3}\nGrid: {grid_ijkcde[size]}""", flush=True)   
    
    X = rng.random((SG3[0], SG3[1], SG3[2], SG3[3], SG3[4]))
    JM = rng.random((SG3[1], RG3[1]))
    KM = rng.random((SG3[2], RG3[2]))
    LM = rng.random((SG3[3], RG3[3]))
    MM = rng.random((SG3[4], RG3[4]))

    runtimes = timeit.repeat(
        """func1(X=X, JM=JM, KM=KM, LM=LM, MM=MM, procs=size,
                 S0=SG3[0], S1=SG3[1], S2=SG3[2], S3=SG3[3], S4=SG3[4],
                 R0=RG3[0], R1=RG3[1], R2=RG3[2], R3=RG3[3], R4=RG3[4]); commworld.Barrier()
        """,
        setup="commworld.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )

    if rank == 0:
        print(f"Median total runtime: {np.median(runtimes)} seconds", flush=True)

        runtimes = timeit.repeat(
            """func4(X=X, JM=JM, KM=KM, LM=LM, MM=MM, procs=size,
                     S0=SG3[0], S1=SG3[1], S2=SG3[2], S3=SG3[3], S4=SG3[4],
                     R0=RG3[0], R1=RG3[1], R2=RG3[2], R3=RG3[3], R4=RG3[4])
            """,
            setup="",
            repeat=10,
            number=1,
            globals=locals()
        )

        print(f"Median compute runtime: {np.median(runtimes)} seconds\n", flush=True)

    ###### Unified Grid #####

    if rank == 0:
        print(f"##### Unified Grid #####\nLocal Sizes: {SGU}, {RGU}\nGrid: {grid_ijklmbcde[size]}""", flush=True)

    X = rng.random((SGU[0], SGU[1], SGU[2], SGU[3], SGU[4]))
    JM = rng.random((SGU[1], RGU[1]))
    KM = rng.random((SGU[2], RGU[2]))
    LM = rng.random((SGU[3], RGU[3]))
    MM = rng.random((SGU[4], RGU[4]))

    runtimes = timeit.repeat(
        """func1(X=X, JM=JM, KM=KM, LM=LM, MM=MM, procs=size,
                 S0=SGU[0], S1=SGU[1], S2=SGU[2], S3=SGU[3], S4=SGU[4],
                 R0=RGU[0], R1=RGU[1], R2=RGU[2], R3=RGU[3], R4=RGU[4]); commworld.Barrier()
        """,
        setup="commworld.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )

    if rank == 0:
        print(f"Median total runtime: {np.median(runtimes)} seconds", flush=True)

        runtimes = timeit.repeat(
            """func5(X=X, JM=JM, KM=KM, LM=LM, MM=MM, procs=size,
                     S0=SGU[0], S1=SGU[1], S2=SGU[2], S3=SGU[3], S4=SGU[4],
                     R0=RGU[0], R1=RGU[1], R2=RGU[2], R3=RGU[3], R4=RGU[4])
            """,
            setup="",
            repeat=10,
            number=1,
            globals=locals()
        )

        print(f"Median compute runtime: {np.median(runtimes)} seconds\n", flush=True)
    
    if rank == 0:
        print(f"Communication Volume for \"One Grid per TensorDot\" algorithm:", flush=True)