# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicit distributed TTMc sample programs. """
import dace
import numpy as np
import opt_einsum as oe

from dace.sdfg import utils
from dace.transformation.auto import auto_optimize


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
P0G1, P1G1, P2G1, P3G1, P4G1 = (dace.symbol(s) for s in ('P0G1', 'P1G1', 'P2G1', 'P3G1', 'P4G1'))
P0G2, P1G2, P2G2, P3G2, P4G2 = (dace.symbol(s) for s in ('P0G2', 'P1G2', 'P2G2', 'P3G2', 'P4G2'))
P0G3, P1G3, P2G3, P3G3, P4G3 = (dace.symbol(s) for s in ('P0G3', 'P1G3', 'P2G3', 'P3G3', 'P4G3'))
P0G4, P1G4, P2G4, P3G4, P4G4 = (dace.symbol(s) for s in ('P0G4', 'P1G4', 'P2G4', 'P3G4', 'P4G4'))
# Process grid lengths for tensor eigenvectors
PR0, PR1, PR2, PR3, PR4 = (dace.symbol(s) for s in ('PR0', 'PR1', 'PR2', 'PR3', 'PR4'))
PR0G1, PR1G1, PR2G1, PR3G1, PR4G1 = (dace.symbol(s) for s in ('PR0G1', 'PR1G1', 'PR2G1', 'PR3G1', 'PR4G1'))
PR0G2, PR1G2, PR2G2, PR3G2, PR4G2 = (dace.symbol(s) for s in ('PR0G2', 'PR1G2', 'PR2G2', 'PR3G2', 'PR4G2'))
PR0G3, PR1G3, PR2G3, PR3G3, PR4G3 = (dace.symbol(s) for s in ('PR0G3', 'PR1G3', 'PR2G3', 'PR3G3', 'PR4G3'))
PR0G4, PR1G4, PR2G4, PR3G4, PR4G4 = (dace.symbol(s) for s in ('PR0G4', 'PR1G4', 'PR2G4', 'PR3G4', 'PR4G4'))


# Einsums
# Tensor order 3
order_3_mode_0_str = 'ijk, jb, kc -> ibc'
# Tensor order 5
order_5_mode_0_str = 'ijklm, jb, kc, ld, me -> ibcde'


dctype = dace.float64
nptype = np.float64


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
    512:  [1, 1, 4, 4, 4, 1, 2, 2, 2],
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


grid_ijkb = {
    #     [i, j,   k, b]
    1:    [1, 1,   1, 1],
    2:    [1, 1,   2, 1],
    4:    [1, 1,   4, 1],
    8:    [1, 1,   8, 1],
    12:   [1, 1,  12, 1],
    16:   [1, 1,  16, 1],
    27:   [1, 1,  27, 1],
    32:   [1, 1,  32, 1],
    64:   [1, 1,  64, 1],
    125:  [1, 1, 125, 1],
    128:  [1, 1, 128, 1],
    256:  [1, 1, 256, 1],
    512:  [1, 1, 512, 1]
}


grid_ikbc = {
    #     [i,   k, b, c]
    1:    [1,   1, 1, 1],
    2:    [1,   2, 1, 1],
    4:    [1,   4, 1, 1],
    8:    [1,   8, 1, 1],
    12:   [1,  12, 1, 1],
    16:   [1,  16, 1, 1],
    27:   [1,  27, 1, 1],
    32:   [1,  32, 1, 1],
    64:   [1,  64, 1, 1],
    125:  [1, 125, 1, 1],
    128:  [1, 128, 1, 1],
    256:  [1, 256, 1, 1],
    512:  [1, 512, 1, 1]
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
def ttmc_order3_mode_0(X: dace.float64[S0G1, S1G1, S2G1],
                       JM: dace.float64[S1G1, R1G1],
                       KM: dace.float64[S2G2, R2G2]) -> dace.float64[S0G2, R1G2, R2G2]:

    # grid: ijkb
    grid1 = dace.comm.Cart_create([P0G1, P1G1, P2G1, PR1G1])
    # out: bik
    grid1_out_gather = dace.comm.Cart_sub(grid1, [True, False, True, True], exact_grid=0)
    grid1_out_reduce = dace.comm.Cart_sub(grid1, [False, True, False, False])
    grid1_out_subarray = dace.comm.Subarray((R1, S0, S2), (R1G1, S0G1, S2G1), dace.float64,
                                            process_grid=grid1_out_gather,
                                            correspondence=(3, 0, 2))

    # grid: ikbc
    grid2 = dace.comm.Cart_create([P0G2, P2G2, PR1G2, PR2G2])
    # in: bik
    grid2_in_scatter = dace.comm.Cart_sub(grid2, [True, True, True, False], exact_grid=0)
    grid2_in_bcast = dace.comm.Cart_sub(grid2, [False, False, False, True])
    grid2_in_subarray = dace.comm.Subarray((R1, S0, S2), (R1G2, S0G2, S2G2), dace.float64,
                                           process_grid=grid2_in_scatter,
                                           correspondence=(2, 0, 1))
    # out: ibc
    grid2_out_reduce = dace.comm.Cart_sub(grid2, [False, True, False, False])

    
    grid1_out = np.tensordot(JM, X, axes=([0], [1]))     # bik
    dace.comm.Reduce(grid1_out, 'MPI_SUM', grid=grid1_out_reduce)
    grid2_in = np.empty_like(grid1_out, shape=(R1G2, S0G2, S2G2))  # Need a nice way to infer the shape here
    dace.comm.Redistribute(grid1_out, grid1_out_subarray, grid2_in, grid2_in_subarray)
    dace.comm.Bcast(grid2_in, grid=grid2_in_bcast)

    grid2_out = np.tensordot(grid2_in, KM, axes=([2], [0]))   # bic
    dace.comm.Allreduce(grid2_out, 'MPI_SUM', grid=grid2_out_reduce)
    return np.transpose(grid2_out, axes=(1, 0, 2))


@dace.program
def mode_0_four_grids(X: dace.float64[S0G1, S1G1, S2G1, S3G1, S4G1],
                      JM: dace.float64[S1G4, R1G4],
                      KM: dace.float64[S2G3, R2G3],
                      LM: dace.float64[S3G2, R3G2],
                      MM: dace.float64[S4G1, R4G1]) -> dace.float64[S0G4, R1G4, R2G4, R3G4, R4G4]:

    # grid: ijklme
    grid1 = dace.comm.Cart_create([P0G1, P1G1, P2G1, P3G1, P4G1,  PR4G1])
    # out: ijkle
    grid1_out_gather = dace.comm.Cart_sub(grid1, [True, True, True, True, False, True], exact_grid=0)
    grid1_out_reduce = dace.comm.Cart_sub(grid1, [False, False, False, False, True, False])
    grid1_out_subarray = dace.comm.Subarray((S0, S1, S2, S3, R4), (S0G1, S1G1, S2G1, S3G1, R4G1), dace.float64, process_grid=grid1_out_gather)

    # grid: ijklde
    grid2 = dace.comm.Cart_create([P0G2, P1G2, P2G2, P3G2, PR3G2,  PR4G2])
    # in: ijkle
    grid2_in_scatter = dace.comm.Cart_sub(grid2, [True, True, True, True, False, True], exact_grid=0)
    grid2_in_bcast = dace.comm.Cart_sub(grid2, [False, False, False, False, True, False])
    grid2_in_subarray = dace.comm.Subarray((S0, S1, S2, S3, R4), (S0G2, S1G2, S2G2, S3G2, R4G2), dace.float64, process_grid=grid2_in_scatter)
    # out: eijkd (ijkde)
    grid2_out_gather = dace.comm.Cart_sub(grid2, [True, True, True, False, True, True], exact_grid=0)
    grid2_out_reduce = dace.comm.Cart_sub(grid2, [False, False, False, True, False, False])
    grid2_out_subarray = dace.comm.Subarray((R4, S0, S1, S2, R3), (R4G2, S0G2, S1G2, S2G2, R3G2), dace.float64, process_grid=grid2_out_gather, correspondence=(4, 0, 1, 2, 3))

    # grid: ijkcde
    grid3 = dace.comm.Cart_create([P0G3, P1G3, P2G3, PR2G3, PR3G3,  PR4G3])
    # in: eijkd (ijkde)
    grid3_in_scatter = dace.comm.Cart_sub(grid3, [True, True, True, False, True, True], exact_grid=0)
    grid3_in_bcast = dace.comm.Cart_sub(grid3, [False, False, False, True, False, False])
    grid3_in_subarray = dace.comm.Subarray((R4, S0, S1, S2, R3), (R4G3, S0G3, S1G3, S2G3, R3G3), dace.float64, process_grid=grid3_in_scatter, correspondence=(4, 0, 1, 2, 3))
    # out: deijc (ijcde)
    grid3_out_gather = dace.comm.Cart_sub(grid3, [True, True, False, True, True, True], exact_grid=0)
    grid3_out_reduce = dace.comm.Cart_sub(grid3, [False, False, True, False, False, False])
    grid3_out_subarray = dace.comm.Subarray((R3, R4, S0, S1, R2), (R3G3, R4G3, S0G3, S1G3, R2G3), dace.float64, process_grid=grid3_out_gather, correspondence=(3, 4, 0, 1, 2))

    # grid: ijbcde
    grid4 = dace.comm.Cart_create([P0G4, P1G4, PR1G4, PR2G4, PR3G4,  PR4G4])
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
def mode_0_ijkcde_grid(X: dace.float64[S0, S1, S2, S3, S4],
                       JM: dace.float64[S1, R1],
                       KM: dace.float64[S2, R2],
                       LM: dace.float64[S3, R3],
                       MM: dace.float64[S4, R4]) -> dace.float64[S0, R1, R2, R3, R4]:

    grid = dace.comm.Cart_create([P0G3, P1G3, P2G3, PR2G3, PR3G3,  PR4G3])

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


if __name__ == "__main__":

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in grid_ijklmbcde:
        raise ValueError("Selected number of MPI processes is not supported.")
    
    def auto_gpu(dcprog):
        sdfg = dcprog.to_sdfg(simplify=True)
        # sdfg.name = f"{sdfg.name}_cupy"
        # for _, arr in sdfg.arrays.items():
        #     if not arr.transient:
        #         arr.storage = dace.dtypes.StorageType.GPU_Global
        return auto_optimize.auto_optimize(sdfg, device=dace.dtypes.DeviceType.GPU)

    sdfg0, sdfg1, sdfg2, sdfg3 = None, None, None, None
    if rank == 0:
        sdfg0 = auto_gpu(mode_0_four_grids)
        # sdfg1 = auto_gpu(mode_0_ijkcde_grid)
        # sdfg2 = auto_gpu(mode_0_unified_grid)
        # sdfg3 = auto_gpu(ttmc_order3_mode_0)
    func0 = utils.distributed_compile(sdfg0, commworld)
    # func1 = utils.distributed_compile(sdfg1, commworld)
    # func2 = utils.distributed_compile(sdfg2, commworld)
    # func3 = utils.distributed_compile(sdfg3, commworld)

    # TTMc order 3
    grid_dims = grid_ijkb[size]
    lcm = np.lcm.reduce(list(grid_dims))

    S, R = np.int32(2 * lcm), np.int32(2 * lcm)

    SG1 = [S // np.int32(p) for p in grid_ijkb[size][:-1]]
    RG1 = [R, R] + [R // np.int32(p) for p in grid_ijkb[size][-1:]]
    SG2 = [S // np.int32(p) for p in grid_ikbc[size][:-2]] + [S]
    RG2 = [R] + [R // np.int32(p) for p in grid_ikbc[size][-2:]]

    PG1 = grid_ijkb[size]
    PG2 = grid_ikbc[size]

    X = np.arange(S**3, dtype=np.float64).reshape(S, S, S).copy()
    JM = np.arange(S*R, dtype=np.float64).reshape(S, R).copy()
    KM = np.arange(S*R, dtype=np.float64).reshape(S, R).copy()

    # ###### Reference ######

    # if rank == 0:
    #     ref = oe.contract(order_3_mode_0_str, X, JM, KM)

    # commworld.Barrier()

#     ##### One Grid per TensorDot #####

#     if rank == 0:
#         print(
#             f"""
# ##### One Grid per TensorDot #####
# jb, ijk -> bik: local sizes {SG1}, {RG1}, grid {grid_ijkb[size]}
# bik, kc -> ibc: local sizes {SG2}, {RG2}, grid {grid_ikbc[size]}""", flush=True
#         )
    
#     cart_comm = commworld.Create_cart(grid_ijklme[size])
#     coords = cart_comm.Get_coords(rank)
#     lX = X[
#         coords[0] * SG1[0]: (coords[0] + 1) * SG1[0],
#         coords[1] * SG1[1]: (coords[1] + 1) * SG1[1],
#         coords[2] * SG1[2]: (coords[2] + 1) * SG1[2]
#     ].copy()
#     lJM = JM[coords[1] * SG1[1]: (coords[1] + 1) * SG1[1], coords[3] * RG1[1]: (coords[3] + 1) * RG1[1]].copy()
#     cart_comm = commworld.Create_cart(grid_ijbcde[size])
#     coords = cart_comm.Get_coords(rank)
#     lKM = KM[coords[1] * SG2[2]: (coords[1] + 1) * SG2[2], coords[3] * RG2[2]: (coords[3] + 1) * RG2[2]].copy()

#     val = func3(X=lX, JM=lJM, KM=lKM,
#                 S0=S, S1=S, S2=S, S3=S, S4=S, R0=R, R1=R, R2=R, R3=R, R4=R,
#                 S0G1=SG1[0], S1G1=SG1[1], S2G1=SG1[2], 
#                 S0G2=SG2[0], S1G2=SG2[1], S2G2=SG2[2], 
#                 R0G1=RG1[0], R1G1=RG1[1], R2G1=RG1[2], 
#                 R0G2=RG2[0], R1G2=RG2[1], R2G2=RG2[2], 
#                 P0G1=PG1[0], P1G1=PG1[1], P2G1=PG1[2], PR1G1=PG1[3],
#                 P0G2=PG2[0], P1G2=PG2[1], P2G2=PG2[2], PR1G2=PG2[2], PR2G2=PG2[3])

#     if rank > 0:
#         commworld.Send(val, 0)
#     else:
#         out = np.ndarray((S, R, R), dtype=nptype)
#         out[
#             coords[0] * SG2[0]: (coords[0] + 1) * SG2[0],
#             coords[2] * RG2[1]: (coords[2] + 1) * RG2[1],
#             coords[3] * RG2[2]: (coords[3] + 1) * RG2[2]
#         ] = val

#         buf = np.ndarray((SG2[0], RG2[1], RG2[2]), dtype=nptype)
#         for r in range(1, size):
#             commworld.Recv(buf, r)
#             coords = cart_comm.Get_coords(r)
#             out[
#                 coords[0] * SG2[0]: (coords[0] + 1) * SG2[0],
#                 coords[2] * RG2[1]: (coords[2] + 1) * RG2[1],
#                 coords[3] * RG2[2]: (coords[3] + 1) * RG2[2]
#             ] = buf

#         print(f"\nRelative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
#         assert(np.allclose(out, ref))
    
#     commworld.Barrier()


    # TTMc order 5

    # grid_dims = set(grid_ijklmbcde[size])
    grid_dims = set()
    grid_dims.update(grid_ijklme[size])
    grid_dims.update(grid_ijklde[size])
    grid_dims.update(grid_ijkcde[size])
    grid_dims.update(grid_ijbcde[size])
    lcm = np.lcm.reduce(list(grid_dims))

    S, R = np.int32(2 * lcm), np.int32(2 * lcm)

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

    PG1 = grid_ijklme[size]
    PG2 = grid_ijklde[size]
    PG3 = grid_ijkcde[size]
    PG4 = grid_ijbcde[size]

    X = np.arange(S**5, dtype=np.float64).reshape(S, S, S, S, S).copy()
    JM = np.arange(S*R, dtype=np.float64).reshape(S, R).copy()
    KM = np.arange(S*R, dtype=np.float64).reshape(S, R).copy()
    LM = np.arange(S*R, dtype=np.float64).reshape(S, R).copy()
    MM = np.arange(S*R, dtype=np.float64).reshape(S, R).copy()

    ###### Reference ######

    if rank == 0:
        ref = oe.contract(order_5_mode_0_str, X, JM, KM, LM, MM)

    commworld.Barrier()

    # ###### Unified Grid #####

    # if rank == 0:
    #     print(f"##### Unified Grid #####\nLocal Sizes: {SGU}, {RGU}\nGrid: {grid_ijklmbcde[size]}""", flush=True)
    
    # cart_comm = commworld.Create_cart(grid_ijklmbcde[size])
    # coords = cart_comm.Get_coords(rank)
    # lX = X[
    #     coords[0] * SGU[0]: (coords[0] + 1) * SGU[0],
    #     coords[1] * SGU[1]: (coords[1] + 1) * SGU[1],
    #     coords[2] * SGU[2]: (coords[2] + 1) * SGU[2],
    #     coords[3] * SGU[3]: (coords[3] + 1) * SGU[3],
    #     coords[4] * SGU[4]: (coords[4] + 1) * SGU[4]
    # ].copy()
    # lJM = JM[coords[1] * SGU[1]: (coords[1] + 1) * SGU[1], coords[5] * RGU[1]: (coords[5] + 1) * RGU[1]].copy()
    # lKM = KM[coords[2] * SGU[2]: (coords[2] + 1) * SGU[2], coords[6] * RGU[2]: (coords[6] + 1) * RGU[2]].copy()
    # lLM = LM[coords[3] * SGU[3]: (coords[3] + 1) * SGU[3], coords[7] * RGU[3]: (coords[7] + 1) * RGU[3]].copy()
    # lMM = MM[coords[4] * SGU[4]: (coords[4] + 1) * SGU[4], coords[8] * RGU[4]: (coords[8] + 1) * RGU[4]].copy()

    # val = func2(X=lX, JM=lJM, KM=lKM, LM=lLM, MM=lMM, procs=size,
    #             S0=SGU[0], S1=SGU[1], S2=SGU[2], S3=SGU[3], S4=SGU[4],
    #             R0=RGU[0], R1=RGU[1], R2=RGU[2], R3=RGU[3], R4=RGU[4])

    # if rank > 0:
    #     commworld.Send(val, 0)
    # else:
    #     out = np.ndarray((S, R, R, R, R), dtype=nptype)
    #     out[
    #         coords[0] * SGU[0]: (coords[0] + 1) * SGU[0],
    #         coords[5] * RGU[1]: (coords[5] + 1) * RGU[1],
    #         coords[6] * RGU[2]: (coords[6] + 1) * RGU[2],
    #         coords[7] * RGU[3]: (coords[7] + 1) * RGU[3],
    #         coords[8] * RGU[4]: (coords[8] + 1) * RGU[4]
    #     ] = val

    #     buf = np.ndarray((SGU[0], RGU[1], RGU[2], RGU[3], RGU[4]), dtype=nptype)
    #     for r in range(1, size):
    #         commworld.Recv(buf, r)
    #         coords = cart_comm.Get_coords(r)
    #         out[
    #             coords[0] * SGU[0]: (coords[0] + 1) * SGU[0],
    #             coords[5] * RGU[1]: (coords[5] + 1) * RGU[1],
    #             coords[6] * RGU[2]: (coords[6] + 1) * RGU[2],
    #             coords[7] * RGU[3]: (coords[7] + 1) * RGU[3],
    #             coords[8] * RGU[4]: (coords[8] + 1) * RGU[4]
    #         ] = buf

    #     print(f"Relative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
    #     assert(np.allclose(out, ref))
    
    # commworld.Barrier()

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
    
    cart_comm = commworld.Create_cart(grid_ijklme[size])
    coords = cart_comm.Get_coords(rank)
    lX = X[
        coords[0] * SG1[0]: (coords[0] + 1) * SG1[0],
        coords[1] * SG1[1]: (coords[1] + 1) * SG1[1],
        coords[2] * SG1[2]: (coords[2] + 1) * SG1[2],
        coords[3] * SG1[3]: (coords[3] + 1) * SG1[3],
        coords[4] * SG1[4]: (coords[4] + 1) * SG1[4]
    ].copy()
    lMM = MM[coords[4] * SG1[4]: (coords[4] + 1) * SG1[4], coords[5] * RG1[4]: (coords[5] + 1) * RG1[4]].copy()
    cart_comm = commworld.Create_cart(grid_ijklde[size])
    coords = cart_comm.Get_coords(rank)
    lLM = LM[coords[3] * SG2[3]: (coords[3] + 1) * SG2[3], coords[4] * RG2[3]: (coords[4] + 1) * RG2[3]].copy()
    cart_comm = commworld.Create_cart(grid_ijkcde[size])
    coords = cart_comm.Get_coords(rank)
    lKM = KM[coords[2] * SG3[2]: (coords[2] + 1) * SG3[2], coords[3] * RG3[2]: (coords[3] + 1) * RG3[2]].copy()
    cart_comm = commworld.Create_cart(grid_ijbcde[size])
    coords = cart_comm.Get_coords(rank)
    lJM = JM[coords[1] * SG4[1]: (coords[1] + 1) * SG4[1], coords[2] * RG4[1]: (coords[2] + 1) * RG4[1]].copy()

    val = func0(X=lX, JM=lJM, KM=lKM, LM=lLM, MM=lMM, procs=size,
                S0=S, S1=S, S2=S, S3=S, S4=S, R0=R, R1=R, R2=R, R3=R, R4=R,
                S0G1=SG1[0], S1G1=SG1[1], S2G1=SG1[2], S3G1=SG1[3], S4G1=SG1[4],
                S0G2=SG2[0], S1G2=SG2[1], S2G2=SG2[2], S3G2=SG2[3], S4G2=SG2[4],
                S0G3=SG3[0], S1G3=SG3[1], S2G3=SG3[2], S3G3=SG3[3], S4G3=SG3[4],
                S0G4=SG4[0], S1G4=SG4[1], S2G4=SG4[2], S3G4=SG4[3], S4G4=SG4[4],
                R0G1=RG1[0], R1G1=RG1[1], R2G1=RG1[2], R3G1=RG1[3], R4G1=RG1[4],
                R0G2=RG2[0], R1G2=RG2[1], R2G2=RG2[2], R3G2=RG2[3], R4G2=RG2[4],
                R0G3=RG3[0], R1G3=RG3[1], R2G3=RG3[2], R3G3=RG3[3], R4G3=RG3[4],
                R0G4=RG4[0], R1G4=RG4[1], R2G4=RG4[2], R3G4=RG4[3], R4G4=RG4[4],
                P0G1=PG1[0], P1G1=PG1[1], P2G1=PG1[2], P3G1=PG1[3], P4G1=PG1[4], PR4G1=PG1[5],
                P0G2=PG2[0], P1G2=PG2[1], P2G2=PG2[2], P3G2=PG2[3], PR3G2=PG2[4], PR4G2=PG2[5],
                P0G3=PG3[0], P1G3=PG3[1], P2G3=PG3[2], PR2G3=PG3[3], PR3G3=PG3[4], PR4G3=PG3[5],
                P0G4=PG4[0], P1G4=PG4[1], PR1G4=PG4[2], PR2G4=PG4[3], PR3G4=PG4[4], PR4G4=PG4[5])

    if rank > 0:
        commworld.Send(val, 0)
    else:
        out = np.ndarray((S, R, R, R, R), dtype=nptype)
        out[
            coords[0] * SG4[0]: (coords[0] + 1) * SG4[0],
            coords[2] * RG4[1]: (coords[2] + 1) * RG4[1],
            coords[3] * RG4[2]: (coords[3] + 1) * RG4[2],
            coords[4] * RG4[3]: (coords[4] + 1) * RG4[3],
            coords[5] * RG4[4]: (coords[5] + 1) * RG4[4]
        ] = val

        buf = np.ndarray((SG4[0], RG4[1], RG4[2], RG4[3], RG4[4]), dtype=nptype)
        for r in range(1, size):
            commworld.Recv(buf, r)
            coords = cart_comm.Get_coords(r)
            out[
                coords[0] * SG4[0]: (coords[0] + 1) * SG4[0],
                coords[2] * RG4[1]: (coords[2] + 1) * RG4[1],
                coords[3] * RG4[2]: (coords[3] + 1) * RG4[2],
                coords[4] * RG4[3]: (coords[4] + 1) * RG4[3],
                coords[5] * RG4[4]: (coords[5] + 1) * RG4[4]
            ] = buf

        print(f"\nRelative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
        assert(np.allclose(out, ref))
    
    commworld.Barrier()

    # ###### Intersection Grid #####

    # if rank == 0:
    #     print(f"##### Intersection Grid #####\nLocal Sizes: {SG3}, {RG3}\nGrid: {grid_ijkcde[size]}""", flush=True)
    
    # cart_comm = commworld.Create_cart(grid_ijkcde[size])
    # coords = cart_comm.Get_coords(rank)
    # lX = X[
    #     coords[0] * SG3[0]: (coords[0] + 1) * SG3[0],
    #     coords[1] * SG3[1]: (coords[1] + 1) * SG3[1],
    #     coords[2] * SG3[2]: (coords[2] + 1) * SG3[2],
    #     :,
    #     :
    # ].copy()
    # lJM = JM[coords[1] * SG3[1]: (coords[1] + 1) * SG3[1], :].copy()
    # lKM = KM[coords[2] * SG3[2]: (coords[2] + 1) * SG3[2], coords[3] * RG3[2]: (coords[3] + 1) * RG3[2]].copy()
    # lLM = LM[:, coords[4] * RG3[3]: (coords[4] + 1) * RG3[3]].copy()
    # lMM = MM[:, coords[5] * RG3[4]: (coords[5] + 1) * RG3[4]].copy()

    # val = func1(X=lX, JM=lJM, KM=lKM, LM=lLM, MM=lMM, procs=size,
    #             S0=SG3[0], S1=SG3[1], S2=SG3[2], S3=SG3[3], S4=SG3[4],
    #             R0=RG3[0], R1=RG3[1], R2=RG3[2], R3=RG3[3], R4=RG3[4],
    #             P0G1=PG1[0], P1G1=PG1[1], P2G1=PG1[2], P3G1=PG1[3], P4G1=PG1[4], PR4G1=PG1[5],
    #             P0G2=PG2[0], P1G2=PG2[1], P2G2=PG2[2], P3G2=PG2[3], PR3G2=PG2[4], PR4G2=PG2[5],
    #             P0G3=PG3[0], P1G3=PG3[1], P2G3=PG3[2], PR2G3=PG3[3], PR3G3=PG3[4], PR4G3=PG3[5],
    #             P0G4=PG4[0], P1G4=PG4[1], PR1G4=PG4[2], PR2G4=PG4[3], PR3G4=PG4[4], PR4G4=PG4[5])

    # if rank > 0:
    #     commworld.Send(val, 0)
    # else:
    #     out = np.ndarray((S, R, R, R, R), dtype=nptype)
    #     out[
    #         coords[0] * SG3[0]: (coords[0] + 1) * SG3[0],
    #         :,
    #         coords[3] * RG3[2]: (coords[3] + 1) * RG3[2],
    #         coords[4] * RG3[3]: (coords[4] + 1) * RG3[3],
    #         coords[5] * RG3[4]: (coords[5] + 1) * RG3[4]
    #     ] = val

    #     buf = np.ndarray((SG3[0], RG3[1], RG3[2], RG3[3], RG3[4]), dtype=nptype)
    #     for r in range(1, size):
    #         commworld.Recv(buf, r)
    #         coords = cart_comm.Get_coords(r)
    #         out[
    #             coords[0] * SG3[0]: (coords[0] + 1) * SG3[0],
    #             :,
    #             coords[3] * RG3[2]: (coords[3] + 1) * RG3[2],
    #             coords[4] * RG3[3]: (coords[4] + 1) * RG3[3],
    #             coords[5] * RG3[4]: (coords[5] + 1) * RG3[4]
    #             ] = buf

    #     print(f"Relative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
    #     assert(np.allclose(out, ref))
    
    # commworld.Barrier()
