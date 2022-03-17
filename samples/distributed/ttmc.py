# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicit distributed TTMc sample programs. """
import dace
import numpy as np
# import opt_einsum as oe
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

grid_ijklme = {
    1: [1, 1, 1, 1, 1, 1],
    64: [1, 4, 4, 4, 1, 1]
}

grid_ijklde = {
    1: [1, 1, 1, 1, 1, 1],
    64: [4, 4, 4, 1, 1, 1]
}

grid_ijkcde = {
    1: [1, 1, 1, 1, 1, 1],
    64: [4, 4, 1, 1, 1, 4]
}

grid_ijbcde = {
    1: [1, 1, 1, 1, 1, 1],
    64: [1, 4, 1, 1, 4, 4]
}


@dace.program
def mode_0(X: dace.float64[S0G1, S1G1, S2G1, S3G1, S4G1],
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
    grid4_out_subarray = dace.comm.Subarray((R2, R3, R4, S0, R1), (R2G4, R3G4, R4G4, S0G4, R1G4), dace.float64, process_grid=grid4_out_gather, correspondence=(2, 3, 4, 0, 1))
    
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
    dace.comm.Reduce(grid4_out, 'MPI_SUM', grid=grid4_out_reduce)

    return np.transpose(grid4_out, axes=[3, 4, 0, 1, 2])  # ibcde


if __name__ == "__main__":

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in grid_ijklme:
        raise ValueError("Selected number of MPI processes is not supported.")
    
    sdfg0, sdfg2, sdfg4 = None, None, None
    if rank == 0:
        sdfg0 = mode_0.to_sdfg(simplify=True, procs=size)
        # sdfg2 = mode_2.to_sdfg(simplify=True)
        # sdfg4 = mode_4.to_sdfg(simplify=True)
    func0 = utils.distributed_compile(sdfg0, commworld)
    # func2 = utils.distributed_compile(sdfg2, commworld)
    # func4 = utils.distributed_compile(sdfg4, commworld)

    S = weak_scaling[size]
    R = 26

    SG1 = [S // p for p in grid_ijklme[size][:-1]] + [S]
    RG1 = [R, R, R, R] + [R // p for p in grid_ijklme[size][-1:]]
    SG2 = [S // p for p in grid_ijklde[size][:-2]] + [S, S]
    RG2 = [R, R, R] + [R // p for p in grid_ijklde[size][-2:]]
    SG3 = [S // p for p in grid_ijkcde[size][:-3]] + [S, S, S]
    RG3 = [R, R] + [R // p for p in grid_ijkcde[size][-3:]]
    SG4 = [S // p for p in grid_ijbcde[size][:-4]]  + [S, S, S, S]
    RG4 = [R] + [R // p for p in grid_ijbcde[size][-4:]]

    if rank == 0:
        print(SG1, RG1, grid_ijklme[size], flush=True)
        print(SG2, RG2, grid_ijklde[size], flush=True)
        print(SG3, RG3, grid_ijkcde[size], flush=True)
        print(SG4, RG4, grid_ijbcde[size], flush=True)

    rng = np.random.default_rng(42)
    X = rng.random((SG1[0], SG1[1], SG1[2], SG1[3], SG1[4]))
    # IM = rng.random((SG1[0], RG1[0]))
    JM = rng.random((SG4[1], RG4[1]))
    KM = rng.random((SG3[2], RG3[2]))
    LM = rng.random((SG2[3], RG2[3]))
    MM = rng.random((SG1[4], RG1[4]))

    out0 = np.zeros((SG4[0], RG4[1], RG4[2], RG4[3], RG4[4]))
    # out2 = np.zeros((LS[2], LR))
    # out4 = np.zeros((LS[4], LR))

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
        """func0(X=X, JM=JM, KM=KM, LM=LM, MM=MM, out=out0, procs=size,
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
        print(f"Mode-0 median runtime: {np.median(runtimes)} seconds")

    # runtimes = timeit.repeat(
    #     """func2(X=X, IM=IM, JM=JM, LM=LM, MM=MM, out=out2,
    #              S0=LS[0], S1=LS[1], S2=LS[2], S3=LS[3], S4=LS[4], R=LR,
    #              P0=pgrid[0], P1=pgrid[1], P2=pgrid[2], P3=pgrid[3], P4=pgrid[4], PR=pgrid[5]); commworld.Barrier()
    #     """,
    #     setup="commworld.Barrier()",
    #     repeat=10,
    #     number=1,
    #     globals=locals()
    # )

    # if rank == 0:
    #     print(f"Mode-2 median runtime: {np.median(runtimes)} seconds")

    # LS = [weak_scaling[size] // p for p in grid4[size][:-1]]
    # LR = 25
    # pgrid = grid4[size]
    # if rank == 0:
    #     print(LS, LR, pgrid, flush=True)

    # runtimes = timeit.repeat(
    #     """func4(X=X, IM=IM, JM=JM, KM=KM, LM=LM, out=out4,
    #              S0=LS[0], S1=LS[1], S2=LS[2], S3=LS[3], S4=LS[4], R=LR,
    #              P0=pgrid[0], P1=pgrid[1], P2=pgrid[2], P3=pgrid[3], P4=pgrid[4], PR=pgrid[5]); commworld.Barrier()
    #     """,
    #     setup="commworld.Barrier()",
    #     repeat=10,
    #     number=1,
    #     globals=locals()
    # )

    # if rank == 0:
    #     print(f"Mode-4 median runtime: {np.median(runtimes)} seconds")


