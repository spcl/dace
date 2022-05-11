# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicitly distributed matrix-multiplication sample programs. """
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


# Einsums
one_mm_str = 'ij, jk -> ik'
two_mm_str = 'ij, jk, kl -> il'
three_mm_str = 'ij, jk, kl, lm -> im'


# Datatypes
dctype = dace.float64
nptype = np.float64


# Grids
grid_ijk = {
    #     [i, j, k]
    1:    [1, 1, 1],
    2:    [1, 1, 2],
    4:    [1, 2, 2],
    8:    [2, 2, 2],
    12:   [2, 2, 3],
    16:   [2, 2, 4],
    27:   [3, 3, 3],
    32:   [2, 4, 4],
    64:   [4, 4, 4],
    125:  [5, 5, 5],
    128:  [4, 4, 8],
    252:  [6, 6, 7],
    256:  [4, 8, 8],
    512:  [8, 8, 8],
}


grid_ikl = {
    #     [i, k, l]
    1:    [1, 1, 1],
    2:    [1, 1, 2],
    4:    [1, 2, 2],
    8:    [2, 2, 2],
    12:   [2, 2, 3],
    16:   [2, 2, 4],
    27:   [3, 3, 3],
    32:   [2, 4, 4],
    64:   [4, 4, 4],
    125:  [5, 5, 5],
    128:  [4, 4, 8],
    252:  [6, 6, 7],
    256:  [4, 8, 8],
    512:  [8, 8, 8],
}


grid_klm = {
    #     [k, l, m]
    1:    [1, 1, 1],
    2:    [1, 1, 2],
    4:    [1, 2, 2],
    8:    [2, 2, 2],
    12:   [2, 2, 3],
    16:   [2, 2, 4],
    27:   [3, 3, 3],
    32:   [2, 4, 4],
    64:   [4, 4, 4],
    125:  [5, 5, 5],
    128:  [4, 4, 8],
    252:  [6, 6, 7],
    256:  [4, 8, 8],
    512:  [8, 8, 8],
}


grid_ikm = {
    #     [i, k, m]
    1:    [1, 1, 1],
    2:    [1, 1, 2],
    4:    [1, 2, 2],
    8:    [2, 2, 2],
    12:   [2, 2, 3],
    16:   [2, 2, 4],
    27:   [3, 3, 3],
    32:   [2, 4, 4],
    64:   [4, 4, 4],
    125:  [5, 5, 5],
    128:  [4, 4, 8],
    252:  [6, 6, 7],
    256:  [4, 8, 8],
    512:  [8, 8, 8],
}


# DaCe Programs
@dace.program(device=dace.dtypes.DeviceType.GPU)
def one_mm(A: dctype[S0, S1], B: dctype[S1, S2]) -> dctype[S0, S2]:

    grid = dace.comm.Cart_create([P0, P1, P2])
    out_reduce = dace.comm.Cart_sub(grid, [False, True, False])

    out = A @ B
    dace.comm.Allreduce(out, 'MPI_SUM', grid=out_reduce)
    return out


@dace.program(device=dace.dtypes.DeviceType.GPU)
def two_mm(A: dctype[S0G1, S1G1], B: dctype[S1G1, S2G1], C: dctype[S2G2, S3G2]) -> dctype[S0G2, S3G2]:

    # grid: ijk
    grid1 = dace.comm.Cart_create([P0G1, P1G1, P2G1])
    # out: ik
    grid1_out_gather = dace.comm.Cart_sub(grid1, [True, False, True], exact_grid=0)
    grid1_out_reduce = dace.comm.Cart_sub(grid1, [False, True, False])
    grid1_out_subarray = dace.comm.Subarray((S0, S2), (S0G1, S2G1), dctype, process_grid=grid1_out_gather)

    # grid: ikl
    grid2 = dace.comm.Cart_create([P0G2, P2G2, P3G2])
    # in: ik
    grid2_in_scatter = dace.comm.Cart_sub(grid2, [True, True, False], exact_grid=0)
    grid2_in_bcast = dace.comm.Cart_sub(grid2, [False, False, True])
    grid2_in_subarray = dace.comm.Subarray((S0, S2), (S0G2, S2G2), dctype, process_grid=grid2_in_scatter)
    # out: il
    grid2_out_reduce = dace.comm.Cart_sub(grid2, [False, True, False])

    grid1_out = A @ B
    dace.comm.Reduce(grid1_out, 'MPI_SUM', grid=grid1_out_reduce)
    grid2_in = np.empty_like(grid1_out, shape=(S0G2, S2G2))
    dace.comm.Redistribute(grid1_out, grid1_out_subarray, grid2_in, grid2_in_subarray)
    dace.comm.Bcast(grid2_in, grid=grid2_in_bcast)

    grid2_out = grid2_in @ C
    dace.comm.Allreduce(grid2_out, 'MPI_SUM', grid=grid2_out_reduce)
    return grid2_out


@dace.program(device=dace.dtypes.DeviceType.GPU)
def three_mm(A: dctype[S0G1, S1G1], B: dctype[S1G1, S2G1],
             C: dctype[S2G2, S3G2], D: dctype[S3G2, S4G2]) -> dctype[S0G3, S4G3]:

    # grid: ijk
    grid1 = dace.comm.Cart_create([P0G1, P1G1, P2G1])
    # out: ik
    grid1_out_gather = dace.comm.Cart_sub(grid1, [True, False, True], exact_grid=0)
    grid1_out_reduce = dace.comm.Cart_sub(grid1, [False, True, False])
    grid1_out_subarray = dace.comm.Subarray((S0, S2), (S0G1, S2G1), dctype, process_grid=grid1_out_gather)

    # grid: klm
    grid2 = dace.comm.Cart_create([P2G2, P3G2, P4G2])
    # out: km
    grid2_out_gather = dace.comm.Cart_sub(grid2, [True, False, True], exact_grid=0)
    grid2_out_reduce = dace.comm.Cart_sub(grid2, [False, True, False])
    grid2_out_subarray = dace.comm.Subarray((S2, S4), (S2G2, S4G2), dctype, process_grid=grid2_out_gather)

    # grid: ikm
    grid3 = dace.comm.Cart_create([P0G3, P2G3, P4G3])
    # in1: ik
    grid3_in1_scatter = dace.comm.Cart_sub(grid3, [True, True, False], exact_grid=0)
    grid3_in1_bcast = dace.comm.Cart_sub(grid3, [False, False, True])
    grid3_in1_subarray = dace.comm.Subarray((S0, S2), (S0G3, S2G3), dctype, process_grid=grid3_in1_scatter)
    # in2: km
    grid3_in2_scatter = dace.comm.Cart_sub(grid3, [False, True, True], exact_grid=0)
    grid3_in2_bcast = dace.comm.Cart_sub(grid3, [True, False, False])
    grid3_in2_subarray = dace.comm.Subarray((S2, S4), (S2G3, S4G3), dctype, process_grid=grid3_in2_scatter)
    # out: im
    grid3_out_reduce = dace.comm.Cart_sub(grid3, [False, True, False])
    
    
    grid1_out = A @ B
    dace.comm.Reduce(grid1_out, 'MPI_SUM', grid=grid1_out_reduce)
    grid3_in1 = np.empty_like(grid1_out, shape=(S0G3, S2G3))
    dace.comm.Redistribute(grid1_out, grid1_out_subarray, grid3_in1, grid3_in1_subarray)
    dace.comm.Bcast(grid3_in1, grid=grid3_in1_bcast)

    grid2_out = C @ D
    dace.comm.Reduce(grid2_out, 'MPI_SUM', grid=grid2_out_reduce)
    grid3_in2 = np.empty_like(grid2_out, shape=(S2G3, S4G3))
    dace.comm.Redistribute(grid2_out, grid2_out_subarray, grid3_in2, grid3_in2_subarray)
    dace.comm.Bcast(grid3_in2, grid=grid3_in2_bcast)

    grid3_out = grid3_in1 @ grid3_in2
    dace.comm.Allreduce(grid3_out, 'MPI_SUM', grid=grid3_out_reduce)
    return grid3_out


if __name__ == "__main__":

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in grid_ijk:
        raise ValueError("Selected number of MPI processes is not supported.")
    
    sdfg1, sdfg2, sdfg3 = (None, ) * 3
    if rank == 0:
        sdfg1 = auto_optimize.auto_optimize(one_mm.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
        sdfg2 = auto_optimize.auto_optimize(two_mm.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
        sdfg3 = auto_optimize.auto_optimize(three_mm.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
    func1 = utils.distributed_compile(sdfg1, commworld)
    func2 = utils.distributed_compile(sdfg2, commworld)
    func3 = utils.distributed_compile(sdfg3, commworld)

    # Single Matrix-Multiplication

    PG = grid_ijk[size]
    lcm = np.lcm.reduce(PG)
    S = np.int32(2 * lcm)
    SG = [S // np.int32(p) for p in PG]

    A = np.arange(S**2, dtype=nptype).reshape(S, S).copy()
    B = np.arange(S**2, dtype=nptype).reshape(S, S).copy()

    ###### Reference ######

    if rank == 0:
        ref = oe.contract(one_mm_str, A, B)

    commworld.Barrier()

    ###### DaCe #####

    if rank == 0:
        print(f"##### 1MM #####\nLocal Sizes: {SG}\nGrid: {PG}""", flush=True)
    
    cart_comm = commworld.Create_cart(PG)
    coords = cart_comm.Get_coords(rank)
    lA = A[coords[0] * SG[0]: (coords[0] + 1) * SG[0], coords[1] * SG[1]: (coords[1] + 1) * SG[1]].copy()
    lB = B[coords[1] * SG[1]: (coords[1] + 1) * SG[1], coords[2] * SG[2]: (coords[2] + 1) * SG[2]].copy()

    val = func1(A=lA, B=lB,
                S0=SG[0], S1=SG[1], S2=SG[2],
                P0=PG[0], P1=PG[1], P2=PG[2])

    if rank > 0:
        commworld.Send(val, 0)
    else:
        out = np.ndarray((S, S), dtype=nptype)
        out[coords[0] * SG[0]: (coords[0] + 1) * SG[0], coords[2] * SG[2]: (coords[2] + 1) * SG[2]] = val

        buf = np.ndarray((SG[0], SG[2]), dtype=nptype)
        for r in range(1, size):
            commworld.Recv(buf, r)
            coords = cart_comm.Get_coords(r)
            out[coords[0] * SG[0]: (coords[0] + 1) * SG[0], coords[2] * SG[2]: (coords[2] + 1) * SG[2]] = buf

        print(f"Relative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
        assert(np.allclose(out, ref))
    
    commworld.Barrier()

    # Two Matrix-Multiplications

    PG1 = grid_ijk[size]
    PG2 = grid_ikl[size]
    lcm = np.lcm.reduce(PG1+PG2)
    S = np.int32(2 * lcm)
    SG1 = [S // np.int32(p) for p in PG1]
    SG2 = [S // np.int32(p) for p in PG2]

    A = np.arange(S**2, dtype=nptype).reshape(S, S).copy()
    B = np.arange(S**2, dtype=nptype).reshape(S, S).copy()
    C = np.arange(S**2, dtype=nptype).reshape(S, S).copy()

    ###### Reference ######

    if rank == 0:
        ref = oe.contract(two_mm_str, A, B, C)

    commworld.Barrier()

    ###### DaCe #####

    if rank == 0:
        print(f"##### 2MM #####\nLocal Sizes: {SG1}, {SG2}\nGrids: {PG1}, {PG2}""", flush=True)
    
    cart_comm = commworld.Create_cart(PG1)
    coords = cart_comm.Get_coords(rank)
    lA = A[coords[0] * SG1[0]: (coords[0] + 1) * SG1[0], coords[1] * SG1[1]: (coords[1] + 1) * SG1[1]].copy()
    lB = B[coords[1] * SG1[1]: (coords[1] + 1) * SG1[1], coords[2] * SG1[2]: (coords[2] + 1) * SG1[2]].copy()
    cart_comm = commworld.Create_cart(PG2)
    coords = cart_comm.Get_coords(rank)
    lC = C[coords[1] * SG2[1]: (coords[1] + 1) * SG2[1], coords[2] * SG2[2]: (coords[2] + 1) * SG2[2]].copy()

    val = func2(A=lA, B=lB, C=lC,
                S0=S, S1=S, S2=S, S3=S,
                S0G1=SG1[0], S1G1=SG1[1], S2G1=SG1[2],
                S0G2=SG2[0], S2G2=SG2[1], S3G2=SG2[2],
                P0G1=PG1[0], P1G1=PG1[1], P2G1=PG1[2],
                P0G2=PG2[0], P2G2=PG2[1], P3G2=PG2[2])

    if rank > 0:
        commworld.Send(val, 0)
    else:
        out = np.ndarray((S, S), dtype=nptype)
        out[coords[0] * SG2[0]: (coords[0] + 1) * SG2[0], coords[2] * SG2[2]: (coords[2] + 1) * SG2[2]] = val

        buf = np.ndarray((SG2[0], SG2[2]), dtype=nptype)
        for r in range(1, size):
            commworld.Recv(buf, r)
            coords = cart_comm.Get_coords(r)
            out[coords[0] * SG2[0]: (coords[0] + 1) * SG2[0], coords[2] * SG2[2]: (coords[2] + 1) * SG2[2]] = buf

        print(f"Relative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
        assert(np.allclose(out, ref))
    
    commworld.Barrier()

    # Three Matrix-Multiplications

    PG1 = grid_ijk[size]
    PG2 = grid_klm[size]
    PG3 = grid_ikm[size]
    lcm = np.lcm.reduce(PG1+PG2+PG3)
    S = np.int32(2 * lcm)
    SG1 = [S // np.int32(p) for p in PG1]
    SG2 = [S // np.int32(p) for p in PG2]
    SG3 = [S // np.int32(p) for p in PG3]

    A = np.arange(S**2, dtype=nptype).reshape(S, S).copy()
    B = np.arange(S**2, dtype=nptype).reshape(S, S).copy()
    C = np.arange(S**2, dtype=nptype).reshape(S, S).copy()
    D = np.arange(S**2, dtype=nptype).reshape(S, S).copy()

    ###### Reference ######

    if rank == 0:
        ref = oe.contract(three_mm_str, A, B, C, D)

    commworld.Barrier()

    ###### DaCe #####

    if rank == 0:
        print(f"##### 3MM #####\nLocal Sizes: {SG1}, {SG2}, {SG3}\nGrids: {PG1}, {PG2}, {PG3}""", flush=True)
    
    cart_comm = commworld.Create_cart(PG1)
    coords = cart_comm.Get_coords(rank)
    lA = A[coords[0] * SG1[0]: (coords[0] + 1) * SG1[0], coords[1] * SG1[1]: (coords[1] + 1) * SG1[1]].copy()
    lB = B[coords[1] * SG1[1]: (coords[1] + 1) * SG1[1], coords[2] * SG1[2]: (coords[2] + 1) * SG1[2]].copy()
    cart_comm = commworld.Create_cart(PG2)
    coords = cart_comm.Get_coords(rank)
    lC = C[coords[0] * SG2[0]: (coords[0] + 1) * SG2[0], coords[1] * SG2[1]: (coords[1] + 1) * SG2[1]].copy()
    lD = D[coords[1] * SG2[1]: (coords[1] + 1) * SG2[1], coords[2] * SG2[2]: (coords[2] + 1) * SG2[2]].copy()
    cart_comm = commworld.Create_cart(PG3)
    coords = cart_comm.Get_coords(rank)

    val = func3(A=lA, B=lB, C=lC, D=lD,
                S0=S, S1=S, S2=S, S3=S, S4=S,
                S0G1=SG1[0], S1G1=SG1[1], S2G1=SG1[2],
                S2G2=SG2[0], S3G2=SG2[1], S4G2=SG2[2],
                S0G3=SG3[0], S2G3=SG3[1], S4G3=SG3[2],
                P0G1=PG1[0], P1G1=PG1[1], P2G1=PG1[2],
                P2G2=PG2[0], P3G2=PG2[1], P4G2=PG2[2],
                P0G3=PG3[0], P2G3=PG3[1], P4G3=PG3[2])

    if rank > 0:
        commworld.Send(val, 0)
    else:
        out = np.ndarray((S, S), dtype=nptype)
        out[coords[0] * SG3[0]: (coords[0] + 1) * SG3[0], coords[2] * SG3[2]: (coords[2] + 1) * SG3[2]] = val

        buf = np.ndarray((SG3[0], SG3[2]), dtype=nptype)
        for r in range(1, size):
            commworld.Recv(buf, r)
            coords = cart_comm.Get_coords(r)
            out[coords[0] * SG3[0]: (coords[0] + 1) * SG3[0], coords[2] * SG3[2]: (coords[2] + 1) * SG3[2]] = buf

        print(f"Relative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
        assert(np.allclose(out, ref))
    
    commworld.Barrier()
