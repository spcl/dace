# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicitly distributed matrix-multiplication sample programs. """
import dace
import numpy as np
import opt_einsum as oe

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
one_mm = 'ij, jk -> ik'
two_mm = 'ij, jk, kl -> il'
three_mm = 'ij, jk, kl, lm -> im'


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


# DaCe Programs
@dace.program
def one_mm(A: dctype[S0, S1], B: dctype[S1, S2]) -> dctype[S0, S2]:

    grid = dace.comm.Cart_create([P0, P1, P2])
    out_reduce = dace.comm.Cart_sub(grid, [False, True, False])

    out = A @ B
    dace.comm.Reduce(out, grid, 'MPI_SUM', grid=out_reduce)
    return out


if __name__ == "__main__":

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in grid_ijk:
        raise ValueError("Selected number of MPI processes is not supported.")
    
    sdfg1 = None
    if rank == 0:
        sdfg1 = one_mm.to_sdfg(simplify=True, procs=size)
    func1 = utils.distributed_compile(sdfg1, commworld)

    # Single Matrix-Multiplication

    PG = grid_ijk[size]
    lcm = np.lcm.reduce(PG)
    S = np.int32(2 * lcm)
    SG = [S // np.int32(p) for p in PG]

    A = np.arange(S**2, dtype=nptype).reshape(S, S).copy()
    B = np.arange(S**2, dtype=nptype).reshape(S, S).copy()

    ###### Reference ######

    if rank == 0:
        ref = oe.contract(one_mm, A, B)

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
            out[coords[0] * SG[0]: (coords[0] + 1) * SG[0], coords[2] * SG[2]: (coords[2] + 1) * SG[2]] = bif

        print(f"Relative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
        assert(np.allclose(out, ref))
    
    commworld.Barrier()
