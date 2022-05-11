# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicitly distributed MTTKRP sample programs. """
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
R, R0, R1, R2, R3, R4 = (dace.symbol(s) for s in ('R', 'R0', 'R1', 'R2', 'R3', 'R4'))
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
PR, PR0, PR1, PR2, PR3, PR4 = (dace.symbol(s) for s in ('PR', 'PR0', 'PR1', 'PR2', 'PR3', 'PR4'))


# Einsums
# Tensor order 3
# mode-0
order_3_mode_0_str = 'ijk, ja, ka -> ia'
# mode-1
order_3_mode_1_str = 'ijk, ia, ka -> ja'
# mode-2
order_3_mode_2_str = 'ijk, ia, ja -> ka'
# Tensor order 5
# mode-0
order_5_mode_0_str = 'ijklm, ja, ka, la, ma -> ia'
# mode-2
order_5_mode_2_str = 'ijklm, ia, ja, la, ma -> ka'
# mode-4
order_5_mode_4_str = 'ijklm, ia, ja, ka, la -> ma'


# Datatypes
dctype = dace.float64
nptype = np.float64


# Grids
grid_ijka = {
    #     [i, j, k, a]
    1:    [1, 1, 1, 1],
    2:    [1, 1, 2, 1],
    4:    [1, 2, 2, 1],
    8:    [2, 2, 2, 1],
    12:   [2, 2, 3, 1],
    16:   [2, 2, 4, 1],
    27:   [3, 3, 3, 1],
    32:   [2, 4, 4, 1],
    64:   [4, 4, 4, 1],
    125:  [5, 5, 5, 1],
    128:  [4, 4, 8, 1],
    252:  [6, 6, 7, 1],
    256:  [4, 8, 8, 1],
    512:  [8, 8, 8, 1],
}


grid_ijklma = {
    #     [i,  j,  k, l, m, a]
    1:    [1,  1,  1, 1, 1, 1],
    2:    [1,  1,  2, 1, 1, 1],
    4:    [1,  2,  2, 1, 1, 1],
    6:    [1,  2,  3, 1, 1, 1],
    8:    [1,  2,  4, 1, 1, 1],
    16:   [1,  4,  4, 1, 1, 1],
    30:   [1,  5,  6, 1, 1, 1],
    32:   [1,  4,  8, 1, 1, 1],
    64:   [1,  8,  8, 1, 1, 1],
    121:  [1, 11, 11, 1, 1, 1],
    128:  [1,  8, 16, 1, 1, 1],
    256:  [1, 16, 16, 1, 1, 1],
    506:  [1, 22, 23, 1, 1, 1],
    512:  [1, 16, 32, 1, 1, 1],
}


# DaCe Programs
@dace.program
def mttkrp_order_3_mode_0(X: dctype[S0, S1, S2],
                          JM: dctype[S1, R],
                          KM: dctype[S2, R]) -> dctype[S0, R]:

    grid = dace.comm.Cart_create([P0, P1, P2, PR])
    out_reduce = dace.comm.Cart_sub(grid, [False, True, True, False])

    # 'ja, ka -> jka'
    tmp = np.ndarray((S1, S2, R), dtype=nptype)
    for j, k, a in dace.map[0:S1, 0:S2, 0:R]:
        tmp[j, k, a] = JM[j, a] * KM[k, a]
    # 'ijk, jka -> ia'
    out = np.tensordot(X, tmp, axes=([1, 2], [0, 1]))
    dace.comm.Allreduce(out, 'MPI_SUM', grid=out_reduce)
    return out


# @dace.program
# def mttkrp_order_3_mode_1(X: dctype[S0, S1, S2],
#                           IM: dctype[S0, R],
#                           KM: dctype[S2, R]) -> dctype[S1, R]:

#     grid = dace.comm.Cart_create([P0, P1, P2, PR])
#     out_reduce = dace.comm.Cart_sub(grid, [True, False, True, False])

#     # 'ka, ia -> kia'
#     tmp = np.ndarray((S2, S0, R), dtype=nptype)
#     for k, i, a in dace.map[0:S2, 0:S0, 0:R]:
#         tmp[k, i, a] = KM[k, a] * IM[i, a]
#     # 'ijk, kia -> ja'
#     out = np.tensordot(X, tmp, axes=([2, 0], [0, 1]))
#     dace.comm.Allreduce(out, 'MPI_SUM', grid=out_reduce)
#     return out


@dace.program
def mttkrp_order_3_mode_1(X: dctype[S0, S1, S2],
                          IM: dctype[S0, R],
                          KM: dctype[S2, R]) -> dctype[S1, R]:

    grid = dace.comm.Cart_create([P0, P1, P2, PR])
    out_reduce = dace.comm.Cart_sub(grid, [True, False, True, False])

    # 'ijk, ka -> ija'
    tmp = np.tensordot(X, KM, axes=([2], [0]))
    # 'ija, ia -> ja'
    out = np.zeros((S1, R), dtype=nptype)
    for j, a in dace.map[0:S1, 0:R]:
        for i in range(S0):
            out[j, a] += tmp[i, j, a] * IM[i, a]
    dace.comm.Allreduce(out, 'MPI_SUM', grid=out_reduce)
    return out


@dace.program
def mttkrp_order_3_mode_2(X: dctype[S0, S1, S2],
                          IM: dctype[S0, R],
                          JM: dctype[S1, R]) -> dctype[S2, R]:

    grid = dace.comm.Cart_create([P0, P1, P2, PR])
    out_reduce = dace.comm.Cart_sub(grid, [True, True, False, False])

    # 'ia, ja -> ija'
    tmp = np.ndarray((S0, S1, R), dtype=nptype)
    for i, j, a in dace.map[0:S0, 0:S1, 0:R]:
        tmp[i, j, a] = IM[i, a] * JM[j, a]
    # 'ijk, ija -> ka'
    out = np.tensordot(X, tmp, axes=([0, 1], [0, 1]))
    dace.comm.Allreduce(out, 'MPI_SUM', grid=out_reduce)
    return out


@dace.program
def mttkrp_order_5_mode_0(X: dctype[S0, S1, S2, S3, S4],
                          JM: dctype[S1, R],
                          KM: dctype[S2, R],
                          LM: dctype[S3, R],
                          MM: dctype[S4, R]) -> dctype[S0, R]:

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
    out = np.zeros((S0, R), dtype=nptype)
    for i, a in dace.map[0:S0, 0:R]:
        for j in range(S1):
            for k in range(S2):
                out[i, a] += tmp2[i, j, k, a] * tmp3[j, k, a]
    
    # Reduce
    dace.comm.Allreduce(out, 'MPI_SUM', grid=reduce_grid)
    return out


@dace.program
def mttkrp_order_5_mode_2(X: dctype[S0, S1, S2, S3, S4],
                          IM: dctype[S0, R],
                          JM: dctype[S1, R],
                          LM: dctype[S3, R],
                          MM: dctype[S4, R]) -> dctype[S2, R]:

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
    out = np.zeros((S2, R), dtype=nptype)
    for k, a in dace.map[0:S2, 0:R]:
        for i in range(S0):
            for j in range(S1):
                out[k, a] += tmp2[i, j, k, a] * tmp3[i, j, a]
    
    # Reduce
    dace.comm.Allreduce(out, 'MPI_SUM', grid=reduce_grid)
    return out


@dace.program
def mttkrp_order_5_mode_4(X: dctype[S0, S1, S2, S3, S4],
                          IM: dctype[S0, R],
                          JM: dctype[S1, R],
                          KM: dctype[S2, R],
                          LM: dctype[S3, R]) -> dctype[S4, R]:

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
    out = np.zeros((S4, R), dtype=nptype)
    for m, a in dace.map[0:S4, 0:R]:
        for k in range(S2):
            for l in range(S3):
                out[m, a] += tmp2[k, l, m, a] * tmp3[k, l, a]
    
    # Reduce
    dace.comm.Allreduce(out, 'MPI_SUM', grid=reduce_grid)
    return out


if __name__ == "__main__":

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in grid_ijka:
        raise ValueError("Selected number of MPI processes is not supported.")
    
    sdfg1, sdfg2, sdfg3 = (None, ) * 3
    sdfg4, sdfg5, sdfg6 = (None, ) * 3
    if rank == 0:
        sdfg1 = auto_optimize.auto_optimize(mttkrp_order_3_mode_0.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
        sdfg2 = auto_optimize.auto_optimize(mttkrp_order_3_mode_1.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
        sdfg3 = auto_optimize.auto_optimize(mttkrp_order_3_mode_2.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
        sdfg4 = auto_optimize.auto_optimize(mttkrp_order_5_mode_0.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
        sdfg5 = auto_optimize.auto_optimize(mttkrp_order_5_mode_2.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
        sdfg6 = auto_optimize.auto_optimize(mttkrp_order_5_mode_4.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
    func1 = utils.distributed_compile(sdfg1, commworld)
    func2 = utils.distributed_compile(sdfg2, commworld)
    func3 = utils.distributed_compile(sdfg3, commworld)
    func4 = utils.distributed_compile(sdfg4, commworld)
    func5 = utils.distributed_compile(sdfg5, commworld)
    func6 = utils.distributed_compile(sdfg6, commworld)

    # MTTKRP, order 3

    PG = grid_ijka[size]
    lcm = np.lcm.reduce(PG)
    S, R = np.int32(2 * lcm), np.int32(2 * lcm)
    SG = [S // np.int32(p) for p in PG]

    X = np.arange(S**3, dtype=nptype).reshape(S, S, S).copy()
    IM = np.arange(S*R, dtype=nptype).reshape(S, R).copy()
    JM = np.arange(S*R, dtype=nptype).reshape(S, R).copy()
    KM = np.arange(S*R, dtype=nptype).reshape(S, R).copy()

    # MTTKRP, order 3, mode 0

    ###### Reference ######

    if rank == 0:
        ref = oe.contract(order_3_mode_0_str, X, JM, KM)

    commworld.Barrier()

    ###### DaCe #####

    if rank == 0:
        print(f"##### MTTKRP, Order 3, Mode 0 #####\nLocal Sizes: {SG}\nGrid: {PG}""", flush=True)
    
    cart_comm = commworld.Create_cart(PG)
    coords = cart_comm.Get_coords(rank)
    lX = X[
        coords[0] * SG[0]: (coords[0] + 1) * SG[0],
        coords[1] * SG[1]: (coords[1] + 1) * SG[1],
        coords[2] * SG[2]: (coords[2] + 1) * SG[2]
    ].copy()
    lJ = JM[coords[1] * SG[1]: (coords[1] + 1) * SG[1], coords[3] * SG[3]: (coords[3] + 1) * SG[3]].copy()
    lK = KM[coords[2] * SG[2]: (coords[2] + 1) * SG[2], coords[3] * SG[3]: (coords[3] + 1) * SG[3]].copy()

    val = func1(X=lX, JM=lJ, KM=lK,
                S0=SG[0], S1=SG[1], S2=SG[2], R=SG[3],
                P0=PG[0], P1=PG[1], P2=PG[2], PR=PG[3])

    if rank > 0:
        commworld.Send(val, 0)
    else:
        out = np.ndarray((S, R), dtype=nptype)
        out[coords[0] * SG[0]: (coords[0] + 1) * SG[0], coords[3] * SG[3]: (coords[3] + 1) * SG[3]] = val

        buf = np.ndarray((SG[0], SG[3]), dtype=nptype)
        for r in range(1, size):
            commworld.Recv(buf, r)
            coords = cart_comm.Get_coords(r)
            out[coords[0] * SG[0]: (coords[0] + 1) * SG[0], coords[3] * SG[3]: (coords[3] + 1) * SG[3]] = buf

        print(f"Relative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
        assert(np.allclose(out, ref))
    
    commworld.Barrier()

    # MTTKRP, order 3, mode 1

    ###### Reference ######

    if rank == 0:
        ref = oe.contract(order_3_mode_1_str, X, IM, KM)

    commworld.Barrier()

    ###### DaCe #####

    if rank == 0:
        print(f"##### MTTKRP, Order 3, Mode 1 #####\nLocal Sizes: {SG}\nGrid: {PG}""", flush=True)
    
    cart_comm = commworld.Create_cart(PG)
    coords = cart_comm.Get_coords(rank)
    lX = X[
        coords[0] * SG[0]: (coords[0] + 1) * SG[0],
        coords[1] * SG[1]: (coords[1] + 1) * SG[1],
        coords[2] * SG[2]: (coords[2] + 1) * SG[2]
    ].copy()
    lI = IM[coords[0] * SG[0]: (coords[0] + 1) * SG[0], coords[3] * SG[3]: (coords[3] + 1) * SG[3]].copy()
    lK = KM[coords[2] * SG[2]: (coords[2] + 1) * SG[2], coords[3] * SG[3]: (coords[3] + 1) * SG[3]].copy()

    val = func2(X=lX, IM=lI, KM=lK,
                S0=SG[0], S1=SG[1], S2=SG[2], R=SG[3],
                P0=PG[0], P1=PG[1], P2=PG[2], PR=PG[3])

    if rank > 0:
        commworld.Send(val, 0)
    else:
        out = np.ndarray((S, R), dtype=nptype)
        out[coords[1] * SG[1]: (coords[1] + 1) * SG[1], coords[3] * SG[3]: (coords[3] + 1) * SG[3]] = val

        buf = np.ndarray((SG[1], SG[3]), dtype=nptype)
        for r in range(1, size):
            commworld.Recv(buf, r)
            coords = cart_comm.Get_coords(r)
            out[coords[1] * SG[1]: (coords[1] + 1) * SG[1], coords[3] * SG[3]: (coords[3] + 1) * SG[3]] = buf

        print(f"Relative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
        assert(np.allclose(out, ref))
    
    commworld.Barrier()

    # MTTKRP, order 3, mode 2

    ###### Reference ######

    if rank == 0:
        ref = oe.contract(order_3_mode_2_str, X, IM, JM)

    commworld.Barrier()

    ###### DaCe #####

    if rank == 0:
        print(f"##### MTTKRP, Order 3, Mode 2 #####\nLocal Sizes: {SG}\nGrid: {PG}""", flush=True)
    
    cart_comm = commworld.Create_cart(PG)
    coords = cart_comm.Get_coords(rank)
    lX = X[
        coords[0] * SG[0]: (coords[0] + 1) * SG[0],
        coords[1] * SG[1]: (coords[1] + 1) * SG[1],
        coords[2] * SG[2]: (coords[2] + 1) * SG[2]
    ].copy()
    lI = IM[coords[0] * SG[0]: (coords[0] + 1) * SG[0], coords[3] * SG[3]: (coords[3] + 1) * SG[3]].copy()
    lJ = JM[coords[1] * SG[1]: (coords[1] + 1) * SG[1], coords[3] * SG[3]: (coords[3] + 1) * SG[3]].copy()

    val = func3(X=lX, IM=lI, JM=lJ,
                S0=SG[0], S1=SG[1], S2=SG[2], R=SG[3],
                P0=PG[0], P1=PG[1], P2=PG[2], PR=PG[3])

    if rank > 0:
        commworld.Send(val, 0)
    else:
        out = np.ndarray((S, R), dtype=nptype)
        out[coords[2] * SG[2]: (coords[2] + 1) * SG[2], coords[3] * SG[3]: (coords[3] + 1) * SG[3]] = val

        buf = np.ndarray((SG[2], SG[3]), dtype=nptype)
        for r in range(1, size):
            commworld.Recv(buf, r)
            coords = cart_comm.Get_coords(r)
            out[coords[2] * SG[2]: (coords[2] + 1) * SG[2], coords[3] * SG[3]: (coords[3] + 1) * SG[3]] = buf

        print(f"Relative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
        assert(np.allclose(out, ref))
    
    commworld.Barrier()

    # MTTKRP, order 5

    if size not in grid_ijklma:
        raise ValueError("Selected number of MPI processes is not supported.")

    PG = grid_ijklma[size]
    lcm = np.lcm.reduce(PG)
    S, R = np.int32(2 * lcm), np.int32(2 * lcm)
    SG = [S // np.int32(p) for p in PG]

    X = np.arange(S**5, dtype=nptype).reshape(S, S, S, S, S).copy()
    IM = np.arange(S*R, dtype=nptype).reshape(S, R).copy()
    JM = np.arange(S*R, dtype=nptype).reshape(S, R).copy()
    KM = np.arange(S*R, dtype=nptype).reshape(S, R).copy()
    LM = np.arange(S*R, dtype=nptype).reshape(S, R).copy()
    MM = np.arange(S*R, dtype=nptype).reshape(S, R).copy()

    # MTTKRP, order 3, mode 0

    ###### Reference ######

    if rank == 0:
        ref = oe.contract(order_5_mode_0_str, X, JM, KM, LM, MM)

    commworld.Barrier()

    ###### DaCe #####

    if rank == 0:
        print(f"##### MTTKRP, Order 5, Mode 0 #####\nLocal Sizes: {SG}\nGrid: {PG}""", flush=True)
    
    cart_comm = commworld.Create_cart(PG)
    coords = cart_comm.Get_coords(rank)
    lX = X[
        coords[0] * SG[0]: (coords[0] + 1) * SG[0],
        coords[1] * SG[1]: (coords[1] + 1) * SG[1],
        coords[2] * SG[2]: (coords[2] + 1) * SG[2],
        coords[3] * SG[3]: (coords[3] + 1) * SG[3],
        coords[4] * SG[4]: (coords[4] + 1) * SG[4],
    ].copy()
    lJ = JM[coords[1] * SG[1]: (coords[1] + 1) * SG[1], coords[5] * SG[5]: (coords[5] + 1) * SG[5]].copy()
    lK = KM[coords[2] * SG[2]: (coords[2] + 1) * SG[2], coords[5] * SG[5]: (coords[5] + 1) * SG[5]].copy()
    lL = LM[coords[3] * SG[3]: (coords[3] + 1) * SG[3], coords[5] * SG[5]: (coords[5] + 1) * SG[5]].copy()
    lM = MM[coords[4] * SG[4]: (coords[4] + 1) * SG[4], coords[5] * SG[5]: (coords[5] + 1) * SG[5]].copy()

    val = func4(X=lX, JM=lJ, KM=lK, LM=lL, MM=lM,
                S0=SG[0], S1=SG[1], S2=SG[2], S3=SG[3], S4=SG[4], R=SG[5],
                P0=PG[0], P1=PG[1], P2=PG[2], P3=PG[3], P4=PG[4], PR=PG[5])

    if rank > 0:
        commworld.Send(val, 0)
    else:
        out = np.ndarray((S, R), dtype=nptype)
        out[coords[0] * SG[0]: (coords[0] + 1) * SG[0], coords[5] * SG[5]: (coords[5] + 1) * SG[5]] = val

        buf = np.ndarray((SG[0], SG[3]), dtype=nptype)
        for r in range(1, size):
            commworld.Recv(buf, r)
            coords = cart_comm.Get_coords(r)
            out[coords[0] * SG[0]: (coords[0] + 1) * SG[0], coords[5] * SG[5]: (coords[5] + 1) * SG[5]] = buf

        print(f"Relative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
        assert(np.allclose(out, ref))
    
    commworld.Barrier()

    # MTTKRP, order 5, mode 2

    ###### Reference ######

    if rank == 0:
        ref = oe.contract(order_5_mode_2_str, X, IM, JM, LM, MM)

    commworld.Barrier()

    ###### DaCe #####

    if rank == 0:
        print(f"##### MTTKRP, Order 5, Mode 2 #####\nLocal Sizes: {SG}\nGrid: {PG}""", flush=True)
    
    cart_comm = commworld.Create_cart(PG)
    coords = cart_comm.Get_coords(rank)
    lX = X[
        coords[0] * SG[0]: (coords[0] + 1) * SG[0],
        coords[1] * SG[1]: (coords[1] + 1) * SG[1],
        coords[2] * SG[2]: (coords[2] + 1) * SG[2],
        coords[3] * SG[3]: (coords[3] + 1) * SG[3],
        coords[4] * SG[4]: (coords[4] + 1) * SG[4],
    ].copy()
    lI = IM[coords[0] * SG[0]: (coords[0] + 1) * SG[0], coords[5] * SG[5]: (coords[5] + 1) * SG[5]].copy()
    lK = KM[coords[2] * SG[2]: (coords[2] + 1) * SG[2], coords[5] * SG[5]: (coords[5] + 1) * SG[5]].copy()
    lL = LM[coords[3] * SG[3]: (coords[3] + 1) * SG[3], coords[5] * SG[5]: (coords[5] + 1) * SG[5]].copy()
    lM = MM[coords[4] * SG[4]: (coords[4] + 1) * SG[4], coords[5] * SG[5]: (coords[5] + 1) * SG[5]].copy()

    val = func5(X=lX, IM=lI, JM=lJ, LM=lL, MM=lM,
                S0=SG[0], S1=SG[1], S2=SG[2], S3=SG[3], S4=SG[4], R=SG[5],
                P0=PG[0], P1=PG[1], P2=PG[2], P3=PG[3], P4=PG[4], PR=PG[5])

    if rank > 0:
        commworld.Send(val, 0)
    else:
        out = np.ndarray((S, R), dtype=nptype)
        out[coords[2] * SG[2]: (coords[2] + 1) * SG[2], coords[5] * SG[5]: (coords[5] + 1) * SG[5]] = val

        buf = np.ndarray((SG[2], SG[5]), dtype=nptype)
        for r in range(1, size):
            commworld.Recv(buf, r)
            coords = cart_comm.Get_coords(r)
            out[coords[2] * SG[2]: (coords[2] + 1) * SG[2], coords[5] * SG[5]: (coords[5] + 1) * SG[5]] = buf

        print(f"Relative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
        assert(np.allclose(out, ref))
    
    commworld.Barrier()

    # MTTKRP, order 5, mode 4

    ###### Reference ######

    if rank == 0:
        ref = oe.contract(order_5_mode_4_str, X, IM, JM, KM, LM)

    commworld.Barrier()

    ###### DaCe #####

    if rank == 0:
        print(f"##### MTTKRP, Order 5, Mode 4 #####\nLocal Sizes: {SG}\nGrid: {PG}""", flush=True)
    
    cart_comm = commworld.Create_cart(PG)
    coords = cart_comm.Get_coords(rank)
    lX = X[
        coords[0] * SG[0]: (coords[0] + 1) * SG[0],
        coords[1] * SG[1]: (coords[1] + 1) * SG[1],
        coords[2] * SG[2]: (coords[2] + 1) * SG[2],
        coords[3] * SG[3]: (coords[3] + 1) * SG[3],
        coords[4] * SG[4]: (coords[4] + 1) * SG[4],
    ].copy()
    lI = IM[coords[0] * SG[0]: (coords[0] + 1) * SG[0], coords[5] * SG[5]: (coords[5] + 1) * SG[5]].copy()
    lJ = JM[coords[1] * SG[1]: (coords[1] + 1) * SG[1], coords[5] * SG[5]: (coords[5] + 1) * SG[5]].copy()
    lK = KM[coords[2] * SG[2]: (coords[2] + 1) * SG[2], coords[5] * SG[5]: (coords[5] + 1) * SG[5]].copy()
    lL = LM[coords[3] * SG[3]: (coords[3] + 1) * SG[3], coords[5] * SG[5]: (coords[5] + 1) * SG[5]].copy()
    
    val = func6(X=lX, IM=lI, JM=lJ, KM=lK, LM=lL,
                S0=SG[0], S1=SG[1], S2=SG[2], S3=SG[3], S4=SG[4], R=SG[5],
                P0=PG[0], P1=PG[1], P2=PG[2], P3=PG[3], P4=PG[4], PR=PG[5])

    if rank > 0:
        commworld.Send(val, 0)
    else:
        out = np.ndarray((S, R), dtype=nptype)
        out[coords[4] * SG[4]: (coords[4] + 1) * SG[4], coords[5] * SG[5]: (coords[5] + 1) * SG[5]] = val

        buf = np.ndarray((SG[4], SG[5]), dtype=nptype)
        for r in range(1, size):
            commworld.Recv(buf, r)
            coords = cart_comm.Get_coords(r)
            out[coords[4] * SG[4]: (coords[4] + 1) * SG[4], coords[5] * SG[5]: (coords[5] + 1) * SG[5]] = buf

        print(f"Relative error: {np.linalg.norm(out-ref) / np.linalg.norm(out)}", flush=True)
        assert(np.allclose(out, ref))
    
    commworld.Barrier()
