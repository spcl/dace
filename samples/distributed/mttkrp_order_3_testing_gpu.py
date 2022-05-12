# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicitly distributed MTTKRP sample programs. """
import csv
import dace
import numpy as np
import timeit

from dace.sdfg import utils
from dace.transformation.auto import auto_optimize
from datetime import datetime
from os.path import exists


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


# Scaling
scaling = {
    1: (1024, 24),
    2: (1218, 30),
    4: (1450, 34),
    8: (1724, 42),
    12: (1908, 48),
    16: (2048, 48),
    27: (2337, 57),
    32: (2436, 60),
    64: (2900, 68),
    125: (3425, 85),
    128: (3448, 88),
    252: (4116, 126),
    256: (4096, 96),
    512: (4872, 120)
}


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


def write_csv(file_name, field_names, values, append=True):
    write_mode = 'w'
    if append:
        write_mode = 'a'
    new_file = not exists(file_name)
    with open(file_name, mode=write_mode) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        if new_file:
            writer.writeheader()
        for entry in values:
            writer.writerow(entry)


def write_time(dtime, bench, frmwrk, nodes, sizes, time_list, file_name, field_names, append=True):
    entries = []
    sockets = MPI.COMM_WORLD.Get_size()
    for t in time_list:
        entries.append(
            dict(datetime=dtime, benchmark=bench, framework=frmwrk, nodes=nodes, sizes=sizes, time=t))
    write_csv(file_name, field_names, entries, append=append)


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


@dace.program
def mttkrp_order_3_mode_0_compute(X: dctype[S0, S1, S2],
                                  JM: dctype[S1, R],
                                  KM: dctype[S2, R]) -> dctype[S0, R]:

    # 'ja, ka -> jka'
    tmp = np.ndarray((S1, S2, R), dtype=nptype)
    for j, k, a in dace.map[0:S1, 0:S2, 0:R]:
        tmp[j, k, a] = JM[j, a] * KM[k, a]
    # 'ijk, jka -> ia'
    return np.tensordot(X, tmp, axes=([1, 2], [0, 1]))


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
def mttkrp_order_3_mode_1_compute(X: dctype[S0, S1, S2],
                                  IM: dctype[S0, R],
                                  KM: dctype[S2, R]) -> dctype[S1, R]:

    # 'ijk, ka -> ija'
    tmp = np.tensordot(X, KM, axes=([2], [0]))
    # 'ija, ia -> ja'
    out = np.zeros((S1, R), dtype=nptype)
    for j, a in dace.map[0:S1, 0:R]:
        for i in range(S0):
            out[j, a] += tmp[i, j, a] * IM[i, a]
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
def mttkrp_order_3_mode_2_compute(X: dctype[S0, S1, S2],
                                  IM: dctype[S0, R],
                                  JM: dctype[S1, R]) -> dctype[S2, R]:

    # 'ia, ja -> ija'
    tmp = np.ndarray((S0, S1, R), dtype=nptype)
    for i, j, a in dace.map[0:S0, 0:S1, 0:R]:
        tmp[i, j, a] = IM[i, a] * JM[j, a]
    # 'ijk, ija -> ka'
    return np.tensordot(X, tmp, axes=([0, 1], [0, 1]))


if __name__ == "__main__":

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in grid_ijka:
        raise ValueError("Selected number of MPI processes is not supported.")

    file_name = "dace_cpu_{n}_nodes.csv".format(n=size)
    field_names = ["datetime", "benchmark", "framework", "nodes", "sizes", "time"]
    
    sdfg1, sdfg1c, sdfg2, sdfg2c, sdfg3, sdfg3c = (None, ) * 6
    if rank == 0:
        sdfg1 = auto_optimize.auto_optimize(mttkrp_order_3_mode_0.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
        sdfg1c = auto_optimize.auto_optimize(mttkrp_order_3_mode_0_compute.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
        sdfg2 = auto_optimize.auto_optimize(mttkrp_order_3_mode_1.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
        sdfg2c = auto_optimize.auto_optimize(mttkrp_order_3_mode_1_compute.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
        sdfg3 = auto_optimize.auto_optimize(mttkrp_order_3_mode_2.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
        sdfg3c = auto_optimize.auto_optimize(mttkrp_order_3_mode_2_compute.to_sdfg(simplify=True), device=dace.dtypes.DeviceType.GPU)
    func1 = utils.distributed_compile(sdfg1, commworld)
    func1c = utils.distributed_compile(sdfg1c, commworld)
    func2 = utils.distributed_compile(sdfg2, commworld)
    func2c = utils.distributed_compile(sdfg2c, commworld)
    func3 = utils.distributed_compile(sdfg3, commworld)
    func3c = utils.distributed_compile(sdfg3c, commworld)

    rng = np.random.default_rng(42)

    # MTTKRP, order 3

    PG = grid_ijka[size]
    S, R = (np.int32(s) for s in scaling[size])
    SG = [S // np.int32(p) for p in PG[:-1]] + [R // np.int32(PG[-1])]

    lX = rng.random((SG[0], SG[1], SG[2]))
    lI = rng.random((SG[0], SG[3]))
    lJ = rng.random((SG[1], SG[3]))
    lK = rng.random((SG[2], SG[3]))

    # MTTKRP, order 3, mode 0

    if rank == 0:
        print(f"##### MTTKRP, Order 3, Mode 0 #####\nLocal Sizes: {SG}\nGrid: {PG}""", flush=True)
    
    runtimes = timeit.repeat(
        """func1(X=lX, JM=lJ, KM=lK,
                 S0=SG[0], S1=SG[1], S2=SG[2], R=SG[3],
                 P0=PG[0], P1=PG[1], P2=PG[2], PR=PG[3]); commworld.Barrier()
        """,
        setup="commworld.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )
    
    if rank == 0:
        print(f"Median total runtime: {np.median(runtimes)} seconds", flush=True)
        write_time(str(datetime.now()), "mttkrp_order_3_mode_0", "dace_gpu", size, (S, S, S, R), runtimes, file_name, field_names, append=True)

        runtimes = timeit.repeat(
            """func1c(X=lX, JM=lJ, KM=lK,
                      S0=SG[0], S1=SG[1], S2=SG[2], R=SG[3],
                      P0=PG[0], P1=PG[1], P2=PG[2], PR=PG[3])
            """,
            setup="",
            repeat=10,
            number=1,
            globals=locals()
        )

        print(f"Median compute runtime: {np.median(runtimes)} seconds")
        write_time(str(datetime.now()),"mttkrp_order_3_mode_0_compute", "dace_gpu", size, (S, S, S, R), runtimes, file_name, field_names, append=True)
    
    # MTTKRP, order 3, mode 1

    if rank == 0:
        print(f"##### MTTKRP, Order 3, Mode 1 #####\nLocal Sizes: {SG}\nGrid: {PG}""", flush=True)
    
    runtimes = timeit.repeat(
        """func2(X=lX, IM=lI, KM=lK,
                 S0=SG[0], S1=SG[1], S2=SG[2], R=SG[3],
                 P0=PG[0], P1=PG[1], P2=PG[2], PR=PG[3]); commworld.Barrier()
        """,
        setup="commworld.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )
    
    if rank == 0:
        print(f"Median total runtime: {np.median(runtimes)} seconds", flush=True)
        write_time(str(datetime.now()), "mttkrp_order_3_mode_1", "dace_gpu", size, (S, S, S, R), runtimes, file_name, field_names, append=True)

        runtimes = timeit.repeat(
            """func2c(X=lX, IM=lI, KM=lK,
                      S0=SG[0], S1=SG[1], S2=SG[2], R=SG[3],
                      P0=PG[0], P1=PG[1], P2=PG[2], PR=PG[3])
            """,
            setup="",
            repeat=10,
            number=1,
            globals=locals()
        )

        print(f"Median compute runtime: {np.median(runtimes)} seconds")
        write_time(str(datetime.now()),"mttkrp_order_3_mode_1_compute", "dace_gpu", size, (S, S, S, R), runtimes, file_name, field_names, append=True)

    # MTTKRP, order 3, mode 2

    if rank == 0:
        print(f"##### MTTKRP, Order 3, Mode 2 #####\nLocal Sizes: {SG}\nGrid: {PG}""", flush=True)
    
    runtimes = timeit.repeat(
        """func3(X=lX, IM=lI, JM=lJ,
                 S0=SG[0], S1=SG[1], S2=SG[2], R=SG[3],
                 P0=PG[0], P1=PG[1], P2=PG[2], PR=PG[3]); commworld.Barrier()
        """,
        setup="commworld.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )
    
    if rank == 0:
        print(f"Median total runtime: {np.median(runtimes)} seconds", flush=True)
        write_time(str(datetime.now()), "mttkrp_order_3_mode_2", "dace_gpu", size, (S, S, S, R), runtimes, file_name, field_names, append=True)

        runtimes = timeit.repeat(
            """func3c(X=lX, IM=lI, JM=lJ,
                      S0=SG[0], S1=SG[1], S2=SG[2], R=SG[3],
                      P0=PG[0], P1=PG[1], P2=PG[2], PR=PG[3])
            """,
            setup="",
            repeat=10,
            number=1,
            globals=locals()
        )

        print(f"Median compute runtime: {np.median(runtimes)} seconds")
        write_time(str(datetime.now()),"mttkrp_order_3_mode_2_compute", "dace_gpu", size, (S, S, S, R), runtimes, file_name, field_names, append=True)
