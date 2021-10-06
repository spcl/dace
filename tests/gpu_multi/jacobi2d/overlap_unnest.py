# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicitly multi GPU distributed Jacobi-2D sample."""
import dace
import numpy as np
import os
import timeit
from typing import List
import argparse

from dace.transformation.dataflow import MapFusion, RedundantArray, RedundantSecondArray
from dace.transformation.interstate import InlineSDFG, StateFusion
from dace.transformation.subgraph import SubgraphFusion
from dace.sdfg import nodes, infer_types
from dace import dtypes
import dace.libraries.nccl as nccl
from dace.libraries.nccl import utils as nutil
np.set_printoptions(linewidth=200, threshold=np.inf)

d_int = dace.int64
d_float = dace.float64
np_float = np.float64

TSTEPS = dace.symbol('TSTEPS', dtype=d_int, integer=True, positive=True)
N = dace.symbol('N', dtype=d_int, integer=True, positive=True)
lNy = dace.symbol('lNy', dtype=d_int, integer=True, positive=True)
Py = dace.symbol('Py', dtype=dace.int32, integer=True, positive=True)
size = dace.symbol('size', dtype=dace.int32, integer=True, positive=True)
# north_neighbor = dace.symbol('north_neighbor',
#                              dtype=dace.int32,
#                              integer=True,
#                              positive=True)
# south_neighbor = dace.symbol('south_neighbor',
#                              dtype=dace.int32,
#                              integer=True,
#                              positive=True)

Ny = size * lNy


@dace.program
def jacobi_2d_shared(A: dace.float64[Ny, N], B: dace.float64[Ny, N]):

    for t in range(TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])


@dace.program
def jacobi_padded(A: dace.float64[Ny, N], B: dace.float64[Ny, N]):
    lA = np.zeros((Ny + 2, N), np_float)
    lB = np.zeros((Ny + 2, N), np_float)
    lA[1:-1] = A[:]
    lB[1:-1] = B[:]
    for t in range(TSTEPS):
        lB[:, 1:-1] = 0.2 * (lA[:, 1:-1] + lA[:, :-2] + lA[:, 2:] +
                             lA[1:, 1:-1] + lA[:-1, 1:-1])
        lA[:, 1:-1] = 0.2 * (lB[:, 1:-1] + lB[:, :-2] + lB[:, 2:] +
                             lB[1:, 1:-1] + lB[:-1, 1:-1])
    A[:] = lA[1:-1]
    B[:] = lB[1:-1]


@dace.program
def jacobi_periodic(A: dace.float64[Ny, N], B: dace.float64[Ny, N]):
    for t in range(TSTEPS):
        # periodic boundary
        B[0, 1:-1] = 0.2 * (A[0, 1:-1] + A[0, :-2] + A[0, 2:] + A[1, 1:-1] +
                            A[-1, 1:-1])
        B[-1, 1:-1] = 0.2 * (A[-1, 1:-1] + A[-1, :-2] + A[-1, 2:] + A[0, 1:-1] +
                             A[-2, 1:-1])
        # interior
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])
        # periodic boundary
        A[0, 1:-1] = 0.2 * (B[0, 1:-1] + B[0, :-2] + B[0, 2:] + B[1, 1:-1] +
                            B[-1, 1:-1])
        A[-1, 1:-1] = 0.2 * (B[-1, 1:-1] + B[-1, :-2] + B[-1, 2:] + B[0, 1:-1] +
                             B[-2, 1:-1])
        # interior
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])


@dace.program
def exchange(arr: d_float[lNy + 2, N]):
    group_handle0 = dace.define_local_scalar(d_int)
    group_handle1 = dace.define_local_scalar(d_int)
    # recv North
    dace.comm.nccl.Send(arr[1], peer=north_neighbor, group_handle=group_handle0)
    # recv South
    dace.comm.nccl.Recv(arr[-1],
                        peer=south_neighbor,
                        group_handle=group_handle0)
    # send South
    dace.comm.nccl.Send(arr[-2],
                        peer=south_neighbor,
                        group_handle=group_handle1)
    # recv North
    dace.comm.nccl.Recv(arr[0], peer=north_neighbor, group_handle=group_handle1)
    # # recv North
    # dace.comm.nccl.Send(arr[1],
    #                     peer=(rank - 1) % size,
    #                     group_handle=group_handle0)
    # # recv South
    # dace.comm.nccl.Recv(arr[-1],
    #                     peer=(rank + 1) % size,
    #                     group_handle=group_handle0)
    # # send South
    # dace.comm.nccl.Send(arr[-2],
    #                     peer=(rank + 1) % size,
    #                     group_handle=group_handle1)
    # # recv North
    # dace.comm.nccl.Recv(arr[0],
    #                     peer=(rank - 1) % size,
    #                     group_handle=group_handle1)


@dace.program
def comp_interior(in_arr: d_float[lNy + 2, N], out_arr: d_float[lNy + 2, N]):
    out_arr[2:-2, 1:-1] = 0.2 * (in_arr[2:-2, 1:-1] + in_arr[2:-2, :-2] +
                                 in_arr[2:-2, 2:] + in_arr[3:-1, 1:-1] +
                                 in_arr[1:-3, 1:-1])


@dace.program
def comp_boundary(in_arr: d_float[lNy + 2, N], out_arr: d_float[lNy + 2, N]):
    # North boundary
    out_arr[1, 1:-1] = 0.2 * (in_arr[1, 1:-1] + in_arr[1, :-2] + in_arr[1, 2:] +
                              in_arr[2, 1:-1] + in_arr[0, 1:-1])
    # South boundary
    out_arr[-2,
            1:-1] = 0.2 * (in_arr[-2, 1:-1] + in_arr[-2, :-2] + in_arr[-2, 2:] +
                           in_arr[-1, 1:-1] + in_arr[-3, 1:-1])


@dace.program
def j_o_un(
    A: d_float[Ny, N],
    B: d_float[Ny, N],
    # lAs: d_float[size, lNy + 2, N],
    # lAe: d_float[size, lNy + 2, N],
    # lBb: d_float[size, lNy + 2, N],
    # lBe: d_float[size, lNy + 2, N],
    # lBi: d_float[size, lNy + 2, N],
):

    for rank in dace.map[0:size]:
        # Local extended domain
        lA = np.zeros((lNy + 2, N), dtype=A.dtype)
        lB = np.zeros((lNy + 2, N), dtype=B.dtype)
        north_neighbor = ((rank - 1) % size)
        south_neighbor = ((rank + 1) % size)

        lA[1:-1, :] = A[rank * lNy:(rank + 1) * lNy, :]
        # lB[1:-1, :] = B[rank * lNy:(rank + 1) * lNy, :]
        # lAs[rank] = lA[:]
        group_handle0 = dace.define_local_scalar(d_int)
        group_handle1 = dace.define_local_scalar(d_int)
        group_handle2 = dace.define_local_scalar(d_int)
        group_handle3 = dace.define_local_scalar(d_int)
        group_handle4 = dace.define_local_scalar(d_int)
        group_handle5 = dace.define_local_scalar(d_int)

        dace.comm.nccl.Send(lA[1],
                            peer=north_neighbor,
                            group_handle=group_handle0)

        dace.comm.nccl.Recv(lA[-1],
                            peer=south_neighbor,
                            group_handle=group_handle0)

        dace.comm.nccl.Send(lA[-2],
                            peer=south_neighbor,
                            group_handle=group_handle1)

        dace.comm.nccl.Recv(lA[0],
                            peer=north_neighbor,
                            group_handle=group_handle1)

        for t in range(TSTEPS):
            # comp_boundary(lA, lB)

            # North boundary
            lB[1, 1:-1] = 0.2 * (lA[1, 1:-1] + lA[1, :-2] + lA[1, 2:] +
                                 lA[2, 1:-1] + lA[0, 1:-1])
            # South boundary
            lB[-2, 1:-1] = 0.2 * (lA[-2, 1:-1] + lA[-2, :-2] + lA[-2, 2:] +
                                  lA[-1, 1:-1] + lA[-3, 1:-1])

            # lBb[rank] = lB[:]
            # recv North
            dace.comm.nccl.Send(lB[1],
                                peer=north_neighbor,
                                group_handle=group_handle2)
            # recv South
            dace.comm.nccl.Recv(lB[-1],
                                peer=south_neighbor,
                                group_handle=group_handle2)
            # send South
            dace.comm.nccl.Send(lB[-2],
                                peer=south_neighbor,
                                group_handle=group_handle3)
            # recv North
            dace.comm.nccl.Recv(lB[0],
                                peer=north_neighbor,
                                group_handle=group_handle3)
            # exchange(lB, rank=rank, size=size)
            # lBe[rank] = lB[:]
            # comp_interior(lA, lB)
            lB[2:-2,
               1:-1] = 0.2 * (lA[2:-2, 1:-1] + lA[2:-2, :-2] + lA[2:-2, 2:] +
                              lA[3:-1, 1:-1] + lA[1:-3, 1:-1])

            # lBi[rank] = lB[:]

            # comp_boundary(lB, lA)
            # North boundary
            lA[1, 1:-1] = 0.2 * (lB[1, 1:-1] + lB[1, :-2] + lB[1, 2:] +
                                 lB[2, 1:-1] + lB[0, 1:-1])
            # South boundary
            lA[-2, 1:-1] = 0.2 * (lB[-2, 1:-1] + lB[-2, :-2] + lB[-2, 2:] +
                                  lB[-1, 1:-1] + lB[-3, 1:-1])
            # if t < TSTEPS - 1:

            # recv North
            dace.comm.nccl.Send(lA[1],
                                peer=north_neighbor,
                                group_handle=group_handle4)
            # recv South
            dace.comm.nccl.Recv(lA[-1],
                                peer=south_neighbor,
                                group_handle=group_handle4)
            # send South
            dace.comm.nccl.Send(lA[-2],
                                peer=south_neighbor,
                                group_handle=group_handle5)
            # recv North
            dace.comm.nccl.Recv(lA[0],
                                peer=north_neighbor,
                                group_handle=group_handle5)
            # comp_interior(lB, lA)
            lA[2:-2,
               1:-1] = 0.2 * (lB[2:-2, 1:-1] + lB[2:-2, :-2] + lB[2:-2, 2:] +
                              lB[3:-1, 1:-1] + lB[1:-3, 1:-1])
        A[rank * lNy:(rank + 1) * lNy] = lA[1:-1, :]
        B[rank * lNy:(rank + 1) * lNy] = lB[1:-1, :]


def init_data(N, datatype):

    A = np.fromfunction(lambda i, j: i * (j + 2) / N,
                        shape=(N, N),
                        dtype=datatype)
    B = np.fromfunction(lambda i, j: i * (j + 3) / N,
                        shape=(N, N),
                        dtype=datatype)

    return A, B


def time_to_ms(raw):
    return int(round(raw * 1000))


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


def find_data_desc(sdfg: dace.SDFG, name: str) -> dace.nodes.MapEntry:
    """ Finds the first access node by the given data name. """
    return next(d for s, n, d in sdfg.arrays_recursive() if n == name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ts", type=int, nargs="?", default=10)
    parser.add_argument("n", type=int, nargs="?", default=128)
    # parser.add_argument("g", type=int, nargs="?", default=4)
    args = vars(parser.parse_args())

    ts = args['ts']
    n = args['n']
    # number_of_gpus = args['g']

    sdfg = j_o_un.to_sdfg(strict=False)
    gpu_map = find_map_by_param(sdfg, 'rank')
    gpu_map.schedule = dace.ScheduleType.GPU_Multidevice

    sdfg.apply_strict_transformations()
    for _, name, array in sdfg.arrays_recursive():
        if name in ['north_neighbor', 'south_neighbor']:
            array.storage = dace.StorageType.CPU_ThreadLocal
    infer_types.set_default_schedule_storage_types_and_location(sdfg)
    # sdfg.expand_library_nodes()
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated([RedundantArray, RedundantSecondArray])
    sdfg.apply_strict_transformations()

    # sdfg.specialize(
    #     dict(size=number_of_gpus, lNy=n // number_of_gpus, TSTEPS=ts, N=n))
    # ss = sdfg.start_state
    # for n in ss.nodes():
    #     if isinstance(n, nodes.NestedSDFG):
    #         nsn = n

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/jacobi/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)
    print('Compiling')
    comp_sdfg = sdfg.compile()
    print('compilation done')
    print(f'number_of_gpus, ts, n, time')
    sdfg.specialize(dict(size=4, lNy=n // 4, TSTEPS=ts, N=n))
    A, B = init_data(n, np_float)
    comp_sdfg(A, B)
    for i in range(10):
        A, B = init_data(n, np_float)
        start = timeit.default_timer()
        comp_sdfg(A, B)  #, lAs, lAe, lBb, lBe, lBi)
        end = timeit.default_timer()
        print(f'4, {ts}, {n}, {end-start}')

    sdfg.specialize(dict(size=2, lNy=n // 2, TSTEPS=ts, N=n))
    A, B = init_data(n, np_float)
    comp_sdfg(A, B)
    for i in range(10):
        A, B = init_data(n, np_float)
        start = timeit.default_timer()
        comp_sdfg(A, B)  #, lAs, lAe, lBb, lBe, lBi)
        end = timeit.default_timer()
        print(f'2, {ts}, {n}, {end-start}')

    print('CPU: start')
    shared_sdfg = jacobi_periodic.to_sdfg()
    shared_sdfg.apply_gpu_transformations()
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated([RedundantArray, RedundantSecondArray])
    shared_sdfg.specialize(dict(size=1, lNy=n // 1, TSTEPS=ts, N=n))
    s_c = shared_sdfg.compile()
    A, B = init_data(n, np_float)
    s_c(A, B)
    for i in range(10):
        A, B = init_data(n, np_float)
        start = timeit.default_timer()
        s_c(A, B)  #, lAs, lAe, lBb, lBe, lBi)
        end = timeit.default_timer()
        print(f'1, {ts}, {n}, {end-start}')

    # # print(f'lAstart:\n{repr(lAs)}\n')
    # # print(f'lAexchange:\n{repr(lAe)}\n')
    # # print(f'lBboundary:\n{repr(lBb)}\n')
    # # print(f'lBexchange:\n{repr(lBe)}\n')
    # # print(f'lBinterior:\n{repr(lBi)}\n')

    # if n <= 16:
    #     print(f'A:\n{repr(A)}\nrefA:\n{repr(refA)}')
    #     print(f'B:\n{repr(B)}\nrefB:\n{repr(refB)}')
    # print("=======Validation=======")
    # assert (np.allclose(A, refA)), f'A:\n{repr(A)}\nrefA:\n{repr(refA)}'
    # assert (np.allclose(B, refB)), f'B:\n{repr(B)}\nrefB:\n{repr(refB)}'
    # print("OK")
