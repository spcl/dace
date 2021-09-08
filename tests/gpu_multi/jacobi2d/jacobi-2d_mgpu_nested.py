# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicitly multi GPU distributed Jacobi-2D sample."""
import dace
import numpy as np
import os
import timeit
from typing import List

from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import InlineSDFG, StateFusion
from dace.sdfg import nodes, infer_types
from dace import dtypes
import dace.libraries.nccl as nccl
from dace.libraries.nccl import utils as nutil

d_int = dace.int64
d_float = dace.float64
np_float = np.float64

TSTEPS = dace.symbol('TSTEPS', dtype=d_int, integer=True, positive=True)
N = dace.symbol('N', dtype=d_int, integer=True, positive=True)
lNy = dace.symbol('lNy', dtype=d_int, integer=True, positive=True)
Py = dace.symbol('Py', dtype=dace.int32, integer=True, positive=True)
pi = dace.symbol('pi', dtype=dace.int32, integer=True, nonnegative=True)
size = dace.symbol('size', dtype=dace.int32, integer=True, positive=True)
Ny = Py * lNy


@dace.program
def jacobi_2d_shared(A: dace.float64[Ny, N], B: dace.float64[Ny, N]):

    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])


@dace.program
def exchange(arr: d_float[lNy + 2, N], rank: d_int):
    group_handle = dace.define_local_scalar(d_int)
    if rank == 0:
        # send South
        dace.comm.nccl.Send(arr[-2], peer=rank + 1, group_handle=group_handle)
        # recv South
        dace.comm.nccl.Recv(arr[-1], peer=rank + 1, group_handle=group_handle)
    elif rank == size - 1:
        # send North
        dace.comm.nccl.Send(arr[1], peer=rank - 1, group_handle=group_handle)
        # recv North
        dace.comm.nccl.Recv(arr[0], peer=rank - 1, group_handle=group_handle)
    else:
        # send North
        dace.comm.nccl.Send(arr[1], peer=rank - 1, group_handle=group_handle)
        # recv South
        dace.comm.nccl.Recv(arr[-1], peer=rank + 1, group_handle=group_handle)
        # send South
        dace.comm.nccl.Send(arr[-2], peer=rank + 1, group_handle=group_handle)
        # recv North
        dace.comm.nccl.Recv(arr[0], peer=rank - 1, group_handle=group_handle)


@dace.program
def comp(in_arr: d_float[lNy + 2, N], out_arr: d_float[lNy + 2, N],
         rank: d_int):
    if rank == 0:
        # noff = 1
        # soff = 0
        # out_arr[1 + noff:-1,
        #         1:-1] = 0.2 * (in_arr[1 + noff:-1, 1:-1] +
        #                        in_arr[1 + noff:-1, :-2] +
        #                        in_arr[1 + noff:-1, 2:] +
        #                        in_arr[2 + noff:, 1:-1] + in_arr[noff:-2, 1:-1])
        out_arr[2:-1, 1:-1] = 0.2 * (in_arr[2:-1, 1:-1] + in_arr[2:-1, :-2] +
                                     in_arr[2:-1, 2:] + in_arr[3:, 1:-1] +
                                     in_arr[1:-2, 1:-1])
    elif rank == size - 1:
        # soff = 1
        # noff = 0
        # out_arr[1:-1 - soff,
        #         1:-1] = 0.2 * (in_arr[1:-1 - soff, 1:-1] +
        #                        in_arr[1:-1 - soff, :-2] +
        #                        in_arr[1:-1 - soff, 2:] + in_arr[2:-soff, 1:-1] +
        #                        in_arr[:-2 - soff, 1:-1])
        out_arr[1:-2, 1:-1] = 0.2 * (in_arr[1:-2, 1:-1] + in_arr[1:-2, :-2] +
                                     in_arr[1:-2, 2:] + in_arr[2:-1, 1:-1] +
                                     in_arr[:-3, 1:-1])
    else:
        # soff = 0
        # noff = 0
        out_arr[1:-1, 1:-1] = 0.2 * (in_arr[1:-1, 1:-1] + in_arr[1:-1, :-2] +
                                     in_arr[1:-1, 2:] + in_arr[2:, 1:-1] +
                                     in_arr[:-2, 1:-1])


@dace.program
def jacobi_2d_mgpu_nested(A: d_float[Ny, N], B: d_float[Ny, N]):

    for rank in dace.map[0:size]:
        # Local extended domain
        lA = np.zeros((lNy + 2, N), dtype=A.dtype)
        lB = np.zeros((lNy + 2, N), dtype=B.dtype)

        lA[1:-1, :] = A[rank * lNy:(rank + 1) * lNy, :]
        lB[1:-1, :] = B[rank * lNy:(rank + 1) * lNy, :]
        # if rank == 0:
        #     noff, soff = 1, 0
        # elif rank == size - 1:
        #     noff, soff = 0, 1
        # else:
        #     noff, soff = 0, 0
        for t in range(1, TSTEPS):
            exchange(lA, rank=rank, size=size)
            comp(lA, lB, rank=rank, size=size)
            # lB[1 + noff:-1 - soff, 1:-1] = 0.2 * (
            #     lA[1 + noff:-1 - soff, 1:-1] + lA[1 + noff:-1 - soff, :-2] +
            #     lA[1 + noff:-1 - soff, 2:] + lA[2 + noff:-soff, 1:-1] +
            #     lA[noff:-2 - soff, 1:-1])
            exchange(lB, rank=rank, size=size)
            comp(lB, lA, rank=rank, size=size)
            # lA[1 + noff:-1 - soff, 1:-1] = 0.2 * (
            #     lB[1 + noff:-1 - soff, 1:-1] + lB[1 + noff:-1 - soff, :-2] +
            #     lB[1 + noff:-1 - soff, 2:] + lB[2 + noff:-soff, 1:-1] +
            #     lB[noff:-2 - soff, 1:-1])

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
    ts, n = 100, 4096
    number_of_gpus = 4
    sdfg = jacobi_2d_mgpu_nested.to_sdfg(strict=False)
    gpu_map = find_map_by_param(sdfg, 'rank')
    gpu_map.schedule = dace.ScheduleType.GPU_Multidevice
    sdfg.specialize(
        dict(Py=number_of_gpus,
             size=number_of_gpus,
             lNy=n // number_of_gpus,
             TSTEPS=ts,
             N=n))
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_strict_transformations()

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/jacobi/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)

    print('GPU: start')
    A, B = init_data(n, np_float)
    sdfg(A, B)
    print('GPU: done')

    print('CPU: start')
    refA, refB = init_data(n, np_float)
    shared_sdfg = jacobi_2d_shared.to_sdfg()
    shared_sdfg.specialize(
        dict(Py=number_of_gpus,
             size=number_of_gpus,
             lNy=n // number_of_gpus,
             TSTEPS=ts,
             N=n))
    shared_sdfg(A=refA, B=refB)
    print('CPU: done')

    print("=======Validation=======")
    assert (np.allclose(A, refA)), f'A:\n{repr(A)}\nrefA:\n{repr(refA)}'
    assert (np.allclose(B, refB)), f'A:\n{repr(B)}\nrefA:\n{repr(refB)}'
    print("OK")
