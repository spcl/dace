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
# r = dace.symbol('r', dtype=dace.int32, integer=True, positive=True)
top_neighbor = dace.symbol('top_neighbor',
                           dtype=dace.int32,
                           integer=True,
                           nonnegative=True)
bottom_neighbor = dace.symbol('bottom_neighbor',
                              dtype=dace.int32,
                              integer=True,
                              nonnegative=True)
Ny = Py * lNy


@dace.program
def jacobi_2d_shared(A: dace.float64[N, Ny], B: dace.float64[N, Ny]):

    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])


@dace.program
def manexchange1(lA: d_float[lNy + 2, N + 2], rank: d_int):
    group_handle = dace.define_local_scalar(d_int,
                                            storage=dace.StorageType.GPU_Global)
    if rank == 0:
        # send North, recv South
        dace.comm.nccl.Recv(lA[-1], peer=size - 1, group_handle=group_handle)
        dace.comm.nccl.Send(lA[1], peer=rank + 1, group_handle=group_handle)
        # send South, recv North
        dace.comm.nccl.Recv(lA[0], peer=rank + 1, group_handle=group_handle)
        dace.comm.nccl.Send(lA[-2], peer=size - 1, group_handle=group_handle)
    elif rank == size - 1:
        # send North, recv South
        dace.comm.nccl.Recv(lA[-1], peer=rank - 1, group_handle=group_handle)
        dace.comm.nccl.Send(lA[1], peer=0, group_handle=group_handle)
        # send South, recv North
        dace.comm.nccl.Recv(lA[0], peer=0, group_handle=group_handle)
        dace.comm.nccl.Send(lA[-2], peer=rank - 1, group_handle=group_handle)
    else:
        # send North, recv South
        dace.comm.nccl.Recv(lA[-1], peer=rank - 1, group_handle=group_handle)
        dace.comm.nccl.Send(lA[1], peer=rank + 1, group_handle=group_handle)
        # send South, recv North
        dace.comm.nccl.Recv(lA[0], peer=rank + 1, group_handle=group_handle)
        dace.comm.nccl.Send(lA[-2], peer=rank - 1, group_handle=group_handle)


@dace.program
def manexchange2(lB: d_float[lNy + 2, N + 2], r: d_int):
    group_handle = dace.define_local_scalar(d_int,
                                            storage=dace.StorageType.GPU_Global)
    if r == 0:
        # send North, recv South
        dace.comm.nccl.Recv(lB[-1], peer=size - 1, group_handle=group_handle)
        dace.comm.nccl.Send(lB[1], peer=r + 1, group_handle=group_handle)
        # send South, recv North
        dace.comm.nccl.Recv(lB[0], peer=r + 1, group_handle=group_handle)
        dace.comm.nccl.Send(lB[-2], peer=size - 1, group_handle=group_handle)
    elif r == size - 1:
        # send North, recv South
        dace.comm.nccl.Recv(lB[-1], peer=r - 1, group_handle=group_handle)
        dace.comm.nccl.Send(lB[1], peer=0, group_handle=group_handle)
        # send South, recv North
        dace.comm.nccl.Recv(lB[0], peer=0, group_handle=group_handle)
        dace.comm.nccl.Send(lB[-2], peer=r - 1, group_handle=group_handle)
    else:
        # send North, recv South
        dace.comm.nccl.Recv(lB[-1], peer=r - 1, group_handle=group_handle)
        dace.comm.nccl.Send(lB[1], peer=r + 1, group_handle=group_handle)
        # send South, recv North
        dace.comm.nccl.Recv(lB[0], peer=r + 1, group_handle=group_handle)
        dace.comm.nccl.Send(lB[-2], peer=r - 1, group_handle=group_handle)


@dace.program
def jacobi_2d_mgpu(A: d_float[Ny, N], B: d_float[Ny, N]):

    for rank in dace.map[0:size]:
        # Local extended domain
        lA = np.zeros((lNy + 2, N + 2), dtype=A.dtype)
        lB = np.zeros((lNy + 2, N + 2), dtype=B.dtype)

        lA[1:-1, 1:-1] = A[rank * lNy:(rank + 1) * lNy, :]
        lB[1:-1, 1:-1] = B[rank * lNy:(rank + 1) * lNy, :]

        for t in range(1, TSTEPS):
            manexchange1(lA, rank=rank, size=size)
            lB[1:-1, 1:-1] = 0.2 * (lA[1:-1, 1:-1] + lA[1:-1, :-2] +
                                    lA[1:-1, 2:] + lA[2:, 1:-1] + lA[:-2, 1:-1])
            manexchange2(lB, r=rank, size=size)
            lA[1:-1, 1:-1] = 0.2 * (lB[1:-1, 1:-1] + lB[1:-1, :-2] +
                                    lB[1:-1, 2:] + lB[2:, 1:-1] + lB[:-2, 1:-1])

        A[rank * lNy:(rank + 1) * lNy] = lA[1:-1, 1:-1]
        B[rank * lNy:(rank + 1) * lNy] = lB[1:-1, 1:-1]


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
    ts, n = 100, 8
    number_of_gpus = 4
    sdfg = jacobi_2d_mgpu.to_sdfg(strict=False)
    gpu_map = find_map_by_param(sdfg, 'rank')
    gpu_map.schedule = dace.ScheduleType.GPU_Multidevice
    # seq_1 = next(s for s in sdfg.sdfg_list if s.name == 'seq1')
    # seq_2 = next(s for s in sdfg.sdfg_list if s.name == 'seq2')
    # seq_1.parent_nsdfg_node.no_inline = True
    # seq_2.parent_nsdfg_node.no_inline = True
    # seq_1.parent_nsdfg_node.schedule = dace.ScheduleType.GPU_Sequential
    # seq_2.parent_nsdfg_node.schedule = dace.ScheduleType.GPU_Sequential
    sdfg.specialize(
        dict(Py=number_of_gpus,
             size=number_of_gpus,
             lNy=n // number_of_gpus,
             TSTEPS=ts,
             N=n))
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_strict_transformations()

    sdfg.name += '_manexchange'
    program_objects = sdfg.generate_code()
    from dace.codegen import compiler
    out_path = '.dacecache/local/jacobi/' + sdfg.name
    program_folder = compiler.generate_program_folder(sdfg, program_objects,
                                                      out_path)

    print('GPU: start')
    A, B = init_data(n, np_float)
    sdfg(A, B)
    print('GPU: done')

    print('CPU: start')
    refA, refB = init_data(n, np.float64)
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
