# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Assortment of distributed Polybench kernels. """

import csv
import dace
import numpy as np
import timeit

from mpi4py import MPI
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.sdfg import utils


# Symbols
# Process grid
P, Px, Py = (dace.symbol(s, positive=True) for s in ('P', 'Px', 'Py'))
# Global sizes
GM, GN, GK, GR, GS, GT = (dace.symbol(s, positive=True) for s in ('GM', 'GN', 'GK', 'GR', 'GS', 'GT'))
# Local sizes
LMx, LMy, LNx, LNy, LKx, LKy = (dace.symbol(s, positive=True) for s in ('LMx', 'LMy', 'LNx', 'LNy', 'LKx', 'LKy'))
LRx, LRy, LSx, LSy, LTx, LTy = (dace.symbol(s, positive=True) for s in ('LRx', 'LRy', 'LSx', 'LSy', 'LTx', 'LTy'))


# Grid sizes
grid = {
    1: (1, 1),
    2: (1, 2),
    4: (2, 2),
    8: (2, 4),
    16: (4, 4),
    32: (4, 8),
    64: (8, 8),
    128: (8, 16),
    256: (16, 16)
}


# Helper methods
def time_to_ms(raw):
    return int(round(raw * 1000))


def l2g(idx, pidx, bsize):
    return idx + pidx * bsize


def int_ceil(divident, divisor):
    return int(np.ceil(divident / divisor))


def adjust_size(size, scal_func, comm_size, divisor):
    candidate = size * scal_func(comm_size)
    if candidate // divisor != candidate:
        candidate = np.ceil(candidate / divisor) * divisor
    return int(candidate)


file_name = "dace_cpu_{n}_processes.csv".format(n=MPI.COMM_WORLD.Get_size())
field_names = ["benchmark", "framework", "processes", "sizes", "time"]


def write_csv(file_name, field_names, values, append=True):
    write_mode = 'w'
    if append:
        write_mode = 'a'
    with open(file_name, mode=write_mode) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        if not append:
            writer.writeheader()
        for entry in values:
            writer.writerow(entry)


def write_time(bench, sz, time_list, append=True):
    entries = []
    processes = MPI.COMM_WORLD.Get_size()
    for t in time_list:
        entries.append(
            dict(benchmark=bench, framework="dace_cpu", processes=processes, sizes=sz, time=t))
    write_csv(file_name, field_names, entries, append=append)


def optimize_compile(program, rank, commworld):
    sdfg = None
    if rank == 0:
        sdfg = program.to_sdfg(simplify=True)
        sdfg = auto_optimize(sdfg, dace.DeviceType.CPU)
    return utils.distributed_compile(sdfg, commworld)


# atax
atax_sizes = [20000, 25000]


@dace.program
def atax(A: dace.float64[LMx, LNy], x: dace.float64[GN], y: dace.float64[GN]):
    tmp = dace.distr.MatMult(A, x, (Px * LMx, Py * LNy), c_block_sizes=(GM, 1))
    y[:] = dace.distr.MatMult(tmp, A, (GM, GN), c_block_sizes=(GN, 1))


def atax_shmem_init(M, N, datatype):
    fn = datatype(N)
    A = np.fromfunction(lambda i, j: ((i + j) % N) / (5 * M),
                        shape=(M, N), dtype=datatype)
    x = np.fromfunction(lambda i: 1 + (i / fn), shape=(N,), dtype=datatype)
    y = np.empty((N,), dtype=datatype)
    return A, x, y


def atax_distr_init(M, N, lM, lN, datatype, pi, pj):
    fn = datatype(N)
    A = np.fromfunction(lambda i, j: ((l2g(i, pi, lM) + l2g(j, pj, lN)) % N) / (5 * M),
                        shape=(lM, lN), dtype=datatype)
    # x = np.fromfunction(lambda i: 1 + (l2g(i, pj, lN) / fn),
    #                     shape=(lN,), dtype=datatype)
    # y = np.empty((lN,), dtype=datatype)
    x = np.fromfunction(lambda i: 1 + (i / fn), shape=(N,), dtype=datatype)
    y = np.empty((N,), dtype=datatype)
    return A, x, y


def run_atax(validate=False):

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    NPx, NPy = grid[size]
    cart_comm = commworld.Create_cart((NPx, NPy))
    i, j = cart_comm.Get_coords(rank)

    sizes = atax_sizes
    if validate:
        sizes = [int_ceil(s, 100) for s in atax_sizes]

    if rank == 0:
        print("===== atax =====")
        print("sizes: {}".format(sizes), flush=True)

    M, N = sizes
    M = adjust_size(M, lambda x: np.sqrt(x), size, max(NPx, NPy))
    N = adjust_size(N, lambda x: np.sqrt(x), size, max(NPx, NPy))
    if rank == 0:
        print("adjusted sizes: {}".format((M, N)), flush=True)

    lM, lN = M // NPx, N // NPy
    lA, x, y = atax_distr_init(M, N, lM, lN, np.float64, i, j)
    if rank == 0:
        print("data initialized", flush=True)
    
    func = optimize_compile(atax, rank, commworld)

    ldict = locals()
    commworld.Barrier()
    func(A=lA, x=x, y=y, LMx=lM, LNy=lN, GM=M, GN=N, Px=NPx, Py=NPy)

    stmt = ("func(A=lA, x=x, y=y, LMx=lM, LNy=lN, GM=M, GN=N, Px=NPx, Py=NPy); commworld.Barrier()")
    setup = "commworld.Barrier()"
    repeat = 10

    raw_time_list = timeit.repeat(stmt,
                                  setup=setup,
                                  repeat=repeat,
                                  number=1,
                                  globals=ldict)
    raw_time = np.median(raw_time_list)

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time), flush=True)
        write_time("atax", (M, N), raw_time_list, append=False)

    if validate and rank == 0:
        Aref, xref, yref = atax_shmem_init(M, N, np.float64)
        yref[:] = Aref.T @ (Aref @ xref)
        if np.allclose(y, yref):
            print("Validation: OK!", flush=True)
        else:
            print("Validation: Failed!", flush=True)


@dace.program
def bicg(A: dace.float64[LMx, LNy], p: dace.float64[GN], r: dace.float64[GM], o1: dace.float64[GN],
            o2: dace.float64[GM]):
    o1[:] = dace.distr.MatMult(r, A, (Px * LMx, Py * LNy), c_block_sizes=(GN, 1))
    o2[:] = dace.distr.MatMult(A, p, (GM, GN), c_block_sizes=(GM, 1))


@dace.program
def gemver(alpha: dace.float64, beta: dace.float64, A: dace.float64[LMx, LNy], u1: dace.float64[LMx],
            v1: dace.float64[LNy], u2: dace.float64[LMx], v2: dace.float64[LNy], w: dace.float64[GM],
            x: dace.float64[GN], y: dace.float64[GM], z: dace.float64[GN]):
    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    tmp1 = dace.distr.MatMult(y, A, (Px * LMx, Py * LNy), c_block_sizes=(GN, 1))
    x += beta * tmp1 + z
    tmp2 = dace.distr.MatMult(A, x, (GM, GN), c_block_sizes=(GM, 1))
    w += alpha * tmp2


@dace.program
def gesummv(alpha: dace.float64, beta: dace.float64, A: dace.float64[LMx, LNy], B: dace.float64[LMx, LNy],
            x: dace.float64[GN], y: dace.float64[GM]):
    tmp1 = dace.distr.MatMult(A, x, (Px * LMx, Py * LNy), c_block_sizes=(GM, 1))
    tmp2 = dace.distr.MatMult(B, x, (GM, GN), c_block_sizes=(GM, 1))
    y[:] = alpha * tmp1 + beta * tmp2


if __name__ == '__main__':

    run_atax(False)


