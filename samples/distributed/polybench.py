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
# Stencil-related
noff, soff, woff, eoff = (dace.symbol(s, nonnegative=True) for s in ('noff', 'soff', 'woff', 'eoff'))
nn, ns, nw, ne = (dace.symbol(s) for s in ('nn', 'ns', 'nw', 'ne'))

# Datatypes
MPI_Request = dace.opaque("MPI_Request")

# Grid sizes
grid = {1: (1, 1), 2: (1, 2), 4: (2, 2), 8: (2, 4), 16: (4, 4), 32: (4, 8), 64: (8, 8), 128: (8, 16), 256: (16, 16)}


# Helper methods
def relerr(ref, val):
    return np.linalg.norm(ref - val) / np.linalg.norm(ref)


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
        entries.append(dict(benchmark=bench, framework="dace_cpu", processes=processes, sizes=sz, time=t))
    write_csv(file_name, field_names, entries, append=append)


def optimize_compile(program, rank, commworld, autoopt=True):
    sdfg = None
    if rank == 0:
        sdfg = program.to_sdfg(simplify=True)
        if autoopt:
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
    A = np.fromfunction(lambda i, j: ((i + j) % N) / (5 * M), shape=(M, N), dtype=datatype)
    x = np.fromfunction(lambda i: 1 + (i / fn), shape=(N, ), dtype=datatype)
    y = np.empty((N, ), dtype=datatype)
    return A, x, y


def atax_distr_init(M, N, lM, lN, datatype, pi, pj):
    fn = datatype(N)
    A = np.fromfunction(lambda i, j: ((l2g(i, pi, lM) + l2g(j, pj, lN)) % N) / (5 * M), shape=(lM, lN), dtype=datatype)
    x = np.fromfunction(lambda i: 1 + (i / fn), shape=(N, ), dtype=datatype)
    y = np.empty((N, ), dtype=datatype)
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

    raw_time_list = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
    raw_time = np.median(raw_time_list)

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time), flush=True)
        write_time("atax", (M, N), raw_time_list)

    if validate and rank == 0:
        Aref, xref, yref = atax_shmem_init(M, N, np.float64)
        yref[:] = Aref.T @ (Aref @ xref)
        if np.allclose(y, yref):
            print("Validation: OK!", flush=True)
        else:
            print("Validation: Failed!", flush=True)


# bicg
bicg_sizes = [25000, 20000]


@dace.program
def bicg(A: dace.float64[LMx, LNy], p: dace.float64[GN], r: dace.float64[GM], o1: dace.float64[GN],
         o2: dace.float64[GM]):
    o1[:] = dace.distr.MatMult(r, A, (Px * LMx, Py * LNy), c_block_sizes=(GN, 1))
    o2[:] = dace.distr.MatMult(A, p, (GM, GN), c_block_sizes=(GM, 1))


def bicg_shmem_init(M, N, datatype):
    A = np.fromfunction(lambda i, j: (i * (j + 1) % M) / M, shape=(M, N), dtype=datatype)
    p = np.fromfunction(lambda i: (i % N) / N, shape=(N, ), dtype=datatype)
    r = np.fromfunction(lambda i: (i % M) / M, shape=(M, ), dtype=datatype)
    o1 = np.empty((N, ), dtype=datatype)
    o2 = np.empty((M, ), dtype=datatype)
    return A, p, r, o1, o2


def bicg_distr_init(M, N, lM, lN, datatype, pi, pj):
    A = np.fromfunction(lambda i, j: (l2g(i, pi, lM) * (l2g(j, pj, lN) + 1) % M) / M, shape=(lM, lN), dtype=datatype)
    p = np.fromfunction(lambda i: (i % N) / N, shape=(N, ), dtype=datatype)
    r = np.fromfunction(lambda i: (i % M) / M, shape=(M, ), dtype=datatype)
    o1 = np.empty((N, ), dtype=datatype)
    o2 = np.empty((M, ), dtype=datatype)
    return A, p, r, o1, o2


def run_bicg(validate=False):

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    NPx, NPy = grid[size]
    cart_comm = commworld.Create_cart((NPx, NPy))
    i, j = cart_comm.Get_coords(rank)

    sizes = bicg_sizes
    if validate:
        sizes = [int_ceil(s, 100) for s in bicg_sizes]

    if rank == 0:
        print("===== bicg =====")
        print("sizes: {}".format(sizes), flush=True)

    M, N = sizes
    M = adjust_size(M, lambda x: np.sqrt(x), size, max(NPx, NPy))
    N = adjust_size(N, lambda x: np.sqrt(x), size, max(NPx, NPy))
    if rank == 0:
        print("adjusted sizes: {}".format((M, N)), flush=True)

    lM, lN = M // NPx, N // NPy
    lA, p, r, o1, o2 = bicg_distr_init(M, N, lM, lN, np.float64, i, j)
    if rank == 0:
        print("data initialized", flush=True)

    func = optimize_compile(bicg, rank, commworld)

    ldict = locals()
    commworld.Barrier()
    func(A=lA, p=p, r=r, o1=o1, o2=o2, LMx=lM, LNy=lN, GM=M, GN=N, Px=NPx, Py=NPy)

    stmt = ("func(A=lA, p=p, r=r, o1=o1, o2=o2, LMx=lM, LNy=lN, GM=M, GN=N, Px=NPx, Py=NPy); commworld.Barrier()")
    setup = "commworld.Barrier()"
    repeat = 10

    raw_time_list = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
    raw_time = np.median(raw_time_list)

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time), flush=True)
        write_time("bicg", (M, N), raw_time_list)

    if validate and rank == 0:
        Aref, pref, rref, o1ref, o2ref = bicg_shmem_init(M, N, np.float64)
        o1ref[:] = Aref.T @ rref
        o2ref[:] = Aref @ pref
        if np.allclose(o1, o1ref) and np.allclose(o2, o2ref):
            print("Validation: OK!", flush=True)
        else:
            print("Validation: Failed!", flush=True)


# doitgen
doitgen_sizes = [128, 512, 512]


@dace.program
def doitgen(A: dace.float64[LKx, GM, GN], C4: dace.float64[GN, GN]):
    for k in range(LKx):
        A[k, :, :] = np.reshape(np.reshape(A[k], (GM, 1, GN)) @ C4, (GM, GN))


def doitgen_shmem_init(NR, NQ, NP, datatype):
    A = np.fromfunction(lambda i, j, k: ((i * j + k) % NP) / NP, shape=(NR, NQ, NP), dtype=datatype)
    C4 = np.fromfunction(lambda i, j: (i * j % NP) / NP, shape=(
        NP,
        NP,
    ), dtype=datatype)
    return A, C4


def doitgen_distr_init(NR, NQ, NP, lR, datatype, p):
    A = np.fromfunction(lambda i, j, k: ((l2g(i, p, lR) * j + k) % NP) / NP, shape=(lR, NQ, NP), dtype=datatype)
    C4 = np.fromfunction(lambda i, j: (i * j % NP) / NP, shape=(
        NP,
        NP,
    ), dtype=datatype)
    return A, C4


def run_doitgen(validate=False):

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    cart_comm = commworld.Create_cart((size, 1))
    i, j = cart_comm.Get_coords(rank)

    sizes = doitgen_sizes
    if validate:
        sizes = [int_ceil(s, 16) for s in doitgen_sizes]

    if rank == 0:
        print("===== doitgen =====")
        print("sizes: {}".format(sizes), flush=True)

    K, M, N = sizes
    K = adjust_size(K, lambda x: x, size, size)
    if rank == 0:
        print("adjusted sizes: {}".format((K, M, N)), flush=True)

    lK = K // size
    lA, C4 = doitgen_distr_init(K, M, N, lK, np.float64, rank)
    if rank == 0:
        print("data initialized", flush=True)

    func = optimize_compile(doitgen, rank, commworld)

    ldict = locals()
    commworld.Barrier()
    func(A=lA, C4=C4, LKx=lK, GM=M, GN=N)

    if validate:
        lAval = np.copy(lA)

    stmt = ("func(A=lA, C4=C4, LKx=lK, GM=M, GN=N); commworld.Barrier()")
    setup = "commworld.Barrier()"
    repeat = 10

    raw_time_list = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
    raw_time = np.median(raw_time_list)

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time), flush=True)
        write_time("doitgen", (K, M, N), raw_time_list)

    if validate:
        Aref, C4ref = doitgen_shmem_init(K, M, N, np.float64)
        Aref[:] = np.reshape(np.reshape(Aref, (K, M, 1, N)) @ C4ref, (K, M, N))
        lAref = Aref[i * lK:(i + 1) * lK]

        if np.allclose(lAval, lAref):
            print(f"Validation (rank {rank}): OK!", flush=True)
        else:
            print(f"Validation (rank {rank}): Failed!", flush=True)


# gemm
gemm_sizes = [8000, 9200, 5200]


@dace.program
def gemm(alpha: dace.float64, beta: dace.float64, C: dace.float64[LMx, LNy], A: dace.float64[LMx, LKy],
         B: dace.float64[LKx, LNy]):
    C[:] = alpha * dace.distr.MatMult(A, B, (LMx * Px, LNy * Py, GK)) + beta * C


def gemm_shmem_init(NI, NJ, NK, datatype):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI, shape=(NI, NJ), dtype=datatype)
    A = np.fromfunction(lambda i, k: (i * (k + 1) % NK) / NK, shape=(NI, NK), dtype=datatype)
    B = np.fromfunction(lambda k, j: (k * (j + 2) % NJ) / NJ, shape=(NK, NJ), dtype=datatype)
    return alpha, beta, C, A, B


def gemm_distr_init(NI, NJ, NK, lNI, lNJ, lNKa, lNKb, datatype, pi, pj):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((l2g(i, pi, lNI) * l2g(j, pj, lNJ) + 1) % NI) / NI,
                        shape=(lNI, lNJ),
                        dtype=datatype)
    A = np.fromfunction(lambda i, k: (l2g(i, pi, lNI) * (l2g(k, pj, lNKa) + 1) % NK) / NK,
                        shape=(lNI, lNKa),
                        dtype=datatype)
    B = np.fromfunction(lambda k, j: (l2g(k, pi, lNKb) * (l2g(j, pj, lNJ) + 2) % NJ) / NJ,
                        shape=(lNKb, lNJ),
                        dtype=datatype)
    return alpha, beta, C, A, B


def run_gemm(validate=False):

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    NPx, NPy = grid[size]
    cart_comm = commworld.Create_cart((NPx, NPy))
    i, j = cart_comm.Get_coords(rank)

    sizes = gemm_sizes
    if validate:
        sizes = [int_ceil(s, 100) for s in gemm_sizes]

    if rank == 0:
        print("===== gemm =====")
        print("sizes: {}".format(sizes), flush=True)

    M, N, K = sizes
    M = adjust_size(M, lambda x: np.cbrt(x), size, max(NPx, NPy))
    N = adjust_size(N, lambda x: np.cbrt(x), size, max(NPx, NPy))
    K = adjust_size(K, lambda x: np.cbrt(x), size, max(NPx, NPy))
    if rank == 0:
        print("adjusted sizes: {}".format((M, N, K)), flush=True)

    lM, lN, lKx, lKy = M // NPx, N // NPy, K // NPx, K // NPy
    alpha, beta, lC, lA, lB = gemm_distr_init(M, N, K, lM, lN, lKy, lKx, np.float64, i, j)
    if rank == 0:
        print("data initialized", flush=True)

    func = optimize_compile(gemm, rank, commworld)

    ldict = locals()
    commworld.Barrier()
    func(alpha=alpha, beta=beta, C=lC, A=lA, B=lB, LMx=lM, LNy=lN, LKx=lKx, LKy=lKy, GK=K, Px=NPx, Py=NPy)

    if validate:
        lCval = np.copy(lC)

    stmt = (
        "func(alpha=alpha, beta=beta, C=lC, A=lA, B=lB, LMx=lM, LNy=lN, LKx=lKx, LKy=lKy, GK=K, Px=NPx, Py=NPy); commworld.Barrier()"
    )
    setup = "commworld.Barrier()"
    repeat = 10

    raw_time_list = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
    raw_time = np.median(raw_time_list)

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time), flush=True)
        write_time("gemm", (M, N, K), raw_time_list)

    if validate:
        alpha, beta, Cref, Aref, Bref = gemm_shmem_init(M, N, K, np.float64)
        Cref[:] = alpha * (Aref @ Bref) + beta * Cref
        lCref = Cref[i * lM:(i + 1) * lM, j * lN:(j + 1) * lN]
        if np.allclose(lCval, lCref):
            print(f"Validation (rank {rank}): OK!", flush=True)
        else:
            print(f"Validation (rank {rank}): Failed!", flush=True)


# gemver
gemver_sizes = [10000]


@dace.program
def gemver(alpha: dace.float64, beta: dace.float64, A: dace.float64[LMx, LNy], u1: dace.float64[LMx],
           v1: dace.float64[LNy], u2: dace.float64[LMx], v2: dace.float64[LNy], w: dace.float64[GM],
           x: dace.float64[GN], y: dace.float64[GM], z: dace.float64[GN]):
    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    tmp1 = dace.distr.MatMult(y, A, (Px * LMx, Py * LNy), c_block_sizes=(GN, 1))
    x += beta * tmp1 + z
    tmp2 = dace.distr.MatMult(A, x, (GM, GN), c_block_sizes=(GM, 1))
    w += alpha * tmp2


def gemver_shmem_init(N, datatype):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    fn = datatype(N)
    A = np.fromfunction(lambda i, j: (i * j % N) / N, shape=(N, N), dtype=datatype)
    u1 = np.fromfunction(lambda i: i, shape=(N, ), dtype=datatype)
    u2 = np.fromfunction(lambda i: ((i + 1) / fn) / 2.0, shape=(N, ), dtype=datatype)
    v1 = np.fromfunction(lambda i: ((i + 1) / fn) / 4.0, shape=(N, ), dtype=datatype)
    v2 = np.fromfunction(lambda i: ((i + 1) / fn) / 6.0, shape=(N, ), dtype=datatype)
    w = np.zeros((N, ), dtype=datatype)
    x = np.zeros((N, ), dtype=datatype)
    y = np.fromfunction(lambda i: ((i + 1) / fn) / 8.0, shape=(N, ), dtype=datatype)
    z = np.fromfunction(lambda i: ((i + 1) / fn) / 9.0, shape=(N, ), dtype=datatype)
    return alpha, beta, A, u1, u2, v1, v2, w, x, y, z


def gemver_distr_init(N, lM, lN, datatype, pi, pj):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    fn = datatype(N)
    A = np.fromfunction(lambda i, j: (l2g(i, pi, lM) * l2g(j, pj, lN) % N) / N, shape=(lM, lN), dtype=datatype)
    u1 = np.fromfunction(lambda i: l2g(i, pi, lM), shape=(lM, ), dtype=datatype)
    u2 = np.fromfunction(lambda i: ((l2g(i, pi, lM) + 1) / fn) / 2.0, shape=(lM, ), dtype=datatype)
    v1 = np.fromfunction(lambda i: ((l2g(i, pj, lN) + 1) / fn) / 4.0, shape=(lN, ), dtype=datatype)
    v2 = np.fromfunction(lambda i: ((l2g(i, pj, lN) + 1) / fn) / 6.0, shape=(lN, ), dtype=datatype)
    w = np.zeros((N, ), dtype=datatype)
    x = np.zeros((N, ), dtype=datatype)
    y = np.fromfunction(lambda i: ((i + 1) / fn) / 8.0, shape=(N, ), dtype=datatype)
    z = np.fromfunction(lambda i: ((i + 1) / fn) / 9.0, shape=(N, ), dtype=datatype)
    return alpha, beta, A, u1, u2, v1, v2, w, x, y, z


def run_gemver(validate=False):

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    NPx, NPy = grid[size]
    cart_comm = commworld.Create_cart((NPx, NPy))
    i, j = cart_comm.Get_coords(rank)

    sizes = gemver_sizes
    if validate:
        sizes = [int_ceil(s, 100) for s in gemver_sizes]

    if rank == 0:
        print("===== gemver =====")
        print("sizes: {}".format(sizes), flush=True)

    N = sizes[0]
    N = adjust_size(N, lambda x: np.sqrt(x), size, max(NPx, NPy))
    if rank == 0:
        print("adjusted sizes: {}".format((N, )), flush=True)

    lNx, lNy = N // NPx, N // NPy
    alpha, beta, lA, lu1, lu2, lv1, lv2, w, x, y, z = gemver_distr_init(N, lNx, lNy, np.float64, i, j)
    if rank == 0:
        print("data initialized", flush=True)

    func = optimize_compile(gemver, rank, commworld)

    ldict = locals()
    commworld.Barrier()
    func(alpha=alpha,
         beta=beta,
         A=lA,
         u1=lu1,
         v1=lv1,
         u2=lu2,
         v2=lv2,
         w=w,
         x=x,
         y=y,
         z=z,
         LMx=lNx,
         LNy=lNy,
         GM=N,
         GN=N,
         Px=NPx,
         Py=NPy)

    if validate and rank == 0:
        wval = np.copy(w)
        xval = np.copy(x)

    stmt = (
        "func(alpha=alpha, beta=beta, A=lA, u1=lu1, v1=lv1, u2=lu2, v2=lv2, w=w, x=x, y=y, z=z, LMx=lNx, LNy=lNy, GM=N, GN=N, Px=NPx, Py=NPy); commworld.Barrier()"
    )
    setup = "commworld.Barrier()"
    repeat = 10

    raw_time_list = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
    raw_time = np.median(raw_time_list)

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time), flush=True)
        write_time("gemver", (N, ), raw_time_list)

    if validate and rank == 0:
        alpha, beta, A, u1, u2, v1, v2, wref, xref, yref, zref = gemver_shmem_init(N, np.float64)
        A[:] += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
        tmp1 = A.T @ yref
        xref[:] += beta * tmp1 + zref
        tmp2 = A @ xref
        wref[:] += alpha * tmp2
        if np.allclose(wval, wref) and np.allclose(xval, xref):
            print("Validation: OK!", flush=True)
        else:
            print("Validation: Failed!", flush=True)


# gesummv
gesummv_sizes = [22400]


@dace.program
def gesummv(alpha: dace.float64, beta: dace.float64, A: dace.float64[LMx, LNy], B: dace.float64[LMx, LNy],
            x: dace.float64[GN], y: dace.float64[GM]):
    tmp1 = dace.distr.MatMult(A, x, (Px * LMx, Py * LNy), c_block_sizes=(GM, 1))
    tmp2 = dace.distr.MatMult(B, x, (GM, GN), c_block_sizes=(GM, 1))
    y[:] = alpha * tmp1 + beta * tmp2


def gesummv_shmem_init(N, datatype):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, shape=(N, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * j + 2) % N) / N, shape=(N, N), dtype=datatype)
    x = np.fromfunction(lambda i: (i % N) / N, shape=(N, ), dtype=datatype)
    y = np.empty((N, ), dtype=datatype)
    return alpha, beta, A, B, x, y


def gesummv_distr_init(N, lM, lN, datatype, pi, pj):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((l2g(i, pi, lM) * l2g(j, pj, lN) + 1) % N) / N, shape=(lM, lN), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((l2g(i, pi, lM) * l2g(j, pj, lN) + 2) % N) / N, shape=(lM, lN), dtype=datatype)
    x = np.fromfunction(lambda i: (i % N) / N, shape=(N, ), dtype=datatype)
    y = np.empty((N, ), dtype=datatype)
    return alpha, beta, A, B, x, y


def run_gesummv(validate=False):

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    NPx, NPy = grid[size]
    cart_comm = commworld.Create_cart((NPx, NPy))
    i, j = cart_comm.Get_coords(rank)

    sizes = gesummv_sizes
    if validate:
        sizes = [int_ceil(s, 100) for s in gesummv_sizes]

    if rank == 0:
        print("===== gesummv =====")
        print("sizes: {}".format(sizes), flush=True)

    N = sizes[0]
    N = adjust_size(N, lambda x: np.sqrt(x), size, max(NPx, NPy))
    if rank == 0:
        print("adjusted sizes: {}".format((N, )), flush=True)

    lNx, lNy = N // NPx, N // NPy
    alpha, beta, lA, lB, x, y = gesummv_distr_init(N, lNx, lNy, np.float64, i, j)
    if rank == 0:
        print("data initialized", flush=True)

    func = optimize_compile(gesummv, rank, commworld)

    ldict = locals()
    commworld.Barrier()
    func(alpha=alpha, beta=beta, A=lA, B=lB, x=x, y=y, LMx=lNx, LNy=lNy, GM=N, GN=N, Px=NPx, Py=NPy)

    if validate and rank == 0:
        yval = np.copy(y)

    stmt = (
        "func(alpha=alpha, beta=beta, A=lA, B=lB, x=x, y=y, LMx=lNx, LNy=lNy, GM=N, GN=N, Px=NPx, Py=NPy); commworld.Barrier()"
    )
    setup = "commworld.Barrier()"
    repeat = 10

    raw_time_list = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
    raw_time = np.median(raw_time_list)

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time), flush=True)
        write_time("gesummv", (N, ), raw_time_list)

    if validate and rank == 0:
        alpha, beta, A, B, xref, yref = gesummv_shmem_init(N, np.float64)
        yref[:] = alpha * A @ xref + beta * B @ xref
        if np.allclose(yval, yref):
            print("Validation: OK!", flush=True)
        else:
            print("Validation: Failed!", flush=True)


# 2mm
k2mm_sizes = [6400, 7200, 4400, 4800]


@dace.program
def k2mm(alpha: dace.float64, beta: dace.float64, A: dace.float64[LMx, LKy], B: dace.float64[LKx, LNy],
         C: dace.float64[LNx, LRy], D: dace.float64[LMx, LRy]):
    tmp = dace.distr.MatMult(A, B, (LMx * Px, LNy * Py, GK))
    D[:] = alpha * dace.distr.MatMult(tmp, C, (GM, GR, GN)) + beta * D


def k2mm_shmem_init(NI, NJ, NK, NL, datatype):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI, shape=(NI, NK), dtype=datatype)
    B = np.fromfunction(lambda i, j: (i * (j + 1) % NJ) / NJ, shape=(NK, NJ), dtype=datatype)
    C = np.fromfunction(lambda i, j: ((i * (j + 3) + 1) % NL) / NL, shape=(NJ, NL), dtype=datatype)
    D = np.fromfunction(lambda i, j: (i * (j + 2) % NK) / NK, shape=(NI, NL), dtype=datatype)
    return alpha, beta, A, B, C, D


def k2mm_distr_init(NI, NJ, NK, NL, lNI, lNJ, lNJx, lNKa, lNKb, lNL, datatype, pi, pj):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((l2g(i, pi, lNI) * l2g(j, pj, lNKa) + 1) % NI) / NI,
                        shape=(lNI, lNKa),
                        dtype=datatype)
    B = np.fromfunction(lambda i, j: (l2g(i, pi, lNKb) * (l2g(j, pj, lNJ) + 1) % NJ) / NJ,
                        shape=(lNKb, lNJ),
                        dtype=datatype)
    C = np.fromfunction(lambda i, j: ((l2g(i, pi, lNJx) * (l2g(j, pj, lNL) + 3) + 1) % NL) / NL,
                        shape=(lNJx, lNL),
                        dtype=datatype)
    D = np.fromfunction(lambda i, j: (l2g(i, pi, lNI) * (l2g(j, pj, lNL) + 2) % NK) / NK,
                        shape=(lNI, lNL),
                        dtype=datatype)
    return alpha, beta, A, B, C, D


def run_k2mm(validate=False):

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    NPx, NPy = grid[size]
    cart_comm = commworld.Create_cart((NPx, NPy))
    i, j = cart_comm.Get_coords(rank)

    sizes = k2mm_sizes
    if validate:
        sizes = [int_ceil(s, 100) for s in k2mm_sizes]

    if rank == 0:
        print("===== k2mm =====")
        print("sizes: {}".format(sizes), flush=True)

    M, N, K, R = sizes
    M = adjust_size(M, lambda x: np.cbrt(x), size, max(NPx, NPy))
    N = adjust_size(N, lambda x: np.cbrt(x), size, max(NPx, NPy))
    K = adjust_size(K, lambda x: np.cbrt(x), size, max(NPx, NPy))
    R = adjust_size(R, lambda x: np.cbrt(x), size, max(NPx, NPy))
    if rank == 0:
        print("adjusted sizes: {}".format((M, N, K, R)), flush=True)

    lMx, lNx, lNy, lKx, lKy, lRy = M // NPx, N // NPx, N // NPy, K // NPx, K // NPy, R // NPy
    alpha, beta, lA, lB, lC, lD = k2mm_distr_init(M, N, K, R, lMx, lNy, lNx, lKy, lKx, lRy, np.float64, i, j)
    if rank == 0:
        print("data initialized", flush=True)

    func = optimize_compile(k2mm, rank, commworld)

    ldict = locals()
    commworld.Barrier()
    func(alpha=alpha,
         beta=beta,
         A=lA,
         B=lB,
         C=lC,
         D=lD,
         LMx=lMx,
         LNy=lNy,
         LKx=lKx,
         LKy=lKy,
         LNx=lNx,
         LRy=lRy,
         GM=M,
         GN=N,
         GK=K,
         GR=R,
         Px=NPx,
         Py=NPy)

    if validate:
        lDval = np.copy(lD)

    stmt = (
        "func(alpha=alpha, beta=beta, A=lA, B=lB, C=lC, D=lD, LMx=lMx, LNy=lNy, LKx=lKx, LKy=lKy, LNx=lNx, LRy=lRy, GM=M, GN=N, GK=K, GR=R, Px=NPx, Py=NPy); commworld.Barrier()"
    )
    setup = "commworld.Barrier()"
    repeat = 10

    raw_time_list = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
    raw_time = np.median(raw_time_list)

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time), flush=True)
        write_time("k2mm", (M, N, K), raw_time_list)

    if validate:
        alpha, beta, Aref, Bref, Cref, Dref = k2mm_shmem_init(M, N, K, R, np.float64)
        Dref[:] = alpha * Aref @ Bref @ Cref + beta * Dref
        lDref = Dref[i * lMx:(i + 1) * lMx, j * lRy:(j + 1) * lRy]
        if np.allclose(lDval, lDref):
            print(f"Validation (rank {rank}): OK!", flush=True)
        else:
            print(f"Validation (rank {rank}): Failed!", flush=True)


# 3mm
k3mm_sizes = [6400, 7200, 4000, 4400, 4800]


@dace.program
def k3mm(A: dace.float64[LMx, LKy], B: dace.float64[LKx, LNy], C: dace.float64[LNx, LRy], D: dace.float64[LRx, LSy],
         E: dace.float64[LMx, LSy]):
    tmp1 = dace.distr.MatMult(A, B, (LMx * Px, LNy * Py, GK))
    tmp2 = dace.distr.MatMult(tmp1, C, (GM, GR, GN))
    E[:] = dace.distr.MatMult(tmp2, D, (GM, GS, GR))


def k3mm_shmem_init(NI, NJ, NK, NM, NL, datatype):
    A = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / (5 * NI), shape=(NI, NK), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * (j + 1) + 2) % NJ) / (5 * NJ), shape=(NK, NJ), dtype=datatype)
    C = np.fromfunction(lambda i, j: (i * (j + 3) % NL) / (5 * NL), shape=(NJ, NM), dtype=datatype)
    D = np.fromfunction(lambda i, j: ((i * (j + 2) + 2) % NK) / (5 * NK), shape=(NM, NL), dtype=datatype)
    E = np.empty((NI, NL), dtype=datatype)
    return A, B, C, D, E


def k3mm_distr_init(NI, NJ, NK, NM, NL, lNI, lNJ, lNJx, lNKa, lNKb, lNMx, lNMy, lNL, datatype, pi, pj):
    A = np.fromfunction(lambda i, j: ((l2g(i, pi, lNI) * l2g(j, pj, lNKa) + 1) % NI) / (5 * NI),
                        shape=(lNI, lNKa),
                        dtype=datatype)
    B = np.fromfunction(lambda i, j: ((l2g(i, pi, lNKb) * (l2g(j, pj, lNJ) + 1) + 2) % NJ) / (5 * NJ),
                        shape=(lNKb, lNJ),
                        dtype=datatype)
    C = np.fromfunction(lambda i, j: (l2g(i, pi, lNJx) * (l2g(j, pj, lNMy) + 3) % NL) / (5 * NL),
                        shape=(lNJx, lNMy),
                        dtype=datatype)
    D = np.fromfunction(lambda i, j: ((l2g(i, pi, lNMx) * (l2g(j, pj, lNL) + 2) + 2) % NK) / (5 * NK),
                        shape=(lNMx, lNL),
                        dtype=datatype)
    E = np.empty((lNI, lNL), dtype=datatype)
    return A, B, C, D, E


def run_k3mm(validate=False):

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    NPx, NPy = grid[size]
    cart_comm = commworld.Create_cart((NPx, NPy))
    i, j = cart_comm.Get_coords(rank)

    sizes = k3mm_sizes
    if validate:
        sizes = [int_ceil(s, 100) for s in k3mm_sizes]

    if rank == 0:
        print("===== k3mm =====")
        print("sizes: {}".format(sizes), flush=True)

    M, N, K, R, S = sizes
    M = adjust_size(M, lambda x: np.cbrt(x), size, max(NPx, NPy))
    N = adjust_size(N, lambda x: np.cbrt(x), size, max(NPx, NPy))
    K = adjust_size(K, lambda x: np.cbrt(x), size, max(NPx, NPy))
    R = adjust_size(R, lambda x: np.cbrt(x), size, max(NPx, NPy))
    S = adjust_size(S, lambda x: np.cbrt(x), size, max(NPx, NPy))
    if rank == 0:
        print("adjusted sizes: {}".format((M, N, K, R, S)), flush=True)

    lMx, lNx, lNy, lKx, lKy, lRx, lRy, lSy = M // NPx, N // NPx, N // NPy, K // NPx, K // NPy, R // NPx, R // NPy, S // NPy
    lA, lB, lC, lD, lE = k3mm_distr_init(M, N, K, R, S, lMx, lNy, lNx, lKy, lKx, lRx, lRy, lSy, np.float64, i, j)

    if rank == 0:
        print("data initialized", flush=True)

    func = optimize_compile(k3mm, rank, commworld)

    ldict = locals()
    commworld.Barrier()
    func(A=lA,
         B=lB,
         C=lC,
         D=lD,
         E=lE,
         LMx=lMx,
         LNy=lNy,
         LKx=lKx,
         LKy=lKy,
         LNx=lNx,
         LRx=lRx,
         LRy=lRy,
         LSy=lSy,
         GM=M,
         GN=N,
         GK=K,
         GR=R,
         GS=S,
         Px=NPx,
         Py=NPy)

    stmt = (
        "func(A=lA, B=lB, C=lC, D=lD, E=lE, LMx=lMx, LNy=lNy, LKx=lKx, LKy=lKy, LNx=lNx, LRx=lRx, LRy=lRy, LSy=lSy, GM=M, GN=N, GK=K, GR=R, GS=S, Px=NPx, Py=NPy); commworld.Barrier()"
    )
    setup = "commworld.Barrier()"
    repeat = 10

    raw_time_list = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
    raw_time = np.median(raw_time_list)

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time), flush=True)
        write_time("k3mm", (M, N, K), raw_time_list)

    if validate:
        Aref, Bref, Cref, Dref, Eref = k3mm_shmem_init(M, N, K, R, S, np.float64)
        Eref[:] = ((Aref @ Bref) @ Cref) @ Dref
        lEref = Eref[i * lMx:(i + 1) * lMx, j * lSy:(j + 1) * lSy]
        if np.allclose(lE, lEref):
            print(f"Validation (rank {rank}): OK!", flush=True)
        else:
            print(f"Validation (rank {rank}): Failed!", flush=True)


# mvt
mvt_sizes = [22000]


@dace.program
def mvt(x1: dace.float64[GM], x2: dace.float64[GN], y_1: dace.float64[GN], y_2: dace.float64[GM], A: dace.float64[LMx,
                                                                                                                  LNy]):
    tmp1 = dace.distr.MatMult(A, y_1, (Px * LMx, Py * LNy), c_block_sizes=(GM, 1))
    tmp2 = dace.distr.MatMult(y_2, A, (GM, GN), c_block_sizes=(GN, 1))
    x1 += tmp1
    x2 += tmp2


def mvt_shmem_init(N, datatype):
    x1 = np.fromfunction(lambda i: (i % N) / N, shape=(N, ), dtype=datatype)
    x2 = np.fromfunction(lambda i: ((i + 1) % N) / N, shape=(N, ), dtype=datatype)
    y_1 = np.fromfunction(lambda i: ((i + 3) % N) / N, shape=(N, ), dtype=datatype)
    y_2 = np.fromfunction(lambda i: ((i + 4) % N) / N, shape=(N, ), dtype=datatype)
    A = np.fromfunction(lambda i, j: (i * j % N) / N, shape=(N, N), dtype=datatype)
    return x1, x2, y_1, y_2, A


def mvt_distr_init(N, lM, lN, datatype, pi, pj):
    x1 = np.fromfunction(lambda i: (i % N) / N, shape=(N, ), dtype=datatype)
    x2 = np.fromfunction(lambda i: ((i + 1) % N) / N, shape=(N, ), dtype=datatype)
    y_1 = np.fromfunction(lambda i: ((i + 3) % N) / N, shape=(N, ), dtype=datatype)
    y_2 = np.fromfunction(lambda i: ((i + 4) % N) / N, shape=(N, ), dtype=datatype)
    A = np.fromfunction(lambda i, j: (l2g(i, pi, lM) * l2g(j, pj, lN) % N) / N, shape=(lM, lN), dtype=datatype)
    return x1, x2, y_1, y_2, A


def run_mvt(validate=False):

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    NPx, NPy = grid[size]
    cart_comm = commworld.Create_cart((NPx, NPy))
    i, j = cart_comm.Get_coords(rank)

    sizes = mvt_sizes
    if validate:
        sizes = [int_ceil(s, 100) for s in mvt_sizes]

    if rank == 0:
        print("===== mvt =====")
        print("sizes: {}".format(sizes), flush=True)

    N = sizes[0]
    N = adjust_size(N, lambda x: np.sqrt(x), size, max(NPx, NPy))
    if rank == 0:
        print("adjusted sizes: {}".format((N, )), flush=True)

    lNx, lNy = N // NPx, N // NPy
    x1, x2, y_1, y_2, lA = mvt_distr_init(N, lNx, lNy, np.float64, i, j)
    if rank == 0:
        print("data initialized", flush=True)

    func = optimize_compile(mvt, rank, commworld)

    ldict = locals()
    commworld.Barrier()
    func(x1=x1, x2=x2, y_1=y_1, y_2=y_2, A=lA, LMx=lNx, LNy=lNy, GM=N, GN=N, Px=NPx, Py=NPy)

    if validate and rank == 0:
        x1val = np.copy(x1)
        x2val = np.copy(x2)

    stmt = (
        "func(x1=x1, x2=x2, y_1=y_1, y_2=y_2, A=lA, LMx=lNx, LNy=lNy, GM=N, GN=N, Px=NPx, Py=NPy); commworld.Barrier()")
    setup = "commworld.Barrier()"
    repeat = 10

    raw_time_list = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
    raw_time = np.median(raw_time_list)

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time), flush=True)
        write_time("mvt", (N, ), raw_time_list)

    if validate and rank == 0:
        x1ref, x2ref, y_1ref, y_2ref, A = mvt_shmem_init(N, np.float64)
        x1ref[:] += A @ y_1ref
        x2ref[:] += A.T @ y_2ref
        if np.allclose(x1val, x1ref) and np.allclose(x2val, x2ref):
            print("Validation: OK!", flush=True)
        else:
            print("Validation: Failed!", flush=True)


# jacobi_1d
jacobi_1d_sizes = [1000, 24000]


@dace.program
def jacobi_1d(TSTEPS: dace.int32, A: dace.float64[LNx + 2], B: dace.float64[LNx + 2]):
    req = np.empty((4, ), dtype=MPI_Request)
    for _ in range(1, TSTEPS):
        dace.comm.Isend(A[1], nw, 3, req[0])
        dace.comm.Isend(A[-2], ne, 2, req[1])
        dace.comm.Irecv(A[0], nw, 2, req[2])
        dace.comm.Irecv(A[-1], ne, 3, req[3])
        dace.comm.Waitall(req)
        B[1 + woff:-1 - eoff] = 0.33333 * (A[woff:-2 - eoff] + A[1 + woff:-1 - eoff] + A[2 + woff:-eoff])
        dace.comm.Isend(B[1], nw, 3, req[0])
        dace.comm.Isend(B[-2], ne, 2, req[1])
        dace.comm.Irecv(B[0], nw, 2, req[2])
        dace.comm.Irecv(B[-1], ne, 3, req[3])
        dace.comm.Waitall(req)
        A[1 + woff:-1 - eoff] = 0.33333 * (B[woff:-2 - eoff] + B[1 + woff:-1 - eoff] + B[2 + woff:-eoff])


def jacobi_1d_shmem_init(N, datatype):
    A = np.fromfunction(lambda i: (i + 2) / N, shape=(N, ), dtype=datatype)
    B = np.fromfunction(lambda i: (i + 3) / N, shape=(N, ), dtype=datatype)
    return A, B


def jacobi_1d_distr_init(N, lN, datatype, p):
    A = np.zeros((lN + 2, ), dtype=datatype)
    B = np.zeros((lN + 2, ), dtype=datatype)
    A[1:-1] = np.fromfunction(lambda i: (l2g(i, p, lN) + 2) / N, shape=(lN, ), dtype=datatype)
    B[1:-1] = np.fromfunction(lambda i: (l2g(i, p, lN) + 3) / N, shape=(lN, ), dtype=datatype)
    return A, B


def run_jacobi_1d(validate=False):

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    cart_comm = commworld.Create_cart((size, 1))
    i, j = cart_comm.Get_coords(rank)

    woff = eoff = 0
    nw = rank - 1
    ne = rank + 1
    if rank == 0:
        woff = 1
        nw = MPI.PROC_NULL
    if rank == size - 1:
        eoff = 1
        ne = MPI.PROC_NULL

    sizes = jacobi_1d_sizes
    if validate:
        sizes = [int_ceil(s, 100) for s in jacobi_1d_sizes]

    if rank == 0:
        print("===== jacobi_1d =====")
        print("sizes: {}".format(sizes), flush=True)

    TSTEPS, N = sizes
    N = adjust_size(N, lambda x: x, size, size)
    if rank == 0:
        print("adjusted sizes: {}".format((TSTEPS, N)), flush=True)

    lN = N // size
    lA, lB = jacobi_1d_distr_init(N, lN, np.float64, rank)
    if rank == 0:
        print("data initialized", flush=True)

    func = optimize_compile(jacobi_1d, rank, commworld)

    ldict = locals()
    commworld.Barrier()
    func(A=lA, B=lB, TSTEPS=TSTEPS, LNx=lN, GN=N, nw=nw, ne=ne, woff=woff, eoff=eoff)

    if validate:
        lAval = np.copy(lA[1:-1])
        lBval = np.copy(lB[1:-1])

    stmt = ("func(A=lA, B=lB, TSTEPS=TSTEPS, LNx=lN, GN=N, nw=nw, ne=ne, woff=woff, eoff=eoff); commworld.Barrier()")
    setup = "commworld.Barrier()"
    repeat = 10

    raw_time_list = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
    raw_time = np.median(raw_time_list)

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time), flush=True)
        write_time("jacobi_1d", (TSTEPS, N), raw_time_list)

    if validate:
        Aref, Bref = jacobi_1d_shmem_init(N, np.float64)
        for _ in range(1, TSTEPS):
            Bref[1:-1] = 0.33333 * (Aref[:-2] + Aref[1:-1] + Aref[2:])
            Aref[1:-1] = 0.33333 * (Bref[:-2] + Bref[1:-1] + Bref[2:])
        lAref = Aref[i * lN:(i + 1) * lN]
        lBref = Bref[i * lN:(i + 1) * lN]

        if np.allclose(lAval, lAref) and np.allclose(lBval, lBref):
            print(f"Validation (rank {rank}): OK!", flush=True)
        else:
            print(f"Validation (rank {rank}): Failed!", flush=True)


# jacobi_2d
jacobi_2d_sizes = [1000, 500]


@dace.program
def jacobi_2d(TSTEPS: dace.int32, A: dace.float64[LMx + 2, LNy + 2], B: dace.float64[LMx + 2, LNy + 2]):
    req = np.empty((8, ), dtype=MPI_Request)
    for _ in range(1, TSTEPS):
        dace.comm.Isend(A[1, 1:-1], nn, 0, req[0])
        dace.comm.Isend(A[-2, 1:-1], ns, 1, req[1])
        dace.comm.Isend(A[1:-1, 1], nw, 2, req[2])
        dace.comm.Isend(A[1:-1, -2], ne, 3, req[3])
        dace.comm.Irecv(A[0, 1:-1], nn, 1, req[4])
        dace.comm.Irecv(A[-1, 1:-1], ns, 0, req[5])
        dace.comm.Irecv(A[1:-1, 0], nw, 3, req[6])
        dace.comm.Irecv(A[1:-1, -1], ne, 2, req[7])
        dace.comm.Waitall(req)

        B[1 + noff:-1 - soff,
          1 + woff:-1 - eoff] = 0.2 * (A[1 + noff:-1 - soff, 1 + woff:-1 - eoff] +
                                       A[1 + noff:-1 - soff, woff:-2 - eoff] + A[1 + noff:-1 - soff, 2 + woff:-eoff] +
                                       A[2 + noff:-soff, 1 + woff:-1 - eoff] + A[noff:-2 - soff, 1 + woff:-1 - eoff])

        dace.comm.Isend(B[1, 1:-1], nn, 0, req[0])
        dace.comm.Isend(B[-2, 1:-1], ns, 1, req[1])
        dace.comm.Isend(B[1:-1, 1], nw, 2, req[2])
        dace.comm.Isend(B[1:-1, -2], ne, 3, req[3])
        dace.comm.Irecv(B[0, 1:-1], nn, 1, req[4])
        dace.comm.Irecv(B[-1, 1:-1], ns, 0, req[5])
        dace.comm.Irecv(B[1:-1, 0], nw, 3, req[6])
        dace.comm.Irecv(B[1:-1, -1], ne, 2, req[7])
        dace.comm.Waitall(req)

        A[1 + noff:-1 - soff,
          1 + woff:-1 - eoff] = 0.2 * (B[1 + noff:-1 - soff, 1 + woff:-1 - eoff] +
                                       B[1 + noff:-1 - soff, woff:-2 - eoff] + B[1 + noff:-1 - soff, 2 + woff:-eoff] +
                                       B[2 + noff:-soff, 1 + woff:-1 - eoff] + B[noff:-2 - soff, 1 + woff:-1 - eoff])


def jacobi_2d_shmem_init(N, datatype):
    A = np.fromfunction(lambda i, j: i * (j + 2) / N, shape=(N, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: i * (j + 3) / N, shape=(N, N), dtype=datatype)
    return A, B


def jacobi_2d_distr_init(N, lM, lN, datatype, pi, pj):
    A = np.zeros((lM + 2, lN + 2), dtype=datatype)
    B = np.zeros((lM + 2, lN + 2), dtype=datatype)
    A[1:-1, 1:-1] = np.fromfunction(lambda i, j: l2g(i, pi, lM) * (l2g(j, pj, lN) + 2) / N,
                                    shape=(lM, lN),
                                    dtype=datatype)
    B[1:-1, 1:-1] = np.fromfunction(lambda i, j: l2g(i, pi, lM) * (l2g(j, pj, lN) + 3) / N,
                                    shape=(lM, lN),
                                    dtype=datatype)
    return A, B


def run_jacobi_2d(validate=False):

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    NPx, NPy = grid[size]
    cart_comm = commworld.Create_cart((NPx, NPy))
    i, j = cart_comm.Get_coords(rank)

    noff = soff = woff = eoff = 0
    nn = (i - 1) * NPy + j
    ns = (i + 1) * NPy + j
    nw = i * NPy + (j - 1)
    ne = i * NPy + (j + 1)
    if i == 0:
        noff = 1
        nn = MPI.PROC_NULL
    if i == NPx - 1:
        soff = 1
        ns = MPI.PROC_NULL
    if j == 0:
        woff = 1
        nw = MPI.PROC_NULL
    if j == NPy - 1:
        eoff = 1
        ne = MPI.PROC_NULL

    sizes = jacobi_2d_sizes
    if validate:
        sizes = [int_ceil(s, 100) for s in jacobi_2d_sizes]

    if rank == 0:
        print("===== jacobi_2d =====")
        print("sizes: {}".format(sizes), flush=True)

    TSTEPS, N = sizes
    N = adjust_size(N, lambda x: x, size, size)
    if rank == 0:
        print("adjusted sizes: {}".format((TSTEPS, N)), flush=True)

    lMx, lNy = N // NPx, N // NPy
    lA, lB = jacobi_2d_distr_init(N, lMx, lNy, np.float64, i, j)
    if rank == 0:
        print("data initialized", flush=True)

    func = optimize_compile(jacobi_2d, rank, commworld)

    ldict = locals()
    commworld.Barrier()
    func(A=lA,
         B=lB,
         TSTEPS=TSTEPS,
         LMx=lMx,
         LNy=lNy,
         noff=noff,
         soff=soff,
         woff=woff,
         eoff=eoff,
         nn=nn,
         ns=ns,
         nw=nw,
         ne=ne)

    if validate:
        lAval = np.copy(lA[1:-1, 1:-1])
        lBval = np.copy(lB[1:-1, 1:-1])

    stmt = (
        "func(A=lA, B=lB, TSTEPS=TSTEPS, LMx=lMx, LNy=lNy, noff=noff, soff=soff, woff=woff, eoff=eoff, nn=nn, ns=ns, nw=nw, ne=ne); commworld.Barrier()"
    )
    setup = "commworld.Barrier()"
    repeat = 10

    raw_time_list = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
    raw_time = np.median(raw_time_list)

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time), flush=True)
        write_time("jacobi_2d", (TSTEPS, N), raw_time_list)

    if validate:
        Aref, Bref = jacobi_2d_shmem_init(N, np.float64)
        for _ in range(1, TSTEPS):
            Bref[1:-1,
                 1:-1] = 0.2 * (Aref[1:-1, 1:-1] + Aref[1:-1, :-2] + Aref[1:-1, 2:] + Aref[2:, 1:-1] + Aref[:-2, 1:-1])
            Aref[1:-1,
                 1:-1] = 0.2 * (Bref[1:-1, 1:-1] + Bref[1:-1, :-2] + Bref[1:-1, 2:] + Bref[2:, 1:-1] + Bref[:-2, 1:-1])
        lAref = Aref[i * lMx:(i + 1) * lMx, j * lNy:(j + 1) * lNy]
        lBref = Bref[i * lMx:(i + 1) * lMx, j * lNy:(j + 1) * lNy]

        if np.allclose(lAval, lAref) and np.allclose(lBval, lBref):
            print(f"Validation (rank {rank}): OK!", flush=True)
        else:
            print(f"Validation (rank {rank}): Failed!", flush=True)


if __name__ == '__main__':

    validate = True

    run_atax(validate)
    run_bicg(validate)
    run_doitgen(validate)
    run_gemm(validate)
    run_gemver(validate)
    run_gesummv(validate)
    run_k2mm(validate)
    run_k3mm(validate)
    run_mvt(validate)
    run_jacobi_1d(validate)
    run_jacobi_2d(validate)
