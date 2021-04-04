# ===== Imports =====

import numpy as np
import dace as dc
import timeit

from mpi4py import MPI
from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL


# ===== Symbols =====

# Process grid
P = dc.symbol('P', dtype=dc.int32, integer=True, positive=True)
Px = dc.symbol('Px', dtype=dc.int32, integer=True, positive=True)
Py = dc.symbol('Py', dtype=dc.int32, integer=True, positive=True)

# Matrix-matrix products
lNI = dc.symbol('lNI', dtype=dc.int32, integer=True, positive=True)
lNJ = dc.symbol('lNJ', dtype=dc.int32, integer=True, positive=True)
lNJx = dc.symbol('lNJx', dtype=dc.int32, integer=True, positive=True)
lNKa = dc.symbol('lNKa', dtype=dc.int32, integer=True, positive=True)
lNKb = dc.symbol('lNKb', dtype=dc.int32, integer=True, positive=True)
lNL = dc.symbol('lNL', dtype=dc.int32, integer=True, positive=True)
lNMx = dc.symbol('lNMx', dtype=dc.int32, integer=True, positive=True)
lNMy = dc.symbol('lNMy', dtype=dc.int32, integer=True, positive=True)
NI = lNI * Px
NJ = lNJ * Py  # == lNJx * Px
NK = lNKa * Py  # == lNKb * Px
NL = lNL * Py
NM = lNMx * Px  # == lNMy * Py

# Matrix-vector products and others
lM = dc.symbol('lM', dtype=dc.int64, integer=True, positive=True)
lN = dc.symbol('lN', dtype=dc.int64, integer=True, positive=True)
lMy = dc.symbol('lMy', dtype=dc.int64, integer=True, positive=True)
lNx = dc.symbol('lNx', dtype=dc.int64, integer=True, positive=True)
M = lM * Px  # == lMy * Py
N = lN * Py  # == lNx * Px


# ===== Helper methods =====

def relerr(ref, val):
    return np.linalg.norm(ref-val) / np.linalg.norm(ref)

def time_to_ms(raw):
    return int(round(raw * 1000))

def l2g(idx, pidx, bsize):
    return idx + pidx * bsize

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


# ===== Programs ==============================================================

# ===== atax =====

atax_sizes = [[1800, 2200], [3600, 4400], [7200, 8800], [14400, 17600]]

@dc.program
def atax_shmem(A: dc.float64[M, N], x: dc.float64[N], y:dc.float64[N]):
    y[:] = (A @ x) @ A

@dc.program
def atax_distr(A: dc.float64[lM, lN], x: dc.float64[lN], y:dc.float64[lN]):
    tmp = distr.MatMult(A, x, (Px*lM, Py*lN), c_block_sizes=(lMy, 1))
    y[:] = distr.MatMult(tmp, A, (M, N))

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
    x = np.fromfunction(lambda i: 1 + (l2g(i, pj, lN) / fn),
                        shape=(lN,), dtype=datatype)
    y = np.empty((lN,), dtype=datatype)
    return A, x, y

def atax(sizes, validate=True):

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px, Py = grid[size]
    pi = rank // Py
    pj = rank % Py

    if rank == 0:
        print("===== atax =====")
        print("sizes: {}".format(sizes), flush=True)

    M, N = sizes

    # Symbolic sizes
    lM = M // Px
    lN = N // Py
    lNx = N // Py
    lMy = M // Py

    lA, lx, ly = atax_distr_init(M, N, lM, lN, np.float64, pi, pj)

    mpi_sdfg = None
    if rank == 0:
        mpi_sdfg = atax_distr.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_func= mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        mpi_sdfg = dc.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=atax_distr.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=atax_distr.name),
            atax_distr.name))

    ldict = locals()

    comm.Barrier()

    mpi_func(A=lA, x=lx, y=ly,
             lM=lM, lN=lN, lNx=lNx, lMy=lMy, Px=Px, Py=Py)
    
    comm.Barrier()

    if validate:

        if rank == 0:
            y = np.empty((N,), dtype=np.float64)
            y[0:lN] = ly
            for i in range(Py):
                if i == pj:
                    continue
                else:
                    comm.Recv(ly, source=i, tag=i)
                    y[i*lN:(i+1)*lN] = ly
        elif pi == 0:
            comm.Send(ly, dest=0, tag=pj)
        
        comm.Barrier()

    stmt = ("mpi_func(A=lA, x=lx, y=ly, "
            "lM=lM, lN=lN, lNx=lNx, lMy=lMy, Px=Px, Py=Py)")
    setup = "comm.Barrier()"
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

    if validate:

        if rank == 0:
            refA, refx, refy = atax_shmem_init(M, N, np.float64)
            shared_sdfg = atax_shmem.compile()
            shared_sdfg(A=refA, x=refx, y=refy,
                        lM=lM, lN=lN, lNx=lNx, lMy=lMy, Px=Px, Py=Py)
            error = relerr(refy, y)
            print("validation: {} ({})".format(error < 1e-12, error), flush=True)


# ===== bicg =====

bicg_sizes = [[2200, 1800], [4400, 3600], [7200, 8800]]

@dc.program
def bicg_shmem(A: dc.float64[M, N], p: dc.float64[N], r: dc.float64[M],
               o1: dc.float64[N], o2: dc.float64[M]):
    o1[:] = r @ A
    o2[:] = A @ p

@dc.program
def bicg_distr(A: dc.float64[lM, lN], p: dc.float64[lN], r: dc.float64[lMy],
               o1: dc.float64[lN], o2: dc.float64[lMy]):
    o1[:] = distr.MatMult(r, A, (Px*lM, Py*lN))
    o2[:] = distr.MatMult(A, p, (M, N), c_block_sizes=(lMy, 1))

def bicg_shmem_init(M, N, datatype):
    A = np.fromfunction(lambda i, j: (i * (j + 1) % M) / M,
                        shape=(M, N), dtype=datatype)
    p = np.fromfunction(lambda i: (i % N) / N, shape=(N,), dtype=datatype)
    r = np.fromfunction(lambda i: (i % M) / M, shape=(M,), dtype=datatype)
    o1 = np.empty((N,), dtype=datatype)
    o2 = np.empty((M,), dtype=datatype)
    return A, p, r, o1, o2

def bicg_distr_init(M, N, lM, lN, lMy, datatype, pi, pj):
    A = np.fromfunction(lambda i, j: (l2g(i, pi, lM) * (l2g(j, pj, lN) + 1) % M) / M,
                        shape=(lM, lN), dtype=datatype)
    p = np.fromfunction(lambda i: (l2g(i, pj, lN) % N) / N, shape=(lN,), dtype=datatype)
    r = np.fromfunction(lambda i: (l2g(i, pj, lMy) % M) / M, shape=(lMy,), dtype=datatype)
    o1 = np.empty((lN,), dtype=datatype)
    o2 = np.empty((lMy,), dtype=datatype)
    return A, p, r, o1, o2

def bicg(sizes, validate=True):

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px, Py = grid[size]
    pi = rank // Py
    pj = rank % Py

    if rank == 0:
        print("===== bicg =====")
        print("sizes: {}".format(sizes), flush=True)

    M, N = sizes

    # Symbolic sizes
    lM = M // Px
    lN = N // Py
    lNx = N // Py
    lMy = M // Py

    lA, lp, lr, lo1, lo2 = bicg_distr_init(M, N, lM, lN, lMy, np.float64, pi, pj)

    mpi_sdfg = None
    if rank == 0:
        mpi_sdfg = bicg_distr.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_func= mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        mpi_sdfg = dc.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=bicg_distr.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=bicg_distr.name),
            bicg_distr.name))

    ldict = locals()

    comm.Barrier()

    mpi_func(A=lA, p=lp, r=lr, o1=lo1, o2=lo2,
             lM=lM, lN=lN, lNx=lNx, lMy=lMy, Px=Px, Py=Py)
    
    comm.Barrier()

    if validate:

        if rank == 0:
            o1 = np.empty((N,), dtype=np.float64)
            o2 = np.empty((M,), dtype=np.float64)
            o1[0:lN] = lo1
            o2[0:lMy] = lo2
            for i in range(Py):
                if i == pj:
                    continue
                else:
                    comm.Recv(lo1, source=i, tag=i)
                    o1[i*lN:(i+1)*lN] = lo1
                    comm.Recv(lo2, source=i, tag=i+Py)
                    o2[i*lMy:(i+1)*lMy] = lo2
        elif pi == 0:
            comm.Send(lo1, dest=0, tag=pj)
            comm.Send(lo2, dest=0, tag=pj+Py)
        
        comm.Barrier()

    stmt = ("mpi_func(A=lA, p=lp, r=lr, o1=lo1, o2=lo2, "
            "lM=lM, lN=lN, lNx=lNx, lMy=lMy, Px=Px, Py=Py)")
    setup = "comm.Barrier()"
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

    if validate:

        if rank == 0:
            refA, refp, refr, refo1, refo2 = bicg_shmem_init(M, N, np.float64)
            shared_sdfg = bicg_shmem.compile()
            shared_sdfg(A=refA, p=refp, r=refr, o1=refo1, o2=refo2,
                        lM=lM, lN=lN, lNx=lNx, lMy=lMy, Px=Px, Py=Py)
            error = relerr(refo1, o1)
            print("validation: {} ({})".format(error < 1e-12, error), flush=True)
            error = relerr(refo2, o2)
            print("validation: {} ({})".format(error < 1e-12, error), flush=True)


# ===== doitgen =====

doitgen_sizes = [[256, 250, 270], [512, 500, 540]]

lR, NQ, NP = (dc.symbol(s, dtype=dc.int32, integer=True, positive=True)
              for s in ('lR', 'NQ', 'NP'))
NR = lR * P

@dc.program
def doitgen_shmem(A: dc.float64[NR, NQ, NP], C4: dc.float64[NP, NP]):
    for r in range(lR*P):
        A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, 1, NP)) @ C4, (NQ, NP))

@dc.program
def doitgen_distr(A: dc.float64[lR, NQ, NP], C4: dc.float64[NP, NP]):
    for r in range(lR):
        A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, 1, NP)) @ C4, (NQ, NP))

def doitgen_shmem_init(NR, NQ, NP, datatype):

    A = np.fromfunction(lambda i, j, k: ((i * j + k) % NP) / NP,
                        shape=(NR, NQ, NP), dtype=datatype)
    C4 = np.fromfunction(lambda i, j: (i * j % NP) / NP,
                         shape=(NP, NP,), dtype=datatype)
    return A, C4

def doitgen_distr_init(NR, NQ, NP, lR, datatype, p):

    A = np.fromfunction(lambda i, j, k: ((l2g(i, p, lR) * j + k) % NP) / NP,
                        shape=(lR, NQ, NP), dtype=datatype)
    C4 = np.fromfunction(lambda i, j: (i * j % NP) / NP,
                         shape=(NP, NP,), dtype=datatype)
    return A, C4

def doitgen(sizes, validate=True):

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("===== doitgen =====")
        print("sizes: {}".format(sizes), flush=True)

    NR, NQ, NP = sizes

    # Symbolic sizes
    lR = NR // size

    lA, C4 = doitgen_distr_init(NR, NQ, NP, lR, np.float64, rank)

    mpi_sdfg = None
    if rank == 0:
        mpi_sdfg = doitgen_distr.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_func= mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        mpi_sdfg = dc.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=doitgen_distr.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=doitgen_distr.name),
            doitgen_distr.name))

    ldict = locals()

    comm.Barrier()

    mpi_func(A=lA, C4=C4,
             lR=lR, NQ=NQ, NP=NP, P=size)
    
    comm.Barrier()

    if validate:

        if rank == 0:
            A = np.empty((NR, NQ, NP), dtype=np.float64)
            A[0:lR, :, :] = lA
            for i in range(size):
                if i == 0:
                    continue
                else:
                    comm.Recv(lA, source=i, tag=i)
                    A[i*lR:(i+1)*lR, :, :] = lA
        else:
            comm.Send(lA, dest=0, tag=rank)
        
        comm.Barrier()

    stmt = ("mpi_func(A=lA, C4=C4, "
            "lR=lR, NQ=NQ, NP=NP, P=size)")
    setup = "comm.Barrier()"
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

    if validate:

        if rank == 0:
            refA, refC4 = doitgen_shmem_init(NR, NQ, NP, np.float64)
            shared_sdfg = doitgen_shmem.compile()
            shared_sdfg(A=refA, C4=refC4,
                        lR=lR, NQ=NQ, NP=NP, P=size)
            error = relerr(refA, A)
            print("validation: {} ({})".format(error < 1e-12, error), flush=True)


# ===== gemm =====

gemm_sizes = [[2000, 2300, 2600], [4000, 4600, 5200]]  #, [8000, 9200, 5200]]

@dc.program
def gemm_shmem(alpha: dc.float64, beta: dc.float64, C: dc.float64[NI, NJ],
               A: dc.float64[NI, NK], B: dc.float64[NK, NJ]):
    C[:] = alpha * A @ B + beta * C 

@dc.program
def gemm_distr(alpha: dc.float64, beta: dc.float64, C: dc.float64[lNI, lNJ],
               A: dc.float64[lNI, lNKa], B: dc.float64[lNKb, lNJ]):

    tmp  = distr.MatMult(A, B, (lNI * Px, lNJ * Py, NK))
    C[:] = alpha * tmp + beta * C

def gemm_shmem_init(NI, NJ, NK, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI,
                        shape=(NI, NJ), dtype=datatype)
    A = np.fromfunction(lambda i, k: (i * (k + 1) % NK) / NK,
                        shape=(NI, NK), dtype=datatype)
    B = np.fromfunction(lambda k, j: (k * (j + 2) % NJ) / NJ,
                        shape=(NK, NJ), dtype=datatype)
    return alpha, beta, C, A, B

def gemm_distr_init(NI, NJ, NK, lNI, lNJ, lNKa, lNKb, datatype, pi, pj):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((l2g(i, pi, lNI) * l2g(j, pj, lNJ) + 1) % NI) / NI,
                        shape=(lNI, lNJ), dtype=datatype)
    A = np.fromfunction(lambda i, k: (l2g(i, pi, lNI) * (l2g(k, pj, lNKa) + 1) % NK) / NK,
                        shape=(lNI, lNKa), dtype=datatype)
    B = np.fromfunction(lambda k, j: (l2g(k, pi, lNKb) * (l2g(j, pj, lNJ) + 2) % NJ) / NJ,
                        shape=(lNKb, lNJ), dtype=datatype)
    return alpha, beta, C, A, B

def gemm(sizes, validate=True):

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px, Py = grid[size]
    # Tmp fix for gemm and non-square grids
    if Px < Py:
        Px, Py = Py, Px
    pi = rank // Py
    pj = rank % Py

    if rank == 0:
        print("===== gemm =====")
        print("sizes: {}".format(sizes), flush=True)

    NI, NJ, NK = sizes

    # Symbolic sizes
    lNI = NI // Px
    lNJ = NJ // Py
    lNKa = NK // Py
    lNKb = NK // Px

    alpha, beta, lC, lA, lB = gemm_distr_init(NI, NJ, NK, lNI, lNJ, lNKa, lNKb, np.float64, pi, pj)

    mpi_sdfg = None
    if rank == 0:
        mpi_sdfg = gemm_distr.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_func= mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        mpi_sdfg = dc.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=gemm_distr.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=gemm_distr.name),
            gemm_distr.name))

    ldict = locals()

    comm.Barrier()

    mpi_func(alpha=alpha, beta=beta, C=lC, A=lA, B=lB,
             lNI=lNI, lNJ=lNJ, lNKa=lNKa, lNKb=lNKb, Px=Px, Py=Py)
    
    comm.Barrier()

    if validate:

        if rank == 0:
            C = np.empty((Px, Py, lNI, lNJ), dtype=np.float64)
        else:
            C = None
        comm.Gather(lC, C)
        if rank == 0:
            C = np.transpose(C, (0, 2, 1, 3)).reshape(NI, NJ).copy()
        
        comm.Barrier()

    stmt = ("mpi_func(alpha=alpha, beta=beta, C=lC, A=lA, B=lB, "
            "lNI=lNI, lNJ=lNJ, lNKa=lNKa, lNKb=lNKb, Px=Px, Py=Py)")
    setup = "comm.Barrier()"
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

    if validate:

        if rank == 0:
            alpha, beta, refC, refA, refB = gemm_shmem_init(NI, NJ, NK, np.float64)
            shared_sdfg = gemm_shmem.compile()
            shared_sdfg(alpha=alpha, beta=beta, C=refC, A=refA, B=refB,
                        lNI=lNI, lNJ=lNJ, lNKa=lNKa, lNKb=lNKb, Px=Px, Py=Py)
            error = relerr(refC, C)
            print("validation: {} ({})".format(error < 1e-12, error), flush=True)


# ==== gemver ====

gemver_sizes = [4000, 8000]

@dc.program
def gemver_shmem(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N],
                 u1: dc.float64[N], v1: dc.float64[N], u2: dc.float64[N],
                 v2: dc.float64[N], w: dc.float64[N], x: dc.float64[N],
                 y: dc.float64[N], z: dc.float64[N]):
    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x

# @dc.program
# def gemver_distr(alpha: dc.float64, beta: dc.float64, A: dc.float64[lNx, lN],
#                  u1: dc.float64[lNx], v1: dc.float64[lN], u2: dc.float64[lNx],
#                  v2: dc.float64[lN], w: dc.float64[lNx], x: dc.float64[lN],
#                  y: dc.float64[lNx], z: dc.float64[lN]):
#     A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
#     tmp1 = distr.MatMult(y, A, (Px*lNx, Py*lN))
#     x += beta * tmp1 + z
#     tmp2 = distr.MatMult(A, x, (N, N), c_block_sizes=(lNx, 1))
#     w += alpha * tmp2
@dc.program
def gemver_distr(alpha: dc.float64, beta: dc.float64, A: dc.float64[lM, lN],
                 u1: dc.float64[lM], v1: dc.float64[lN], u2: dc.float64[lM],
                 v2: dc.float64[lN], w: dc.float64[lMy], x: dc.float64[lN],
                 y: dc.float64[lMy], z: dc.float64[lN]):
    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    tmp1 = distr.MatMult(y, A, (Px*lM, Py*lN))
    x += beta * tmp1 + z
    tmp2 = distr.MatMult(A, x, (N, N), c_block_sizes=(lMy, 1))
    w += alpha * tmp2

def gemver_shmem_init(N, datatype):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    fn = datatype(N)
    A = np.fromfunction(lambda i, j: (i * j % N) / N,
                        shape=(N, N), dtype=datatype)
    u1 = np.fromfunction(lambda i: i, shape=(N,), dtype=datatype)
    u2 = np.fromfunction(lambda i: ((i + 1) / fn) / 2.0, shape=(N,), dtype=datatype)
    v1 = np.fromfunction(lambda i: ((i + 1) / fn) / 4.0, shape=(N,), dtype=datatype)
    v2 = np.fromfunction(lambda i: ((i + 1) / fn) / 6.0, shape=(N,), dtype=datatype)
    w = np.zeros((N,), dtype=datatype)
    x = np.zeros((N,), dtype=datatype)
    y = np.fromfunction(lambda i: ((i + 1) / fn) / 8.0, shape=(N,), dtype=datatype)
    z = np.fromfunction(lambda i: ((i + 1) / fn) / 9.0, shape=(N,), dtype=datatype)
    return alpha, beta, A, u1, u2, v1, v2, w, x, y, z

# def gemver_distr_init(N, lNx, lN, datatype, pi, pj):
#     alpha = datatype(1.5)
#     beta = datatype(1.2)
#     fn = datatype(N)
#     A = np.fromfunction(lambda i, j: (l2g(i, pi, lNx) * l2g(j, pj, lN) % N) / N,
#                         shape=(lNx, lN), dtype=datatype)
#     u1 = np.fromfunction(lambda i: l2g(i, pi, lNx), shape=(lNx,), dtype=datatype)
#     u2 = np.fromfunction(lambda i: ((l2g(i, pi, lNx) + 1) / fn) / 2.0, shape=(lNx,), dtype=datatype)
#     v1 = np.fromfunction(lambda i: ((l2g(i, pj, lN) + 1) / fn) / 4.0, shape=(lN,), dtype=datatype)
#     v2 = np.fromfunction(lambda i: ((l2g(i, pj, lN) + 1) / fn) / 6.0, shape=(lN,), dtype=datatype)
#     w = np.zeros((lNx,), dtype=datatype)
#     x = np.zeros((lN,), dtype=datatype)
#     y = np.fromfunction(lambda i: ((l2g(i, pi, lNx) + 1) / fn) / 8.0, shape=(lNx,), dtype=datatype)
#     z = np.fromfunction(lambda i: ((l2g(i, pj, lN) + 1) / fn) / 9.0, shape=(lN,), dtype=datatype)
#     return alpha, beta, A, u1, u2, v1, v2, w, x, y, z
def gemver_distr_init(N, lM, lN, lMy, datatype, pi, pj):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    fn = datatype(N)
    A = np.fromfunction(lambda i, j: (l2g(i, pi, lM) * l2g(j, pj, lN) % N) / N,
                        shape=(lM, lN), dtype=datatype)
    u1 = np.fromfunction(lambda i: l2g(i, pi, lM), shape=(lM,), dtype=datatype)
    u2 = np.fromfunction(lambda i: ((l2g(i, pi, lM) + 1) / fn) / 2.0, shape=(lM,), dtype=datatype)
    v1 = np.fromfunction(lambda i: ((l2g(i, pj, lN) + 1) / fn) / 4.0, shape=(lN,), dtype=datatype)
    v2 = np.fromfunction(lambda i: ((l2g(i, pj, lN) + 1) / fn) / 6.0, shape=(lN,), dtype=datatype)
    w = np.zeros((lMy,), dtype=datatype)
    x = np.zeros((lN,), dtype=datatype)
    y = np.fromfunction(lambda i: ((l2g(i, pj, lMy) + 1) / fn) / 8.0, shape=(lMy,), dtype=datatype)
    z = np.fromfunction(lambda i: ((l2g(i, pj, lN) + 1) / fn) / 9.0, shape=(lN,), dtype=datatype)
    return alpha, beta, A, u1, u2, v1, v2, w, x, y, z

def gemver(sizes, validate=True):

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px, Py = grid[size]
    pi = rank // Py
    pj = rank % Py

    if rank == 0:
        print("===== gemver =====")
        print("sizes: {}".format(sizes), flush=True)

    M = N = sizes

    # Symbolic sizes
    lM = M // Px
    lN = N // Py
    lMy = M // Py

    alpha, beta, lA, lu1, lu2, lv1, lv2, lw, lx, ly, lz = gemver_distr_init(
        N, lM, lN, lMy, np.float64, pi, pj)

    mpi_sdfg = None
    if rank == 0:
        mpi_sdfg = gemver_distr.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_func= mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        mpi_sdfg = dc.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=gemver_distr.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=gemver_distr.name),
            gemver_distr.name))

    ldict = locals()

    comm.Barrier()

    mpi_func(alpha=alpha, beta=beta, A=lA, u1=lu1, v1=lv1, u2=lu2, v2=lv2,
             w=lw, x=lx, y=ly, z=lz,
             lM=lM, lN=lN, lMy=lMy, Px=Px, Py=Py)
    
    comm.Barrier()

    if validate:

        if rank == 0:
            A = np.empty((Px, Py, lM, lN), dtype=np.float64)
        else:
            A = None
        comm.Gather(lA, A)
        if rank == 0:
            A = np.transpose(A, (0, 2, 1, 3)).reshape(N, N).copy()

        if rank == 0:
            x = np.empty((N,), dtype=np.float64)
            w = np.empty((N,), dtype=np.float64)
            x[0:lN] = lx
            w[0:lMy] = lw
            for i in range(Py):
                if i == pj:
                    continue
                else:
                    comm.Recv(lx, source=i, tag=i)
                    x[i*lN:(i+1)*lN] = lx
                    comm.Recv(lw, source=i, tag=i+Py)
                    w[i*lMy:(i+1)*lMy] = lw
        elif pi == 0:
            comm.Send(lx, dest=0, tag=pj)
            comm.Send(lw, dest=0, tag=pj+Py)
        
        comm.Barrier()

    stmt = ("mpi_func(alpha=alpha, beta=beta, A=lA, u1=lu1, v1=lv1, u2=lu2, v2=lv2, "
            "w=lw, x=lx, y=ly, z=lz, "
            "lM=lM, lN=lN, lMy=lMy, Px=Px, Py=Py)")
    setup = "comm.Barrier()"
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

    if validate:

        if rank == 0:
            alpha, beta, refA, refu1, refu2, refv1, refv2, refw, refx, refy, refz = gemver_shmem_init(N, np.float64)
            shared_sdfg = gemver_shmem.compile()
            shared_sdfg(alpha=alpha, beta=beta, A=refA, u1=refu1, v1=refv1,
                        u2=refu2, v2=refv2,
                        w=refw, x=refx, y=refy, z=refz,
                        lM=lM, lN=lN, lMy=lMy, Px=Px, Py=Py)
            error = relerr(refA, A)
            print("validation: {} ({})".format(error < 1e-12, error), flush=True)
            error = relerr(refx, x)
            print("validation: {} ({})".format(error < 1e-12, error), flush=True)
            error = relerr(refw, w)
            print("validation: {} ({})".format(error < 1e-12, error), flush=True)


# ===== gesummv =====

gesummv_sizes = [2800, 5600, 11200]

@dc.program
def gesummv_shmem(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N],
                  B: dc.float64[N, N], x: dc.float64[N], y: dc.float64[N]):
    y[:] = alpha * A @ x + beta * B @ x

@dc.program
def gesummv_distr(alpha: dc.float64, beta: dc.float64, A: dc.float64[lM, lN],
                  B: dc.float64[lM, lN], x: dc.float64[lN], y: dc.float64[lMy]):
    
    tmp1 = distr.MatMult(A, x, (Px*lM, Py*lN), c_block_sizes=(lMy, 1))
    tmp2 = distr.MatMult(B, x, (M, N), c_block_sizes=(lMy, 1))
    y[:] = alpha * tmp1 + beta * tmp2

def gesummv_shmem_init(N, datatype):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N,
                        shape=(N, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * j + 2) % N) / N,
                        shape=(N, N), dtype=datatype)
    x = np.fromfunction(lambda i: (i % N) / N, shape=(N,), dtype=datatype)
    y = np.empty((N,), dtype=datatype)
    return alpha, beta, A, B, x, y

def gesummv_distr_init(N, lM, lN, lMy, datatype, pi, pj):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((l2g(i, pi, lM) * l2g(j, pj, lN) + 1) % N) / N,
                        shape=(lM, lN), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((l2g(i, pi, lM) * l2g(j, pj, lN) + 2) % N) / N,
                        shape=(lM, lN), dtype=datatype)
    x = np.fromfunction(lambda i: (l2g(i, pj, lN) % N) / N, shape=(lN,), dtype=datatype)
    y = np.empty((lMy,), dtype=datatype)
    return alpha, beta, A, B, x, y

def gesummv(sizes, validate=True):

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px, Py = grid[size]
    pi = rank // Py
    pj = rank % Py

    if rank == 0:
        print("===== gesummv =====")
        print("sizes: {}".format(sizes), flush=True)

    M = N = sizes

    # Symbolic sizes
    lM = M // Px
    lN = N // Py
    lMy = M // Py

    alpha, beta, lA, lB, lx, ly = gesummv_distr_init(N, lM, lN, lMy, np.float64, pi, pj)

    mpi_sdfg = None
    if rank == 0:
        mpi_sdfg = gesummv_distr.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_func= mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        mpi_sdfg = dc.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=gesummv_distr.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=gesummv_distr.name),
            gesummv_distr.name))

    ldict = locals()

    comm.Barrier()

    mpi_func(alpha=alpha, beta=beta, A=lA, B=lB, x=lx, y=ly,
             lM=lM, lN=lN, lMy=lMy, Px=Px, Py=Py)
    
    comm.Barrier()

    if validate:

        if rank == 0:
            y = np.empty((N,), dtype=np.float64)
            y[0:lMy] = ly
            for i in range(Py):
                if i == pj:
                    continue
                else:
                    comm.Recv(ly, source=i, tag=i)
                    y[i*lMy:(i+1)*lMy] = ly
        elif pi == 0:
            comm.Send(ly, dest=0, tag=pj)
        
        comm.Barrier()

    stmt = ("mpi_func(alpha=alpha, beta=beta, A=lA, B=lB, x=lx, y=ly, "
            "lM=lM, lN=lN, lMy=lMy, Px=Px, Py=Py)")
    setup = "comm.Barrier()"
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

    if validate:

        if rank == 0:
            alpha, beta, refA, refB, refx, refy = gesummv_shmem_init(N, np.float64)
            shared_sdfg = gesummv_shmem.compile()
            shared_sdfg(alpha=alpha, beta=beta, A=refA, B=refB, x=refx, y=refy,
                        lM=lM, lN=lN, lMy=lMy, Px=Px, Py=Py)
            error = relerr(refy, y)
            print("validation: {} ({})".format(error < 1e-12, error), flush=True)


# ===== 2mm =====

k2mm_sizes = [[1600, 1800, 2200, 2400], [3200, 3600, 4400, 4800]]  #, [6400, 7200, 8800, 4800]]

@dc.program
def k2mm_shmem(alpha: dc.float64, beta: dc.float64, A: dc.float64[NI, NK],
               B: dc.float64[NK, NJ], C: dc.float64[NJ, NL],
               D: dc.float64[NI, NL]):
    D[:] = alpha * A @ B @ C + beta * D

@dc.program
def k2mm_distr(alpha: dc.float64, beta: dc.float64, A: dc.float64[lNI, lNKa],
               B: dc.float64[lNKb, lNJ], C: dc.float64[lNJx, lNL],
               D: dc.float64[lNI, lNL]):
    tmp1 = distr.MatMult(A, B, (lNI * Px, lNJ * Py, NK))
    tmp2 = distr.MatMult(tmp1, C, (NI, NL, NJ))
    D[:] = alpha * tmp2 + beta * D

def k2mm_shmem_init(NI, NJ, NK, NL, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI,
                        shape=(NI, NK), dtype=datatype)
    B = np.fromfunction(lambda i, j: (i * (j + 1) % NJ) / NJ,
                        shape=(NK, NJ), dtype=datatype)
    C = np.fromfunction(lambda i, j: ((i * (j + 3) + 1) % NL) / NL,
                        shape=(NJ, NL), dtype=datatype)
    D = np.fromfunction(lambda i, j: (i * (j + 2) % NK) / NK,
                        shape=(NI, NL), dtype=datatype)
    return alpha, beta, A, B, C, D

def k2mm_distr_init(NI, NJ, NK, NL, lNI, lNJ, lNJx, lNKa, lNKb, lNL, datatype, pi, pj):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((l2g(i, pi, lNI) * l2g(j, pj, lNKa) + 1) % NI) / NI,
                        shape=(lNI, lNKa), dtype=datatype)
    B = np.fromfunction(lambda i, j: (l2g(i, pi, lNKb) * (l2g(j, pj, lNJ) + 1) % NJ) / NJ,
                        shape=(lNKb, lNJ), dtype=datatype)
    C = np.fromfunction(lambda i, j: ((l2g(i, pi, lNJx) * (l2g(j, pj, lNL) + 3) + 1) % NL) / NL,
                        shape=(lNJx, lNL), dtype=datatype)
    D = np.fromfunction(lambda i, j: (l2g(i, pi, lNI) * (l2g(j, pj, lNL) + 2) % NK) / NK,
                        shape=(lNI, lNL), dtype=datatype)
    return alpha, beta, A, B, C, D

def k2mm(sizes, validate=True):

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px, Py = grid[size]
    # Tmp fix for gemm and non-square grids
    if Px < Py:
        Px, Py = Py, Px
    pi = rank // Py
    pj = rank % Py

    if rank == 0:
        print("===== k2mm =====")
        print("sizes: {}".format(sizes), flush=True)

    NI, NJ, NK, NL = sizes

    # Symbolic sizes
    lNI = NI // Px
    lNJ = NJ // Py
    lNJx = NJ // Px
    lNKa = NK // Py
    lNKb = NK // Px
    lNL = NL // Py

    alpha, beta, lA, lB, lC, lD = k2mm_distr_init(NI, NJ, NK, NL, lNI, lNJ, lNJx,
                                                  lNKa, lNKb, lNL, np.float64, pi, pj)

    mpi_sdfg = None
    if rank == 0:
        mpi_sdfg = k2mm_distr.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_func= mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        mpi_sdfg = dc.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=k2mm_distr.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=k2mm_distr.name),
            k2mm_distr.name))

    ldict = locals()

    comm.Barrier()

    mpi_func(alpha=alpha, beta=beta, A=lA, B=lB, C=lC, D=lD,
             lNI=lNI, lNJ=lNJ, lNJx=lNJx, lNKa=lNKa, lNKb=lNKb, lNL=lNL, Px=Px, Py=Py)
    
    comm.Barrier()

    if validate:

        if rank == 0:
            D = np.empty((Px, Py, lNI, lNL), dtype=np.float64)
        else:
            D = None
        comm.Gather(lD, D)
        if rank == 0:
            D = np.transpose(D, (0, 2, 1, 3)).reshape(NI, NL).copy()
        
        comm.Barrier()

    stmt = ("mpi_func(alpha=alpha, beta=beta, A=lA, B=lB, C=lC, D=lD, "
            "lNI=lNI, lNJ=lNJ, lNJx=lNJx, lNKa=lNKa, lNKb=lNKb, lNL=lNL, Px=Px, Py=Py)")
    setup = "comm.Barrier()"
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

    if validate:

        if rank == 0:
            alpha, beta, refA, refB, refC, refD = k2mm_shmem_init(NI, NJ, NK, NL, np.float64)
            shared_sdfg = k2mm_shmem.compile()
            shared_sdfg(alpha=alpha, beta=beta, A=refA, B=refB, C=refC, D=refD,
                        lNI=lNI, lNJ=lNJ, lNJx=lNJx, lNKa=lNKa, lNKb=lNKb, lNL=lNL, Px=Px, Py=Py)
            error = relerr(refD, D)
            print("validation: {} ({})".format(error < 1e-12, error), flush=True)


# ===== 3mm =====

k3mm_sizes = [[1600, 1800, 2000, 2200, 2400], [3200, 3600, 4000, 4400, 4800]]  #, [6400, 3600, 8000, 8800, 9600]]

@dc.program
def k3mm_shmem(A: dc.float64[NI, NK], B: dc.float64[NK, NJ],
               C: dc.float64[NJ, NM], D: dc.float64[NM, NL],
               E: dc.float64[NI, NL]):
    E[:] = A @ B @ C @ D

@dc.program
def k3mm_distr(A: dc.float64[lNI, lNKa], B: dc.float64[lNKb, lNJ],
               C: dc.float64[lNJx, lNMy], D: dc.float64[lNMx, lNL],
               E: dc.float64[lNI, lNL]):
    tmp1 = distr.MatMult(A, B, (lNI * Px, lNJ * Py, NK))
    tmp2 = distr.MatMult(tmp1, C, (NI, NM, NJ))
    E[:] = distr.MatMult(tmp2, D, (NI, NL, NM))

def k3mm_shmem_init(NI, NJ, NK, NM, NL, datatype):

    A = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / (5 * NI),
                        shape=(NI, NK), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * (j + 1) + 2) % NJ) / (5 * NJ),
                        shape=(NK, NJ), dtype=datatype)
    C = np.fromfunction(lambda i, j: (i * (j + 3) % NL) / (5 * NL),
                        shape=(NJ, NM), dtype=datatype)
    D = np.fromfunction(lambda i, j: ((i * (j + 2) + 2) % NK) / ( 5 * NK),
                        shape=(NM, NL), dtype=datatype)
    E = np.empty((NI, NL), dtype=datatype)
    return A, B, C, D, E

def k3mm_distr_init(NI, NJ, NK, NM, NL, lNI, lNJ, lNJx, lNKa, lNKb, lNMx, lNMy, lNL, datatype, pi, pj):

    A = np.fromfunction(lambda i, j: ((l2g(i, pi, lNI) * l2g(j, pj, lNKa) + 1) % NI) / (5 * NI),
                        shape=(lNI, lNKa), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((l2g(i, pi, lNKb) * (l2g(j, pj, lNJ) + 1) + 2) % NJ) / (5 * NJ),
                        shape=(lNKa, lNJ), dtype=datatype)
    C = np.fromfunction(lambda i, j: (l2g(i, pi, lNJx) * (l2g(j, pj, lNMy) + 3) % NL) / (5 * NL),
                        shape=(lNJx, lNMy), dtype=datatype)
    D = np.fromfunction(lambda i, j: ((l2g(i, pi, lNMx) * (l2g(j, pj, lNL) + 2) + 2) % NK) / ( 5 * NK),
                        shape=(lNMx, lNL), dtype=datatype)
    E = np.empty((lNI, lNL), dtype=datatype)
    return A, B, C, D, E

def k3mm(sizes, validate=True):

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px, Py = grid[size]
    # Tmp fix for gemm and non-square grids
    if Px < Py:
        Px, Py = Py, Px
    pi = rank // Py
    pj = rank % Py

    if rank == 0:
        print("===== k3mm =====")
        print("sizes: {}".format(sizes), flush=True)

    NI, NJ, NK, NL, NM = sizes

    # Symbolic sizes
    lNI = NI // Px
    lNJ = NJ // Py
    lNJx = NJ // Px
    lNKa = NK // Py
    lNKb = NK // Px
    lNL = NL // Py
    lNMx = NM // Px
    lNMy = NM // Py

    lA, lB, lC, lD, lE = k3mm_distr_init(NI, NJ, NK, NM, NL, lNI, lNJ, lNJx,
                                         lNKa, lNKb, lNMx, lNMy, lNL, np.float64, pi, pj)

    mpi_sdfg = None
    if rank == 0:
        mpi_sdfg = k3mm_distr.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_func= mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        mpi_sdfg = dc.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=k3mm_distr.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=k3mm_distr.name),
            k3mm_distr.name))

    ldict = locals()

    comm.Barrier()

    mpi_func(A=lA, B=lB, C=lC, D=lD, E=lE,
             lNI=lNI, lNJ=lNJ, lNJx=lNJx, lNKa=lNKa, lNKb=lNKb,
             lNMx=lNMx, lNMy=lNMy, lNL=lNL, Px=Px, Py=Py)
    
    comm.Barrier()

    if validate:

        if rank == 0:
            E = np.empty((Px, Py, lNI, lNL), dtype=np.float64)
        else:
            E = None
        comm.Gather(lE, E)
        if rank == 0:
            E = np.transpose(E, (0, 2, 1, 3)).reshape(NI, NL).copy()
        
        comm.Barrier()

    stmt = ("mpi_func(A=lA, B=lB, C=lC, D=lD, E=lE, "
            "lNI=lNI, lNJ=lNJ, lNJx=lNJx, lNKa=lNKa, lNKb=lNKb, "
            "lNMx=lNMx, lNMy=lNMy, lNL=lNL, Px=Px, Py=Py)")
    setup = "comm.Barrier()"
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

    if validate:

        if rank == 0:
            refA, refB, refC, refD, refE = k3mm_shmem_init(NI, NJ, NK, NM, NL, np.float64)
            shared_sdfg = k3mm_shmem.compile()
            shared_sdfg(A=refA, B=refB, C=refC, D=refD, E=refE,
                        lNI=lNI, lNJ=lNJ, lNJx=lNJx, lNKa=lNKa, lNKb=lNKb,
                        lNMx=lNMx, lNMy=lNMy, lNL=lNL, Px=Px, Py=Py)
            error = relerr(refE, E)
            print("validation: {} ({})".format(error < 1e-12, error), flush=True)


if __name__ == "__main__":

    # for sizes in atax_sizes:
    #     atax(sizes)
    # for sizes in bicg_sizes:
    #     bicg(sizes)
    # for sizes in doitgen_sizes:
    #     doitgen(sizes)
    # for sizes in gemm_sizes:
    #     gemm(sizes)
    # for sizes in gemver_sizes:
    #     gemver(sizes)
    # for sizes in gesummv_sizes:
    #     gesummv(sizes)
    # for sizes in k2mm_sizes:
    #     k2mm(sizes)
    for sizes in k3mm_sizes:
        k3mm(sizes)
