# ===== Imports =====

import argparse
import csv
import legate.numpy as np
import timeit

# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
rank = 0
setup = ""


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

def adjust_size(size, scal_func, comm_size, divisor):
    candidate = size * scal_func(comm_size)
    if candidate // divisor != candidate:
        candidate = np.ceil(candidate / divisor) * divisor
    return int(candidate)

# CSV headers
file_name = "legate_cpu_x_sockets.csv"
field_names = ["benchmark", "framework", "sockets", "sizes", "time"]

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

def write_time(bench, sockets, sz, time_list, append=True):
    entries = []
    for t in time_list:
        entries.append(
            dict(benchmark=bench, framework="legate_cpu", sockets=sockets, sizes=sz, time=t))
    write_csv(file_name, field_names, entries, append=append)


# ===== Programs ==============================================================

# ===== atax =====

atax_sizes = [[10000, 12500]] #[[20000, 25000]]  #[[1800, 2200], [3600, 4400], [7200, 8800], [14400, 17600]]

def atax_legate(A, x, y):
    y[:] = (A @ x) @ A

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

def atax(sockets, sizes, validate=True):

    # rank = 0
    size = sockets
    Px, Py = grid[size]
    # Fix for grid issue with gemv
    if Px != Py:
        Px, Py = 1, size
    # pi = rank // Py
    # pj = rank % Py

    if rank == 0:
        print("===== atax =====")
        print("sizes: {}".format(sizes), flush=True)

    M, N = sizes
    M = adjust_size(M, lambda x: np.sqrt(x), size, max(Px, Py))
    N = adjust_size(N, lambda x: np.sqrt(x), size, max(Px, Py))
    if rank == 0:
        print("adjusted sizes: {}".format((M, N)), flush=True)

    A, x, y = atax_shmem_init(M, N, np.float64)
    if rank == 0:
        print("data initialized", flush=True)

    # ldict = locals()
    # ldict["np"] = np
    # ldict["atax_legate"] = atax_legate
    # ldict["atax_shmem_init"] = atax_shmem_init

    atax_legate(A, x, y)
    assert not np.isnan(np.sum(y))
    ldict = {**globals(), **locals()}

    stmt = "atax_legate(A, x, y); assert not np.isnan(np.sum(y))"
    # setup = "A, x, y = atax_shmem_init(M, N, np.float64)"
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
        write_time("atax", sockets, (M, N), raw_time_list, append=False)


# ===== bicg =====

bicg_sizes = [[12500, 10000]] #[[25000, 20000]]  # [[2200, 1800], [4400, 3600], [7200, 8800]]

def bicg_legate(A, p, r, o1, o2):
    o1[:] = r @ A
    o2[:] = A @ p

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

def bicg(sockets, sizes, validate=True):

    # rank = 0
    size = sockets
    Px, Py = grid[size]
    # Fix for grid issue with gemv
    if Px != Py:
        Px, Py = 1, size

    if rank == 0:
        print("===== bicg =====")
        print("sizes: {}".format(sizes), flush=True)

    M, N = sizes
    M = adjust_size(M, lambda x: np.sqrt(x), size, max(Px, Py))
    N = adjust_size(N, lambda x: np.sqrt(x), size, max(Px, Py))
    if rank == 0:
        print("adjusted sizes: {}".format((M, N)), flush=True)

    A, p, r, o1, o2 = bicg_shmem_init(M, N, np.float64)
    if rank == 0:
        print("data initialized", flush=True)

    # ldict = locals()
    # ldict["np"] = np
    # ldict["bicg_legate"] = bicg_legate

    bicg_legate(A, p, r, o1, o2)
    assert not np.isnan(np.sum(o1))
    assert not np.isnan(np.sum(o2))
    ldict = {**globals(), **locals()}

    stmt = ("bicg_legate(A, p, r, o1, o2); "
            "assert not np.isnan(np.sum(o1)); "
            "assert not np.isnan(np.sum(o2));")
    # setup = "A, p, r, o1, o2 = bicg_shmem_init(M, N, np.float64)"
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
        write_time("bicg", sockets, (M, N), raw_time_list)


# ===== doitgen =====

doitgen_sizes = [[16, 16, 512]]#[[128, 512, 512]]  #[[256, 250, 270], [512, 500, 540]]

def doitgen_legate(NR, NQ, NP, A, C4):
    for r in range(NR):
        for q in range(NQ):
            A[r, q, :] = A[r, q] @ C4

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

def doitgen(sockets, sizes, validate=True):

    # rank = 0
    size = sockets

    if rank == 0:
        print("===== doitgen =====")
        print("sizes: {}".format(sizes), flush=True)

    NR, NQ, NP = sizes
    NR = adjust_size(NR, lambda x: x, size, size)
    if rank == 0:
        print("adjusted sizes: {}".format((NR, NQ, NP)), flush=True)


    A, C4 = doitgen_shmem_init(NR, NQ, NP, np.float64)
    if rank == 0:
        print("data initialized", flush=True)


    # ldict = locals()
    # ldict["np"] = np
    # ldict["doitgen_legate"] = doitgen_legate

    doitgen_legate(NR, NQ, NP, A, C4)
    assert not np.isnan(np.sum(A))
    ldict = {**globals(), **locals()}

    stmt = "doitgen_legate(NR, NQ, NP, A, C4); assert not np.isnan(np.sum(A))"
    # setup = "A, C4 = doitgen_shmem_init(NR, NQ, NP, np.float64)"
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
        write_time("doitgen", sockets, (NR, NQ, NP), raw_time_list)


# ===== gemm =====

gemm_sizes = [[4000, 4600, 5200]] #[[8000, 9200, 5200]]  # [[2000, 2300, 2600], [4000, 4600, 5200]]  #, [8000, 9200, 5200]]

def gemm_legate(alpha, beta, C, A, B):
    C[:] = alpha * A @ B + beta * C 

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

def gemm(sockets, sizes, validate=True):

    # rank = 0
    size = sockets
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
    NI = adjust_size(NI, lambda x: np.cbrt(x), size, max(Px, Py))
    NJ = adjust_size(NJ, lambda x: np.cbrt(x), size, max(Px, Py))
    NK = adjust_size(NK, lambda x: np.cbrt(x), size, max(Px, Py))
    if rank == 0:
        print("adjusted sizes: {}".format((NI, NJ, NK)), flush=True)


    alpha, beta, C, A, B = gemm_shmem_init(NI, NJ, NK, np.float64)
    if rank == 0:
        print("data initialized", flush=True)

    # ldict = locals()
    # ldict["np"] = np
    # ldict["gemm_legate"] = gemm_legate

    gemm_legate(alpha, beta, C, A, B)
    assert not np.isnan(np.sum(C))
    ldict = {**globals(), **locals()}

    stmt = "gemm_legate(alpha, beta, C, A, B); assert not np.isnan(np.sum(C))"
    # setup = "alpha, beta, C, A, B = gemm_shmem_init(NI, NJ, NK, np.float64)"
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
        write_time("gemm", sockets, (NI, NJ, NK), raw_time_list)


# # ==== gemver ====

gemver_sizes = [10000]  #[4000, 8000]

def gemver_legate(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x

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

def gemver(sockets, sizes, validate=True):

    # rank = 0
    size = sockets
    Px, Py = grid[size]
    # Fix for grid issue with gemv
    if Px != Py:
        Px, Py = 1, size

    if rank == 0:
        print("===== gemver =====")
        print("sizes: {}".format(sizes), flush=True)

    M = N = sizes
    M = N = adjust_size(N, lambda x: np.sqrt(x), size, max(Px, Py))
    if rank == 0:
        print("adjusted sizes: {}".format((N,)), flush=True)

    alpha, beta, A, u1, u2, v1, v2, w, x, y, z = gemver_shmem_init(N, np.float64)
    if rank == 0:
        print("data initialized", flush=True)

    # ldict = locals()
    # ldict["np"] = np
    # ldict["gemver_legate"] = gemver_legate

    gemver_legate(alpha, beta, A, u1, v1, u2, v2, w, x, y, z)
    assert not np.isnan(np.sum(w))
    ldict = {**globals(), **locals()}

    stmt = ("gemver_legate(alpha, beta, A, u1, v1, u2, v2, w, x, y, z); "
            "assert not np.isnan(np.sum(w))")
    # setup = "alpha, beta, A, u1, u2, v1, v2, w, x, y, z = gemver_shmem_init(N, np.float64)"
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
        write_time("gemver", sockets, (N,), raw_time_list)


# ===== gesummv =====

gesummv_sizes = [11200] #[22400] #[2800, 5600, 11200]

def gesummv_legate(alpha, beta, A, B, x, y):
    y[:] = alpha * A @ x + beta * B @ x

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

def gesummv(sockets, sizes, validate=True):

    # rank = 0
    size = sockets
    Px, Py = grid[size]
    # Fix for grid issue with gemv
    if Px != Py:
        Px, Py = 1, size

    if rank == 0:
        print("===== gesummv =====")
        print("sizes: {}".format(sizes), flush=True)

    M = N = sizes
    M = N = adjust_size(N, lambda x: np.sqrt(x), size, max(Px, Py))
    if rank == 0:
        print("adjusted sizes: {}".format((N,)), flush=True)

    alpha, beta, A, B, x, y = gesummv_shmem_init(N, np.float64)
    if rank == 0:
        print("data initialized", flush=True)

    # ldict = locals()
    # ldict["np"] = np
    # ldict["gesummv_legate"] = gesummv_legate

    gesummv_legate(alpha, beta, A, B, x, y)
    assert not np.isnan(np.sum(y))
    ldict = {**globals(), **locals()}

    stmt = ("gesummv_legate(alpha, beta, A, B, x, y); "
            "assert not np.isnan(np.sum(y))")
    # setup = "alpha, beta, A, B, x, y = gesummv_shmem_init(N, np.float64)"
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
        write_time("gesummv", sockets, (N,), raw_time_list)


# ===== 2mm =====

k2mm_sizes = [[3200, 3600, 4400, 4800]] #[[6400, 7200, 4400, 4800]]  # [[1600, 1800, 2200, 2400], [3200, 3600, 4400, 4800]]  #, [6400, 7200, 8800, 4800]]

def k2mm_legate(alpha, beta, A, B, C, D):
    D[:] = alpha * A @ B @ C + beta * D

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

def k2mm(sockets, sizes, validate=True):

    # rank = 0
    size = sockets
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
    NI = adjust_size(NI, lambda x: np.cbrt(x), size, max(Px, Py))
    NJ = adjust_size(NJ, lambda x: np.cbrt(x), size, max(Px, Py))
    NK = adjust_size(NK, lambda x: np.cbrt(x), size, max(Px, Py))
    NL = adjust_size(NL, lambda x: np.cbrt(x), size, max(Px, Py))
    if rank == 0:
        print("adjusted sizes: {}".format((NI, NJ, NK, NL)), flush=True)

    alpha, beta, A, B, C, D = k2mm_shmem_init(NI, NJ, NK, NL, np.float64)
    if rank == 0:
        print("data initialized", flush=True)

    k2mm_legate(alpha, beta, A, B, C, D)
    assert not np.isnan(np.sum(D))
    ldict = {**globals(), **locals()}

    stmt = ("k2mm_legate(alpha, beta, A, B, C, D); "
            "assert not np.isnan(np.sum(D))")
    # setup = "alpha, beta, A, B, C, D = k2mm_shmem_init(NI, NJ, NK, NL, np.float64)"
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
        write_time("k2mm", sockets, (NI, NJ, NK, NL), raw_time_list)


# ===== 3mm =====

k3mm_sizes = [[3200, 3600, 4000, 4400, 4800]] #[[6400, 7200, 4000, 4400, 4800]]  # [[1600, 1800, 2000, 2200, 2400], [3200, 3600, 4000, 4400, 4800]]  #, [6400, 3600, 8000, 8800, 9600]]

def k3mm_legate(A, B, C, D, E):
    E[:] = A @ B @ C @ D

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

def k3mm(sockets, sizes, validate=True):

    # rank = 0
    size = sockets
    Px, Py = grid[size]
    # Tmp fix for gemm and non-square grids
    if Px < Py:
        Px, Py = Py, Px


    if rank == 0:
        print("===== k3mm =====")
        print("sizes: {}".format(sizes), flush=True)

    NI, NJ, NK, NL, NM = sizes
    NI = adjust_size(NI, lambda x: np.cbrt(x), size, max(Px, Py))
    NJ = adjust_size(NJ, lambda x: np.cbrt(x), size, max(Px, Py))
    NK = adjust_size(NK, lambda x: np.cbrt(x), size, max(Px, Py))
    NL = adjust_size(NL, lambda x: np.cbrt(x), size, max(Px, Py))
    NM = adjust_size(NM, lambda x: np.cbrt(x), size, max(Px, Py))
    if rank == 0:
        print("adjusted sizes: {}".format((NI, NJ, NK, NL, NM)), flush=True)

    A, B, C, D, E = k3mm_shmem_init(NI, NJ, NK, NM, NL, np.float64)
    if rank == 0:
        print("data initialized", flush=True)

    k3mm_legate(A, B, C, D, E)
    assert not np.isnan(np.sum(E))
    ldict = {**globals(), **locals()}

    stmt = ("k3mm_legate(A, B, C, D, E); "
            "assert not np.isnan(np.sum(E));")
    # setup = "A, B, C, D, E = k3mm_shmem_init(NI, NJ, NK, NM, NL, np.float64)"
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
        write_time("k3mm", sockets, (NI, NJ, NK, NM, NL), raw_time_list)


# ===== mvt =====

mvt_sizes = [11000] #[22000]  # [4000, 8000, 16000]

def mvt_legate(x1, x2, y_1, y_2, A):
    x1 += A @ y_1
    x2 += y_2 @ A

def mvt_shmem_init(N, datatype):
    x1 = np.fromfunction(lambda i: (i % N) / N, shape=(N,), dtype=datatype)
    x2 = np.fromfunction(lambda i: ((i + 1) % N) / N, shape=(N,), dtype=datatype)
    y_1 = np.fromfunction(lambda i: ((i + 3) % N) / N, shape=(N,), dtype=datatype)
    y_2 = np.fromfunction(lambda i: ((i + 4) % N) / N, shape=(N,), dtype=datatype)
    A = np.fromfunction(lambda i, j: (i * j % N) / N, shape=(N,N), dtype=datatype)
    return x1, x2, y_1, y_2, A

def mvt_distr_init(N, lM, lN, lMy, datatype, pi, pj):
    x1 = np.fromfunction(lambda i: (l2g(i, pj, lMy) % N) / N, shape=(lMy,), dtype=datatype)
    x2 = np.fromfunction(lambda i: ((l2g(i, pj, lN) + 1) % N) / N, shape=(lN,), dtype=datatype)
    y_1 = np.fromfunction(lambda i: ((l2g(i, pj, lN) + 3) % N) / N, shape=(lN,), dtype=datatype)
    y_2 = np.fromfunction(lambda i: ((l2g(i, pj, lMy) + 4) % N) / N, shape=(lMy,), dtype=datatype)
    A = np.fromfunction(lambda i, j: (l2g(i, pi, lM) * l2g(j, pj, lN) % N) / N, shape=(lM,lN), dtype=datatype)
    return x1, x2, y_1, y_2, A

def mvt(sockets, sizes, validate=True):

    # rank = 0
    size = sockets
    Px, Py = grid[size]
    # Fix for grid issue with gemv
    if Px != Py:
        Px, Py = 1, size

    if rank == 0:
        print("===== mvt =====")
        print("sizes: {}".format(sizes), flush=True)

    M = N = sizes
    M = N = adjust_size(N, lambda x: np.sqrt(x), size, max(Px, Py))
    if rank == 0:
        print("adjusted sizes: {}".format((N,)), flush=True)

    x1, x2, y_1, y_2, A = mvt_shmem_init(N, np.float64)
    if rank == 0:
        print("data initialized", flush=True)

    mvt_legate(x1, x2, y_1, y_2, A)
    assert not np.isnan(np.sum(A))
    ldict = {**globals(), **locals()}

    stmt = ("mvt_legate(x1, x2, y_1, y_2, A); "
            "assert not np.isnan(np.sum(A))")
    # setup = "x1, x2, y_1, y_2, A = mvt_shmem_init(N, np.float64)"
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
        write_time("mvt", sockets, (N,), raw_time_list)


# ===== jacobi_1d =====

jacobi_1d_sizes = [[1000, 12000]] #[[1000, 24000]] #[[1000, 4000], [2000, 8000], [4000, 16000]]

def jacobi_1d_legate(TSTEPS, A, B):   
    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])

def jacobi_1d_shmem_init(N, datatype):
    A = np.fromfunction(lambda i: (i + 2) / N, shape=(N,), dtype=datatype)
    B = np.fromfunction(lambda i: (i + 3) / N, shape=(N,), dtype=datatype)
    return A, B

def jacobi_1d_distr_init(N, lN, datatype, p):
    A = np.zeros((lN+2,), dtype=datatype)
    B = np.zeros((lN+2,), dtype=datatype)
    A[1:-1] = np.fromfunction(lambda i: (l2g(i, p, lN) + 2) / N,
                              shape=(lN,), dtype=datatype)
    B[1:-1] = np.fromfunction(lambda i: (l2g(i, p, lN) + 3) / N,
                              shape=(lN,), dtype=datatype)
    return A, B

def jacobi_1d(sockets, sizes, validate=True):

    # rank = 0
    size = sockets

    if rank == 0:
        print("===== jacobi_1d =====")
        print("sizes: {}".format(sizes), flush=True)

    TSTEPS, NR = sizes
    NR = adjust_size(NR, lambda x: x, size, size)
    if rank == 0:
        print("adjusted sizes: {}".format((TSTEPS, NR)), flush=True)

    A, B = jacobi_1d_shmem_init(NR, np.float64)
    if rank == 0:
        print("data initialized", flush=True)
    
    jacobi_1d_legate(TSTEPS, A, B)
    assert not np.isnan(np.sum(A))
    ldict = {**globals(), **locals()}

    stmt = ("jacobi_1d_legate(TSTEPS, A, B); "
            "assert not np.isnan(np.sum(A))")
    # setup = "A, B = jacobi_1d_shmem_init(NR, np.float64)"
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
        write_time("jacobi_1d", sockets, (TSTEPS, NR), raw_time_list)


# ===== jacobi_2d =====

jacobi_2d_sizes = [[1000, 500]]  #[[1000, 500]]  # [[10, 2800], [10, 5600], [10, 11200]]

def jacobi_2d_legate(TSTEPS, A, B):
    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] +
                                 A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] +
                                 B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])

def jacobi_2d_shmem_init(N, datatype):
    A = np.fromfunction(lambda i, j: i * (j + 2) / N, shape=(N, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: i * (j + 3) / N, shape=(N, N), dtype=datatype)
    return A, B

def jacobi_2d_distr_init(N, lM, lN, datatype, pi, pj):
    A = np.zeros((lM+2, lN+2), dtype=datatype)
    B = np.zeros((lM+2, lN+2), dtype=datatype)
    A[1:-1, 1:-1] = np.fromfunction(lambda i, j: l2g(i, pi, lM) * (l2g(j, pj, lN) + 2) / N,
                                    shape=(lM, lN), dtype=datatype)
    B[1:-1, 1:-1] = np.fromfunction(lambda i, j: l2g(i, pi, lM) * (l2g(j, pj, lN) + 3) / N,
                                    shape=(lM, lN), dtype=datatype)
    return A, B

def jacobi_2d(sockets, sizes, validate=True):

    # rank = 0
    size = sockets
    Px, Py = grid[size]

    if rank == 0:
        print("===== jacobi_2d =====")
        print("sizes: {}".format(sizes), flush=True)

    TSTEPS, N = sizes
    N = adjust_size(N, lambda x: np.sqrt(x), size, max(Px, Py))
    if rank == 0:
        print("adjusted sizes: {}".format((TSTEPS, N)), flush=True)
    M = N

    A, B = jacobi_2d_shmem_init(N, np.float64)
    if rank == 0:
        print("data initialized", flush=True)

    jacobi_2d_legate(TSTEPS, A, B)
    assert not np.isnan(np.sum(A))
    ldict = {**globals(), **locals()}

    stmt = ("jacobi_2d_legate(TSTEPS, A, B); "
            "assert not np.isnan(np.sum(A))")
    # setup = "A, B = jacobi_2d_shmem_init(N, np.float64)"
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
        write_time("jacobi_2d", sockets, (TSTEPS, N), raw_time_list)


# # ===== heat_3d =====

# heat_3d_sizes = [[10, 200], [10, 300], [10, 450]]

# S = dc.symbol('S', dtype=dc.int32, integer=True, positive=True)
# @dc.program
# def heat_3d_shmem(TSTEPS: dc.int32, A: dc.float64[S, S, S], B: dc.float64[S, S, S]):
#     for t in range(1, TSTEPS):
#         B[1:-1, 1:-1, 1:-1] = (
#             0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] +
#                      A[:-2, 1:-1, 1:-1]) +
#             0.125 * (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] +
#                      A[1:-1, :-2, 1:-1]) +
#             0.125 * (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] +
#                      A[1:-1, 1:-1, 0:-2]) +
#             A[1:-1, 1:-1, 1:-1])
#         A[1:-1, 1:-1, 1:-1] = (
#             0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] +
#                      B[:-2, 1:-1, 1:-1]) +
#             0.125 * (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] +
#                      B[1:-1, :-2, 1:-1]) +
#             0.125 * (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] +
#                      B[1:-1, 1:-1, 0:-2]) +
#             B[1:-1, 1:-1, 1:-1])

# @dc.program
# def heat_3d_distr(TSTEPS: dc.int32, A: dc.float64[lM+2, lN+2, N], B: dc.float64[lM+2, lN+2, N]):   
#     req = np.empty((8,), dtype=MPI_Request)
#     for t in range(1, TSTEPS):
#         dc.comm.Isend(A[1, 1:-1], nn, 0, req[0])
#         dc.comm.Isend(A[-2, 1:-1], ns, 1, req[1])
#         dc.comm.Isend(A[1:-1, 1], nw, 2, req[2])
#         dc.comm.Isend(A[1:-1, -2], ne, 3, req[3])
#         dc.comm.Irecv(A[0, 1:-1], nn, 1, req[4])
#         dc.comm.Irecv(A[-1, 1:-1], ns, 0, req[5])
#         dc.comm.Irecv(A[1:-1, 0], nw, 3, req[6])
#         dc.comm.Irecv(A[1:-1, -1], ne, 2, req[7])
#         dc.comm.Waitall(req)

#         B[1+noff:-1-soff, 1+woff:-1-eoff, 1:-1] = (
#             0.125 * (A[2+noff:-soff, 1+woff:-1-eoff, 1:-1] - 2.0 *
#                      A[1+noff:-1-soff, 1+woff:-1-eoff, 1:-1] +
#                      A[noff:-2-soff, 1+woff:-1-eoff, 1:-1]) +
#             0.125 * (A[1+noff:-1-soff, 2+woff:-eoff, 1:-1] - 2.0 *
#                      A[1+noff:-1-soff, 1+woff:-1-eoff, 1:-1] +
#                      A[1+noff:-1-soff, woff:-2-eoff, 1:-1]) +
#             0.125 * (A[1+noff:-1-soff, 1+woff:-1-eoff, 2:] - 2.0 *
#                      A[1+noff:-1-soff, 1+woff:-1-eoff, 1:-1] +
#                      A[1+noff:-1-soff, 1+woff:-1-eoff, 0:-2]) +
#                      A[1+noff:-1-soff, 1+woff:-1-eoff, 1:-1])

#         dc.comm.Isend(B[1, 1:-1], nn, 0, req[0])
#         dc.comm.Isend(B[-2, 1:-1], ns, 1, req[1])
#         dc.comm.Isend(B[1:-1, 1], nw, 2, req[2])
#         dc.comm.Isend(B[1:-1, -2], ne, 3, req[3])
#         dc.comm.Irecv(B[0, 1:-1], nn, 1, req[4])
#         dc.comm.Irecv(B[-1, 1:-1], ns, 0, req[5])
#         dc.comm.Irecv(B[1:-1, 0], nw, 3, req[6])
#         dc.comm.Irecv(B[1:-1, -1], ne, 2, req[7])
#         dc.comm.Waitall(req)

#         A[1+noff:-1-soff, 1+woff:-1-eoff, 1:-1] = (
#             0.125 * (B[2+noff:-soff, 1+woff:-1-eoff, 1:-1] - 2.0 *
#                      B[1+noff:-1-soff, 1+woff:-1-eoff, 1:-1] +
#                      B[noff:-2-soff, 1+woff:-1-eoff, 1:-1]) +
#             0.125 * (B[1+noff:-1-soff, 2+woff:-eoff, 1:-1] - 2.0 *
#                      B[1+noff:-1-soff, 1+woff:-1-eoff, 1:-1] +
#                      B[1+noff:-1-soff, woff:-2-eoff, 1:-1]) +
#             0.125 * (B[1+noff:-1-soff, 1+woff:-1-eoff, 2:] - 2.0 *
#                      B[1+noff:-1-soff, 1+woff:-1-eoff, 1:-1] +
#                      B[1+noff:-1-soff, 1+woff:-1-eoff, 0:-2]) +
#                      B[1+noff:-1-soff, 1+woff:-1-eoff, 1:-1])

# def heat_3d_shmem_init(N, datatype):
#     # A = np.fromfunction(lambda i, j, k: (i + j + (N - k)) * 10 / N, shape=(N, N, N), dtype=datatype)
#     # B = np.fromfunction(lambda i, j, k: (i + j + (N - k)) * 10 / N, shape=(N, N, N), dtype=datatype)
#     A = np.empty((N, N, N), dtype=datatype)
#     B = np.empty((N, N, N), dtype=datatype)
#     for i in range(N):
#         for j in range(N):
#             for k in range(N):
#                 A[i, j, k] = B[i, j, k] = (i + j + (N - k)) * 10 / N
#     return A, B

# def heat_3d_distr_init(N, lM, lN, datatype, pi, pj):
#     A = np.zeros((lM+2, lN+2, N), dtype=datatype)
#     B = np.zeros((lM+2, lN+2, N), dtype=datatype)
#     A[1:-1, 1:-1] = np.fromfunction(lambda i, j, k: (l2g(i, pi, lM) + l2g(j, pj, lN) + (N - k)) * 10 / N,
#                                     shape=(lM, lN, N), dtype=datatype)
#     B[1:-1, 1:-1] = np.fromfunction(lambda i, j, k: (l2g(i, pi, lM) + l2g(j, pj, lN) + (N - k)) * 10 / N,
#                                     shape=(lM, lN, N), dtype=datatype)
#     return A, B

# def heat_3d(sizes, validate=True):

#     # MPI
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     Px, Py = grid[size]
#     pi = rank // Py
#     pj = rank % Py
#     noff = soff = woff = eoff = 0
#     nn = (pi-1)*Py + pj
#     ns = (pi+1)*Py + pj
#     nw = pi*Py + (pj-1)
#     ne = pi*Py + (pj+1)
#     if pi == 0:
#         noff = 1
#         nn = MPI.PROC_NULL
#     if pi == Px - 1:
#         soff = 1
#         ns = MPI.PROC_NULL
#     if pj == 0:
#         woff = 1
#         nw = MPI.PROC_NULL
#     if pj == Py - 1:
#         eoff = 1
#         ne = MPI.PROC_NULL

#     if rank == 0:
#         print("===== heat_3d =====")
#         print("sizes: {}".format(sizes), flush=True)

#     TSTEPS, N = sizes
#     N = adjust_size(N, lambda x: np.cbrt(x), size, max(Px, Py))
#     if rank == 0:
#         print("adjusted sizes: {}".format((TSTEPS, N)), flush=True)
#     S = M = N

#     # Symbolic sizes
#     lM = M // Px
#     lN = N // Py

#     lA, lB = heat_3d_distr_init(N, lM, lN, np.float64, pi, pj)
#     if rank == 0:
#         print("data initialized", flush=True)

#     mpi_sdfg = heat_3d_distr.to_sdfg(strict=False)
#     if rank == 0:
#         mpi_sdfg.apply_strict_transformations()
#         mpi_sdfg.apply_transformations_repeated([MapFusion])
#         mpi_sdfg.apply_strict_transformations()
#         mpi_func= mpi_sdfg.compile()
#     comm.Barrier()
#     if rank > 0:
#         mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
#             ".dacecache/{n}/build/lib{n}.so".format(n=heat_3d_distr.name),
#             heat_3d_distr.name))

#     ldict = locals()

#     comm.Barrier()

#     mpi_func(A=lA, B=lB, TSTEPS=TSTEPS, lM=lM, lN=lN, Px=Px, Py=Py,
#              noff=noff, soff=soff, woff=woff, eoff=eoff,
#              nn=nn, ns=ns, nw=nw, ne=ne)
    
#     comm.Barrier()

#     if validate:

#         tA = lA[1:-1, 1:-1].copy()
#         tB = lB[1:-1, 1:-1].copy()
#         A = B = None
#         if rank == 0:
#             A = np.empty((Px, Py, lM, lN, N), dtype=np.float64)
#             B = np.empty((Px, Py, lM, lN, N), dtype=np.float64)
#         comm.Gather(tA, A)
#         comm.Gather(tB, B)
#         if rank == 0:
#             A = np.transpose(A, (0, 2, 1, 3, 4)).reshape(N, N, N).copy()
#             B = np.transpose(B, (0, 2, 1, 3, 4)).reshape(N, N, N).copy()

#     stmt = ("mpi_func(A=lA, B=lB, TSTEPS=TSTEPS, lM=lM, lN=lN, Px=Px, Py=Py, "
#             "noff=noff, soff=soff, woff=woff, eoff=eoff, "
#             "nn=nn, ns=ns, nw=nw, ne=ne)")
#     setup = "comm.Barrier()"
#     repeat = 10

#     raw_time_list = timeit.repeat(stmt,
#                                   setup=setup,
#                                   repeat=repeat,
#                                   number=1,
#                                   globals=ldict)
#     raw_time = np.median(raw_time_list)

#     if rank == 0:
#         ms_time = time_to_ms(raw_time)
#         print("Median is {}ms".format(ms_time), flush=True)
#         write_time("heat_3d", (TSTEPS, N), raw_time_list)

#     if validate:

#         if rank == 0:
#             # refA, refB = heat_3d_shmem_init(S, np.float64)
#             # shared_sdfg = heat_3d_shmem.to_sdfg()
#             # shared_sdfg.apply_strict_transformations()
#             # shared_sdfg.apply_transformations_repeated([MapFusion])
#             # shared_sdfg.apply_strict_transformations()
#             # shared_func= shared_sdfg.compile()
#             # shared_func(A=refA, B=refB, TSTEPS=TSTEPS, S=S)
#             refA, refB = heat_3d_distr_init(N, N, N, np.float64, 0, 0)
#             mpi_func(A=refA, B=refB, TSTEPS=TSTEPS, lM=N, lN=N, Px=1, Py=1,
#              noff=1, soff=1, woff=1, eoff=1,
#              nn=MPI.PROC_NULL, ns=MPI.PROC_NULL, nw=MPI.PROC_NULL, ne=MPI.PROC_NULL)
#             error = relerr(refA[1:-1, 1:-1], A)
#             print("validation: {} ({})".format(error < 1e-12, error), flush=True)
#             error = relerr(refB[1:-1, 1:-1], B)
#             print("validation: {} ({})".format(error < 1e-12, error), flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sockets", type=int, nargs="?", default=1)
    args = vars(parser.parse_args())
    sockets = args["sockets"]
    file_name = "legate_cpu_{}_sockets.csv".format(sockets)

    # for sizes in atax_sizes:
    #     atax(sockets, sizes)
    # for sizes in bicg_sizes:
    #     bicg(sockets, sizes)
    # for sizes in doitgen_sizes:
    #     doitgen(sockets, sizes)
    # for sizes in gemm_sizes:
    #     gemm(sockets, sizes)
    # for sizes in gemver_sizes:
    #     gemver(sockets, sizes)
    # for sizes in gesummv_sizes:
    #     gesummv(sockets, sizes)
    # for sizes in k2mm_sizes:
    #     k2mm(sockets, sizes)
    # for sizes in k3mm_sizes:
    #     k3mm(sockets, sizes)
    # for sizes in mvt_sizes:
    #     mvt(sockets, sizes)
    for sizes in jacobi_1d_sizes:
        jacobi_1d(sockets, sizes)
    for sizes in jacobi_2d_sizes:
        jacobi_2d(sockets, sizes)
    # for sizes in heat_3d_sizes:
    #     heat_3d(sizes, validate=True)
