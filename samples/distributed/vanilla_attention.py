# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicit distributed vanilla attention sample programs. """
import csv
from enum import auto
import dace
import numpy as np
import timeit

from dace.transformation.auto.auto_optimize import auto_optimize
from dace.sdfg import utils
from datetime import datetime
from os.path import exists
from scipy import sparse


"""
formula: σ((A ⊙ (H x HT)) x H x W)

σ: ReLU activation function
A: Sparse (CSR) matrix (Arows, Acols). It is an adjacency matrix, i.e. its values are 0 and 1.
H: Dense matrix (Hrows, Hcols)
HT: H transposed
W: Dense matrix (Wrows, Wcols)

⊙: Hadamard product
x: Dot product (matrix multiplication)

Notes:
- H x HT is a dense matrix (Hrows, Hrows).
- Since it must be possible to compute the Hadamard product of A with (H x HT), Arows = Acols = Hrows.
- Since it must be possible to compute the dot product of H with W, Hcols = Wrows.
- A ⊙ (H x HT) is a sparse matrix (Arows, Acols) with the non-zeros at exactly the same places as A.
- The result is a dense matrix (Arows, Wcols).

Decomposition:
- We assume that Arows = Acols = Hrows >> Hcols = Wrows ≈ Wcols.
- We use a 2D process grid (Px, Py).
- A is split into (Px, Py) blocks. Each block is assigned to a single process.
- H is distributed twice:
  - H1 is split into (Px,) blocks. Each block is replicated in Py processes. Used as H in (A ⊙ (H x HT)).
  - H2 is split into (Py,) blocks. Each block is replicated in Px processes. Used as HT in (A ⊙ (H x HT)) and as H in (H x W).
- W is replicated in all processes.

Computation:
- We assume that A has nnz non-zero elements.
- A ⊙ (H x HT) is essentially nnz vector dot products, i.e. it costs 2 * nnz * Hcols FLOPs.
- H x W costs 2 * Arows * Hcols * Wcols FLOPs.
- (A ⊙ (H x HT)) x (H x W) costs 2 * nnz * Wcols FLOPs.
- ReLU costs Arows * Wcols FLOPs.
- Total (exact) is 2 * nnz * (Hcols + Wcols) + Arows * Wcols * (1 + 2 * Hcols) FLOPs.
- Total (taking into account Arows >> Hcols ≈ Wcols) is O(nnz + Arows).
"""

dctype = dace.float64
nptype = np.float64


grid = {
    #     [Px, Py]
    1:    [ 1,  1],
    2:    [ 1,  2],
    4:    [ 2,  2],
    8:    [ 2,  4],
    16:   [ 4,  4],
    32:   [ 4,  8],
    64:   [ 8,  8],
    128:  [ 8, 16],
    256:  [16, 16],
    512:  [16, 32],
}


# Each node does 28.2 GFLOPs.
# Scaling formula is for A rows is ceiling(base * sqrt(nodes) / nodes) * nodes
weak_scaling = {
    #:   ( Arows, Hcols, Wcols)
    1:   ( 20480,   128,   128),
    2:   ( 28964,   128,   128),
    4:   ( 40960,   128,   128),
    8:   ( 57928,   128,   128),
    16:  ( 81920,   128,   128),
    32:  (115872,   128,   128),
    64:  (163840,   128,   128),
    128: (231808,   128,   128),
    256: (327680,   128,   128),
    512: (463872,   128,   128),
}


# Global symbols
GArows, GAnnz, GHcols, GWcols = (dace.symbol(s) for s in ('GArows', 'GAnnz', 'GHcols', 'GWcols'))
# Local symbols
LArows, LAcols, LAnnz, LHcols, LWcols = (dace.symbol(s) for s in ('LArows', 'LAcols', 'LAnnz', 'LHcols', 'LWcols'))
# Process grid
Px, Py = (dace.symbol(s) for s in ('Px', 'Py'))
# Layer symbols
num_layers = dace.symbol('num_layers')


@dace.program
def vanilla_dace(A_rowptr: dace.int32[LArows+1],
                 A_rowidx: dace.int32[LAnnz],
                 A_colidx: dace.int32[LAnnz],
                 A_data: dctype[LAnnz],
                 H1: dctype[LArows, LHcols],
                #  H2: dctype[LAcols, LHcols],
                 W: dctype[LHcols, LWcols]) -> dctype[LArows, LWcols]:
    
    # Process grid
    parent_grid = dace.comm.Cart_create([Px, Py, 1])
    reduce_grid = dace.comm.Cart_sub(parent_grid, [False, True, False])
    h1_grid = dace.comm.Cart_sub(parent_grid, [True, False, True], exact_grid=0)
    h2_grid = dace.comm.Cart_sub(parent_grid, [False, True, True], exact_grid=0)
    bcast_grid = dace.comm.Cart_sub(parent_grid, [True, False, False])

    H2 = np.empty((LAcols, LHcols), dtype=H1.dtype)

    arr_h1 = dace.comm.Subarray((GArows, GHcols), H1, process_grid=h1_grid)
    arr_h2 = dace.comm.Subarray((GArows, GHcols), H2, process_grid=h2_grid)
    dace.comm.Redistribute(H1, arr_h1, H2, arr_h2)
    dace.comm.Bcast(H2, grid=bcast_grid)
    
    # HW = H x W
    HW = H2 @ W

    # S = A ⊙ (H x HT)
    values = np.zeros_like(A_data)
    dace.ahht(A_rowidx, A_colidx, H1, H2, values)
    
    # S x W
    out = np.empty((LArows, LWcols), dtype=nptype)
    dace.csrmm(A_rowptr, A_colidx, values, HW, out, 1, 0)

    # Reduce
    dace.comm.Allreduce(out, 'MPI_SUM', grid=reduce_grid)

    # ReLU
    return np.maximum(out, 0)


@dace.program
def vanilla_dace_loop(A_rowptr: dace.int32[LArows+1],
                      A_rowidx: dace.int32[LAnnz],
                      A_colidx: dace.int32[LAnnz],
                      A_data: dctype[LAnnz],
                      H1: dctype[LArows, LHcols],
                      W1: dctype[LHcols, LWcols],
                      W2: dctype[num_layers, LWcols, LWcols]) -> dctype[LArows, LWcols]:

    # Process grid
    parent_grid = dace.comm.Cart_create([Px, Py, 1])
    reduce_grid = dace.comm.Cart_sub(parent_grid, [False, True, False])
    h1_grid = dace.comm.Cart_sub(parent_grid, [True, False, True], exact_grid=0)
    h2_grid = dace.comm.Cart_sub(parent_grid, [False, True, True], exact_grid=0)
    bcast_grid = dace.comm.Cart_sub(parent_grid, [True, False, False])

    H2 = np.empty((LAcols, LHcols), dtype=H1.dtype)
    arr_h1 = dace.comm.Subarray((GArows, GHcols), H1, process_grid=h1_grid)
    arr_h2 = dace.comm.Subarray((GArows, GHcols), H2, process_grid=h2_grid)
    dace.comm.Redistribute(H1, arr_h1, H2, arr_h2)
    dace.comm.Bcast(H2, grid=bcast_grid)
    
    # HW = H x W
    HW = H2 @ W1
    # S = A ⊙ (H x HT)
    values = np.zeros_like(A_data)
    dace.ahht(A_rowidx, A_colidx, H1, H2, values)
    # S x W
    out = np.empty((LArows, LWcols), dtype=nptype)
    dace.csrmm(A_rowptr, A_colidx, values, HW, out, 1, 0)
    # Reduce
    dace.comm.Allreduce(out, 'MPI_SUM', grid=reduce_grid)
    # ReLU
    H1_new = np.maximum(out, 0)

    H2_new = np.empty_like(H1_new)

    for i in range(num_layers):
        arr_h1b = dace.comm.Subarray((GArows, GHcols), H1_new, process_grid=h1_grid)
        arr_h2b = dace.comm.Subarray((GArows, GHcols), H2_new, process_grid=h2_grid)
        dace.comm.Redistribute(H1_new, arr_h1b, H2_new, arr_h2b)
        dace.comm.Bcast(H2_new, grid=bcast_grid)

        # HW = H x W
        HW_new = H2_new @ W2[i]
        # S = A ⊙ (H x HT)
        values = np.zeros_like(A_data)
        dace.ahht(A_rowidx, A_colidx, H1_new, H2_new, values)
        # S x W
        out = np.empty((LArows, LWcols), dtype=nptype)
        dace.csrmm(A_rowptr, A_colidx, values, HW_new, out, 1, 0)
        # Reduce
        dace.comm.Allreduce(out, 'MPI_SUM', grid=reduce_grid)
        # ReLU
        H1_new[:] = np.maximum(out, 0)
    
    return H1_new



def vanilla_npsp(A: sparse.csr_matrix,
                 H: np.ndarray,
                 W: np.ndarray) -> np.ndarray:
    
    """ For single-node validation. """

    HW = H @ W
    HHT = H @ H.T
    for i in range(A.indptr.size - 1):
        start = A.indptr[i]
        finish = A.indptr[i+1]
        for j in range(start, finish):
            A.data[j] = HHT[i, A.indices[j]]
    out = A @ HW
    return np.maximum(out, 0)


def vanilla_npsp_loop(A: sparse.csr_matrix,
                      H: np.ndarray,
                      W1: np.ndarray,
                      W2: np.ndarray,
                      num_layers: int) -> np.ndarray:
    
    """ For single-node validation. """

    HW = H @ W1
    HHT = H @ H.T
    for i in range(A.indptr.size - 1):
        start = A.indptr[i]
        finish = A.indptr[i+1]
        for j in range(start, finish):
            A.data[j] = HHT[i, A.indices[j]]
    out = A @ HW
    H_new = np.maximum(out, 0)

    for i in range(num_layers):

        HW = H_new @ W2[i]
        HHT = H_new @ H_new.T
        for i in range(A.indptr.size - 1):
            start = A.indptr[i]
            finish = A.indptr[i+1]
            for j in range(start, finish):
                A.data[j] = HHT[i, A.indices[j]]
        out = A @ HW
        H_new = np.maximum(out, 0)

    return H_new



def vanilla_npsp2(A: sparse.csr_matrix,
                  H: np.ndarray,
                  W: np.ndarray) -> np.ndarray:
    
    """ For single-node validation. """

    HW = H @ W
    HHT = H @ H.T
    A_coo = A.tocoo()
    for i, (row, col) in enumerate(zip(A_coo.row, A_coo.col)):
        A.data[i] = HHT[row, col]
    out = A @ HW
    return np.maximum(out, 0)


# NOTE: The following should be used only with small sizes
# def vanilla_npsp(A: sparse.csr_matrix,
#                  H: np.ndarray,
#                  W: np.ndarray) -> np.ndarray:
    
#     """ For single-node validation. """

#     return np.maximum((A * (H @ H.T)) @ H @ W, 0)


def csr_to_coo(rowptr: np.ndarray) -> np.ndarray:
    """ Converts CSR row-pointer representation to COO row-indices. """
    nnz = rowptr[-1]  # Is this always correct?
    row_indices = np.empty((nnz,), dtype=rowptr.dtype)

    row = 0
    for i in range(rowptr.size - 1):
        row_indices[rowptr[i]:rowptr[i+1]] = row
        row += 1
    
    return row_indices


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


if __name__ == '__main__':

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in grid:
        raise ValueError("Selected number of MPI processes is not supported.")

    file_name = "dace_cpu_{n}_nodes.csv".format(n=size)
    field_names = ["datetime", "benchmark", "framework", "nodes", "sizes", "time"]
    
    sdfg, sdfgc = (None, ) * 2
    if rank == 0:
        # sdfg = vanilla_dace.to_sdfg(simplify=True)
        sdfg = vanilla_dace_loop.to_sdfg(simplify=True)
        sdfg = auto_optimize(sdfg, dace.DeviceType.CPU)
    func = utils.distributed_compile(sdfg, commworld)

    rng = np.random.default_rng(42)

    # Global sizes
    Nx, Ny = grid[size]
    NArows, NHcols, NWcols = weak_scaling[size]
    density = 0.01
    NArows = 2048
    num_layers = 2

    # Global data
    A = sparse.random(NArows, NArows, density=density, format='csr', dtype=nptype, random_state=rng)
    H = rng.random((NArows, NHcols), dtype=nptype)
    W = rng.random((NHcols, NWcols), dtype=nptype)
    W2 = rng.random((num_layers, NWcols, NWcols), dtype=nptype)

    # Local data
    cart_comm = commworld.Create_cart((Nx, Ny))
    x, y = cart_comm.Get_coords(rank)
    tx, ty = NArows // Nx, NArows // Ny
    lA = A[x*tx:(x+1)*tx, y*ty:(y+1)*ty]
    A_rowptr = lA.indptr.copy()
    A_rowidx = csr_to_coo(A_rowptr)
    A_colidx = lA.indices.copy()
    A_data = lA.data.copy()
    H1 = H[x*tx:(x+1)*tx, :].copy()
    H2 = H[y*ty:(y+1)*ty, :].copy()

    out = np.ndarray((tx, NWcols), dtype=nptype)

    # lA_coo = lA.tocoo()
    # assert(np.allclose(A_rowidx, lA_coo.row))
    # assert(np.allclose(A_colidx, lA_coo.col))

    if rank == 0:
        print(f"##### Vanilla Attention #####\nGlobal Sizes: {weak_scaling[size]}\nGrid: {grid[size]}""", flush=True)
    
    # runtimes = timeit.repeat(
    #     """out[:] = func(A_rowptr=A_rowptr, A_rowidx=A_rowidx, A_colidx=A_colidx, A_data=A_data, H1=H1, H2=H2, W=W,
    #                      LArows=tx, LAcols=ty, LAnnz=A_data.size, LHcols=NHcols, LWcols=NWcols,
    #                      Px=Nx, Py=Ny); commworld.Barrier()
    #     """,
    #     setup="commworld.Barrier()",
    #     repeat=10,
    #     number=1,
    #     globals=locals()
    # )
    # runtimes = timeit.repeat(
    #     """out[:] = func(A_rowptr=A_rowptr, A_rowidx=A_rowidx, A_colidx=A_colidx, A_data=A_data, H1=H1, W=W,
    #                      GArows=NArows, GAcols=NArows, GHcols=NHcols,
    #                      LArows=tx, LAcols=ty, LAnnz=A_data.size, LHcols=NHcols, LWcols=NWcols,
    #                      Px=Nx, Py=Ny); commworld.Barrier()
    #     """,
    #     setup="commworld.Barrier()",
    #     repeat=10,
    #     number=1,
    #     globals=locals()
    # )

    runtimes = timeit.repeat(
        """out[:] = func(A_rowptr=A_rowptr, A_rowidx=A_rowidx, A_colidx=A_colidx, A_data=A_data, H1=H1, W1=W, W2=W2,
                         num_layers=num_layers, GArows=NArows, GAcols=NArows, GHcols=NHcols,
                         LArows=tx, LAcols=ty, LAnnz=A_data.size, LHcols=NHcols, LWcols=NWcols,
                         Px=Nx, Py=Ny, print=print); commworld.Barrier()
        """,
        setup="commworld.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )
    
    if rank == 0:
        print(f"Median total runtime: {np.median(runtimes)} seconds", flush=True)
        write_time(str(datetime.now()), "vanilla", "dace_cpu", size, weak_scaling[size], runtimes, file_name, field_names, append=True)

    # ref = vanilla_npsp(A, H, W)
    ref = vanilla_npsp_loop(A, H, W, W2, num_layers)
    lref = ref[x*tx:(x+1)*tx, :]
    # ref2 = vanilla_npsp2(A, H, W)
    # lref2 = ref2[x*tx:(x+1)*tx, :]
    # assert(np.allclose(ref2, ref))
    # assert(np.allclose(out, lref2))
    assert(np.allclose(out, lref))
