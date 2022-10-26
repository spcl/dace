""" Explicit distributed C-GNN and A-GNN sample programs running on Kronecker graphs. """
import argparse
import csv
import cupy
import dace
import numpy as np
import os
import timeit

from dace.transformation.auto.auto_optimize import auto_optimize
from dace.sdfg import utils
from datetime import datetime
from os.path import exists
from scipy import sparse

from cgnn_gpu import cgnn_dace_loop, cgnn_dace_loop_compute
from vanilla_attention_gpu import vanilla_dace_loop, vanilla_dace_loop_compute
from agnn_gpu import agnn_dace_loop, agnn_dace_loop_compute
from gat_gpu import GAT_dace_loop, GAT_dace_loop_compute


dctype = dace.float32
nptype = np.float32


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


def normalize_mat(X):
    for row in X:
        row = row /np.sum(row)
    return X

def normalize_vec(x):
    return x/np.sum(x)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, nargs="?", default="kronecker")
    parser.add_argument("-f", "--file", type=str, nargs="?", default="h-dar___symmetrized-sorted.npy")
    parser.add_argument("-m", "--model", choices=['cgnn', 'vanilla', 'agnn', 'gat'], nargs="?", default='cgnn')
    args = vars(parser.parse_args())

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in grid:
        raise ValueError("Selected number of MPI processes is not supported.")

    file_name = "real_dace_gpu_{n}_nodes.csv".format(n=size)
    field_names = ["datetime", "benchmark", "framework", "nodes", "sizes", "time"]

    def auto_gpu(dcprog):
        sdfg = dcprog.to_sdfg(simplify=True)
        sdfg.name = f"{sdfg.name}_cupy"
        for _, arr in sdfg.arrays.items():
            if not arr.transient:
                arr.storage = dace.StorageType.GPU_Global
        return auto_optimize(sdfg, device=dace.DeviceType.GPU)

    normal_program = None
    compute_program = None
    kwargs = None
    model = args['model']
    if model == 'cgnn':
        normal_program = cgnn_dace_loop
        compute_program = cgnn_dace_loop_compute
        kwargs = 'A_rowptr=A_rowptr, A_rowidx=A_rowidx, A_colidx=A_colidx, A_data=A_data, H2=H1, W1=lW, W2=lW2'
    elif model == 'vanilla':
        normal_program = vanilla_dace_loop
        compute_program = vanilla_dace_loop_compute
        kwargs = 'A_rowptr=A_rowptr, A_rowidx=A_rowidx, A_colidx=A_colidx, A_data=A_data, H1=H1, W1=lW, W2=lW2'
    elif model == 'agnn':
        normal_program = agnn_dace_loop
        compute_program = agnn_dace_loop_compute
        kwargs = 'A_rowptr=A_rowptr, A_rowidx=A_rowidx, A_colidx=A_colidx, A_data=A_data, H1=H1, W1=lW, W2=lW2'
    elif model == 'gat':
        normal_program = GAT_dace_loop
        compute_program = GAT_dace_loop_compute
        kwargs = 'A_rowptr=A_rowptr, A_rowidx=A_rowidx, A_colidx=A_colidx, A_data=A_data, H1=H1, W1=lW, W2=lW2, aL=laL, aR=laR'
    
    sdfg, sdfgc = (None, ) * 2
    if rank == 0:
        print("Generating SDFG and compiling ...", flush=True)
        sdfg = auto_gpu(normal_program)
        sdfgc = auto_gpu(compute_program)
    func = utils.distributed_compile(sdfg, commworld)
    funcc = utils.distributed_compile(sdfgc, commworld)

    rng = np.random.default_rng(42)

    Nx, Ny = grid[size]
    num_layers = 2  # (+1)

    # Graph file
    folder = args['directory']
    file = args['file']
    assert file.endswith('.npy')

    # Parameters
    # tokens = file.split('_')
    # NArows = 2 ** int(tokens[1][2:])
    # sp = float(tokens[3][2:-4])
    # NHcols = NWcols = 128
    NArows_orig = 22744077
    lcm = np.lcm(Nx, Ny)
    NArows = int(np.ceil(NArows_orig / lcm)) * lcm
    NHcols = NWrows = NWcols = 128

    # Data
    if rank == 0:
        print(f"Reading {file} ...", flush=True)
        gA = np.load(os.path.join(folder, file))
        gnnz = np.array([gA.shape[0]], dtype=np.int32)
    else:
        gnnz = np.array([0], dtype=np.int32)
    commworld.Bcast(gnnz)
    gnnz = gnnz[0]
    # print(f"rank {rank}: {gnnz}", flush=True)

    if rank == 0:
        print("Bcasting rows/cols ...", flush=True)
        grow = gA[:, 0].copy()
        gcol = gA[:, 1].copy()
    else:
        grow = np.empty((gnnz, ), dtype=np.int32)
        gcol = np.empty((gnnz, ), dtype=np.int32)
    commworld.Bcast(grow)
    commworld.Bcast(gcol)
    if rank == 0:
        print("... done!", flush=True)
    
    gdata = np.ones((gnnz, ), dtype=nptype)
    gA = sparse.coo_matrix((gdata, (grow, gcol)), shape=(NArows, NArows)).tocsr()
    del grow
    del gcol

    sp = gnnz / (NArows_orig * NArows_orig)

    # Local data and sizes
    cart_comm = commworld.Create_cart((Nx, Ny))
    x, y = cart_comm.Get_coords(rank)
    tx, ty = NArows // Nx, NArows // Ny
    lA = gA[x*tx:(x+1)*tx, y*ty:(y+1)*ty]
    lA.data[:] = 1
    if model == 'cgnn':
        H = normalize_mat(rng.random((ty, NHcols), dtype=nptype))
    else:
        H = normalize_mat(rng.random((tx, NHcols), dtype=nptype))
    W = normalize_mat(rng.random((NHcols, NWcols), dtype=nptype))
    W2 = normalize_mat(rng.random((num_layers, NWcols, NWcols), dtype=nptype))
    aR = normalize_vec(rng.random((NWcols, 1), dtype=nptype))
    aL = normalize_vec(rng.random((NWcols, 1), dtype=nptype))

    A_rowptr = cupy.asarray(lA.indptr)
    A_rowidx = cupy.asarray(csr_to_coo(lA.indptr))
    A_colidx = cupy.asarray(lA.indices)
    A_data = cupy.asarray(lA.data)
    H1 = cupy.asarray(H)
    lW = cupy.asarray(W)
    lW2 = cupy.asarray(W2)
    laR = cupy.asarray(aR)
    laL = cupy.asarray(aL)

    out = cupy.asarray(np.ndarray((tx, NWcols), dtype=nptype))

    if rank == 0:
        print(f"##### Graph {file} - Model {model} #####\nGlobal Sizes: ({NArows}, 128, 128, {sp})\nGrid: {grid[size]}""", flush=True)
    
    runtimes = timeit.repeat(
        f"""out[:] = func({kwargs},
                          num_layers=num_layers, GArows=NArows, GAcols=NArows, GHcols=NHcols,
                          LArows=tx, LAcols=ty, LAnnz=A_data.size, LHcols=NHcols, LWcols=NWcols,
                          Px=Nx, Py=Ny); commworld.Barrier()
        """,
        setup="commworld.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )
    
    if rank == 0:
        print(f"Median total runtime: {np.median(runtimes)} seconds", flush=True)
        write_time(str(datetime.now()), f"{model}", "dace_gpu", size, (NArows, 128, 128, sp), runtimes, file_name, field_names, append=True)

        runtimes = timeit.repeat(
            f"""out[:] = funcc({kwargs},
                                num_layers=num_layers, GArows=NArows, GAcols=NArows, GHcols=NHcols,
                                LArows=tx, LAcols=ty, LAnnz=A_data.size, LHcols=NHcols, LWcols=NWcols,
                                Px=Nx, Py=Ny);
            """,
            setup="",
            repeat=10,
            number=1,
            globals=locals()
        )

        print(f"Median compute runtime: {np.median(runtimes)} seconds")
        write_time(str(datetime.now()), f"{model}_compute", "dace_gpu", size, (NArows, 128, 128, sp), runtimes, file_name, field_names, append=True)
