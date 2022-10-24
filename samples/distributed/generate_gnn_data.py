import argparse
import numpy as np
import scipy as sp

from scipy import sparse


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


nptype = np.float32
scalf = np.sqrt(2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-as", "--a-size", type=int, nargs="?", default=131072)
    parser.add_argument("-hc", "--h-cols", type=int, nargs="?", default=128)
    parser.add_argument("-sp", "--sparsity", type=float, nargs="?", default=1e-2)
    parser.add_argument("-nn", "--num-nodes", type=int, nargs="?", default=1)

    args = vars(parser.parse_args())

    print(f"Script called with options: {args}")

    nodes = args['num_nodes']
    assert nodes in grid

    rng = np.random.default_rng(42)

    Nx, Ny = grid[nodes]
    NArows = int(np.ceil(args['a_size'] * np.sqrt(nodes)/ nodes)) * nodes
    LArows = NArows // Nx
    LAcols = NArows // Ny
    NHcols = NWcols = args['h_cols']
    density = args['sparsity']

    lA = sparse.random(LArows, LAcols, density=density, format='csr', dtype=nptype, random_state=rng)
    A = sparse.bmat([[lA]*Ny]*Nx, format='coo', dtype=nptype)
    sparse.save_npz(f'reddit_n{nodes}_s{NArows}_graph.npz', A)

    nnz = len(A.data)
    lH = rng.random((LArows, NHcols), dtype=nptype)
    H = np.repeat(lH, Nx, axis=0)
    assert H.shape == (NArows, NHcols)
    data = {}
    data['feature'] = H
    data['node_types'] = np.ones((nnz, ), dtype=np.int32)
    data['node_ids'] = np.arange(nnz, dtype=np.int32)
    data['label'] = rng.integers(0, 128, size=(nnz, ), dtype=np.int32)
    np.savez('reddit_data.npz', data)
