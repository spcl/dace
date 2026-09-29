# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that a loop-dependent index that is not affine in the loop variable keeps the whole array dimension. """
import numpy as np

import dace

N = dace.symbol('N')


@dace.program
def rotating_table_read(a: dace.float64[N, 3], tbl: dace.int32[3], out: dace.float64[N, 3], K: dace.int64):
    for i in dace.map[0:N]:
        for k in range(K):
            for j in dace.map[0:3]:
                for t in range(k + 1):
                    out[i, j] = out[i, j] + a[i, j] * tbl[(j + t) % 3] + t


def test_modulo_index_in_nested_loop_passes_the_whole_table():
    sdfg = rotating_table_read.to_sdfg(simplify=False)
    table_shapes = [
        n.sdfg.arrays[e.dst_conn].shape for n, state in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.NestedSDFG) for e in state.in_edges(n)
        if n.sdfg.arrays[e.dst_conn].dtype == dace.int32
    ]
    assert table_shapes and all(shape == (3, ) for shape in table_shapes), table_shapes

    rng = np.random.default_rng(0)
    a = rng.random((5, 3))
    tbl = np.array([2, 3, -1], dtype=np.int32)
    out = np.zeros((5, 3))
    sdfg(a=a, tbl=tbl, out=out, N=5, K=4)
    ref = np.zeros((5, 3))
    for k in range(4):
        for j in range(3):
            for t in range(k + 1):
                ref[:, j] += a[:, j] * tbl[(j + t) % 3] + t
    assert np.allclose(out, ref)


if __name__ == '__main__':
    test_modulo_index_in_nested_loop_passes_the_whole_table()
