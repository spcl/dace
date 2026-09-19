# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that a broadcast advanced index reaches the memlet as an expression, not a symbol name. """
import numpy as np

import dace as dc

N = dc.symbol('N', dtype=dc.int64, positive=True)


def test_an_open_mesh_gather_indexes_a_size_one_dimension():
    """broadcast_together spells a size-1 dimension as the literal 0, and an index that is not a
    valid identifier cannot name a symbol, so an np.ix_ gather refused to parse at all."""

    @dc.program
    def gather(field: dc.float64[N, N, N], xs: dc.int64[2], ys: dc.int64[2], zs: dc.int64[2], out: dc.float64[2, 2, 2]):
        gx, gy, gz = np.ix_(xs, ys, zs)
        out[:] = field[gx, gy, gz]

    size = 4
    field = np.zeros((size, size, size), dtype=np.float64)
    field[:] = np.arange(size**3, dtype=np.float64).reshape(size, size, size)
    xs = np.array([0, 3], dtype=np.int64)
    ys = np.array([1, 2], dtype=np.int64)
    zs = np.array([2, 3], dtype=np.int64)
    out = np.zeros((2, 2, 2), dtype=np.float64)

    gather(field=field, xs=xs, ys=ys, zs=zs, out=out, N=size)
    want = field[np.ix_(xs, ys, zs)]
    assert np.array_equal(out, want), (out, want)


def test_a_one_dimensional_gather_still_indexes_by_symbol():
    """The fix parses every index rather than naming it, so the ordinary case where the index IS an
    identifier has to keep working."""

    @dc.program
    def take(field: dc.float64[N], idx: dc.int64[3], out: dc.float64[3]):
        out[:] = field[idx]

    field = np.arange(8, dtype=np.float64)
    idx = np.array([7, 0, 4], dtype=np.int64)
    out = np.zeros(3, dtype=np.float64)

    take(field=field, idx=idx, out=out, N=8)
    assert np.array_equal(out, field[idx]), (out, field[idx])


if __name__ == '__main__':
    test_an_open_mesh_gather_indexes_a_size_one_dimension()
    test_a_one_dimensional_gather_still_indexes_by_symbol()
