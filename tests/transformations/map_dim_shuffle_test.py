# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.dataflow import MapDimShuffle


@dace.program
def miprog(A: dace.float64[20, 30, 40], B: dace.float64[40, 30, 20]):
    for i, j, k in dace.map[0:20, 0:30, 0:40]:
        with dace.tasklet:
            a << A[i, j, k]
            b >> B[k, j, i]
            b = a + 5


def test_map_dim_shuffle():
    A = np.random.rand(20, 30, 40)
    B = np.random.rand(40, 30, 20)
    expected = np.transpose(A, axes=(2, 1, 0)) + 5

    sdfg = miprog.to_sdfg()
    sdfg.simplify()
    sdfg.validate()

    # Apply map dim_shuffle
    state = sdfg.node(0)
    me = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry))
    assert me.map.params == ["i", "j", "k"]
    MapDimShuffle.apply_to(sdfg, map_entry=me, options={"parameters": ["k", "i", "j"]})
    assert me.map.params == ["k", "i", "j"]

    # Validate memlets
    sdfg.validate()

    # Validate correctness
    sdfg(A=A, B=B)
    assert np.allclose(B, expected)


if __name__ == '__main__':
    test_map_dim_shuffle()
