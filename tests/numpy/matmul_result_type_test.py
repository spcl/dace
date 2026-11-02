# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The type of a matrix product of 16-bit floats, which depends on the array it is assigned to."""

import dace
from dace.libraries.blas import MatMul

M, N, K = 8, 4, 16


def product_type(program) -> dace.typeclass:
    sdfg = program.to_sdfg()
    (node,) = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MatMul)]
    (state,) = [s for s in sdfg.states() if node in s.nodes()]
    return sdfg.arrays[state.out_edges(node)[0].data.data].dtype


def test_into_wider_float_keeps_the_accumulation_type():
    """Assigned to a float32 array, the product is not rounded to 16 bits first (as BLAS products with a wider output
    type compute it)."""

    @dace.program
    def into_float32(a: dace.float16[M, K], b: dace.float16[N, K], c: dace.float32[M, N]):
        c[:] = a @ b.T

    assert product_type(into_float32) == dace.float32


def test_otherwise_numpy_type():
    @dace.program
    def into_float16(a: dace.float16[M, K], b: dace.float16[K, N], c: dace.float16[M, N]):
        c[:] = a @ b

    @dace.program
    def returned(a: dace.float16[M, K], b: dace.float16[K, N]):
        return a @ b

    assert product_type(into_float16) == dace.float16
    assert product_type(returned) == dace.float16


if __name__ == "__main__":
    test_into_wider_float_keeps_the_accumulation_type()
    test_otherwise_numpy_type()
