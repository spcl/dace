# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``np.einsum`` over a strided view: the GEMM lowering must read the view's own strides, not dense ones."""

import numpy as np

import dace

N, C, O, H, W, K = (dace.symbol(s) for s in "NCOHWK")


@dace.program
def one_tap(x: dace.float64[N, C, H, W], w: dace.float64[O, C, K, K], out: dace.float64[N, O, H - K + 1, W - K + 1]):
    out[:] = np.einsum("nchw,oc->nohw", x[:, :, 1 : 1 + (H - K + 1), 2 : 2 + (W - K + 1)], w[:, :, 1, 2])


@dace.program
def strided_matmul(a: dace.float64[N, C, K], b: dace.float64[C, O], out: dace.float64[N, O]):
    out[:] = np.einsum("nc,co->no", a[:, :, 1], b)


def test_einsum_tap_of_weight_view():
    rng = np.random.default_rng(0)
    x = rng.random((2, 3, 8, 9))
    w = rng.random((4, 3, 3, 3))
    out = np.zeros((2, 4, 6, 7))
    one_tap(x=x, w=w, out=out)
    expected = np.einsum("nchw,oc->nohw", x[:, :, 1:7, 2:9], w[:, :, 1, 2])
    assert np.allclose(out, expected)


def test_einsum_strided_matrix_operand():
    rng = np.random.default_rng(1)
    a = rng.random((5, 6, 3))
    b = rng.random((6, 4))
    out = np.zeros((5, 4))
    strided_matmul(a=a, b=b, out=out)
    assert np.allclose(out, a[:, :, 1] @ b)


if __name__ == "__main__":
    test_einsum_tap_of_weight_view()
    test_einsum_strided_matrix_operand()
