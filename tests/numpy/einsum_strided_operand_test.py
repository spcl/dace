# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``np.einsum`` over a strided view: the GEMM lowering must read the view's own strides, not dense ones.

A packed column-major operand is contiguous too (``Data.is_packed_fortran_strides``). It is not copied; the GEMM
lowering passes its leading stride through. A column-major batch is packed storage but not a BLAS layout, so it is
copied, same as a strided slice.
"""

import numpy as np

import dace
from dace.frontend.common.einsum import create_einsum_sdfg

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
    assert any("_dense" in name for name in strided_matmul.to_sdfg().arrays)


def test_einsum_column_major_matrix_is_not_copied():
    """A packed column-major operand stays in place: BLAS reads the unit leading stride."""
    rng = np.random.default_rng(2)
    rows, contracted, cols = 5, 6, 4
    a = np.asfortranarray(rng.random((rows, contracted)))
    b = rng.random((contracted, cols))
    out = np.zeros((rows, cols))
    sdfg = dace.SDFG("einsum_column_major")
    sdfg.add_array("A", [rows, contracted], dace.float64, strides=[1, rows])
    sdfg.add_array("B", [contracted, cols], dace.float64, strides=[cols, 1])
    sdfg.add_array("C", [rows, cols], dace.float64, strides=[cols, 1])
    create_einsum_sdfg(sdfg, sdfg.add_state(), "nk,km->nm", "A", "B", output="C")
    assert not any("_dense" in name for name in sdfg.arrays)
    gemm = next(nested for nested in sdfg.all_sdfgs_recursive() if nested is not sdfg)
    assert list(gemm.arrays["X"].strides) == [1, rows]
    sdfg(A=a, B=b, C=out)
    assert np.allclose(out, np.einsum("nk,km->nm", a, b))


def test_einsum_column_major_output():
    """The product is written with the output's own column-major strides, not a row-major guess."""
    rng = np.random.default_rng(3)
    rows, contracted, cols = 5, 6, 4
    a = rng.random((rows, contracted))
    b = np.asfortranarray(rng.random((contracted, cols)))
    out = np.zeros((rows, cols), order="F")
    sdfg = dace.SDFG("einsum_column_major_output")
    sdfg.add_array("A", [rows, contracted], dace.float64, strides=[contracted, 1])
    sdfg.add_array("B", [contracted, cols], dace.float64, strides=[1, contracted])
    sdfg.add_array("C", [rows, cols], dace.float64, strides=[1, rows])
    create_einsum_sdfg(sdfg, sdfg.add_state(), "nk,km->nm", "A", "B", output="C")
    assert not any("_dense" in name for name in sdfg.arrays)
    sdfg(A=a, B=b, C=out)
    assert np.allclose(out, np.einsum("nk,km->nm", a, b))


def test_einsum_column_major_batch_is_copied():
    """A column-major batch is packed, but the unit stride is the batch. The GEMM path copies it to row-major."""
    rng = np.random.default_rng(4)
    batch, rows, contracted, cols = 2, 3, 4, 5
    a = np.asfortranarray(rng.random((batch, rows, contracted)))
    b = rng.random((batch, contracted, cols))
    out = np.zeros((batch, rows, cols))
    specs = {
        "A": ([batch, rows, contracted], [1, batch, batch * rows]),
        "B": ([batch, contracted, cols], [contracted * cols, cols, 1]),
        "C": ([batch, rows, cols], [rows * cols, cols, 1]),
    }
    sdfg = dace.SDFG("einsum_column_major_batch")
    for name, (shape, strides) in specs.items():
        sdfg.add_array(name, shape, dace.float64, strides=strides)
    create_einsum_sdfg(sdfg, sdfg.add_state(), "bmk,bkn->bmn", "A", "B", output="C")
    assert any(name.startswith("A_dense") for name in sdfg.arrays)
    sdfg(A=a, B=b, C=out)
    assert np.allclose(out, np.einsum("bmk,bkn->bmn", a, b))


if __name__ == "__main__":
    test_einsum_tap_of_weight_view()
    test_einsum_strided_matrix_operand()
    test_einsum_column_major_matrix_is_not_copied()
    test_einsum_column_major_output()
    test_einsum_column_major_batch_is_copied()
