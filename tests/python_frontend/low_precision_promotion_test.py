# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Arithmetic on a low-precision float keeps its type against an integer size, as NumPy's weak promotion does
for ``float16`` (and ``ml_dtypes`` does for ``bfloat16``)."""

import pytest

import dace

N, K = dace.symbol("N"), dace.symbol("K")


@pytest.mark.parametrize("dtype", [dace.float16, dace.bfloat16, dace.float32])
def test_a_low_precision_array_keeps_its_type_against_integer_sizes(dtype):
    """A bf16 array divided or scaled by an integer size came out ``float64``: ``bfloat16`` was
    classed as an unsigned integer, so the true division of two "integers" was double. HB's bf16 kernels then
    fed a double block into a bf16 ``tensordot`` (conv_transpose3d_avg_pool_clamp_softmax_multiply)."""

    @dace.program
    def scaled(a: dtype[N], b: dtype[N]):
        c = a / (K * K)
        d = a * K
        b[:] = c + d

    sdfg = scaled.to_sdfg(simplify=False)
    transients = {name: desc.dtype for name, desc in sdfg.arrays.items() if desc.transient}
    assert transients and all(t == dtype for t in transients.values()), transients


@pytest.mark.parametrize("dtype", [dace.float16, dace.bfloat16])
def test_a_low_precision_array_times_an_integer_array_promotes_as_float16_does(dtype):
    """``bfloat16 * int64`` has no promotion in ``ml_dtypes``, and the frontend raised ``DTypePromotionError`` on
    HB's hinge_loss (bf16 predictions times int64 labels). bfloat16 promotes as float16 does: to float64 here."""

    @dace.program
    def product(a: dtype[N], b: dace.int64[N], c: dace.float64[N]):
        c[:] = a * b

    sdfg = product.to_sdfg(simplify=False)
    transients = {name: desc.dtype for name, desc in sdfg.arrays.items() if desc.transient}
    assert transients and all(t == dace.float64 for t in transients.values()), transients


@pytest.mark.parametrize("dtype", [dace.float16, dace.bfloat16, dace.float32])
def test_a_python_float_does_not_widen_a_low_precision_array(dtype):
    """``1.0 / (1.0 + np.exp(-x))`` on a bf16 array came out ``float64``: numpy widens ml_dtypes' bfloat16 against a
    weak Python float, and HB's dist_mlp_tp then passed a double into a helper declared bf16. A Python scalar keeps
    a low-precision float's type, as NEP 50 does for float16."""

    @dace.program
    def sigmoid(a: dtype[N], b: dtype[N]):
        d = 1.0 + a
        b[:] = 1.0 / d

    sdfg = sigmoid.to_sdfg(simplify=False)
    transients = {name: desc.dtype for name, desc in sdfg.arrays.items() if desc.transient}
    assert transients and all(t == dtype for t in transients.values()), transients


if __name__ == "__main__":
    for dtype in (dace.float16, dace.bfloat16, dace.float32):
        test_a_low_precision_array_keeps_its_type_against_integer_sizes(dtype)
    for dtype in (dace.float16, dace.bfloat16):
        test_a_low_precision_array_times_an_integer_array_promotes_as_float16_does(dtype)
    for dtype in (dace.float16, dace.bfloat16, dace.float32):
        test_a_python_float_does_not_widen_a_low_precision_array(dtype)
