# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that the backward pass keeps one dtype per symbol name through library expansions and inlining."""
import numpy as np
import pytest

import dace
from dace.autodiff import add_backward_pass

M, N = (dace.symbol(s, dtype=dace.int64) for s in ('M', 'N'))


@dace.program
def upper_row_products(data: dace.float64[N, M]):
    corr = np.zeros((M, M), dtype=np.float64)
    for i in range(M - 1):
        corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]
    return corr


@pytest.mark.autodiff
def test_a_product_sized_by_the_loop_variable_keeps_its_dtype_through_inlining():
    """The gemv expansion parses ``M - i`` from strings; an ``i`` minted at the default dtype would meet the
    declared int64 ``i`` once the expansion is inlined (polybench correlation)."""

    @dace.program
    def outer(data: dace.float64[6, 8]):
        return np.sum(upper_row_products(data))

    sdfg = outer.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["data"], outputs=["__return"])
    sdfg.validate()
