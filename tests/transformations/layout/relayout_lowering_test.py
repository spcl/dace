# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for build_relayout: lower a layout-algebra op sequence to a materialized copy."""
import numpy
import dace

from dace.libraries.layout.algebra import Permute, Block, Unblock
from dace.libraries.layout.lowering import build_relayout

N = dace.symbol("N")
M = dace.symbol("M")


def _run_relayout(in_shape_syms, ops, in_np):
    sdfg = dace.SDFG("relayout_test")
    sdfg.add_array("A_in", in_shape_syms, dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    build_relayout(sdfg, state, "A_in", "A_out", ops)
    sdfg.validate()
    out_desc = sdfg.arrays["A_out"]
    out_shape = tuple(int(dace.symbolic.evaluate(s, {N: in_np.shape[0], M: in_np.shape[-1]})) for s in out_desc.shape)
    A_out = numpy.zeros(out_shape, dtype=numpy.float64)
    symmap = {}
    if len(in_np.shape) >= 1:
        symmap["N"] = in_np.shape[0]
    if len(in_np.shape) >= 2:
        symmap["M"] = in_np.shape[1]
    sdfg(A_in=in_np.copy(), A_out=A_out, **symmap)
    return A_out


def test_relayout_permute_is_transpose():
    A = numpy.random.rand(6, 8)
    out = _run_relayout([N, M], [Permute((1, 0))], A)
    assert out.shape == (8, 6)
    assert numpy.array_equal(out, A.T)


def test_relayout_block_is_reshape():
    n = 64
    A = numpy.random.rand(n)
    out = _run_relayout([N], [Block(0, 16)], A)
    assert out.shape == (n // 16, 16)
    assert numpy.array_equal(out, A.reshape(n // 16, 16))


def test_relayout_block_unblock_is_copy():
    n = 64
    A = numpy.random.rand(n)
    # Block∘Unblock simplifies to [] -> plain copy, output shape == input.
    out = _run_relayout([N], [Block(0, 16), Unblock(0, 16)], A)
    assert out.shape == (n, )
    assert numpy.array_equal(out, A)


if __name__ == "__main__":
    test_relayout_permute_is_transpose()
    test_relayout_block_is_reshape()
    test_relayout_block_unblock_is_copy()
    print("relayout lowering tests PASS")
