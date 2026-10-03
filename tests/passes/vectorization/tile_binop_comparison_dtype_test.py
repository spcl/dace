# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``TileBinop`` comparison answers ``bool``, so its operand never meets the output dtype.

TSVC ``s341`` / ``s342`` stream compaction builds its mask as ``__out = (b_index > 0.0)``: a double
operand into an ``int8`` mask. ``TileBinop.validate`` checked the double against the int8 output as if
``>`` promoted it, called that narrowing, and the vectorizer refused both kernels whole.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop

WIDTH = 8


def compare_against_zero(name: str, op: str, operand_dtype: dace.typeclass,
                         out_dtype: dace.typeclass) -> tuple[dace.SDFG, dace.SDFGState, TileBinop]:
    """``c = a <op> 0.0`` over one tile, with ``a`` and ``c`` typed by the caller."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", (WIDTH, ), operand_dtype)
    sdfg.add_array("c", (WIDTH, ), out_dtype)
    state = sdfg.add_state(is_start_block=True)
    node = TileBinop("cmp", widths=(WIDTH, ), op=op, kind_a="Tile", kind_b="Symbol", expr_b="0.0")
    state.add_node(node)
    state.add_edge(state.add_access("a"), None, node, "_a", dace.Memlet(f"a[0:{WIDTH}]"))
    state.add_edge(node, "_c", state.add_access("c"), None, dace.Memlet(f"c[0:{WIDTH}]"))
    return sdfg, state, node


def test_a_double_compared_into_an_int8_mask_validates():
    sdfg, state, sut = compare_against_zero("double_gt_into_int8", ">", dace.float64, dace.int8)

    sut.validate(sdfg, state)


def test_the_int8_mask_holds_each_double_compared_at_its_own_precision():
    """``0.25`` and ``1e-300`` read as ``0`` if the operand were cast to the int8 output first."""
    sdfg, _, _ = compare_against_zero("double_gt_into_int8_run", ">", dace.float64, dace.int8)
    a = np.array([-1.5, 0.25, 0.0, 3.0, -0.0, 1e-300, -2.0, 7.0])
    c = np.full(WIDTH, 5, dtype=np.int8)

    sdfg(a=a, c=c)

    np.testing.assert_array_equal(c, np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8))


def test_arithmetic_on_a_double_into_an_int8_output_is_still_narrowing():
    sdfg, state, sut = compare_against_zero("double_plus_into_int8", "+", dace.float64, dace.int8)

    with pytest.raises(NotImplementedError, match="narrowing"):
        sut.validate(sdfg, state)
