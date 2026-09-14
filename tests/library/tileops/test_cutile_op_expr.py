# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The cuTile expression table spells each tasklet operator with its SDFG meaning."""
import numpy
import pytest

from dace.libraries.tileops.nodes.tile_binop import CUTE_OP_EXPR


def test_cutile_modulo_text_takes_the_sign_of_the_dividend():
    """``%`` in a tasklet is C's truncating remainder, so the cuTile text puts the dividend's sign on ``|a| % |b|``."""
    emitted = CUTE_OP_EXPR["%"].format(lhs="a", rhs="b")
    assert emitted == "ct.where(a < 0, -(ct.abs(a) % ct.abs(b)), ct.abs(a) % ct.abs(b))"


@pytest.mark.parametrize("dtype", [numpy.int32, numpy.int64, numpy.float64])
def test_cutile_modulo_matches_c_remainder_on_mixed_signs(dtype):
    """Evaluated with NumPy's floored ``%`` standing in for cuTile's, the text computes C's ``fmod``."""
    a = numpy.array([7, -7, 7, -7, 6, -6, 0, 1], dtype=dtype)
    b = numpy.array([3, 3, -3, -3, 3, -3, 5, -5], dtype=dtype)
    emitted = CUTE_OP_EXPR["%"].format(lhs="a", rhs="b")
    result = eval(emitted, {"ct": numpy, "a": a, "b": b})
    numpy.testing.assert_array_equal(result, numpy.array([1, -1, 1, -1, 0, 0, 0, 1], dtype=dtype))
    numpy.testing.assert_array_equal(result, numpy.fmod(a, b))
