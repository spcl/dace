# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Type-cast-on-lowering tests for :class:`TileBinop`.

A Tile operand whose dtype differs from the output is promoted to the output
dtype before the op. Widening (int -> wider int, int -> float / double, float ->
double) is allowed and must be numerically correct; narrowing (float / double ->
int, double -> float, int narrowing) raises ``NotImplementedError`` at expansion.
Covers both the ``pure`` (implicit C++ conversion) and ``scalar`` backend (an
explicit per-lane cast into the output-dtype buffer) lowerings.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop


def _build(a_dt, b_dt, c_dt, impl, name):
    """Two tile inputs (dtypes ``a_dt`` / ``b_dt``) -> TileBinop(+) -> ``c_dt`` tile.

    :param impl: Expansion implementation to stamp before lowering
        (``"pure"`` or ``"scalar"``).
    """
    widths = (8, )
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", widths, a_dt)
    sdfg.add_array("B", widths, b_dt)
    sdfg.add_array("C", widths, c_dt)
    state = sdfg.add_state("main")
    a, b, c = state.add_access("A"), state.add_access("B"), state.add_access("C")
    node = TileBinop(name="tb", widths=widths, op="+")
    node.implementation = impl
    state.add_node(node)
    state.add_edge(a, None, node, "_a", dace.Memlet("A[0:8]"))
    state.add_edge(b, None, node, "_b", dace.Memlet("B[0:8]"))
    state.add_edge(node, "_c", c, None, dace.Memlet("C[0:8]"))
    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


# (a_dtype, b_dtype, c_dtype, numpy a-dtype, numpy b-dtype, numpy c-dtype) — all widening.
_WIDENING = [
    (dace.int32, dace.float64, dace.float64, np.int32, np.float64, np.float64),
    (dace.float32, dace.float64, dace.float64, np.float32, np.float64, np.float64),
    (dace.int32, dace.int64, dace.int64, np.int32, np.int64, np.int64),
    (dace.int32, dace.float32, dace.float32, np.int32, np.float32, np.float32),
]


@pytest.mark.parametrize("impl", ["pure", "scalar"])
@pytest.mark.parametrize("a_dt,b_dt,c_dt,na,nb,nc", _WIDENING)
def test_tile_binop_widening_promotes(impl, a_dt, b_dt, c_dt, na, nb, nc):
    """A widening Tile operand is promoted to the output dtype; result matches numpy."""
    tag = f"{impl}_{a_dt.to_string()}_{b_dt.to_string()}"
    sdfg = _build(a_dt, b_dt, c_dt, impl, f"promote_{tag}")
    rng = np.random.default_rng(seed=hash(tag) & 0xFFFF)
    A = (rng.integers(1, 9, size=8).astype(na) if np.issubdtype(na, np.integer) else rng.random(8).astype(na))
    B = (rng.integers(1, 9, size=8).astype(nb) if np.issubdtype(nb, np.integer) else rng.random(8).astype(nb))
    C = np.zeros(8, dtype=nc)
    sdfg(A=A, B=B, C=C)
    ref = (A.astype(nc) + B.astype(nc)).astype(nc)
    np.testing.assert_allclose(C, ref, rtol=0, atol=0)


@pytest.mark.parametrize("c_dt", [dace.int32, dace.int64])
def test_tile_binop_rejects_float_to_int_narrowing(c_dt):
    """A fp64 Tile operand into an integer output is narrowing -> raises."""
    with pytest.raises(NotImplementedError, match="narrowing"):
        _build(dace.float64, dace.float64, c_dt, "pure", f"narrow_{c_dt.to_string()}")


def test_tile_binop_rejects_double_to_float_narrowing():
    """fp64 operand into an fp32 output is narrowing -> raises."""
    with pytest.raises(NotImplementedError, match="narrowing"):
        _build(dace.float64, dace.float32, dace.float32, "pure", "narrow_d2f")
