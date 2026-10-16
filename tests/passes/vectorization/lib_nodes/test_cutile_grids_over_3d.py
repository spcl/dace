# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for cuTile launch grids with more than 3 dimensions.

The ``cuda.tile`` runtime caps the launch grid at three axes (``ct.launch``
takes a ``Dim3`` and ``ct.bid(axis)`` only accepts ``axis in {0, 1, 2}``). A
tiled ``CuTile`` map with more than three dimensions (e.g. a 4-D or 5-D
elementwise kernel, ``softmax``, ``conv2d``) therefore has its extra grid
dimensions FOLDED, row-major, onto grid axis 0 and recovered inside the kernel
via integer div/mod. The single canonical fold layout lives in
:mod:`dace.libraries.tileops.cutile` and is shared by the map-entry
codegen (``codegen/py/cutile_target.py``) and every tile-op ``cutile``
expansion, so map-entry block IDs and tile-op block IDs agree.

These tests cover:

* the pure fold helpers (``cutile_launch_grid_dims`` / ``cutile_bid_expr``),
* the generated code structure (no ``ct.bid`` axis exceeds 2; the launch grid
  folds the leading dims onto axis 0),
* end-to-end 4-D and 5-D elementwise kernels compiled and run on the GPU and
  compared against NumPy (including non-divisible tile boundaries).
"""
import ast

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.tileops.cutile import cutile_bid_expr, cutile_launch_grid_dims
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile

_N = dace.symbol("N")
_A = dace.symbol("A")
_B = dace.symbol("B")
_C = dace.symbol("C")
_D = dace.symbol("D")


@dace.program
def _add4d(x: dace.float32[_N, _A, _B, _C], y: dace.float32[_N, _A, _B, _C], z: dace.float32[_N, _A, _B, _C]):
    z[:] = x + y


@dace.program
def _fma5d(x: dace.float32[_N, _A, _B, _C, _D], y: dace.float32[_N, _A, _B, _C, _D],
           z: dace.float32[_N, _A, _B, _C, _D]):
    z[:] = x * y + x


def _lower(prog, widths):
    """Canonicalize + apply the cuTile pipeline, returning the Python-backend SDFG."""
    sdfg = prog.to_sdfg(simplify=True)
    # ``assumption_guard=False`` mirrors ``canonicalize_for_cutile``: the guard-on
    # canonicalize injects a CPP ``__builtin_trap`` tasklet the Python backend
    # cannot codegen.
    canonicalize(sdfg)
    VectorizeCuTile(widths=widths).apply_pass(sdfg, {})
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def _main_code(sdfg):
    """Return the generated Python code object body for the main SDFG."""
    return next(co for co in sdfg.generate_code() if co.name == sdfg.name).code


def _max_bid_axis(code):
    """Return the largest ``ct.bid(<axis>)`` axis literal in generated code."""
    import re
    return max((int(m) for m in re.findall(r"ct\.bid\((\d+)\)", code)), default=-1)


# ============================================================
# Pure fold-helper unit tests (no GPU)
# ============================================================


class TestFoldHelpers:
    """Unit tests for the canonical grid-fold contract."""

    def test_launch_grid_dims_le_3_padded(self):
        assert cutile_launch_grid_dims(["a"]) == ["a", "1", "1"]
        assert cutile_launch_grid_dims(["a", "b"]) == ["a", "b", "1"]
        assert cutile_launch_grid_dims(["a", "b", "c"]) == ["a", "b", "c"]

    def test_launch_grid_dims_4d_folds_leading_two(self):
        # 4 dims: leading 2 fold onto axis 0, inner 2 -> axes 1, 2.
        assert cutile_launch_grid_dims(["a", "b", "c", "d"]) == ["(a) * (b)", "c", "d"]

    def test_launch_grid_dims_5d_folds_leading_three(self):
        assert cutile_launch_grid_dims(["a", "b", "c", "d", "e"]) == ["(a) * (b) * (c)", "d", "e"]

    def test_bid_expr_identity_when_le_3(self):
        for m in (1, 2, 3):
            for d in range(m):
                assert cutile_bid_expr(d, m, ["g"] * m) == f"ct.bid({d})"

    def test_bid_expr_4d_recovery(self):
        g = ["gN", "gA", "gB", "gC"]
        # tail dims read axes 1, 2 directly
        assert cutile_bid_expr(2, 4, g) == "ct.bid(1)"
        assert cutile_bid_expr(3, 4, g) == "ct.bid(2)"
        # innermost folded dim (axis 1 in map order) is a pure modulo
        assert cutile_bid_expr(1, 4, g) == "(ct.bid(0) % (gA))"
        # outermost folded dim is a pure quotient
        assert cutile_bid_expr(0, 4, g) == "(ct.bid(0) // ((gA)))"

    def test_bid_expr_5d_recovery(self):
        g = ["gN", "gK", "gKK", "gCin", "gCout"]
        assert cutile_bid_expr(3, 5, g) == "ct.bid(1)"
        assert cutile_bid_expr(4, 5, g) == "ct.bid(2)"
        assert cutile_bid_expr(2, 5, g) == "(ct.bid(0) % (gKK))"
        assert cutile_bid_expr(1, 5, g) == "((ct.bid(0) // ((gKK))) % (gK))"
        assert cutile_bid_expr(0, 5, g) == "(ct.bid(0) // ((gK) * (gKK)))"


# ============================================================
# Codegen-structure tests (no GPU)
# ============================================================


class TestCodegenStructure:
    """The generated kernel must never emit an out-of-range ``ct.bid`` axis."""

    def test_4d_no_bid_axis_above_2(self):
        sdfg = _lower(_add4d, (8, 8, 8))
        code = _main_code(sdfg)
        ast.parse(code)  # valid Python
        assert _max_bid_axis(code) <= 2, "generated code uses a ct.bid axis > 2"
        # launch grid folds the leading dims onto axis 0 (a product expression)
        assert "ct.launch(" in code
        assert "*" in code  # folded axis-0 product
        # folded block-id recovery is present in the kernel
        assert "ct.bid(0) %" in code and "ct.bid(0) //" in code

    def test_5d_no_bid_axis_above_2(self):
        sdfg = _lower(_fma5d, (8, 8, 8))
        code = _main_code(sdfg)
        ast.parse(code)
        assert _max_bid_axis(code) <= 2, "generated code uses a ct.bid axis > 2"


# ============================================================
# End-to-end GPU runtime tests
# ============================================================


@pytest.mark.gpu
class TestRuntimeOver3DGrid:
    """4-D / 5-D elementwise kernels lower, compile, run, and match NumPy."""

    @pytest.mark.parametrize("shape", [(2, 8, 8, 8), (2, 3, 10, 10)])
    def test_add4d_matches_numpy(self, shape):
        sdfg = _lower(_add4d, (8, 8, 8))
        csdfg = sdfg.compile()
        x = np.random.rand(*shape).astype(np.float32)
        y = np.random.rand(*shape).astype(np.float32)
        z = np.zeros(shape, dtype=np.float32)
        N, A, B, C = shape
        csdfg(x=x, y=y, z=z, N=N, A=A, B=B, C=C)
        np.testing.assert_allclose(z, x + y, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("shape", [(2, 3, 8, 8, 8), (2, 3, 4, 10, 10)])
    def test_fma5d_matches_numpy(self, shape):
        sdfg = _lower(_fma5d, (8, 8, 8))
        csdfg = sdfg.compile()
        x = np.random.rand(*shape).astype(np.float32)
        y = np.random.rand(*shape).astype(np.float32)
        z = np.zeros(shape, dtype=np.float32)
        N, A, B, C, D = shape
        csdfg(x=x, y=y, z=z, N=N, A=A, B=B, C=C, D=D)
        np.testing.assert_allclose(z, x * y + x, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
