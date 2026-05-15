# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that ``_tile_extent`` returns the static tile width for a tiled inner-map extent so the
lifted transient's shape does not leak an out-of-scope outer-loop symbol into ``cudaMalloc``."""
import sympy

from dace.transformation.passes.move_array_out_of_kernel import _tile_extent


def test_tile_extent_recognises_min_pattern():
    """For a ``Min``-bounded inner-map extent, ``_tile_extent`` returns the static tile width 32."""
    b_i = sympy.Symbol('b_i')
    N = sympy.Symbol('N')
    max_elem = sympy.Min(N - 1, b_i + 31)
    min_elem = b_i
    extent = _tile_extent(max_elem, min_elem)
    assert extent == 32, f"expected 32, got {extent}"
    assert b_i not in extent.free_symbols, f"tile extent leaks outer-loop symbol: {extent.free_symbols}"


def test_tile_extent_falls_back_for_plain_range():
    """No ``Min`` in the upper bound: the symbolic extent is returned unchanged."""
    W = sympy.Symbol('W')
    extent = _tile_extent(W - 1, sympy.Integer(0))
    assert sympy.simplify(extent - W) == 0, f"expected W, got {extent}"


def test_tile_extent_handles_outer_block_strided_loop():
    """Outer strided GPU_Device map ``b_i = 0:N:32``: the fallback returns the host-visible ``N``."""
    N = sympy.Symbol('N')
    # max_element() of a strided range comes back as ``N - 1``; pin that and check there is no leak.
    extent = _tile_extent(N - 1, sympy.Integer(0))
    assert sympy.simplify(extent - N) == 0
    assert sympy.Symbol('b_i') not in extent.free_symbols
