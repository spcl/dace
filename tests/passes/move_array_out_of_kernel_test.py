# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``MoveArrayOutOfKernel`` shape derivation.

The lift prepends per-iteration dimensions onto a transient that lives
inside a ``GPU_Device`` / ``GPU_ThreadBlock`` map scope. For a tiled inner
map of the form ``i = start : Min(X, start+Y) + 1`` the per-iteration
extent ``-start + Min(X, start+Y) + 1`` references the outer-loop symbol
``start``, which is not in scope at the lift destination (host level)
and would leak into ``cudaMalloc`` size expressions. The fix lives in
``_tile_extent`` and substitutes the static tile width ``Y + 1``.
"""
import sympy

from dace.transformation.passes.move_array_out_of_kernel import _tile_extent


def test_tile_extent_recognises_min_pattern():
    """``i = b_i : Min(N-1, b_i+31) + 1`` — the inner GPU_ThreadBlock map
    DaCe produces for a block-stride of 32. ``_tile_extent`` must
    return the static tile width 32 (independent of ``b_i``)."""
    b_i = sympy.Symbol('b_i')
    N = sympy.Symbol('N')
    max_elem = sympy.Min(N - 1, b_i + 31)
    min_elem = b_i
    extent = _tile_extent(max_elem, min_elem)
    assert extent == 32, f"expected 32, got {extent}"
    assert b_i not in extent.free_symbols, f"tile extent leaks outer-loop symbol: {extent.free_symbols}"


def test_tile_extent_falls_back_for_plain_range():
    """No ``Min`` in the upper bound — the existing symbolic form is
    returned unchanged. Used by ``Sequential`` inner maps and any
    non-tiled GPU_Device map."""
    W = sympy.Symbol('W')
    extent = _tile_extent(W - 1, sympy.Integer(0))
    assert sympy.simplify(extent - W) == 0, f"expected W, got {extent}"


def test_tile_extent_handles_outer_block_strided_loop():
    """Outer GPU_Device map of the form ``b_i = 0:N:32`` (no ``Min`` in
    the upper bound). The fallback returns ``N`` (max+1-min for the
    range), which is host-visible and correct for the outer dim."""
    N = sympy.Symbol('N')
    # max_element of ``0:N:32`` is N - (N - 1) % 32 ... in DaCe this
    # comes back as ``N - 1`` for max_element() of strided ranges; the
    # test fixes that as the boundary condition and just checks no leak.
    extent = _tile_extent(N - 1, sympy.Integer(0))
    assert sympy.simplify(extent - N) == 0
    assert sympy.Symbol('b_i') not in extent.free_symbols
