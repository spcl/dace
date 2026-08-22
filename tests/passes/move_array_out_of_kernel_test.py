# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that ``_tile_extent`` returns the static tile width for a tiled inner-map extent so the
lifted transient's shape does not leak an out-of-scope outer-loop symbol into ``cudaMalloc``."""
import sympy

import dace
from dace.transformation.passes.move_array_out_of_kernel import _tile_extent, MoveArrayOutOfKernel


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


def test_get_new_shape_info_multidim_prepend_strides():
    """A GPU map that prepends >1 dimension must yield packed C-layout strides.

    Lifting an ``[64]`` transient out of a 2-D kernel ``map[0:128, 0:32]`` gives shape
    ``[128, 32, 64]``; the packed strides are ``[2048, 64, 1]``. Regression: the stride loop
    inserted the running accumulator *before* multiplying and iterated ``range_size[:-1]``, so
    it produced ``[64, 64, 1]`` -- both prepended dims wrongly shared stride 64.
    """
    sdfg = dace.SDFG('move_array_strides')
    state = sdfg.add_state('s')
    me, _mx = state.add_map('kernel', dict(i='0:128', j='0:32'), schedule=dace.dtypes.ScheduleType.GPU_Device)

    arr = dace.data.Array(dace.float64, [64])
    new_shape, new_strides, new_total, _new_offsets = MoveArrayOutOfKernel().get_new_shape_info(arr, [me])

    assert [int(s) for s in new_shape] == [128, 32, 64], new_shape
    assert [int(s) for s in new_strides] == [2048, 64, 1], new_strides
    assert int(new_total) == 128 * 32 * 64, new_total
