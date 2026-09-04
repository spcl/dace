# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Stencil`` over a SUBSET of a container, and the size check that guards it.

``parse_connectors`` inferred the iteration space from the data DESCRIPTORS, so a stencil reading a
window of a larger buffer was rejected as "Inconsistent input sizes" -- the same defect as the
linalg nodes in ``tests/library/linalg_subset_test.py``. The extents belong to the memlets; what the
expansions genuinely cannot do is address a region that is not contiguous in its container, and that
is refused explicitly instead of read at the wrong offsets.
"""
import numpy as np
import pytest

import dace
from dace.libraries.stencil import Stencil

SIZE = dace.symbol('size')
ROWS = dace.symbol('rows')
COLS = dace.symbol('cols')

CODE_1D = """\
tmp0 = (a[0] + a[1])
tmp1 = (tmp0 + a[2])
res[1] = (dace.float32(0.3333) * tmp1)"""


def make_1d_sdfg(name: str, a_shape, a_memlet: str) -> dace.SDFG:
    """A three-point 1-D stencil whose input arrives through ``a_memlet``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', a_shape, dtype=dace.float32)
    _, res_desc = sdfg.add_array('res', (SIZE, ), dtype=dace.float32)

    state = sdfg.add_state('s', is_start_block=True)
    node = Stencil('stencil_test', CODE_1D, inputs={'a'}, outputs={'res'})
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, 'a', dace.Memlet(a_memlet))
    state.add_edge(node, 'res', state.add_write('res'), None, dace.Memlet.from_array('res', res_desc))
    return sdfg


def three_point_reference(a: np.ndarray, n: int) -> np.ndarray:
    expected = np.zeros(n, dtype=np.float32)
    expected[1:n - 1] = np.float32(0.3333) * (a[0:n - 2] + a[1:n - 1] + a[2:n])
    return expected


def test_stencil_over_a_window_of_a_larger_buffer():
    """The iteration space is the window the memlet carries, not the buffer it is cut from. Reading
    the descriptor instead raised ``Inconsistent input sizes: (2*size,) vs. (size,)``."""
    n = 16
    sdfg = make_1d_sdfg('stencil_window', (2 * SIZE, ), 'a[0:size]')
    sdfg.validate()

    a = np.arange(2 * n, dtype=np.float32)
    res = np.zeros(n, dtype=np.float32)
    sdfg(a=a, res=res, size=n)
    assert np.allclose(res, three_point_reference(a, n)), f'the window read the wrong data: {res}'


def test_a_window_at_a_nonzero_offset_reads_from_the_offset():
    """The pointer arrives at the subset origin, so the same stencil over the SECOND half of the
    buffer must read the second half -- proving the fix moves the window, not just its length."""
    n = 16
    sdfg = make_1d_sdfg('stencil_window_offset', (2 * SIZE, ), 'a[size:2*size]')
    sdfg.validate()

    a = np.arange(2 * n, dtype=np.float32)
    res = np.zeros(n, dtype=np.float32)
    sdfg(a=a, res=res, size=n)
    assert np.allclose(res, three_point_reference(a[n:], n)), f'the window read the wrong half: {res}'


def test_a_noncontiguous_region_is_refused_not_miscompiled():
    """The expansion builds its connector arrays packed, so a region whose container is strided
    around it cannot be addressed. Refusing is the point: before the extents came from the memlet
    this raised a size error, and reading it packed would silently return the wrong elements."""
    sdfg = dace.SDFG('stencil_strided_window')
    sdfg.add_array('a', (ROWS, COLS), dtype=dace.float32)
    _, res_desc = sdfg.add_array('res', (ROWS, 4), dtype=dace.float32)

    state = sdfg.add_state('s', is_start_block=True)
    node = Stencil('stencil_test', """\
res[0, 0] = (a[0, 0] + a[0, 1])""", inputs={'a'}, outputs={'res'})
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, 'a', dace.Memlet('a[0:rows, 4:8]'))
    state.add_edge(node, 'res', state.add_write('res'), None, dace.Memlet.from_array('res', res_desc))

    with pytest.raises(NotImplementedError, match='not contiguous'):
        sdfg.expand_library_nodes()


def test_inconsistent_sizes_are_still_rejected():
    """The size agreement check still has to fail: two connectors of the same rank that move
    different extents are not one iteration space."""
    sdfg = make_1d_sdfg('stencil_mismatch', (2 * SIZE, ), 'a[0:2*size]')
    with pytest.raises(ValueError, match='Inconsistent input sizes'):
        sdfg.expand_library_nodes()


if __name__ == '__main__':
    test_stencil_over_a_window_of_a_larger_buffer()
    test_a_window_at_a_nonzero_offset_reads_from_the_offset()
    test_a_noncontiguous_region_is_refused_not_miscompiled()
    test_inconsistent_sizes_are_still_rejected()
