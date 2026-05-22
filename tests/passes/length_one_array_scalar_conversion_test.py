# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the length-1 ``Array`` <-> ``Scalar`` conversion passes."""
import dace
from dace.transformation.passes.length_one_array_scalar_conversion import (ConvertLengthOneArraysToScalars,
                                                                           ConvertScalarsToLengthOneArrays)


def test_scalarize_rewrites_length_one_array():
    """A shape-``(1,)`` array becomes a true ``Scalar`` and its ``[0]`` accessor is dropped."""
    sdfg = dace.SDFG('scalarize')
    sdfg.add_array('a', (1, ), dace.float64)
    s0, s1 = sdfg.add_state('s0'), sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'k': 'a[0] + 1'}))

    ConvertLengthOneArraysToScalars(recursive=False).apply_pass(sdfg, {})

    assert isinstance(sdfg.arrays['a'], dace.data.Scalar)
    assert list(sdfg.all_interstate_edges())[0].data.assignments['k'] == 'a + 1'


def test_scalarize_keeps_overlapping_name_subscript():
    """A scalarized name that is a suffix of another array must not eat that
    array's literal ``[0]`` index (scalarized ``ar`` vs multi-element ``bar``)."""
    sdfg = dace.SDFG('overlap')
    sdfg.add_array('ar', (1, ), dace.float64)
    sdfg.add_array('bar', (4, ), dace.float64)
    s0, s1 = sdfg.add_state('s0'), sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'k': 'ar[0] + bar[0]'}))

    ConvertLengthOneArraysToScalars(recursive=False).apply_pass(sdfg, {})

    assert isinstance(sdfg.arrays['ar'], dace.data.Scalar)
    assert isinstance(sdfg.arrays['bar'], dace.data.Array)
    assert list(sdfg.all_interstate_edges())[0].data.assignments['k'] == 'ar + bar[0]'


def test_collapsed_memlet_preserves_dynamic():
    """Collapsing a scalarized array's memlet to element 0 keeps the dynamic flag."""
    sdfg = dace.SDFG('dynmem')
    sdfg.add_array('a', (1, ), dace.float64, transient=True)
    sdfg.add_array('b', (1, ), dace.float64)
    state = sdfg.add_state('s')
    an_a, an_b = state.add_access('a'), state.add_access('b')
    state.add_nedge(an_a, an_b, dace.Memlet(data='a', subset='0', dynamic=True))

    ConvertLengthOneArraysToScalars(recursive=False).apply_pass(sdfg, {})

    assert isinstance(sdfg.arrays['a'], dace.data.Scalar)
    assert state.edges()[0].data.dynamic is True


def test_roundtrip_scalar_to_array_and_back():
    """``Scalar`` -> length-1 ``Array`` -> ``Scalar`` returns to the original descriptor kind."""
    sdfg = dace.SDFG('roundtrip')
    sdfg.add_scalar('s', dace.float64, transient=True)
    sdfg.add_state('only')

    ConvertScalarsToLengthOneArrays(recursive=False).apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays['s'], dace.data.Array)
    assert tuple(sdfg.arrays['s'].shape) == (1, )

    ConvertLengthOneArraysToScalars(recursive=False).apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays['s'], dace.data.Scalar)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
