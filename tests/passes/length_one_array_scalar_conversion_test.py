# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the length-1 ``Array`` <-> ``Scalar`` conversion passes."""
import dace
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars


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


# ---------------------------------------------------------------------------
# ``filter`` knob: when None, every eligible array is scalarized; when a set is
# provided, only listed names are scalarized AND being in the filter overrides
# the ``transient_only`` check.
# ---------------------------------------------------------------------------
def _build_two_length_one_arrays_sdfg():
    """SDFG with two length-1 arrays: one transient and one non-transient. Both eligible
    under ``transient_only=False`` defaults."""
    sdfg = dace.SDFG('flt')
    sdfg.add_array('keep_me', (1, ), dace.float64, transient=False)
    sdfg.add_array('skip_me', (1, ), dace.float64, transient=False)
    sdfg.add_array('local', (1, ), dace.float64, transient=True)
    s0, s1 = sdfg.add_state('s0'), sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'k': 'keep_me[0] + skip_me[0] + local[0]'}))
    return sdfg


def test_filter_none_is_default_no_filtering():
    """``filter=None`` (the default) keeps existing behaviour: every length-1 array
    eligible under ``transient_only`` is scalarized."""
    sdfg = _build_two_length_one_arrays_sdfg()
    ConvertLengthOneArraysToScalars(recursive=False).apply_pass(sdfg, {})
    for name in ('keep_me', 'skip_me', 'local'):
        assert isinstance(sdfg.arrays[name], dace.data.Scalar), name


def test_filter_set_restricts_to_listed_names():
    """A non-empty filter set scalarizes only the listed arrays; the others stay arrays."""
    sdfg = _build_two_length_one_arrays_sdfg()
    ConvertLengthOneArraysToScalars(recursive=False, filter={'keep_me'}).apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays['keep_me'], dace.data.Scalar)
    assert isinstance(sdfg.arrays['skip_me'], dace.data.Array)
    assert isinstance(sdfg.arrays['local'], dace.data.Array)
    # ``[0]`` only stripped for the scalarized one; the other two keep their subscripts.
    assigns = list(sdfg.all_interstate_edges())[0].data.assignments['k']
    assert assigns == 'keep_me + skip_me[0] + local[0]', assigns


def test_filter_overrides_transient_only_for_listed_name():
    """A name in the filter is scalarized even when ``transient_only=True`` would otherwise
    skip it; arrays not in the filter are NOT scalarized regardless of ``transient_only``
    (the filter is the exclusive allow-list at the root level)."""
    sdfg = _build_two_length_one_arrays_sdfg()
    ConvertLengthOneArraysToScalars(recursive=False, transient_only=True, filter={'keep_me'}).apply_pass(sdfg, {})
    # keep_me is non-transient, but the filter overrides the transient_only restriction.
    assert isinstance(sdfg.arrays['keep_me'], dace.data.Scalar)
    # skip_me is non-transient and not in the filter -> left alone.
    assert isinstance(sdfg.arrays['skip_me'], dace.data.Array)
    # ``local`` is transient and would otherwise pass transient_only, but the filter is
    # the exclusive allow-list at root level: only listed names are touched.
    assert isinstance(sdfg.arrays['local'], dace.data.Array)


def test_filter_with_unknown_name_does_nothing():
    """A filter that names no real array yields no rewrites; the pass returns ``None``."""
    sdfg = _build_two_length_one_arrays_sdfg()
    result = ConvertLengthOneArraysToScalars(recursive=False, filter={'no_such_array'}).apply_pass(sdfg, {})
    assert result is None
    for name in ('keep_me', 'skip_me', 'local'):
        assert isinstance(sdfg.arrays[name], dace.data.Array), name


def test_filter_only_gates_root_level_nested_recursion_unaffected():
    """The filter applies to root-SDFG names only. Inner SDFG recursion (transient-only)
    proceeds independently and rewrites the nested transient length-1 array even when the
    filter is restrictive at the root."""
    inner = dace.SDFG('inner')
    inner.add_array('inner_local', (1, ), dace.float64, transient=True)
    inner.add_state('s')

    outer = dace.SDFG('outer')
    outer.add_array('outer_arr', (1, ), dace.float64, transient=False)
    outer.add_array('outer_unrelated', (1, ), dace.float64, transient=False)
    ostate = outer.add_state()
    ostate.add_nested_sdfg(sdfg=inner, inputs=set(), outputs=set())

    ConvertLengthOneArraysToScalars(recursive=True, filter={'outer_arr'}).apply_pass(outer, {})
    assert isinstance(outer.arrays['outer_arr'], dace.data.Scalar)
    assert isinstance(outer.arrays['outer_unrelated'], dace.data.Array)
    # Inner recursion follows transient_only=True (the default for nested) and is NOT gated
    # by the outer filter; the inner transient length-1 array gets scalarized.
    assert isinstance(inner.arrays['inner_local'], dace.data.Scalar)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
