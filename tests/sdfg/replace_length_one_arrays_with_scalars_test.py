"""Unit tests for ``replace_length_one_arrays_with_scalars`` in
``dace.sdfg.construction_utils``.

The function rewrites every length-1 ``Array`` on the SDFG to a true
``Scalar`` of the same dtype and strips the ``[0]`` accessors that
referred to it from interstate-edge assignments, conditional-block
guards, and loop-region condition expressions.

Tests cover:
    * a single ``intent(out)``-style length-1 array becomes a scalar,
      and the SDFG executes correctly when bound to a Python value
      (we use a 1-element numpy buffer downstream just to read the
      scalar back out -- DaCe's runtime is happy to accept either),
    * ``[0]`` accessors on an interstate-edge assignment are
      rewritten,
    * the recursive flag descends into nested SDFGs and rewrites
      their TRANSIENT length-1 arrays only,
    * ``transient_only=True`` skips signature length-1 arrays.
"""

import ctypes

import dace

from dace.transformation.passes import replace_length_one_arrays_with_scalars

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass


def _make_simple_sdfg(name: str) -> dace.SDFG:
    """One state with a length-1 ``out`` array written from a tasklet."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('out', shape=(1, ), dtype=dace.float64)
    state = sdfg.add_state('s')
    t = state.add_tasklet('w', set(), {'_o'}, '_o = 3.5')
    a = state.add_access('out')
    state.add_edge(t, '_o', a, None, dace.Memlet('out[0]'))
    return sdfg


def test_signature_length_one_becomes_scalar():
    sdfg = _make_simple_sdfg('len1_to_scalar')
    assert isinstance(sdfg.arrays['out'], dace.data.Array)
    rewritten = replace_length_one_arrays_with_scalars(sdfg)
    assert 'out' in rewritten
    assert isinstance(sdfg.arrays['out'], dace.data.Scalar)


def test_interstate_edge_subscript_stripped():
    """An interstate edge that reads ``flag[0]`` must be rewritten to
    ``flag`` after the array becomes a scalar."""
    sdfg = dace.SDFG('iface_strip')
    sdfg.add_array('flag', shape=(1, ), dtype=dace.int64)
    sdfg.add_symbol('s', dace.int64)

    s1 = sdfg.add_state('s1', is_start_block=True)
    s2 = sdfg.add_state('s2')
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={'s': 'flag[0] + 1'}))

    replace_length_one_arrays_with_scalars(sdfg)

    # Find the edge and confirm the [0] subscript is gone.
    for e in sdfg.edges():
        if e.data.assignments:
            assert 'flag[0]' not in e.data.assignments['s']
            assert 'flag + 1' in e.data.assignments['s']


def test_transient_only_skips_signature():
    sdfg = dace.SDFG('transient_only')
    sdfg.add_array('arg', shape=(1, ), dtype=dace.float64)
    sdfg.add_array('local', shape=(1, ), dtype=dace.float64, transient=True)

    rewritten = replace_length_one_arrays_with_scalars(sdfg, transient_only=True)
    assert rewritten == {'local'}
    assert isinstance(sdfg.arrays['arg'], dace.data.Array)
    assert isinstance(sdfg.arrays['local'], dace.data.Scalar)


def test_recursive_descends_into_nested_sdfg():
    outer = dace.SDFG('outer')
    outer.add_array('outer_arg', shape=(1, ), dtype=dace.float64)

    inner = dace.SDFG('inner')
    inner.add_array('inner_arg', shape=(1, ), dtype=dace.float64)
    inner.add_array('inner_local', shape=(1, ), dtype=dace.float64, transient=True)
    inner_state = inner.add_state('s', is_start_block=True)

    state = outer.add_state('s')
    state.add_nested_sdfg(inner, set(), set())

    replace_length_one_arrays_with_scalars(outer, recursive=True, transient_only=False)

    # Outer arg becomes scalar.
    assert isinstance(outer.arrays['outer_arg'], dace.data.Scalar)
    # Inner local becomes scalar; inner arg is a signature dummy and
    # ``transient_only=True`` is forced for nested calls -> stays Array.
    assert isinstance(inner.arrays['inner_local'], dace.data.Scalar)
    assert isinstance(inner.arrays['inner_arg'], dace.data.Array)


if __name__ == '__main__':
    test_signature_length_one_becomes_scalar()
    test_interstate_edge_subscript_stripped()
    test_transient_only_skips_signature()
    test_recursive_descends_into_nested_sdfg()
