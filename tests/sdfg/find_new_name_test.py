# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that name minting consults every namespace without materializing them.

Both cases below were kept because a mutation proved the rest of the suite misses them: a view
that answers ``in`` from arrays and symbols but FORGETS constants leaves 787 sdfg + frontend
tests passing, and mints a name straight on top of an existing constant.
"""

import dace
from dace.sdfg.sdfg import _UsedNames


def test_used_names_view_covers_arrays_symbols_and_constants():
    """The view reports a name from any of the three namespaces, and nothing else."""
    sdfg = dace.SDFG('used_names')
    sdfg.add_array('an_array', [1], dace.float64)
    sdfg.add_symbol('a_symbol', dace.int64)
    sdfg.add_constant('a_constant', 1)

    view = _UsedNames(sdfg)
    assert 'an_array' in view
    assert 'a_symbol' in view
    assert 'a_constant' in view
    assert 'unused' not in view


def test_minted_names_avoid_every_namespace():
    """A mint collides with an array, a symbol or a constant alike -- the union's whole point."""
    sdfg = dace.SDFG('mint')
    sdfg.add_array('x', [1], dace.float64)
    sdfg.add_symbol('y', dace.int64)
    sdfg.add_constant('z', 1)

    assert sdfg._find_new_name('x') == 'x_0'
    assert sdfg._find_new_name('y') == 'y_0'
    assert sdfg._find_new_name('z') == 'z_0'
    assert sdfg._find_new_name('free') == 'free'

    # A transient minted with find_new_name=True lands beside, not on top of, the existing one.
    second, _ = sdfg.add_transient('x', [1], dace.float64, find_new_name=True)
    assert second != 'x' and second in sdfg.arrays and 'x' in sdfg.arrays


if __name__ == '__main__':
    test_used_names_view_covers_arrays_symbols_and_constants()
    test_minted_names_avoid_every_namespace()
