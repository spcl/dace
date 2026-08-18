# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that name minting consults every namespace without materializing them. """

import dace
from dace.sdfg.sdfg import _UsedNames
from dace.utils import find_new_name


def test_find_new_name_only_tests_membership():
    """``find_new_name`` must never iterate its argument, so a view can answer ``in`` lazily."""

    class ContainerOnly:

        def __init__(self, taken):
            self.taken = taken
            self.probes = 0

        def __contains__(self, name):
            self.probes += 1
            return name in self.taken

        def __iter__(self):
            raise AssertionError('find_new_name iterated its argument')

    names = ContainerOnly({'A', 'A_0'})
    assert find_new_name('A', names) == 'A_1'
    assert names.probes == 3
    assert find_new_name('B', names) == 'B'


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
    test_find_new_name_only_tests_membership()
    test_used_names_view_covers_arrays_symbols_and_constants()
    test_minted_names_avoid_every_namespace()
