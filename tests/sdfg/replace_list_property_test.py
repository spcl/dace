# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that replacing symbols in list properties only rewrites the lists that actually hold
names or expressions. Lists are not uniformly symbolic -- they also hold booleans, C declarations,
opaque identifiers and plain objects -- and round-tripping those through sympy corrupts them. """

import numpy as np
import sympy as sp

import dace
from dace.sdfg.replace import _replace_list_item


def _repl():
    """ ``{i -> j, N -> M}`` as the name and symbolic dictionaries ``replace()`` passes down. """
    return ({'i': 'j', 'N': 'M'}, {sp.Symbol('i'): sp.Symbol('j'), sp.Symbol('N'): sp.Symbol('M')})


def test_bools_in_an_int_list_survive():
    """``ProcessGrid.color`` is a ``ListProperty(int)`` that actually holds booleans (``Sub([False,
    True])``). ``pystr_to_symbolic('False')`` is ``BooleanFalse``, which ``int()`` then rejects."""
    repl, symrepl = _repl()
    for value in (False, True):
        result = _replace_list_item(value, int, repl, symrepl)
        assert result is value, result
        assert isinstance(result, bool), type(result)


def test_code_fragments_in_a_string_list_survive():
    """``Dummy.fields`` and ``CodeNode.state_fields`` hold C declarations, which do not sympify."""
    repl, symrepl = _repl()
    field = 'MPI_Comm __pgrid_0;'
    assert _replace_list_item(field, str, repl, symrepl) == field


def test_tuples_in_an_object_list_survive():
    """``LogicalGroup.nodes`` holds tuples, and ``'(0, 1)'`` sympifies to the *list* ``[0, 1]``."""
    repl, symrepl = _repl()
    node = (0, 1)
    result = _replace_list_item(node, tuple, repl, symrepl)
    assert result == (0, 1)
    assert isinstance(result, tuple), type(result)


def test_names_in_a_string_list_are_replaced():
    """The case the list-property replacement exists for: renaming ``Map.params``."""
    repl, symrepl = _repl()
    assert _replace_list_item('i', str, repl, symrepl) == 'j'
    assert _replace_list_item('k', str, repl, symrepl) == 'k'


def test_non_identifier_replacements_are_skipped():
    """A map parameter cannot be named by an expression, so such a replacement is not written."""
    repl = {'i': 'k + 1'}
    symrepl = {sp.Symbol('i'): sp.Symbol('k') + 1}
    assert _replace_list_item('i', str, repl, symrepl) == 'i'


def test_symbolic_lists_are_replaced():
    """``Array.offset`` is a ``ListProperty(sp.Basic)`` and must still be substituted into."""
    repl, symrepl = _repl()
    assert _replace_list_item(sp.Symbol('N'), sp.Basic, repl, symrepl) == sp.Symbol('M')


def test_map_params_are_renamed_end_to_end():
    """The same thing through the public API, so the dispatch is exercised as ``replace()`` uses it."""
    sdfg = dace.SDFG('list_property_replace')
    sdfg.add_array('out', [8], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    entry, exit_node = state.add_map('m', {'i': '0:8'})
    tasklet = state.add_tasklet('w', {}, {'o'}, 'o = 1.0')
    state.add_edge(entry, None, tasklet, None, dace.Memlet())
    exit_node.add_in_connector('IN_o')
    exit_node.add_out_connector('OUT_o')
    state.add_edge(tasklet, 'o', exit_node, 'IN_o', dace.Memlet('out[i]'))
    state.add_edge(exit_node, 'OUT_o', state.add_write('out'), None, dace.Memlet('out[0:8]'))
    sdfg.validate()

    state.replace_dict({'i': 'k'})
    assert entry.map.params == ['k'], entry.map.params
    sdfg.validate()

    result = np.zeros(8, dtype=np.float64)
    sdfg(out=result)
    assert np.allclose(result, 1.0)


if __name__ == '__main__':
    test_bools_in_an_int_list_survive()
    test_code_fragments_in_a_string_list_survive()
    test_tuples_in_an_object_list_survive()
    test_names_in_a_string_list_are_replaced()
    test_non_identifier_replacements_are_skipped()
    test_symbolic_lists_are_replaced()
    test_map_params_are_renamed_end_to_end()
