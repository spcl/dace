# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the validation of nested SDFG connector descriptors against the containers they are connected to. """
from typing import Dict, List, Sequence

import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.validation import InvalidSDFGNodeError


def _nested_write(
    outer_shape: List[str],
    inner_shape: List[str],
    symbol_mapping: Dict[str, str],
    inner_symbols: Sequence[str] = ('N', ),
    outer_symbols: Sequence[str] = ('M', )
) -> dace.SDFG:
    """
    Creates an SDFG that writes to ``A`` through a nested SDFG connector ``a``, without integrating the nested SDFG.

    :param outer_shape: The shape of ``A`` in the parent SDFG.
    :param inner_shape: The shape of ``a`` in the nested SDFG.
    :param symbol_mapping: The symbol mapping of the nested SDFG node.
    :param inner_symbols: The symbols declared in the nested SDFG.
    :param outer_symbols: The symbols declared in the parent SDFG.
    :return: The parent SDFG.
    """
    inner = dace.SDFG('inner')
    for s in inner_symbols:
        inner.add_symbol(s, dace.int64)
    inner.add_array('a', inner_shape, dace.float64)
    istate = inner.add_state()
    t = istate.add_tasklet('t', {}, {'o'}, 'o = 1')
    istate.add_edge(t, 'o', istate.add_write('a'), None, dace.Memlet('a[' + ', '.join(['0'] * len(inner_shape)) + ']'))

    outer = dace.SDFG('outer')
    for s in outer_symbols:
        outer.add_symbol(s, dace.int64)
    outer.add_array('A', outer_shape, dace.float64)
    state = outer.add_state()
    node = state.add_nested_sdfg(inner, {}, {'a'}, symbol_mapping)
    state.add_edge(node, 'a', state.add_write('A'), None, dace.Memlet.from_array('A', outer.arrays['A']))
    return outer


def _nested_node(sdfg: dace.SDFG) -> nodes.NestedSDFG:
    return next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))


def test_mapped_expression():
    sdfg = _nested_write(['M + 1'], ['N'], {'N': 'M + 1'})
    sdfg.validate()


def test_mapped_expression_in_inner_shape():
    sdfg = _nested_write(['M + K - 1'], ['N + 1'], {'N': 'M + K - 2'}, outer_symbols=('M', 'K'))
    sdfg.validate()


def test_mapped_expression_multidimensional():
    # The strides of both descriptors are derived from their shapes, and must match after mapping as well
    sdfg = _nested_write(['M + 1', 'K - 1'], ['N', 'P'], {
        'N': 'M + 1',
        'P': 'K - 1'
    },
                         inner_symbols=('N', 'P'),
                         outer_symbols=('M', 'K'))
    sdfg.validate()


def test_mapped_expression_mismatch():
    sdfg = _nested_write(['M + 1'], ['N'], {'N': 'M + 2'})
    with pytest.raises(InvalidSDFGNodeError, match='not equivalent'):
        sdfg.validate()


def test_mapped_strides_mismatch():
    sdfg = _nested_write(['M', 'K'], ['N', 'P'], {
        'N': 'M',
        'P': 'K'
    },
                         inner_symbols=('N', 'P'),
                         outer_symbols=('M', 'K'))
    sdfg.arrays['A'].strides = (1, sdfg.arrays['A'].shape[0])
    with pytest.raises(InvalidSDFGNodeError, match='not equivalent'):
        sdfg.validate()


def test_swapped_symbol_names():
    sdfg = _nested_write(['M', 'N'], ['N', 'M'], {
        'N': 'M',
        'M': 'N'
    },
                         inner_symbols=('N', 'M'),
                         outer_symbols=('M', 'N'))
    sdfg.validate()


def test_swapped_symbol_names_written_alike():
    # The descriptors read the same, but the inner ``N`` is the outer ``M`` and vice versa
    sdfg = _nested_write(['N', 'M'], ['N', 'M'], {
        'N': 'M',
        'M': 'N'
    },
                         inner_symbols=('N', 'M'),
                         outer_symbols=('M', 'N'))
    with pytest.raises(InvalidSDFGNodeError, match='not equivalent'):
        sdfg.validate()


def test_shadowed_symbol_written_alike():
    # The inner ``N`` is bound to the outer ``N + 1``, so ``a[N]`` does not describe ``A[N]``
    sdfg = _nested_write(['N'], ['N'], {'N': 'N + 1'}, outer_symbols=('N', ))
    with pytest.raises(InvalidSDFGNodeError, match='not equivalent'):
        sdfg.validate()


def test_undeclared_inner_symbol():
    # The inner descriptor reads the same as the outer one, but its symbol is not a symbol of the nested SDFG,
    # so there is nothing to compare it with
    sdfg = _nested_write(['Q'], ['Q'], {}, inner_symbols=(), outer_symbols=('Q', ))
    node = _nested_node(sdfg)
    node.sdfg.remove_symbol('Q')
    node.symbol_mapping.pop('Q', None)
    with pytest.raises(InvalidSDFGNodeError, match='Q'):
        sdfg.validate()


def test_unmapped_inner_symbol():
    # The inner ``N`` is a symbol of the nested SDFG, but nothing outside gives it a value
    sdfg = _nested_write(['N'], ['N'], {}, outer_symbols=('N', ))
    _nested_node(sdfg).symbol_mapping.pop('N', None)
    with pytest.raises(InvalidSDFGNodeError, match='symbol mapping'):
        sdfg.validate()


def test_mapped_symbol_not_declared():
    # ``N`` is mapped and used by a transient, but is not a symbol of the nested SDFG, so it has no type inside
    sdfg = _nested_write(['M'], ['M'], {'M': 'M', 'N': 'M + 1'}, inner_symbols=('M', 'N'))
    node = _nested_node(sdfg)
    node.sdfg.add_transient('tmp', ['N'], dace.float64)
    istate = node.sdfg.start_block
    t = istate.add_tasklet('w', {}, {'o'}, 'o = 2')
    istate.add_edge(t, 'o', istate.add_write('tmp'), None, dace.Memlet('tmp[0]'))
    node.sdfg.remove_symbol('N')
    node.symbol_mapping['N'] = 'M + 1'  # Removing the symbol also unmaps it
    with pytest.raises(InvalidSDFGNodeError, match='not declared'):
        sdfg.validate()


def test_mapped_expression_after_integration():
    sdfg = _nested_write(['M + 1'], ['N'], {'N': 'M + 1'})
    _nested_node(sdfg).integrate_into_parent()
    sdfg.validate()


if __name__ == '__main__':
    test_mapped_expression()
    test_mapped_expression_in_inner_shape()
    test_mapped_expression_multidimensional()
    test_mapped_expression_mismatch()
    test_mapped_strides_mismatch()
    test_swapped_symbol_names()
    test_swapped_symbol_names_written_alike()
    test_shadowed_symbol_written_alike()
    test_undeclared_inner_symbol()
    test_unmapped_inner_symbol()
    test_mapped_symbol_not_declared()
    test_mapped_expression_after_integration()
