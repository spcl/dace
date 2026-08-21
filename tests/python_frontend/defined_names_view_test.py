# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ProgramVisitor.defined`` is a view; it must answer exactly what the merged dict holds.

The view resolves a name by walking its sources in reverse precedence and stopping at the first
hit, while ``materialize()`` merges them all and lets the last write win. Those two are only equal
if the walk order is the exact reverse of the merge order, and nothing but a test says so -- the
frontend consults ``defined`` on nearly every name it touches, so a single inverted pair would
resolve some names to the wrong descriptor and lower a silently different program.
"""
import pytest

import dace
from dace.frontend.python.newast import ProgramVisitor


def make_visitor() -> ProgramVisitor:
    """A visitor whose tables exercise every source ``defined`` merges, including the overlaps."""
    sdfg = dace.SDFG('defined_names_view')
    sdfg.add_array('A', [12], dace.float64)
    sdfg.add_scalar('s', dace.float64)
    sdfg.add_array('shadowed', [4], dace.float64)
    sdfg.add_symbol('N', dace.int64)

    scope_arrays = {'shadowed': dace.data.Array(dace.float32, [7]), 'outer': dace.data.Array(dace.float32, [3])}

    visitor = ProgramVisitor(name='defined_names_view',
                             filename='<test>',
                             line_offset=0,
                             col_offset=0,
                             global_vars={
                                 'g_sym': dace.symbol('g_sym'),
                                 'g_plain': 5
                             },
                             constants={},
                             scope_arrays=scope_arrays,
                             scope_vars={
                                 'from_scope': 'outer',
                                 'both_tables': 'shadowed',
                                 'to_sdfg_array': 'A',
                                 'contested': 'outer',
                             })
    visitor.sdfg = sdfg
    # ``contested`` is deliberately in BOTH tables, resolving to a DIFFERENT descriptor in each:
    # that is the only shape that pins which table wins, and without it an inverted walk order
    # passes every assertion here (verified by mutation).
    visitor.variables = {'var_to_array': 'A', 'var_to_symbol': 'N', 'A': 's', 'contested': 'A'}
    return visitor


def test_the_view_and_the_merged_dict_agree_entry_for_entry():
    """Keys, values and membership, on tables where four of the sources overlap."""
    visitor = make_visitor()
    view = visitor.defined
    merged = view.materialize()

    assert set(view) == set(merged)
    assert len(view) == len(merged)
    for key, value in merged.items():
        assert key in view, f'{key} is in the merged dict and missing from the view'
        assert view[key] is value, f'{key} resolves to a different descriptor through the view'

    # `g_plain` is a closure global that is not a symbol, so it is not a defined NAME.
    for absent in ('nothing_defined_by_this_name', 'g_plain'):
        assert absent not in view
        assert absent not in merged
        with pytest.raises(KeyError):
            view[absent]

    # The visitor seeds `scope_vars` from the enclosing scope's arrays, so those names resolve too
    # -- asserted here rather than assumed, since it is why the two tables overlap at all.
    assert 'outer' in view
    assert view['outer'] is view.pv.scope_arrays['outer']


def test_the_view_resolves_each_source_the_way_the_merge_order_says():
    """The precedence itself, spelled out -- an equivalence test alone would pass on two
    implementations that are consistently WRONG in the same direction."""
    view = make_visitor().defined
    sdfg = view.pv.sdfg

    # An SDFG array beats a variable of the same name: `arrays` is merged after `variables`, so
    # `A` is the array A, never the scalar `s` that `variables` maps it to.
    assert view['A'] is sdfg.arrays['A']
    # A variable resolves through its SDFG-level name.
    assert view['var_to_array'] is sdfg.arrays['A']
    assert view['var_to_symbol'] is sdfg.symbols['N']
    # A scope variable resolves against the ENCLOSING scope's descriptors...
    assert view['from_scope'] is view.pv.scope_arrays['outer']
    assert view['both_tables'] is view.pv.scope_arrays['shadowed']
    # ...but a name in BOTH tables resolves through ``variables``, which the merge applies LAST.
    # This is the assertion that fails if the walk order is inverted.
    assert view['contested'] is sdfg.arrays['A']
    assert view['to_sdfg_array'] is sdfg.arrays['A']
    # A closure symbol is the lowest-precedence source, and a non-symbol global is not defined.
    assert view['g_sym'] is view.pv.globals['g_sym']
    assert 'g_plain' not in view


if __name__ == '__main__':
    test_the_view_and_the_merged_dict_agree_entry_for_entry()
    test_the_view_resolves_each_source_the_way_the_merge_order_says()
