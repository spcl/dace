# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that the frontend's scope symbol table always answers what a fresh one would.

``ProgramVisitor.scope_symbol_table`` hands ``add_mapped_tasklet`` a table instead of letting it
walk every descriptor again, and folds that table forward over the descriptors added since it was
built. Each case below grows the SDFG the way one frontend statement does and demands the table
equal ``sdfg_scope_symbols`` right after -- a table the visitor reused where it should have rebuilt
answers with a missing or wrongly typed name, and memlet propagation then widens on it.
"""

import dace
from dace.frontend.python.newast import ProgramVisitor
from dace.sdfg.state import sdfg_scope_symbols


def visitor_with_an_array() -> ProgramVisitor:
    pv = ProgramVisitor('table', 'scope_symbol_table_test.py', 0, 0, {}, {}, {}, {})
    pv.sdfg.add_symbol('N', dace.int64)
    pv.sdfg.add_array('a', [dace.symbol('N', dace.int64)], dace.float64)
    return pv


def assert_table_is_fresh(pv: ProgramVisitor) -> None:
    assert dict(pv.scope_symbol_table(pv.sdfg)) == dict(sdfg_scope_symbols(pv.sdfg))


def test_table_equals_a_fresh_one_when_nothing_changed():
    pv = visitor_with_an_array()
    assert_table_is_fresh(pv)
    assert_table_is_fresh(pv)


def test_a_descriptor_naming_a_new_symbol_is_folded_in():
    """Written into ``arrays`` rather than through ``add_datadesc``, which declares the symbol as
    well -- the same door ``ProgramVisitor.repl_callback`` renames a descriptor through."""
    pv = visitor_with_an_array()
    assert 'M' not in pv.scope_symbol_table(pv.sdfg)
    pv.sdfg.arrays['t'] = dace.data.Array(dace.float64, [dace.symbol('M', dace.int64)], transient=True)

    assert 'M' in pv.scope_symbol_table(pv.sdfg)
    assert_table_is_fresh(pv)


def test_a_declared_symbol_arriving_rebuilds_the_table():
    pv = visitor_with_an_array()
    assert_table_is_fresh(pv)
    pv.sdfg.add_symbol('K', dace.int32)

    assert pv.scope_symbol_table(pv.sdfg)['K'] == dace.int32
    assert_table_is_fresh(pv)


def test_a_redeclared_symbol_rebuilds_the_table():
    """A declaration is folded FIRST, so a changed dtype cannot be folded forward."""
    pv = visitor_with_an_array()
    assert pv.scope_symbol_table(pv.sdfg)['N'] == dace.int64
    pv.sdfg.remove_symbol('N')
    pv.sdfg.add_symbol('N', dace.int32)

    assert_table_is_fresh(pv)


def test_a_removed_descriptor_rebuilds_the_table():
    pv = visitor_with_an_array()
    pv.sdfg.arrays['t'] = dace.data.Array(dace.float64, [dace.symbol('M', dace.int64)], transient=True)
    assert 'M' in pv.scope_symbol_table(pv.sdfg)
    del pv.sdfg.arrays['t']

    assert 'M' not in pv.scope_symbol_table(pv.sdfg)
    assert_table_is_fresh(pv)


def test_an_edge_into_the_start_block_puts_the_table_back_on_the_full_path():
    """``sdfg_scope_symbols`` folds interstate assignments LAST, so one reaching the start block
    cannot be folded forward either."""
    pv = visitor_with_an_array()
    entry = pv.sdfg.start_block
    assert_table_is_fresh(pv)
    latch = pv.sdfg.add_state('latch')
    pv.sdfg.add_edge(entry, latch, dace.InterstateEdge())
    pv.sdfg.add_edge(latch, entry, dace.InterstateEdge(assignments={'bound': 'N + 1'}))

    assert 'bound' in pv.scope_symbol_table(pv.sdfg)
    assert_table_is_fresh(pv)


def test_an_sdfg_the_visitor_does_not_own_gets_no_table():
    assert visitor_with_an_array().scope_symbol_table(dace.SDFG('elsewhere')) is None


if __name__ == '__main__':
    test_table_equals_a_fresh_one_when_nothing_changed()
    test_a_descriptor_naming_a_new_symbol_is_folded_in()
    test_a_declared_symbol_arriving_rebuilds_the_table()
    test_a_redeclared_symbol_rebuilds_the_table()
    test_a_removed_descriptor_rebuilds_the_table()
    test_an_edge_into_the_start_block_puts_the_table_back_on_the_full_path()
    test_an_sdfg_the_visitor_does_not_own_gets_no_table()
