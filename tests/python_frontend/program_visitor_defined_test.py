# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the name resolution table ``ProgramVisitor.defined`` hands to memlet and symbol parsing. """
import dace
from dace import data
from dace.frontend.python.newast import ProgramVisitor


def _visitor(name: str) -> ProgramVisitor:
    return ProgramVisitor(name=name,
                          filename=__file__,
                          line_offset=0,
                          col_offset=0,
                          global_vars={},
                          constants={},
                          scope_arrays={},
                          scope_vars={})


def test_defined_resolves_arrays_symbols_and_process_grids():
    pv = _visitor('defined_table')
    sdfg = pv.sdfg
    sdfg.add_array('A', (20, ), dace.float64)
    sdfg.add_symbol('N', dace.int32)
    pgrid = sdfg.add_pgrid(shape=[2, 2])

    # The visitor's own variable names, as an assignment in the parsed program would bind them.
    pv.variables['a'] = 'A'
    pv.variables['n'] = 'N'
    pv.variables['grid'] = pgrid

    defined = pv.defined

    assert defined['a'] is sdfg.arrays['A']
    assert defined['n'] is sdfg.symbols['N']
    # Keyed by the DESCRIPTOR name, not the program-side name: that is what the process-grid entry
    # has always promised its consumers, and it is the entry a lookup of the grid resolves through.
    assert defined[pgrid] is sdfg.process_grids[pgrid]
    assert isinstance(defined[pgrid], data.distributed.ProcessGrid)


def test_defined_resolves_a_process_grid_under_every_name_bound_to_it():
    pv = _visitor('defined_two_names_one_grid')
    pgrid = pv.sdfg.add_pgrid(shape=[2, 2])
    pv.variables['grid'] = pgrid
    pv.variables['alias'] = pgrid

    # Every binding resolves to the SAME descriptor, and the descriptor name resolves whether or not
    # a variable is bound to it (`add_pgrid` files the grid in the SDFG's descriptor repository).
    defined = pv.defined
    assert defined[pgrid] is pv.sdfg.process_grids[pgrid]
    assert isinstance(defined[pgrid], data.distributed.ProcessGrid)


if __name__ == '__main__':
    test_defined_resolves_arrays_symbols_and_process_grids()
    test_defined_resolves_a_process_grid_under_every_name_bound_to_it()
