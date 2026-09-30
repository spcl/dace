# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" A symbol declared on the SDFG keeps its declared dtype in ``defined_symbols`` over a default-typed array extent. """
import dace


def test_a_declared_symbol_wins_over_the_default_typed_extent_of_an_array():
    sdfg = dace.SDFG('declared_extent')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [dace.symbol('N')], dace.float32)
    state = sdfg.add_state()
    assert state.defined_symbols()['N'] == dace.int64


if __name__ == '__main__':
    test_a_declared_symbol_wins_over_the_default_typed_extent_of_an_array()
