# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An interstate assignment that READS an array is data flow, and must reach the tiler as data.

The frontend routes an array element into a body through an interstate assignment
(``b_index = b[i]``), so the consuming state reads a bare symbol that names no map parameter yet
holds a different value per lane. ``ConvertTaskletsToTileOps`` classifies a symbol operand as
tile-invariant, and rather than splat lane 0 across the tile it refuses -- for the WHOLE SDFG, not
the one kernel. CloudSC hit exactly that on
``za_slice_plus_zsolac_slice = zsolac[jl - 1] + za[jk - 1, jl - 1]``: 974 maps were classified as
tileable and none of them was tiled.

``DemoteDataReadingInterstateSymbols`` gives the read back its data flow before anything classifies
operands. The tests below pin the two halves that make it safe: the data-reading symbol IS demoted,
and a symbol carrying graph structure is NOT.
"""
import dace
from dace import nodes
from dace.transformation.passes.vectorization.demote_data_reading_interstate_symbols import (
    DemoteDataReadingInterstateSymbols, data_reading_assigned_symbols)

N = 16


def indexed_read_sdfg() -> dace.SDFG:
    """``elem = b[3]`` on an interstate edge, then a state that uses ``elem`` in a tasklet."""
    sdfg = dace.SDFG('indexed_interstate_read')
    sdfg.add_array('b', (N, ), dace.float64)
    sdfg.add_array('out', (N, ), dace.float64)
    sdfg.add_symbol('elem', dace.float64)

    entry = sdfg.add_state('entry', is_start_block=True)
    use = sdfg.add_state('use')
    sdfg.add_edge(entry, use, dace.InterstateEdge(assignments={'elem': 'b[3]'}))

    tasklet = use.add_tasklet('scale', {'_in'}, {'_out'}, '_out = _in * elem')
    use.add_edge(use.add_access('b'), None, tasklet, '_in', dace.Memlet('b[0]'))
    use.add_edge(tasklet, '_out', use.add_access('out'), None, dace.Memlet('out[0]'))
    return sdfg


def loop_bound_sdfg() -> dace.SDFG:
    """The control: ``n`` is assigned a plain expression and BOUNDS a loop -- structure, not data."""
    sdfg = dace.SDFG('loop_bound_symbol')
    sdfg.add_array('out', (N, ), dace.float64)
    sdfg.add_symbol('n', dace.int64)

    entry = sdfg.add_state('entry', is_start_block=True)
    loop = dace.sdfg.state.LoopRegion('walk', 'i < n', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(entry, loop, dace.InterstateEdge(assignments={'n': '4'}))
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('one', {}, {'_out'}, '_out = 1.0')
    body.add_edge(tasklet, '_out', body.add_access('out'), None, dace.Memlet('out[i]'))
    return sdfg


def test_an_indexed_read_is_recognised_as_one():
    assert 'elem' in data_reading_assigned_symbols(indexed_read_sdfg())


def test_a_plain_expression_is_not_a_read():
    assert 'n' not in data_reading_assigned_symbols(loop_bound_sdfg())


def test_the_read_becomes_a_scalar_and_a_tasklet():
    """The symbol is gone, a Register scalar stands in its place, and the read is a memlet."""
    sdfg = indexed_read_sdfg()
    assert DemoteDataReadingInterstateSymbols().apply_pass(sdfg, {}) == 1

    assert 'elem' not in sdfg.symbols, 'the demoted name is still a symbol'
    assert 'elem' in sdfg.arrays, 'no scalar took the symbol\'s place'
    assert sdfg.arrays['elem'].storage == dace.dtypes.StorageType.Register
    assert sdfg.arrays['elem'].dtype == dace.float64, 'the scalar must keep the symbol\'s own dtype'

    # The assignment is gone from every edge, and some state now reads ``b`` into ``elem``.
    assert not any('elem' in e.data.assignments for e in sdfg.all_interstate_edges())
    writes_elem = [
        state.label for state in sdfg.all_states() for n in state.nodes()
        if isinstance(n, nodes.AccessNode) and n.data == 'elem' and state.in_degree(n) > 0
    ]
    assert writes_elem, 'nothing writes the scalar the assignment was demoted into'
    sdfg.validate()


def test_a_structural_symbol_is_left_alone():
    """A loop bound stays a symbol: demoting it would rewrite control flow, not a read."""
    sdfg = loop_bound_sdfg()
    assert DemoteDataReadingInterstateSymbols().apply_pass(sdfg, {}) is None
    assert 'n' in sdfg.symbols
    sdfg.validate()


def test_the_demoted_read_still_computes_the_same_value():
    """Executable check: ``out[0] = b[0] * b[3]`` before and after the demotion."""
    import numpy as np

    def run(sdfg: dace.SDFG) -> np.ndarray:
        b = np.arange(1, N + 1, dtype=np.float64)
        out = np.zeros(N, dtype=np.float64)
        sdfg(b=b, out=out)
        return out

    before = run(indexed_read_sdfg())
    demoted = indexed_read_sdfg()
    DemoteDataReadingInterstateSymbols().apply_pass(demoted, {})
    demoted.name = 'indexed_interstate_read_demoted'
    assert np.allclose(run(demoted), before)
    assert before[0] == 1.0 * 4.0, 'the fixture stopped computing b[0] * b[3]'
