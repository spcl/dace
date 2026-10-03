# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Contract for the symbol phase of :func:`_structural_cleanup`, which runs at EVERY stage boundary.

Running a prune ~15 times earlier than it used to run has one specific way to go wrong: a symbol
that is declared and live but referenced only somewhere the "is it used" scan does not look. The
reserved ``__dace_num_threads`` is exactly that shape -- frame code declares it and
``chunk_anti_dependence`` sizes its seam transient ``__dace_num_threads + 1``, so the only
reference is an array SHAPE. Prune it and the emitted C++ fails to compile on an undeclared
identifier, which no SDFG-level assertion would catch.

The order is pinned too. Symbols are folded before the state machine is rewritten, and
``StateFusionExtended`` applies once everywhere rather than to a fixpoint -- the helper is meant
to be cheap per boundary and run often, not to converge at each of ~15 boundaries.
"""
import dace
from dace.transformation.passes.canonicalize.pipeline import _structural_cleanup
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols

#: The reserved symbol whose only reference is a transient's shape.
NUM_THREADS = '__dace_num_threads'


def _shape_only_symbol_sdfg() -> dace.SDFG:
    """An SDFG whose ``__dace_num_threads`` is referenced by nothing but a transient's shape."""
    sdfg = dace.SDFG('shape_only_symbol')
    sdfg.add_symbol(NUM_THREADS, dace.int32)
    sdfg.add_array('a', [16], dace.float64)
    sdfg.add_transient('seam', [dace.symbol(NUM_THREADS, dtype=dace.int32) + 1], dace.float64)
    state = sdfg.add_state('s')
    tasklet = state.add_tasklet('copy', {'i'}, {'o'}, 'o = i')
    state.add_edge(state.add_read('a'), None, tasklet, 'i', dace.Memlet('a[0]'))
    state.add_edge(tasklet, 'o', state.add_write('a'), None, dace.Memlet('a[1]'))
    return sdfg


def test_shape_only_symbol_counts_as_used():
    """``RemoveUnusedSymbols`` must look at descriptor shapes, not only code and edges."""
    sdfg = _shape_only_symbol_sdfg()
    assert NUM_THREADS in RemoveUnusedSymbols().used_symbols(sdfg)


def test_shape_only_symbol_survives_the_prune():
    """The failure this guards is a C++ compile error, so nothing at SDFG level would catch it."""
    sdfg = _shape_only_symbol_sdfg()
    assert RemoveUnusedSymbols().apply_pass(sdfg, {}) is None
    assert NUM_THREADS in sdfg.symbols
    assert str(sdfg.arrays['seam'].shape[0]) == f'{NUM_THREADS} + 1'


def test_shape_only_symbol_survives_the_whole_cleanup():
    """The prune does not run alone -- the rest of the boundary must not strip it either."""
    sdfg = _shape_only_symbol_sdfg()
    for _label, unit in _structural_cleanup('t'):
        unit.apply_pass(sdfg, {})
    assert NUM_THREADS in sdfg.symbols
