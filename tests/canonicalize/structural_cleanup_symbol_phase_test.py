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
from dace.transformation.passes.canonicalize.symbol_dedup import SymbolDedup
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace.transformation.passes.pattern_matching import PatternApplyOnceEverywhere, PatternMatchAndApplyRepeated
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols
from dace.transformation.passes.symbol_propagation import SymbolPropagation

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


def test_symbol_phase_precedes_the_structural_phase():
    """Symbols are folded before the state machine is rewritten, and the phase closes on a dedup.

    The second ``SymbolDedup`` closes the phase, and it must come AFTER the prune: propagation and
    constant folding rewrite the assignments the first dedup merged, and ``RemoveUnusedSymbols``
    then deletes assignments, which can make two previously-different edge sets identical. Order
    it before the prune and 7 corpus kernels keep a mergeable pair, ``scatter_accum_dup`` among
    them. Leaving one behind is what makes a syntactic same-slot test answer "different slots" --
    the defect that cost ``scatter_accum_dup`` its WCR and left it aborting on duplicate indices.
    """
    members = [type(unit) for _label, unit in _structural_cleanup('t')]
    symbol_phase = [SymbolDedup, SymbolPropagation, ConstantPropagation, RemoveUnusedSymbols, SymbolDedup]
    assert members[:len(symbol_phase)] == symbol_phase
    assert members.count(SymbolDedup) == 2
    assert members.index(RemoveUnusedSymbols) < len(symbol_phase) - 1, 'the prune must precede the closing dedup'


def test_state_fusion_applies_once_not_to_a_fixpoint():
    """Cheap per boundary, repeated often -- not a fixpoint at each of ~15 boundaries."""
    fusions = [unit for _label, unit in _structural_cleanup('t') if isinstance(unit, PatternMatchAndApplyRepeated)]
    assert len(fusions) == 1
    assert isinstance(fusions[0], PatternApplyOnceEverywhere)
