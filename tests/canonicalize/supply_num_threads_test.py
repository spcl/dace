# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``__dace_num_threads``: declared by the pass, defined by frame code, kept out of the ABI.

The thread count has to be readable where a buffer is SIZED, but a caller cannot be asked for a
machine property. An allocation is emitted at program entry, ahead of every state, so nothing that
runs as part of the graph can define the symbol in time -- frame code does, at the top of the
program function. These pin that contract (declared once, defined before the first allocation, only
when something reads it, never in the signature) plus the hazards that make it subtle: the value
must be read where no parallel region is open, and a symbol the parent maps but the child never
declares fails validation.
"""
import numpy as np
import pytest

import dace
from dace import dtypes, symbolic
from dace.codegen.targets import framecode
from dace.transformation.passes.canonicalize.supply_num_threads import DTYPE, SupplyNumThreads, ensure_in_scope

N = dace.symbol('N', dtype=dace.int64)
NT = symbolic.pystr_to_symbolic(symbolic.NUM_THREADS_SYMBOL)


@dace.program
def scale(a: dace.float64[N], out: dace.float64[N]):
    out[:] = a * 2.0


def supplied() -> dace.SDFG:
    sdfg = scale.to_sdfg(simplify=False)
    SupplyNumThreads().apply_pass(sdfg, {})
    return sdfg


def per_thread_sdfg(name: str) -> dace.SDFG:
    """A transient whose SIZE names the symbol -- the use that has no other spelling, because it is
    read by the allocation rather than by an edge, a memlet or a tasklet."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_transient('buf', [NT], dace.float64, lifetime=dtypes.AllocationLifetime.State)
    first = sdfg.add_state()
    writer = first.add_tasklet('w', {}, {'o'}, 'o = 1.0')
    first.add_edge(writer, 'o', first.add_access('buf'), None, dace.Memlet('buf[0]'))
    second = sdfg.add_state_after(first)
    reader = second.add_tasklet('r', {'i'}, {'o'}, 'o = i')
    second.add_edge(second.add_access('buf'), None, reader, 'i', dace.Memlet('buf[0]'))
    second.add_edge(reader, 'o', second.add_access('out'), None, dace.Memlet('out[0]'))
    SupplyNumThreads().apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def generated(sdfg: dace.SDFG) -> str:
    return '\n'.join(block.clean_code for block in sdfg.generate_code())


def test_the_pass_declares_the_symbol_at_int64():
    """int64, not int32: it divides int64 extents, and a narrower type makes the emitted
    ``int_ceil`` deduce a mixed-width result a loop counter cannot take."""
    sdfg = supplied()
    assert sdfg.symbols[symbolic.NUM_THREADS_SYMBOL] == DTYPE
    assert DTYPE == dtypes.int64


def test_the_pass_leaves_no_probe_state_behind():
    """The definition is frame code's. A graph-side probe would be a SECOND mechanism defining one
    symbol, and the two could disagree."""
    sdfg = supplied()
    assert not [b for b in sdfg.all_control_flow_blocks() if 'num_threads' in b.label]
    assert not [e for e in sdfg.all_interstate_edges() if symbolic.NUM_THREADS_SYMBOL in e.data.assignments]


def test_frame_code_defines_it_before_the_first_allocation():
    """The whole point of moving it: the allocation that names the symbol is emitted at program
    entry, so the definition has to be above it."""
    code = generated(per_thread_sdfg('nt_before_alloc'))
    decl = code.index(f'{DTYPE.ctype} {symbolic.NUM_THREADS_SYMBOL} = omp_get_max_threads()')
    alloc = code.index(f'double[{symbolic.NUM_THREADS_SYMBOL}]')
    assert decl < alloc, code


def test_the_definition_reads_the_pool_size_and_is_guarded():
    """``omp_get_max_threads`` reports the TEAM size inside a parallel region, so it must be read
    where none is open -- the top of the program function is that place by construction. A build
    without OpenMP must still compile, answering 1 rather than guessing."""
    code = generated(per_thread_sdfg('nt_guarded'))
    assert '#ifdef _OPENMP' in code
    assert f'{DTYPE.ctype} {symbolic.NUM_THREADS_SYMBOL} = omp_get_max_threads();' in code
    assert f'{DTYPE.ctype} {symbolic.NUM_THREADS_SYMBOL} = 1;' in code
    assert '#include <omp.h>' in code


def test_the_definition_uses_the_declared_width():
    """One name, one width. A definition that disagreed with ``sdfg.symbols`` would leave codegen
    emitting the same symbol at two widths."""
    sdfg = per_thread_sdfg('nt_width')
    assert f'const {DTYPE.ctype} {symbolic.NUM_THREADS_SYMBOL}' in generated(sdfg)
    assert sdfg.symbols[symbolic.NUM_THREADS_SYMBOL] == DTYPE


def test_a_graph_that_never_sizes_per_thread_pays_nothing():
    """Declared on the SDFG so a later pass may use it, but nothing reads it here, so no definition
    and no OpenMP header are emitted."""
    sdfg = supplied()
    assert symbolic.NUM_THREADS_SYMBOL in sdfg.symbols
    assert not framecode.num_threads_is_used(sdfg)
    assert f'{DTYPE.ctype} {symbolic.NUM_THREADS_SYMBOL} =' not in generated(sdfg)


def test_a_size_only_use_counts_as_a_use():
    """An allocation is not an edge, a memlet or a tasklet, so a graph-only scan would report the
    symbol unused and skip the very definition the allocation needs."""
    assert framecode.num_threads_is_used(per_thread_sdfg('nt_size_use'))


def test_the_symbol_never_reaches_the_abi():
    """A caller cannot supply a machine property. The ``__dace`` prefix is what keeps it out, on
    both the scalar and the free-symbol path."""
    for sdfg in (supplied(), per_thread_sdfg('nt_abi')):
        assert symbolic.NUM_THREADS_SYMBOL not in sdfg.signature()
        assert symbolic.NUM_THREADS_SYMBOL not in sdfg.arglist()


def test_supplying_twice_declares_it_once():
    """Idempotent: the recipe may run it more than once, and two declarations of one symbol could
    disagree on the width."""
    sdfg = supplied()
    assert SupplyNumThreads().apply_pass(sdfg, {}) is None
    assert sdfg.symbols[symbolic.NUM_THREADS_SYMBOL] == DTYPE


def test_a_nested_sdfg_does_not_declare_its_own():
    """A nested graph receives the value through ``symbol_mapping``; declaring it again there would
    shadow the parent's and could disagree with it."""
    inner = dace.SDFG('inner_probe')
    inner.add_state()
    outer = dace.SDFG('outer_probe')
    state = outer.add_state()
    nsdfg = state.add_nested_sdfg(inner, {}, {})
    assert SupplyNumThreads().apply_pass(inner, {}) is None
    assert symbolic.NUM_THREADS_SYMBOL not in nsdfg.sdfg.symbols


def test_ensure_in_scope_maps_the_symbol_into_a_nested_sdfg():
    """A transformation that introduces a USE below the top level must top up the mapping, or the
    inner graph names a symbol nothing defines -- which validation rejects."""
    inner = dace.SDFG('inner_scope')
    inner.add_state()
    outer = dace.SDFG('outer_scope')
    state = outer.add_state()
    nsdfg = state.add_nested_sdfg(inner, {}, {})
    assert symbolic.NUM_THREADS_SYMBOL not in nsdfg.symbol_mapping

    ensure_in_scope(nsdfg)
    assert symbolic.NUM_THREADS_SYMBOL in nsdfg.symbol_mapping
    assert symbolic.NUM_THREADS_SYMBOL in nsdfg.sdfg.symbols

    ensure_in_scope(nsdfg)  # idempotent
    assert str(nsdfg.symbol_mapping[symbolic.NUM_THREADS_SYMBOL]) == symbolic.NUM_THREADS_SYMBOL


def test_ensure_in_scope_is_what_makes_a_nested_use_resolvable():
    """The defect ``ensure_in_scope`` exists for: a child whose SHAPE names the symbol while the
    parent maps nothing. That is the "Missing symbols on nested SDFG" failure seen on covariance --
    declaring the symbol without using it does not reproduce it, which is why this asserts on the
    use, not on the declaration."""
    inner = dace.SDFG('inner_use')
    inner.add_symbol(symbolic.NUM_THREADS_SYMBOL, dtypes.int64)
    inner.add_array('buf', [NT], dace.float64)
    outer = dace.SDFG('outer_use')
    state = outer.add_state()
    nsdfg = state.add_nested_sdfg(inner, {}, {'buf'})

    used = {str(sym) for sym in inner.free_symbols}
    assert symbolic.NUM_THREADS_SYMBOL in used, 'the child must genuinely use the symbol'
    # Building the node maps the child's free symbols for you, so the hazard is NOT construction.
    assert symbolic.NUM_THREADS_SYMBOL in nsdfg.symbol_mapping

    # The real case is a use introduced AFTER the node exists -- a transformation that moves a map
    # sized per thread into an existing nested SDFG. Nothing re-derives the mapping then, and the
    # graph fails codegen with "Missing symbols on nested SDFG".
    del nsdfg.symbol_mapping[symbolic.NUM_THREADS_SYMBOL]
    ensure_in_scope(nsdfg)
    assert symbolic.NUM_THREADS_SYMBOL in nsdfg.symbol_mapping, \
        'a use added after construction is only resolvable once the mapping is topped up'


def test_a_nested_per_thread_buffer_sees_the_definition():
    """A nested SDFG is inlined into the same program function, so the one frame-code definition
    covers a buffer the CHILD sizes -- which is the arrangement ``ensure_in_scope`` sets up."""
    inner = dace.SDFG('inner_nt')
    inner.add_symbol(symbolic.NUM_THREADS_SYMBOL, dtypes.int64)
    inner.add_array('o', [1], dace.float64)
    inner.add_transient('ibuf', [NT], dace.float64, lifetime=dtypes.AllocationLifetime.State)
    first = inner.add_state()
    writer = first.add_tasklet('w', {}, {'x'}, 'x = 2.0')
    first.add_edge(writer, 'x', first.add_access('ibuf'), None, dace.Memlet('ibuf[0]'))
    second = inner.add_state_after(first)
    reader = second.add_tasklet('r', {'x'}, {'y'}, 'y = x')
    second.add_edge(second.add_access('ibuf'), None, reader, 'x', dace.Memlet('ibuf[0]'))
    second.add_edge(reader, 'y', second.add_access('o'), None, dace.Memlet('o[0]'))

    outer = dace.SDFG('outer_nt')
    outer.add_array('out', [1], dace.float64)
    state = outer.add_state()
    nsdfg = state.add_nested_sdfg(inner, {}, {'o'})
    state.add_edge(nsdfg, 'o', state.add_access('out'), None, dace.Memlet('out[0]'))
    ensure_in_scope(nsdfg)
    SupplyNumThreads().apply_pass(outer, {})
    outer.validate()

    assert symbolic.NUM_THREADS_SYMBOL not in outer.arglist()
    assert generated(outer).count(f'{DTYPE.ctype} {symbolic.NUM_THREADS_SYMBOL} = omp_get_max_threads();') == 1
    out = np.zeros(1)
    outer.compile()(out=out)
    assert np.allclose(out, 2.0)


def test_the_supplied_value_compiles_and_runs():
    """End to end: the symbol must survive codegen and the program must be right."""
    sdfg = supplied()
    csdfg = sdfg.compile()
    a = np.arange(1.0, 9.0)
    out = np.zeros_like(a)
    csdfg(a=a, out=out, N=a.size)
    assert np.allclose(out, a * 2.0)


def test_a_per_thread_buffer_compiles_and_runs():
    """The case the frame-code definition exists for: a buffer whose size is the thread count."""
    out = np.zeros(1)
    per_thread_sdfg('nt_run').compile()(out=out)
    assert np.allclose(out, 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
