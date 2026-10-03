# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One name, one dtype: ``EqualizeSymbolDtypes`` rebuilds every spelling of a name at the dtype its scope declares.

A symbol's dtype is part of its identity, so ``N`` at ``int32`` and ``N`` at ``int64`` are unrelated sympy symbols that
never cancel. The tests build graphs where one name is spelled at two dtypes in different places and ask for the single
declared one afterwards.
"""
from typing import Dict, Set

import sympy

import dace
from dace import dtypes, subsets, symbolic
from dace.sdfg import nodes
from dace.transformation.passes.equalize_symbol_dtypes import equalize, equalized

N = dace.symbol('N', dtype=dace.int64)


def bound_symbols(expr) -> Set[symbolic.symbol]:
    if isinstance(expr, symbolic.SymExpr):
        return bound_symbols(expr.expr) | bound_symbols(expr.approx)
    return {s for s in getattr(expr, 'free_symbols', ()) if isinstance(s, symbolic.symbol)}


def subset_symbols(subset: subsets.Subset) -> Set[symbolic.symbol]:
    return {s for rng in subset.ndrange() for bound in rng for s in bound_symbols(bound)}


def dtypes_of(sdfg: dace.SDFG) -> Dict[str, Set[dtypes.typeclass]]:
    """Every dtype each symbol name carries in a map range, memlet subset or descriptor extent of the whole tree."""
    seen: Dict[str, Set[dtypes.typeclass]] = {}

    def add(symbols):
        for sym in symbols:
            seen.setdefault(sym.name, set()).add(sym.dtype)

    for nested in sdfg.all_sdfgs_recursive():
        for desc in nested.arrays.values():
            add(s for extent in desc.shape for s in bound_symbols(extent))
        for state in nested.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    add(subset_symbols(node.map.range))
            for edge in state.edges():
                for subset in (edge.data.subset, edge.data.other_subset):
                    if subset is not None:
                        add(subset_symbols(subset))
    return seen


def respell(subset: subsets.Range, name: str, dtype: dtypes.typeclass) -> subsets.Range:
    """``subset`` with the symbol ``name`` rebuilt at ``dtype``."""
    ranges = []
    for rng in subset.ranges:
        ranges.append(
            tuple(
                bound.xreplace({s: symbolic.symbol(name, dtype)
                                for s in bound_symbols(bound)
                                if s.name == name}) if isinstance(bound, sympy.Basic) else bound for bound in rng))
    return subsets.Range(ranges)


def copy_with_range_at_int32(label: str) -> dace.SDFG:
    """``B[0:N] = A[0:N]`` under a map over ``N`` (int64), whose input memlet spells ``N`` at int32."""
    sdfg = dace.SDFG(label)
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('copy', {'i': '0:N'}, {'a': dace.Memlet('A[i]')},
                             'b = a', {'b': dace.Memlet('B[i]')},
                             external_edges=True)
    for edge in state.edges():
        if edge.data.data == 'A' and isinstance(edge.src, nodes.AccessNode):
            edge.data.subset = subsets.Range([(0, symbolic.symbol('N', dace.int32) - 1, 1)])
    return sdfg


def test_a_memlet_spelling_a_declared_symbol_at_another_dtype_is_rebuilt():
    sdfg = copy_with_range_at_int32('equalize_memlet_spelling')
    assert dtypes_of(sdfg)['N'] == {dace.int64, dace.int32}

    assert equalize(sdfg) is not None

    assert dtypes_of(sdfg)['N'] == {dace.int64}
    sdfg.validate()


def test_a_consistent_sdfg_is_left_alone():
    sdfg = copy_with_range_at_int32('equalize_consistent')
    equalize(sdfg)

    assert equalize(sdfg) is None
    assert dtypes_of(sdfg)['N'] == {dace.int64}


def test_a_loop_iterator_takes_the_dtype_the_loop_declares():

    @dace.program
    def loop_copy(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            B[i] = A[i]

    sdfg = loop_copy.to_sdfg(simplify=False)
    sdfg.name = 'equalize_loop_iterator'
    iterator_dtype = None
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.subset is not None and 'i' in {s.name for s in subset_symbols(edge.data.subset)}:
                iterator_dtype = next(s.dtype for s in subset_symbols(edge.data.subset) if s.name == 'i')
                edge.data.subset = respell(edge.data.subset, 'i', dace.int8)
    assert iterator_dtype is not None and iterator_dtype != dace.int8
    assert dtypes_of(sdfg)['i'] == {dace.int8}

    equalize(sdfg)

    assert dtypes_of(sdfg)['i'] == {iterator_dtype}


def test_a_nested_declaration_follows_the_symbol_it_is_mapped_onto():
    inner = dace.SDFG('equalize_nested_inner')
    inner.add_symbol('N', dace.int32)
    inner.add_array('X', [N], dace.float64)
    inner.add_state()
    outer = dace.SDFG('equalize_nested_outer')
    outer.add_symbol('N', dace.int64)
    outer.add_array('X', [N], dace.float64)
    state = outer.add_state()
    node = state.add_nested_sdfg(inner, {'X'}, set(), symbol_mapping={'N': 'N'})
    state.add_edge(state.add_read('X'), None, node, 'X', dace.Memlet('X[0:N]'))

    equalize(outer)

    assert inner.symbols['N'] == dace.int64
    assert dtypes_of(outer)['N'] == {dace.int64}


def test_the_stage_reads_names_from_text_at_the_declared_dtype():
    sdfg = copy_with_range_at_int32('equalize_stage_authority')

    with equalized(sdfg):
        assert symbolic.pystr_to_symbolic('N').dtype == dace.int64
        assert symbolic.declared_symbol_dtype('i') is not None

    assert symbolic.declared_symbol_dtype('N') is None


def test_a_name_minted_inside_the_stage_is_rebuilt_on_exit():
    sdfg = copy_with_range_at_int32('equalize_stage_exit')
    equalize(sdfg)

    with equalized(sdfg):
        for edge in sdfg.state(0).edges():
            if edge.data.data == 'B' and isinstance(edge.dst, nodes.AccessNode):
                edge.data.subset = subsets.Range([(0, symbolic.symbol('N', dace.int32) - 1, 1)])

    assert dtypes_of(sdfg)['N'] == {dace.int64}


if __name__ == '__main__':
    test_a_memlet_spelling_a_declared_symbol_at_another_dtype_is_rebuilt()
    test_a_consistent_sdfg_is_left_alone()
    test_a_loop_iterator_takes_the_dtype_the_loop_declares()
    test_a_nested_declaration_follows_the_symbol_it_is_mapped_onto()
    test_the_stage_reads_names_from_text_at_the_declared_dtype()
    test_a_name_minted_inside_the_stage_is_rebuilt_on_exit()
