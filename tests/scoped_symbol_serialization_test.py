# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scope-driven symbol-dtype serialization.

A symbol's serialized dtype is a deterministic function of the enclosing-scope
authority (the SDFG's registered symbols, overridden by any shadowing map/consume
scope), not of the symbol instance's own (SymPy-cache-stale) dtype. Outside SDFG
serialization the instance dtype is used.
"""
import json

import dace
from dace import symbolic


def _resave(sdfg: dace.SDFG):
    """Save -> load -> save; returns the two JSON strings (must match for a stable roundtrip)."""
    s1 = json.dumps(sdfg.to_json())
    s2 = json.dumps(dace.SDFG.from_json(json.loads(s1)).to_json())
    return s1, s2


def test_scope_authority_overrides_and_restores():
    """The active authority decides the dtype; opening a scope overrides and closing restores it."""
    sym = symbolic.symbol('x', dtype=dace.int64)

    # No scope active: the symbol's own (instance) dtype is used.
    assert symbolic.serialize_symbolic(sym) == 'symbol($x, dtype=dace.int64)'

    with symbolic.serialization_symbol_dtypes({'x': dace.int32}):
        assert symbolic.serialize_symbolic(sym) == '$x'  # int32 == DEFAULT_SYMBOL_TYPE -> bare
        with symbolic.serialization_symbol_dtypes({'x': dace.int64}):
            assert symbolic.serialize_symbolic(sym) == 'symbol($x, dtype=dace.int64)'
        assert symbolic.serialize_symbolic(sym) == '$x'  # inner scope closed, outer restored

    assert symbolic.serialize_symbolic(sym) == 'symbol($x, dtype=dace.int64)'  # back to no-scope


def test_scope_authority_only_overrides_named_symbols():
    """The authority overrides only the names it declares; an absent name keeps its own dtype."""
    sym = symbolic.symbol('y', dtype=dace.int64)
    with symbolic.serialization_symbol_dtypes({'other': dace.int64}):
        assert symbolic.serialize_symbolic(sym) == 'symbol($y, dtype=dace.int64)'  # absent -> instance


def test_registered_sdfg_symbol_roundtrips_its_dtype():
    """A symbol registered on the SDFG keeps its declared dtype through a roundtrip."""
    sdfg = dace.SDFG('registered')
    sdfg.add_symbol('N', dace.uint64)
    sdfg.add_array('A', (symbolic.symbol('N', dace.uint64), ), dace.float64)

    s1, s2 = _resave(sdfg)
    assert s1 == s2
    assert 'symbol($N, dtype=dace.uint64)' in s1

    restored = dace.SDFG.from_json(json.loads(s1))
    assert restored.symbols['N'] == dace.uint64
    shape_sym = next(iter(restored.arrays['A'].shape[0].free_symbols))
    assert shape_sym.dtype == dace.uint64


def test_map_iterator_dtype_follows_scope():
    """A map iterator's dtype is inferred from its range under the scope authority and round-trips."""
    sdfg = dace.SDFG('mapscope')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', (symbolic.symbol('N', dace.int64), ), dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('m',
                             map_ranges={'i': '0:N'},
                             inputs={},
                             code='b = 1.0',
                             outputs={'b': dace.Memlet(data='A', subset='i')},
                             external_edges=True)

    s1, s2 = _resave(sdfg)
    assert s1 == s2
    # result_type_of(int32 literal, int64 N) == int64, so the iterator is int64 in scope.
    assert 'symbol($i, dtype=dace.int64)' in s1


def test_nested_symbol_mapping_referencing_outer_map_param_roundtrips():
    """The map_fission ``test_dependent_symbol`` shape: an outer map param used in a nested SDFG's
    ``symbol_mapping`` serializes deterministically (regression for the flaky ``$i`` roundtrip)."""
    sdfg = dace.SDFG('outer')
    sdfg.add_symbol('fidx', dace.int32)
    sdfg.add_symbol('lidx', dace.int32)
    sdfg.add_array('A', (2, 10), dace.int32)
    sdfg.add_array('B', (2, 10), dace.int32)

    inner = dace.SDFG('inner')
    inner.add_symbol('first', dace.int32)
    inner.add_symbol('last', dace.int32)
    inner.add_array('A0', (10, ), dace.int32)
    inner.add_array('B0', (10, ), dace.int32)
    istate = inner.add_state('s', is_start_block=True)
    istate.add_mapped_tasklet('plus',
                              map_ranges={'j': 'first:last'},
                              inputs={'__a': dace.Memlet(data='A0', subset='j')},
                              code='__b = __a + 1',
                              outputs={'__b': dace.Memlet(data='B0', subset='j')},
                              external_edges=True)

    state = sdfg.add_state('outer', is_start_block=True)
    a = state.add_access('A')
    b = state.add_access('B')
    me, mx = state.add_map('map', {'i': '0:2'})
    nsdfg = state.add_nested_sdfg(inner, {'A0'}, {'B0'},
                                  symbol_mapping={
                                      'first': 'max(0, i - fidx)',
                                      'last': 'min(10, i + lidx)'
                                  })
    state.add_memlet_path(a, me, nsdfg, memlet=dace.Memlet(data='A', subset='0, 0:10'), dst_conn='A0')
    state.add_memlet_path(nsdfg, mx, b, memlet=dace.Memlet(data='B', subset='0, 0:10'), src_conn='B0')

    s1, s2 = _resave(sdfg)
    assert s1 == s2


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
