# Copyright 2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the per-SDFG symbol object registry, focused on nested SDFGs.

A symbol is fully identified by (name, dtype, sympy assumptions). The registry
must ensure that:

  1. Re-adding the same name with matching conditions is idempotent.
  2. Re-adding the same name with conflicting conditions is rejected.
  3. Inner (nested) and outer SDFGs keep independent registries, so a symbol
     named 'N' may exist in each -- they're different objects.
  4. When a symbol flows from outer to inner via ``symbol_mapping``, the inner
     SDFG ends up with a registration that matches the outer one.
  5. ``get_symbol`` returns a stable, canonical object within one SDFG.
"""
import pytest

import dace
from dace import dtypes


def _make_leaf(name: str = 'leaf') -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_state('s')
    return sdfg


def test_inner_and_outer_independent_registries():
    outer = dace.SDFG('outer')
    inner = _make_leaf('inner')

    outer.add_symbol('N', dtypes.int64)
    inner.add_symbol('N', dtypes.int64, positive=True)

    # The two registries are independent -- same name, different objects.
    assert outer.get_symbol('N') is not inner.get_symbol('N')
    assert outer.get_symbol('N').assumptions0.get('positive') is None
    assert inner.get_symbol('N').assumptions0.get('positive') is True


def test_outer_add_then_inner_add_must_match_conditions():
    outer = dace.SDFG('outer')
    inner = _make_leaf('inner')

    outer.add_symbol('N', dtypes.int64, positive=True)
    # Inner adds with the same conditions -- must be idempotent.
    inner.add_symbol('N', dtypes.int64, positive=True)
    inner.add_symbol('N', dtypes.int64, positive=True)
    assert inner.get_symbol('N').assumptions0.get('positive') is True


def test_re_add_with_conflicting_assumption_is_rejected():
    sdfg = dace.SDFG('host')
    sdfg.add_symbol('N', dtypes.int64)

    with pytest.raises(FileExistsError):
        sdfg.add_symbol('N', dtypes.int64, positive=True)
    with pytest.raises(FileExistsError):
        sdfg.add_symbol('N', dtypes.int64, nonnegative=True)


def test_re_add_with_conflicting_dtype_is_rejected():
    sdfg = dace.SDFG('host')
    sdfg.add_symbol('N', dtypes.int64)

    with pytest.raises(FileExistsError):
        sdfg.add_symbol('N', dtypes.float32)


def test_find_new_name_only_kicks_in_on_conflict():
    sdfg = dace.SDFG('host')
    n = sdfg.add_symbol('N', dtypes.int64, find_new_name=True)
    assert n == 'N', 'find_new_name must not rename when no conflict exists'

    # Re-adding with the same conditions is still a no-op even with
    # find_new_name=True -- we must not create an 'N_0' duplicate.
    n2 = sdfg.add_symbol('N', dtypes.int64, find_new_name=True)
    assert n2 == 'N'
    assert len(sdfg.symbols) == 1

    # A real conflict (different assumptions) under find_new_name must rename.
    n3 = sdfg.add_symbol('N', dtypes.int64, find_new_name=True, positive=True)
    assert n3 == 'N_0'
    assert sdfg.get_symbol(n3).assumptions0.get('positive') is True


def test_get_symbol_returns_stable_object():
    sdfg = dace.SDFG('host')
    sdfg.add_symbol('N', dtypes.int64, nonnegative=True)

    s1 = sdfg.get_symbol('N')
    s2 = sdfg.get_symbol('N')
    assert s1 is s2, 'get_symbol must return the same canonical Python object'
    assert s1.dtype == dtypes.int64
    assert s1.assumptions0.get('nonnegative') is True


def test_nested_sdfg_propagates_outer_symbol():
    """When the inner SDFG references a symbol declared in the outer SDFG, the
    outer SDFG's registry entry must survive inner registration via
    ``add_nested_sdfg``'s auto-symbol propagation path."""
    outer = dace.SDFG('outer')
    outer.add_array('A', [20], dtypes.float64)
    outer.add_array('B', [20], dtypes.float64)
    outer.add_symbol('N', dtypes.int64, positive=True)

    inner = _make_leaf('inner')
    inner.add_array('a', [20], dtypes.float64)
    inner.add_array('b', [20], dtypes.float64)
    # Inner references 'N' via a mapped tasklet subset.
    inner_state = inner.nodes()[0]
    inner_state.add_mapped_tasklet(
        'copy',
        map_ranges={'i': '0:N'},
        inputs={'__in': dace.Memlet('a[i]')},
        outputs={'__out': dace.Memlet('b[i]')},
        code='__out = __in',
        external_edges=True,
    )

    outer_state = outer.add_state('outer_state')
    a = outer_state.add_access('A')
    b = outer_state.add_access('B')
    nsdfg_node = outer_state.add_nested_sdfg(inner, {'a'}, {'b'})
    outer_state.add_edge(a, None, nsdfg_node, 'a', dace.Memlet.from_array('A', outer.arrays['A']))
    outer_state.add_edge(nsdfg_node, 'b', b, None, dace.Memlet.from_array('B', outer.arrays['B']))

    # After add_nested_sdfg, the inner SDFG must have 'N' registered with a
    # dtype compatible with what it had (or int by default).
    assert 'N' in inner.symbols

    # Re-adding 'N' to the outer with the *same* conditions must be idempotent.
    outer.add_symbol('N', dtypes.int64, positive=True)
    assert len([s for s in outer.symbols if s == 'N']) == 1


def test_legacy_entry_without_object_can_be_upgraded_then_locked():
    """Simulates a symbol that was registered pre-registry (only dtype stored).
    The first add_symbol with matching dtype + no assumptions fills in the
    registry. Subsequent adds with different assumptions must be rejected.
    """
    sdfg = dace.SDFG('host')
    # Simulate a legacy registration (dtype-only, no canonical object).
    sdfg.symbols['K'] = dtypes.int32

    sdfg.add_symbol('K', dtypes.int32)  # upgrades registry
    assert 'K' in sdfg._symbol_objects
    assert sdfg.get_symbol('K').dtype == dtypes.int32

    # Now adding with a new assumption must fail.
    with pytest.raises(FileExistsError):
        sdfg.add_symbol('K', dtypes.int32, positive=True)


def test_placeholder_name_still_accepted():
    """The UndefinedSymbol placeholder uses the name '?' which is not a valid
    identifier. add_symbol must still accept it (graceful fallback) without
    crashing during symbol object construction.
    """
    sdfg = dace.SDFG('host')
    sdfg.add_symbol('?', dtypes.int64)
    assert '?' in sdfg.symbols
    # Idempotent re-add must also work.
    sdfg.add_symbol('?', dtypes.int64)
    assert len(sdfg.symbols) == 1


if __name__ == '__main__':
    test_inner_and_outer_independent_registries()
    test_outer_add_then_inner_add_must_match_conditions()
    test_re_add_with_conflicting_assumption_is_rejected()
    test_re_add_with_conflicting_dtype_is_rejected()
    test_find_new_name_only_kicks_in_on_conflict()
    test_get_symbol_returns_stable_object()
    test_nested_sdfg_propagates_outer_symbol()
    test_legacy_entry_without_object_can_be_upgraded_then_locked()
    test_placeholder_name_still_accepted()
    print('All nested symbol-registry tests passed.')
