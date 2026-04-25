# Copyright 2026 ETH Zurich and the DaCe authors. All rights reserved.
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


def test_direct_mutation_of_symbols_is_blocked():
    """The strict guard on ``sdfg.symbols`` rejects all mutation paths that
    bypass the official API."""
    sdfg = dace.SDFG('host')
    sdfg.add_symbol('K', dtypes.int32)
    with pytest.raises(RuntimeError):
        sdfg.symbols['K'] = dtypes.float64
    with pytest.raises(RuntimeError):
        del sdfg.symbols['K']
    with pytest.raises(RuntimeError):
        sdfg.symbols.pop('K')
    with pytest.raises(RuntimeError):
        sdfg.symbols.popitem()
    with pytest.raises(RuntimeError):
        sdfg.symbols.update({'M': dtypes.int64})
    with pytest.raises(RuntimeError):
        sdfg.symbols.setdefault('M', dtypes.int64)
    with pytest.raises(RuntimeError):
        sdfg.symbols.clear()


def test_dict_class_method_bypass_is_blocked():
    """Calling ``dict.__setitem__`` / ``dict.update`` etc. directly on the
    instance is a known way to bypass Python-level overrides on ``dict``
    subclasses. ``_SymbolDict`` is a ``MutableMapping`` (composition, not
    inheritance), so those C-level slots simply do not apply to it."""
    sdfg = dace.SDFG('host')
    sdfg.add_symbol('K', dtypes.int32)
    for fn in (
        lambda: dict.__setitem__(sdfg.symbols, 'M', dtypes.int64),
        lambda: dict.__delitem__(sdfg.symbols, 'K'),
        lambda: dict.pop(sdfg.symbols, 'K'),
        lambda: dict.update(sdfg.symbols, {'M': dtypes.int64}),
        lambda: dict.clear(sdfg.symbols),
    ):
        with pytest.raises(TypeError):
            fn()


def test_replacing_backing_storage_is_blocked():
    """Writing ``sdfg._symbols = some_plain_dict`` -- attempting to swap the
    underlying storage with an unguarded mapping -- is rejected by
    ``SDFG.__setattr__``. Note: assignments via ``object.__setattr__`` or
    ``sdfg.__dict__`` deliberately bypass any class-level ``__setattr__`` and
    are out of scope -- they are explicit Python introspection, not normal
    code paths."""
    sdfg = dace.SDFG('host')
    sdfg.add_symbol('K', dtypes.int32)
    with pytest.raises(RuntimeError):
        sdfg._symbols = {'M': dtypes.int64}


def test_property_assignment_rewraps():
    """Assigning ``sdfg.symbols = {...}`` (the property path) re-wraps the
    new dict so it stays guarded."""
    sdfg = dace.SDFG('host')
    sdfg.symbols = {'Q': dtypes.int64}
    assert type(sdfg.symbols).__name__ == '_SymbolDict'
    with pytest.raises(RuntimeError):
        sdfg.symbols['R'] = dtypes.int64


def test_set_symbol_type_swaps_dtype():
    """``set_symbol_type`` is the official path for in-place dtype swaps; it
    rebuilds the canonical object so dtype and registry stay in sync."""
    sdfg = dace.SDFG('host')
    sdfg.add_symbol('K', dtypes.int32)
    sdfg.set_symbol_type('K', dtypes.float64)
    assert sdfg.symbols['K'] == dtypes.float64
    assert sdfg.get_symbol('K').dtype == dtypes.float64


def test_remove_symbol_clears_registry():
    sdfg = dace.SDFG('host')
    sdfg.add_symbol('N', dtypes.int64, positive=True)
    assert 'N' in sdfg._symbol_objects
    sdfg.remove_symbol('N')
    assert 'N' not in sdfg.symbols
    assert 'N' not in sdfg._symbol_objects
    # After removal we must be able to re-register with new conditions.
    sdfg.add_symbol('N', dtypes.float32)
    assert sdfg.get_symbol('N').dtype == dtypes.float32


def test_replace_renames_in_registry():
    sdfg = dace.SDFG('host')
    sdfg.add_symbol('N', dtypes.int64, positive=True)
    sdfg.replace('N', 'M')
    assert 'N' not in sdfg.symbols
    assert 'M' in sdfg.symbols
    # After rename the canonical object's name must follow.
    new_sym = sdfg.get_symbol('M')
    assert new_sym.name == 'M'
    assert new_sym.dtype == dtypes.int64
    assert new_sym.assumptions0.get('positive') is True


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
    test_direct_mutation_of_symbols_is_blocked()
    test_dict_class_method_bypass_is_blocked()
    test_replacing_backing_storage_is_blocked()
    test_property_assignment_rewraps()
    test_set_symbol_type_swaps_dtype()
    test_remove_symbol_clears_registry()
    test_replace_renames_in_registry()
    test_placeholder_name_still_accepted()
    print('All nested symbol-registry tests passed.')
