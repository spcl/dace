# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``ArgumentSignatureProperty`` (shared by ``SDFG.arg_names``
and ``SDFG.user_args``)."""

import pytest

import dace
from dace.properties import ArgumentSignatureProperty, make_properties


@make_properties
class _Probe:
    sig_flat = ArgumentSignatureProperty(allow_nested=False, desc='flat signature')
    sig_nested = ArgumentSignatureProperty(allow_nested=True, desc='nested signature')

    def __init__(self):
        self.sig_flat = []
        self.sig_nested = []


def _flat_prop() -> ArgumentSignatureProperty:
    return _Probe.__properties__['sig_flat']


def _nested_prop() -> ArgumentSignatureProperty:
    return _Probe.__properties__['sig_nested']


def test_argsig_flat_accepts_strings_and_canonicalizes():
    p = _Probe()
    p.sig_flat = ('A', 'b')
    assert p.sig_flat == ['A', 'b']
    assert isinstance(p.sig_flat, list)


def test_argsig_flat_rejects_nesting():
    p = _Probe()
    with pytest.raises(TypeError):
        p.sig_flat = [('A', 'B')]


def test_argsig_flat_rejects_bare_string():
    # Unlike ListProperty(str), a bare string does not explode into characters.
    p = _Probe()
    with pytest.raises(TypeError):
        p.sig_flat = 'abc'


def test_argsig_flat_rejects_non_string_entries():
    p = _Probe()
    with pytest.raises(TypeError):
        p.sig_flat = ['A', 1]


def test_argsig_nested_canonicalizes_to_inner_tuples():
    p = _Probe()
    p.sig_nested = [['A', ['B'], 'c'], 'd']
    assert p.sig_nested == [('A', ('B', ), 'c'), 'd']


def test_argsig_nested_rejects_empty_tuple():
    p = _Probe()
    with pytest.raises(ValueError):
        p.sig_nested = [()]


def test_argsig_to_json_is_self_describing():
    assert _flat_prop().to_json(['A', 'b']) == \
        {'type': 'ArgumentSignature', 'allow_nested': False, 'args': ['A', 'b']}
    assert _nested_prop().to_json([('A', 'B'), 'alpha']) == \
        {'type': 'ArgumentSignature', 'allow_nested': True, 'args': [['A', 'B'], 'alpha']}
    assert _flat_prop().to_json(None) is None


def test_argsig_new_format_roundtrip():
    for prop, value in ((_flat_prop(), ['A', 'b']), (_nested_prop(), [('A', ('B', )), 'alpha'])):
        assert prop.from_json(prop.to_json(value)) == value
    assert _flat_prop().from_json(None) is None


def test_argsig_legacy_list_loads_as_flat():
    # The old ListProperty format: a plain list of strings. It loads into
    # either mode as a flat signature.
    assert _flat_prop().from_json(['A', 'b']) == ['A', 'b']
    assert _nested_prop().from_json(['A', 'b']) == ['A', 'b']


def test_argsig_legacy_list_with_nesting_rejected():
    # Nested entries never existed in the legacy format, in neither mode.
    with pytest.raises(TypeError):
        _flat_prop().from_json([['A', 'B']])
    with pytest.raises(TypeError):
        _nested_prop().from_json([['A', 'B']])


def test_argsig_nested_data_refused_by_flat_property():
    # The stored flag is a contract: allow_nested=True data cannot load into a
    # flat-mode property, even if its entries happen to be flat.
    nested_json = _nested_prop().to_json(['A', 'b'])
    assert nested_json['allow_nested'] is True
    with pytest.raises(TypeError):
        _flat_prop().from_json(nested_json)
    # The other direction is fine: flat data is valid nested-mode data.
    flat_json = _flat_prop().to_json(['A', 'b'])
    assert _nested_prop().from_json(flat_json) == ['A', 'b']


def test_argsig_string_roundtrip():
    for prop, value in ((_flat_prop(), ['A', 'b']), (_nested_prop(), [('A', ('B', )), 'alpha'])):
        assert prop.from_string(prop.to_string(value)) == value


def test_argsig_sdfg_writes_new_format():
    sdfg = dace.SDFG('argsig_write_probe')
    sdfg.arg_names = ['A', 'b']
    sdfg.user_args = [('A', 'B'), 'alpha']
    attrs = sdfg.to_json()['attributes']
    assert attrs['arg_names'] == {'type': 'ArgumentSignature', 'allow_nested': False, 'args': ['A', 'b']}
    assert attrs['user_args'] == \
        {'type': 'ArgumentSignature', 'allow_nested': True, 'args': [['A', 'B'], 'alpha']}


def test_argsig_sdfg_reads_old_format():
    # An old-style .sdfg JSON carries arg_names as a plain list of strings.
    sdfg = dace.SDFG('argsig_read_probe')
    sdfg.arg_names = ['A', 'b']
    j = sdfg.to_json()
    j['attributes']['arg_names'] = ['A', 'b']
    restored = dace.SDFG.from_json(j)
    assert restored.arg_names == ['A', 'b']


if __name__ == '__main__':
    import sys
    pytest.main([__file__] + sys.argv[1:])
