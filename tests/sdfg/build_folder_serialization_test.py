# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Serialization of an explicitly-set build folder.

An explicitly-set ``build_folder`` is part of the user's contract and survives
a JSON round-trip. A configuration-derived folder (``_build_folder is None``)
is environment state: it is deliberately NOT serialized - the key is omitted
entirely, which keeps the serialized form (and thus hashes and build caches)
of such SDFGs unchanged - and it is restored as derived (``None``), which also
covers files written before the key existed.
"""

import pathlib
import warnings

import dace


def _make_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('bf_ser_probe')
    sdfg.add_array('A', [4], dace.float64)
    return sdfg


def test_explicit_build_folder_roundtrip():
    sdfg = _make_sdfg()
    sdfg.build_folder = '/some/explicit/folder'

    j = sdfg.to_json()
    assert j['attributes']['build_folder'] == '/some/explicit/folder'

    with warnings.catch_warnings():
        # A leftover 'build_folder' key would trigger the "Unused properties"
        # warning of set_properties_from_json; it must be consumed cleanly.
        warnings.simplefilter('error')
        restored = dace.SDFG.from_json(j)
    assert restored._build_folder == '/some/explicit/folder'
    assert restored.build_folder == '/some/explicit/folder'


def test_relative_explicit_build_folder_kept_verbatim():
    sdfg = _make_sdfg()
    sdfg.build_folder = 'relative/dir'
    restored = dace.SDFG.from_json(sdfg.to_json())
    assert restored._build_folder == 'relative/dir'


def test_pathlib_build_folder_serializes_as_string():
    sdfg = _make_sdfg()
    sdfg.build_folder = pathlib.Path('/pathlib/folder')
    j = sdfg.to_json()
    assert j['attributes']['build_folder'] == '/pathlib/folder'
    assert isinstance(j['attributes']['build_folder'], str)
    assert dace.SDFG.from_json(j)._build_folder == '/pathlib/folder'


def test_derived_build_folder_not_serialized():
    # No explicit folder: the key is absent - the serialized form of such
    # SDFGs is byte-identical to before the feature (no hash/cache impact).
    sdfg = _make_sdfg()
    j = sdfg.to_json()
    assert 'build_folder' not in j['attributes']

    restored = dace.SDFG.from_json(j)
    assert restored._build_folder is None


def test_legacy_json_without_key_restores_derived():
    # Files written before the key existed: nothing to consume, folder derives.
    sdfg = _make_sdfg()
    sdfg.build_folder = '/explicit/but/stripped'
    j = sdfg.to_json()
    del j['attributes']['build_folder']

    restored = dace.SDFG.from_json(j)
    assert restored._build_folder is None


def test_explicit_build_folder_enters_hash():
    # Pinning a build location is content: it changes the SDFG hash.
    plain = _make_sdfg().hash_sdfg()
    pinned = _make_sdfg()
    pinned.build_folder = '/some/explicit/folder'
    assert pinned.hash_sdfg() != plain


if __name__ == '__main__':
    test_explicit_build_folder_roundtrip()
    test_relative_explicit_build_folder_kept_verbatim()
    test_pathlib_build_folder_serializes_as_string()
    test_derived_build_folder_not_serialized()
    test_legacy_json_without_key_restores_derived()
    test_explicit_build_folder_enters_hash()
