# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``compiler.interface=auto``: per-SDFG selection between the nanobind
and the ctypes interface, decided by ``compiler.resolve_compiler_interface``."""
import numpy as np
import pytest

import dace
from dace.config import Config, set_temporary


@pytest.fixture
def clean_interface_env(monkeypatch):
    """Drops the ``DACE_compiler_interface`` export so ``set_temporary`` (and the
    schema default) decide - the env var overrides ``Config`` reads."""
    monkeypatch.delenv('DACE_compiler_interface', raising=False)
    yield


def _supported_sdfg(name: str) -> dace.SDFG:
    """A trivially nanobind-supported SDFG: one primitive array, one scalar."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_scalar('alpha', dace.float64)
    state = sdfg.add_state()
    t = state.add_tasklet('t', {'a_in', 's_in'}, {'a_out'}, 'a_out = a_in + s_in')
    state.add_edge(state.add_read('A'), None, t, 'a_in', dace.Memlet('A[0]'))
    state.add_edge(state.add_read('alpha'), None, t, 's_in', dace.Memlet('alpha[0]'))
    state.add_edge(t, 'a_out', state.add_write('A'), None, dace.Memlet('A[0]'))
    return sdfg


def _callback_sdfg(name: str) -> dace.SDFG:
    """An SDFG with a callback argument - out of the nanobind interface's scope."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_symbol('cb', dace.callback(dace.float64, dace.float64))
    state = sdfg.add_state()
    t = state.add_tasklet('t', {}, {'o'}, 'o = cb(1.0)')
    state.add_edge(t, 'o', state.add_write('A'), None, dace.Memlet('A[0]'))
    return sdfg


def _structure_sdfg(name: str) -> dace.SDFG:
    """An SDFG with a Structure argument - out of the nanobind interface's scope."""
    csr = dace.data.Structure(dict(data=dace.float32[4]), name='OnlyData')
    sdfg = dace.SDFG(name)
    sdfg.add_datadesc('A', csr)
    sdfg.add_state()
    return sdfg


def test_interface_default_is_auto(clean_interface_env):
    assert Config.get_default('compiler', 'interface') == 'auto'


def test_resolve_compiler_interface(clean_interface_env):
    """The external decision function: explicit values pass through, auto
    inspects the SDFG, and no SDFG means the ctypes-compatible answer."""
    from dace.codegen.compiler import resolve_compiler_interface

    supported = _supported_sdfg('resolve_supported_probe')
    unsupported = _callback_sdfg('resolve_callback_probe')

    with set_temporary('compiler', 'interface', value='auto'):
        assert resolve_compiler_interface(supported) == 'nanobind'
        assert resolve_compiler_interface(unsupported) == 'ctypes'
        assert resolve_compiler_interface(None) == 'ctypes'
    with set_temporary('compiler', 'interface', value='ctypes'):
        assert resolve_compiler_interface(supported) == 'ctypes'
        assert resolve_compiler_interface(unsupported) == 'ctypes'
    with set_temporary('compiler', 'interface', value='nanobind'):
        # Explicit nanobind is a demand, not a hint: the decision function
        # honors it and the codegen refusal (fail-fast) is the backstop.
        assert resolve_compiler_interface(supported) == 'nanobind'
        assert resolve_compiler_interface(unsupported) == 'nanobind'
    with set_temporary('compiler', 'interface', value='no_such_interface'):
        with pytest.raises(ValueError, match='compiler.interface'):
            resolve_compiler_interface(supported)


def test_auto_falls_back_to_ctypes_without_nanobind_package(clean_interface_env, monkeypatch):
    """``auto`` in an environment without the nanobind package resolves to ctypes instead of
    crashing at compile time, and says so once -- nanobind is a declared dependency, so its
    absence usually means a broken installation and a silent fallback would mask it."""
    import warnings

    from dace.codegen import compiler

    monkeypatch.setattr(compiler, '_nanobind_available', lambda: False)
    compiler._warn_nanobind_unavailable.cache_clear()
    supported = _supported_sdfg('no_nanobind_probe')
    with set_temporary('compiler', 'interface', value='auto'):
        with pytest.warns(UserWarning, match='nanobind package is not installed'):
            assert compiler.resolve_compiler_interface(supported) == 'ctypes'
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # the warning fires once per process, not per compile
            assert compiler.resolve_compiler_interface(supported) == 'ctypes'
    compiler._warn_nanobind_unavailable.cache_clear()


def test_explicit_nanobind_ignores_availability_probe(clean_interface_env, monkeypatch):
    """Explicit ``nanobind`` stays a demand even without the package: resolution passes it
    through unchanged, and the hard CompilerConfigurationError at compile time is the backstop."""
    from dace.codegen import compiler

    monkeypatch.setattr(compiler, '_nanobind_available', lambda: False)
    with set_temporary('compiler', 'interface', value='nanobind'):
        assert compiler.resolve_compiler_interface(_supported_sdfg('demand_probe')) == 'nanobind'


def test_auto_selects_nanobind_for_supported_sdfg(clean_interface_env):
    from dace.codegen.nanobind_compiled_sdfg import NanobindCompiledSDFG

    with set_temporary('compiler', 'interface', value='auto'):
        csdfg = _supported_sdfg('auto_supported_prog').compile()
        assert isinstance(csdfg, NanobindCompiledSDFG)
        a = np.zeros(10)
        csdfg(A=a, alpha=2.0)
        assert a[0] == 2.0


def test_auto_falls_back_to_ctypes_for_callbacks(clean_interface_env):
    from dace.codegen.ctypes_compiled_sdfg import CtypesCompiledSDFG

    with set_temporary('compiler', 'interface', value='auto'):
        csdfg = _callback_sdfg('auto_callback_prog').compile()
        assert isinstance(csdfg, CtypesCompiledSDFG)
        a = np.zeros(10)
        csdfg(A=a, cb=lambda x: x + 41.0)
        assert a[0] == 42.0


def test_auto_falls_back_to_ctypes_for_structures(clean_interface_env):
    from dace.codegen.ctypes_compiled_sdfg import CtypesCompiledSDFG

    with set_temporary('compiler', 'interface', value='auto'):
        # Compile only (no execution: a Structure call needs a built ctypes
        # object; the selection is what is under test).
        csdfg = _structure_sdfg('auto_structure_prog').compile()
        assert isinstance(csdfg, CtypesCompiledSDFG)


def test_explicit_nanobind_refuses_unsupported_sdfg(clean_interface_env):
    with set_temporary('compiler', 'interface', value='nanobind'):
        with pytest.raises(NotImplementedError, match='ctypes'):
            _callback_sdfg('explicit_nanobind_callback_prog').compile()


def test_explicit_ctypes_still_ctypes(clean_interface_env):
    from dace.codegen.ctypes_compiled_sdfg import CtypesCompiledSDFG

    with set_temporary('compiler', 'interface', value='ctypes'):
        csdfg = _supported_sdfg('explicit_ctypes_prog').compile()
        assert isinstance(csdfg, CtypesCompiledSDFG)


if __name__ == '__main__':
    test_resolve_compiler_interface()
    test_auto_selects_nanobind_for_supported_sdfg()
    test_auto_falls_back_to_ctypes_for_callbacks()
    test_auto_falls_back_to_ctypes_for_structures()
    test_explicit_nanobind_refuses_unsupported_sdfg()
    test_explicit_ctypes_still_ctypes()
