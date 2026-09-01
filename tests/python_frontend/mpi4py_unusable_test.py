# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An mpi4py that is installed but cannot reach an MPI runtime must read as "no MPI".

The wheel carries no MPI of its own: it ``dlopen``s ``libmpi`` when the ``MPI`` submodule is first
imported, so on a machine with the package and no MPI runtime that import raises ``RuntimeError:
cannot load MPI library`` rather than ``ImportError``. The frontend guarded only for ``ImportError``,
so the RuntimeError escaped ``resolve_names`` and EVERY ``@dace.program`` -- including ones that
never mention MPI -- failed to parse. A stray ``pip install mpi4py`` was enough to brick DaCe.
"""

import sys
import types

import numpy as np
import pytest

import dace
from dace.frontend.python.preprocessing import mpi4py_is_usable

N = dace.symbol('N')


@dace.program
def scale(x: dace.float64[N]):
    """A program with nothing to do with MPI. It must parse whatever mpi4py is doing."""
    x[:] = x * 2.0


class UnusableMPI4Py(types.ModuleType):
    """Importable ``mpi4py`` whose ``MPI`` submodule cannot load libmpi."""

    def __getattr__(self, name):
        if name == 'MPI':
            raise RuntimeError('cannot load MPI library')
        raise AttributeError(name)


@pytest.fixture
def unusable_mpi4py(monkeypatch):
    """Swap in an mpi4py that imports but whose MPI submodule raises, and drop the cached verdict."""
    monkeypatch.setitem(sys.modules, 'mpi4py', UnusableMPI4Py('mpi4py'))
    monkeypatch.delitem(sys.modules, 'mpi4py.MPI', raising=False)
    mpi4py_is_usable.cache_clear()
    yield
    mpi4py_is_usable.cache_clear()


def test_unusable_mpi4py_reads_as_absent(unusable_mpi4py):
    """The RuntimeError path must answer False, not propagate."""
    assert mpi4py_is_usable() is False


def test_program_without_mpi_still_parses_and_runs(unusable_mpi4py):
    """The regression itself: a non-MPI program parses, compiles and computes with mpi4py broken."""
    x = np.ones(8)
    scale.__sdfg__(x=x)  # force a fresh parse under the patched module
    scale(x=x, N=8)
    assert np.allclose(x, 2.0)


def test_missing_mpi4py_also_reads_as_absent(monkeypatch):
    """The pre-existing ImportError path must keep working -- this fix widens it, not replaces it."""

    def refuse(name, *args, **kwargs):
        if name.startswith('mpi4py'):
            raise ImportError('No module named mpi4py')
        return original(name, *args, **kwargs)

    original = __builtins__['__import__'] if isinstance(__builtins__, dict) else __builtins__.__import__
    monkeypatch.delitem(sys.modules, 'mpi4py', raising=False)
    monkeypatch.delitem(sys.modules, 'mpi4py.MPI', raising=False)
    monkeypatch.setattr('builtins.__import__', refuse)
    mpi4py_is_usable.cache_clear()
    try:
        assert mpi4py_is_usable() is False
    finally:
        mpi4py_is_usable.cache_clear()


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
