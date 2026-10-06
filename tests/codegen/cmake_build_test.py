# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CMake build: the runnable library and its stub, and the opt-in static archive beside it."""
import os
import subprocess

import numpy as np

import dace
from dace.config import set_temporary


def archive_path(lib: str) -> str:
    """The ``lib<name>.a`` that sits next to the runnable ``lib<name>.so`` at ``lib``."""
    return os.path.join(os.path.dirname(lib), os.path.basename(lib)[:-len('.so')] + '.a')


def test_cmake_build_produces_a_library_and_its_stub(tmp_path):

    @dace.program
    def axpy_cmake(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        c[:] = 2.0 * a + b

    a = np.random.rand(32)
    b = np.random.rand(32)
    c = np.zeros(32)
    sdfg = axpy_cmake.to_sdfg()
    sdfg.build_folder = str(tmp_path / 'cache')
    csdfg = sdfg.compile()
    lib = str(csdfg._lib._library_filename)
    stub = os.path.join(os.path.dirname(lib), 'libdacestub_' + os.path.basename(lib)[3:])
    assert os.path.isfile(lib) and os.path.isfile(stub)
    csdfg(a=a, b=b, c=c)
    assert np.allclose(c, 2.0 * a + b)


def test_cmake_static_archive_emitted_alongside_so(tmp_path):
    """With ``compiler.static_archive`` on, the build ALSO emits ``lib<name>.a`` from the same objects
    as the shared library; the ``.so`` is untouched and runs bit-exact."""

    @dace.program
    def axpy_ar_cm(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        c[:] = 2.0 * a + b

    a, b, c = np.random.rand(32), np.random.rand(32), np.zeros(32)
    sdfg = axpy_ar_cm.to_sdfg()
    sdfg.build_folder = str(tmp_path / 'cache')
    with set_temporary('compiler', 'static_archive', value=True):
        csdfg = sdfg.compile()
    lib = str(csdfg._lib._library_filename)
    archive = archive_path(lib)
    assert os.path.isfile(lib)
    assert os.path.isfile(archive)
    members = subprocess.check_output(['ar', 't', archive]).decode()
    assert members.strip() and any(m.endswith('.o') for m in members.split())
    csdfg(a=a, b=b, c=c)
    assert np.allclose(c, 2.0 * a + b)


def test_cmake_no_static_archive_by_default(tmp_path):

    @dace.program
    def axpy_noar_cm(a: dace.float64[16], b: dace.float64[16], c: dace.float64[16]):
        c[:] = a + b

    sdfg = axpy_noar_cm.to_sdfg()
    sdfg.build_folder = str(tmp_path / 'cache')
    csdfg = sdfg.compile()
    assert not os.path.isfile(archive_path(str(csdfg._lib._library_filename)))
