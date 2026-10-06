# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CMake build: the runnable library and its stub, and the opt-in static archive beside it."""
import os
import subprocess

import numpy as np

import dace
from dace import dtypes, library
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


def library_environment(name: str, libraries):
    fields = dict(cmake_minimum_version=None,
                  cmake_packages=[],
                  cmake_variables={},
                  cmake_includes=[],
                  cmake_libraries=list(libraries),
                  cmake_compile_flags=[],
                  cmake_link_flags=[],
                  cmake_files=[],
                  headers=[],
                  state_fields=[],
                  init_code='',
                  finalize_code='',
                  dependencies=[],
                  __module__=__name__)
    return library.environment(type(name, (), fields))


def test_an_environment_links_its_libraries_in_the_order_it_lists_them(tmp_path):
    """A static archive listed before the shared library it needs must stay before it: sorted, ``-ldep``
    came first, ``--as-needed`` dropped it, and the link failed on the archive's undefined reference."""
    dep, archive = tmp_path / 'dep', tmp_path / 'libk.a'
    dep.mkdir()
    (dep / 'dep.c').write_text('double dep_factor(void) { return 3.0; }\n')
    subprocess.run(['gcc', '-fPIC', '-shared', str(dep / 'dep.c'), '-o', str(dep / 'libdep.so')], check=True)
    (tmp_path /
     'k.c').write_text('double dep_factor(void);\ndouble kernel_value(void) { return 2.0 * dep_factor(); }\n')
    subprocess.run(['gcc', '-fPIC', '-c', str(tmp_path / 'k.c'), '-o', str(tmp_path / 'k.o')], check=True)
    subprocess.run(['ar', 'rcs', str(archive), str(tmp_path / 'k.o')], check=True)
    env = library_environment('OrderedLinkEnv', [str(archive), f'-L{dep}', '-ldep', f'-Wl,-rpath,{dep}'])
    sdfg = dace.SDFG('ordered_link')
    sdfg.add_array('out', [1], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('call', {}, {'o'},
                                'o = kernel_value();',
                                language=dtypes.Language.CPP,
                                code_global='extern "C" double kernel_value();')
    tasklet.environments = {env.full_class_path()}
    state.add_edge(tasklet, 'o', state.add_write('out'), None, dace.Memlet('out[0]'))
    sdfg.build_folder = str(tmp_path / 'cache')
    out = np.zeros(1)

    sdfg(out=out)

    assert out[0] == 6.0
