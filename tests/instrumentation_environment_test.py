# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Instrumentation must declare its build requirements, not append them to the global config.

``Config.append('compiler', 'cpu', 'args', ...)`` mutates process-wide state with no restore, and the
providers did it from ``on_sdfg_begin`` -- once per SDFG. So the flags accumulated without bound and
leaked into every later SDFG in the session, instrumented or not. DaCe already has the right
mechanism: an environment's ``cmake_compile_flags``/``cmake_libraries`` are a deduplicated set scoped
to the SDFGs that use it.

These tests need neither LIKWID nor PAPI installed, which matters because CI has neither -- the
regression they guard would otherwise be invisible until a user hit it.
"""
import ast
import pathlib

import pytest

from dace.codegen.instrumentation.likwid import LIKWID, LIKWIDNvmon, LIKWIDPerfmon
from dace.codegen.instrumentation.papi import PAPI
from dace.codegen.compiler import get_environment_flags
from dace.config import Config, set_temporary
from dace.library import get_environments_and_dependencies

#: Instrumentation providers that used to append to the global config.
PROVIDER_SOURCES = ('dace/codegen/instrumentation/likwid.py', 'dace/codegen/instrumentation/papi.py')

#: Config keys whose global mutation is the bug under test. ``libs`` is included because PAPI
#: appended ``papi`` there for the same reason it appended flags.
FORBIDDEN_KEYS = (('compiler', 'cpu', 'args'), ('compiler', 'cpu', 'libs'))


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def config_append_targets(source: pathlib.Path):
    """Every ``Config.append(...)`` key tuple in ``source``, as tuples of literal strings."""
    tree = ast.parse(source.read_text())
    targets = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != 'append' or not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != 'Config':
            continue
        targets.append(tuple(a.value for a in node.args if isinstance(a, ast.Constant)))
    return targets


@pytest.mark.parametrize('relative', PROVIDER_SOURCES)
def test_providers_do_not_mutate_global_compiler_config(relative):
    """The regression guard: a provider must not reach for ``Config.append`` again.

    Asserted against the source rather than by running the providers, because doing it the other way
    needs LIKWID and PAPI installed -- and on a machine without them the provider returns early, so a
    reintroduced append would go unnoticed exactly where CI runs.
    """
    source = repo_root() / relative
    assert source.is_file(), f'{relative} not found; update PROVIDER_SOURCES'
    offenders = [t for t in config_append_targets(source) if t in FORBIDDEN_KEYS]
    assert not offenders, (f'{relative} appends to global config {offenders}. Declare the flags on the provider\'s '
                           'library environment instead -- a global append leaks into every later SDFG in the '
                           'process and, from on_sdfg_begin, accumulates once per SDFG.')


def test_likwid_environments_carry_their_defines():
    """Each marker API is its own environment: the CPU and GPU providers need different defines."""
    assert LIKWIDPerfmon.cmake_compile_flags == ['-DLIKWID_PERFMON']
    assert LIKWIDNvmon.cmake_compile_flags == ['-DLIKWID_NVMON']
    assert LIKWID in LIKWIDPerfmon.dependencies and LIKWID in LIKWIDNvmon.dependencies
    # Defining both would activate a marker API whose initialization the generated code never emits.
    assert LIKWIDPerfmon.cmake_compile_flags != LIKWIDNvmon.cmake_compile_flags


def test_likwid_does_not_re_add_openmp():
    """``CMakeLists.txt`` already does ``find_package(OpenMP REQUIRED)`` and links OpenMP::OpenMP_CXX,
    so the provider adding ``-fopenmp`` only duplicated it on every compile line."""
    for env in (LIKWID, LIKWIDPerfmon, LIKWIDNvmon):
        assert '-fopenmp' not in env.cmake_compile_flags


def test_papi_environment_declares_the_library():
    assert 'papi' in PAPI.cmake_libraries


def test_papi_declares_no_python_chosen_compile_flags():
    """The vectorization flag is not a Python-side string any more; it is a CMake fragment."""
    assert PAPI.cmake_compile_flags == []


@pytest.mark.parametrize('enabled', [True, False])
def test_papi_vectorization_fragment_follows_the_config(enabled):
    """Evaluated per build, so toggling the setting is honoured rather than baked in at the first
    instrumented SDFG the way the previous one-shot global append was."""
    with set_temporary('instrumentation', 'papi', 'vectorization_analysis', value=enabled):
        assert PAPI.cmake_files() == (['papi_vectorization'] if enabled else [])


def test_papi_vectorization_fragment_is_compiler_aware():
    """The report flag differs per compiler, and CMake is what picks the compiler.

    Choosing in CMake from CMAKE_CXX_COMPILER_ID means there is no second detector in Python that
    has to predict what CMake is about to pick -- the failure mode where flags chosen for one
    compiler are handed to another. Asserted on the fragment's text so it holds on a machine that
    has only one of these compilers installed, which is every CI runner.
    """
    fragment = pathlib.Path(__file__).resolve().parents[1] / 'dace' / 'codegen' / 'instrumentation'
    text = (fragment / 'papi_vectorization.cmake').read_text()
    for compiler_id, flag in (('GNU', '-fopt-info-vec'), ('Clang', '-Rpass=loop-vectorize'), ('NVHPC', '-Minfo=vect'),
                              ('IntelLLVM', '-qopt-report')):
        assert compiler_id in text and flag in text, f'no {compiler_id} spelling in papi_vectorization.cmake'
    # Placed via add_compile_options so the build type's flags cannot override them.
    assert 'add_compile_options' in text


def test_papi_vectorization_fragment_resolves_to_a_real_file():
    """``cmake_files`` entries are resolved against the environment's module and get ``.cmake``
    appended; a name that does not resolve fails at configure time, not here."""
    with set_temporary('instrumentation', 'papi', 'vectorization_analysis', value=True):
        envs = get_environments_and_dependencies({PAPI.full_class_path()})
        flags, _link = get_environment_flags(envs)
    declared = [f for f in flags if 'DACE_ENV_CMAKE_FILES' in f]
    assert declared, 'the environment declared no cmake files'
    paths = declared[0].split('=', 1)[1].strip('"').split(';')
    assert any(pathlib.Path(p).is_file() for p in paths if p), f'declared cmake file does not exist: {paths}'


def test_compiler_args_survive_importing_the_providers():
    """Import alone must not have moved the global compiler flags."""
    before = Config.get('compiler', 'cpu', 'args')
    import dace.codegen.instrumentation.likwid  # noqa: F401
    import dace.codegen.instrumentation.papi  # noqa: F401
    assert Config.get('compiler', 'cpu', 'args') == before


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
