# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for library environments requested by the model and materialized on disk. """

import os

import pytest

import dace
import dace.library
from dace.libraries.ai import environments as ai_environments
from dace.libraries.ai.backend import EnvironmentSpec, TaskletSpec
from dace.libraries.ai.exceptions import AIExpansionError

SPEC = EnvironmentSpec(name='TestFFT',
                       headers=['fftw3.h'],
                       cmake_packages=['PkgConfig'],
                       cmake_libraries=['fftw3'],
                       cmake_compile_flags=['-DTEST_FFT=1'],
                       cmake_minimum_version='3.15',
                       state_fields=['int test_fft_plan;'],
                       init_code='// set up',
                       finalize_code='// tear down')


def test_render_produces_every_required_field():
    source = ai_environments.render_environment(SPEC)

    # dace.library.environment enforces all thirteen fields
    for field in ('cmake_minimum_version', 'cmake_packages', 'cmake_variables', 'cmake_includes', 'cmake_libraries',
                  'cmake_compile_flags', 'cmake_link_flags', 'cmake_files', 'headers', 'state_fields', 'init_code',
                  'finalize_code', 'dependencies'):
        assert f'    {field} = ' in source, f'{field} is missing from the generated module'
    assert 'class TestFFT:' in source
    assert "'fftw3.h'" in source


def test_materialize_writes_and_registers(tmp_path):
    with dace.config.set_temporary('ai', 'environment_dir', value=str(tmp_path)):
        env = ai_environments.materialize(SPEC)

        path = tmp_path / 'dace_ai_env_testfft.py'
        assert path.exists(), 'the environment was not written to disk'

        assert env._dace_library_environment
        assert env.cmake_libraries == ['fftw3']
        assert env.state_fields == ['int test_fft_plan;']
        # Registered under its full class path, which is what expansions store on the node
        assert dace.library.get_environment(env.full_class_path()) is env
        # The path is resolvable, which is what @dace.library.environment needs it for
        assert os.path.exists(env._dace_file_path)


def test_generated_environments_reload(tmp_path):
    with dace.config.set_temporary('ai', 'environment_dir', value=str(tmp_path)):
        ai_environments.materialize(SPEC)
        # Forget the registration, as a fresh interpreter would
        registry = dace.library._DACE_REGISTERED_ENVIRONMENTS
        stale = [k for k in registry if k.startswith('dace_ai_env_testfft')]
        for key in stale:
            del registry[key]

        loaded = ai_environments.load_generated_environments()

    assert any(env.__name__ == 'TestFFT' for env in loaded)
    assert any(k.startswith('dace_ai_env_testfft') for k in dace.library._DACE_REGISTERED_ENVIRONMENTS)


def test_broken_module_is_skipped(tmp_path):
    (tmp_path / 'dace_ai_env_broken.py').write_text('this is not valid python !!!\n')
    with dace.config.set_temporary('ai', 'environment_dir', value=str(tmp_path)):
        # A hand-edited or truncated module must not make importing DaCe fail
        assert ai_environments.load_generated_environments() == []


def test_names_are_sanitized(tmp_path):
    spec = EnvironmentSpec(name='My Env-2!', headers=['x.h'])
    with dace.config.set_temporary('ai', 'environment_dir', value=str(tmp_path)):
        env = ai_environments.materialize(spec)
    assert env.__name__ == 'My_Env_2'
    assert (tmp_path / 'dace_ai_env_my_env_2.py').exists()


def test_environment_dir_is_not_created_by_reading(tmp_path):
    target = tmp_path / 'does_not_exist_yet'
    with dace.config.set_temporary('ai', 'environment_dir', value=str(target)):
        assert ai_environments.environment_dir() == str(target)
        assert not target.exists(), 'reading the environment directory must not create it'
        ai_environments.load_generated_environments()
        assert not target.exists()


def test_requested_environment_reaches_the_generated_tasklet(tmp_path):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from ai_test_utils import stub_provider

    from dace import nodes
    from dace.libraries.ai.backend import TaskletSpec
    from dace.libraries.ai.nodes import AINode

    sdfg = dace.SDFG('ai_env_flow')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    state = sdfg.add_state()
    node = AINode('needs_a_library', 'Copy the value.', inputs={'_in'}, outputs={'_out'})
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_in', dace.Memlet('A[0]'))
    state.add_edge(node, '_out', state.add_write('B'), None, dace.Memlet('B[0]'))

    spec = TaskletSpec(code='_out = _in;', environments=[SPEC])
    with dace.config.set_temporary('ai', 'environment_dir', value=str(tmp_path)):
        with dace.config.set_temporary('ai', 'verify', value=False):
            with stub_provider(spec):
                node.expand(state, 'ai')

    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    # ExpandTransformation.apply stores full class paths, which the code generator resolves later
    assert any(path.endswith('.TestFFT') for path in tasklet.environments), tasklet.environments
    for path in tasklet.environments:
        assert dace.library.get_environment(path).cmake_libraries == ['fftw3']


def test_an_existing_environment_can_be_named_instead_of_described(tmp_path):
    """ Reusing an installed environment must not write anything to disk. """
    with dace.config.set_temporary('ai', 'environment_dir', value=str(tmp_path)):
        env = ai_environments.materialize(SPEC)
    path = env.full_class_path()

    other = tmp_path / 'nothing_written_here'
    with dace.config.set_temporary('ai', 'environment_dir', value=str(other)):
        assert ai_environments.resolve_installed([path]) == [env]
        # Naming the same environment twice attaches it once
        spec = TaskletSpec(code='_out = _in;', use_environments=[path, path])
        assert ai_environments.collect(spec) == [env]
    assert not other.exists(), 'reusing an installed environment wrote a module'


def test_an_unknown_environment_reference_is_reported():
    with pytest.raises(AIExpansionError, match='not registered') as info:
        ai_environments.resolve_installed(['dace.libraries.nowhere.NoSuchEnvironment'])

    # The message must name what was asked for, so a repair or a human knows what to correct
    assert 'NoSuchEnvironment' in str(info.value)


def test_reused_environments_reach_the_tasklet(tmp_path):
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from ai_test_utils import stub_provider

    from dace import nodes
    from dace.libraries.ai.nodes import AINode

    with dace.config.set_temporary('ai', 'environment_dir', value=str(tmp_path)):
        env = ai_environments.materialize(SPEC)

    sdfg = dace.SDFG('ai_env_reuse')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    state = sdfg.add_state()
    node = AINode('reuses_a_library', 'Copy the value.', inputs={'_in'}, outputs={'_out'})
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_in', dace.Memlet('A[0]'))
    state.add_edge(node, '_out', state.add_write('B'), None, dace.Memlet('B[0]'))

    spec = TaskletSpec(code='_out = _in;', use_environments=[env.full_class_path()])
    with dace.config.set_temporary('ai', 'verify', value=False):
        with stub_provider(spec):
            node.expand(state, 'ai')

    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert list(tasklet.environments) == [env.full_class_path()]


if __name__ == '__main__':
    test_render_produces_every_required_field()
    pytest.main([__file__])
