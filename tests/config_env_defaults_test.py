# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Environment variables are consumed when the configuration is loaded.

``DACE_<option>`` variables are applied at load time with the precedence
(increasing priority): schema default, configuration file, environment.
Values set explicitly afterwards (``Config.set``, ``set_temporary``,
``temporary_config``) have the highest priority, and changing the environment
after the configuration is loaded has no effect on ``get()``.
"""
import io
import warnings

import pytest

from dace.config import Config, set_temporary, temporary_config


def _reload_from(yaml_text: str = ''):
    """Reload the in-memory configuration from the given yaml content."""
    Config.load(file=io.StringIO(yaml_text))


def test_env_seeds_default_on_load():
    with temporary_config():
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv('DACE_compiler_build_type', 'FromEnv')
            _reload_from()
            assert Config.get('compiler', 'build_type') == 'FromEnv'


def test_set_temporary_overrides_env_seeded_value():
    with temporary_config():
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv('DACE_compiler_build_type', 'FromEnv')
            _reload_from()
            with set_temporary('compiler', 'build_type', value='FromSet'):
                assert Config.get('compiler', 'build_type') == 'FromSet'
            assert Config.get('compiler', 'build_type') == 'FromEnv'


def test_temporary_config_overrides_env_seeded_value():
    with temporary_config():
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv('DACE_compiler_build_type', 'FromEnv')
            _reload_from()
            with temporary_config():
                Config.set('compiler', 'build_type', value='FromSet')
                assert Config.get('compiler', 'build_type') == 'FromSet'


def test_config_set_overrides_env_seeded_value():
    with temporary_config():
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv('DACE_compiler_build_type', 'FromEnv')
            _reload_from()
            Config.set('compiler', 'build_type', value='FromSet')
            assert Config.get('compiler', 'build_type') == 'FromSet'


def test_env_overrides_config_file():
    with temporary_config():
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv('DACE_compiler_build_type', 'FromEnv')
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                _reload_from('compiler:\n  build_type: FromFile\n')
            assert Config.get('compiler', 'build_type') == 'FromEnv'


def test_precedence_default_file_env_set():
    """Increasing priority: schema default, configuration file, environment, set()."""
    with temporary_config():
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv('DACE_compiler_build_type', raising=False)
            _reload_from('compiler:\n  build_type: FromFile\n')
            assert Config.get('compiler', 'build_type') == 'FromFile'
            mp.setenv('DACE_compiler_build_type', 'FromEnv')
            _reload_from('compiler:\n  build_type: FromFile\n')
            assert Config.get('compiler', 'build_type') == 'FromEnv'
            Config.set('compiler', 'build_type', value='FromSet')
            assert Config.get('compiler', 'build_type') == 'FromSet'


def test_env_change_after_load_is_inert():
    with temporary_config():
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv('DACE_compiler_build_type', raising=False)
            _reload_from()
            default = Config.get_default('compiler', 'build_type')
            assert Config.get('compiler', 'build_type') == default
            mp.setenv('DACE_compiler_build_type', 'FromEnv')
            assert Config.get('compiler', 'build_type') == default
            # A reload picks the new environment up.
            _reload_from()
            assert Config.get('compiler', 'build_type') == 'FromEnv'


def test_env_bool_coercion():
    with temporary_config():
        with pytest.MonkeyPatch.context() as mp:
            for raw, expected in [('1', True), ('true', True), ('on', True), ('0', False), ('false', False),
                                  ('off', False)]:
                mp.setenv('DACE_debugprint', raw)
                _reload_from()
                assert Config.get('debugprint') is expected, f'env value {raw!r}'
                assert Config.get_bool('debugprint') is expected, f'env value {raw!r}'


def test_env_int_coercion():
    with temporary_config():
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv('DACE_compiler_max_stack_array_size', '1024')
            _reload_from()
            assert Config.get('compiler', 'max_stack_array_size') == 1024


def test_env_bad_int_warns_and_falls_back_to_default():
    with temporary_config():
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv('DACE_compiler_max_stack_array_size', 'not_a_number')
            with pytest.warns(UserWarning, match='DACE_compiler_max_stack_array_size'):
                _reload_from()
            assert Config.get('compiler', 'max_stack_array_size') == \
                Config.get_default('compiler', 'max_stack_array_size')


def test_set_has_no_autosave():
    """``Config.set``/``append`` never write the configuration file: the
    ``autosave`` parameter is removed (nothing in the tree ever passed True)."""
    with temporary_config():
        with pytest.raises(TypeError):
            Config.set('debugprint', value=True, autosave=True)
        with pytest.raises(TypeError):
            Config.append('compiler', 'cpu', 'args', value=' -O2', autosave=True)


def test_legacy_config_file_is_migrated_on_load(tmp_path):
    """A config file in the old format (marked by the legacy ``execution``
    entry, which old DaCe versions wrote because they saved EVERY entry) is
    rewritten once, in the new nondefaults-only format. The environment has
    been applied by then, so ``DACE_*`` values set during the migration are
    persisted into the file as well - documented behavior."""
    from dace.config import _ConfigData

    cfg = tmp_path / 'legacy.conf'
    cfg.write_text('execution: {}\ndebugprint: true\n')
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('DACE_CONFIG', str(cfg))
        mp.setenv('DACE_profiling', '1')
        mp.chdir(tmp_path)
        _ConfigData()  # A fresh store runs _initialize -> load -> migration.
    migrated = cfg.read_text()
    assert 'execution' not in migrated  # The legacy marker is gone...
    assert 'debugprint: true' in migrated  # ...nondefaults are kept...
    assert 'profiling: true' in migrated  # ...and env-set values are persisted.


def test_load_does_not_rewrite_a_modern_config_file(tmp_path):
    """A file already in the new format is never written back: neither the
    environment nor anything else leaks into the user's file on load."""
    from dace.config import _ConfigData

    cfg = tmp_path / 'modern.conf'
    cfg.write_text('debugprint: true\n')
    before = cfg.read_text()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('DACE_CONFIG', str(cfg))
        mp.setenv('DACE_profiling', '1')
        mp.chdir(tmp_path)
        store = _ConfigData()
        assert store.get('debugprint') is True  # The file was read...
        assert store.get('profiling') is True  # ...the environment applied,
    assert cfg.read_text() == before  # ...and nothing wrote the file back.


if __name__ == '__main__':
    test_env_seeds_default_on_load()
    test_set_temporary_overrides_env_seeded_value()
    test_temporary_config_overrides_env_seeded_value()
    test_config_set_overrides_env_seeded_value()
    test_env_overrides_config_file()
    test_precedence_default_file_env_set()
    test_env_change_after_load_is_inert()
    test_env_bool_coercion()
    test_env_int_coercion()
    test_env_bad_int_warns_and_falls_back_to_default()
