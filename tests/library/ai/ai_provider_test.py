# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for provider selection, configuration and failure messages. """

import builtins
import contextlib

import pytest

import dace
from dace.libraries.ai import backend
from dace.libraries.ai.exceptions import AIExpansionError


@contextlib.contextmanager
def _hide_module(name: str):
    """
    Makes importing a module fail, as it would on a machine without the SDK installed.

    :param name: The module to hide.
    """
    real_import = builtins.__import__

    def fake_import(module, *args, **kwargs):
        if module == name or module.startswith(name + '.'):
            raise ImportError(f'No module named {name!r}')
        return real_import(module, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        yield
    finally:
        builtins.__import__ = real_import


def test_missing_sdk_points_at_the_extra():
    with dace.config.set_temporary('ai', 'provider', value='anthropic'):
        with _hide_module('anthropic'):
            with pytest.raises(AIExpansionError) as info:
                backend.get_provider()

    message = str(info.value)
    assert 'anthropic' in message
    assert "dace[ai-anthropic]" in message
    assert 'DACE_ai_provider' in message


def test_missing_responses_sdk_points_at_its_own_extra():
    with dace.config.set_temporary('ai', 'provider', value='responses'):
        with _hide_module('openai'):
            with pytest.raises(AIExpansionError) as info:
                backend.get_provider()

    assert 'openai' in str(info.value)
    assert 'dace[ai-openai]' in str(info.value)


def test_unknown_provider_is_rejected():
    with dace.config.set_temporary('ai', 'provider', value='nonexistent'):
        with pytest.raises(AIExpansionError, match='Unknown AI provider') as info:
            backend.get_provider()

    # The message lists what the user could have meant
    for provider in backend.SUPPORTED_PROVIDERS:
        assert provider in str(info.value)


def test_manual_provider_needs_no_sdk_and_no_key():
    # It is the one backend that works with a chat subscription instead of API credits
    with _hide_module('anthropic'):
        with _hide_module('openai'):
            with dace.config.set_temporary('ai', 'provider', value='manual'):
                with dace.config.set_temporary('ai', 'api_key_envvar', value='DACE_NO_SUCH_KEY'):
                    provider = backend.get_provider()

    assert type(provider).__name__ == 'ManualProvider'


def test_credential_error_names_the_configured_variable():
    error = backend.missing_credentials_error('anthropic', RuntimeError('no key'))
    assert 'ANTHROPIC_API_KEY' in str(error)

    with dace.config.set_temporary('ai', 'api_key_envvar', value='MY_PROJECT_KEY'):
        error = backend.missing_credentials_error('anthropic', RuntimeError('no key'))
    assert 'MY_PROJECT_KEY' in str(error)
    assert 'DACE_ai_api_key_envvar' in str(error)


def test_api_key_is_read_from_the_configured_variable(monkeypatch):
    monkeypatch.setenv('DACE_TEST_AI_KEY', 'secret-value')
    with dace.config.set_temporary('ai', 'api_key_envvar', value='DACE_TEST_AI_KEY'):
        assert backend.api_key('anthropic') == 'secret-value'

    monkeypatch.delenv('DACE_TEST_AI_KEY')
    with dace.config.set_temporary('ai', 'api_key_envvar', value='DACE_TEST_AI_KEY'):
        assert backend.api_key('anthropic') is None


def test_response_schema_is_strict_mode_shaped():
    schema = backend.RESPONSE_SCHEMA

    # Structured output requires "additionalProperties: false" and an exhaustive "required" list at
    # every level; a missing entry is rejected by the provider rather than silently ignored
    assert schema['additionalProperties'] is False
    assert set(schema['required']) == set(schema['properties'])

    item = schema['properties']['environments']['items']
    assert item['additionalProperties'] is False
    assert set(item['required']) == set(item['properties'])


def test_empty_response_is_rejected():
    with pytest.raises(AIExpansionError, match='empty tasklet body'):
        backend.spec_from_dict({'code': '   '})


def test_spec_round_trips_every_field():
    spec = backend.spec_from_dict({
        'code':
        '_out = _in;',
        'language':
        'CPP',
        'code_global':
        '#include <cmath>',
        'code_init':
        'int x = 0;',
        'code_exit':
        '// done',
        'state_fields': ['int counter;'],
        'side_effects':
        True,
        'ignored_symbols': ['i'],
        'notes':
        'why',
        'use_environments': ['dace.libraries.blas.environments.openblas.OpenBLAS'],
        'environments': [{
            'name': 'FFTW',
            'headers': ['fftw3.h'],
            'cmake_libraries': ['fftw3'],
        }],
    })

    assert spec.code_global == '#include <cmath>'
    assert spec.state_fields == ['int counter;']
    assert spec.side_effects is True
    assert spec.ignored_symbols == ['i']
    assert spec.use_environments == ['dace.libraries.blas.environments.openblas.OpenBLAS']
    assert len(spec.environments) == 1
    assert spec.environments[0].headers == ['fftw3.h']
    assert spec.environments[0].cmake_packages == []


if __name__ == '__main__':
    test_missing_sdk_points_at_the_extra()
    test_missing_responses_sdk_points_at_its_own_extra()
    test_unknown_provider_is_rejected()
    test_credential_error_names_the_configured_variable()
    test_empty_response_is_rejected()
    test_spec_round_trips_every_field()
