# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for provider selection, configuration and failure messages."""

import builtins
import contextlib
import json

import pytest

import dace
from dace.libraries.ai import backend
from dace.libraries.ai import cache as ai_cache
from dace.libraries.ai.exceptions import AIExpansionError
from tests.library.ai.ai_test_utils import offline


@contextlib.contextmanager
def _hide_module(name: str):
    """
    Makes importing a module fail, as it would on a machine without the SDK installed.

    :param name: The module to hide.
    """
    real_import = builtins.__import__

    def fake_import(module, *args, **kwargs):
        if module == name or module.startswith(name + "."):
            raise ImportError(f"No module named {name!r}")
        return real_import(module, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        yield
    finally:
        builtins.__import__ = real_import


def test_missing_sdk_points_at_the_extra():
    with dace.config.set_temporary("ai", "provider", value="anthropic"):
        with _hide_module("anthropic"):
            with pytest.raises(AIExpansionError) as info:
                backend.get_provider()

    message = str(info.value)
    assert "anthropic" in message
    assert "dace[agent-anthropic]" in message
    assert "DACE_ai_provider" in message


def test_missing_openai_sdk_points_at_its_own_extra():
    with dace.config.set_temporary("ai", "provider", value="openai"):
        with _hide_module("openai"):
            with pytest.raises(AIExpansionError) as info:
                backend.get_provider()

    assert "openai" in str(info.value)
    assert "dace[agent-openai]" in str(info.value)


def test_unknown_provider_is_rejected():
    with dace.config.set_temporary("ai", "provider", value="nonexistent"):
        with pytest.raises(AIExpansionError, match="Unknown AI provider") as info:
            backend.get_provider()

    # The message lists what the user could have meant
    for provider in backend.SUPPORTED_PROVIDERS:
        assert provider in str(info.value)


def test_manual_provider_needs_no_sdk_and_no_key():
    # It is the one backend that works with a chat subscription instead of API credits
    with _hide_module("anthropic"):
        with _hide_module("openai"):
            with dace.config.set_temporary("ai", "provider", value="manual"):
                with dace.config.set_temporary("ai", "api_key_envvar", value="DACE_NO_SUCH_KEY"):
                    provider = backend.get_provider()

    assert type(provider).__name__ == "ManualProvider"


def test_credential_error_names_the_configured_variable():
    error = backend.missing_credentials_error("anthropic", RuntimeError("no key"))
    assert "ANTHROPIC_API_KEY" in str(error)

    with dace.config.set_temporary("ai", "api_key_envvar", value="MY_PROJECT_KEY"):
        error = backend.missing_credentials_error("anthropic", RuntimeError("no key"))
    assert "MY_PROJECT_KEY" in str(error)
    assert "DACE_ai_api_key_envvar" in str(error)


def test_api_key_is_read_from_the_configured_variable(monkeypatch):
    monkeypatch.setenv("DACE_TEST_AI_KEY", "secret-value")
    with dace.config.set_temporary("ai", "api_key_envvar", value="DACE_TEST_AI_KEY"):
        assert backend.api_key("anthropic") == "secret-value"

    monkeypatch.delenv("DACE_TEST_AI_KEY")
    with dace.config.set_temporary("ai", "api_key_envvar", value="DACE_TEST_AI_KEY"):
        assert backend.api_key("anthropic") is None


def test_response_schema_is_strict_mode_shaped():
    schema = backend.RESPONSE_SCHEMA

    # Structured output requires "additionalProperties: false" and an exhaustive "required" list at
    # every level, and no references or defaults
    def check(sub):
        if "properties" in sub:
            assert sub["additionalProperties"] is False
            assert set(sub["required"]) == set(sub["properties"])
            for prop in sub["properties"].values():
                check(prop)
        if "items" in sub:
            check(sub["items"])
        assert "$ref" not in sub and "default" not in sub

    check(schema)
    environment = schema["properties"]["environments"]["items"]
    assert "cmake_variables" in environment["properties"]


def test_empty_response_is_rejected():
    with pytest.raises(AIExpansionError, match="tasklet body is empty"):
        backend.parse_response(json.dumps({"code": "   "}))


def test_unknown_field_is_rejected():
    with pytest.raises(AIExpansionError, match="does not match the response schema"):
        backend.parse_response(json.dumps({"code": "_out = _in;", "codee": ""}))


def test_spec_round_trips_every_field():
    text = json.dumps(
        {
            "code": "_out = _in;",
            "language": "CPP",
            "code_global": "#include <cmath>",
            "code_init": "int x = 0;",
            "code_exit": "// done",
            "state_fields": ["int counter;"],
            "side_effects": True,
            "ignored_symbols": ["i"],
            "notes": "why",
            "use_environments": ["dace.libraries.blas.environments.openblas.OpenBLAS"],
            "environments": [
                {
                    "name": "FFTW",
                    "headers": ["fftw3.h"],
                    "cmake_libraries": ["fftw3"],
                    "cmake_variables": [{"name": "FFTW_ROOT", "value": "/opt/fftw"}],
                }
            ],
        }
    )
    spec = backend.parse_response(text)

    assert spec.raw_response == text
    assert spec.code_global == "#include <cmath>"
    assert spec.state_fields == ["int counter;"]
    assert spec.side_effects is True
    assert spec.ignored_symbols == ["i"]
    assert spec.use_environments == ["dace.libraries.blas.environments.openblas.OpenBLAS"]
    assert len(spec.environments) == 1
    assert spec.environments[0].headers == ["fftw3.h"]
    assert spec.environments[0].cmake_packages == []
    assert spec.environments[0].cmake_variables[0].value == "/opt/fftw"
    assert backend.parse_response(spec.model_dump_json()).model_dump() == spec.model_dump()


def test_client_options_pass_the_base_url(monkeypatch):
    monkeypatch.setenv("DACE_TEST_AI_KEY", "secret-value")
    with dace.config.set_temporary("ai", "api_key_envvar", value="DACE_TEST_AI_KEY"):
        with dace.config.set_temporary("ai", "base_url", value="http://localhost:8080/v1"):
            options = backend.client_options("openai")
        assert "base_url" not in backend.client_options("openai")

    assert options["base_url"] == "http://localhost:8080/v1"
    assert options["api_key"] == "secret-value"


def test_cache_key_depends_on_request_settings():
    messages = [{"role": "user", "content": "prompt"}]
    base = ai_cache.key("system", messages)
    with dace.config.set_temporary("ai", "max_tokens", value=123):
        assert ai_cache.key("system", messages) != base
    with dace.config.set_temporary("ai", "base_url", value="http://localhost:8080/v1"):
        assert ai_cache.key("system", messages) != base
    assert ai_cache.key("system", messages) == base


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_sdk_clients_use_the_base_url(provider, monkeypatch):
    pytest.importorskip(provider)
    monkeypatch.setenv("DACE_TEST_AI_KEY", "secret-value")
    with dace.config.set_temporary("ai", "provider", value=provider):
        with dace.config.set_temporary("ai", "api_key_envvar", value="DACE_TEST_AI_KEY"):
            with dace.config.set_temporary("ai", "base_url", value="http://localhost:8080/v1"):
                client = backend.get_provider()._client
    assert str(client.base_url).rstrip("/") == "http://localhost:8080/v1"


def test_missing_openai_key_is_a_credentials_error(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DACE_NO_SUCH_KEY", raising=False)
    with dace.config.set_temporary("ai", "provider", value="openai"):
        with dace.config.set_temporary("ai", "api_key_envvar", value="DACE_NO_SUCH_KEY"):
            with pytest.raises(AIExpansionError, match="no usable credentials"):
                backend.get_provider()


if __name__ == "__main__":
    with offline():
        test_missing_sdk_points_at_the_extra()
        test_missing_openai_sdk_points_at_its_own_extra()
        test_unknown_provider_is_rejected()
        test_manual_provider_needs_no_sdk_and_no_key()
        test_credential_error_names_the_configured_variable()
        test_response_schema_is_strict_mode_shaped()
        test_empty_response_is_rejected()
        test_unknown_field_is_rejected()
        test_spec_round_trips_every_field()
        test_cache_key_depends_on_request_settings()
