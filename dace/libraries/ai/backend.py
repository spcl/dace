# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Provider-independent interface to the language model: the response model, its parser, and provider selection."""

import hashlib
import json
import os
from typing import Any, Literal, Protocol

import pydantic

from dace.config import Config
from dace.libraries.ai.exceptions import AIExpansionError

#: Environment variable names used when ``ai.api_key_envvar`` is empty.
DEFAULT_KEY_ENVVARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}

#: Providers accepted by ``ai.provider``.
SUPPORTED_PROVIDERS = ("anthropic", "openai", "manual")


class CMakeVariable(pydantic.BaseModel):
    """A CMake variable set by an environment."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    value: str


class EnvironmentSpec(pydantic.BaseModel):
    """A DaCe library environment requested by the model (see ``dace.library.environment``)."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str = pydantic.Field(description='A short identifier for the environment, e.g. "FFTW".')
    headers: list[str] = []
    cmake_packages: list[str] = []
    cmake_libraries: list[str] = []
    cmake_includes: list[str] = []
    cmake_compile_flags: list[str] = []
    cmake_link_flags: list[str] = []
    cmake_variables: list[CMakeVariable] = []
    cmake_minimum_version: str | None = None
    state_fields: list[str] = []
    init_code: str = ""
    finalize_code: str = ""


class TaskletSpec(pydantic.BaseModel):
    """A tasklet as described by the model."""

    model_config = pydantic.ConfigDict(extra="forbid")

    notes: str = pydantic.Field("", description="Why the code is written this way, and any assumption it relies on.")
    language: Literal["CPP", "Python"] = pydantic.Field(
        "CPP",
        description=(
            "Language of the tasklet body. Use CPP: the whole contract above -- code_global, "
            'include placement, state_fields, the ban on "return" -- describes a C++ tasklet, and '
            "anything but a bare Python assignment expression needs it. Choose Python only for a "
            "body that is a single assignment of a Python expression and needs none of those."
        ),
    )
    code: str = pydantic.Field(description="The tasklet body. Never empty.")
    code_global: str = pydantic.Field(
        "", description="File-scope code: includes, helper functions, type definitions. May be empty."
    )
    code_init: str = pydantic.Field(
        "", description="Host code run once at program initialization. __state is in scope. May be empty."
    )
    code_exit: str = pydantic.Field(
        "", description="Host code run once at program finalization. __state is in scope. May be empty."
    )
    state_fields: list[str] = pydantic.Field(
        [], description='Raw C++ member declarations added to the program state struct, e.g. "fftw_plan p0;".'
    )
    side_effects: bool = pydantic.Field(
        False, description="True if the code performs I/O or mutates global state and must not be eliminated."
    )
    ignored_symbols: list[str] = pydantic.Field(
        [], description="Identifiers in the code that coincide with SDFG symbol names but are local to the code."
    )
    use_environments: list[str] = pydantic.Field(
        [],
        description=(
            "Environments that already exist on this machine, by the exact class path listed in the "
            'context, e.g. "dace.libraries.blas.environments.openblas.OpenBLAS". Prefer this over '
            'describing the same library again in "environments".'
        ),
    )
    environments: list[EnvironmentSpec] = pydantic.Field(
        [],
        description=(
            "External libraries the code needs and that are NOT already listed in the context. Empty "
            "unless a new dependency is truly required."
        ),
    )

    #: The provider's answer as it arrived, for the session record.
    _raw: str = pydantic.PrivateAttr("")

    @pydantic.field_validator("code")
    @classmethod
    def _code_not_empty(cls, code: str) -> str:
        # An empty body would also drop the global code and state fields from the generated program
        if not code.strip():
            raise ValueError("the tasklet body is empty")
        return code

    @property
    def raw_response(self) -> str:
        """The text this specification was parsed from, or an empty string if it was built directly."""
        return self._raw


def _strict(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """
    Rewrites a pydantic JSON schema into the form structured output accepts.

    :param schema: A (sub)schema.
    :param defs: The ``$defs`` of the root schema, inlined in place of references.
    :return: The schema with references inlined, every property required, and no defaults or titles.
    """
    if "$ref" in schema:
        return _strict(defs[schema["$ref"].rsplit("/", 1)[-1]], defs)
    result = {key: value for key, value in schema.items() if key not in ("default", "title", "$defs")}
    if "properties" in result:
        result["properties"] = {name: _strict(sub, defs) for name, sub in result["properties"].items()}
        result["required"] = list(result["properties"])
        result["additionalProperties"] = False
    for key in ("items", "anyOf"):
        if isinstance(result.get(key), dict):
            result[key] = _strict(result[key], defs)
        elif isinstance(result.get(key), list):
            result[key] = [_strict(sub, defs) for sub in result[key]]
    return result


def response_schema() -> dict[str, Any]:
    """
    :return: The JSON schema of :class:`TaskletSpec` in strict structured-output form.
    """
    schema = TaskletSpec.model_json_schema()
    return _strict(schema, schema.get("$defs", {}))


#: The schema sent to providers and shown to the user by the manual provider.
RESPONSE_SCHEMA: dict[str, Any] = response_schema()

#: Identifies the schema in cache keys, so that changing the model invalidates cached answers.
SCHEMA_HASH = hashlib.sha256(json.dumps(RESPONSE_SCHEMA, sort_keys=True).encode()).hexdigest()


def parse_response(text: str, described: str = "the node") -> TaskletSpec:
    """
    Parses and validates a model answer.

    :param text: The answer, a single JSON object matching :data:`RESPONSE_SCHEMA`.
    :param described: What the answer is for, used in error messages.
    :return: The tasklet specification, with ``raw_response`` set to ``text``.
    :raises AIExpansionError: If the answer is not valid JSON or does not match the schema.
    """
    text = text.strip()
    if not text:
        raise AIExpansionError(f"The answer for {described} is empty.")
    try:
        spec = TaskletSpec.model_validate_json(text)
    except pydantic.ValidationError as e:
        raise AIExpansionError(
            f"The answer for {described} does not match the response schema:\n{e}\n"
            "It must be a single JSON object matching the schema at the end of the prompt."
        ) from e
    spec._raw = text
    return spec


class LLMProvider(Protocol):
    """Minimal interface a language model backend must implement."""

    def generate(self, system: str, messages: list[dict[str, str]]) -> TaskletSpec:
        """
        Requests a tasklet from the model.

        :param system: The system prompt.
        :param messages: The conversation so far, as ``{'role': ..., 'content': ...}`` dicts.
        :return: The tasklet described by the model.
        """
        ...


def api_key(provider: str) -> str | None:
    """
    Reads the API key for a provider from the configured environment variable.

    :param provider: The configured provider name.
    :return: The key, or ``None`` if the variable is unset or empty.
    """
    envvar = Config.get("ai", "api_key_envvar") or DEFAULT_KEY_ENVVARS.get(provider, "")
    if not envvar:
        return None
    return os.environ.get(envvar) or None


def client_options(provider: str) -> dict[str, Any]:
    """
    Returns the constructor arguments shared by the SDK clients.

    Options that are not configured are left out, so that the SDK applies its own defaults and
    environment variables (e.g. ``ANTHROPIC_BASE_URL``).

    :param provider: The configured provider name.
    :return: Keyword arguments for the client constructor.
    """
    options: dict[str, Any] = {"timeout": float(Config.get("ai", "timeout"))}
    key = api_key(provider)
    if key:
        options["api_key"] = key
    base_url = Config.get("ai", "base_url")
    if base_url:
        options["base_url"] = base_url
    return options


def missing_sdk_error(provider: str, package: str, extra: str, exc: BaseException) -> AIExpansionError:
    """
    Builds the error raised when a provider's SDK is not installed.

    :param provider: The configured provider name.
    :param package: The name of the missing Python package.
    :param extra: The DaCe optional-dependency extra that installs it.
    :param exc: The original :class:`ImportError`.
    :return: An error with installation instructions.
    """
    error = AIExpansionError(
        f"The '{provider}' provider needs the {package} Python package, which is not installed.\n"
        f"    pip install 'dace[{extra}]'   (in a DaCe checkout: uv sync --extra {extra})\n"
        f"Configured provider: {provider} (change it with DACE_ai_provider)."
    )
    error.__cause__ = exc
    return error


def missing_credentials_error(provider: str, detail: str | None = None) -> AIExpansionError:
    """
    Builds the error raised when a provider has no usable credentials.

    :param provider: The configured provider name.
    :param detail: What the provider's SDK reported, if anything.
    :return: An error naming the environment variable that is consulted.
    """
    envvar = Config.get("ai", "api_key_envvar") or DEFAULT_KEY_ENVVARS.get(provider, "(unset)")
    message = (
        f"The '{provider}' provider has no usable credentials.\n"
        f"    export {envvar}=...   (the variable name comes from DACE_ai_api_key_envvar)"
    )
    if detail:
        message += f"\nUnderlying error: {detail}"
    return AIExpansionError(message)


def request_settings() -> dict[str, Any]:
    """
    :return: Every setting that changes a provider's answer besides the conversation itself, for cache keys.
    """
    return {
        "provider": Config.get("ai", "provider"),
        "base_url": Config.get("ai", "base_url"),
        "model": Config.get("ai", "model"),
        "effort": Config.get("ai", "effort"),
        "max_tokens": int(Config.get("ai", "max_tokens")),
        "schema": SCHEMA_HASH,
    }


def get_provider() -> LLMProvider:
    """
    Instantiates the provider selected by the ``ai.provider`` configuration entry.

    :return: A ready-to-use provider.
    :raises AIExpansionError: If the provider is unknown or cannot be set up.
    """
    # Imported here: the provider modules import this one
    provider = (Config.get("ai", "provider") or "anthropic").strip().lower()
    if provider == "anthropic":
        from dace.libraries.ai.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if provider == "openai":
        from dace.libraries.ai.providers.openai_provider import OpenAIProvider

        return OpenAIProvider()
    if provider == "manual":
        from dace.libraries.ai.providers.manual_provider import ManualProvider

        return ManualProvider()
    raise AIExpansionError(
        f"Unknown AI provider '{provider}'. Supported providers are "
        f"{', '.join(SUPPORTED_PROVIDERS)} (set DACE_ai_provider)."
    )
