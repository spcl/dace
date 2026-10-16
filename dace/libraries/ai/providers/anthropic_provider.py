# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Anthropic Messages API provider."""

from dace.config import Config
from dace.libraries.ai import backend
from dace.libraries.ai.exceptions import AIExpansionError


class AnthropicProvider:
    """Generates tasklets through the Anthropic Messages API."""

    def __init__(self) -> None:
        try:
            import anthropic  # Optional dependency (dace[agent-anthropic])
        except ImportError as e:
            raise backend.missing_sdk_error("anthropic", "anthropic", "agent-anthropic", e)

        self._anthropic = anthropic
        try:
            self._client = anthropic.Anthropic(**backend.client_options("anthropic"))
        except anthropic.AnthropicError as e:
            raise AIExpansionError(f"Could not create the Anthropic client: {e}") from e

        # Without a key the SDK falls back to its own credential chain, and only reports that the
        # chain is empty once a request is made
        if not (self._client.api_key or self._client.auth_token or self._client.credentials):
            raise backend.missing_credentials_error(
                "anthropic",
                "the Anthropic SDK found no API key, auth token, or stored credentials. A Claude "
                "subscription does not include API access; the Messages API needs a key from the "
                "Anthropic Console (console.anthropic.com).",
            )

    def generate(self, system: str, messages: list[dict[str, str]]) -> backend.TaskletSpec:
        """
        Requests a tasklet from the model.

        :param system: The system prompt.
        :param messages: The conversation so far.
        :return: The tasklet described by the model.
        :raises AIExpansionError: If the request fails or the response cannot be interpreted.
        """
        model = Config.get("ai", "model")
        try:
            with self._client.messages.stream(
                model=model,
                max_tokens=int(Config.get("ai", "max_tokens")),
                system=system,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": Config.get("ai", "effort"),
                    "format": {"type": "json_schema", "schema": backend.RESPONSE_SCHEMA},
                },
                messages=messages,
            ) as stream:
                response = stream.get_final_message()
        except self._anthropic.AuthenticationError as e:
            raise backend.missing_credentials_error("anthropic", str(e)) from e
        except self._anthropic.APIError as e:
            raise AIExpansionError(f"The Anthropic API request failed for model {model}: {e}") from e

        if response.stop_reason == "refusal":
            raise AIExpansionError("The model declined to generate this tasklet.")
        if response.stop_reason == "max_tokens":
            raise AIExpansionError("The answer was cut off at ai.max_tokens; raise it (DACE_ai_max_tokens).")
        text = "".join(block.text for block in response.content if block.type == "text")
        return backend.parse_response(text)
