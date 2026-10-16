# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""OpenAI Responses API provider."""

from dace.config import Config
from dace.libraries.ai import backend
from dace.libraries.ai.exceptions import AIExpansionError


class OpenAIProvider:
    """Generates tasklets through the OpenAI Responses API."""

    def __init__(self) -> None:
        try:
            import openai  # Optional dependency (dace[agent-openai])
        except ImportError as e:
            raise backend.missing_sdk_error("openai", "openai", "agent-openai", e)

        self._openai = openai
        options = backend.client_options("openai")
        try:
            self._client = openai.OpenAI(**options)
        except openai.OpenAIError as e:
            if "api_key" not in options:
                raise backend.missing_credentials_error("openai", str(e)) from e
            raise AIExpansionError(f"Could not create the OpenAI client: {e}") from e

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
            response = self._client.responses.create(
                model=model,
                instructions=system,
                input=[{"role": m["role"], "content": m["content"]} for m in messages],
                max_output_tokens=int(Config.get("ai", "max_tokens")),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "dace_tasklet",
                        "strict": True,
                        "schema": backend.RESPONSE_SCHEMA,
                    },
                },
            )
        except self._openai.AuthenticationError as e:
            raise backend.missing_credentials_error("openai", str(e)) from e
        except self._openai.OpenAIError as e:
            raise AIExpansionError(f"The OpenAI API request failed for model {model}: {e}") from e

        return backend.parse_response(response.output_text)
