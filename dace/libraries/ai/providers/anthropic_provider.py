# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Anthropic backend for AI-generated library node expansions. """

import json
from typing import Dict, List

from dace.config import Config
from dace.libraries.ai import backend
from dace.libraries.ai.exceptions import AIExpansionError


class AnthropicProvider:
    """
    Generates tasklets through the Anthropic Messages API.

    The ``anthropic`` package is imported in the constructor rather than at module load, so that
    importing :mod:`dace.libraries.ai` -- which happens whenever any library node is expanded --
    never requires the SDK to be installed.
    """

    def __init__(self) -> None:
        try:
            import anthropic  # Optional dependency: only needed when the 'ai' expansion is used
        except ImportError as e:
            raise backend.missing_sdk_error('anthropic', 'anthropic', 'ai-anthropic', e)

        self._anthropic = anthropic
        key = backend.api_key('anthropic')
        timeout = float(Config.get('ai', 'timeout'))
        try:
            # Without an explicit key the SDK falls back to its own credential chain (an auth token,
            # or a profile stored by `ant auth login`), which is a legitimate way to authenticate.
            self._client = anthropic.Anthropic(api_key=key, timeout=timeout) if key else anthropic.Anthropic(
                timeout=timeout)
        except Exception as e:
            raise backend.missing_credentials_error('anthropic', str(e)) from e

        # The client resolves credentials lazily, and reports their absence only once a request is
        # made -- as a bare TypeError. Check up front so the failure is reported here, where it can
        # say which environment variable was consulted.
        if not any(getattr(self._client, attr, None) for attr in ('api_key', 'auth_token', 'credentials')):
            raise backend.missing_credentials_error(
                'anthropic', 'the Anthropic SDK found no API key, auth token, or stored credentials. Note that a '
                'Claude Pro or Max subscription does not include API access: the Messages API needs a key from '
                'the Anthropic Console (console.anthropic.com), billed separately.')

    def generate(self, system: str, messages: List[Dict[str, str]]) -> backend.TaskletSpec:
        """
        Requests a tasklet from the model.

        :param system: The system prompt.
        :param messages: The conversation so far.
        :return: The tasklet described by the model.
        :raises AIExpansionError: If the request fails or the response cannot be interpreted.
        """
        model = Config.get('ai', 'model')
        try:
            with self._client.messages.stream(
                    model=model,
                    max_tokens=int(Config.get('ai', 'max_tokens')),
                    system=system,
                    thinking={'type': 'adaptive'},
                    output_config={
                        'effort': Config.get('ai', 'effort'),
                        'format': {
                            'type': 'json_schema',
                            'schema': backend.RESPONSE_SCHEMA,
                        },
                    },
                    messages=messages,
            ) as stream:
                response = stream.get_final_message()
        except self._anthropic.APIError as e:
            raise AIExpansionError(f'The Anthropic API request failed for model {model}: {e}') from e

        if response.stop_reason == 'refusal':
            raise AIExpansionError('The model declined to generate this tasklet.')

        text = ''.join(block.text for block in response.content if block.type == 'text')
        if not text.strip():
            raise AIExpansionError(f'The model returned no content (stop reason: {response.stop_reason}).')
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise AIExpansionError(f'The model response was not valid JSON: {e}') from e
        return backend.spec_from_dict(payload)
