# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Responses API backend for AI-generated library node expansions. """

import json
from typing import Dict, List

from dace.config import Config
from dace.libraries.ai import backend
from dace.libraries.ai.exceptions import AIExpansionError


class ResponsesProvider:
    """
    Generates tasklets through the Responses API of the ``openai`` package.

    Selected with ``DACE_ai_provider=responses``. Like the Anthropic provider, the SDK is imported
    in the constructor so that it is only required when this provider is actually used.
    """

    def __init__(self) -> None:
        try:
            import openai  # Optional dependency: only needed when the 'ai' expansion is used
        except ImportError as e:
            raise backend.missing_sdk_error('responses', 'openai', 'ai-openai', e)

        self._openai = openai
        key = backend.api_key('responses')
        timeout = float(Config.get('ai', 'timeout'))
        try:
            self._client = openai.OpenAI(api_key=key, timeout=timeout) if key else openai.OpenAI(timeout=timeout)
        except Exception as e:
            raise backend.missing_credentials_error('responses', str(e)) from e

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
            response = self._client.responses.create(
                model=model,
                instructions=system,
                input=[{
                    'role': m['role'],
                    'content': m['content']
                } for m in messages],
                max_output_tokens=int(Config.get('ai', 'max_tokens')),
                text={
                    'format': {
                        'type': 'json_schema',
                        'name': 'dace_tasklet',
                        'strict': True,
                        'schema': backend.RESPONSE_SCHEMA,
                    },
                },
            )
        except self._openai.OpenAIError as e:
            raise AIExpansionError(f'The Responses API request failed for model {model}: {e}') from e

        text = getattr(response, 'output_text', '') or ''
        if not text.strip():
            raise AIExpansionError('The model returned no content.')
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise AIExpansionError(f'The model response was not valid JSON: {e}') from e
        return backend.spec_from_dict(payload)
