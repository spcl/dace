# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the Anthropic provider, run against a local HTTP server rather than the real API.

These exercise the parts that would otherwise only be reachable with a live API key: that the
request DaCe builds is accepted by the installed SDK, that it carries the model, system prompt,
thinking configuration and structured-output schema, and that a streamed response is turned back
into a :class:`~dace.libraries.ai.backend.TaskletSpec`. They are skipped when the SDK is not
installed, and never reach the network.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import dace
from dace.libraries.ai import backend
from dace.libraries.ai.exceptions import AIExpansionError

anthropic = pytest.importorskip('anthropic', reason='needs the Anthropic SDK (pip install dace[ai-anthropic])')

from dace.libraries.ai.providers.anthropic_provider import AnthropicProvider  # noqa: E402

RESPONSE = {
    'notes': 'doubling',
    'language': 'CPP',
    'code': '_out = 2.0 * _in;',
    'code_global': '',
    'code_init': '',
    'code_exit': '',
    'state_fields': [],
    'side_effects': False,
    'ignored_symbols': [],
    'environments': [],
}


def _sse(events) -> bytes:
    """
    Encodes a list of Messages API events as a server-sent event stream.

    :param events: ``(event name, payload)`` pairs.
    :return: The encoded stream.
    """
    return ''.join(f'event: {name}\ndata: {json.dumps(payload)}\n\n' for name, payload in events).encode()


def _message_stream(text: str, stop_reason: str = 'end_turn') -> bytes:
    """
    Builds a minimal but well-formed streamed message carrying one text block.

    :param text: The text the assistant returns.
    :param stop_reason: The reason the message ended.
    :return: The encoded stream.
    """
    message = {
        'id': 'msg_test',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-opus-5',
        'content': [],
        'stop_reason': None,
        'stop_sequence': None,
        'usage': {
            'input_tokens': 1,
            'output_tokens': 1
        },
    }
    return _sse([
        ('message_start', {
            'type': 'message_start',
            'message': message
        }),
        ('content_block_start', {
            'type': 'content_block_start',
            'index': 0,
            'content_block': {
                'type': 'text',
                'text': ''
            }
        }),
        ('content_block_delta', {
            'type': 'content_block_delta',
            'index': 0,
            'delta': {
                'type': 'text_delta',
                'text': text
            }
        }),
        ('content_block_stop', {
            'type': 'content_block_stop',
            'index': 0
        }),
        ('message_delta', {
            'type': 'message_delta',
            'delta': {
                'stop_reason': stop_reason,
                'stop_sequence': None
            },
            'usage': {
                'output_tokens': 1
            }
        }),
        ('message_stop', {
            'type': 'message_stop'
        }),
    ])


class _Server:
    """ A local stand-in for the Messages API that records the request it received. """

    def __init__(self, body: bytes):
        """
        :param body: The server-sent event stream to reply with.
        """
        self.requests = []
        server_self = self

        class Handler(BaseHTTPRequestHandler):

            def do_POST(self):
                length = int(self.headers.get('Content-Length', 0))
                server_self.requests.append(json.loads(self.rfile.read(length)))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._httpd = HTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self._httpd.server_address[1]}'

    def __enter__(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def local_api(monkeypatch):
    """
    Points the Anthropic SDK at a local server with a dummy key.

    :return: A factory taking a response body and yielding the running server.
    """
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test-not-a-real-key')

    def factory(body: bytes):
        server = _Server(body)
        monkeypatch.setenv('ANTHROPIC_BASE_URL', server.url)
        return server

    return factory


def test_request_carries_the_configured_model_and_schema(local_api):
    with local_api(_message_stream(json.dumps(RESPONSE))) as server:
        with dace.config.set_temporary('ai', 'model', value='claude-opus-5'):
            with dace.config.set_temporary('ai', 'effort', value='high'):
                spec = AnthropicProvider().generate('SYSTEM', [{'role': 'user', 'content': 'USER'}])

    assert spec.code == '_out = 2.0 * _in;'
    assert spec.language == 'CPP'
    assert spec.notes == 'doubling'

    request = server.requests[0]
    assert request['model'] == 'claude-opus-5'
    assert request['system'] == 'SYSTEM'
    assert request['messages'] == [{'role': 'user', 'content': 'USER'}]
    assert request['stream'] is True
    assert request['thinking'] == {'type': 'adaptive'}
    assert request['output_config']['effort'] == 'high'
    assert request['output_config']['format']['type'] == 'json_schema'
    assert request['output_config']['format']['schema'] == backend.RESPONSE_SCHEMA


def test_refusal_is_reported_clearly(local_api):
    with local_api(_message_stream('', stop_reason='refusal')):
        with pytest.raises(AIExpansionError, match='declined'):
            AnthropicProvider().generate('SYSTEM', [{'role': 'user', 'content': 'USER'}])


def test_non_json_response_is_reported_clearly(local_api):
    with local_api(_message_stream('here is your tasklet, boss')):
        with pytest.raises(AIExpansionError, match='not valid JSON'):
            AnthropicProvider().generate('SYSTEM', [{'role': 'user', 'content': 'USER'}])


def test_missing_credentials_are_detected_before_any_request(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('ANTHROPIC_AUTH_TOKEN', raising=False)
    with dace.config.set_temporary('ai', 'api_key_envvar', value='DACE_NO_SUCH_AI_KEY'):
        with pytest.raises(AIExpansionError) as info:
            AnthropicProvider()

    message = str(info.value)
    assert 'DACE_NO_SUCH_AI_KEY' in message
    # The distinction that actually trips people up
    assert 'Pro' in message and 'console.anthropic.com' in message


if __name__ == '__main__':
    pytest.main([__file__])
