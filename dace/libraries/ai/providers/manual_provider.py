# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Copy-and-paste backend for AI-generated library node expansions. """

import hashlib
import json
import os
import re
import sys
import tempfile
from typing import Dict, List

from dace.config import Config
from dace.libraries.ai import backend
from dace.libraries.ai.exceptions import AIExpansionError

#: Appended to the prompt, since a chat interface has no structured-output mode to enforce it.
_JSON_INSTRUCTIONS = """
# Response format

Reply with a single JSON object and nothing else -- no commentary before or after it. It must
match this JSON schema exactly:

{schema}
""".strip()

_INSTRUCTIONS = """
================================================================================
DaCe needs an implementation for {node_type} "{node_name}".

The prompt has been written to:

    {prompt_path}

Paste its contents into a chat with a language model, then return its answer in one of two ways:

  * paste the JSON reply here and press Ctrl-D (Ctrl-Z then Enter on Windows), or
  * save the reply to the file below, then either press Ctrl-D now or run the program again:

    {response_path}

A ```json fenced block is fine; the fences are stripped. A saved answer is reused on later runs,
so re-running this program will not ask again unless the prompt itself changes.
================================================================================
"""


class ManualProvider:
    """
    Asks the user to relay the prompt to a language model by hand.

    This backend needs no API key and no SDK, so it works with a chat subscription rather than API
    credits. It is meant for trying out and debugging AI expansion -- reading the prompt a library
    node produces, and pasting back a reply -- rather than for unattended compilation. The repair
    loop works as usual: a second round-trip is requested if the generated code does not compile.
    """

    def __init__(self) -> None:
        self._directory = Config.get('ai', 'manual_dir') or os.path.join(tempfile.gettempdir(), 'dace_ai_prompts')
        os.makedirs(self._directory, exist_ok=True)

    def _paths(self, prompt: str):
        """
        Returns the prompt and response file paths for one exchange.

        The names are derived from the prompt's content rather than from the process, so that
        re-running the same program finds the answer given last time and proceeds without asking
        again. A repair round asks a different question, so it gets its own pair of files.

        :param prompt: The full prompt text for this exchange.
        :return: A tuple of (prompt path, response path).
        """
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        stem = os.path.join(self._directory, f'dace_ai_{digest}')
        return f'{stem}_prompt.md', f'{stem}_response.json'

    def generate(self, system: str, messages: List[Dict[str, str]]) -> backend.TaskletSpec:
        """
        Writes the prompt out, then reads the model's reply from the user.

        :param system: The system prompt.
        :param messages: The conversation so far.
        :return: The tasklet described in the reply.
        :raises AIExpansionError: If no reply is provided, or it cannot be interpreted.
        """
        schema = json.dumps(backend.RESPONSE_SCHEMA, indent=2)
        conversation = '\n\n'.join(f'## {m["role"]}\n\n{m["content"]}' for m in messages)
        prompt = f'# System\n\n{system}\n\n# Conversation\n\n{conversation}\n\n'
        prompt += _JSON_INSTRUCTIONS.format(schema=schema)
        prompt_path, response_path = self._paths(prompt)
        node_type, node_name = _describe(messages)

        # An answer to this exact prompt from an earlier run is reused, so that re-running a
        # program does not ask the same question again
        reply = _read(response_path)
        if reply:
            print(f'Reusing the saved answer for {node_type} "{node_name}" from {response_path}.',
                  file=sys.stderr,
                  flush=True)
        else:
            with open(prompt_path, 'w') as fp:
                fp.write(prompt)
            print(_INSTRUCTIONS.format(node_type=node_type,
                                       node_name=node_name,
                                       prompt_path=prompt_path,
                                       response_path=response_path),
                  file=sys.stderr,
                  flush=True)
            reply = sys.stdin.read().strip() or _read(response_path)

        if not reply:
            raise AIExpansionError(f'No response was provided for {node_type} "{node_name}". Paste the model\'s '
                                   f'JSON reply on standard input, or save it to {response_path} and run again. '
                                   f'The prompt is in {prompt_path}.\nNote that the manual provider needs an '
                                   'interactive terminal; set DACE_ai_provider to a different backend for '
                                   'unattended runs.')

        try:
            payload = json.loads(_strip_fences(reply))
        except json.JSONDecodeError as e:
            raise AIExpansionError(f'The reply for {node_type} "{node_name}" is not valid JSON: {e}\n'
                                   f'It must be a single JSON object matching the schema in {prompt_path}.') from e
        return backend.spec_from_dict(payload)


def _read(path: str) -> str:
    """
    Reads a saved reply, if there is one.

    :param path: Path of the response file.
    :return: Its contents, or an empty string if it does not exist or is empty.
    """
    try:
        with open(path, 'r') as fp:
            return fp.read().strip()
    except OSError:
        return ''


def _strip_fences(text: str) -> str:
    """
    Extracts the JSON object from a reply copied out of a chat interface.

    Such a reply is rarely bare JSON: it usually arrives inside a Markdown code fence, often with
    a sentence of commentary before and after it.

    :param text: The reply as pasted.
    :return: The JSON object it contains, or the input unchanged if none can be located.
    """
    stripped = text.strip()

    fenced = re.search(r'```[a-zA-Z0-9_+-]*[ \t]*\r?\n(.*?)```', stripped, re.DOTALL)
    if fenced is not None:
        return fenced.group(1).strip()

    # Unfenced, but possibly wrapped in commentary
    start, end = stripped.find('{'), stripped.rfind('}')
    if start != -1 and end > start:
        return stripped[start:end + 1]
    return stripped


def _describe(messages: List[Dict[str, str]]):
    """
    Recovers the node type and name from the prompt, for the on-screen instructions.

    :param messages: The conversation so far.
    :return: A tuple of (node type, node name), falling back to placeholders.
    """
    node_type, node_name = 'a library node', '?'
    for line in messages[0]['content'].splitlines():
        if line.startswith('Library node type: '):
            node_type = line.split(': ', 1)[1]
        elif line.startswith('Node name: '):
            node_name = line.split(': ', 1)[1]
            break
    return node_type, node_name
