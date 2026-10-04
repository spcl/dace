# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Exchanging prompts and responses with a language model by hand.

The ``manual`` provider asks for one reply at a time, while the expansion is running. This module
offers the same exchange as a batch: write out the prompts for every AI-expanded library node in an
SDFG, answer them at leisure (in a chat interface, or with any other tool), and read the responses
back in one go::

    dace.libraries.ai.generate_prompts(sdfg, 'prompts')
    # ... save a <name>_<hash>_response.json next to every <name>_<hash>_prompt.md ...
    dace.libraries.ai.read_prompt_responses(sdfg, 'prompts')

The node-level equivalents are :meth:`dace.sdfg.nodes.LibraryNode.generate_prompt` and
:meth:`dace.sdfg.nodes.LibraryNode.read_prompt_response`. Both flows use the same files, so a
response saved for one is found by the other.
"""

import hashlib
import json
import os
import tempfile
import warnings
from typing import Dict, List, Optional, Tuple

from dace.config import Config
from dace.libraries.ai import backend, prompts
from dace.libraries.ai.context import collect_context
from dace.libraries.ai.exceptions import AIExpansionError
from dace.libraries.ai.nodes import AITasklet
from dace.sdfg import nodes
from dace.sdfg import SDFG, SDFGState

#: Appended to the prompt, since a chat interface has no structured-output mode to enforce it.
JSON_INSTRUCTIONS = """
# Response format

Reply with a single JSON object and nothing else -- no commentary before or after it. It must
match this JSON schema exactly:

{schema}
""".strip()


def prompt_directory(directory: Optional[str] = None) -> str:
    """
    Returns the directory prompts are written to and responses are read from, creating it.

    :param directory: An explicit directory, or ``None`` for ``ai.manual_dir`` (by default a
                      ``dace_ai_prompts`` directory under the system temporary directory).
    :return: The directory.
    """
    directory = directory or Config.get('ai', 'manual_dir') or os.path.join(tempfile.gettempdir(), 'dace_ai_prompts')
    os.makedirs(directory, exist_ok=True)
    return directory


def render_prompt(system: str, messages: List[Dict[str, str]]) -> str:
    """
    Renders a conversation as one self-contained text, to be pasted into a chat interface.

    :param system: The system prompt.
    :param messages: The conversation so far.
    :return: The full prompt, ending with the required response format.
    """
    schema = json.dumps(backend.RESPONSE_SCHEMA, indent=2)
    conversation = '\n\n'.join(f'## {m["role"]}\n\n{m["content"]}' for m in messages)
    return (f'# System\n\n{system}\n\n# Conversation\n\n{conversation}\n\n' + JSON_INSTRUCTIONS.format(schema=schema))


def exchange_paths(directory: str, prompt: str, name: str) -> Tuple[str, str]:
    """
    Returns the prompt and response file paths for one exchange.

    The names are derived from the prompt's content rather than from the process, so that
    re-running the same program finds the answer given last time. A repair round asks a different
    question, so it gets its own pair of files.

    :param directory: The exchange directory.
    :param prompt: The full prompt text.
    :param name: The library node's name, to make the files recognizable.
    :return: A tuple of (prompt path, response path).
    """
    digest = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    readable = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
    stem = os.path.join(directory, f'{readable}_{digest}')
    return f'{stem}_prompt.md', f'{stem}_response.json'


def read_saved(path: str) -> str:
    """
    Reads a saved response, if there is one.

    :param path: Path of the response file.
    :return: Its contents, or an empty string if it does not exist or is empty.
    """
    try:
        with open(path, 'r') as fp:
            return fp.read().strip()
    except OSError:
        return ''


def parse_response(response: str, described: str) -> backend.TaskletSpec:
    """
    Interprets a response written by a model.

    :param response: The response, which must be a single JSON object matching
                     :data:`dace.libraries.ai.backend.RESPONSE_SCHEMA`.
    :param described: The node the response is for, for error messages.
    :return: The tasklet the response describes.
    :raises AIExpansionError: If the response is not valid JSON or has no tasklet body.
    """
    response = response.strip()
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as e:
        raise AIExpansionError(f'The response for {described} is not valid JSON: {e}\nIt must be a single JSON '
                               'object matching the schema at the end of the prompt.') from e
    return backend.spec_from_dict(payload, raw=response)


def _describe(node: nodes.LibraryNode) -> str:
    """
    Names a node in messages.

    :param node: The library node.
    :return: Its type and name.
    """
    return f'{type(node).__name__} "{node.name}"'


def generate_prompt(node: nodes.LibraryNode, state: SDFGState) -> str:
    """
    Returns the prompt the ``'ai'`` implementation would send to a model for this node.

    :param node: The library node.
    :param state: The state containing it.
    :return: The full, self-contained prompt.
    """
    ctx = collect_context(node, state, state.sdfg)
    return render_prompt(prompts.SYSTEM_PROMPT, [{'role': 'user', 'content': prompts.build_user_prompt(ctx)}])


def read_prompt_response(node: nodes.LibraryNode, state: SDFGState, response: str) -> AITasklet:
    """
    Expands a library node with a model's response to :func:`generate_prompt`.

    The response goes through the same verification as one obtained from a provider, but a
    response that does not compile is reported rather than sent back for repair.

    :param node: The library node.
    :param state: The state containing it.
    :param response: The model's reply, a single JSON object.
    :return: The tasklet that replaced the node.
    :raises AIExpansionError: If the response cannot be interpreted or does not compile.
    """
    spec = parse_response(response, _describe(node))
    before = set(state.nodes())
    node.expand(state, nodes.AI_IMPLEMENTATION_NAME, response=spec)
    return next(n for n in state.nodes() if n not in before and isinstance(n, AITasklet))


def _ai_nodes(sdfg: SDFG) -> List[Tuple[nodes.LibraryNode, SDFGState]]:
    """
    Lists the library nodes of an SDFG, including nested ones, that expand with ``'ai'``.

    :param sdfg: The SDFG to search.
    :return: A list of (node, containing state).
    """
    result = []
    for node, state in sdfg.all_nodes_recursive():
        if not isinstance(node, nodes.LibraryNode):
            continue
        if (node.implementation or type(node).default_implementation) == nodes.AI_IMPLEMENTATION_NAME:
            result.append((node, state))
    return result


def generate_prompts(sdfg: SDFG, directory: Optional[str] = None) -> List[str]:
    """
    Writes out the prompt for every library node in an SDFG that expands with ``'ai'``.

    Each prompt is written to ``<node name>_<hash>_prompt.md``. Its answer is expected next to it,
    as ``<node name>_<hash>_response.json``, for :func:`read_prompt_responses` to pick up.

    :param sdfg: The SDFG, including nested SDFGs.
    :param directory: Where to write the prompts; ``ai.manual_dir`` if not given.
    :return: The paths of the prompt files written.
    """
    directory = prompt_directory(directory)
    written = []
    for node, state in _ai_nodes(sdfg):
        prompt = generate_prompt(node, state)
        prompt_path, _ = exchange_paths(directory, prompt, node.name)
        with open(prompt_path, 'w') as fp:
            fp.write(prompt)
        written.append(prompt_path)
    return written


def read_prompt_responses(sdfg: SDFG, directory: Optional[str] = None) -> List[AITasklet]:
    """
    Expands every library node whose response was saved next to its prompt.

    The prompts are regenerated and matched to their response files by content, so this must run on
    the SDFG the prompts were generated from, before it is compiled. Nodes without a saved response
    are left as library nodes, and are listed in a warning.

    :param sdfg: The SDFG, including nested SDFGs.
    :param directory: Where the responses are; ``ai.manual_dir`` if not given.
    :return: The tasklets that replaced the nodes.
    :raises AIExpansionError: If a response cannot be interpreted or does not compile.
    """
    directory = prompt_directory(directory)

    # Matched up front: every prompt must be computed against the SDFG as it was generated from,
    # not one in which some of the nodes have already been expanded
    answered, missing = [], []
    for node, state in _ai_nodes(sdfg):
        prompt_path, response_path = exchange_paths(directory, generate_prompt(node, state), node.name)
        response = read_saved(response_path)
        if response:
            answered.append((node, state, response))
        else:
            missing.append(f'{_describe(node)} ({prompt_path})')

    if missing:
        listing = '\n    '.join(missing)
        warnings.warn(f'No saved response for {len(missing)} node(s), which remain library nodes:\n    {listing}')
    return [read_prompt_response(node, state, response) for node, state, response in answered]
