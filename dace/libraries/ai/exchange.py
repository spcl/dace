# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Prompts and responses exchanged through files, for every ``'ai'`` node of an SDFG at once::

    dace.libraries.ai.generate_prompts(sdfg, 'prompts')
    # ... save <name>_<hash>_response.json next to each <name>_<hash>_prompt.md ...
    dace.libraries.ai.read_prompt_responses(sdfg, 'prompts')

The ``manual`` provider and :meth:`dace.sdfg.nodes.LibraryNode.generate_prompt` use the same files.
"""

import hashlib
import json
import os
import warnings

from dace.libraries.ai import backend, cache, prompts
from dace.libraries.ai.context import collect_context
from dace.libraries.ai.nodes import AITasklet
from dace.sdfg import SDFG, SDFGState, nodes

#: Appended to the prompt, since a chat interface does not enforce the response schema.
JSON_INSTRUCTIONS = """
# Response format

Reply with a single JSON object and nothing else -- no commentary before or after it. It must
match this JSON schema exactly:

{schema}
""".strip()


def prompt_directory(directory: str | None = None) -> str:
    """
    Returns the directory prompts are written to and responses are read from, creating it.

    :param directory: An explicit directory, or ``None`` for ``<build folder root>/ai-prompts``.
    :return: The directory.
    """
    if directory is None:
        return cache.storage_dir("prompts", create=True)
    os.makedirs(directory, exist_ok=True)
    return directory


def render_prompt(system: str, messages: list[dict[str, str]]) -> str:
    """
    Renders a conversation as one text, to be pasted into a chat interface.

    :param system: The system prompt.
    :param messages: The conversation so far.
    :return: The full prompt, ending with the required response format.
    """
    schema = json.dumps(backend.RESPONSE_SCHEMA, indent=2)
    conversation = "\n\n".join(f"## {m['role']}\n\n{m['content']}" for m in messages)
    return f"# System\n\n{system}\n\n# Conversation\n\n{conversation}\n\n" + JSON_INSTRUCTIONS.format(schema=schema)


def exchange_paths(directory: str, prompt: str, name: str) -> tuple[str, str]:
    """
    Returns the prompt and response file paths for a prompt, named after its content, so that
    re-running a program finds the saved answer.

    :param directory: The exchange directory.
    :param prompt: The full prompt text.
    :param name: The library node's name, to make the files recognizable.
    :return: A tuple of (prompt path, response path).
    """
    digest = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    readable = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    stem = os.path.join(directory, f"{readable}_{digest}")
    return f"{stem}_prompt.md", f"{stem}_response.json"


def read_saved(path: str) -> str:
    """
    Reads a saved response, if there is one.

    :param path: Path of the response file.
    :return: Its contents, or an empty string if it does not exist or is empty.
    """
    try:
        with open(path, encoding="utf-8") as fp:
            return fp.read().strip()
    except OSError:
        return ""


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
    return render_prompt(prompts.SYSTEM_PROMPT, [{"role": "user", "content": prompts.build_user_prompt(ctx)}])


def read_prompt_response(node: nodes.LibraryNode, state: SDFGState, response: str) -> AITasklet:
    """
    Expands a library node with a model's response to :func:`generate_prompt`. The response is
    compile-checked; failures raise an error instead of a repair round.

    :param node: The library node.
    :param state: The state containing it.
    :param response: The model's reply, a single JSON object.
    :return: The tasklet that replaced the node.
    :raises AIExpansionError: If the response cannot be interpreted or does not compile.
    """
    spec = backend.parse_response(response, _describe(node))
    before = set(state.nodes())
    node.expand(state, nodes.AI_IMPLEMENTATION_NAME, response=spec)
    return next(n for n in state.nodes() if n not in before and isinstance(n, AITasklet))


def _ai_nodes(sdfg: SDFG) -> list[tuple[nodes.LibraryNode, SDFGState]]:
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


def generate_prompts(sdfg: SDFG, directory: str | None = None) -> list[str]:
    """
    Writes ``<node name>_<hash>_prompt.md`` for every library node in an SDFG that expands with
    ``'ai'``. :func:`read_prompt_responses` reads the answers from ``<node name>_<hash>_response.json``.

    :param sdfg: The SDFG, including nested SDFGs.
    :param directory: Where to write the prompts; ``<build folder root>/ai-prompts`` if not given.
    :return: The paths of the prompt files written.
    """
    directory = prompt_directory(directory)
    written = []
    for node, state in _ai_nodes(sdfg):
        prompt = generate_prompt(node, state)
        prompt_path, _ = exchange_paths(directory, prompt, node.name)
        with open(prompt_path, "w", encoding="utf-8") as fp:
            fp.write(prompt)
        written.append(prompt_path)
    return written


def read_prompt_responses(sdfg: SDFG, directory: str | None = None) -> list[AITasklet]:
    """
    Expands every library node whose response was saved next to its prompt. Responses are matched
    by prompt content, so this must run on the SDFG the prompts were generated from. Nodes without a
    response remain library nodes, with a warning.

    :param sdfg: The SDFG, including nested SDFGs.
    :param directory: Where the responses are; ``<build folder root>/ai-prompts`` if not given.
    :return: The tasklets that replaced the nodes.
    :raises AIExpansionError: If a response cannot be interpreted or does not compile.
    """
    directory = prompt_directory(directory)

    # All prompts are computed before any node is expanded
    answered, missing = [], []
    for node, state in _ai_nodes(sdfg):
        prompt_path, response_path = exchange_paths(directory, generate_prompt(node, state), node.name)
        response = read_saved(response_path)
        if response:
            answered.append((node, state, response))
        else:
            missing.append(f"{_describe(node)} ({prompt_path})")

    if missing:
        listing = "\n    ".join(missing)
        warnings.warn(f"No saved response for {len(missing)} node(s), which remain library nodes:\n    {listing}")
    return [read_prompt_response(node, state, response) for node, state, response in answered]
