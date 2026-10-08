# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Copy-and-paste backend for AI-generated library node expansions."""

import sys
from typing import Dict, List

from dace.libraries.ai import backend, exchange
from dace.libraries.ai.exceptions import AIExpansionError

_INSTRUCTIONS = """
================================================================================
DaCe needs an implementation for {node_type} "{node_name}".

The prompt has been written to:

    {prompt_path}

Paste its contents into a chat with a language model, then return its answer in one of two ways:

  * paste the JSON reply here and press Ctrl-D (Ctrl-Z then Enter on Windows), or
  * save the reply to the file below, then either press Ctrl-D now or run the program again:

    {response_path}

Paste only the JSON object itself -- if the chat shows it in a code block, use that block's copy
button. A saved answer is reused on later runs, so re-running this program will not ask again
unless the prompt itself changes. To answer every node's prompt at once instead, see
dace.libraries.ai.generate_prompts.
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
        self._directory = exchange.prompt_directory()

    def generate(self, system: str, messages: List[Dict[str, str]]) -> backend.TaskletSpec:
        """
        Writes the prompt out, then reads the model's reply from the user.

        :param system: The system prompt.
        :param messages: The conversation so far.
        :return: The tasklet described in the reply.
        :raises AIExpansionError: If no reply is provided, or it cannot be interpreted.
        """
        prompt = exchange.render_prompt(system, messages)
        node_type, node_name = _describe(messages)
        prompt_path, response_path = exchange.exchange_paths(self._directory, prompt, node_name)

        # An answer to this exact prompt from an earlier run is reused, so that re-running a
        # program does not ask the same question again
        reply = exchange.read_saved(response_path)
        if reply:
            print(
                f'Reusing the saved answer for {node_type} "{node_name}" from {response_path}.',
                file=sys.stderr,
                flush=True,
            )
        else:
            with open(prompt_path, "w") as fp:
                fp.write(prompt)
            print(
                _INSTRUCTIONS.format(
                    node_type=node_type, node_name=node_name, prompt_path=prompt_path, response_path=response_path
                ),
                file=sys.stderr,
                flush=True,
            )
            reply = sys.stdin.read().strip() or exchange.read_saved(response_path)

        if not reply:
            raise AIExpansionError(
                f'No response was provided for {node_type} "{node_name}". Paste the model\'s '
                f"JSON reply on standard input, or save it to {response_path} and run again. "
                f"The prompt is in {prompt_path}.\nNote that the manual provider needs an "
                "interactive terminal; set DACE_ai_provider to a different backend for "
                "unattended runs."
            )

        return exchange.parse_response(reply, f'{node_type} "{node_name}"')


def _describe(messages: List[Dict[str, str]]):
    """
    Recovers the node type and name from the prompt, for the on-screen instructions.

    :param messages: The conversation so far.
    :return: A tuple of (node type, node name), falling back to placeholders.
    """
    node_type, node_name = "a library node", "?"
    for line in messages[0]["content"].splitlines():
        if line.startswith("Library node type: "):
            node_type = line.split(": ", 1)[1]
        elif line.startswith("Node name: "):
            node_name = line.split(": ", 1)[1]
            break
    return node_type, node_name
