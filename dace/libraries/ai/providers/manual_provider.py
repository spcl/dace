# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Provider that has the user relay prompts to a language model."""

import sys

from dace.libraries.ai import backend, exchange
from dace.libraries.ai.exceptions import AIExpansionError

_INSTRUCTIONS = """
================================================================================
DaCe needs an implementation for node "{node_name}".

The prompt has been written to:

    {prompt_path}

Paste its contents into a chat with a language model, then return its answer in one of two ways:

  * paste the JSON reply here and press Ctrl-D (Ctrl-Z then Enter on Windows), or
  * save the reply to the file below, then either press Ctrl-D now or run the program again:

    {response_path}

Paste only the JSON object. A saved answer is reused while the prompt does not change. To answer
the prompts of all nodes at once, see dace.libraries.ai.generate_prompts.
================================================================================
"""


class ManualProvider:
    """
    Writes each prompt to a file and reads the reply from standard input or a response file. Needs
    no SDK or API key, and an interactive terminal.
    """

    def __init__(self) -> None:
        self._directory = exchange.prompt_directory()

    def generate(self, system: str, messages: list[dict[str, str]], node_name: str = "") -> backend.TaskletSpec:
        """
        Writes the prompt out, then reads the model's reply from the user.

        :param system: The system prompt.
        :param messages: The conversation so far.
        :param node_name: Name of the library node, for the file names and messages.
        :return: The tasklet described in the reply.
        :raises AIExpansionError: If no reply is provided, or it cannot be interpreted.
        """
        prompt = exchange.render_prompt(system, messages)
        prompt_path, response_path = exchange.exchange_paths(self._directory, prompt, node_name)

        reply = exchange.read_saved(response_path)
        if reply:
            print(
                f'Reusing the saved answer for node "{node_name}" from {response_path}.',
                file=sys.stderr,
                flush=True,
            )
        else:
            with open(prompt_path, "w", encoding="utf-8") as fp:
                fp.write(prompt)
            print(
                _INSTRUCTIONS.format(node_name=node_name, prompt_path=prompt_path, response_path=response_path),
                file=sys.stderr,
                flush=True,
            )
            reply = sys.stdin.read().strip() or exchange.read_saved(response_path)

        if not reply:
            raise AIExpansionError(
                f'No response was provided for node "{node_name}". Paste the model\'s '
                f"JSON reply on standard input, or save it to {response_path} and run again. "
                f"The prompt is in {prompt_path}.\nNote that the manual provider needs an "
                "interactive terminal; set DACE_ai_provider to a different backend for "
                "unattended runs."
            )

        return backend.parse_response(reply, f'node "{node_name}"')
