# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
The ``"ai"`` library node expansion.

``"ai"`` is a *reserved* implementation name rather than an entry registered into every library
node's ``implementations`` dictionary: :meth:`dace.sdfg.nodes.LibraryNode.expand` recognizes it and
resolves it here. That way a library node defined outside DaCe -- by a downstream project or a
plugin loaded at run time -- also expands with ``"ai"``, without anything having had to run when
its class was defined.

:class:`ExpandTransformation` identifies the node it replaces through the class attribute
``_match_node``, so a single shared class cannot serve every library node type.
:meth:`ExpandAI.for_node_class` synthesizes (and caches) one subclass per node type instead.

This module deliberately imports nothing beyond :mod:`dace.library` and the transformation base
class at load time; context collection, prompting, provider access and verification are all
imported inside :meth:`ExpandAI.expansion`.
"""

import functools
import logging
from typing import Any, Dict, List, Type

import dace.library
from dace import dtypes
from dace.sdfg import nodes
from dace.sdfg import SDFG, SDFGState
from dace.transformation.transformation import ExpandTransformation

logger = logging.getLogger(__name__)


class ExpandAI(ExpandTransformation):
    """
    Expands any library node into a tasklet written by a language model.

    The model is given the node's description and properties, the connectors and the data
    descriptors behind them, every enclosing map, loop and nested SDFG up to the top-level SDFG,
    what the generated code is allowed to do at that point (GPU device code? is ``__state`` in
    scope?), and the compiler and architecture the code will be built for.
    """

    environments: List[Any] = []

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def for_node_class(node_class: Type[nodes.LibraryNode]) -> Type['ExpandAI']:
        """
        Returns the expansion class bound to a specific library node type.

        Only the synthesized class is cached; the generated code never is. Two nodes of the same
        type in different parts of an SDFG are therefore generated independently, which is what
        makes context-sensitive expansion work.

        :param node_class: The library node class being expanded.
        :return: A subclass of :class:`ExpandAI` whose ``_match_node`` is ``node_class``.
        """
        subclass = type(f'ExpandAI_{node_class.__name__}', (ExpandAI, ), {'environments': []})
        subclass = dace.library.expansion(subclass)
        subclass._match_node = node_class
        return subclass

    @classmethod
    def expansion(cls, node: nodes.LibraryNode, parent_state: SDFGState, parent_sdfg: SDFG, **kwargs) -> nodes.Tasklet:
        """
        Generates a tasklet for a library node.

        :param cls: The per-node-class expansion produced by :meth:`for_node_class`. Any
                    environments the generated code needs are recorded on it, since
                    :meth:`dace.transformation.transformation.ExpandTransformation.apply` reads
                    ``type(self).environments`` once this method returns.
        :param node: The library node to replace.
        :param parent_state: The state containing the node.
        :param parent_sdfg: The SDFG containing the state.
        :param kwargs: Ignored; accepted so that the expansion can be invoked like any other.
        :return: The generated tasklet.
        :raises AIExpansionError: If no tasklet could be generated.
        """
        # Imported here so that resolving the reserved 'ai' implementation stays cheap and free of
        # import cycles; none of this is needed unless an expansion actually runs.
        from dace.config import Config
        from dace.libraries.ai import environments as ai_environments
        from dace.libraries.ai import prompts, verify
        from dace.libraries.ai.backend import get_provider
        from dace.libraries.ai.context import collect_context
        from dace.libraries.ai.exceptions import AIExpansionError

        ctx = collect_context(node, parent_state, parent_sdfg)
        provider = get_provider()
        system = prompts.SYSTEM_PROMPT
        messages: List[Dict[str, str]] = [{'role': 'user', 'content': prompts.build_user_prompt(ctx)}]

        attempts = max(0, int(Config.get('ai', 'max_repair_attempts'))) + 1
        should_verify = Config.get_bool('ai', 'verify')

        spec = None
        environments: List[Any] = []
        for attempt in range(attempts):
            spec = provider.generate(system, messages)
            environments = ai_environments.collect(spec)
            if not should_verify:
                break

            result = verify.probe_compile(spec, ctx, environments)
            if result.ok or result.inconclusive:
                break
            if attempt == attempts - 1:
                raise AIExpansionError(
                    f'The generated code for {type(node).__name__} "{node.name}" still does not compile after '
                    f'{attempts} attempt(s). Last diagnostics:\n\n{result.stderr}')
            logger.info('AI expansion of %s "%s" failed to compile; asking for a repair (attempt %d/%d).',
                        type(node).__name__, node.name, attempt + 2, attempts)
            messages.append({'role': 'assistant', 'content': _echo(spec)})
            messages.append({'role': 'user', 'content': prompts.build_repair_prompt(result.stderr, result.command)})

        cls.environments = list(environments)
        return ExpandAI._make_tasklet(node, spec)

    @staticmethod
    def _make_tasklet(node: nodes.LibraryNode, spec: Any) -> nodes.Tasklet:
        """
        Turns a generated specification into a tasklet.

        :param node: The library node being replaced.
        :param spec: The generated tasklet specification.
        :return: The tasklet that replaces the library node.
        """
        language = dtypes.Language.Python if spec.language.upper() == 'PYTHON' else dtypes.Language.CPP
        tasklet = nodes.Tasklet(node.name,
                                inputs=dict(node.in_connectors),
                                outputs=dict(node.out_connectors),
                                code=spec.code,
                                language=language,
                                state_fields=list(spec.state_fields),
                                code_global=spec.code_global,
                                code_init=spec.code_init,
                                code_exit=spec.code_exit,
                                side_effects=spec.side_effects or None,
                                ignored_symbols=set(spec.ignored_symbols))
        if spec.notes:
            logger.info('AI expansion of %s "%s": %s', type(node).__name__, node.name, spec.notes)
        return tasklet


def _echo(spec: Any) -> str:
    """
    Renders a generated specification back into the conversation for a repair round.

    :param spec: The specification the model returned.
    :return: The text to replay as the assistant's previous turn.
    """
    import dataclasses
    import json

    return json.dumps(
        {
            'notes': spec.notes,
            'language': spec.language,
            'code': spec.code,
            'code_global': spec.code_global,
            'code_init': spec.code_init,
            'code_exit': spec.code_exit,
            'state_fields': spec.state_fields,
            'side_effects': spec.side_effects,
            'ignored_symbols': spec.ignored_symbols,
            # Echoed too: the repair instruction asks for the complete object back, and the schema
            # marks these required, so omitting them would invite an invalid reply
            'use_environments': spec.use_environments,
            'environments': [dataclasses.asdict(env) for env in spec.environments],
        },
        indent=1)
