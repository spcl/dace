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
from typing import Any, Dict, List, Optional, Type

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
        from dace.libraries.ai import cache as ai_cache
        from dace.libraries.ai import environments as ai_environments
        from dace.libraries.ai import prompts, transcript, verify
        from dace.libraries.ai.backend import get_provider
        from dace.libraries.ai.context import collect_context
        from dace.libraries.ai.exceptions import AIExpansionError

        described = f'{type(node).__name__} "{node.name}"'
        ctx = collect_context(node, parent_state, parent_sdfg)
        record = transcript.begin(node, parent_state, parent_sdfg)
        if record.path:
            _status(f'{described}: transcript in {record.path}')

        # Resolved on first use rather than up front, so that an expansion answered entirely from
        # the cache needs neither a provider SDK nor an API key
        provider = None
        system = prompts.SYSTEM_PROMPT
        record.record_system_prompt(system)
        _detail('system prompt', system)

        prompt = prompts.build_user_prompt(ctx)
        messages: List[Dict[str, str]] = [{'role': 'user', 'content': prompt}]

        attempts = max(0, int(Config.get('ai', 'max_repair_attempts'))) + 1
        should_verify = Config.get_bool('ai', 'verify')

        spec = None
        environments: List[Any] = []
        try:
            for attempt in range(attempts):
                entry = record.begin_attempt(attempt + 1, prompt)
                _detail('user prompt' if attempt == 0 else 'repair prompt', prompt)

                cache_key = ai_cache.key(system, messages)
                spec = ai_cache.lookup(cache_key)
                cached = spec is not None
                if cached:
                    _status(f'{described}: reusing the cached answer for this prompt ({cache_key[:12]}), '
                            f'attempt {attempt + 1}/{attempts}')
                else:
                    _status(f'{described}: asking {Config.get("ai", "provider")} '
                            f'({Config.get("ai", "model")}), attempt {attempt + 1}/{attempts}')
                    provider = provider or get_provider()
                    spec = provider.generate(system, messages)
                    ai_cache.store(cache_key, spec, system, messages)
                record.record_answer(entry, spec, cached=cached)
                _detail('answer', spec.raw_response or _echo(spec))
                environments = ai_environments.collect(spec)
                if not should_verify:
                    _status(f'{described}: verification is off (ai.verify), taking the code as generated')
                    break

                result = verify.probe_compile(spec, ctx, environments)
                record.record_verification(entry, result, result.source, ctx.capabilities is not None
                                           and ctx.capabilities.device_level)
                _detail('probe source', result.source)
                if result.inconclusive:
                    _status(f'{described}: probe compilation was inconclusive, taking the code as generated')
                    break
                if result.ok:
                    _status(f'{described}: probe compiled cleanly ({result.command})')
                    break

                _status(f'{described}: probe compilation failed ({result.command})')
                _detail('probe diagnostics', result.stderr)
                if attempt == attempts - 1:
                    raise AIExpansionError(f'The generated code for {described} still does not compile after '
                                           f'{attempts} attempt(s). Last diagnostics:\n\n{result.stderr}')
                logger.info('AI expansion of %s failed to compile; asking for a repair (attempt %d/%d).', described,
                            attempt + 2, attempts)
                _status(f'{described}: asking for a repair')
                prompt = prompts.build_repair_prompt(result.stderr, result.command)
                messages.append({'role': 'assistant', 'content': _echo(spec)})
                messages.append({'role': 'user', 'content': prompt})
        except Exception as e:
            record.record_outcome('failed', str(e))
            if record.path is None:
                raise
            # Named in the message rather than only logged: the run that produced this cost a model
            # call, and the transcript is the only copy of what it said.
            if isinstance(e, AIExpansionError):
                raise AIExpansionError(f'{e}\n\nThe full prompts and answers are in {record.path}') from e
            logger.warning('AI expansion of %s failed; the prompts and answers are in %s', described, record.path)
            raise

        cls.environments = list(environments)
        record.record_outcome(
            'expanded', environments=[env.full_class_path() for env in environments if hasattr(env, 'full_class_path')])
        _status(f'{described}: expanded into a tasklet' +
                (f' using {len(environments)} environment(s)' if environments else ''))
        return ExpandAI._make_tasklet(node, spec, ctx)

    @staticmethod
    def _make_tasklet(node: nodes.LibraryNode, spec: Any, ctx: Any) -> nodes.Tasklet:
        """
        Turns a generated specification into a tasklet.

        The connector types resolved during context collection are stamped onto the tasklet rather
        than left for :func:`dace.sdfg.infer_types.infer_connector_types` to fill in later. They
        have to be: the model was told those types and the probe compiled against them, while
        inference re-derives them from the *tasklet*, and one of its rules -- never pass GPU global
        memory by value -- only applies while the node is still a library node. Leaving it to
        inference would turn a GPU operand the model was told to treat as a pointer into a value
        that the host loads directly out of device memory.

        :param node: The library node being replaced.
        :param spec: The generated tasklet specification.
        :param ctx: The context the tasklet was generated for.
        :return: The tasklet that replaces the library node.
        """
        inputs = dict(node.in_connectors)
        outputs = dict(node.out_connectors)
        for conn in ctx.connectors:
            if conn.conntype is not None:
                (inputs if conn.direction == 'in' else outputs)[conn.name] = conn.conntype

        language = dtypes.Language.Python if spec.language.upper() == 'PYTHON' else dtypes.Language.CPP
        tasklet = nodes.Tasklet(node.name,
                                inputs=inputs,
                                outputs=outputs,
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


def _status(message: str) -> None:
    """
    Prints one line describing a step of the expansion, when ``debugprint`` is set at all.

    An expansion makes a paid network request and may compile and re-ask several times, so it is
    worth being able to watch it happen rather than waiting in silence.

    :param message: The message, without a trailing newline.
    """
    from dace.config import Config

    if Config.get_bool('debugprint'):
        print(f'[ai] {message}', flush=True)


def _detail(label: str, body: Optional[str]) -> None:
    """
    Prints a block of text that a step acted on, only under ``debugprint=verbose``.

    The same text is written to the transcript regardless; this is for watching a run live.

    :param label: A short heading, e.g. ``'user prompt'``.
    :param body: The text. Nothing is printed when it is empty.
    """
    import textwrap

    from dace.config import Config

    if Config.get('debugprint') != 'verbose' or not body or not body.strip():
        return
    print(f'[ai] --- {label} ---', flush=True)
    print(textwrap.indent(body.rstrip('\n'), '    '), flush=True)
    print(f'[ai] --- end {label} ---', flush=True)


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
