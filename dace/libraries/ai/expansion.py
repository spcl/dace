# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
The ``'ai'`` expansion. The implementation name is reserved: :meth:`dace.sdfg.nodes.LibraryNode.expand`
resolves it here for every library node class, including ones defined outside DaCe.
"""

import functools
import os
import textwrap
import warnings
from typing import Any

import dace.library
from dace import dtypes
from dace.config import Config
from dace.libraries.ai import backend, iterate, prompts, verify
from dace.libraries.ai import cache as ai_cache
from dace.libraries.ai import environments as ai_environments
from dace.libraries.ai import session as ai_session
from dace.libraries.ai.context import collect_context
from dace.libraries.ai.exceptions import AIExpansionError
from dace.libraries.ai.nodes import AITasklet
from dace.sdfg import SDFG, SDFGState, nodes
from dace.transformation.transformation import ExpandTransformation


class ExpandAI(ExpandTransformation):
    """
    Expands a library node into a tasklet written by a language model, given the node, its
    connectors and containers, the enclosing scopes, and the target (see :mod:`~dace.libraries.ai.context`).
    """

    environments: list[Any] = []

    @staticmethod
    @functools.cache
    def for_node_class(node_class: type[nodes.LibraryNode]) -> type["ExpandAI"]:
        """
        Returns the expansion class of a library node type (``ExpandTransformation`` matches nodes
        through the class attribute ``_match_node``).

        :param node_class: The library node class being expanded.
        :return: A subclass of :class:`ExpandAI` whose ``_match_node`` is ``node_class``.
        """
        subclass = type(f"ExpandAI_{node_class.__name__}", (ExpandAI,), {"environments": []})
        subclass = dace.library.expansion(subclass)
        subclass._match_node = node_class
        return subclass

    @classmethod
    def expansion(cls, node: nodes.LibraryNode, parent_state: SDFGState, parent_sdfg: SDFG, **kwargs) -> nodes.Tasklet:
        """
        Generates a tasklet for a library node.

        :param cls: The expansion class from :meth:`for_node_class`. The environments of the
                    generated code are set on it, where ``ExpandTransformation.apply`` reads them.
        :param node: The library node to replace.
        :param parent_state: The state containing the node.
        :param parent_sdfg: The SDFG containing the state.
        :param kwargs: ``feedback`` and ``session`` start a new round of an existing session
                       (:func:`dace.libraries.ai.iterate.refine`). ``response`` is a
                       :class:`~dace.libraries.ai.backend.TaskletSpec` to use instead of asking a
                       provider (:func:`dace.libraries.ai.exchange.read_prompt_response`).
        :return: The generated tasklet.
        :raises AIExpansionError: If no tasklet could be generated.
        """
        feedback = kwargs.get("feedback") or ""
        response = kwargs.get("response")
        described = f'{type(node).__name__} "{node.name}"'
        ctx = collect_context(node, parent_state, parent_sdfg)
        record = ai_session.begin(node, parent_state, parent_sdfg, session_id=kwargs.get("session"))
        record.record_node(node.to_json(parent_state))
        record.begin_round(feedback)
        if record.path:
            status(f"{described}: session {record.id}, round {record.number} ({record.path})")

        # Created on first use, so that cached answers need no SDK or key
        provider = None
        system = prompts.SYSTEM_PROMPT
        record.record_system_prompt(system)
        detail("system prompt", system)

        # A round with feedback continues the conversation
        if feedback and record.conversation:
            messages: list[dict[str, str]] = list(record.conversation)
            prompt = prompts.build_feedback_prompt(feedback, prompts.build_user_prompt(ctx))
        else:
            messages = []
            prompt = prompts.build_user_prompt(ctx)
            if feedback:
                prompt = prompts.build_feedback_prompt(feedback, prompt, standalone=True)
        messages.append({"role": "user", "content": prompt})

        attempts = max(0, int(Config.get("ai", "max_repair_attempts"))) + 1
        should_check = Config.get_bool("ai", "compile_check")

        spec = None
        environments: list[Any] = []
        written: frozenset[str] = frozenset()  # Environments this expansion created, which repairs may redefine
        try:
            for attempt in range(attempts):
                entry = record.begin_attempt(attempt + 1, prompt)
                detail("user prompt" if attempt == 0 else "repair prompt", prompt)

                # A new round without feedback must not get the previous round's cached answer
                salt = str(record.number) if not feedback and record.number > 1 else ""
                cache_key = ai_cache.key(system, messages, salt=salt)
                spec = None if response is not None else ai_cache.lookup(cache_key)
                cached = spec is not None
                if response is not None:
                    status(f"{described}: using the supplied response")
                    spec = response
                    ai_cache.store(cache_key, spec, system, messages)
                elif cached:
                    status(
                        f"{described}: reusing the cached answer for this prompt ({cache_key[:12]}), "
                        f"attempt {attempt + 1}/{attempts}"
                    )
                else:
                    status(
                        f"{described}: asking {Config.get('ai', 'provider')} "
                        f"({Config.get('ai', 'model')}), attempt {attempt + 1}/{attempts}"
                    )
                    provider = provider or backend.get_provider()
                    spec = provider.generate(system, messages, node_name=node.name)
                    ai_cache.store(cache_key, spec, system, messages)
                record.record_answer(entry, spec, cached=cached)
                detail("answer", spec.raw_response or spec.model_dump_json(indent=1))
                try:
                    environments = ai_environments.collect(spec, written)
                    written |= {env.name for env in spec.environments}
                except ai_environments.EnvironmentConflict as e:
                    status(f"{described}: {e}")
                    problem = str(e)
                    prompt = prompts.build_rejection_prompt(problem)
                else:
                    if not should_check:
                        status(
                            f"{described}: the compile check is off (ai.compile_check), taking the code as generated"
                        )
                        break

                    check_folder = os.path.join(ai_cache.storage_dir("check"), *record.id.split("/"))
                    result = verify.compile_check(
                        ExpandAI._make_tasklet(node, spec, ctx),
                        node,
                        parent_state,
                        parent_sdfg,
                        ctx,
                        environments,
                        check_folder,
                    )
                    record.record_check(entry, result)
                    if result.skipped:
                        status(f"{described}: compile check skipped ({result.skipped}), taking the code as generated")
                        break
                    if result.ok:
                        status(f"{described}: compile check passed")
                        break

                    status(f"{described}: compile check failed")
                    detail("compiler diagnostics", result.diagnostics)
                    problem = f"The code does not compile. Diagnostics:\n\n{result.diagnostics}"
                    prompt = prompts.build_repair_prompt(result.diagnostics, result.sources)

                if response is not None:
                    # Not sent for repair: whoever supplied it chose not to use a provider
                    raise AIExpansionError(f"The supplied response for {described} cannot be used. {problem}")
                if attempt == attempts - 1:
                    raise AIExpansionError(
                        f"The generated code for {described} is still not usable after {attempts} attempt(s). {problem}"
                    )
                status(f"{described}: asking for a repair (attempt {attempt + 2}/{attempts})")
                messages.append({"role": "assistant", "content": spec.model_dump_json(indent=1)})
                messages.append({"role": "user", "content": prompt})
        except Exception as e:
            record.record_outcome("failed", str(e))
            if record.path is None:
                raise
            if isinstance(e, AIExpansionError):
                raise AIExpansionError(f"{e}\n\nThe full prompts and answers are in {record.path}") from e
            warnings.warn(f"AI expansion of {described} failed; the prompts and answers are in {record.path}")
            raise

        cls.environments = list(environments)
        env_paths = [env.full_class_path() for env in environments]
        record.record_outcome("expanded", spec=spec, environments=env_paths)

        record.conversation = messages + [{"role": "assistant", "content": spec.model_dump_json(indent=1)}]
        record.flush()

        status(
            f"{described}: expanded into a tasklet"
            + (f" using {len(environments)} environment(s)" if environments else "")
        )
        return ExpandAI._make_tasklet(node, spec, ctx, parent_state, record)

    @staticmethod
    def _make_tasklet(
        node: nodes.LibraryNode, spec: Any, ctx: Any, state: SDFGState = None, record: Any = None
    ) -> nodes.Tasklet:
        """
        Turns a generated specification into a tasklet, with the connector types the model was
        told. Inference would type a single element of GPU memory by value on a tasklet, since its
        pointer rule only applies to library nodes. The library node is recorded on the tasklet
        for :func:`~dace.libraries.ai.iterate.refine`.

        :param node: The library node being replaced.
        :param spec: The generated tasklet specification.
        :param ctx: The context the tasklet was generated for.
        :param state: The state containing the node, needed to serialize it.
        :param record: The session this round belongs to.
        :return: The tasklet that replaces the library node.
        """
        inputs = dict(node.in_connectors)
        outputs = dict(node.out_connectors)
        for conn in ctx.connectors:
            if conn.conntype is not None:
                (inputs if conn.direction == "in" else outputs)[conn.name] = conn.conntype

        language = dtypes.Language.Python if spec.language.upper() == "PYTHON" else dtypes.Language.CPP
        tasklet = AITasklet(
            node.name,
            inputs=inputs,
            outputs=outputs,
            code=spec.code,
            language=language,
            state_fields=list(spec.state_fields),
            code_global=spec.code_global,
            code_init=spec.code_init,
            code_exit=spec.code_exit,
            side_effects=spec.side_effects or None,
            ignored_symbols=set(spec.ignored_symbols),
        )
        if record is not None and state is not None:
            iterate.stamp(tasklet, session=record.id, round_number=record.number, node_json=node.to_json(state))
        if spec.notes:
            status(f'{type(node).__name__} "{node.name}": {spec.notes}')
        return tasklet


def status(message: str) -> None:
    """
    Prints a step of the expansion when ``debugprint`` is set.

    :param message: The message.
    """
    if Config.get_bool("debugprint"):
        print(f"[ai] {message}", flush=True)


def detail(label: str, body: str | None) -> None:
    """
    Prints a block of text, e.g. a prompt, when ``debugprint`` is ``verbose``.

    :param label: A short heading, e.g. ``'user prompt'``.
    :param body: The text. Nothing is printed when it is empty.
    """
    if Config.get("debugprint") != "verbose" or not body or not body.strip():
        return
    print(f"[ai] --- {label} ---", flush=True)
    print(textwrap.indent(body.rstrip("\n"), "    "), flush=True)
    print(f"[ai] --- end {label} ---", flush=True)
