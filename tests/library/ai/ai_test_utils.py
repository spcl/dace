# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Helpers shared by the tests of the AI library node expansion. """

import contextlib
from typing import Callable, Dict, List, Optional

from dace.libraries.ai import backend


class StubProvider:
    """
    A provider that returns a canned tasklet instead of querying a model.

    Lets the plumbing around the ``'ai'`` implementation -- resolution, context collection,
    verification, tasklet construction and code generation -- be tested offline.
    """

    def __init__(self, spec_factory: Callable[[str, List[Dict[str, str]]], backend.TaskletSpec]) -> None:
        """
        :param spec_factory: Called with the system prompt and the conversation; returns the
                             tasklet to pretend the model produced.
        """
        self._spec_factory = spec_factory
        self.calls: List[List[Dict[str, str]]] = []
        self.system: Optional[str] = None

    def generate(self, system: str, messages: List[Dict[str, str]]) -> backend.TaskletSpec:
        """
        Records the prompt and returns the canned tasklet.

        :param system: The system prompt.
        :param messages: The conversation so far.
        :return: The tasklet from the factory.
        """
        self.system = system
        self.calls.append(list(messages))
        return self._spec_factory(system, messages)


@contextlib.contextmanager
def stub_provider(spec: backend.TaskletSpec):
    """
    Installs a :class:`StubProvider` for the duration of the context.

    :param spec: The tasklet every generation request should return.
    :return: The installed provider, so that the prompts it saw can be inspected.
    """
    provider = StubProvider(lambda system, messages: spec)
    original = backend.get_provider
    backend.get_provider = lambda: provider
    try:
        yield provider
    finally:
        backend.get_provider = original


@contextlib.contextmanager
def stub_provider_sequence(specs: List[backend.TaskletSpec]):
    """
    Installs a provider that returns a different tasklet on each request.

    Used to exercise the repair loop, where the first response fails to compile and a later one
    succeeds. The last entry is repeated if more requests are made than there are entries.

    :param specs: The tasklets to return, in order.
    :return: The installed provider.
    """
    remaining = list(specs)

    def factory(system, messages):
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    provider = StubProvider(factory)
    original = backend.get_provider
    backend.get_provider = lambda: provider
    try:
        yield provider
    finally:
        backend.get_provider = original


def prompt_of(provider: StubProvider) -> str:
    """
    Returns the user prompt from a provider's first request.

    :param provider: The stub provider that was used.
    :return: The rendered user prompt.
    """
    return provider.calls[0][0]['content']
