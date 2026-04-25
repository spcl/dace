# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the pluggable GPU stream-scheduling strategy.

The pipeline ships :class:`NaiveGPUStreamScheduler` as its default. Users with
specialised needs (e.g. annotation-driven scheduling, single-stream
reproducibility, priority-based queues) plug in their own strategy by
subclassing :class:`GPUStreamSchedulingStrategy` and overriding ``assign``,
then registering it via :func:`register_gpu_stream_scheduler`. ``None``
clears the registration and falls back to the naive default.
"""
from typing import Dict

import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (
    GPUStreamSchedulingStrategy,
    NaiveGPUStreamScheduler,
    get_gpu_stream_scheduler,
    register_gpu_stream_scheduler,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """ The registry is process-global; every test starts from the default. """
    register_gpu_stream_scheduler(None)
    yield
    register_gpu_stream_scheduler(None)


def test_default_is_naive():
    assert isinstance(get_gpu_stream_scheduler(), NaiveGPUStreamScheduler)


def test_abstract_assign_raises():
    """ Calling ``assign`` on the base class is a contract violation -- it has
        to be a NotImplementedError so the user gets a precise message. """
    with pytest.raises(NotImplementedError, match='Subclass'):
        GPUStreamSchedulingStrategy().assign(dace.SDFG('abc'))


def test_abstract_apply_pass_also_raises():
    """ ``apply_pass`` routes through ``assign``, so the same contract holds
        when the pass machinery invokes the strategy. """
    with pytest.raises(NotImplementedError):
        GPUStreamSchedulingStrategy().apply_pass(dace.SDFG('abc'), {})


def test_register_custom_strategy():
    """ A user-registered strategy supersedes the default. """

    class SingleStreamScheduler(GPUStreamSchedulingStrategy):
        """Pin every node to stream 0 -- useful for debugging."""
        def assign(self, sdfg) -> Dict[nodes.Node, int]:
            return {n: 0 for s in sdfg.states() for n in s.nodes()}

    register_gpu_stream_scheduler(SingleStreamScheduler)
    s = get_gpu_stream_scheduler()
    assert isinstance(s, SingleStreamScheduler)


def test_register_none_restores_default():
    class Dummy(GPUStreamSchedulingStrategy):
        def assign(self, sdfg) -> Dict[nodes.Node, int]:
            return {}

    register_gpu_stream_scheduler(Dummy)
    assert isinstance(get_gpu_stream_scheduler(), Dummy)
    register_gpu_stream_scheduler(None)
    assert isinstance(get_gpu_stream_scheduler(), NaiveGPUStreamScheduler)


def test_register_rejects_non_strategy_class():
    """ Refuse to register classes that don't implement the strategy
        interface -- catches the typo case where a user passes a bare class
        and wonders why the pipeline blows up later. """
    class NotAStrategy:
        pass

    with pytest.raises(TypeError, match='GPUStreamSchedulingStrategy'):
        register_gpu_stream_scheduler(NotAStrategy)


def test_naive_still_callable_directly():
    """ The naive scheduler must keep working when invoked directly so that
        the existing pipeline (which constructs it explicitly) is unaffected
        by the registry being introduced. """
    sdfg = dace.SDFG('empty')
    sdfg.add_state('s')
    assignments = NaiveGPUStreamScheduler().assign(sdfg)
    assert isinstance(assignments, dict)


def test_get_returns_fresh_instance():
    """ Each call returns a fresh instance so internal state (e.g. caches in
        future strategies) is not shared across pipeline runs. """
    a = get_gpu_stream_scheduler()
    b = get_gpu_stream_scheduler()
    assert a is not b
    assert type(a) is type(b)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
