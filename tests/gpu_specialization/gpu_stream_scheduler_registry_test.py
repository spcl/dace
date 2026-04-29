# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Strategy-selection tests.

The strategy is chosen via the pipeline constructor argument
``GPUStreamPipeline(scheduling_strategy=…)``. This file pins the
selection contract.
"""
from typing import Dict

import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (GPUStreamSchedulingStrategy,
                                                                                 MonolithicSingleStreamGPUScheduler,
                                                                                 NaiveGPUStreamScheduler)

# ---------------------------------------------------------------------------
# Pipeline-level config
# ---------------------------------------------------------------------------


def test_pipeline_default_strategy_is_naive():
    pipe = GPUStreamPipeline()
    assert isinstance(pipe._scheduling_strategy, NaiveGPUStreamScheduler)


def test_pipeline_accepts_explicit_strategy_instance():
    strategy = MonolithicSingleStreamGPUScheduler()
    pipe = GPUStreamPipeline(scheduling_strategy=strategy)
    assert pipe._scheduling_strategy is strategy


def test_pipeline_rejects_non_strategy_argument():
    with pytest.raises(TypeError, match="GPUStreamSchedulingStrategy"):
        GPUStreamPipeline(scheduling_strategy="not a strategy")


def test_pipeline_accepts_user_defined_strategy():
    """A user-defined strategy that subclasses the base class is accepted."""

    class DummyScheduler(GPUStreamSchedulingStrategy):

        def assign_streams(self, sdfg) -> Dict[nodes.Node, int]:
            return {}

        def insert_sync_tasklets(self, sdfg, assignments) -> None:
            pass

    pipe = GPUStreamPipeline(scheduling_strategy=DummyScheduler())
    assert isinstance(pipe._scheduling_strategy, DummyScheduler)


# ---------------------------------------------------------------------------
# Strategy contract
# ---------------------------------------------------------------------------


def test_abstract_assign_streams_raises():
    """The base class enforces the contract — a strategy must override
    ``assign_streams``."""
    with pytest.raises(NotImplementedError, match="assign_streams"):
        GPUStreamSchedulingStrategy().assign_streams(dace.SDFG('abc'))


def test_abstract_apply_pass_also_raises():
    """``apply_pass`` routes through ``assign_streams``, so the same
    contract holds when the pass machinery invokes the strategy."""
    with pytest.raises(NotImplementedError):
        GPUStreamSchedulingStrategy().apply_pass(dace.SDFG('abc'), {})


def test_apply_pass_rejects_non_root_sdfg():
    """Stream scheduling must run on the root SDFG only."""
    outer = dace.SDFG('outer')
    inner = dace.SDFG('inner')
    inner._parent_sdfg = outer
    with pytest.raises(ValueError, match="root SDFG"):
        NaiveGPUStreamScheduler().apply_pass(inner, {})


def test_naive_assign_streams_callable_directly():
    """The naive scheduler must keep working when invoked directly."""
    sdfg = dace.SDFG('empty')
    sdfg.add_state('s')
    assignments = NaiveGPUStreamScheduler().assign_streams(sdfg)
    assert isinstance(assignments, dict)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
