# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Strategy-selection tests.

The strategy is chosen via the pipeline constructor argument
``GPUStreamPipeline(scheduling_strategy=...)``. This file pins the
selection contract.
"""

from typing import Dict

import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (
    AutoGPUStreamScheduler,
    GPUStreamSchedulingStrategy,
    PerComponentGPUStreamScheduler,
)

# Pipeline-level config.


def test_pipeline_default_strategy_is_auto():
    """The pipeline's default strategy is the auto/single-stream classifier; it falls back to
    :class:`PerComponentGPUStreamScheduler` internally when its analysis says so."""
    pipe = GPUStreamPipeline()
    assert isinstance(pipe._scheduling_strategy, AutoGPUStreamScheduler)


def test_pipeline_accepts_explicit_strategy_instance():
    strategy = AutoGPUStreamScheduler(monolithic=True)
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

        def insert_sync_tasklets(self, sdfg, assignments):
            pass

    pipe = GPUStreamPipeline(scheduling_strategy=DummyScheduler())
    assert isinstance(pipe._scheduling_strategy, DummyScheduler)


# Strategy contract.


def test_abstract_assign_streams_raises():
    """A strategy must override ``assign_streams`` (base class enforces it)."""
    with pytest.raises(NotImplementedError, match="assign_streams"):
        GPUStreamSchedulingStrategy().assign_streams(dace.SDFG("abc_abstract_assign_streams_raises"))


def test_abstract_apply_pass_also_raises():
    """``apply_pass`` routes through ``assign_streams``, so the contract holds
    via the pass machinery too."""
    with pytest.raises(NotImplementedError):
        GPUStreamSchedulingStrategy().apply_pass(dace.SDFG("abc_abstract_apply_pass_also_raises"), {})


def test_apply_pass_rejects_non_root_sdfg():
    """Stream scheduling must run on the root SDFG only."""
    outer = dace.SDFG("outer_apply_pass_rejects_non_root_sdfg")
    inner = dace.SDFG("inner_apply_pass_rejects_non_root_sdfg")
    inner._parent_sdfg = outer
    with pytest.raises(ValueError, match="root SDFG"):
        PerComponentGPUStreamScheduler().apply_pass(inner, {})


def test_per_component_assign_streams_callable_directly():
    """The per-component scheduler must keep working when invoked directly."""
    sdfg = dace.SDFG("empty_per_component_assign_streams_callable_directly")
    sdfg.add_state("s")
    assignments = PerComponentGPUStreamScheduler().assign_streams(sdfg)
    assert isinstance(assignments, dict)


if __name__ == "__main__":
    test_pipeline_default_strategy_is_auto()
    test_pipeline_accepts_explicit_strategy_instance()
    test_pipeline_rejects_non_strategy_argument()
    test_pipeline_accepts_user_defined_strategy()
    test_abstract_assign_streams_raises()
    test_abstract_apply_pass_also_raises()
    test_apply_pass_rejects_non_root_sdfg()
    test_per_component_assign_streams_callable_directly()
