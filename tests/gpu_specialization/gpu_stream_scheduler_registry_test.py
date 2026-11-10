# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Strategy-selection tests.

The strategy is chosen via the pipeline constructor argument
``GPUStreamPipeline(scheduling_strategy=...)``. This file pins the
selection contract.
"""

import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (
    GPUStreamSchedulingStrategy,
    PerComponentGPUStreamScheduler,
)


def test_pipeline_accepts_user_defined_strategy():
    """A user-defined strategy that subclasses the base class is accepted."""

    class DummyScheduler(GPUStreamSchedulingStrategy):
        def assign_streams(self, sdfg) -> dict[nodes.Node, int]:
            return {}

        def insert_sync_tasklets(self, sdfg, assignments):
            pass

    pipe = GPUStreamPipeline(scheduling_strategy=DummyScheduler())
    assert isinstance(pipe._scheduling_strategy, DummyScheduler)


def test_apply_pass_rejects_non_root_sdfg():
    """Stream scheduling must run on the root SDFG only."""
    outer = dace.SDFG("outer")
    inner = dace.SDFG("inner")
    inner._parent_sdfg = outer
    with pytest.raises(ValueError, match="root SDFG"):
        PerComponentGPUStreamScheduler().apply_pass(inner, {})


if __name__ == "__main__":
    test_pipeline_accepts_user_defined_strategy()
    test_apply_pass_rejects_non_root_sdfg()
