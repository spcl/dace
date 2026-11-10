# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Strategy-selection tests.

The strategy is chosen via the pipeline constructor argument
``GPUStreamPipeline(scheduling_strategy=...)``. This file pins the
selection contract.
"""

import pytest

import dace
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (
    PerComponentGPUStreamScheduler,
)


def test_apply_pass_rejects_non_root_sdfg():
    """Stream scheduling must run on the root SDFG only."""
    outer = dace.SDFG("outer")
    inner = dace.SDFG("inner")
    inner._parent_sdfg = outer
    with pytest.raises(ValueError, match="root SDFG"):
        PerComponentGPUStreamScheduler().apply_pass(inner, {})


if __name__ == "__main__":
    test_apply_pass_rejects_non_root_sdfg()
