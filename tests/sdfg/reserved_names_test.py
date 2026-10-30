# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPU stream pipeline refuses to reuse a user array that squats on its reserved name."""

import pytest

import dace
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import allocate_stream_array
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import get_gpu_stream_array_name


def test_reserved_name_collision_rejected():
    sdfg = dace.SDFG("reserved_name_collision")
    sdfg.add_array(get_gpu_stream_array_name(), [4], dace.float64)
    with pytest.raises(NameError):
        allocate_stream_array(sdfg, 4)


if __name__ == "__main__":
    test_reserved_name_collision_rejected()
