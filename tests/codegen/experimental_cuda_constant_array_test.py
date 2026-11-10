# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A constant array read inside a kernel is declared in the kernel: device code cannot read a host ``constexpr``."""

import numpy as np
import pytest

import dace
from dace import dtypes

TABLE = np.array([1.0, 2.0, 0.5, -3.0])


def table_kernel_sdfg() -> dace.SDFG:
    """``B[i] = A[i] * table[i % 4]`` in a GPU_Device map, ``table`` an SDFG constant."""
    sdfg = dace.SDFG("experimental_cuda_constant_array")
    sdfg.add_constant("table", TABLE, dace.data.Array(dace.float64, [4]))
    for name in "AB":
        sdfg.add_array(name, [16], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "scale",
        {"i": "0:16"},
        {"a": dace.Memlet("A[i]")},
        "b = a * table[i % 4]",
        {"b": dace.Memlet("B[i]")},
        schedule=dtypes.ScheduleType.GPU_Device,
        external_edges=True,
    )
    return sdfg


@pytest.mark.gpu
def test_a_kernel_reads_a_constant_array():
    import cupy

    sdfg = table_kernel_sdfg()
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        device = next(code for code in sdfg.generate_code() if code.language == "cu").clean_code
        a = cupy.arange(16, dtype=cupy.float64)
        b = cupy.zeros(16, dtype=cupy.float64)
        sdfg(A=a, B=b)

    kernel = device[device.index("__global__") :]
    assert "constexpr double table[4]" in kernel
    np.testing.assert_allclose(b.get(), np.arange(16) * np.tile(TABLE, 4))


if __name__ == "__main__":
    test_a_kernel_reads_a_constant_array()
