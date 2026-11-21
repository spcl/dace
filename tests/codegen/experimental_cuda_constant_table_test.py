# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A constant array read inside a kernel is declared in the kernel body by the experimental codegen, not passed
as a kernel argument."""

import numpy as np
import pytest

import dace
from dace import dtypes

N = 16


def constant_table_sdfg() -> dace.SDFG:
    """A[i] = 2 * TAB[i] in one GPU kernel, TAB a constant array."""
    sdfg = dace.SDFG("constant_table_in_kernel")
    sdfg.add_constant("TAB", np.arange(N, dtype=np.float64))
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_transient("gpu_A", [N], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    gpu_a = state.add_access("gpu_A")
    entry = state.add_mapped_tasklet(
        "scale",
        {"i": f"0:{N}"},
        {},
        "a = TAB[i] * 2",
        {"a": dace.Memlet("gpu_A[i]")},
        schedule=dtypes.ScheduleType.GPU_Device,
        external_edges=True,
        output_nodes={"gpu_A": gpu_a},
    )[1]
    entry.map.gpu_block_size = [N, 1, 1]
    state.add_nedge(gpu_a, state.add_write("A"), dace.Memlet("A[0:16]"))
    sdfg.validate()
    return sdfg


def test_a_constant_table_is_declared_in_the_kernel():
    sdfg = constant_table_sdfg()
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        objects = sdfg.generate_code()
    kernels = [code.clean_code for code in objects if "__global__" in code.clean_code]
    assert len(kernels) == 1, "expected exactly one kernel"

    kernel = kernels[0]
    signature = kernel[kernel.index("__global__") : kernel.index("{", kernel.index("__global__"))]
    assert "TAB" not in signature, "the constant table is passed as a kernel argument"
    body = kernel[kernel.index("{", kernel.index("__global__")) :]
    assert "TAB[16]" in body, "the constant table is not declared inside the kernel"


@pytest.mark.gpu
def test_a_constant_table_is_read_inside_the_kernel():
    sdfg = constant_table_sdfg()
    a = np.zeros(N)

    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        sdfg(A=a)

    np.testing.assert_allclose(a, 2 * np.arange(N))


if __name__ == "__main__":
    test_a_constant_table_is_declared_in_the_kernel()
    test_a_constant_table_is_read_inside_the_kernel()
