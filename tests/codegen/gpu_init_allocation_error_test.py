# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A device allocation the initializer cannot satisfy fails the call, instead of handing a kernel a null buffer."""

import pytest

import dace
from dace import dtypes

N = dace.symbol("N", dtype=dace.int64)


def persistent_device_buffer() -> dace.SDFG:
    """A kernel writing one element of a persistent device buffer of ``N`` doubles, which the initializer
    allocates."""
    sdfg = dace.SDFG("gpu_init_allocation_error")
    sdfg.add_array(
        "big",
        [N],
        dace.float64,
        storage=dtypes.StorageType.GPU_Global,
        transient=True,
        lifetime=dtypes.AllocationLifetime.Persistent,
    )
    sdfg.add_array("out", [1], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("fill")
    big = state.add_access("big")
    state.add_mapped_tasklet(
        "one",
        {"i": "0:1"},
        {},
        "b = 1.0",
        {"b": dace.Memlet("big[i]")},
        schedule=dtypes.ScheduleType.GPU_Device,
        output_nodes={"big": big},
        external_edges=True,
    )
    state.add_nedge(big, state.add_write("out"), dace.Memlet("big[0]"))
    return sdfg


@pytest.mark.gpu
def test_an_initializer_whose_device_allocation_fails_raises():
    """2**50 doubles cannot be allocated: the call raises, naming the failed initialization."""
    import cupy

    out = cupy.zeros(1)
    compiled = persistent_device_buffer().compile()
    with pytest.raises(RuntimeError, match="failed to initialize"):
        compiled(out=out, N=2**50)


if __name__ == "__main__":
    test_an_initializer_whose_device_allocation_fails_raises()
