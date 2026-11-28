# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A host transient that only kernels and copies touch goes to device memory (SC40 CUDA icon_velocity_tendencies)."""

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize.finalize import place_kernel_buffers_on_device


def block_loop_sdfg(host_reads: bool = False) -> dace.SDFG:
    """``out[b, 3:N+3] = 2 * a[b, :]`` per block ``b``: a host loop over the blocks launches a kernel into ``tmp``, and
    ``tmp`` is copied out through the loop's exit. ``host_reads`` adds a host tasklet reading ``tmp[0]``."""
    sdfg = dace.SDFG("block_loop_kernel_buffer" + ("_host_reads" if host_reads else ""))
    sdfg.add_array("a", ["NB", "N"], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("out", ["NB", "N + 3"], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("first", ["NB"], dace.float64)
    sdfg.add_transient("tmp", ["N"], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map("blocks", {"b": "0:NB"}, schedule=dace.ScheduleType.Sequential)
    entry, exit_ = state.add_map("kernel", {"j": "0:N"}, schedule=dace.ScheduleType.GPU_Device)
    twice = state.add_tasklet("twice", {"x"}, {"y"}, "y = 2 * x")
    tmp = state.add_access("tmp")
    state.add_memlet_path(state.add_read("a"), outer_entry, entry, twice, dst_conn="x", memlet=dace.Memlet("a[b, j]"))
    state.add_memlet_path(twice, exit_, tmp, src_conn="y", memlet=dace.Memlet("tmp[j]"))
    state.add_memlet_path(
        tmp,
        outer_exit,
        state.add_write("out"),
        memlet=dace.Memlet(data="out", subset="b, 3:N + 3", other_subset="0:N"),
    )
    if host_reads:
        peek = state.add_tasklet("peek", {"x"}, {"y"}, "y = x")
        state.add_edge(tmp, None, peek, "x", dace.Memlet("tmp[0]"))
        state.add_memlet_path(peek, outer_exit, state.add_write("first"), src_conn="y", memlet=dace.Memlet("first[b]"))
    else:
        state.add_memlet_path(outer_entry, outer_exit, memlet=dace.Memlet())
    return sdfg


def test_a_buffer_between_a_kernel_and_a_copy_goes_to_device_memory():
    sdfg = block_loop_sdfg()

    moved = place_kernel_buffers_on_device(sdfg)

    assert moved == [f"{sdfg.name}.tmp"]
    assert sdfg.arrays["tmp"].storage == dace.StorageType.GPU_Global


def test_a_buffer_host_code_reads_stays_on_the_host():
    sdfg = block_loop_sdfg(host_reads=True)

    assert place_kernel_buffers_on_device(sdfg) == []
    assert sdfg.arrays["tmp"].storage == dace.StorageType.Default


@pytest.mark.gpu
def test_the_moved_buffer_computes_what_numpy_computes():
    """Before the move the kernel wrote host memory: an illegal memory access."""
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.

    sdfg = block_loop_sdfg()
    place_kernel_buffers_on_device(sdfg)
    blocks, length = 3, 1000
    a = np.random.default_rng(1).random((blocks, length))
    out = cupy.zeros((blocks, length + 3))

    sdfg(a=cupy.asarray(a), out=out, first=np.zeros(blocks), NB=blocks, N=length)

    np.testing.assert_allclose(cupy.asnumpy(out)[:, 3:], 2 * a, rtol=0, atol=0)


if __name__ == "__main__":
    test_a_buffer_between_a_kernel_and_a_copy_goes_to_device_memory()
    test_a_buffer_host_code_reads_stays_on_the_host()
    test_the_moved_buffer_computes_what_numpy_computes()
