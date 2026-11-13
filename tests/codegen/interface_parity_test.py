# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that must behave identically on BOTH Python <-> C interfaces.

Every test takes the ``compiler_interface`` fixture and therefore runs once
under ``ctypes`` and once under ``nanobind``.
"""

import numpy as np
import pytest

import dace
from dace.config import set_temporary


@pytest.fixture(params=["ctypes", "nanobind"])
def compiler_interface(request):
    with set_temporary("compiler", "interface", value=request.param):
        yield request.param


def test_vector_return_exotic_strides(compiler_interface):
    """A vector-dtype return with PERMUTED, padded strides allocates
    correctly: the lane dimension is appended trailing with scalar stride 1,
    wherever the vector-stride-1 dimension sits. Input and output use
    DIFFERENT strides and the copy goes through a MAP - an identical-layout
    copy would lower to one memcpy and never exercise the per-dimension
    addressing."""
    vec2 = dace.vector(dace.float64, 2)
    sdfg = dace.SDFG(f"vec_ret_exotic_{compiler_interface}")
    sdfg.add_array("A", [2, 3, 4], vec2, strides=[12, 4, 1], total_size=24)
    sdfg.add_array("__return", [2, 3, 4], vec2, strides=[3, 1, 12], total_size=48)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "copy",
        dict(i="0:2", j="0:3", k="0:4"),
        dict(inp=dace.Memlet("A[i, j, k]")),
        "out = inp",
        dict(out=dace.Memlet("__return[i, j, k]")),
        external_edges=True,
    )

    csdfg = sdfg.compile()

    # Dense C-order input: vector strides (12, 4, 1) = a contiguous
    # (2, 3, 4, 2) float64 array.
    a = np.arange(48, dtype=np.float64).reshape(2, 3, 4, 2)

    result = csdfg(A=a)
    # The lane dimension trails with scalar stride 1; the descriptor's
    # (vector) strides scale to bytes: (3*16, 1*16, 12*16, 8).
    assert result.shape == (2, 3, 4, 2)
    assert result.strides == (48, 16, 192, 8)
    # Element-wise equality across the two different memory layouts: correct
    # only if the returned array's stride metadata matches the addressing the
    # kernel actually wrote with.
    assert np.array_equal(result, a)


def test_is_initialized_lifecycle(compiler_interface):
    """``is_initialized`` reports the state lifecycle identically on both
    interfaces: False after compile, True after ``initialize()``, False after
    ``finalize()``. Library code (e.g. the torch dispatchers) consults it
    instead of the ctypes-private ``_initialized`` attribute."""
    sdfg = dace.SDFG(f"is_init_prog_{compiler_interface}")
    sdfg.add_array("A", [4], dace.float64)
    state = sdfg.add_state()
    t = state.add_tasklet("t", {"i"}, {"o"}, "o = i + 1.0")
    state.add_edge(state.add_read("A"), None, t, "i", dace.Memlet("A[0]"))
    state.add_edge(t, "o", state.add_write("A"), None, dace.Memlet("A[0]"))

    csdfg = sdfg.compile()
    assert csdfg.is_initialized is False
    csdfg.initialize(A=np.zeros(4))
    assert csdfg.is_initialized is True
    csdfg.finalize()
    assert csdfg.is_initialized is False


if __name__ == "__main__":
    for interface in ("ctypes", "nanobind"):
        with set_temporary("compiler", "interface", value=interface):
            test_vector_return_exotic_strides(interface)
