# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""finalize_transient_storage turns an all-ones transient ((1, 1), (1, 1, 1)) into a scalar, as it does a (1,) one.

ShrinkMapLocalTransients shrinks a per-iteration temporary to the box it names, so a 2-D scratch buffer indexed at
[i, j] inside the map lands on (1, 1); left an array, CPF declares and indexes a stack array per iteration (icon
gather buffers, lulesh, cp2k in the SciComp review of 11-23).
"""

import numpy as np

import dace
import dace.data as dd
from dace.transformation.passes.canonicalize.finalize import finalize_transient_storage


def two_d_scratch_sdfg() -> dace.SDFG:
    """``b[i] = (a[i] * 2) + 1`` through a ``(1, 1)`` transient ``tmp`` inside the map."""
    sdfg = dace.SDFG("single_element_scratch")
    sdfg.add_array("a", [16], dace.float64)
    sdfg.add_array("b", [16], dace.float64)
    sdfg.add_array("tmp", [1, 1], dace.float64, transient=True)
    state = sdfg.add_state("main")
    entry, exit_ = state.add_map("m", {"i": "0:16"})
    first = state.add_tasklet("first", {"x"}, {"y"}, "y = x * 2.0")
    second = state.add_tasklet("second", {"x"}, {"y"}, "y = x + 1.0")
    read, write, tmp = state.add_read("a"), state.add_write("b"), state.add_access("tmp")
    state.add_memlet_path(read, entry, first, dst_conn="x", memlet=dace.Memlet("a[i]"))
    state.add_edge(first, "y", tmp, None, dace.Memlet("tmp[0, 0]"))
    state.add_edge(tmp, None, second, "x", dace.Memlet("tmp[0, 0]"))
    state.add_memlet_path(second, exit_, write, src_conn="y", memlet=dace.Memlet("b[i]"))
    return sdfg


def test_all_ones_transient_becomes_scalar():
    sdfg = two_d_scratch_sdfg()
    finalize_transient_storage(sdfg, dace.DeviceType.CPU)
    assert isinstance(sdfg.arrays["tmp"], dd.Scalar)
    sdfg.validate()
    a = np.arange(16, dtype=np.float64)
    b = np.zeros(16, dtype=np.float64)
    sdfg(a=a, b=b)
    assert np.allclose(b, a * 2.0 + 1.0)


if __name__ == "__main__":
    test_all_ones_transient_becomes_scalar()
