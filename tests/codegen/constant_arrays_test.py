# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace


def test_nsdfg_input():
    """Tests constexpr array passed as input argument to a NestedSDFG."""

    @dace.program
    def constexpr_nsdfg():
        a = np.array([1.0, 2.0, 3.0])
        b = np.max(a)

    with dace.config.set_temporary("compiler", "inline_sdfgs", value=False):
        constexpr_nsdfg()


def test_tasklet_input_cpu():
    """Tests constexpr array passed as input argument to a Tasklet (CPU)."""

    @dace.program
    def constexpr_tasklet_cpu():
        a = np.array([1.0, 2.0, 3.0])
        b = np.max(a)

    with dace.config.set_temporary("optimizer", "autooptimize", value=True):
        constexpr_tasklet_cpu()


def test_a_column_major_constant_is_read_by_index():
    """A constant is written in its descriptor's memory order: written in index order, a column-major table was read
    as its transpose."""
    table = np.arange(6, dtype=np.float64).reshape(2, 3)
    sdfg = dace.SDFG("column_major_constant")
    sdfg.add_array("T", [2, 3], dace.float64, strides=[1, 2], transient=True)
    sdfg.add_constant("T", table, sdfg.arrays["T"])
    sdfg.add_array("out", [2, 3], dace.float64)
    sdfg.add_state().add_mapped_tasklet(
        "read",
        {"i": "0:2", "j": "0:3"},
        {"t": dace.Memlet("T[i, j]")},
        "o = t",
        {"o": dace.Memlet("out[i, j]")},
        external_edges=True,
    )
    out = np.zeros((2, 3))

    sdfg(out=out)

    np.testing.assert_array_equal(out, table)


if __name__ == "__main__":
    test_nsdfg_input()
    test_tasklet_input_cpu()
    test_a_column_major_constant_is_read_by_index()
