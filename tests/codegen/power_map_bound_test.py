# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map whose extent is a power of a symbol compiles under OpenMP: the bound is an integer, not ``pow``'s double."""

import numpy as np

import dace


def test_a_parallel_map_up_to_a_power_of_a_symbol_compiles():
    """spgemm_hash: ``for i in range(32 * 2**b)`` gave ``i < 32*pow(2, b)``, an invalid OpenMP predicate."""
    sdfg = dace.SDFG("power_map_bound")
    sdfg.add_symbol("b", dace.int64)
    sdfg.add_array("A", [64], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "fill",
        {"i": "0:2*2**b"},
        {},
        "out = 1.0",
        {"out": dace.Memlet("A[i]")},
        schedule=dace.ScheduleType.CPU_Multicore,
        external_edges=True,
    )
    A = np.zeros(64)
    sdfg(A=A, b=3)
    assert np.all(A[:16] == 1.0) and np.all(A[16:] == 0.0)


if __name__ == "__main__":
    test_a_parallel_map_up_to_a_power_of_a_symbol_compiles()
