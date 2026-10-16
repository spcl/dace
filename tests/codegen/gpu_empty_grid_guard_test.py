# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A GPU launch is skipped on an empty grid unless the facts at the kernel prove every grid dimension positive."""

import dace
from dace import symbolic


def kernel_code(positive: bool) -> str:
    sdfg = dace.SDFG(f"empty_grid_guard_{'positive' if positive else 'unknown'}")
    sdfg.add_symbol("N", dace.int64)
    if positive:
        sdfg.set_symbol_assumptions("N", frozenset({symbolic.Predicate.POSITIVE}))
    sdfg.add_array("A", ["N"], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    state.add_mapped_tasklet("fill", {"i": "0:N"}, {}, "a = 1.0", {"a": dace.Memlet("A[i]")}, external_edges=True)
    sdfg.apply_gpu_transformations()
    return next(code.clean_code for code in sdfg.generate_code() if code.title == "CUDA")


def empty_grid_checks(code: str) -> list[str]:
    return [line for line in code.splitlines() if "<= 0)" in line and "if (" in line]


def test_an_unproven_grid_dimension_keeps_the_empty_launch_check():
    assert empty_grid_checks(kernel_code(positive=False))


def test_a_grid_dimension_positive_by_the_facts_launches_unchecked():
    assert not empty_grid_checks(kernel_code(positive=True))


if __name__ == "__main__":
    test_an_unproven_grid_dimension_keeps_the_empty_launch_check()
    test_a_grid_dimension_positive_by_the_facts_launches_unchecked()
