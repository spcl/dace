# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A lifted scan reads its input in place when it is a plain read of an array the loop does not write, and
reads a ``_scan_in_`` copy when the loop computes it."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.scan import INPUT_CONNECTOR_NAME, Scan
from dace.transformation.passes.lift_preprocess import LiftPreprocess
from dace.transformation.passes.loop_to_scan import LoopToScan

N = dace.symbol("N")


def carried_sum(distance: int) -> dace.SDFG:
    @dace.program
    def program(a: dace.float64[N], x: dace.float64[N]):
        for i in range(distance, N):
            a[i] = a[i - distance] + x[i]

    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = f"direct_read_sum_{distance}"
    return sdfg


def carried_sum_of_products(distance: int) -> dace.SDFG:
    @dace.program
    def program(a: dace.float64[N], x: dace.float64[N], y: dace.float64[N]):
        for i in range(distance, N):
            a[i] = a[i - distance] + x[i] * y[i]

    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = f"copied_input_sum_{distance}"
    return sdfg


def lifted(sdfg: dace.SDFG) -> dace.SDFG:
    LiftPreprocess().apply_pass(sdfg, {})
    assert LoopToScan().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    return sdfg


def scan_inputs(sdfg: dace.SDFG) -> list[str]:
    return [
        edge.data.data
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, Scan)
        for edge in state.in_edges(node)
        if edge.dst_conn == INPUT_CONNECTOR_NAME
    ]


def copies(sdfg: dace.SDFG) -> list[str]:
    return [name for name in sdfg.arrays if name.startswith("_scan_in_")]


@pytest.mark.parametrize("distance", [1, 2, 3])
# Reproducer: LoopToScan: a scan whose input is a plain read of an array the loop does not write reads i... (5abf740137)
def test_a_scan_over_an_array_the_loop_does_not_write_reads_it_in_place(distance: int) -> None:
    sut = lifted(carried_sum(distance))
    assert scan_inputs(sut) == ["x"]
    assert copies(sut) == []
    rng = np.random.default_rng(distance)
    for n in (distance + 1, 4096 * 3 + 5, (1 << 17) + 3):
        x = rng.uniform(-1.0, 1.0, n)
        a = rng.uniform(-1.0, 1.0, n)
        expected = a.copy()
        for i in range(distance, n):
            expected[i] = expected[i - distance] + x[i]
        sut(a=a, x=x, N=n)
        np.testing.assert_allclose(a, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("distance", [1, 2])
# Reproducer: LoopToScan: a scan whose input is a plain read of an array the loop does not write reads i... (5abf740137)
def test_a_scan_over_a_computed_delta_reads_a_copy(distance: int) -> None:
    sut = lifted(carried_sum_of_products(distance))
    assert len(copies(sut)) == 1
    assert scan_inputs(sut) == copies(sut)
    rng = np.random.default_rng(distance)
    n = 4096 * 3 + 5
    x, y, a = rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n)
    expected = a.copy()
    for i in range(distance, n):
        expected[i] = expected[i - distance] + x[i] * y[i]
    sut(a=a, x=x, y=y, N=n)
    np.testing.assert_allclose(a, expected, rtol=1e-9, atol=1e-11)


def carried_sum_of_a_column(columns: int) -> dace.SDFG:
    @dace.program
    def program(a: dace.float64[N], x: dace.float64[N, columns]):
        for i in range(1, N):
            a[i] = a[i - 1] + x[i, 0]

    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = f"column_sum_{columns}"
    return sdfg


@pytest.mark.parametrize(("columns", "in_place"), [(1, True), (3, False)])
def test_a_scan_over_a_unit_stride_column_reads_it_in_place(columns: int, in_place: bool) -> None:
    """A scalar-expanded ``s[i, 0]`` of shape ``(N, 1)`` (the CPF running example) is one contiguous run and is read
    in place; a column of an ``(N, 3)`` array has stride 3 and keeps the copy."""
    sut = lifted(carried_sum_of_a_column(columns))
    assert scan_inputs(sut) == (["x"] if in_place else copies(sut))
    assert len(copies(sut)) == (0 if in_place else 1)
    rng = np.random.default_rng(columns)
    n = 4096 * 3 + 5
    x, a = rng.uniform(-1.0, 1.0, (n, columns)), rng.uniform(-1.0, 1.0, n)
    expected = a.copy()
    for i in range(1, n):
        expected[i] = expected[i - 1] + x[i, 0]
    sut(a=a, x=x, N=n)
    np.testing.assert_allclose(a, expected, rtol=1e-9, atol=1e-11)


if __name__ == "__main__":
    for distance in (1, 2, 3):
        test_a_scan_over_an_array_the_loop_does_not_write_reads_it_in_place(distance)
    for distance in (1, 2):
        test_a_scan_over_a_computed_delta_reads_a_copy(distance)
    for columns, in_place in ((1, True), (3, False)):
        test_a_scan_over_a_unit_stride_column_reads_it_in_place(columns, in_place)
