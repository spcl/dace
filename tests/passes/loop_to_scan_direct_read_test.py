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


if __name__ == "__main__":
    for distance in (1, 2, 3):
        test_a_scan_over_an_array_the_loop_does_not_write_reads_it_in_place(distance)
    for distance in (1, 2):
        test_a_scan_over_a_computed_delta_reads_a_copy(distance)
