# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``numpy.cumsum`` / ``numpy.cumprod``, which lower onto the ``Scan`` library node.

A prefix scan is not a reduction -- every partial result stays visible -- so these cannot go
through ``Reduce``. Without the replacement the frontend fell back to a Python callback typed
``pyobject``, which parses and then fails to compile against a ``double``, so the structural
assertion below (a ``Scan`` node is actually in the graph) is the one that pins the fix: a
loop-lowered scan would satisfy the numeric ones just as well and hand the GPU backend the shape
only the CPU wants.
"""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.scan import Scan, ScanOp
from common import compare_numpy_output


@compare_numpy_output(positive=True)
def test_cumsum_1d(A: dace.float64[64]):
    return np.cumsum(A)


@compare_numpy_output(positive=True, max_value=2)
def test_cumprod_1d(A: dace.float64[16]):
    return np.cumprod(A)


@compare_numpy_output(positive=True)
def test_cumsum_last_axis(A: dace.float64[5, 40]):
    return np.cumsum(A, axis=1)


@compare_numpy_output(positive=True)
def test_cumsum_negative_axis(A: dace.float64[4, 3, 12]):
    return np.cumsum(A, axis=-1)


@compare_numpy_output(positive=True, max_value=2)
def test_cumprod_last_axis(A: dace.float64[6, 8]):
    return np.cumprod(A, axis=-1)


@compare_numpy_output(positive=True, check_dtype=True)
def test_cumsum_int32_accumulates_in_int64(A: dace.int32[32]):
    """numpy's cumulative rule widens every signed integer to int64, whatever the input width."""
    return np.cumsum(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_cumsum_uint8_accumulates_in_uint64(A: dace.uint8[32]):
    """...and every unsigned width to uint64, keeping the signedness."""
    return np.cumsum(A)


@compare_numpy_output(positive=True, check_dtype=True)
def test_cumsum_float32_keeps_its_dtype(A: dace.float32[32]):
    """A float is NOT widened -- the integer rule is specific to integers."""
    return np.cumsum(A, axis=0)


def scan_nodes(program):
    """Every ``Scan`` library node in ``program``'s parsed SDFG, recursively."""
    sdfg = program.to_sdfg(simplify=False)
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan)]


def test_cumsum_lowers_to_a_scan_libnode():
    """The point of the replacement: the op reaches the backends as a Scan, not as a callback and
    not as a hand-lowered loop the GPU expansion can no longer recognise."""

    @dace.program
    def prog(a: dace.float64[64], out: dace.float64[64]):
        out[:] = np.cumsum(a)

    nodes = scan_nodes(prog)
    assert len(nodes) == 1
    assert nodes[0].op == ScanOp.SUM
    assert not nodes[0].exclusive  # numpy's cumulative functions are inclusive


def test_cumprod_carries_the_product_op():
    """SUM and PRODUCT are the same node with one property apart; a copy-paste that left the op at
    its SUM default would still produce a Scan, and every numeric test above would still pass."""

    @dace.program
    def prog(a: dace.float64[16], out: dace.float64[16]):
        out[:] = np.cumprod(a)

    nodes = scan_nodes(prog)
    assert len(nodes) == 1
    assert nodes[0].op == ScanOp.PRODUCT


def test_a_batched_scan_keeps_the_batch_axis_parallel():
    """Over a rank > 1 operand the leading axes become a Map and the sequential recurrence stays
    inside it. Scanning the whole thing in one sequential nest would compute the same numbers and
    throw the only parallelism the op has."""

    @dace.program
    def prog(a: dace.float64[5, 40], out: dace.float64[5, 40]):
        out[:] = np.cumsum(a, axis=1)

    sdfg = prog.to_sdfg(simplify=False)
    scans = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan)]
    assert len(scans) == 1
    state = next(s for s in sdfg.states() if scans[0] in s.nodes())
    assert state.entry_node(scans[0]) is not None, "the scan is not inside the batch map"


def test_an_inner_axis_is_refused_rather_than_scanned_as_the_last():
    """axis=0 of a 2-D operand is a strided chain per column. Silently scanning the last axis
    instead would return the right SHAPE holding the wrong numbers."""

    @dace.program
    def prog(a: dace.float64[5, 40], out: dace.float64[5, 40]):
        out[:] = np.cumsum(a, axis=0)

    with pytest.raises(Exception, match="last axis only"):
        prog.to_sdfg()


def test_an_axis_less_scan_over_a_matrix_is_refused():
    """numpy FLATTENS here, which is a reshape only when the operand is contiguous."""

    @dace.program
    def prog(a: dace.float64[5, 40], out: dace.float64[200]):
        out[:] = np.cumsum(a)

    with pytest.raises(Exception, match="flattens"):
        prog.to_sdfg()


if __name__ == '__main__':
    test_cumsum_1d()
    test_cumprod_1d()
    test_cumsum_last_axis()
    test_cumsum_negative_axis()
    test_cumprod_last_axis()
    test_cumsum_int32_accumulates_in_int64()
    test_cumsum_uint8_accumulates_in_uint64()
    test_cumsum_float32_keeps_its_dtype()
    test_cumsum_lowers_to_a_scan_libnode()
    test_cumprod_carries_the_product_op()
    test_a_batched_scan_keeps_the_batch_axis_parallel()
    test_an_inner_axis_is_refused_rather_than_scanned_as_the_last()
    test_an_axis_less_scan_over_a_matrix_is_refused()
