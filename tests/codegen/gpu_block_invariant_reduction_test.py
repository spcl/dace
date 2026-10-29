# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A kernel's map-exit WCR into an element indexed by a parameter every thread of a block shares folds by
``gpucub::BlockReduce`` and one atomic per block, as a scalar accumulator does; one indexed by a parameter the
block's threads vary is a scatter. Asserted on the generated code, without a GPU."""

import re

import numpy as np
import pytest

import dace


def row_sums(block: list[int]) -> dace.SDFG:
    """``out[i] += A[i, j]`` over a ``GPU_Device`` map ``(i, j)`` launched with ``block``."""
    sdfg = dace.SDFG(f"row_sums_{'x'.join(map(str, block))}")
    sdfg.add_array("A", ["N", "M"], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("out", ["N"], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    _, _, exit_node = state.add_mapped_tasklet(
        "row_sum",
        {"i": "0:N", "j": "0:M"},
        {"a": dace.Memlet("A[i, j]")},
        "o = a",
        {"o": dace.Memlet("out[i]", wcr="lambda x, y: x + y")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    entry = state.entry_node(exit_node)
    entry.map.gpu_block_size = block
    return sdfg


def row_forces(block: list[int]) -> dace.SDFG:
    """``f[i, c] += A[i, j] * w[c]`` for the three components ``c``, written as three WCR edges, launched with
    ``block``."""
    sdfg = dace.SDFG(f"row_forces_{'x'.join(map(str, block))}")
    sdfg.add_array("A", ["N", "M"], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("f", ["N", 3], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    _, entry, _ = state.add_mapped_tasklet(
        "row_force",
        {"i": "0:N", "j": "0:M"},
        {"a": dace.Memlet("A[i, j]")},
        "fx = a\nfy = 2 * a\nfz = 3 * a",
        {c: dace.Memlet(f"f[i, {k}]", wcr="lambda x, y: x + y") for k, c in enumerate(("fx", "fy", "fz"))},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    entry.map.gpu_block_size = block
    return sdfg


def kernel_code(sdfg: dace.SDFG) -> str:
    return "\n".join(obj.clean_code for obj in sdfg.generate_code() if "__global__" in obj.clean_code)


def test_a_row_shared_by_the_block_folds_once_per_block():
    """Block ``256 x 1``: ``i`` (the ``y`` axis) is the block's own, so the row is one accumulator for all of it."""
    code = kernel_code(row_sums([256, 1, 1]))
    assert "BlockReduce" in code, code
    assert "reduce_atomic(out + ((i) + __bk_" in code, "the block's sum goes to its own row, once"


def test_a_row_the_block_spans_is_a_scatter():
    """Block ``16 x 16``: the threads of a block span sixteen rows, so no one accumulator holds them."""
    code = kernel_code(row_sums([16, 16, 1]))
    assert "BlockReduce" not in code, code


def test_the_components_of_a_shared_row_fold_as_one_span():
    """Three WCR edges into ``f[i, 0:3]``: one partial of three slots, drained by three atomics per block."""
    code = kernel_code(row_forces([256, 1, 1]))
    assert re.search(r"double __bpart_\w+\[3\];", code), code
    assert code.count("reduce_atomic(f + ") == 1, "every component drains through the one block fold"


@pytest.mark.gpu
def test_a_row_shared_by_the_block_sums_as_numpy():
    import cupy as cp

    sdfg = row_sums([256, 1, 1])
    A = np.random.default_rng(3).random((7, 1000))
    out = cp.zeros(7)
    sdfg(A=cp.asarray(A), out=out, N=7, M=1000)
    assert np.allclose(cp.asnumpy(out), A.sum(axis=1), rtol=1e-12)


if __name__ == "__main__":
    test_a_row_shared_by_the_block_folds_once_per_block()
    test_a_row_the_block_spans_is_a_scatter()
    test_the_components_of_a_shared_row_fold_as_one_span()
