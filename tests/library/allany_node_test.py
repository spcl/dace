# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``All`` / ``Any`` library nodes (Fortran ``ALL`` / ``ANY`` logical reductions): a dim-wise
reduce through the default ``reduction`` expansion, and a LOGICAL(4) result with non-1 .TRUE. mask values through both
the ``reduction`` and the short-circuiting ``sequential`` expansion."""

import ctypes

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import AllNode, AnyNode

# DaCe-compiled SOs link against libgomp; preload with RTLD_GLOBAL.
try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass


def _build_allany_sdfg(
    tag, op, mask_shape, mask_dtype, dim, out_shape, out_dtype, *, implementation="reduction", mask_subset=None
):
    """One-state SDFG wiring an ``All`` / ``Any`` node from a mask access into an
    output access.  ``mask_subset`` (list of ``(lo, hi)`` per dim, 0-based
    inclusive-exclusive) restricts the input edge to a section."""
    sdfg = dace.SDFG(f"allany_{tag}")
    sdfg.add_array("mask", mask_shape, mask_dtype, transient=False)
    out_shape_used = out_shape or [1]
    sdfg.add_array("out", out_shape_used, dace.bool_, transient=False)  # ALL/ANY return bool
    state = sdfg.add_state("s")

    node = (AllNode if op == "all" else AnyNode)("aa", dim=dim)
    node.implementation = implementation
    state.add_node(node)

    if mask_subset is None:
        msub = ", ".join(f"0:{s}" for s in mask_shape)
    else:
        msub = ", ".join(f"{lo}:{hi}" for (lo, hi) in mask_subset)
    osub = ", ".join(f"0:{s}" for s in out_shape_used)
    state.add_edge(state.add_access("mask"), None, node, AllNode.INPUT_CONNECTOR_NAME, dace.Memlet(f"mask[{msub}]"))
    state.add_edge(node, AllNode.OUTPUT_CONNECTOR_NAME, state.add_access("out"), None, dace.Memlet(f"out[{osub}]"))
    sdfg.validate()
    return sdfg


# ---------------------------------------------------------------------------
# reduction expansion -- whole-array reduce, 1-D
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# reduction expansion -- whole-array reduce, 2-D
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# reduction expansion -- per-dim reduce (dim=k Fortran 1-based)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op,dim", [("all", 1), ("all", 2), ("any", 1), ("any", 2)])
def test_reduction_dimwise_reduce(op, dim):
    rng = np.random.default_rng(2)
    mask = (rng.random((4, 5)) > 0.4).astype(np.int32)
    # Fortran dim=k reduces axis k (1-based); numpy axis = k-1.
    out_shape = [mask.shape[1]] if dim == 1 else [mask.shape[0]]
    sdfg = _build_allany_sdfg(f"dimr_{op}_{dim}", op, [4, 5], dace.int32, dim, out_shape, dace.int32)
    out = np.zeros(out_shape, dtype=np.bool_)
    sdfg(mask=mask.copy(), out=out)
    np_axis = dim - 1
    expected = (mask.all(axis=np_axis) if op == "all" else mask.any(axis=np_axis)).astype(np.int32)
    np.testing.assert_array_equal(out.astype(np.int32), expected)


# ---------------------------------------------------------------------------
# reduction expansion -- sectioned input
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# reduction expansion -- non-int (LOGICAL(1) <-> uint8) mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("implementation", ["reduction", "sequential"])
@pytest.mark.parametrize("op,expected", [("all", 1), ("any", 1)])
def test_logical4_output_and_foreign_true(op, expected, implementation):
    """A Fortran LOGICAL(4) result is 4-byte storage: the expansion writes the output's own dtype (0 / 1), and any
    non-zero mask value (-1, HUGE) is .TRUE."""
    mask = np.array([1, 0xFFFFFFFF, 0x7FFFFFFF, 2], dtype=np.uint32)
    sdfg = _build_allany_sdfg(
        f"l4_{op}_{implementation}", op, [4], dace.uint32, -1, None, dace.uint32, implementation=implementation
    )
    sdfg.arrays["out"].dtype = dace.uint32
    out = np.full(1, 7, dtype=np.uint32)
    sdfg(mask=mask, out=out)
    assert out[0] == expected


# ---------------------------------------------------------------------------
# default implementation is ``reduction``
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# sequential expansion -- short-circuit (break) via the Python frontend
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    for op, dim in [("all", 1), ("all", 2), ("any", 1), ("any", 2)]:
        test_reduction_dimwise_reduce(op, dim)
    for implementation in ("reduction", "sequential"):
        for op in ("all", "any"):
            test_logical4_output_and_foreign_true(op, 1, implementation)
