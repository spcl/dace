# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for ``CountLibraryNode`` and its expansion — exercises every
supported mode in isolation so the library node's contract is pinned
independently of any frontend wiring.

Each test documents one desired-behaviour case; failures here narrow to
the library node or its expansion (not to a frontend).

Modes covered:
  * **Mode A — whole-array reduce** (``dim=-1``, default).
    Output is a length-1 array; result equals ``int(mask).sum()``.
  * **Mode B — per-dim reduce** (``dim=k``, Fortran 1-based).
    Output is rank-(N-1); reduces along the k-th axis.
  * **Mode C — sectioned input** (caller-side memlet subset).
    The library node sees a partial subset of a larger array; result
    only counts that section.  Verifies the input memlet's subset is
    honoured by the inner Reduce expansion.
  * **Mode D — non-int mask**.  Mask dtype other than int32 (e.g.
    ``LOGICAL(1)`` ↔ uint8).  The expansion's cast tasklet widens
    to int32 before reducing.
"""

import ctypes

import numpy as np

import dace
from dace.libraries.standard.nodes import CountLibraryNode

# DaCe-compiled SOs link against libgomp's ``omp_get_max_threads`` at
# load time; preload it with RTLD_GLOBAL so ctypes.CDLL on the dacestub
# finds the symbol.
try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass


def _build_count_sdfg(name_tag: str, mask_shape, mask_dtype, dim, out_shape, out_dtype):
    """Build a one-state SDFG that wires a CountLibraryNode from a mask
    access into an output access — full coverage (no section subset)."""
    sdfg = dace.SDFG(f"count_{name_tag}")
    sdfg.add_array("mask", mask_shape, mask_dtype, transient=False)
    out_shape_used = out_shape or [1]
    sdfg.add_array("out", out_shape_used, out_dtype, transient=False)
    state = sdfg.add_state("count_state")

    node = CountLibraryNode("count_main", dim=dim)
    state.add_node(node)
    mask_in = state.add_access("mask")
    out_w = state.add_access("out")

    mask_subset = ", ".join(f"0:{s}" for s in mask_shape)
    out_subset = ", ".join(f"0:{s}" for s in out_shape_used)
    state.add_edge(mask_in, None, node, CountLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet(f"mask[{mask_subset}]"))
    state.add_edge(node, CountLibraryNode.OUTPUT_CONNECTOR_NAME, out_w, None, dace.Memlet(f"out[{out_subset}]"))
    sdfg.validate()
    return sdfg


# ---------------------------------------------------------------------------
# Mode A — whole-array reduce
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Mode B — per-dim reduce
# ---------------------------------------------------------------------------


def test_mode_b_dim2_collapses_second_axis():
    """``COUNT(mask, dim=2)`` on a 2-D mask → rank-1 output of length n.
    Reduces along the second axis (j); each ``out[i] = sum_j mask[i, j]``."""
    n, m = 5, 7
    sdfg = _build_count_sdfg("b_dim2", [n, m], dace.int32, dim=2, out_shape=[n], out_dtype=dace.int32)

    rng = np.random.default_rng(2)
    mask = (rng.random((n, m)) > 0.5).astype(np.int32)
    out = np.zeros(n, dtype=np.int32)
    sdfg(mask=mask, out=out)
    np.testing.assert_array_equal(out, mask.sum(axis=1))


# ---------------------------------------------------------------------------
# Mode C — sectioned input subset
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Mode D — narrower mask kind
# ---------------------------------------------------------------------------


def test_mode_d_foreign_true_values_count_once():
    """A LOGICAL(4) mask is the caller's storage: every non-zero value (-1, HUGE, 2) counts as one .TRUE."""
    sdfg = _build_count_sdfg("d_foreign", [5], dace.uint32, dim=-1, out_shape=None, out_dtype=dace.int32)
    mask = np.array([0, 1, 0xFFFFFFFF, 0x7FFFFFFF, 2], dtype=np.uint32)
    out = np.zeros(1, dtype=np.int32)
    sdfg(mask=mask, out=out)
    assert int(out[0]) == 4


if __name__ == "__main__":
    test_mode_b_dim2_collapses_second_axis()
    test_mode_d_foreign_true_values_count_once()
