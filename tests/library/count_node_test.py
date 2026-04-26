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

import dace
from dace.libraries.standard.nodes import CountLibraryNode

import numpy as np

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
    out_shape_used = out_shape if out_shape else [1]
    sdfg.add_array("out", out_shape_used, out_dtype, transient=False)
    state = sdfg.add_state("count_state")

    node = CountLibraryNode("count_main", dim=dim)
    state.add_node(node)
    mask_in = state.add_access("mask")
    out_w = state.add_access("out")

    mask_subset = ", ".join(f"0:{s}" for s in mask_shape)
    out_subset = ", ".join(f"0:{s}" for s in out_shape_used)
    state.add_edge(mask_in, None, node, "_mask", dace.Memlet(f"mask[{mask_subset}]"))
    state.add_edge(node, "_out", out_w, None, dace.Memlet(f"out[{out_subset}]"))
    sdfg.validate()
    return sdfg


# ---------------------------------------------------------------------------
# Mode A — whole-array reduce
# ---------------------------------------------------------------------------


def test_mode_a_whole_array_int_mask_1d():
    """``COUNT(mask)`` on a 1-D int mask → scalar count.  Default ``dim=-1``."""
    n = 16
    sdfg = _build_count_sdfg("a_int1d", [n], dace.int32, dim=-1, out_shape=None, out_dtype=dace.int32)

    rng = np.random.default_rng(0)
    mask = (rng.random(n) > 0.5).astype(np.int32)
    out = np.zeros(1, dtype=np.int32)
    sdfg(mask=mask, out=out)
    assert int(out[0]) == int(mask.sum())


def test_mode_a_whole_array_int_mask_2d():
    """``COUNT(mask)`` on a 2-D int mask → scalar count covering both
    axes.  Verifies the inner Reduce reduces along all dimensions when
    ``dim=-1``."""
    n, m = 6, 8
    sdfg = _build_count_sdfg("a_int2d", [n, m], dace.int32, dim=-1, out_shape=None, out_dtype=dace.int32)

    rng = np.random.default_rng(1)
    mask = (rng.random((n, m)) > 0.5).astype(np.int32)
    out = np.zeros(1, dtype=np.int32)
    sdfg(mask=mask, out=out)
    assert int(out[0]) == int(mask.sum())


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


def test_mode_b_dim1_collapses_first_axis():
    """``COUNT(mask, dim=1)`` on a 2-D mask → rank-1 output of length m.
    Reduces along the first axis (i); each ``out[j] = sum_i mask[i, j]``."""
    n, m = 4, 6
    sdfg = _build_count_sdfg("b_dim1", [n, m], dace.int32, dim=1, out_shape=[m], out_dtype=dace.int32)

    rng = np.random.default_rng(3)
    mask = (rng.random((n, m)) > 0.5).astype(np.int32)
    out = np.zeros(m, dtype=np.int32)
    sdfg(mask=mask, out=out)
    np.testing.assert_array_equal(out, mask.sum(axis=0))


# ---------------------------------------------------------------------------
# Mode C — sectioned input subset
# ---------------------------------------------------------------------------


def test_mode_c_sectioned_input_subset():
    """The caller-side memlet covers only a section of the source mask
    (``mask[2:6]`` of an 8-element array).  The library node must
    honour the subset and only count those 4 elements — verifies the
    inner Reduce expansion uses the incoming subset, not the array's
    full descriptor shape."""
    n_full = 8
    sdfg = dace.SDFG("count_section")
    sdfg.add_array("mask", [n_full], dace.int32, transient=False)
    sdfg.add_array("out", [1], dace.int32, transient=False)
    state = sdfg.add_state("count_state")

    node = CountLibraryNode("count_section", dim=-1)
    state.add_node(node)
    mask_in = state.add_access("mask")
    out_w = state.add_access("out")
    # Section of length 4 (indices 2..5 inclusive).
    state.add_edge(mask_in, None, node, "_mask", dace.Memlet("mask[2:6]"))
    state.add_edge(node, "_out", out_w, None, dace.Memlet("out[0]"))
    sdfg.validate()

    mask = np.array([1, 0, 1, 1, 0, 1, 1, 0], dtype=np.int32)  # 5 ones total
    out = np.zeros(1, dtype=np.int32)
    sdfg(mask=mask, out=out)
    # Section [2:6] = [1, 1, 0, 1] → 3 ones.
    assert int(out[0]) == 3


# ---------------------------------------------------------------------------
# Mode D — narrower mask kind
# ---------------------------------------------------------------------------


def test_mode_d_uint8_mask_widens_to_int32():
    """``LOGICAL(1)`` ↔ uint8 mask — the expansion's cast tasklet widens
    each element to int32 before the Reduce.  Verifies type-coercion
    inside the library node (so the bridge can pass any-kind logical
    masks without per-test type wrangling)."""
    n = 12
    sdfg = _build_count_sdfg("d_uint8", [n], dace.uint8, dim=-1, out_shape=None, out_dtype=dace.int32)

    mask = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0], dtype=np.uint8)
    out = np.zeros(1, dtype=np.int32)
    sdfg(mask=mask, out=out)
    assert int(out[0]) == 7
