# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``All`` / ``Any`` library nodes (Fortran ``ALL`` /
``ANY`` logical reductions) and their expansions -- pins the nodes'
contract independently of any frontend wiring.

Covered (both nodes):
  * **whole-array reduce** (``dim=-1``) on 1-D and 2-D logical masks ->
    logical scalar.
  * **dim-wise reduce** (``dim=k``, Fortran 1-based) -> rank-(N-1) result.
  * **sectioned input** (caller-side memlet subset) -> only the section
    is reduced.
  * **non-int mask dtype** (``LOGICAL(1)`` <-> uint8).
  * **implementations**: ``reduction`` (default, ``dace.reduce`` -> a
    ``Reduce`` node, so OpenMP &&/|| on CPU + CUB on GPU) and
    ``sequential`` (short-circuit ``break`` loop) -- both numeric, plus
    the ``sequential`` dim-wise fall-back-to-reduction contract.
"""
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


def _build_allany_sdfg(tag, op, mask_shape, mask_dtype, dim, out_shape, out_dtype, *,
                       implementation="reduction", mask_subset=None):
    """One-state SDFG wiring an ``All`` / ``Any`` node from a mask access into an
    output access.  ``mask_subset`` (list of ``(lo, hi)`` per dim, 0-based
    inclusive-exclusive) restricts the input edge to a section."""
    sdfg = dace.SDFG(f"allany_{tag}")
    sdfg.add_array("mask", mask_shape, mask_dtype, transient=False)
    out_shape_used = out_shape if out_shape else [1]
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
    state.add_edge(state.add_access("mask"), None, node, AllNode.INPUT_CONNECTOR_NAME,
                   dace.Memlet(f"mask[{msub}]"))
    state.add_edge(node, AllNode.OUTPUT_CONNECTOR_NAME, state.add_access("out"), None,
                   dace.Memlet(f"out[{osub}]"))
    sdfg.validate()
    return sdfg


# ---------------------------------------------------------------------------
# reduction expansion -- whole-array reduce, 1-D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op,mask,expected", [
    ("all", [1, 1, 1, 1, 1], 1),   # every element true
    ("all", [1, 1, 0, 1, 1], 0),   # one false -> false
    ("all", [0, 0, 0, 0, 0], 0),
    ("any", [0, 0, 0, 0, 0], 0),   # every element false
    ("any", [0, 0, 1, 0, 0], 1),   # one true -> true
    ("any", [1, 1, 1, 1, 1], 1),
])
def test_reduction_whole_array_1d(op, mask, expected):
    n = len(mask)
    sdfg = _build_allany_sdfg(f"red_{op}_{expected}", op, [n], dace.int32, -1, None, dace.int32)
    out = np.zeros(1, dtype=np.bool_)
    sdfg(mask=np.array(mask, dtype=np.int32), out=out)
    assert int(out[0]) == expected


# ---------------------------------------------------------------------------
# reduction expansion -- whole-array reduce, 2-D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["all", "any"])
def test_reduction_whole_array_2d(op):
    rng = np.random.default_rng(1)
    mask = (rng.random((4, 5)) > 0.5).astype(np.int32)
    sdfg = _build_allany_sdfg(f"red2d_{op}", op, [4, 5], dace.int32, -1, None, dace.int32)
    out = np.zeros(1, dtype=np.bool_)
    sdfg(mask=mask.copy(), out=out)
    expected = int(mask.all()) if op == "all" else int(mask.any())
    assert int(out[0]) == expected


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


@pytest.mark.parametrize("op", ["all", "any"])
def test_reduction_sectioned_input(op):
    # 10-element mask; reduce only [2:7).  Put the deciding element OUTSIDE
    # the section to prove the subset is honoured.
    mask = np.ones(10, dtype=np.int32)
    if op == "all":
        mask[0] = 0   # outside [2:7) -> ALL of the section is still true
        expected = 1
    else:
        mask[:] = 0
        mask[9] = 1   # outside [2:7) -> ANY of the section is still false
        expected = 0
    sdfg = _build_allany_sdfg(f"sect_{op}", op, [10], dace.int32, -1, None, dace.int32,
                              mask_subset=[(2, 7)])
    out = np.zeros(1, dtype=np.bool_)
    sdfg(mask=mask, out=out)
    assert int(out[0]) == expected


# ---------------------------------------------------------------------------
# reduction expansion -- non-int (LOGICAL(1) <-> uint8) mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op,expected", [("all", 0), ("any", 1)])
def test_reduction_uint8_mask(op, expected):
    mask = np.array([1, 0, 1, 1], dtype=np.uint8)  # all->0 (has a 0), any->1
    sdfg = _build_allany_sdfg(f"u8_{op}", op, [4], dace.uint8, -1, None, dace.int32)
    out = np.zeros(1, dtype=np.bool_)
    sdfg(mask=mask, out=out)
    assert int(out[0]) == expected


# ---------------------------------------------------------------------------
# default implementation is ``reduction``
# ---------------------------------------------------------------------------


def test_default_implementation_is_reduction():
    assert AllNode.default_implementation == "reduction"
    node = AllNode("aa")
    # Not selecting an implementation must still expand (via the default).
    sdfg = dace.SDFG("aa_default")
    sdfg.add_array("mask", [4], dace.int32, transient=False)
    sdfg.add_array("out", [1], dace.bool_, transient=False)
    st = sdfg.add_state()
    st.add_node(node)
    st.add_edge(st.add_access("mask"), None, node, AllNode.INPUT_CONNECTOR_NAME, dace.Memlet("mask[0:4]"))
    st.add_edge(node, AllNode.OUTPUT_CONNECTOR_NAME, st.add_access("out"), None, dace.Memlet("out[0]"))
    sdfg.expand_library_nodes()  # must not raise
    out = np.zeros(1, dtype=np.bool_)
    sdfg(mask=np.array([1, 1, 1, 1], dtype=np.int32), out=out)
    assert int(out[0]) == 1


# ---------------------------------------------------------------------------
# sequential expansion -- short-circuit (break) via the Python frontend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op,mask,expected", [
    ("all", [1, 1, 1, 1, 1], 1),
    ("all", [1, 0, 1, 1, 1], 0),   # decides early (index 1) -> break
    ("all", [0, 0, 0, 0, 0], 0),
    ("any", [0, 0, 0, 0, 0], 0),
    ("any", [1, 0, 0, 0, 0], 1),   # decides early (index 0) -> break
    ("any", [0, 0, 0, 0, 1], 1),
])
def test_sequential_short_circuit(op, mask, expected):
    """The ``sequential`` expansion stops at the first deciding element
    (``break``) but returns the same logical result as ``reduction``."""
    n = len(mask)
    sdfg = _build_allany_sdfg(f"seq_{op}_{expected}", op, [n], dace.int32, -1, None, dace.int32,
                              implementation="sequential")
    out = np.zeros(1, dtype=np.bool_)
    sdfg(mask=np.array(mask, dtype=np.int32), out=out)
    assert int(out[0]) == expected


def test_sequential_dimwise_falls_back():
    """``sequential`` only handles a rank-1 whole-array reduce; a
    ``dim``-wise request raises so the library default ('reduction') is used."""
    sdfg = _build_allany_sdfg("seq_dim", "all", [3, 4], dace.int32, 1, [4], dace.int32,
                              implementation="sequential")
    with pytest.raises(NotImplementedError, match="rank-1 whole-array"):
        sdfg.expand_library_nodes()
