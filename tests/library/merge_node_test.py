# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for ``MergeLibraryNode`` and its expansion — exercises every
supported variant in isolation so the library node's contract is pinned
independently of any frontend wiring.

Variants follow the Fortran ``MERGE(tsource, fsource, mask)`` shapes;
each test documents one desired-behaviour case.

  * **V1 — all scalar** (``MERGE(t, f, mask_scalar)``):
    Each operand is a single value → result is a single value.
    Degenerate, but the node accepts it via length-1 input arrays.
  * **V2 — all-array** (``MERGE(t_arr, f_arr, mask_arr)``):
    Per-element select; all four cover the same shape.
  * **V3 — broadcast both scalars** (``MERGE(t_scalar, f_scalar,
    mask_arr)``):
    Each output element picks one of two scalars via the mask array.
    Both ``_t`` and ``_f`` arrive as length-1 inputs; ``_mask`` and
    ``_out`` are full arrays.  The expansion broadcasts the scalars
    by indexing them as ``[0]`` per iteration.
  * **V4 — t scalar, f array** (``MERGE(t_scalar, f_arr, mask_arr)``):
    Asymmetric broadcast — ``_t`` is broadcast, ``_f`` per-element.
  * **V5 — t array, f scalar** (``MERGE(t_arr, f_scalar, mask_arr)``):
    Mirror of V4 — ``_f`` broadcast, ``_t`` per-element.
"""
import dace
from dace.libraries.standard.nodes import MergeLibraryNode

import numpy as np


def _build_merge_sdfg(name_tag: str, t_shape, f_shape, m_shape, out_shape, strides=None):
    """Build a one-state SDFG that wires a MergeLibraryNode for the
    given operand shapes.  Each input is wired with a memlet covering
    its full descriptor; the expansion picks broadcast vs per-element
    behaviour based on the memlet subset's volume.  ``strides`` gives
    every operand a non-default layout; ``None`` leaves them packed."""
    sdfg = dace.SDFG(f"merge_{name_tag}")
    sdfg.add_array("t", t_shape, dace.float64, transient=False, strides=strides)
    sdfg.add_array("f", f_shape, dace.float64, transient=False, strides=strides)
    sdfg.add_array("mask", m_shape, dace.int32, transient=False, strides=strides)
    sdfg.add_array("out", out_shape, dace.float64, transient=False, strides=strides)
    state = sdfg.add_state("merge_state")

    node = MergeLibraryNode("merge_main")
    state.add_node(node)
    t_in = state.add_access("t")
    f_in = state.add_access("f")
    m_in = state.add_access("mask")
    out_w = state.add_access("out")

    state.add_edge(t_in, None, node, MergeLibraryNode.TRUE_CONNECTOR_NAME,
                   dace.Memlet(f"t[{', '.join(f'0:{s}' for s in t_shape)}]"))
    state.add_edge(f_in, None, node, MergeLibraryNode.FALSE_CONNECTOR_NAME,
                   dace.Memlet(f"f[{', '.join(f'0:{s}' for s in f_shape)}]"))
    state.add_edge(m_in, None, node, MergeLibraryNode.MASK_CONNECTOR_NAME,
                   dace.Memlet(f"mask[{', '.join(f'0:{s}' for s in m_shape)}]"))
    state.add_edge(node, MergeLibraryNode.OUTPUT_CONNECTOR_NAME, out_w, None,
                   dace.Memlet(f"out[{', '.join(f'0:{s}' for s in out_shape)}]"))
    sdfg.validate()
    return sdfg


# ---------------------------------------------------------------------------
# V1 — all scalar
# ---------------------------------------------------------------------------


def test_v1_all_scalar():
    """``MERGE(t, f, mask)`` with t, f, mask, out each length-1.
    Degenerate variant; result picks t or f by the single mask bit."""
    sdfg = _build_merge_sdfg("v1_scalar", [1], [1], [1], [1])
    for mask_val, expected in [(1, 7.5), (0, -3.25)]:
        t = np.array([7.5], dtype=np.float64)
        f = np.array([-3.25], dtype=np.float64)
        mask = np.array([mask_val], dtype=np.int32)
        out = np.zeros(1, dtype=np.float64)
        sdfg(t=t, f=f, mask=mask, out=out)
        assert float(out[0]) == expected


# ---------------------------------------------------------------------------
# V2 — all-array
# ---------------------------------------------------------------------------


def test_v2_all_array_pointwise():
    """``MERGE(t_arr, f_arr, mask_arr)`` — pointwise select, all same
    shape.  The common case.  Result is per-element ``np.where``."""
    n = 16
    sdfg = _build_merge_sdfg("v2_array", [n], [n], [n], [n])

    rng = np.random.default_rng(0)
    t = rng.standard_normal(n, dtype=np.float64)
    f = rng.standard_normal(n, dtype=np.float64)
    mask = (rng.random(n) > 0.5).astype(np.int32)
    out = np.zeros(n, dtype=np.float64)
    sdfg(t=t, f=f, mask=mask, out=out)
    np.testing.assert_array_equal(out, np.where(mask.astype(bool), t, f))


def _provenance_report(got, expected, **operands) -> str:
    """Every wrong cell, and WHERE its value came from.

    A wrong result is either a value the kernel had no business reading -- so the read address is
    off -- or a write that never happened, leaving the caller's zero. Naming the operand and index
    each wrong value actually occupies separates those two, which the default
    ``assert_array_equal`` message cannot: it prints five indices and truncates the arrays.
    """
    lines = [f'{np.count_nonzero(got != expected)} / {got.size} cells wrong']
    for idx in zip(*np.nonzero(got != expected)):
        value = got[idx]
        source = 'never written (still the caller zero)' if value == 0 else 'not in any operand'
        for oname, operand in operands.items():
            where = np.nonzero(operand == value)
            if where[0].size:
                source = f'{oname}{tuple(int(c[0]) for c in where)}'
                break
        lines.append(f'  out{tuple(int(i) for i in idx)} = {value!r} <- {source}, wanted {expected[idx]!r}')
    return '\n'.join(lines)


def test_v2_all_array_2d():
    """``MERGE`` on a 2-D pointwise shape — verifies the iteration
    domain matches the output's shape across rank > 1."""
    n, m = 6, 8
    sdfg = _build_merge_sdfg("v2_2d", [n, m], [n, m], [n, m], [n, m])

    rng = np.random.default_rng(1)
    t = rng.standard_normal((n, m), dtype=np.float64)
    f = rng.standard_normal((n, m), dtype=np.float64)
    mask = (rng.random((n, m)) > 0.5).astype(np.int32)
    out = np.zeros((n, m), dtype=np.float64)
    sdfg(t=t, f=f, mask=mask, out=out)
    expected = np.where(mask.astype(bool), t, f)
    assert np.array_equal(out, expected), _provenance_report(out, expected, t=t, f=f)


# ---------------------------------------------------------------------------
# V3 — both sources scalar, mask array (scalar-broadcast)
# ---------------------------------------------------------------------------


def test_v3_both_scalars_array_mask():
    """``MERGE(t_scalar, f_scalar, mask_arr)`` — both ``_t`` and ``_f``
    are length-1; the expansion's per-input broadcast detection makes
    each iteration read element ``[0]`` of the scalar inputs."""
    n = 10
    sdfg = _build_merge_sdfg("v3", [1], [1], [n], [n])

    t = np.array([42.0], dtype=np.float64)
    f = np.array([-1.5], dtype=np.float64)
    mask = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.int32)
    out = np.zeros(n, dtype=np.float64)
    sdfg(t=t, f=f, mask=mask, out=out)
    expected = np.where(mask.astype(bool), 42.0, -1.5)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# V4 — t scalar, f array, mask array
# ---------------------------------------------------------------------------


def test_v4_t_scalar_f_array():
    """``MERGE(t_scalar, f_arr, mask_arr)`` — only ``_t`` broadcasts;
    ``_f`` and ``_mask`` are per-element."""
    n = 12
    sdfg = _build_merge_sdfg("v4", [1], [n], [n], [n])

    t = np.array([99.0], dtype=np.float64)
    rng = np.random.default_rng(2)
    f = rng.standard_normal(n, dtype=np.float64)
    mask = (rng.random(n) > 0.5).astype(np.int32)
    out = np.zeros(n, dtype=np.float64)
    sdfg(t=t, f=f, mask=mask, out=out)
    expected = np.where(mask.astype(bool), 99.0, f)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# V5 — t array, f scalar, mask array
# ---------------------------------------------------------------------------


def test_v5_t_array_f_scalar():
    """Mirror of V4 — only ``_f`` broadcasts; ``_t`` and ``_mask`` are
    per-element."""
    n = 14
    sdfg = _build_merge_sdfg("v5", [n], [1], [n], [n])

    rng = np.random.default_rng(3)
    t = rng.standard_normal(n, dtype=np.float64)
    f = np.array([-7.0], dtype=np.float64)
    mask = (rng.random(n) > 0.5).astype(np.int32)
    out = np.zeros(n, dtype=np.float64)
    sdfg(t=t, f=f, mask=mask, out=out)
    expected = np.where(mask.astype(bool), t, -7.0)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# Operand layout — descriptors that are not packed C
# ---------------------------------------------------------------------------


def test_fortran_layout_operands():
    """Every operand is Fortran-strided.  Each connector on the expanded
    SDFG is a view onto the caller's buffer, so it has to carry that
    operand's own strides — assuming a packed C layout reads the wrong
    addresses and the select returns other elements' values."""
    n, m = 6, 8
    fstrides = (1, n)
    sdfg = _build_merge_sdfg("fortran_2d", [n, m], [n, m], [n, m], [n, m], strides=fstrides)

    state = sdfg.states()[0]
    node = next(nd for nd in state.nodes() if isinstance(nd, MergeLibraryNode))
    node.expand(state)
    inner = next(nd.sdfg for nd in state.nodes() if isinstance(nd, dace.nodes.NestedSDFG))
    for conn in (MergeLibraryNode.TRUE_CONNECTOR_NAME, MergeLibraryNode.FALSE_CONNECTOR_NAME,
                 MergeLibraryNode.MASK_CONNECTOR_NAME, MergeLibraryNode.OUTPUT_CONNECTOR_NAME):
        assert tuple(inner.arrays[conn].strides) == fstrides, \
            f"connector '{conn}' dropped the operand layout: {tuple(inner.arrays[conn].strides)} != {fstrides}"

    rng = np.random.default_rng(1)
    t = np.asfortranarray(rng.standard_normal((n, m), dtype=np.float64))
    f = np.asfortranarray(rng.standard_normal((n, m), dtype=np.float64))
    mask = np.asfortranarray((rng.random((n, m)) > 0.5).astype(np.int32))
    out = np.zeros((n, m), order="F", dtype=np.float64)
    sdfg(t=t, f=f, mask=mask, out=out)
    np.testing.assert_array_equal(out, np.where(mask.astype(bool), t, f))


def test_fortran_layout_scalar_broadcast():
    """Fortran-strided array operands alongside a length-1 one: the
    broadcast operand stays a scalar view while the array operands keep
    their own strides."""
    n, m = 5, 7
    fstrides = (1, n)
    sdfg = dace.SDFG("merge_fortran_bcast")
    sdfg.add_array("t", [1], dace.float64, transient=False)
    sdfg.add_array("f", [n, m], dace.float64, transient=False, strides=fstrides)
    sdfg.add_array("mask", [n, m], dace.int32, transient=False, strides=fstrides)
    sdfg.add_array("out", [n, m], dace.float64, transient=False, strides=fstrides)
    state = sdfg.add_state("merge_state")
    node = MergeLibraryNode("merge_main")
    state.add_node(node)
    state.add_edge(state.add_access("t"), None, node, MergeLibraryNode.TRUE_CONNECTOR_NAME, dace.Memlet("t[0:1]"))
    state.add_edge(state.add_access("f"), None, node, MergeLibraryNode.FALSE_CONNECTOR_NAME,
                   dace.Memlet(f"f[0:{n}, 0:{m}]"))
    state.add_edge(state.add_access("mask"), None, node, MergeLibraryNode.MASK_CONNECTOR_NAME,
                   dace.Memlet(f"mask[0:{n}, 0:{m}]"))
    state.add_edge(node, MergeLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access("out"), None,
                   dace.Memlet(f"out[0:{n}, 0:{m}]"))
    sdfg.validate()

    node.expand(state)
    inner = next(nd.sdfg for nd in state.nodes() if isinstance(nd, dace.nodes.NestedSDFG))
    assert tuple(inner.arrays[MergeLibraryNode.TRUE_CONNECTOR_NAME].shape) == (1, )
    for conn in (MergeLibraryNode.FALSE_CONNECTOR_NAME, MergeLibraryNode.MASK_CONNECTOR_NAME,
                 MergeLibraryNode.OUTPUT_CONNECTOR_NAME):
        assert tuple(inner.arrays[conn].strides) == fstrides, \
            f"connector '{conn}' dropped the operand layout: {tuple(inner.arrays[conn].strides)} != {fstrides}"

    rng = np.random.default_rng(4)
    t = np.array([13.5], dtype=np.float64)
    f = np.asfortranarray(rng.standard_normal((n, m), dtype=np.float64))
    mask = np.asfortranarray((rng.random((n, m)) > 0.5).astype(np.int32))
    out = np.zeros((n, m), order="F", dtype=np.float64)
    sdfg(t=t, f=f, mask=mask, out=out)
    np.testing.assert_array_equal(out, np.where(mask.astype(bool), 13.5, f))
