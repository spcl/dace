# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end correctness for the ``pure`` expansion of :class:`TileReduce`.

Covers nine combinations of tile shape, reduction op, axis (full vs single),
and mask presence. Each test compiles + runs the lib node and compares the
result against a plain NumPy reduction — the project rule is that the
reference must be a non-transformed scalar evaluation, not a different
SDFG variant.

Single vectorization config is sufficient since the reduction emission is
invariant across knobs (the lib node owns the per-arch lowering).
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileReduce

_IDENT = {
    "+": 0.0,
    "*": 1.0,
    "min": np.inf,
    "max": -np.inf,
}


def _np_reduce(op: str, arr: np.ndarray, axis):
    """Compute the NumPy reference for the supported op set."""
    if op == "+":
        return np.sum(arr, axis=axis)
    if op == "*":
        return np.prod(arr, axis=axis)
    if op == "min":
        return np.min(arr, axis=axis)
    if op == "max":
        return np.max(arr, axis=axis)
    raise ValueError(f"unknown op {op!r}")


def _build_reduce_sdfg(widths, op: str, axis, has_mask: bool, dtype=dace.float64):
    """Build a minimal SDFG: SRC tile (+ optional MASK) -> TileReduce -> DST."""
    K = len(widths)
    if axis is None:
        dst_shape = (1, )
    else:
        dst_shape = tuple(w for d, w in enumerate(widths) if d != axis) or (1, )
    op_tag = {"+": "add", "*": "mul", "min": "min", "max": "max"}[op]
    sdfg = dace.SDFG(f"tile_reduce_pure_{'x'.join(str(w) for w in widths)}_{op_tag}_"
                     f"{'full' if axis is None else f'ax{axis}'}_"
                     f"{'m' if has_mask else 'nm'}")
    sdfg.add_array("SRC", widths, dtype, transient=False)
    sdfg.add_array("DST", dst_shape, dtype, transient=False)
    if has_mask:
        sdfg.add_array("M", widths, dace.bool_, transient=False)

    state = sdfg.add_state("main")
    src_node = state.add_access("SRC")
    dst_node = state.add_access("DST")
    node = TileReduce(name="tr", widths=widths, op=op, axis=axis, has_mask=has_mask)
    state.add_node(node)

    src_subset = ",".join(f"0:{w}" for w in widths)
    dst_subset = ",".join(f"0:{w}" for w in dst_shape)
    state.add_edge(src_node, None, node, "_src", dace.Memlet(f"SRC[{src_subset}]"))
    state.add_edge(node, "_dst", dst_node, None, dace.Memlet(f"DST[{dst_subset}]"))
    if has_mask:
        m_node = state.add_access("M")
        state.add_edge(m_node, None, node, "_mask", dace.Memlet(f"M[{src_subset}]"))

    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg, dst_shape


def _run_reduce(widths, op, axis, has_mask, seed, mask_arr=None):
    sdfg, dst_shape = _build_reduce_sdfg(widths, op, axis, has_mask)
    rng = np.random.default_rng(seed=seed)
    SRC = rng.random(widths)
    DST = np.zeros(dst_shape)
    kwargs = dict(SRC=SRC, DST=DST)
    if has_mask:
        assert mask_arr is not None
        kwargs["M"] = mask_arr
    sdfg(**kwargs)
    return SRC, DST, dst_shape


# -------------------- K=1 unmasked --------------------


def test_k1_sum_full():
    """Sum-reduce a K=1 tile to a scalar."""
    SRC, DST, _ = _run_reduce(widths=(8, ), op="+", axis=None, has_mask=False, seed=1)
    ref = _np_reduce("+", SRC, axis=None)
    np.testing.assert_allclose(DST.flatten(), [ref], rtol=1e-12, atol=1e-12)


def test_k1_max_full():
    """Max-reduce a K=1 tile to a scalar."""
    SRC, DST, _ = _run_reduce(widths=(8, ), op="max", axis=None, has_mask=False, seed=2)
    ref = _np_reduce("max", SRC, axis=None)
    np.testing.assert_allclose(DST.flatten(), [ref], rtol=1e-12, atol=1e-12)


# -------------------- K=2 full reduction --------------------


def test_k2_sum_full():
    """Sum-reduce a K=2 tile to a scalar (all lanes combined)."""
    SRC, DST, _ = _run_reduce(widths=(4, 8), op="+", axis=None, has_mask=False, seed=3)
    ref = _np_reduce("+", SRC, axis=None)
    np.testing.assert_allclose(DST.flatten(), [ref], rtol=1e-12, atol=1e-12)


def test_k2_prod_full():
    """Product-reduce a K=2 tile to a scalar."""
    SRC, DST, _ = _run_reduce(widths=(4, 8), op="*", axis=None, has_mask=False, seed=4)
    ref = _np_reduce("*", SRC, axis=None)
    np.testing.assert_allclose(DST.flatten(), [ref], rtol=1e-12, atol=1e-12)


# -------------------- K=2 axis reduction --------------------


def test_k2_sum_axis0():
    """Sum-reduce a K=2 tile along axis 0 — output shape (W1,)."""
    SRC, DST, dst_shape = _run_reduce(widths=(4, 8), op="+", axis=0, has_mask=False, seed=5)
    assert dst_shape == (8, )
    ref = _np_reduce("+", SRC, axis=0)
    np.testing.assert_allclose(DST, ref, rtol=1e-12, atol=1e-12)


def test_k2_sum_axis1():
    """Sum-reduce a K=2 tile along axis 1 — output shape (W0,)."""
    SRC, DST, dst_shape = _run_reduce(widths=(4, 8), op="+", axis=1, has_mask=False, seed=6)
    assert dst_shape == (4, )
    ref = _np_reduce("+", SRC, axis=1)
    np.testing.assert_allclose(DST, ref, rtol=1e-12, atol=1e-12)


# -------------------- masked variants --------------------


def test_k1_sum_full_masked():
    """Masked sum reduction: inactive lanes contribute the ``+`` identity (0)."""
    widths = (8, )
    rng = np.random.default_rng(seed=21)
    M = (rng.random(widths) > 0.5)
    SRC, DST, _ = _run_reduce(widths=widths, op="+", axis=None, has_mask=True, seed=21, mask_arr=M)
    masked_src = np.where(M, SRC, _IDENT["+"])
    ref = _np_reduce("+", masked_src, axis=None)
    np.testing.assert_allclose(DST.flatten(), [ref], rtol=1e-12, atol=1e-12)


def test_k2_sum_full_masked():
    """Masked sum reduction on a K=2 tile to a scalar."""
    widths = (4, 8)
    rng = np.random.default_rng(seed=22)
    M = (rng.random(widths) > 0.3)
    SRC, DST, _ = _run_reduce(widths=widths, op="+", axis=None, has_mask=True, seed=22, mask_arr=M)
    masked_src = np.where(M, SRC, _IDENT["+"])
    ref = _np_reduce("+", masked_src, axis=None)
    np.testing.assert_allclose(DST.flatten(), [ref], rtol=1e-12, atol=1e-12)


def test_k2_sum_axis0_masked():
    """Masked single-axis reduction: inactive lanes contribute identity (0)
    so each kept-dim slot is the sum of active lanes mapping to it."""
    widths = (4, 8)
    rng = np.random.default_rng(seed=23)
    M = (rng.random(widths) > 0.4)
    SRC, DST, dst_shape = _run_reduce(widths=widths, op="+", axis=0, has_mask=True, seed=23, mask_arr=M)
    assert dst_shape == (8, )
    masked_src = np.where(M, SRC, _IDENT["+"])
    ref = _np_reduce("+", masked_src, axis=0)
    np.testing.assert_allclose(DST, ref, rtol=1e-12, atol=1e-12)
