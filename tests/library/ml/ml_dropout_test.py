# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the pure expansion of dace.libraries.ml.Dropout."""

import numpy as np

import dace
import dace.libraries.ml as ml


def _build(shape, p, seed, offset, mask, dtype=dace.float32, mask_dtype=dace.uint8, nsym=None):
    n = nsym if nsym is not None else shape[0]
    full = [n] + list(shape[1:])
    sdfg = dace.SDFG("dropout")
    sdfg.add_array("x", full, dtype)
    sdfg.add_array("y", full, dtype)
    if mask:
        sdfg.add_array("m", full, mask_dtype)
    state = sdfg.add_state()
    node = ml.Dropout("drop", p=p, seed=seed, offset=offset)
    state.add_node(node)
    state.add_edge(state.add_read("x"), None, node, "_x", dace.Memlet("x"))
    state.add_edge(node, "_y", state.add_write("y"), None, dace.Memlet("y"))
    if mask:
        node.add_out_connector("_mask")
        state.add_edge(node, "_mask", state.add_write("m"), None, dace.Memlet("m"))
    return sdfg


def test_determinism_same_seed_offset():
    shape = (32, 48)
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    y1 = np.zeros(shape, np.float32)
    m1 = np.zeros(shape, np.uint8)
    y2 = np.zeros(shape, np.float32)
    m2 = np.zeros(shape, np.uint8)
    _build(shape, 0.3, 42, 0, True)(x=x, y=y1, m=m1)
    _build(shape, 0.3, 42, 0, True)(x=x, y=y2, m=m2)
    assert np.array_equal(m1, m2)
    assert np.array_equal(y1, y2)


def test_different_offsets_differ():
    shape = (32, 48)
    x = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    m0 = np.zeros(shape, np.uint8)
    m1 = np.zeros(shape, np.uint8)
    y = np.zeros(shape, np.float32)
    _build(shape, 0.5, 42, 0, True)(x=x, y=y, m=m0)
    _build(shape, 0.5, 42, 1000, True)(x=x, y=y, m=m1)
    assert not np.array_equal(m0, m1)


def test_mask_mean_matches_keep_probability():
    shape = (256, 256)
    x = np.random.default_rng(2).standard_normal(shape).astype(np.float32)
    y = np.zeros(shape, np.float32)
    m = np.zeros(shape, np.uint8)
    _build(shape, 0.2, 7, 0, True)(x=x, y=y, m=m)
    assert abs(m.mean() - 0.8) < 0.01


def test_output_equals_scaled_input():
    shape = (128, 97)
    p = 0.35
    x = np.random.default_rng(3).standard_normal(shape).astype(np.float32)
    y = np.zeros(shape, np.float32)
    m = np.zeros(shape, np.uint8)
    _build(shape, p, 11, 0, True)(x=x, y=y, m=m)
    expected = x * m.astype(np.float32) / (1 - p)
    assert np.allclose(y, expected, atol=1e-5)


def test_p_zero_keeps_all():
    shape = (64, 64)
    x = np.random.default_rng(4).standard_normal(shape).astype(np.float32)
    y = np.zeros(shape, np.float32)
    m = np.zeros(shape, np.uint8)
    _build(shape, 0, 5, 0, True)(x=x, y=y, m=m)
    assert m.all()
    assert np.allclose(y, x, atol=1e-6)


def test_no_mask_output():
    shape = (16, 16)
    x = np.random.default_rng(5).standard_normal(shape).astype(np.float32)
    y = np.zeros(shape, np.float32)
    _build(shape, 0.4, 9, 0, False)(x=x, y=y)
    # Every output element is either 0 or x/(1-p).
    scaled = x / 0.6
    close_zero = np.isclose(y, 0.0, atol=1e-6)
    close_scaled = np.isclose(y, scaled, atol=1e-4)
    assert np.all(close_zero | close_scaled)


def test_bool_mask_symbolic_batch():
    """Mirrors the frontend case: bool mask and a symbolic leading dimension."""
    N = dace.symbol("N")
    shape = (8, 1000)
    seed = dace.symbol("dace_random_seed")
    x = np.random.default_rng(6).standard_normal(shape).astype(np.float32)
    y = np.zeros(shape, np.float32)
    m = np.zeros(shape, dtype=np.bool_)
    sdfg = _build(shape, 0.3, seed, 0, True, mask_dtype=dace.bool, nsym=N)
    sdfg(x=x, y=y, m=m, N=shape[0], dace_random_seed=123)
    expected = x * m.astype(np.float32) / 0.7
    assert np.allclose(y, expected, atol=1e-4)
    assert m.dtype == np.bool_


def test_save_roundtrip(tmp_path):
    sdfg = _build((8, 8), 0.25, 99, 3, True)
    node = next(x for s in sdfg.states() for x in s.nodes() if isinstance(x, ml.Dropout))
    fname = str(tmp_path / "dropout.sdfg")
    sdfg.save(fname)
    sdfg2 = dace.SDFG.from_file(fname)
    node2 = next(x for s in sdfg2.states() for x in s.nodes() if isinstance(x, ml.Dropout))
    assert str(node2.p) == str(node.p)
    assert str(node2.seed) == str(node.seed)
    assert str(node2.offset) == str(node.offset)


if __name__ == "__main__":
    test_determinism_same_seed_offset()
    test_different_offsets_differ()
    test_mask_mean_matches_keep_probability()
    test_output_equals_scaled_input()
    test_p_zero_keeps_all()
    test_no_mask_output()
    test_bool_mask_symbolic_batch()
    print("ok")
