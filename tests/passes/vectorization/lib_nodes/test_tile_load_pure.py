# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end correctness for the ``pure`` expansion of :class:`TileLoad`.

The lib node receives the full memlet of the source array and uses the
in-edge subset to locate the K-dim tile region. K=1 and K=2 are covered
contiguously; non-unit ``dim_strides`` are covered in T6 once the
strided-load AVX-512 intrinsic lands.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileLoad


def _build_load_sdfg(src_shape, widths, has_mask, dtype=dace.float64):
    """Build a minimal SDFG: source array -> TileLoad -> tile transient."""
    sdfg = dace.SDFG(f"tile_load_pure_{'x'.join(str(w) for w in widths)}_{'m' if has_mask else 'nm'}")
    sdfg.add_array("SRC", src_shape, dtype, transient=False)
    sdfg.add_array("DST", widths, dtype, transient=False)
    if has_mask:
        sdfg.add_array("M", widths, dace.bool_, transient=False)

    state = sdfg.add_state("main")
    src_node = state.add_access("SRC")
    dst_node = state.add_access("DST")
    node = TileLoad(name="tl", widths=widths, has_mask=has_mask)
    state.add_node(node)

    src_subset = ",".join(f"0:{w}" for w in widths)
    dst_subset = ",".join(f"0:{w}" for w in widths)
    state.add_edge(src_node, None, node, "_src", dace.Memlet(f"SRC[{src_subset}]"))
    state.add_edge(node, "_dst", dst_node, None, dace.Memlet(f"DST[{dst_subset}]"))
    if has_mask:
        m_node = state.add_access("M")
        state.add_edge(m_node, None, node, "_mask", dace.Memlet(f"M[{dst_subset}]"))

    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("widths", [(8, ), (4, 8)])
def test_tile_load_pure_unmasked_contiguous(widths):
    """Unmasked load copies the leading tile region of SRC into DST."""
    sdfg = _build_load_sdfg(src_shape=widths, widths=widths, has_mask=False)
    rng = np.random.default_rng(seed=11)
    SRC = rng.random(widths)
    DST = np.zeros(widths)
    sdfg(SRC=SRC, DST=DST)
    np.testing.assert_allclose(DST, SRC, rtol=0, atol=0)


def test_tile_load_pure_masked_writes_zero_on_inactive_lanes():
    """Inactive mask lanes write 0 into DST (matches AVX-512 maskz semantic)."""
    widths = (4, 8)
    sdfg = _build_load_sdfg(src_shape=widths, widths=widths, has_mask=True)
    rng = np.random.default_rng(seed=12)
    SRC = rng.random(widths)
    DST = np.full(widths, 99.0)
    M = np.zeros(widths, dtype=bool)
    M[:, :4] = True
    sdfg(SRC=SRC, DST=DST, M=M)
    ref = np.where(M, SRC, 0.0)
    np.testing.assert_allclose(DST, ref, rtol=0, atol=0)


def test_tile_load_rejects_invalid_K():
    """Constructor refuses K outside ``{1, 2, 3}`` and stride / width length mismatch."""
    with pytest.raises(ValueError, match="length in"):
        TileLoad(name="bad_K", widths=())
    with pytest.raises(ValueError, match="dim_strides length"):
        TileLoad(name="bad_stride_len", widths=(8, ), dim_strides=(1, 1))


# ---- Replicate-factor spectrum ---------------------------------------


def _build_replicate_load_sdfg(src_shape, widths, replicate_factor_per_dim):
    """Build a minimal SDFG exercising a TileLoad with a non-trivial
    ``replicate_factor_per_dim``: source array is W/k elements per
    replicate dim; destination is W lanes per dim."""
    sdfg = dace.SDFG(f"tile_load_replicate_{'x'.join(str(w) for w in widths)}_"
                     f"{'x'.join(str(k) for k in replicate_factor_per_dim)}")
    sdfg.add_array("SRC", src_shape, dace.float64, transient=False)
    sdfg.add_array("DST", widths, dace.float64, transient=False)
    state = sdfg.add_state("main")
    src_node = state.add_access("SRC")
    dst_node = state.add_access("DST")
    node = TileLoad(name="tl_rep", widths=widths, has_mask=False, replicate_factor_per_dim=replicate_factor_per_dim)
    state.add_node(node)
    src_subset = ",".join(f"0:{s}" for s in src_shape)
    dst_subset = ",".join(f"0:{w}" for w in widths)
    state.add_edge(src_node, None, node, "_src", dace.Memlet(f"SRC[{src_subset}]"))
    state.add_edge(node, "_dst", dst_node, None, dace.Memlet(f"DST[{dst_subset}]"))
    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("k", [2, 4])
def test_tile_load_pure_replicate_k1(k):
    """K=1 with ``replicate_factor=k`` (1 < k < W): source has W/k > 1
    distinct elements; the pure expansion broadcasts each to ``k``
    consecutive lanes. The degenerate endpoint ``k = W`` is the full
    broadcast case -- covered by ``src_kind='Scalar'``, not this
    spectrum."""
    W = 8
    sdfg = _build_replicate_load_sdfg(src_shape=(W // k, ), widths=(W, ), replicate_factor_per_dim=(k, ))
    rng = np.random.default_rng(seed=21)
    SRC = rng.random(W // k)
    DST = np.zeros(W)
    sdfg(SRC=SRC, DST=DST)
    # Lane ``l`` should read SRC[l // k]
    expected = np.array([SRC[l // k] for l in range(W)])
    np.testing.assert_allclose(DST, expected, rtol=0, atol=0)


def test_tile_load_pure_replicate_factor_1_is_contiguous():
    """``replicate_factor=1`` is exactly the contiguous endpoint of the
    spectrum -- the codegen reduces to a plain TileLoad."""
    sdfg = _build_replicate_load_sdfg(src_shape=(8, ), widths=(8, ), replicate_factor_per_dim=(1, ))
    rng = np.random.default_rng(seed=22)
    SRC = rng.random(8)
    DST = np.zeros(8)
    sdfg(SRC=SRC, DST=DST)
    np.testing.assert_allclose(DST, SRC, rtol=0, atol=0)


def test_tile_load_rejects_invalid_replicate_factor():
    """Constructor refuses replicate factors that don't divide widths
    or that are < 1."""
    with pytest.raises(ValueError, match="replicate_factor_per_dim"):
        TileLoad(name="bad_factor_dim", widths=(8, ), replicate_factor_per_dim=(1, 2))
    with pytest.raises(ValueError, match="must be >= 1"):
        TileLoad(name="bad_factor_zero", widths=(8, ), replicate_factor_per_dim=(0, ))
    with pytest.raises(ValueError, match="must divide"):
        TileLoad(name="bad_factor_no_div", widths=(8, ), replicate_factor_per_dim=(3, ))
