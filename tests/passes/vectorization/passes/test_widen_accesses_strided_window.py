# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Widened tile windows must carry the per-lane stride, not assume contiguity.

Lane ``l`` of a ``W``-wide tile evaluates the index expression at ``iter_var + l``, so
``a[i * inc]`` touches ``inc`` cells apart -- the contiguous ``i*inc : i*inc + W`` window that
:class:`WidenAccesses` used to emit named the wrong cells for every ``inc != 1`` while the
``tile_load`` / ``tile_store`` intrinsics strode by ``inc`` regardless (they take the stride from
``dim_strides``, not from the memlet). The memlet was therefore a lie about which cells the map
iteration touches, which every downstream analysis and ``validate`` reads.
"""
import os

os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')

import warnings

import numpy as np
import pytest

import dace
from dace import subsets
from dace.libraries.tileops.nodes.tile_store import TileStore
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses

N = dace.symbol('N')


def widened(index: str, iter_var: str = 'i', width: int = 8):
    """Widen a single-element subset ``[index]`` and return its one ``(begin, end, step)``."""
    new = WidenAccesses(widths=(width, ))._widen_subset_inplace(subsets.Range.from_string(index), (iter_var, ))
    assert new is not None, f"{index!r} was not widened at all"
    return new.ranges[0], new


def test_symbolic_lane_stride_becomes_the_window_step():
    (beg, end, step), rng = widened('i * inc')
    i, inc = dace.symbolic.pystr_to_symbolic('i'), dace.symbolic.pystr_to_symbolic('inc')
    assert dace.symbolic.simplify(beg - i * inc) == 0
    assert dace.symbolic.simplify(step - inc) == 0
    # ``beg + inc*W - 1`` (not ``beg + inc*(W-1)``): both cover lanes 0..W-1, but this form lets
    # ``ceiling((end + 1 - beg) / step)`` fold to exactly W without knowing anything about ``inc``.
    assert dace.symbolic.simplify(end - (i * inc + inc * 8 - 1)) == 0
    assert dace.symbolic.simplify(rng.size()[0] - 8) == 0


def test_affine_lane_stride_keeps_the_offset_and_strides_by_the_coefficient():
    (beg, end, step), rng = widened('3 * i + 2')
    i = dace.symbolic.pystr_to_symbolic('i')
    assert dace.symbolic.simplify(beg - (3 * i + 2)) == 0
    assert step == 3
    assert dace.symbolic.simplify(rng.size()[0] - 8) == 0


def test_unit_stride_window_is_unchanged():
    (beg, end, step), rng = widened('i + 1')
    i = dace.symbolic.pystr_to_symbolic('i')
    assert dace.symbolic.simplify(beg - (i + 1)) == 0
    assert dace.symbolic.simplify(end - (i + 8)) == 0
    assert step == 1


@pytest.mark.parametrize('index', ['i % 4', '-i'])
def test_non_affine_and_descending_indices_are_left_to_the_contiguous_fallback(index):
    """``i % 4`` is not affine in ``i`` and ``-i`` walks backwards; neither has a per-lane step
    this window form can express (the tile-op base pointer is lane 0's address). Both keep the
    legacy contiguous widening rather than growing a bogus stride."""
    (_, _, step), _ = widened(index)
    assert step == 1


def assert_window_strides_by(store, edge, stride_name: str, width: int = 8):
    """The store window must be ``beg : beg + stride*W : stride`` with ``stride`` the node's own
    per-lane stride. Every comparison stays inside the SDFG's own symbol instances -- a
    ``pystr_to_symbolic(name)`` here would mint a differently-assumed symbol and ``inc - inc``
    would not cancel."""
    beg, end, step = edge.data.subset.ranges[0]
    assert dace.symbolic.symstr(step) == stride_name, f'window step {step} is not the lane stride'
    assert dace.symbolic.symstr(list(store.dim_strides)[0]) == stride_name
    assert dace.symbolic.simplify(end - beg - width * step + 1) == 0, f'window {edge.data.subset} is not {width} lanes'


def canonicalized_and_vectorized(prog):
    """Canonicalize + vectorize ``prog``, returning the SDFG and any refusal messages."""
    sdfg = prog.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        VectorizeCPUMultiDim(
            VectorizeConfig(widths=(8, ),
                            target_isa='SCALAR',
                            remainder_strategy='masked_tail',
                            branch_mode='merge',
                            validate_all=True)).apply_pass(sdfg, {})
        refusals = [str(m.message) for m in caught if 'refusing to vectorize' in str(m.message)]
    sdfg.validate()
    return sdfg, refusals


@dace.program
def strided_store(a: dace.float64[N], b: dace.float64[N], n3: dace.int64):
    for i in range(0, N, n3):
        a[i] = a[i] + b[i]


def test_strided_store_window_matches_the_intrinsic_stride():
    """The store memlet must name the cells ``tile_store`` actually writes.

    ``tile_store`` takes its stride from ``dim_strides``, so a contiguous window compiled to the
    right code while describing the wrong cells -- wrong for propagation, for out-of-bounds
    validation, and for any pass that reads the write set.
    """
    sdfg, refusals = canonicalized_and_vectorized(strided_store)
    assert not refusals, refusals
    stores = [(n, e) for sd in sdfg.all_sdfgs_recursive() for st in sd.states() for n in st.nodes()
              if isinstance(n, TileStore) for e in st.out_edges(n) if e.src_conn == '_dst']
    assert stores, 'kernel was not tiled -- nothing to check'
    for node, edge in stores:
        assert_window_strides_by(node, edge, 'n3')

    a, b = np.random.rand(64), np.random.rand(64)
    ref = a.copy()
    ref[0:64:3] += b[0:64:3]
    got = a.copy()
    sdfg.compile()(a=got, b=b, n3=3, N=64)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12)


@dace.program
def symbolic_index_rmw(a: dace.float64[N], b: dace.float64[N], inc: dace.int64):
    for i in range(N):
        a[i * inc] = a[i * inc] + b[i]


def test_promoted_product_index_reaches_the_tile_store_as_a_stride():
    """``a[i*inc]`` is promoted by the frontend to the opaque symbol ``a[__sym_i_times_inc]``.
    Canonicalize inlines it back before the lifting stages, so the vectorizer sees the direct
    arithmetic and stores with lane stride ``inc`` -- rather than leaving the per-lane value
    stranded in a parent-declared scalar, which used to narrow the tile to lane 0."""
    sdfg, refusals = canonicalized_and_vectorized(symbolic_index_rmw)
    assert not refusals, refusals
    stores = [(n, e) for sd in sdfg.all_sdfgs_recursive() for st in sd.states() for n in st.nodes()
              if isinstance(n, TileStore) for e in st.out_edges(n) if e.src_conn == '_dst']
    assert stores, 'kernel was not tiled -- nothing to check'
    for node, edge in stores:
        assert_window_strides_by(node, edge, 'inc')

    a, b = np.random.rand(64), np.random.rand(64)
    got = a.copy()
    sdfg.compile()(a=got, b=b, inc=1, N=64)
    assert np.allclose(got, a + b, rtol=1e-12, atol=1e-12)


@dace.program
def linearized_write(flat: dace.float64[N * N], aa: dace.float64[N, N], bb: dace.float64[N, N]):
    for i in range(N):
        for j in range(N):
            flat[i * N + j] = aa[i, j] + bb[i, j]


def test_linearized_multi_var_index_is_refused_not_broadcast():
    """``flat[i*N + j]`` under a ``(i, j)`` tile is a 2-D access flattened onto a 1-D array. No
    single iter-var owns that array dim, so the classifier reports AFFINE with no stride and the
    padding step used to hand the second iter-var a BROADCAST stride of 0 -- every lane of that
    dim racing for one cell on a write, and one cell splatted across lanes on a read. The lane set
    is a ``W x W`` box at strides ``(N, 1)`` over ONE array dim, which a per-array-dim tile window
    cannot spell, so the kernel must stay un-tiled and correct."""
    sdfg = linearized_write.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        VectorizeCPUMultiDim(
            VectorizeConfig(widths=(8, 8),
                            target_isa='SCALAR',
                            remainder_strategy='masked_tail',
                            branch_mode='merge',
                            validate_all=True)).apply_pass(sdfg, {})
        refusals = [str(m.message) for m in caught if 'refusing to vectorize' in str(m.message)]
    sdfg.validate()
    assert any('indexed jointly by several of the tile iter-vars' in r for r in refusals), refusals

    n = 16
    rng = np.random.default_rng(3)
    aa, bb = rng.random((n, n)), rng.random((n, n))
    got = np.zeros(n * n)
    sdfg.compile()(flat=got, aa=aa, bb=bb, N=n)
    assert np.allclose(got.reshape(n, n), aa + bb, rtol=1e-12, atol=1e-12)
