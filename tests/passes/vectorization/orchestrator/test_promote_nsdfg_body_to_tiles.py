# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""NSDFG-body tiling on the walker-primary path.

A ``vbor``-style per-iteration scalar chain is lowered by the frontend into a
multi-state body NestedSDFG (``loop_body``) inside the tile map's scope. The
legacy ``PromoteNSDFGBodyToTiles`` descent that used to tile such a body in
place was DELETED in the walker-primary migration; :class:`VectorizeCPUMultiDim`
now owns this end to end (``NestInnermostMapBodyIntoNSDFG`` mints the body NSDFG,
then ``WidenAccesses`` / ``InsertTileLoadStore`` / ``ConvertTaskletsToTileOps``
turn connector reads into :class:`TileLoad`, the split scalar chain into
:class:`TileBinop` / :class:`TileUnop`, and connector writes into
:class:`TileStore`). These tests pin that the walker:

* tiles a const-store output (``a[i] = 3.0``) correctly, masked tail included;
* emits TileLoad / TileBinop / TileStore for the vbor scalar chain;
* emits TileUnop for unary-minus tasklets in a reused-scalar chain;
* vectorizes the carried-dependency ``s231`` nest correctly -- the inner ``j``
  loop carries a dependency so LoopToMap leaves it sequential and only the
  parallel outer ``i`` becomes the tiled map (the legacy descent clean-skipped
  this; the walker handles it).
"""

import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop, TileLoad, TileStore, TileUnop
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.transformation.interstate import LoopToMap

L = dace.symbol("LEN_2D")


@dace.program
def _vbor(a: dace.float64[L], b: dace.float64[L], c: dace.float64[L], d: dace.float64[L], e: dace.float64[L],
          x: dace.float64[L]):
    for i in range(L):
        a1 = a[i]
        b1 = b[i]
        c1 = c[i]
        d1 = d[i]
        e1 = e[i]
        f1 = a[i]
        a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 + a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 +
              a1 * d1 * e1 + a1 * d1 * f1 + a1 * e1 * f1)
        b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 + b1 * d1 * f1 + b1 * e1 * f1)
        c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
        d1 = d1 * e1 * f1
        x[i] = a1 * b1 * c1 * d1


@dace.program
def _unop_chain(a: dace.float64[L], b: dace.float64[L], c: dace.float64[L], d: dace.float64[L], e: dace.float64[L],
                x: dace.float64[L]):
    # vbor verbatim with two unary-minus unops injected on reused scalars: same
    # boundary-connector structure as _vbor, so the e2e isolates the TileUnop path.
    for i in range(L):
        a1 = a[i]
        b1 = b[i]
        c1 = c[i]
        d1 = d[i]
        e1 = e[i]
        f1 = a[i]
        a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 + a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 +
              a1 * d1 * e1 + a1 * d1 * f1 + a1 * e1 * f1)
        a1 = -a1  # unary-minus unop on a reused scalar
        b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 + b1 * d1 * f1 + b1 * e1 * f1)
        c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
        c1 = -c1  # unary-minus unop
        d1 = d1 * e1 * f1
        x[i] = a1 * b1 * c1 * d1


@dace.program
def _const_store(a: dace.float64[L]):
    for i in range(L):
        a[i] = 3.0


@dace.program
def _s231(aa: dace.float64[L, L], bb: dace.float64[L, L]):
    for i in range(L):
        for j in range(1, L):
            aa[j, i] = aa[j - 1, i] + bb[j, i]


def _build(prog, name):
    sdfg = prog.to_sdfg(simplify=False)
    sdfg.name = name
    sdfg.simplify(validate=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()
    return sdfg


def _vectorize(prog, name, expand=True):
    """Build the kernel + run the walker-primary ``VectorizeCPUMultiDim``.

    ``expand=False`` leaves the tile lib nodes in place (for structural
    assertions); ``expand=True`` expands them to tasklets (for e2e compile/run)."""
    sdfg = _build(prog, name)
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR,
                                         expand_tile_nodes=expand)).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("n", [16, 17])
def test_const_store_to_output_matches_reference(n):
    """``a[i] = 3.0`` into an output connector, nested into an NSDFG body, tiles
    through the walker (const-fill tile + masked TileStore) and matches the
    reference (n=17 forces the masked tail -- the mask must keep the const out of
    the OOB lanes of the output array)."""
    ref = _build(_const_store, f"const_ref{n}")
    vec = _vectorize(_const_store, f"const_vec{n}")
    ra = np.full(n, -1.0)
    va = np.full(n, -1.0)
    ref.compile()(a=ra, LEN_2D=n)
    vec.compile()(a=va, LEN_2D=n)
    np.testing.assert_allclose(va, ra, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [16, 17])
def test_vbor_nsdfg_body_matches_reference(n):
    """vbor's flat body-NSDFG is tiled in place and matches the
    unvectorized reference (n=17 forces the masked i-tail)."""
    rng = np.random.default_rng(seed=n)
    arrays = {k: rng.random(n) for k in "abcde"}
    arrays["x"] = np.zeros(n)
    ref = _build(_vbor, f"vbor_ref{n}")
    vec = _vectorize(_vbor, f"vbor_vec{n}")

    rf = {k: v.copy() for k, v in arrays.items()}
    vf = {k: v.copy() for k, v in arrays.items()}
    ref.compile()(**rf, LEN_2D=n)
    vec.compile()(**vf, LEN_2D=n)
    np.testing.assert_allclose(vf["x"], rf["x"], rtol=1e-12, atol=1e-12)


def test_vbor_emits_tile_ops():
    """The walker leaves TileLoad / TileBinop / TileStore lib nodes inside the
    body NSDFG (before ``expand_library_nodes``)."""
    sdfg = _vectorize(_vbor, "vbor_struct", expand=False)
    loads = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileLoad)]
    binops = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileBinop)]
    stores = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileStore)]
    assert loads, "expected TileLoad nodes for the connector reads"
    assert binops, "expected TileBinop nodes for the scalar chain"
    assert stores, "expected a TileStore for x[i]"


@pytest.mark.parametrize("n", [16, 17])
def test_unop_chain_matches_reference(n):
    """A reused-scalar chain with unary-minus unops tiles through the walker
    (NSDFG body) and matches the unvectorized reference (n=17 forces the
    masked i-tail). Proves the TileUnop path is numerically correct."""
    rng = np.random.default_rng(seed=n)
    arrays = {k: rng.random(n) for k in "abcde"}
    arrays["x"] = np.zeros(n)
    ref = _build(_unop_chain, f"unop_ref{n}")
    vec = _vectorize(_unop_chain, f"unop_vec{n}")

    rf = {k: v.copy() for k, v in arrays.items()}
    vf = {k: v.copy() for k, v in arrays.items()}
    ref.compile()(**rf, LEN_2D=n)
    vec.compile()(**vf, LEN_2D=n)
    np.testing.assert_allclose(vf["x"], rf["x"], rtol=1e-12, atol=1e-12)


def test_unop_chain_emits_tile_unop():
    """The walker emits a TileUnop for the unary-minus tasklets in a reused-
    scalar (NSDFG-body) chain."""
    sdfg = _vectorize(_unop_chain, "unop_struct", expand=False)
    unops = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileUnop)]
    assert unops, "expected TileUnop nodes for the unary-minus tasklets"


@pytest.mark.parametrize("n", [16, 17])
def test_s231_carried_dep_vectorizes_on_parallel_dim(n):
    """The carried-dependency ``s231`` nest: the inner ``j`` loop carries a
    dependency (``aa[j] = aa[j-1] + bb[j]``), so LoopToMap leaves it sequential
    and only the parallel outer ``i`` becomes the tiled map. The walker
    vectorizes that ``i`` map correctly -- bit-equal to the unvectorized
    reference (the legacy descent clean-skipped this kernel; the walker now
    handles it, so the prior ``NotImplementedError`` expectation is obsolete)."""
    rng = np.random.default_rng(seed=n)
    aa = rng.random((n, n))
    bb = rng.random((n, n))
    ref = _build(_s231, f"s231_ref{n}")
    vec = _vectorize(_s231, f"s231_vec{n}")
    ar = aa.copy()
    av = aa.copy()
    ref.compile()(aa=ar, bb=bb.copy(), LEN_2D=n)
    vec.compile()(aa=av, bb=bb.copy(), LEN_2D=n)
    np.testing.assert_allclose(av, ar, rtol=1e-12, atol=1e-12)
