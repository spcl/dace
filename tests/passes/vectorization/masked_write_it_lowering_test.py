# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Masked-write lowering: a frontend boolean-masked assignment ``A[mask] = value``
lowers (``newast.py``) to a bare-if tasklet ``if __in_cond: __out = value``.

``NormalizeMaskedWriteTasklets`` rewrites that (in the tiled bodies only) to the
first-class write-only conditional-write function ``__out = IT(__in_cond, value)``
-- write ``value`` where the condition holds, else leave the destination unchanged,
with NO old-value read (unlike ``ITE(c, t, e)``). ``ConvertTaskletsToTileOps`` then
lowers ``IT`` to a masked ``TileStore`` (the ``cond`` gates the store; inactive lanes
are left untouched). This test pins both the structural rewrite and the end-to-end
numerics against NumPy.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import ast

import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.normalize_masked_write_tasklets import NormalizeMaskedWriteTasklets
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')


def _bare_if_tasklets(sdfg):
    hits = []
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nd.Tasklet) and n.code.language == dace.dtypes.Language.Python:
            try:
                body = ast.parse(n.code.as_string).body
            except (SyntaxError, ValueError):
                continue
            if any(isinstance(s, ast.If) for s in body):
                hits.append(n)
    return hits


@dace.program
def masked_zero(A: dace.float64[N], thresh: dace.float64):
    A[A > thresh] = 0.0


@dace.program
def masked_val(A: dace.float64[N], x: dace.float64[N], m: dace.bool[N]):
    A[m] = x


def _base(prog):
    sdfg = prog.to_sdfg(simplify=False)
    sdfg.simplify(validate=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.simplify(validate=True)
    return sdfg


def test_masked_write_bare_if_becomes_it():
    """The frontend bare-if masked-write tasklet is rewritten to the write-only
    ``IT(cond, value)`` form -- no bare-if survives and NO ``*_old`` self-read
    connector is added (``IT`` never reads the destination's prior value)."""
    sdfg = masked_zero.to_sdfg(simplify=True)
    assert len(_bare_if_tasklets(sdfg)) >= 1, "test setup: expected a frontend bare-if tasklet"
    n = NormalizeMaskedWriteTasklets().apply_pass(sdfg, {})
    assert n and n >= 1
    survivors = _bare_if_tasklets(sdfg)
    assert not survivors, f"bare-if tasklets survived: {[t.code.as_string for t in survivors]}"
    it_tasklets = [
        t for t, _ in sdfg.all_nodes_recursive()
        if isinstance(t, nd.Tasklet) and t.code.language == dace.dtypes.Language.Python and "IT(" in t.code.as_string
    ]
    assert it_tasklets, "expected an ``IT(...)`` conditional-write tasklet"
    for t in it_tasklets:
        # Write-only: the IT rewrite must NOT introduce an old-value self-read connector.
        assert not any(c.endswith("_old") for c in t.in_connectors), \
            f"IT tasklet must not read the destination's old value: {sorted(t.in_connectors)}"
        assert t.code.as_string.strip().startswith(next(iter(t.out_connectors)) + " = IT(")
    sdfg.validate()


def test_normalize_skips_scalar_tail():
    """Normalize (run standalone) rewrites every masked write to ``IT`` -- but inside
    the full pipeline it skips scalar-tail scopes, which keep the bare-if. Here, with no
    tail markers present, all masked writes are rewritten (structural precondition)."""
    sdfg = masked_zero.to_sdfg(simplify=True)
    NormalizeMaskedWriteTasklets().apply_pass(sdfg, {})
    assert not _bare_if_tasklets(sdfg)


@pytest.mark.parametrize("isa", ["SCALAR", "AVX512"])
@pytest.mark.parametrize("remainder", ["scalar_postamble", "masked_tail"])
def test_masked_const_write_matches_numpy(isa, remainder):
    """``A[A > thresh] = 0`` lowers through the tile pipeline (interior masked store +
    scalar/masked tail) bit-exact vs NumPy, at a non-tile-divisible size."""
    sdfg = _base(masked_zero)
    VectorizeCPUMultiDim(widths=(8, ), target_isa=isa, remainder_strategy=remainder, branch_mode="merge",
                         validate_all=True).apply_pass(sdfg, {})
    rng = np.random.default_rng(0)
    Nval = 37
    A = rng.random(Nval)
    thresh = 0.5
    ref = A.copy()
    ref[ref > thresh] = 0.0
    work = A.copy()
    sdfg(A=work, thresh=thresh, N=Nval)
    assert np.array_equal(work, ref), f"{work[:6]} != {ref[:6]}"


@pytest.mark.parametrize("isa", ["SCALAR", "AVX512"])
@pytest.mark.parametrize("remainder", ["scalar_postamble", "masked_tail"])
def test_masked_value_write_matches_numpy(isa, remainder):
    """``A[m] = x`` (value tile, not a constant) lowers bit-exact vs NumPy. The
    ``masked_tail`` config exercises the AND-combine of ``cond`` with the tile
    iteration mask on the remainder store."""
    sdfg = _base(masked_val)
    VectorizeCPUMultiDim(widths=(8, ), target_isa=isa, remainder_strategy=remainder, branch_mode="merge",
                         validate_all=True).apply_pass(sdfg, {})
    rng = np.random.default_rng(1)
    Nval = 37
    A = rng.random(Nval)
    x = rng.random(Nval)
    m = rng.random(Nval) > 0.5
    ref = A.copy()
    ref[m] = x[m]
    work = A.copy()
    sdfg(A=work, x=x.copy(), m=m.copy(), N=Nval)
    assert np.array_equal(work, ref), f"{work[:6]} != {ref[:6]}"


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
