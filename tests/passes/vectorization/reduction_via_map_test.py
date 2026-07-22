# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Standalone reproducers for reduction-expressed-as-``dace.map`` shapes that the
multi-dim tile vectorizer must handle end-to-end (extracted from npbench
``azimint_naive`` / ``azimint_hist``).

A scalar accumulation inside a parallel ``dace.map`` -- ``for j in dace.map[0:N]:
acc += x[j]`` -- is valid dace code: the frontend lowers the augmented assignment
to a WCR reduction (a ``-> map_exit`` WCR). The vectorizer normalizes that WCR
(lifts it to a ``Reduce``). This test pins the two shapes:

* ``scalar_reduce``  -- an unmasked WCR sum. Must vectorize + run bit-exact.
* ``masked_reduce``  -- a masked WCR sum + count (``if mask[j]: acc += x[j]``),
  the ``azimint_naive`` shape. The conditional is lowered to a per-lane select
  whose result is folded into the accumulator.

Both are run at K=1 SCALAR and AVX512 against the numpy reference.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')


@dace.program
def scalar_reduce(data: dace.float64[N], res: dace.float64[1]):
    tmp = dace.float64(0)
    for j in dace.map[0:N]:
        tmp += data[j]
    res[0] = tmp


@dace.program
def masked_reduce(data: dace.float64[N], mask: dace.int64[N], res: dace.float64[1]):
    tmp = dace.float64(0)
    cnt = dace.float64(0)
    for j in dace.map[0:N]:
        if mask[j] > 0:
            tmp += data[j]
            cnt += 1.0
    res[0] = tmp / cnt


def _run(prog, kwargs, ref, isa):
    from dace.libraries.tileops import TileReduce
    sdfg = prog.to_sdfg(simplify=True)
    cfg = VectorizeConfig(widths=(8, ), target_isa=isa, remainder_strategy=RemainderStrategy.SCALAR_POSTAMBLE,
                          expand_tile_nodes=False)
    VectorizeCPUMultiDim(cfg).apply_pass(sdfg, {})
    # The scalar accumulation must have been LOWERED to a horizontal TileReduce -- i.e. actually
    # vectorized, not silently refused (a refuse would leave a plain per-lane scalar reduction and
    # still run bit-exact, hiding a no-op). Assert the tile fold is present before expanding.
    assert any(isinstance(n, TileReduce) for n, _ in sdfg.all_nodes_recursive()), \
        f"{prog.name}/{isa}: no TileReduce -- reduction was not vectorized (silent refuse?)"
    sdfg.expand_library_nodes()
    sdfg.validate()
    work = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in kwargs.items()}
    sdfg(**work)
    assert np.allclose(work['res'][0], ref, rtol=1e-9, atol=1e-12), f"{prog.name}/{isa}: {work['res'][0]} != {ref}"


@pytest.mark.parametrize("isa", ["SCALAR", "AVX512"])
def test_scalar_reduce_via_map(isa):
    """Unmasked WCR sum over a ``dace.map`` -> lifted to Reduce, vectorized."""
    rng = np.random.default_rng(0)
    n = 20
    data = rng.random(n)
    _run(scalar_reduce, dict(data=data, res=np.zeros(1), N=n), data.sum(), isa)


@pytest.mark.parametrize("isa", ["SCALAR", "AVX512"])
def test_masked_reduce_via_map(isa):
    """Masked WCR sum + count (``azimint_naive`` shape): conditional accumulation
    inside a ``dace.map``.

    The frontend lowers ``if mask[j]: acc += x[j]`` to a masked in-nsdfg WCR that
    ``NormalizeWCR`` turns into a seeded body-local accumulator + a plain copyback into a
    scalar sink. ``WidenAccesses`` (Step 0) rewrites that copyback into a ``reduce_accum`` fold
    -- read off the boundary WCR op -- so it lowers to a horizontal ``TileReduce`` and the scalar
    sink is NOT over-widened. Bit-exact at K=1 SCALAR and AVX512."""
    rng = np.random.default_rng(1)
    n = 20
    data = rng.random(n)
    mask = (rng.random(n) > 0.5).astype(np.int64)
    ref = data[mask > 0].sum() / (mask > 0).sum()
    _run(masked_reduce, dict(data=data, mask=mask, res=np.zeros(1), N=n), ref, isa)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
