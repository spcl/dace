# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`FuseChainedScalarReductions`."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

from typing import List, Optional, Tuple

import numpy as np
import pytest

import dace
from dace.sdfg import SDFGState, nodes
from dace.transformation.passes.canonicalize.fuse_chained_scalar_reductions import (FuseChainedScalarReductions,
                                                                                    _binop_op)

N = dace.symbol('N')


@dace.program
def s319(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], e: dace.float64[N],
         out: dace.float64[1]):
    s = 0.0
    for i in range(N):
        a[i] = c[i] + d[i]
        s = s + a[i]
        b[i] = c[i] + e[i]
        s = s + b[i]
    out[0] = s


def chain_state(sdfg: dace.SDFG) -> Optional[Tuple[SDFGState, List[nodes.Tasklet]]]:
    """The loop-body state holding the chained accumulations, with its foldable binops."""
    for state in sdfg.states():
        binops = [n for n in state.nodes() if isinstance(n, nodes.Tasklet) and _binop_op(n) is not None]
        if len(binops) >= 2:
            return state, binops
    return None


def reference(c: np.ndarray, d: np.ndarray, e: np.ndarray) -> float:
    return float((c + d).sum() + (c + e).sum())


def run(sdfg: dace.SDFG, n: int) -> float:
    rng = np.random.default_rng(319)
    c, d, e = rng.random(n), rng.random(n), rng.random(n)
    out = np.zeros(1)
    sdfg(a=np.zeros(n), b=np.zeros(n), c=c, d=d, e=e, out=out, N=n)
    assert np.allclose(out[0], reference(c, d, e)), f'got {out[0]}, want {reference(c, d, e)}'
    return out[0]


def test_chained_accumulations_fold():
    sdfg = s319.to_sdfg(simplify=True)
    assert chain_state(sdfg) is not None, 'fixture must produce the chained-accumulator shape'
    assert FuseChainedScalarReductions().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    run(sdfg, 24)


def test_ordering_memlet_on_a_chain_node_refuses_the_fold():
    """The fold deletes the downstream chain nodes; an ordering memlet on one of them would go
    with it, so the fold must decline."""
    sdfg = s319.to_sdfg(simplify=True)
    found = chain_state(sdfg)
    assert found is not None
    state, binops = found
    anchor = next(iter(state.data_nodes()))
    state.add_edge(anchor, None, binops[-1], None, dace.Memlet())
    sdfg.validate()

    assert FuseChainedScalarReductions().apply_pass(sdfg, {}) is None, 'the fold must decline'
    assert any(e.data.is_empty() for e in state.edges()), 'the ordering memlet must survive'
    sdfg.validate()
    run(sdfg, 24)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
