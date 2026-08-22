# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize on data-dependent MERGE-select and config-flag branch shapes.

* **MERGE / upwind select** -- ``mo_solve_nonhydro.f90:970-1001``
  (``lvn_pos = vn >= 0; ilc0 = MERGE(idx1, idx2, lvn_pos)`` feeding a
  gather). The select reads a per-edge value, so it must stay INSIDE the
  per-element Map (canonicalize must not hoist the data-dependent select
  out -- doing so would be wrong, the condition varies per lane).

* **Config-flag two-array branches** -- cloudsc ``IWARMRAIN == 1/2`` /
  ``IEVAPRAIN`` (1833-1931): a loop-invariant config flag selects between
  two incompatible bodies that write DIFFERENT arrays. The guard is
  invariant, so canonicalize may hoist it to the SDFG top level, leaving
  each branch as its own clean parallel nest.

Both are value-preserving against numpy oracles.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _ncond_blocks(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock))


# ----------------------------------------------------------------------
# Data-dependent MERGE / upwind select feeding a gather
# ----------------------------------------------------------------------


@dace.program
def upwind_select_gather(vn: dace.float64[N], field: dace.float64[N], idx1: dace.int32[N], idx2: dace.int32[N],
                         out: dace.float64[N]):
    """``out[je] = field[ MERGE(idx1, idx2, vn>=0) ]`` -- the upwind
    neighbour is chosen per edge by the sign of ``vn``. Data-dependent
    select; must stay inside the per-edge Map."""
    for je in dace.map[0:N]:
        if vn[je] >= 0.0:
            out[je] = field[idx1[je]]
        else:
            out[je] = field[idx2[je]]


def _upwind_oracle(vn, field, idx1, idx2):
    n = vn.shape[0]
    out = np.zeros(n)
    for je in range(n):
        out[je] = field[idx1[je]] if vn[je] >= 0.0 else field[idx2[je]]
    return out


def test_upwind_select_gather_value_preserving():
    n = 14
    rng = np.random.default_rng(60)
    vn = rng.standard_normal(n)
    field = rng.standard_normal(n)
    idx1 = rng.integers(0, n, n).astype(np.int32)
    idx2 = rng.integers(0, n, n).astype(np.int32)
    exp = _upwind_oracle(vn, field, idx1, idx2)
    sdfg = upwind_select_gather.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    out = np.zeros(n)
    sdfg(vn=vn, field=field, idx1=idx1, idx2=idx2, out=out, N=n)
    assert np.allclose(out, exp), 'upwind select mis-canonicalized'


def test_upwind_select_stays_inside_map():
    """The data-dependent select reads ``vn[je]`` per edge, so it must NOT
    be hoisted to the SDFG top level (no top-level ConditionalBlock); the
    per-edge Map survives."""
    sdfg = upwind_select_gather.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nmaps(sdfg) >= 1, 'the per-edge select must keep a Map'
    top_conds = [c for c in sdfg.nodes() if isinstance(c, ConditionalBlock)]
    assert not top_conds, 'data-dependent per-edge select must not be hoisted to SDFG top level'


# ----------------------------------------------------------------------
# Config-flag selecting between two different-array branches
# ----------------------------------------------------------------------


@dace.program
def config_two_array_branches(a: dace.float64[N], outA: dace.float64[N], outB: dace.float64[N], scheme: dace.int32):
    """``scheme`` is a loop-invariant config flag selecting between two
    incompatible bodies writing DIFFERENT outputs (cloudsc IWARMRAIN
    shape). The guard may hoist to the SDFG top level."""
    for i in dace.map[0:N]:
        if scheme == 1:
            outA[i] = a[i] * 2.0
        else:
            outB[i] = a[i] + 1.0


def test_config_two_array_branches_value_preserving():
    n = 12
    rng = np.random.default_rng(61)
    a = rng.standard_normal(n)
    for scheme in (1, 0):
        sdfg = config_two_array_branches.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True)
        sdfg.validate()
        outA = np.zeros(n)
        outB = np.zeros(n)
        sdfg(a=a, outA=outA, outB=outB, scheme=np.int32(scheme), N=n)
        if scheme == 1:
            assert np.allclose(outA, a * 2.0), f'outA mismatch scheme={scheme}'
        else:
            assert np.allclose(outB, a + 1.0), f'outB mismatch scheme={scheme}'


def test_config_two_array_branches_guard_present():
    """The invariant ``scheme`` guard is preserved (canonicalize keeps a
    ConditionalBlock selecting the two incompatible branches); value
    correctness is the primary contract."""
    sdfg = config_two_array_branches.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _ncond_blocks(sdfg) >= 1, 'the config-flag guard must survive to select the two branches'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
