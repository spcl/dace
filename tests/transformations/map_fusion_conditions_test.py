# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Map fusion across conditions: two adjacent identical-condition guarded
    computations (the shape branch-replicated fission produces) must
    recombine -- ConditionFusion merges the ConditionalBlocks, then map
    fusion fuses the now-co-located maps. Kernels use the dace Python
    frontend; every test checks numerical equivalence (condition taken and
    not-taken) against a deep-copied pre-pass run.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg.state import ConditionalBlock
from dace.sdfg import nodes
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.interstate.condition_fusion import ConditionFusion
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def two_guarded(a: dace.float64[N], A: dace.float64[N], B: dace.float64[N]):
    # Symbol condition -> no condition-prep state, so the two Conditional
    # blocks are directly consecutive with an identical condition.
    if M > 0:
        for i in dace.map[0:N]:
            A[i] = a[i] + 1.0
    if M > 0:
        for i in dace.map[0:N]:
            B[i] = a[i] * 2.0


def _with_M(sdfg):
    """A condition-only symbol is not auto-registered in ``sdfg.symbols``
    (frontend quirk); add it so the SDFG is callable."""
    if 'M' not in sdfg.symbols:
        sdfg.add_symbol('M', dace.int64)
    return sdfg


def _conds(sdfg):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock)]


def _maps(sdfg):
    return [n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.MapEntry)]


def test_map_fusion_recombines_identical_guards():
    """Two consecutive `if c: <map>` blocks recombine into one guard whose
    maps fuse; value-preserving for c taken and not-taken."""
    n = 16
    a = np.random.rand(n)
    base = _with_M(two_guarded.to_sdfg(simplify=True))
    n_conds_before = len(_conds(base))
    assert n_conds_before >= 2, "expected two separate guarding conditionals"

    for mval in (1, 0):
        ref_A, ref_B = np.full(n, 9.0), np.full(n, 9.0)
        copy.deepcopy(base)(a=a.copy(), A=ref_A, B=ref_B, N=n, M=mval)

        sdfg = _with_M(two_guarded.to_sdfg(simplify=True))
        # The fuse-stage recipe: merge identical guards, structural-clean, fuse maps.
        PatternMatchAndApplyRepeated([ConditionFusion()]).apply_pass(sdfg, {})
        PatternMatchAndApplyRepeated([StateFusionExtended()]).apply_pass(sdfg, {})
        PatternMatchAndApplyRepeated([InlineSDFG()]).apply_pass(sdfg, {})
        PatternMatchAndApplyRepeated([MapFusionVertical(), MapFusionHorizontal()]).apply_pass(sdfg, {})
        sdfg.validate()
        assert len(_conds(sdfg)) < n_conds_before, "guards were not recombined"

        out_A, out_B = np.full(n, 9.0), np.full(n, 9.0)
        sdfg(a=a.copy(), A=out_A, B=out_B, N=n, M=mval)
        assert np.allclose(out_A, ref_A) and np.allclose(out_B, ref_B), f"mismatch M={mval}"
        if mval > 0:
            assert np.allclose(out_A, a + 1.0) and np.allclose(out_B, a * 2.0)
        else:
            assert np.allclose(out_A, 9.0) and np.allclose(out_B, 9.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
