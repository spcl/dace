# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalization must fission a mixed parallel/sequential loop body.

A loop that carries BOTH a true recurrence and an independent statement

    for i in range(1, N):
        a[i] = a[i - 1] + x[i]   # sequential (prefix-sum recurrence)
        b[i] = y[i] * 2.0        # independent (embarrassingly parallel)

cannot become one parallel map -- the ``a[i-1]`` carry bridges every iteration.
Naive loop-to-map therefore parallelizes *nothing*. Canonicalization must first
``LoopFission`` the body so the independent statement lifts to a parallel map
while the recurrence stays a sequential loop: the "one parallel loop + one
sequential compute" decomposition. This test asserts fission unlocks a parallel
map the fused form cannot, that the recurrence survives as a sequential region,
and that the result stays numerically identical to the numpy oracle.
"""
import contextlib
import inspect
import io
import os

os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.interstate.loop_to_map import LoopToMap
from tests.corpus.tsvc_2_5 import tsvc_2_5, tsvc_2_5_numpy


def _program(name: str):
    return [p for p in tsvc_2_5.collect() if p.name.endswith(name)][0]


def _top_maps(sdfg: dace.SDFG) -> int:
    return sum(1 for st in sdfg.all_states() for n in st.nodes()
               if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None)


def _loop_regions(sdfg: dace.SDFG) -> int:
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


@pytest.mark.parametrize("kernel", ["fission_dep_then_indep", "fission_dep_const_offset"])
def test_fission_splits_recurrence_from_parallel_body(kernel):
    program = _program(kernel)

    # Baseline: loop-to-map on the fused body parallelizes nothing (the carry blocks it).
    baseline = program.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([LoopToMap()]).apply_pass(baseline, {})
    assert _top_maps(baseline) == 0, "the recurrence-carrying fused loop must not map as-is"

    # Canonicalized: fission -> one parallel map (independent body) + one sequential loop (recurrence).
    canon = program.to_sdfg(simplify=True)
    with contextlib.redirect_stdout(io.StringIO()):
        canonicalize(canon, validate=True, peel_limit=4, break_anti_dependence=True, unroll_limit=4)
    assert _top_maps(canon) >= 1, "fission must lift the independent statement to a parallel map"
    assert _loop_regions(canon) >= 1, "the recurrence must survive as a sequential loop"

    # Value-preserving vs the numpy oracle.
    arrays, scalars = tsvc_2_5.make_inputs(program)
    oracle = getattr(tsvc_2_5_numpy, "ref_" + program.name.rsplit("tsvc_2_5_", 1)[-1])
    pool = {
        **{
            n: a.copy()
            for n, a in arrays.items()
        },
        **scalars,
        **{
            s.lower(): v
            for s, v in tsvc_2_5.SIZES.items()
        }, "n": tsvc_2_5.SIZES["LEN_1D"]
    }
    oracle(**{p: pool[p] for p in inspect.signature(oracle).parameters})
    ref = {n: pool[n] for n in arrays}

    free = {str(s) for s in canon.free_symbols}
    for s in free:
        if s not in canon.symbols:
            canon.add_symbol(s, dace.int64)
    symbols = {s: tsvc_2_5.SIZES[s] for s in tsvc_2_5.SIZES if s in free}
    got = {n: a.copy() for n, a in arrays.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        canon.compile()(**got, **scalars, **symbols)
    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue
        assert np.allclose(ref[name], got[name], rtol=1e-9, atol=1e-9, equal_nan=True), \
            f"{program.name}/{name}: fissioned canon diverges from numpy oracle"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
