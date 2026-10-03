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
from dace.libraries.standard.nodes.scan import Scan
from dace.transformation.passes.canonicalize.pipeline import _build_stages, canonicalize
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
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


def _carried_sequentially(sdfg: dace.SDFG) -> int:
    """Places the recurrence can legally end up: a sequential loop, or the Scan it lifts to."""
    return _loop_regions(sdfg) + sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan))


def _assert_matches_oracle(program, canon):
    """The canonicalized kernel must reproduce the numpy oracle element for element.

    Structure is read off ``canon`` BEFORE this runs: compiling is one-way, and the SDFG a
    ``CompiledSDFG`` still points at is not the object to inspect afterwards.
    """
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
        csdfg = canon.compile()
    csdfg(**got, **scalars, **symbols)
    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue
        assert np.allclose(ref[name], got[name], rtol=1e-9, atol=1e-9, equal_nan=True), \
            f"{program.name}/{name}: fissioned canon diverges from numpy oracle"


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
    assert _carried_sequentially(canon) >= 1, "the recurrence must survive as a sequential loop or a Scan"

    _assert_matches_oracle(program, canon)


def _split_only(program):
    """``program`` run through the recipe up to and including ``SplitStatements``.

    The full recipe lowers the split's loops to maps and lifts the recurrence to a ``Scan``, so
    how many loops the split itself emitted -- and which array each writes -- is only readable here.
    """
    sdfg = program.to_sdfg(simplify=True)
    for _label, unit in _build_stages():
        unit.apply_pass(sdfg, {})
        if isinstance(unit, SplitStatements):
            break
    sdfg.reset_cfg_list()
    return sdfg


def _loop_writes(sdfg):
    """The non-transient arrays each residual loop stores to, sorted per loop and across loops."""
    out = []
    for region in sdfg.all_control_flow_regions(recursive=True):
        if not (isinstance(region, LoopRegion) and region.loop_variable):
            continue
        out.append(
            sorted({
                node.data
                for state in region.all_states()
                for node in state.data_nodes()
                if not state.sdfg.arrays[node.data].transient and any(e.data is not None and not e.data.is_empty()
                                                                      for e in state.in_edges(node))
            }))
    return sorted(out)


def test_fission_dep_then_indep_yields_one_loop_per_statement():
    """``a[i] = a[i-1] + x[i]; b[i] = y[i]*2`` -- the UNORDERED-sibling case, asserted structurally.

    The two groups share no array at all, so no order between the clones has to be proved and the
    split is the plainest one the pass documents. What it must produce is exactly two loops, one
    writing ``a`` and one writing ``b`` -- and after the rest of the recipe, ``b``'s loop is a
    parallel map and ``a``'s is a ``Scan``, with no sequential ``LoopRegion`` left at all. The
    existing test above admits ``>= 1`` of each, which an un-split loop that merely mapped its
    tail would also satisfy.
    """
    program = _program("fission_dep_then_indep")

    split = _split_only(program)
    assert _loop_writes(split) == [["a"], ["b"]], "the split must emit one loop per output statement"

    canon = program.to_sdfg(simplify=True)
    with contextlib.redirect_stdout(io.StringIO()):
        canonicalize(canon, validate=True, peel_limit=4, break_anti_dependence=True, unroll_limit=4)
    assert _loop_regions(canon) == 0, "nothing may stay a sequential loop: b maps, a becomes a Scan"
    assert _top_maps(canon) == 1, "the independent statement is the one map"
    assert sum(1 for n, _ in canon.all_nodes_recursive() if isinstance(n, Scan)) == 1, \
        "the prefix-sum recurrence must be lifted to exactly one Scan"
    _assert_matches_oracle(program, canon)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
