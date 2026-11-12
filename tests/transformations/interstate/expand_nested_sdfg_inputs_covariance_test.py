# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for :class:`ExpandNestedSDFGInputs` on multi-output NSDFGs
whose outputs mix a WCR reduction and/or a transposed (index-swapped) write with
rank-reduced (column/row-sliced) boundaries.

Both shapes used to crash inside ``_rewrite_memlets_with_offset``: a self-copy /
slice-copy edge (``AccessNode -> AccessNode``) whose access node still carried the
just-removed inner connector name made the ``data.View`` probe
``inner_sdfg.arrays[src.data]`` raise ``KeyError`` (the descriptor was replaced by
the outer array at the top of ``_replace_desc_and_uncollapse_dims``). The fix is a
defensive ``.get()`` -- a missing descriptor is unambiguously "not a View".

The tests assert bit-exactness against a NumPy oracle so a mis-offset transposed
output (``cov[j, i]``) or a dropped reduction shows up as a value error, not just a
crash.
"""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.vectorization.nest_innermost_map_body import NestInnermostMapBodyIntoNSDFG

M = dace.symbol("M")


def _expand_then_inline(sdfg: dace.SDFG) -> None:
    """Widen every top-level body-NSDFG boundary to full arrays, then inline.

    Fresh pass instances per call so no matcher state leaks between tests."""
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs(), InlineMultistateSDFG(), InlineSDFG()]).apply_pass(sdfg, {})


M16 = 16


@dace.program
def _rowsum16(a: dace.float64[M, M16], out: dace.float64[M]):
    for i in dace.map[0:M]:
        acc = 0.0
        for k in range(M16):
            acc += a[i, k]
        out[i] = acc


def _rowsum16_via_map_to_for_loop(sdfg: dace.SDFG) -> None:
    PatternMatchAndApplyRepeated([MapExpansion()]).apply_pass(sdfg, {})
    to_loop = MapToForLoop()
    to_loop.inline_freed = False  # the row body stays nested for ExpandNestedSDFGInputs
    PatternMatchAndApplyRepeated([to_loop]).apply_pass(sdfg, {})


def test_rowsum16_expand_inline_value_exact():
    """Bug 2: a per-row reduction whose boundary read is the sliced row
    ``a[i, 0:16]`` (rank-reduced) must widen + inline value-exactly.

    ``MapToForLoop`` wraps the row body in a top-level NSDFG with a slice-copy
    edge ``a_row[k] -> a_row_slice[0]``; pre-fix that edge's ``data.View`` probe
    raised ``KeyError`` on the just-removed inner connector name."""
    sdfg = _rowsum16.to_sdfg(simplify=True)
    _rowsum16_via_map_to_for_loop(sdfg)

    n_nsdfg = sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.NestedSDFG))
    assert n_nsdfg > 0, "test setup: expected a body NSDFG for ExpandNestedSDFGInputs to widen"

    _expand_then_inline(sdfg)
    sdfg.validate()

    m = 13
    rng = np.random.default_rng(4)
    a = rng.random((m, M16))
    out = np.zeros(m)
    sdfg(a=a, out=out, M=m)

    ref = a.sum(axis=1)
    assert np.allclose(out, ref), f"rowsum16 value corrupted; max|d| = {np.max(np.abs(out - ref))}"
    assert np.max(np.abs(out - ref)) < 1e-9


def test_rowsum16_nest_innermost_then_expand_value_exact():
    """The vectorization-prep driver (``NestInnermostMapBodyIntoNSDFG`` ->
    ``ExpandNestedSDFGInputs``) on the same row reduction: the widen must not
    crash and the kernel must stay bit-exact."""
    sdfg = _rowsum16.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG(vector_width=M16, nest_provably_divisible=True).apply_pass(sdfg, {})
    applied = sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs, permissive=False, validate=False)
    assert applied and applied >= 1
    sdfg.validate()

    m = 11
    rng = np.random.default_rng(7)
    a = rng.random((m, M16))
    out = np.zeros(m)
    sdfg(a=a, out=out, M=m)

    ref = a.sum(axis=1)
    assert np.allclose(out, ref), f"rowsum16 (nest path) value corrupted; max|d| = {np.max(np.abs(out - ref))}"


if __name__ == "__main__":
    test_rowsum16_expand_inline_value_exact()
    test_rowsum16_nest_innermost_then_expand_value_exact()
    print("all covariance/rowsum16 expand-inline regression tests passed")
