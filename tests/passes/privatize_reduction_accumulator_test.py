# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``PrivatizeReductionAccumulator``.

The pass converts WCR-on-array-element reductions (e.g. ``dot[0] WCR-+= val``
inside a parallel map) into WCR-on-scalar + init + writeback. The resulting
SDFG has a transient ``Scalar`` accumulator the OMP codegen can name in a
``reduction(op:scalar)`` clause.
"""
import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from dace.transformation.interstate.state_fusion import StateFusion
from dace.transformation.passes.canonicalize.privatize_reduction_accumulator import PrivatizeReductionAccumulator
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = dace.symbol("N")


def _build_wcr_map(prog) -> dace.SDFG:
    """Build the simplified SDFG and run the AugAssignToWCR + LoopToMap +
    state-fusion + inline path that produces a parallel WCR-map with the WCR
    target as an array element. This matches the canonicalize order in which
    ``PrivatizeReductionAccumulator`` is wired."""
    sdfg = prog.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(AugAssignToWCR, validate=False, validate_all=False, permissive=True)
    sdfg.apply_transformations_repeated(LoopToMap, validate=False, validate_all=False)
    PatternMatchAndApplyRepeated([StateFusion()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([InlineSDFG()]).apply_pass(sdfg, {})
    return sdfg


def _count_wcr_edges(sdfg: dace.SDFG):
    wcr_arr_elem = 0
    wcr_scalar = 0
    for st in sdfg.all_states():
        for e in st.edges():
            if e.data is None or e.data.wcr is None:
                continue
            desc = sdfg.arrays.get(e.data.data)
            if desc is None:
                continue
            from dace import data
            if isinstance(desc, data.Scalar):
                wcr_scalar += 1
            else:
                wcr_arr_elem += 1
    return wcr_arr_elem, wcr_scalar


# ---------------------------------------------------------------------------
# Basic shape: a dot product whose accumulator is an array slot.
# ---------------------------------------------------------------------------


@dace.program
def _array_slot_dot(a: dace.float64[N], b: dace.float64[N], dot: dace.float64[N]):
    dot[0] = 0.0
    for i in range(N):
        dot[0] = dot[0] + a[i] * b[i]


def test_array_slot_dot_product_privatized():
    """``dot[0] WCR-+= a[i]*b[i]`` -- WCR target becomes a transient Scalar,
    init state seeds it from ``dot[0]``, writeback state copies back."""
    sdfg = _build_wcr_map(_array_slot_dot)
    arr_wcr_before, scalar_wcr_before = _count_wcr_edges(sdfg)
    # Pre-condition: the AugAssignToWCR + L2M path produced an array-element
    # WCR write.
    assert arr_wcr_before >= 1
    assert scalar_wcr_before == 0

    lifted = PrivatizeReductionAccumulator().apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 1

    # Post-condition: the WCR target is now a Scalar, no array-elem WCR left.
    arr_wcr_after, scalar_wcr_after = _count_wcr_edges(sdfg)
    assert arr_wcr_after == 0
    assert scalar_wcr_after >= 1

    # New ``_priv_*`` transient scalar appears.
    priv_arrays = sorted(k for k in sdfg.arrays if k.startswith("_priv_"))
    assert len(priv_arrays) == 1, priv_arrays
    from dace import data
    assert isinstance(sdfg.arrays[priv_arrays[0]], data.Scalar)

    # Numerical correctness.
    n = 64
    rng = np.random.default_rng(0)
    a = rng.random(n)
    b = rng.random(n)
    dot = np.zeros(n)
    sdfg(a=a, b=b, dot=dot, N=n)
    assert np.isclose(dot[0], float((a * b).sum()))


# ---------------------------------------------------------------------------
# Multiple reductions in the same SDFG -- the pass must mint UNIQUE scalar
# names across loop nests (no collisions).
# ---------------------------------------------------------------------------


@dace.program
def _two_independent_reductions(a: dace.float64[N], b: dace.float64[N], dot: dace.float64[N], out: dace.float64[N]):
    # First reduction: dot += a*b.
    dot[0] = 0.0
    for i in range(N):
        dot[0] = dot[0] + a[i] * b[i]
    # Second reduction (target lives in a DIFFERENT array): out += a.
    out[0] = 0.0
    for i in range(N):
        out[0] = out[0] + a[i]


def test_two_reductions_get_unique_scalar_names():
    """Two parallel WCR-maps targeting different array elements -- each must
    get its OWN transient scalar; names must not collide."""
    sdfg = _build_wcr_map(_two_independent_reductions)
    arr_wcr_before, _ = _count_wcr_edges(sdfg)
    assert arr_wcr_before >= 2

    lifted = PrivatizeReductionAccumulator().apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 2

    priv_arrays = sorted(k for k in sdfg.arrays if k.startswith("_priv_"))
    # Two distinct privatized scalars, no duplicate names.
    assert len(priv_arrays) == 2
    assert len(set(priv_arrays)) == 2

    n = 64
    rng = np.random.default_rng(1)
    a = rng.random(n)
    b = rng.random(n)
    dot = np.zeros(n)
    out = np.zeros(n)
    sdfg(a=a, b=b, dot=dot, out=out, N=n)
    assert np.isclose(dot[0], float((a * b).sum()))
    assert np.isclose(out[0], float(a.sum()))


# ---------------------------------------------------------------------------
# Two reductions targeting the SAME array name (e.g. ``acc[0]`` and ``acc[1]``)
# -- still must mint distinct scalar names, AND the writeback subsets must
# point at the correct slots.
# ---------------------------------------------------------------------------


@dace.program
def _two_reductions_same_array(a: dace.float64[N], b: dace.float64[N], acc: dace.float64[N]):
    acc[0] = 0.0
    acc[1] = 0.0
    for i in range(N):
        acc[0] = acc[0] + a[i] * b[i]
    for i in range(N):
        acc[1] = acc[1] + a[i]


def test_two_reductions_same_array_get_unique_names():
    """Both reductions write into different slots of the same array."""
    sdfg = _build_wcr_map(_two_reductions_same_array)
    lifted = PrivatizeReductionAccumulator().apply_pass(sdfg, {})
    sdfg.validate()
    assert lifted == 2

    priv_arrays = sorted(k for k in sdfg.arrays if k.startswith("_priv_acc"))
    # Both scalars share the prefix '_priv_acc' but each carries a distinct
    # suffix from ``find_new_name=True``.
    assert len(priv_arrays) == 2
    assert len(set(priv_arrays)) == 2

    n = 64
    rng = np.random.default_rng(2)
    a = rng.random(n)
    b = rng.random(n)
    acc = np.zeros(n)
    sdfg(a=a, b=b, acc=acc, N=n)
    assert np.isclose(acc[0], float((a * b).sum()))
    assert np.isclose(acc[1], float(a.sum()))


# ---------------------------------------------------------------------------
# Idempotence: a second invocation must NOT re-privatize already-scalar WCRs.
# ---------------------------------------------------------------------------


def test_idempotent_no_double_privatize():
    sdfg = _build_wcr_map(_array_slot_dot)
    PrivatizeReductionAccumulator().apply_pass(sdfg, {})
    arrays_after_first = set(sdfg.arrays.keys())
    # Second invocation: pass must NOT add any new ``_priv_`` arrays nor
    # split any further states (the WCR target is already a Scalar).
    lifted2 = PrivatizeReductionAccumulator().apply_pass(sdfg, {})
    assert lifted2 is None
    assert set(sdfg.arrays.keys()) == arrays_after_first


if __name__ == "__main__":
    test_array_slot_dot_product_privatized()
    test_two_reductions_get_unique_scalar_names()
    test_two_reductions_same_array_get_unique_names()
    test_idempotent_no_double_privatize()
