# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`PrepareReductionForWidening`.

The pass scalar-localizes an ARRAY-SLOT WCR reduction (``s[3] += a[i] * b[i]`` -- a map-exit WCR
into one element of a multi-element array) into a private transient scalar accumulator + init +
writeback, so the tile/SIMD widener folds it to per-lane partial sums + one horizontal
``TileReduce`` instead of bailing (``no loose WCR in the map body``).

Coverage:

* POSITIVE e2e: a dot product / a sum whose accumulator is a genuine array slot -- BAILS through the
  vectorize pipeline with the prep disabled (proven via a no-op patch), widens + is numerically
  correct with the prep on.
* POSITIVE structural: the array-slot WCR target becomes a transient ``Scalar`` after the pass.
* NEGATIVE: a plain scalar / length-1 accumulator (already widenable), a ``min`` RMW (not a WCR),
  and a cross-iteration recurrence are all left untouched -- value-preserving.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from unittest import mock

import numpy as np
import pytest

import dace
from dace import data
from dace.sdfg import nodes
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, RemainderStrategy, BranchMode
from dace.transformation.passes.vectorization.reduction_scalar_local_prep import PrepareReductionForWidening
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol("N")

# ---------------------------------------------------------------------------
# Kernels: reductions whose accumulator is a genuine multi-element ARRAY SLOT.
# ---------------------------------------------------------------------------


@dace.program
def dot_into_slot(a: dace.float64[N], b: dace.float64[N], s: dace.float64[8]):
    for i in dace.map[0:N]:
        s[3] += a[i] * b[i]


@dace.program
def sum_into_slot(a: dace.float64[N], s: dace.float64[4]):
    for i in dace.map[0:N]:
        s[2] += a[i]


@dace.program
def prod_into_slot(a: dace.float64[N], s: dace.float64[4]):
    for i in dace.map[0:N]:
        s[1] *= a[i]


# Negative: RMW min into an array slot -- the frontend emits ``s[2] = min(s[2], a[i])`` as a
# read+write, NOT a WCR, so the prep must not match it.
@dace.program
def min_rmw_slot(a: dace.float64[N], s: dace.float64[4]):
    for i in dace.map[0:N]:
        s[2] = min(s[2], a[i])


# Negative: a genuine scan recurrence (``a[i] = a[i-1] + a[i]``) -- LoopToMap keeps it sequential,
# no WCR is minted, so nothing to privatize; must stay value-preserving.
@dace.program
def scan_recurrence(a: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i - 1] + a[i]


def _wcr_targets(sdfg: dace.SDFG):
    """(#WCR edges into a multi-element Array, #WCR edges into a Scalar) across all states."""
    arr_elem, scalar = 0, 0
    for st in sdfg.all_states():
        for e in st.edges():
            if e.data is None or e.data.wcr is None:
                continue
            desc = sdfg.arrays.get(e.data.data)
            if desc is None:
                continue
            if isinstance(desc, data.Scalar) or desc.total_size == 1:
                scalar += 1
            else:
                arr_elem += 1
    return arr_elem, scalar


# ---------------------------------------------------------------------------
# Structural: the pass rewrites the array-slot WCR into a scalar accumulator.
# ---------------------------------------------------------------------------


def test_array_slot_wcr_becomes_scalar():
    """``s[3] += a[i]*b[i]`` -- the array-element WCR target becomes a transient Scalar; an init +
    writeback state materialize; count == 1."""
    sdfg = dot_into_slot.to_sdfg(simplify=True)
    arr_before, scalar_before = _wcr_targets(sdfg)
    assert arr_before == 1 and scalar_before == 0, "fixture must start with one array-slot WCR"
    n_states_before = len(list(sdfg.all_states()))

    count = PrepareReductionForWidening().apply_pass(sdfg, {})
    sdfg.validate()
    assert count == 1

    arr_after, scalar_after = _wcr_targets(sdfg)
    assert arr_after == 0 and scalar_after == 1, "WCR must now target a scalar, no array-slot WCR left"
    priv = [k for k, v in sdfg.arrays.items() if k.startswith("_priv_") and isinstance(v, data.Scalar) and v.transient]
    assert len(priv) == 1, priv
    # Cross-state form: an init state (seed from the slot) + a writeback state were added.
    assert len(list(sdfg.all_states())) == n_states_before + 2


@pytest.mark.parametrize("prog", [min_rmw_slot, sum_into_slot, dot_into_slot, prod_into_slot])
def test_prep_is_value_preserving_structural(prog):
    """After the pass the SDFG still validates and holds no array-slot WCR (either privatized to a
    scalar, or -- for the RMW ``min`` -- never a WCR to begin with)."""
    sdfg = prog.to_sdfg(simplify=True)
    PrepareReductionForWidening().apply_pass(sdfg, {})
    sdfg.validate()
    assert _wcr_targets(sdfg)[0] == 0


def test_scalar_accumulator_is_left_alone():
    """A length-1 accumulator ``s[0] += a[i]`` (already widenable) must NOT be rewritten."""

    @dace.program
    def sum_scalar(a: dace.float64[N], s: dace.float64[1]):
        for i in dace.map[0:N]:
            s[0] += a[i]

    sdfg = sum_scalar.to_sdfg(simplify=True)
    count = PrepareReductionForWidening().apply_pass(sdfg, {})
    assert not count, "a scalar / length-1 accumulator already widens -- leave it alone"
    assert not any(k.startswith("_priv_") for k in sdfg.arrays)


def test_min_rmw_slot_is_left_alone():
    """``s[2] = min(s[2], a[i])`` is an RMW read+write, not a WCR reduction -- not matched."""
    sdfg = min_rmw_slot.to_sdfg(simplify=True)
    count = PrepareReductionForWidening().apply_pass(sdfg, {})
    assert not count


def test_recurrence_is_left_alone():
    """A scan recurrence carries no WCR -- the pass must not touch it."""
    sdfg = scan_recurrence.to_sdfg(simplify=True)
    count = PrepareReductionForWidening().apply_pass(sdfg, {})
    assert not count


# ---------------------------------------------------------------------------
# e2e: through the multi-dim tile vectorizer -- bails without prep, correct with it.
# ---------------------------------------------------------------------------


def _vectorize(prog, name):
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.name = name
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR, remainder_strategy=RemainderStrategy.MASKED_TAIL,
                        branch_mode=BranchMode.MERGE)).apply_pass(sdfg, {})
    sdfg.validate()
    for sym in {str(x) for x in sdfg.free_symbols}:
        if sym not in sdfg.symbols:
            sdfg.add_symbol(sym, dace.int64)
    return sdfg


def test_array_slot_dot_bails_without_prep_and_widens_with_it():
    """The dot-into-slot reduction BAILS through the vectorizer with the prep disabled, and widens
    to a correct result with it on -- the "no-widen before, correct after" case."""
    # BEFORE: with the prep patched to a no-op, the array-slot WCR reaches the tiler as a loose
    # in-body WCR and the pipeline's precondition fires.
    with mock.patch.object(PrepareReductionForWidening, "apply_pass", lambda self, sdfg, res: None):
        with pytest.raises((AssertionError, dace.sdfg.validation.InvalidSDFGError)):
            _vectorize(dot_into_slot, "dot_into_slot_noprep")

    # AFTER: the wired-in prep scalar-localizes it and the widened result matches numpy.
    n = 60
    a = np.random.random(n)
    b = np.random.random(n)
    s = np.zeros(8)
    ref = np.zeros(8)
    ref[3] = float((a * b).sum())
    sdfg = _vectorize(dot_into_slot, "dot_into_slot_prep")
    sdfg.compile()(a=a.copy(), b=b.copy(), s=s, N=n)
    assert np.allclose(s, ref, rtol=1e-9, atol=1e-12), f"got {s}, ref {ref}"


def test_sum_into_slot_widens_and_correct():
    """Sum into a fixed slot ``s[2] += a[i]`` -- widened result matches numpy (all-zero elsewhere)."""
    n = 55
    a = np.random.random(n)
    s = np.zeros(4)
    ref = np.zeros(4)
    ref[2] = float(a.sum())
    sdfg = _vectorize(sum_into_slot, "sum_into_slot_prep")
    sdfg.compile()(a=a.copy(), s=s, N=n)
    assert np.allclose(s, ref, rtol=1e-9, atol=1e-12), f"got {s}, ref {ref}"


def test_prod_into_slot_widens_and_correct():
    """Product into a fixed slot ``s[1] *= a[i]`` -- widened result matches numpy."""
    n = 40
    a = np.random.random(n) * 0.5 + 0.75  # keep away from 0 so the product is well-conditioned
    s = np.ones(4)  # product seed 1.0 in slot 1
    ref = np.ones(4)
    ref[1] = float(np.prod(a))
    sdfg = _vectorize(prod_into_slot, "prod_into_slot_prep")
    sdfg.compile()(a=a.copy(), s=s, N=n)
    assert np.allclose(s, ref, rtol=1e-9, atol=1e-12), f"got {s}, ref {ref}"


def test_recurrence_value_preserving_through_pipeline():
    """A scan recurrence stays sequential + correct through the pipeline (prep is a no-op on it)."""
    n = 32
    a = np.random.random(n)
    ref = a.copy()
    for i in range(1, n):
        ref[i] = ref[i - 1] + ref[i]
    sdfg = _vectorize(scan_recurrence, "scan_recurrence_prep")
    got = a.copy()
    sdfg.compile()(a=got, N=n)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12), f"got {got[:6]}, ref {ref[:6]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
