# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One unit test per ``subsets.Range.squeeze()`` call site in the DaCe Python frontend.

``squeeze()`` drops every size-1 dimension and cannot, on its own, tell a length-1 SLICE
(``k:k+1``, whose axis must survive to match numpy) from a scalar INDEX (``k``, whose axis
should collapse). Each frontend site that squeezes an access subset is therefore a candidate
length-1-slice miscompile. This file pins one representative access per site so a regression at
any single site is caught, and documents -- via ``xfail(strict=True)`` -- the sites that still
diverge from numpy (those flip to XPASS, the trigger to promote, once fixed).

Sites (in ``dace/frontend/python/newast.py``), from the 2026-07 squeeze-site audit:
  * ``_add_read_slice``            -- returns a View sized from the squeezed subset.  FIXED.
  * ``make_slice``                 -- sizes a slice transient (closure/global read).
  * ``_add_assignment``            -- squeezes both sides of ``lhs[...] = rhs[...]`` to broadcast.
  * ``_add_aug_assignment``        -- squeezes then UNSQUEEZES (balanced); a safe regression guard.
  * ``add_indirection_subgraph``   -- squeezes to build the indirection map; size-1 dims are trivial
                                      iterations (safe), guarded here so that stays true.
The bit-exact value oracle is numpy; integer data makes equality meaningful.
"""
import numpy as np
import pytest

import dace

N, M = 4, 5


def _2d():
    return np.arange(N * M, dtype=np.int64).reshape(N, M)


# --------------------------------------------------------------------------- #
# _add_read_slice (FIXED): a returned length-1 slice keeps its axis.
# --------------------------------------------------------------------------- #
@dace.program
def read_slice_view(A):
    return A[:, 1:2]


def test_add_read_slice_keeps_length1_axis():
    arr = _2d()
    got = np.asarray(read_slice_view(arr.copy()))
    oracle = arr[:, 1:2]
    assert got.shape == oracle.shape == (N, 1)
    assert np.array_equal(got, oracle)


# --------------------------------------------------------------------------- #
# make_slice: a length-1 slice off a CLOSURE / global array.
# --------------------------------------------------------------------------- #
_GLOBAL = np.arange(N * M, dtype=np.int64).reshape(N, M)


@dace.program
def read_closure_slice():
    return _GLOBAL[:, 1:2]


@pytest.mark.xfail(strict=True,
                   reason="make_slice (newast.py:5661) squeezes with no ignore_indices, and the "
                   "closure-global slice-return path is itself unsupported (raises on the captured "
                   "view). Both are follow-on work; the primary _add_read_slice site is fixed. XPASS "
                   "here is the trigger to promote once make_slice threads slice provenance.")
def test_make_slice_closure_keeps_length1_axis():
    got = np.asarray(read_closure_slice())
    oracle = _GLOBAL[:, 1:2]
    assert got.shape == oracle.shape == (N, 1), f"make_slice site: got {got.shape}, want {oracle.shape}"
    assert np.array_equal(got, oracle)


# --------------------------------------------------------------------------- #
# _add_assignment: broadcasting a length-1 slice across an assignment target. This is a VALUE
# check, not just shape -- the audit flagged this site as a silent miscompile.
# --------------------------------------------------------------------------- #
@dace.program
def assign_broadcast_from_slice(A: dace.int64[N, M], B: dace.int64[N, M]):
    A[:, :] = B[:, 1:2]


def test_add_assignment_broadcast_from_slice_is_value_exact():
    rng = np.random.default_rng(0)
    B = rng.integers(0, 100, size=(N, M)).astype(np.int64)
    A = np.zeros((N, M), dtype=np.int64)
    oracle = A.copy()
    oracle[:, :] = B[:, 1:2]  # numpy broadcasts column 1 across every column

    got = A.copy()
    assign_broadcast_from_slice(A=got, B=B.copy())
    assert np.array_equal(got, oracle), f"broadcast-assign miscompile: got\n{got}\noracle\n{oracle}"


# --------------------------------------------------------------------------- #
# _add_aug_assignment (SAFE: squeeze + unsqueeze balanced) -- matching-rank slice aug-assign.
# --------------------------------------------------------------------------- #
@dace.program
def aug_assign_matching_slice(A: dace.int64[N, M], B: dace.int64[N, M]):
    A[:, 1:2] += B[:, 1:2]


def test_add_aug_assignment_matching_slice_is_value_exact():
    rng = np.random.default_rng(1)
    A0 = rng.integers(0, 100, size=(N, M)).astype(np.int64)
    B = rng.integers(0, 100, size=(N, M)).astype(np.int64)
    oracle = A0.copy()
    oracle[:, 1:2] += B[:, 1:2]

    got = A0.copy()
    aug_assign_matching_slice(A=got, B=B.copy())
    assert np.array_equal(got, oracle)


# --------------------------------------------------------------------------- #
# add_indirection_subgraph (SAFE): a length-1 slice alongside an indirect index. The size-1 dim
# is a trivial map iteration; the user-visible result must still match numpy.
# --------------------------------------------------------------------------- #
@dace.program
def indirect_with_slice(A: dace.int64[N, M], idx: dace.int64[N]):
    out = np.zeros((N, 1), dtype=np.int64)
    for i in dace.map[0:N]:
        out[i, :] = A[idx[i], 1:2]
    return out


def test_add_indirection_subgraph_with_slice_is_value_exact():
    rng = np.random.default_rng(2)
    A = rng.integers(0, 100, size=(N, M)).astype(np.int64)
    idx = rng.integers(0, N, size=(N, )).astype(np.int64)
    got = np.asarray(indirect_with_slice(A.copy(), idx.copy()))
    oracle = np.stack([A[idx[i], 1:2] for i in range(N)])
    assert got.shape == oracle.shape
    assert np.array_equal(got, oracle)


if __name__ == "__main__":
    test_add_read_slice_keeps_length1_axis()
    test_make_slice_closure_keeps_length1_axis()
    test_add_assignment_broadcast_from_slice_is_value_exact()
    test_add_aug_assignment_matching_slice_is_value_exact()
    test_add_indirection_subgraph_with_slice_is_value_exact()
    print("OK")
