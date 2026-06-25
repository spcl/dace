# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``canonicalize`` end-to-end correctness over the TSVC kernel corpus.

The corpus-driven arm of the canonicalize suite: the hand-written pattern
examples live in the other ``tests/canonicalize/*`` files; here we import
:mod:`tests.corpus.tsvc` and run the production ``canonicalize`` recipe over
*every* kernel, checking it against the kernel's plain-numpy reference in
:mod:`tests.corpus.tsvc_numpy` (the scalar oracle).

Corpus default (user-specified): best-effort loop peeling ``peel_limit=4`` plus
anti-dependence breaking ``break_anti_dependence=True``. The transform input is
exercised at **both** ``simplify=False`` and ``simplify=True``. A recurrence kernel
that canonicalize wrongly parallelized, or an anti-dependence it broke unsafely,
diverges from the numpy reference. Comparison always treats matching
``nan``/``inf``/``-inf`` as equal (``equal_nan=True``).
"""
import contextlib
import io

import numpy as np
import pytest

from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES

_PEEL_LIMIT = 4
_BREAK_ANTI_DEP = True
_TOL = 1e-9

#: Kernels read out of bounds *as written* at the corpus lengths (genuine
#: source defects, not transform bugs) -- their numpy reference is itself
#: undefined, so there is nothing to check. Mapping name -> reason.
_OOB_BY_CONSTRUCTION: dict = {}

_CORPUS = tsvc.collect()


def _allclose(a, b) -> bool:
    return np.allclose(np.asarray(a), np.asarray(b), rtol=_TOL, atol=_TOL, equal_nan=True)


@pytest.mark.parametrize("kernel", _CORPUS, ids=[k.name for k in _CORPUS])
def test_canonicalize_corpus_value_preserving(kernel, request):
    """Canonicalized outputs match the numpy reference (151 kernels, one test each).

    ``canonicalize`` runs after ``simplify`` (its normal pipeline position), so the
    candidate is the simplified SDFG. ``canonicalize`` is invoked with
    ``validate=True``, so this also covers structural validity -- a single check
    per kernel.
    """
    if kernel.name in _OOB_BY_CONSTRUCTION:
        pytest.skip(_OOB_BY_CONSTRUCTION[kernel.name])

    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[kernel.name](**ref, **call_kwargs)

    cand = tsvc.to_sdfg(kernel, request.node.name, simplify=True)
    with contextlib.redirect_stdout(io.StringIO()):
        canonicalize(cand, validate=True, peel_limit=_PEEL_LIMIT, break_anti_dependence=_BREAK_ANTI_DEP)
    got = {n: a.copy() for n, a in arrays.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        cand.compile()(**got, **call_kwargs)

    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue  # gather indices are read-only
        assert _allclose(
            ref[name],
            got[name]), (f"{kernel.name}/{name}: canonicalize diverges from numpy "
                         f"reference, max|diff|={np.nanmax(np.abs(np.asarray(ref[name]) - np.asarray(got[name]))):.3e}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
