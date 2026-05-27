# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` end-to-end correctness over the TSVC kernel corpus.

The corpus-driven arm of the LoopToMap suite: the hand-written accept/refuse
contracts live in the other ``loop_to_map_*`` files (e.g.
:mod:`tests.transformations.loop_to_map_carried_symbol_test`); here we import
:mod:`tests.corpus.tsvc` and apply ``LoopToMap`` repeatedly over *every* kernel,
checking the result against the kernel's plain-numpy reference in
:mod:`tests.corpus.tsvc_numpy`.

This is the safety net for parallelization correctness: a TSVC kernel with a
loop-carried dependence (a recurrence such as ``s212``'s ``a[i]=a[i-1]*b[i]``, the
wrap-around induction ``s291``, or a prefix scan) must either be *refused* by
LoopToMap (stays a sequential ``LoopRegion``) or, if turned into a Map, still
compute the reference result. A Map that silently pins a carried value and diverges
is caught here. Comparison treats matching ``nan``/``inf``/``-inf`` as equal.
"""
import contextlib
import copy
import io

import numpy as np
import pytest

from dace.transformation.interstate import LoopToMap
from tests.corpus import tsvc
from tests.corpus.tsvc_numpy import REFERENCES

_TOL = 1e-9

#: Kernels read out of bounds *as written* at the corpus lengths (genuine source
#: defects; the numpy reference is itself undefined). Mapping name -> reason.
_OOB_BY_CONSTRUCTION: dict = {}

_CORPUS = tsvc.collect()


def _allclose(a, b) -> bool:
    return np.allclose(np.asarray(a), np.asarray(b), rtol=_TOL, atol=_TOL, equal_nan=True)


@pytest.mark.parametrize("kernel", _CORPUS, ids=[k.name for k in _CORPUS])
def test_loop_to_map_corpus_value_preserving(kernel, request):
    """``LoopToMap`` (applied to fixpoint on the simplified SDFG) keeps the
    numpy-reference result.

    LoopToMap runs after ``simplify`` (its normal position in the pipeline), so the
    candidate is the simplified SDFG. Whatever LoopToMap chooses to parallelize, the
    outputs must still match the reference -- a carried dependence it parallelized
    unsafely diverges.
    """
    if kernel.name in _OOB_BY_CONSTRUCTION:
        pytest.skip(_OOB_BY_CONSTRUCTION[kernel.name])

    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[kernel.name](**ref, **call_kwargs)

    cand = tsvc.to_sdfg(kernel, request.node.name, simplify=True)
    with contextlib.redirect_stdout(io.StringIO()):
        cand.apply_transformations_repeated(LoopToMap())
    got = {n: a.copy() for n, a in arrays.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        cand.compile()(**got, **call_kwargs)

    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue  # gather indices are read-only
        assert _allclose(
            ref[name],
            got[name]), (f"{kernel.name}/{name}: LoopToMap diverges from numpy "
                         f"reference, max|diff|={np.nanmax(np.abs(np.asarray(ref[name]) - np.asarray(got[name]))):.3e}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
