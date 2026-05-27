# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TSVC baseline integration test: the un-transformed DaCe SDFG of every corpus
kernel computes what its numpy reference does.

This is the *baseline* arm of the TSVC suite -- the foundation the canonicalize
and LoopToMap value tests build on (a transform can only be checked value-preserving
once the un-transformed program is itself correct). For every kernel in
:mod:`tests.corpus.tsvc`, at **both** ``simplify=False`` and ``simplify=True``, we
build the SDFG, run it, and compare its outputs to the plain-numpy reference in
:mod:`tests.corpus.tsvc_numpy` on identical inputs.

Comparison always treats matching ``nan`` / ``inf`` / ``-inf`` as equal
(``equal_nan=True``; ``np.allclose`` already matches infinities of equal sign), so
a kernel whose reference legitimately produces a non-finite value is fine as long
as the SDFG produces the same one.
"""
import contextlib
import io

import numpy as np
import pytest

from tests.corpus import tsvc
from tests.corpus.tsvc_numpy import REFERENCES

_CORPUS = tsvc.collect()
_TOL = 1e-9


def _allclose(a, b) -> bool:
    return np.allclose(np.asarray(a), np.asarray(b), rtol=_TOL, atol=_TOL, equal_nan=True)


@pytest.mark.parametrize("simplify", [False, True], ids=["nosimplify", "simplify"])
@pytest.mark.parametrize("kernel", _CORPUS, ids=[k.name for k in _CORPUS])
def test_tsvc_baseline_matches_numpy(kernel, simplify, request):
    """The un-transformed SDFG (at this simplify level) matches the numpy reference."""
    l1, l2 = tsvc.lengths(kernel)
    arrays = tsvc.allocate(kernel, l1, l2, np.random.default_rng(1234))
    call_kwargs = {**tsvc.scalar_params(kernel, l1), **tsvc.symbols(kernel, l1, l2)}

    # numpy oracle (mutates its own copy in place)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[kernel.name](**ref, **call_kwargs)

    # DaCe scalar SDFG at the requested simplify level
    sdfg = tsvc.to_sdfg(kernel, request.node.name, simplify=simplify)
    got = {n: a.copy() for n, a in arrays.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        sdfg.compile()(**got, **call_kwargs)

    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue  # gather indices are read-only inputs
        assert _allclose(ref[name], got[name]), (
            f"{kernel.name}/{name} (simplify={simplify}): SDFG baseline diverges from numpy "
            f"reference, max|diff|={np.nanmax(np.abs(np.asarray(ref[name]) - np.asarray(got[name]))):.3e}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
