# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TSVC ``VectorizeCPU`` correctness sweep over the whole kernel corpus.

Every kernel in :mod:`tests.corpus.tsvc` is run through ``VectorizeCPU`` and
compared, output array by output array, against the un-vectorized reference
(``simplify`` + ``LoopToMap``). This replaces the former per-block files
(``tsvc_vectorization_test_block{1,2,3,4}`` and ``tsvc_vectorization_test_2d``):
the kernels now live once in the corpus and the matrix is built from there.

Each kernel is parametrised across:

* ``LEN`` -- ``{64, 65}`` for 1D-regime kernels (64 is divisible by ``W = 8`` so
  P2 proves divisibility and emits no remainder; 65 forces a non-empty
  remainder) and ``{16, 17}`` for 2D-regime kernels.
* ``remainder_strategy`` in ``{scalar, masked}`` -- the remainder *shape*.
* ``branch_mode`` in ``{merge, fp_factor}`` -- ``fp_factor`` paired with
  ``masked`` is rejected by VectorizeCPU's locked plan rule and skipped;
  ``fp_factor`` on a branchless kernel produces an identical SDFG to ``merge``
  and is likewise pruned (see :func:`tests.passes._tsvc_harness_helper.build_tsvc_matrix`).
"""
import copy

import numpy as np
import pytest

from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from tests.corpus import tsvc
from tests.passes._tsvc_harness_helper import build_tsvc_matrix

_G1D = tsvc.collect(regime="1d")
_G2D = tsvc.collect(regime="2d")


def _matrix():
    """Build the parametrize matrix from the corpus, one LEN regime at a time.

    ``build_tsvc_matrix`` keys its branch/remainder pruning off ``kt[0]`` (the
    ``@dace.program``), so each kernel is passed as ``(program, TSVCKernel)``;
    the kernel record travels through as ``kt[1]``. IDs use the short kernel name.
    """
    params = (build_tsvc_matrix([(k.program, k) for k in _G1D],
                                (64, 65))[0] + build_tsvc_matrix([(k.program, k) for k in _G2D], (16, 17))[0])
    # param = (program, kernel, remainder_strategy, branch_mode, LEN)
    ids = [f"{p[1].name}-{p[2]}-{p[3]}-{p[4]}" for p in params]
    return params, ids


_MATRIX, _IDS = _matrix()


@pytest.mark.parametrize("program,kernel,remainder_strategy,branch_mode,len_val", _MATRIX, ids=_IDS)
def test_tsvc_vectorization(program, kernel, remainder_strategy, branch_mode, len_val, request):
    if kernel.regime == "1d":
        l1, l2 = len_val, tsvc.LEN_2D_FIXED
    else:
        l1, l2 = tsvc.LEN_2D_FIXED, len_val  # l1 is unused for pure-2D kernels

    rng = np.random.default_rng(seed=len_val)
    arrays_ref = tsvc.allocate(kernel, l1, l2, rng)
    arrays_vec = {name: arr.copy() for name, arr in arrays_ref.items()}
    sym = tsvc.symbols(kernel, l1, l2)
    sparams = tsvc.scalar_params(kernel, l1)

    # Uniquely-named per variant (xdist-safe; see tsvc.to_sdfg).
    sdfg = tsvc.to_sdfg(kernel, f"ref-{request.node.name}")
    sdfg.simplify(validate=True, validate_all=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = sdfg.name + "_vec"

    if branch_mode == "fp_factor":
        branch_kwargs = dict(use_fp_factor=True, branch_normalization=False)
    else:
        branch_kwargs = dict(use_fp_factor=False, branch_normalization=True)

    # NotImplementedError propagates as a real test failure (per design
    # directive: an unsupported kernel pattern must surface, never silently
    # skip). Configuration-mismatch skips happen UPSTREAM in the
    # parametrize matrix.
    VectorizeCPU(vector_width=8,
                 fail_on_unvectorizable=False,
                 remainder_strategy=remainder_strategy,
                 **branch_kwargs).apply_pass(vsdfg, {})

    c_ref = sdfg.compile()
    c_vec = vsdfg.compile()

    c_ref(**arrays_ref, **sparams, **sym)
    c_vec(**arrays_vec, **sparams, **sym)

    for name, code in kernel.args.items():
        if code in tsvc.INDEX_CODES:
            continue  # gather indices are read-only, never mutated
        # Some TSVC recurrences (s232's ``aa[j, i] = aa[j, i - 1]**2 +
        # bb[j, i]`` squares on every step and diverges to +inf on random
        # [0, 1) data); the reference and the vectorized output both
        # produce the same inf / nan pattern. ``np.allclose(equal_nan=
        # True)`` treats matching nans / infs as equal so the carried-dep
        # divergence is not a false positive — a genuine numerical
        # mismatch still trips the rtol/atol threshold.
        assert np.allclose(arrays_ref[name], arrays_vec[name], rtol=1e-10, atol=1e-10, equal_nan=True), (
            f"{kernel.name}/{name}: max abs diff = "
            f"{np.nanmax(np.abs(arrays_ref[name] - arrays_vec[name]))}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
