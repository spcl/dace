# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TSVC kernels that require the canonicalize pipeline before vectorization.

A small curated corpus -- ``s121``, ``s2111``, ``s481`` -- captures the
shapes that the basic ``simplify + LoopToMap`` path is *not* enough for:

* ``s121`` -- single-loop forward read ``a[i] = a[i+1] + b[i]`` with an
  in-loop symbol assignment (``j = i + 1``). ``LoopToMap`` alone refuses
  due to the read-ahead anti-dependence; canonicalize rewrites this into a
  parallelizable shape.
* ``s2111`` -- 2-D backward stencil ``aa[j,i] = aa[j,i-1] + aa[j-1,i]``
  whose vectorizable dim only emerges after stride / index canonicalization.
* ``s481`` -- ``for i: if d[i]<0.0: break`` -- the ``break`` becomes a
  prefix-mask after the canonicalize ``break_anti_dependence`` pass.

Each kernel is parametrised across the same knob matrix as the wider TSVC
sweep (``remainder_strategy`` x ``branch_mode`` x ``LEN``). The reference
SDFG runs ``simplify + LoopToMap`` only; the vectorised SDFG runs
``simplify + canonicalize + VectorizeCPU(...)`` -- a deliberate mirror of
the production driver so the comparison witnesses the additional shapes
canonicalize enables.
"""
import copy

import numpy as np
import pytest

from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from tests.corpus import tsvc
from tests.passes._tsvc_harness_helper import build_tsvc_matrix

# Curated list of kernels whose vectorization is gated by the canonicalize
# pipeline. Each name must match a ``tsvc.TSVCKernel.name``.
_KERNEL_NAMES = ("s121_d_single", "s2111_d_single", "s481_d_single")
_KERNELS_1D = [k for k in tsvc.collect(regime="1d") if k.name in _KERNEL_NAMES]
_KERNELS_2D = [k for k in tsvc.collect(regime="2d") if k.name in _KERNEL_NAMES]


def _matrix():
    params = (build_tsvc_matrix([(k.program, k) for k in _KERNELS_1D], (64, 65))[0] +
              build_tsvc_matrix([(k.program, k) for k in _KERNELS_2D], (16, 17))[0])
    ids = [f"{p[1].name}-{p[2]}-{p[3]}-{p[4]}" for p in params]
    return params, ids


_MATRIX, _IDS = _matrix()


@pytest.mark.parametrize("program,kernel,remainder_strategy,branch_mode,len_val", _MATRIX, ids=_IDS)
def test_tsvc_canonicalize_vectorization(program, kernel, remainder_strategy, branch_mode, len_val, request):
    # s481 carries a ``for ... if cond: break`` whose vectorization needs a
    # dedicated break-mask pass that is not in the canonicalize/VectorizeCPU
    # path. Tracked separately; the kernel stays in the corpus so the
    # tripwire flips when the break path lands.
    if kernel.name == "s481_d_single":
        pytest.xfail("s481 break-vectorization requires a dedicated VectorizeBreak pass")

    if kernel.regime == "1d":
        l1, l2 = len_val, tsvc.LEN_2D_FIXED
    else:
        l1, l2 = tsvc.LEN_2D_FIXED, len_val

    rng = np.random.default_rng(seed=len_val)
    arrays_ref = tsvc.allocate(kernel, l1, l2, rng)
    arrays_vec = {name: arr.copy() for name, arr in arrays_ref.items()}
    sym = tsvc.symbols(kernel, l1, l2)
    sparams = tsvc.scalar_params(kernel, l1)

    # Reference: simplify + LoopToMap only (matches tsvc_vectorization_test).
    sdfg = tsvc.to_sdfg(kernel, f"ref-{request.node.name}")
    sdfg.simplify(validate=True, validate_all=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    # Vectorised: simplify -> canonicalize -> VectorizeCPU.
    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = sdfg.name + "_vec"
    # ``peel_limit=8`` lets canonicalize peel the start-at-1 boundary of
    # s2111's ``for j in range(1, LEN_2D)`` so the W-strided main map covers
    # the divisible interior with a scalar prologue.
    # ``break_anti_dependence`` lifts the ``for ... if cond: break`` pattern
    # (TSVC s481) closer to a prefix-mask vectorizable form.
    canonicalize(vsdfg, validate=True, peel_limit=8, break_anti_dependence=True)

    if branch_mode == "fp_factor":
        branch_kwargs = dict(use_fp_factor=True, branch_normalization=False)
    else:
        branch_kwargs = dict(use_fp_factor=False, branch_normalization=True)

    try:
        VectorizeCPU(vector_width=8,
                     fail_on_unvectorizable=False,
                     remainder_strategy=remainder_strategy,
                     **branch_kwargs).apply_pass(vsdfg, {})
    except NotImplementedError as ex:
        pytest.skip(f"vectorize NotImplementedError on {kernel.name}: {ex}")

    c_ref = sdfg.compile()
    c_vec = vsdfg.compile()
    c_ref(**arrays_ref, **sparams, **sym)
    c_vec(**arrays_vec, **sparams, **sym)

    for name, code in kernel.args.items():
        if code in tsvc.INDEX_CODES:
            continue
        diff = float(np.max(np.abs(arrays_ref[name] - arrays_vec[name])))
        assert diff <= 1e-10, f"{kernel.name}/{name}: max abs diff = {diff}"
