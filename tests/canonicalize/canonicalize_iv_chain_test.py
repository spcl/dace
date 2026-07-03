# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""E2e: induction-variable chains that free a loop to parallelize.

``s128`` is the headline case: a DERIVED-IV chain where the primary IV increment
``j := j + 2`` sits *between* two content blocks and a derived IV ``k := j + 1``
feeds the array gathers ``b[k]`` / ``c[k]``. Substituting the between-blocks
``j`` rewrites the derived iedge to ``k := 2 * i`` (affine), which symbol
propagation folds into the gathers, so ``LoopToMap`` parallelizes the whole loop.
The kernel is run to a numpy reference to confirm the substitution is correct,
not just that it parallelizes.
"""

import numpy as np
import pytest

from dace.sdfg.state import LoopRegion
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES


def _canonicalize_counts(name):
    kernel = [k for k in tsvc.collect() if k.name == name][0]
    sdfg = tsvc.to_sdfg(kernel, 'iv_' + name, simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4)

    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[kernel.name](**ref, **call_kwargs)
    got = {n: a.copy() for n, a in arrays.items()}
    sdfg.compile()(**got, **call_kwargs)
    for n, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue
        assert np.allclose(ref[n], got[n], equal_nan=True), f"{name}: value mismatch on {n}"

    nloops = sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)
    nmaps = sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry))
    return nloops, nmaps


def test_s128_derived_iv_chain_parallelizes():
    """s128: derived-IV chain (k := j+1 before j := j+2, between content blocks).
    Must reduce to affine k=2i and fully parallelize, value-preserving."""
    nloops, nmaps = _canonicalize_counts('s128_d_single')
    assert nloops == 0 and nmaps >= 1, \
        f"s128 (derived-IV chain) should fully parallelize, got loops={nloops} maps={nmaps}"


def test_s124_branch_uniform_iv_parallelizes():
    """s124: ``j += 1`` in BOTH branches of the conditional (branch-uniform IV).
    Hoisting the common increment out of the conditional lets IV substitution
    close it to ``j = i``, so ``a[j]`` becomes the parallel ``a[i]``."""
    nloops, nmaps = _canonicalize_counts('s124_d_single')
    assert nloops == 0 and nmaps >= 1, \
        f"s124 (branch-uniform IV) should fully parallelize, got loops={nloops} maps={nmaps}"


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
