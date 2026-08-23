# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""E2e: induction-variable chains that free a loop to parallelize.

``s128`` is the headline case: a DERIVED-IV chain where the primary IV increment
``j := j + 2`` sits *between* two content blocks and a derived IV ``k := j + 1``
feeds the array gathers ``b[k]`` / ``c[k]``. Substituting the between-blocks
``j`` rewrites the derived iedge to ``k := 2 * i`` (affine), which symbol
propagation folds into the gathers, so ``LoopToMap`` parallelizes the whole loop.
The kernel is run to a numpy reference to confirm the substitution is correct,
not just that it parallelizes.

``s453`` and ``s122`` are the two shapes that need the closed form expanded at the
IV's USE SITES rather than the loop eliminated: ``s453``'s accumulator is READ by
the statement next to it (so it is neither eliminable nor fissionable), and
``s122``'s counter lives in a loop whose start AND stride are symbolic.
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
    return nloops, nmaps, sdfg


def test_s128_derived_iv_chain_parallelizes():
    """s128: derived-IV chain (k := j+1 before j := j+2, between content blocks).
    Must reduce to affine k=2i and fully parallelize, value-preserving."""
    nloops, nmaps, _sdfg = _canonicalize_counts('s128_d_single')
    assert nloops == 0 and nmaps >= 1, \
        f"s128 (derived-IV chain) should fully parallelize, got loops={nloops} maps={nmaps}"


def test_s124_branch_uniform_iv_parallelizes():
    """s124: ``j += 1`` in BOTH branches of the conditional (branch-uniform IV).
    Hoisting the common increment out of the conditional lets IV substitution
    close it to ``j = i``, so ``a[j]`` becomes the parallel ``a[i]``."""
    nloops, nmaps, _sdfg = _canonicalize_counts('s124_d_single')
    assert nloops == 0 and nmaps >= 1, \
        f"s124 (branch-uniform IV) should fully parallelize, got loops={nloops} maps={nmaps}"


def test_s453_use_site_iv_parallelizes():
    """s453: ``s = s + 2.0; a[i] = s * b[i]`` -- the IV is a DATA accumulator that the other
    statement READS, so it is neither eliminable nor fissionable. Expanding its closed form at
    the use site (``s == s_entry + 2.0*(i+1)`` after this iteration's update) leaves a pure
    per-element body."""
    nloops, nmaps, _sdfg = _canonicalize_counts('s453_d_single')
    assert nloops == 0 and nmaps >= 1, \
        f"s453 (use-site data IV) should fully parallelize, got loops={nloops} maps={nmaps}"


def test_s122_symbolic_start_and_stride_iv_parallelizes():
    """s122: ``for i in range(n1-1, LEN_1D, n3): k = k + j; a[i] = a[i] + b[LEN_1D-k]`` -- BOTH
    the start and the stride are symbolic, so the counter's closed form needs the trip index
    ``t = int_floor(i - start, stride)`` rather than ``i - start``."""
    nloops, nmaps, _sdfg = _canonicalize_counts('s122_d_single')
    assert nloops == 0 and nmaps >= 1, \
        f"s122 (symbolic start/stride IV) should fully parallelize, got loops={nloops} maps={nmaps}"


def test_s318_staged_counter_iv_lifts_to_an_arg_reduction():
    """s318: ``k += inc``, where ``inc`` is a scalar ARGUMENT rather than a literal.

    The frontend cannot put that increment on an interstate edge -- the step is data -- so it emits
    a tasklet writing a transient scalar and binds ``k`` to that slot. Promoting the read-only
    argument to a symbol and reading the binding through the one staging hop closes ``k`` to
    ``inc * i``, which turns the guarded body into the strided gather ``a[inc*i]`` the arg-reduction
    lift already handles. Both halves are load-bearing: without the closure the loop keeps a
    per-iteration tasklet and the lift refuses the body outright.
    """
    from dace.libraries.standard.nodes import ArgReduce
    nloops, nmaps, sdfg = _canonicalize_counts('s318_d_single')
    assert nloops == 0 and nmaps >= 1, \
        f"s318 (staged counter IV) should fully parallelize, got loops={nloops} maps={nmaps}"
    assert sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, ArgReduce)) == 1, \
        "the strided argmax must lift to a single ArgReduce, not to a value-only reduction"


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
