# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end semantic-op detection through the full canonicalization pipeline.

Locks in that the reduction / argmax targets are lifted or parallelized (not left
as sequential loops) AND stay value-preserving vs the numpy oracle:

* ``s313`` (vdot, ``dot += a[i] * b[i]``) -> a parallel Map with a WCR reduction.
* ``s315`` (1-D argmax tracking value **and** index) -> an ``ArgReduce`` libnode.
* ``s13110`` (2-D argmax, three carriers ``maxv`` / ``xindex`` / ``yindex``) ->
  an ``ArgReduce`` libnode.

These are exercised as full dace-Python (numpy-derived) kernels from the TSVC
corpus, so the assertions cover both structural detection and numerical result.
"""

import numpy as np
import pytest

from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES


def _canonicalize_and_check(name):
    """Canonicalize the named TSVC kernel, assert value-preservation vs the numpy
    oracle, and return (n_sequential_loops, {node_type: count}, has_wcr)."""
    kernel = [k for k in tsvc.collect() if k.name == name][0]
    sdfg = tsvc.to_sdfg(kernel, 'e2e_' + name, simplify=True)
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
    types = {}
    for node, _ in sdfg.all_nodes_recursive():
        types[type(node).__name__] = types.get(type(node).__name__, 0) + 1
    has_wcr = any(e.data is not None and e.data.wcr is not None for st in sdfg.all_states() for e in st.edges())
    return nloops, types, has_wcr


def test_vdot_s313_parallelizes_as_wcr_reduction():
    """vdot must become a parallel reduction (Map + WCR), not a sequential loop."""
    nloops, types, has_wcr = _canonicalize_and_check('s313_d_single')
    assert nloops == 0, "vdot must not remain a sequential loop"
    assert types.get('MapEntry', 0) >= 1 and has_wcr, "vdot should be a parallel WCR-map reduction"


def test_s315_argmax_with_index_lifts_to_argreduce():
    """1-D argmax tracking both the max value and its index must lift to ArgReduce."""
    nloops, types, _ = _canonicalize_and_check('s315_d_single')
    assert types.get('ArgReduce', 0) >= 1, "s315 (value+index argmax) must lift to an ArgReduce libnode"
    assert nloops == 0, "the argmax loop must be gone"


def test_s13110_2d_argmax_lifts_to_argreduce():
    """2-D argmax with three carriers (maxv / xindex / yindex) must lift to ArgReduce."""
    nloops, types, _ = _canonicalize_and_check('s13110_d_single')
    assert types.get('ArgReduce', 0) >= 1, "s13110 (2D 3-carrier argmax) must lift to an ArgReduce libnode"


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
