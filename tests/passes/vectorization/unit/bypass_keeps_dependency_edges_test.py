# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``BypassTrivialAssignTasklets`` must never touch a dependency edge.

Bypassing ``AN(src) -> [_out = _in] -> AN(dst) -> C`` re-sources each consumer ``C`` directly from
``src``, rebuilding the memlet from both ends of the bypassed chain. An EMPTY memlet carries no
data -- it only orders two nodes -- so rebuilding one as dataflow invents a copy the program never
had. When the destination is an array the enclosing nested SDFG only takes as an INPUT, that
invented copy is a write to a read-only input, which validation rejects.

TSVC s471 is the shape::

    for i in range(N):
        x[i] = b[i] + d[i] * d[i]
        b[i] = c[i] + d[i] * e[i]

``b`` is read then written in the same iteration, and the ordering edges that sequence the two
statements ran into ``d`` and ``e``. Rebuilt as data they became ``b -> d`` and ``b -> e``.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')


@dace.program
def read_then_overwrite(x: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N],
                        e: dace.float64[N]):
    for i in range(N):
        x[i] = b[i] + d[i] * d[i]
        b[i] = c[i] + d[i] * e[i]


def _vectorized(tag):
    sdfg = read_then_overwrite.to_sdfg(simplify=True)
    sdfg.name = tag
    canonicalize(sdfg, validate=True, peel_limit=4)
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def test_read_only_inputs_are_never_written():
    """The validator's own rule, asserted directly: nothing writes ``d`` or ``e``."""
    sdfg = _vectorized('bypass_dep_struct')
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if not isinstance(node, nd.AccessNode) or node.data not in ('d', 'e'):
                    continue
                real = [ed for ed in state.in_edges(node) if ed.data is not None and not ed.data.is_empty()]
                assert not real, f'write to read-only input {node.data} in {state.label}: {[str(r.data) for r in real]}'


def test_value_preserving():
    n = 64
    rng = np.random.default_rng(7)
    b, c, d, e = rng.random(n), rng.random(n), rng.random(n), rng.random(n)
    want_x = b + d * d
    want_b = c + d * e

    sdfg = _vectorized('bypass_dep_value')
    got_x, got_b = np.zeros(n), b.copy()
    sdfg.compile()(x=got_x, b=got_b, c=c, d=d, e=e, N=n)
    assert np.allclose(got_x, want_x, rtol=1e-12, atol=1e-12)
    assert np.allclose(got_b, want_b, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
