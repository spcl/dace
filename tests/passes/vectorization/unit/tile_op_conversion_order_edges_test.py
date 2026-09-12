# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ConvertTaskletsToTileOps`` must not read an ordering edge as dataflow.

An EMPTY memlet carries no data: ``StateFusionExtended`` mints one to sequence a WAR/WAW
hazard, and it has no connector. Every conversion in the pass resolves the produced value as
``out_edges[0]`` and then bulk-removes ``in_edges.values() + out_edges``, so an ordering edge
sorting first both steals the result connector and gets deleted along with the tasklet.

TSVC s1251 is the shape::

    for i in range(N):
        s = b[i] + c[i]
        b[i] = a[i] + d[i]
        a[i] = s * e[i]

``a`` and ``b`` are each read before being overwritten in the same iteration, so the two
statements are sequenced by ordering edges. Before the fix ``TileBinop._c`` was wired to the
empty edge and the real output dropped, failing validation with ``Node validation failed: None``.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop, TileUnop
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')


@dace.program
def read_then_overwrite_both(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N],
                             e: dace.float64[N]):
    for i in range(N):
        s = b[i] + c[i]
        b[i] = a[i] + d[i]
        a[i] = s * e[i]


def _vectorized(tag):
    sdfg = read_then_overwrite_both.to_sdfg(simplify=True)
    sdfg.name = tag
    canonicalize(sdfg, validate=True, peel_limit=4)
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def _compute_ops(sdfg):
    return [(state, node) for sd in sdfg.all_sdfgs_recursive() for state in sd.states() for node in state.nodes()
            if isinstance(node, (TileBinop, TileUnop))]


def test_every_compute_op_produces_a_real_output():
    """The defect directly: a result connector wired to an empty (data-less) memlet."""
    sdfg = _vectorized('tileop_order_struct')
    ops = _compute_ops(sdfg)
    assert ops, 'kernel did not vectorize -- the regression it guards would be untestable'
    for state, node in ops:
        results = [ed for ed in state.out_edges(node) if ed.src_conn in ('_c', '_o')]
        assert results, f'{node.label} has no result edge'
        for ed in results:
            assert ed.data is not None and not ed.data.is_empty(), (
                f'{node.label} result connector {ed.src_conn} carries an empty memlet')


def test_ordering_edges_survive_conversion():
    """A converted tasklet's happens-before edges must re-anchor onto the lib node, not vanish.

    Counted per state so a body that legitimately needed no ordering (fully independent
    statements) does not fail the assertion -- only a body that HAD ordering before the
    conversion and lost all of it would.
    """
    sdfg = read_then_overwrite_both.to_sdfg(simplify=True)
    sdfg.name = 'tileop_order_keep'
    canonicalize(sdfg, validate=True, peel_limit=4)
    before = sum(1 for sd in sdfg.all_sdfgs_recursive() for st in sd.states() for ed in st.edges()
                 if ed.data is not None and ed.data.is_empty())
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})
    sdfg.validate()
    after = sum(1 for sd in sdfg.all_sdfgs_recursive() for st in sd.states() for ed in st.edges()
                if ed.data is not None and ed.data.is_empty())
    if before:
        assert after, 'conversion dropped every happens-before edge in the body'


@pytest.mark.parametrize('n', [64, 61])
def test_value_preserving(n):
    """``n=61`` exercises the masked remainder tile as well as the full ones."""
    rng = np.random.default_rng(1251)
    a, b, c, d, e = (rng.random(n) for _ in range(5))
    ref_a, ref_b = a.copy(), b.copy()
    for i in range(n):
        s = ref_b[i] + c[i]
        ref_b[i] = ref_a[i] + d[i]
        ref_a[i] = s * e[i]

    sdfg = _vectorized(f'tileop_order_val_{n}')
    got_a, got_b = a.copy(), b.copy()
    sdfg(a=got_a, b=got_b, c=c.copy(), d=d.copy(), e=e.copy(), N=n)
    assert np.allclose(got_a, ref_a)
    assert np.allclose(got_b, ref_b)


if __name__ == '__main__':
    test_every_compute_op_produces_a_real_output()
    test_ordering_edges_survive_conversion()
    test_value_preserving(64)
    test_value_preserving(61)
