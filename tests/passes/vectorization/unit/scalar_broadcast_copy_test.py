# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A bare scalar-to-tile copy must lower to a ``TileLoad(src_kind='Scalar')`` broadcast.

TSVC s293 is the shape::

    a0 = a[0]
    for i in range(N):
        a[i] = a0

``parallelize`` leaves that body as a single AccessNode -> AccessNode copy — no tasklet
anywhere. ``InsertTileLoadStore`` classifies the ``a0`` read as CONSTANT and used to stage it
through a ``Scalar`` bridge, which only works when a lib node CONSUMES the scalar and splats it.
A plain copy has no such consumer: the bridge -> output rewire then dropped the write's
``other_subset`` (the ``a[i:i+W]`` window), leaving a write no later pass could classify and an
orphaned ``_tile_iter_mask`` behind.

The source of a bare copy into a global array must therefore become a real ``(W,)`` tile via a
broadcast ``TileLoad``, paired with the ``TileStore`` that writes the window.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileLoad, TileStore
from dace.transformation.passes.parallelize import parallelize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')


@dace.program
def broadcast_scalar(a: dace.float64[N]):
    a0 = a[0]
    for i in range(N):
        a[i] = a0


def _vectorized(tag):
    sdfg = broadcast_scalar.to_sdfg(simplify=True)
    sdfg.name = tag
    parallelize(sdfg, validate=True, validate_all=False, peel_limit=4)
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def test_lowers_to_a_broadcast_load_and_a_tile_store():
    sdfg = _vectorized('bcast_copy_struct')
    loads = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileLoad)]
    stores = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileStore)]
    assert loads, 'the scalar source never became a tile'
    assert all(ld.src_kind == 'Scalar' for ld in loads), [ld.src_kind for ld in loads]
    assert stores, 'the tile is never stored back to the global array'


def test_the_stored_window_is_the_whole_tile():
    """The write subset must stay the per-tile window; collapsing it to a single element is the
    regression this guards (the dropped ``other_subset``)."""
    sdfg = _vectorized('bcast_copy_window')
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if not isinstance(node, TileStore):
                    continue
                out = [e for e in state.out_edges(node) if e.src_conn == '_dst']
                assert out, f'{node.label} has no _dst edge'
                for edge in out:
                    sizes = edge.data.subset.size()
                    assert any(str(s) == '8' for s in sizes), f'{node.label} stores {edge.data.subset}, not a tile'


def test_value_preserving():
    n = 61  # not a multiple of the width: exercises the masked remainder body
    rng = np.random.default_rng(11)
    a = rng.random(n)
    want = np.full(n, a[0])

    got = a.copy()
    _vectorized('bcast_copy_value').compile()(a=got, N=n)
    assert np.array_equal(got, want)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
