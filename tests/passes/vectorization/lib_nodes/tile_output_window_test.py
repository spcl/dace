# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A tile op may write a WINDOW of a larger array, so the output-kind rule reads the memlet.

Design 6.2 says any Tile input means a tile-shaped output, and every tile node checked that against
the output DESCRIPTOR. ``WidenAccesses`` widens the memlet of a lane-indexed array in place instead
of swapping its descriptor -- CloudSC's ``zsolqa`` keeps its ``(nclv, nclv, klon)`` shape -- so the
tile write lands in a window of a larger array and the descriptor check called a perfectly good tile
a rule violation: "kind_a='Tile', kind_b='Scalar' (has Tile input) but '_c' descriptor is not
tile-shape (8,)". The vectorizer then refused the whole kernel and left it un-tiled.
"""
import os

os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')

import warnings

import numpy as np
import pytest

import dace
from dace.libraries.tileops.nodes.tile_binop import edge_moves_a_tile
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')


def edge_with_subset(subset: str, shape):
    """A single edge into an access node of ``shape``, carrying memlet ``subset``."""
    sdfg = dace.SDFG('window')
    sdfg.add_array('buf', shape, dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    tasklet = state.add_tasklet('w', dict(), {'_c'}, '_c = 1.0')
    return state.add_edge(tasklet, '_c', state.add_write('buf'), None, dace.Memlet(f'buf[{subset}]'))


@pytest.mark.parametrize('subset,shape', [('0, i:i+8', [3, 64]), ('i:i+8', [64]), ('0, 0, i:i+8', [2, 3, 64])])
def test_a_window_of_a_bigger_array_is_a_tile(subset, shape):
    """Leading single elements then the tile itself: exactly what ``_c[off]`` walks."""
    assert edge_moves_a_tile(edge_with_subset(subset, shape), (8, ))


@pytest.mark.parametrize('subset,shape', [('0, i:i+4', [3, 64]), ('0:3, i:i+8', [3, 64]), ('0, 0', [3, 64])])
def test_anything_else_is_not_a_tile(subset, shape):
    """A short window, a non-degenerate leading dim, and a scalar write all stay refusals."""
    assert not edge_moves_a_tile(edge_with_subset(subset, shape), (8, ))


@dace.program
def lane_indexed_buffer(a: dace.float64[N], out: dace.float64[N]):
    """CloudSC's ``zsolqa`` in miniature: a transient whose LAST dim is the vectorized one."""
    buf = np.zeros((3, N), dtype=np.float64)
    for i in range(N):
        for j in range(3):
            buf[j, i] = a[i] * (j + 1)
        out[i] = buf[0, i] + buf[1, i] + buf[2, i]


def test_a_windowed_tile_write_vectorizes_and_keeps_its_numbers():
    """End to end: the kernel tiles, and the widened window writes the cells it names."""
    sdfg = lane_indexed_buffer.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        VectorizeCPUMultiDim(
            VectorizeConfig(widths=(8, ), target_isa='SCALAR', remainder_strategy='masked_tail',
                            validate_all=True)).apply_pass(sdfg, {})
    sdfg.validate()
    # A refusal leaves the kernel correct but un-tiled, which would pass the numbers below without
    # exercising any of this -- the assertion has to be that it TILED.
    refusals = [str(m.message) for m in caught if 'refusing to vectorize' in str(m.message)]
    assert not refusals, refusals
    assert any(
        type(node).__name__.startswith('Tile') for sd in sdfg.all_sdfgs_recursive() for state in sd.states()
        for node in state.nodes()), 'kernel produced no tile lib nodes'

    a = np.random.rand(64)
    out = np.zeros(64)
    sdfg.compile()(a=a, out=out, N=64)
    assert np.allclose(out, a * 6.0, rtol=1e-12, atol=1e-12), f'{out[:4]} != {(a * 6.0)[:4]}'


if __name__ == '__main__':
    test_a_windowed_tile_write_vectorizes_and_keeps_its_numbers()
