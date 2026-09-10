# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" An example and test for the DoubleBuffering transformation. """
import dace
import numpy as np

from dace.transformation.passes.pattern_matching import match_patterns
from dace.transformation.dataflow import DoubleBuffering, InLocalStorage


@dace.program
def mm_double_buffered(A: dace.float32[256, 256], B: dace.float32[256, 256], C: dace.float32[256, 256]):
    # Write to C in 128x128 output tiles
    for tile_i, tile_j in dace.map[0:256:128, 0:256:128]:
        # Load inputs in increments of 8 (128x8 tiles)
        for tile_k in dace.map[0:256:8]:
            # Compute outer products on input tiles
            for k, i, j in dace.map[0:8, 0:128, 0:128]:
                with dace.tasklet:
                    a << A[tile_i + i, tile_k + k]
                    b << B[tile_k + k, tile_j + j]
                    c >> C(1, lambda x, y: x + y)[tile_i + i, tile_j + j]
                    c = a * b


def test_double_buffering():
    A = np.random.rand(256, 256).astype(np.float32)
    B = np.random.rand(256, 256).astype(np.float32)
    expected_C = A @ B
    C = np.zeros((256, 256), dtype=np.float32)

    sdfg = mm_double_buffered.to_sdfg()
    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(expected_C - C) / (256 * 256)
    print('Difference (before):', diff)

    # Apply local storage transformation on inner map (last two transformations)
    sdfg.simplify()
    for i in range(2):
        for match in reversed(list(match_patterns(sdfg, InLocalStorage, states=[sdfg.node(0)]))):
            match.apply(sdfg.node(0), sdfg)
            break
        else:
            raise ValueError('Local storage transformation not applied')

    applied = sdfg.apply_transformations(DoubleBuffering)
    if applied != 1:
        raise ValueError('Double-buffering transformation not applied')
    C = np.zeros((256, 256), dtype=np.float32)
    sdfg(A=A, B=B, C=C)

    diff2 = np.linalg.norm(expected_C - C) / (256 * 256)
    print('Difference (after):', diff2)

    assert (diff <= 1e-5 and diff2 <= 1e-5)


ROWS, COLS = 4, 3


def _row_sum_body():
    """Sums the tile it is given into ``o[k]``, as a nested SDFG over the whole containers."""
    sdfg = dace.SDFG('row_sum')
    sdfg.add_array('t', [COLS], dace.float64)
    sdfg.add_array('o', [ROWS], dace.float64)
    sdfg.add_symbol('k', dace.int64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map('r', dict(q='0:%d' % COLS))
    tasklet = state.add_tasklet('t', {'x'}, {'y'}, 'y = x')
    state.add_memlet_path(state.add_read('t'), entry, tasklet, dst_conn='x', memlet=dace.Memlet('t[q]'))
    state.add_memlet_path(tasklet,
                          exit_,
                          state.add_write('o'),
                          src_conn='y',
                          memlet=dace.Memlet('o[k]', wcr='lambda a, b: a + b'))
    return sdfg


def _tiled_row_sum():
    """A one-dimensional map staging ``A[k, :]`` in a transient tile read by a nested SDFG."""
    sdfg = dace.SDFG('double_buffered_nested')
    sdfg.add_array('A', [ROWS, COLS], dace.float64)
    sdfg.add_array('B', [ROWS], dace.float64)
    sdfg.add_transient('tile', [COLS], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map('m', dict(k='0:%d' % ROWS))
    tile = state.add_access('tile')
    node = state.add_nested_sdfg(_row_sum_body(), {'t'}, {'o'}, {'k': 'k'})
    state.add_memlet_path(state.add_read('A'), entry, tile, memlet=dace.Memlet('A[k, 0:%d]' % COLS))
    state.add_edge(tile, None, node, 't', dace.Memlet('tile[0:%d]' % COLS))
    state.add_memlet_path(node,
                          exit_,
                          state.add_write('B'),
                          src_conn='o',
                          memlet=dace.Memlet('B[k]', wcr='lambda a, b: a + b'))
    return sdfg, state, entry, tile


def test_double_buffering_of_a_nested_sdfg_input():
    """The tile gains a buffer dimension, so the connector reading it becomes a view of one buffer."""
    A = np.arange(ROWS * COLS, dtype=np.float64).reshape(ROWS, COLS).copy()
    expected = A.sum(axis=1)

    sdfg, state, entry, tile = _tiled_row_sum()
    sdfg.validate()

    DoubleBuffering.apply_to(sdfg, map_entry=entry, transient=tile, verify=True, save=False)
    sdfg.validate()

    B = np.zeros(ROWS)
    sdfg(A=A, B=B)
    assert np.allclose(B, expected)


if __name__ == '__main__':
    test_double_buffering()
    test_double_buffering_of_a_nested_sdfg_input()
