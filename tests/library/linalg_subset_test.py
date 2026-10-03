# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TensorTranspose`` and ``TensorDot`` on a SUBSET of a container, and the extents that guard them.

Every pre-existing case moves a whole array, so the descriptor and the memlet agreed and reading
the wrong one of the two was invisible. A write into a slice is the shape that separates them:
densenet's dense block appends 32 new channels into channels 64:96 of a 256-channel concatenation
buffer, and comparing the moved shape against the 256-channel CONTAINER rejects a write that is
correct.
"""
import numpy as np
import pytest

import dace
from dace import symbolic
from dace.libraries.linalg import TensorDot, TensorTranspose

B = dace.symbol('B', dtype=dace.int64)
H = dace.symbol('H', dtype=dace.int64)
W = dace.symbol('W', dtype=dace.int64)
K = dace.symbol('K', dtype=dace.int64)
P = dace.symbol('P', dtype=dace.int64)
N = dace.symbol('N', dtype=dace.int64)


def transpose_nodes(sdfg: dace.SDFG) -> list:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TensorTranspose)]


@dace.program
def append_channels(x: dace.float64[B, H, W, 32], y: dace.float64[B, 256, H, W]):
    y[:, 64:96] = np.transpose(x, (0, 3, 1, 2))


def test_transpose_into_a_slice_of_a_larger_container():
    """The permuted input matches the SUBSET the output memlet carries, not the container it is cut
    from. Reading the descriptor instead raised ``The permutation of the input shape does not match
    the output shape`` on a 32-into-256 channel append."""
    sdfg = append_channels.to_sdfg(simplify=True)
    sdfg.validate()
    assert transpose_nodes(sdfg), 'the transpose must still be a library node after simplify'

    b, h, w = 2, 3, 5
    x = np.random.default_rng(20260830).random((b, h, w, 32))
    y = np.full((b, 256, h, w), -1.0)
    sdfg(x=x.copy(), y=y, B=b, H=h, W=w)
    assert np.allclose(y[:, 64:96], np.transpose(x, (0, 3, 1, 2))), 'the slice got the wrong data'
    # The container is what a descriptor-shaped transpose would have written over.
    assert np.all(y[:, :64] == -1.0) and np.all(y[:, 96:] == -1.0), 'the transpose wrote outside its subset'


def test_transpose_out_of_a_slice_of_a_larger_container():
    """The input side is the same rule and the sharper failure: the extents drive the expansion's
    map range, so a container-shaped read walks 256 channels of a 32-channel move. Built by hand
    because the frontend materializes ``x[:, :, :, 64:96]`` into a transient first -- which is why
    only the output side of this ever showed up."""
    sdfg = dace.SDFG('slice_source_transpose')
    sdfg.add_array('x', (B, H, W, 256), dace.float64)
    sdfg.add_array('y', (B, 32, H, W), dace.float64)
    state = sdfg.add_state(is_start_block=True)
    node = TensorTranspose('_TensorTranspose', axes=[0, 3, 1, 2])
    state.add_node(node)
    state.add_edge(state.add_read('x'), None, node, '_inp_tensor', dace.Memlet('x[0:B, 0:H, 0:W, 64:96]'))
    state.add_edge(node, '_out_tensor', state.add_write('y'), None, dace.Memlet('y[0:B, 0:32, 0:H, 0:W]'))
    sdfg.validate()

    b, h, w = 2, 3, 5
    x = np.random.default_rng(20260831).random((b, h, w, 256))
    y = np.zeros((b, 32, h, w))
    sdfg(x=x.copy(), y=y, B=b, H=h, W=w)
    assert np.allclose(y, np.transpose(x[:, :, :, 64:96], (0, 3, 1, 2))), 'the slice read the wrong data'


def make_slice_transpose_sdfg(out_channels: int) -> tuple:
    """A hand-built ``[B,H,W,32] -> y[:, 64:96]`` transpose into a container ``out_channels`` wide."""
    sdfg = dace.SDFG('slice_transpose')
    sdfg.add_array('x', (B, H, W, 32), dace.float64)
    sdfg.add_array('y', (B, out_channels, H, W), dace.float64)
    state = sdfg.add_state(is_start_block=True)
    node = TensorTranspose('_TensorTranspose', axes=[0, 3, 1, 2])
    state.add_node(node)
    state.add_edge(state.add_read('x'), None, node, '_inp_tensor', dace.Memlet('x[0:B, 0:H, 0:W, 0:32]'))
    state.add_edge(node, '_out_tensor', state.add_write('y'), None, dace.Memlet('y[0:B, 64:96, 0:H, 0:W]'))
    return sdfg, state, node


def test_validate_returns_the_extents_the_memlets_move():
    """``validate`` hands the expansions the extents, and every expansion sizes its call from them.
    They are the subset's, so a slice write is 32 channels wide even though its container is 256."""
    sdfg, state, node = make_slice_transpose_sdfg(256)
    inp_tensor, out_tensor, inp_shape, out_shape = node.validate(sdfg, state)

    assert list(out_tensor.shape) == [B, 256, H, W], 'the descriptor is still the container'
    assert symbolic.shapes_equal(out_shape, [B, 32, H, W]), f'out extents {out_shape} are not the subset'
    assert symbolic.shapes_equal(inp_shape, [B, H, W, 32]), f'inp extents {inp_shape} are not the subset'
    # Extents from the memlet, strides from the container -- that pair is what makes the slice
    # addressable at all, so the container's layout must survive the narrowing.
    assert symbolic.shapes_equal(out_tensor.strides, [256 * H * W, H * W, W, 1]), f'{out_tensor.strides}'


def test_extents_are_compared_by_name_not_by_structure():
    """The comparator half. One extent reaches the two sides through different rewrites and arrives
    as two sympy instances that raw ``!=`` calls unequal -- here the frontend's int64 ``H``/``W``
    against a consumer's re-minted int32 -- and the check must not reject shapes that match."""
    h32 = dace.symbol('H', dtype=dace.int32)
    w32 = dace.symbol('W', dtype=dace.int32)
    # Non-vacuity: without a spelling difference this test proves nothing about the comparator.
    assert [H, W] != [h32, w32], 'the two spellings must be structurally unequal for this to bite'

    sdfg = dace.SDFG('respelled_transpose')
    sdfg.add_array('x', (B, H, W, 32), dace.float64)
    sdfg.add_array('y', (B, 256, h32, w32), dace.float64)
    state = sdfg.add_state(is_start_block=True)
    node = TensorTranspose('_TensorTranspose', axes=[0, 3, 1, 2])
    state.add_node(node)
    state.add_edge(state.add_read('x'), None, node, '_inp_tensor', dace.Memlet('x[0:B, 0:H, 0:W, 0:32]'))
    state.add_edge(node, '_out_tensor', state.add_write('y'), None, dace.Memlet('y[0:B, 64:96, 0:H, 0:W]'))

    sdfg.validate()


def test_a_genuine_extent_mismatch_is_still_rejected():
    """The check still has to fail: the subset is what it reads, so a subset that is not the
    permuted input is the error the message names."""
    sdfg, state, node = make_slice_transpose_sdfg(256)
    for edge in state.out_edges(node):
        edge.data.subset = dace.subsets.Range.from_string('0:B, 64:80, 0:H, 0:W')
    with pytest.raises(ValueError, match='does not match the output shape'):
        node.validate(sdfg, state)


@dace.program
def dot_into_slice(a: dace.float64[B, K, P], b: dace.float64[P, N], y: dace.float64[B, K, 2 * N]):
    y[:, :, 0:N] = np.tensordot(a, b, axes=([2], [0]))


def test_contraction_into_a_slice_of_a_larger_container():
    """``TensorDot`` is the same defect in the sibling node: the dot-product shape was compared
    against the output CONTAINER, and every expansion sized its call from the descriptor."""
    sdfg = dot_into_slice.to_sdfg(simplify=True)
    sdfg.validate()
    assert [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TensorDot)], 'the contraction must be a libnode'

    b, k, p, n = 2, 3, 4, 5
    rng = np.random.default_rng(20260901)
    a, bb = rng.random((b, k, p)), rng.random((p, n))
    y = np.full((b, k, 2 * n), -1.0)
    sdfg(a=a.copy(), b=bb.copy(), y=y, B=b, K=k, P=p, N=n)
    assert np.allclose(y[:, :, 0:n], np.tensordot(a, bb, axes=([2], [0]))), 'the slice got the wrong data'
    assert np.all(y[:, :, n:] == -1.0), 'the contraction wrote outside its subset'


def test_contraction_extents_are_compared_by_name_not_by_structure():
    """The comparator half for ``TensorDot``: the contracting modes and the output shape are matched
    by name, so an extent respelled on one side does not reject a contraction that agrees."""
    p32 = dace.symbol('P', dtype=dace.int32)
    n32 = dace.symbol('N', dtype=dace.int32)
    assert [P, N] != [p32, n32], 'the two spellings must be structurally unequal for this to bite'

    sdfg = dace.SDFG('respelled_dot')
    sdfg.add_array('a', (B, K, P), dace.float64)
    sdfg.add_array('b', (p32, n32), dace.float64)
    sdfg.add_array('y', (B, K, N), dace.float64)
    state = sdfg.add_state(is_start_block=True)
    node = TensorDot('_TensorDot_', left_axes=[2], right_axes=[0])
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_left_tensor', dace.Memlet('a[0:B, 0:K, 0:P]'))
    state.add_edge(state.add_read('b'), None, node, '_right_tensor', dace.Memlet('b[0:P, 0:N]'))
    state.add_edge(node, '_out_tensor', state.add_write('y'), None, dace.Memlet('y[0:B, 0:K, 0:N]'))

    sdfg.validate()


if __name__ == '__main__':
    test_transpose_into_a_slice_of_a_larger_container()
    test_transpose_out_of_a_slice_of_a_larger_container()
    test_validate_returns_the_extents_the_memlets_move()
    test_extents_are_compared_by_name_not_by_structure()
    test_a_genuine_extent_mismatch_is_still_rejected()
    test_contraction_into_a_slice_of_a_larger_container()
    test_contraction_extents_are_compared_by_name_not_by_structure()
