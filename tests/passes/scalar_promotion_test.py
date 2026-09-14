# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PromoteScalarOutputsToArrays``: written signature scalars become addressable, read-only ones do not.

The pass exists because a non-transient ``Scalar`` has no addressable spelling in either signature it
can appear in -- ``T name`` on the entry point, ``T &name`` on a nested SDFG connector -- so a written
one either loses the result or fails to render as C. What each test pins is therefore the SIGNATURE
and the descriptor, not only that the pass returned a count.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.scalar_promotion import PromoteScalarOutputsToArrays


def scalar_output_sdfg(name: str = 'scalar_out') -> dace.SDFG:
    """``out = a[0] * 3`` with ``out`` a written non-transient scalar."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_scalar('out', dace.float64, transient=False)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('scale', {'i'}, {'o'}, 'o = i * 3.0')
    state.add_edge(state.add_access('a'), None, tasklet, 'i', dace.Memlet('a[0]'))
    state.add_edge(tasklet, 'o', state.add_access('out'), None, dace.Memlet('out[0]'))
    return sdfg


def test_written_signature_scalar_becomes_a_pointer_parameter():
    sdfg = scalar_output_sdfg()
    assert 'double out' in sdfg.signature(), 'the premise is gone: a signature Scalar is no longer by value'

    assert PromoteScalarOutputsToArrays().apply_pass(sdfg, {}) == 1
    descriptor = sdfg.arrays['out']
    assert isinstance(descriptor, dace.data.Array) and descriptor.shape == (1, ), descriptor
    assert 'double * __restrict__ out' in sdfg.signature()
    sdfg.validate()


def test_read_only_signature_scalar_is_left_by_value():
    """By value is correct for a read-only scalar; promoting it would churn every such signature."""
    sdfg = dace.SDFG('scalar_in')
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_array('b', [20], dace.float64)
    sdfg.add_scalar('alpha', dace.float64, transient=False)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('scale', {'i', 's'}, {'o'}, 'o = i * s')
    state.add_edge(state.add_access('a'), None, tasklet, 'i', dace.Memlet('a[0]'))
    state.add_edge(state.add_access('alpha'), None, tasklet, 's', dace.Memlet('alpha[0]'))
    state.add_edge(tasklet, 'o', state.add_access('b'), None, dace.Memlet('b[0]'))

    assert PromoteScalarOutputsToArrays().apply_pass(sdfg, {}) is None
    assert isinstance(sdfg.arrays['alpha'], dace.data.Scalar)
    assert 'double alpha' in sdfg.signature()


def test_transient_scalar_is_left_alone():
    """A transient scalar is a local, not a signature entry -- it is already addressable."""
    sdfg = scalar_output_sdfg('scalar_transient')
    sdfg.arrays['out'].transient = True

    assert PromoteScalarOutputsToArrays().apply_pass(sdfg, {}) is None
    assert isinstance(sdfg.arrays['out'], dace.data.Scalar)


def test_nested_scalar_connector_is_promoted_with_its_parent():
    """The connector's INNER descriptor must be promoted too, or the nested call still binds ``T &``."""
    sdfg = dace.SDFG('nested_scalar_out')
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_scalar('out', dace.float64, transient=False)
    state = sdfg.add_state()
    nested = scalar_output_sdfg('inner')
    node = state.add_nested_sdfg(nested, {'a'}, {'out'})
    state.add_edge(state.add_access('a'), None, node, 'a', dace.Memlet('a[0:20]'))
    state.add_edge(node, 'out', state.add_access('out'), None, dace.Memlet('out[0]'))

    # One promotion, not two: promoting the outer descriptor CASCADES into the connector it binds,
    # so the inner scalar is never reached as an independent candidate.
    assert PromoteScalarOutputsToArrays().apply_pass(sdfg, {}) == 1
    assert isinstance(sdfg.arrays['out'], dace.data.Array)
    assert isinstance(nested.arrays['out'], dace.data.Array), ('the inner descriptor is still a Scalar, so the '
                                                               'nested call would bind it by reference')
    sdfg.validate()


def test_promoted_sdfg_matches_a_hand_written_array_output():
    """Promotion must land on exactly the SDFG one would have written with an array output.

    The unpromoted form cannot be the reference here: its ``out`` is by value, which is the defect
    the pass exists to remove, so running it would compare against a result that never comes back.
    """
    a = np.random.rand(20)

    reference = dace.SDFG('promote_ref')
    reference.add_array('a', [20], dace.float64)
    reference.add_array('out', [1], dace.float64)
    state = reference.add_state()
    tasklet = state.add_tasklet('scale', {'i'}, {'o'}, 'o = i * 3.0')
    state.add_edge(state.add_access('a'), None, tasklet, 'i', dace.Memlet('a[0]'))
    state.add_edge(tasklet, 'o', state.add_access('out'), None, dace.Memlet('out[0]'))
    ref_out = np.zeros(1)
    reference(a=a, out=ref_out)

    promoted = scalar_output_sdfg('promote_run')
    PromoteScalarOutputsToArrays().apply_pass(promoted, {})
    assert promoted.signature() == reference.signature(), (f'{promoted.signature()!r} is not the signature a hand-'
                                                           f'written array output gives: {reference.signature()!r}')
    out = np.zeros(1)
    promoted(a=a, out=out)

    assert np.allclose(out, ref_out)
    assert np.allclose(out[0], a[0] * 3.0)


N = dace.symbol('N')


def map_over_written_scalar_sdfg(name: str) -> dace.SDFG:
    """``width = min(width, N); for ii in 0:width: B[ii] = A[ii]``, the range read through a connector named ``width``.

    The connector spelling is the one ``LoopToMap`` gives a scalar a loop bound reads (cegterg's ``nvecx``).
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_scalar('width', dace.int64)
    clamp = sdfg.add_state('clamp', is_start_block=True)
    tasklet = clamp.add_tasklet('clamp', {'w_in'}, {'w_out'}, 'w_out = min(w_in, N)')
    clamp.add_edge(clamp.add_read('width'), None, tasklet, 'w_in', dace.Memlet('width'))
    clamp.add_edge(tasklet, 'w_out', clamp.add_write('width'), None, dace.Memlet('width'))
    copy = sdfg.add_state_after(clamp, 'copy')
    entry, exit_node = copy.add_map('copy', dict(ii='0:width'))
    entry.add_in_connector('width')
    copy.add_edge(copy.add_read('width'), None, entry, 'width', dace.Memlet('width'))
    body = copy.add_tasklet('copy', {'a'}, {'b'}, 'b = a')
    copy.add_memlet_path(copy.add_read('A'), entry, body, dst_conn='a', memlet=dace.Memlet('A[ii]'))
    copy.add_memlet_path(body, exit_node, copy.add_write('B'), src_conn='b', memlet=dace.Memlet('B[ii]'))
    sdfg.validate()
    return sdfg


def test_a_map_range_over_a_promoted_scalar_reads_its_value():
    """Code generation defines no local for a range input named like its container, so the bound compared the loop
    index against the promoted array's pointer and the unit did not compile."""
    sdfg = map_over_written_scalar_sdfg('promote_map_range')
    assert PromoteScalarOutputsToArrays().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    a = np.arange(1.0, 9.0)
    b = np.zeros(8)
    width = np.array([5], dtype=np.int64)
    sdfg(A=a, B=b, width=width, N=8)
    assert np.array_equal(b, [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0]), b


if __name__ == '__main__':
    pytest.main([__file__])
