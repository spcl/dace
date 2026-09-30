# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that :class:`FindAccessNodes` classifies by DATA, not by degree.

An empty memlet is an ordering edge -- it transfers nothing -- so an access node reached only by
such edges is neither read nor written. Counting raw degree made one an apparent write, and
``ScalarWriteShadowScopes`` then reported it both as a read of the real write and as a write scope
of its own; ``ScalarFission`` renamed the node twice and versioned a later read onto a container
that nothing ever writes.
"""
import dace
from dace.transformation.passes.analysis import FindAccessNodes


def ordering_edge_sdfg() -> dace.SDFG:
    """One real write to ``s``, then a second ``s`` node reached only by an ordering edge."""
    sdfg = dace.SDFG('ordering_edge_access')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_scalar('s', dace.float64, transient=True)

    state = sdfg.add_state('main', is_start_block=True)
    a_read = state.add_read('A')
    written = state.add_access('s')
    ordered = state.add_access('s')
    out_write = state.add_write('out')

    producer = state.add_tasklet('producer', {'i'}, {'o'}, 'o = i * 2.0')
    consumer = state.add_tasklet('consumer', {'i'}, {'o'}, 'o = i + 1.0')

    state.add_edge(a_read, None, producer, 'i', dace.Memlet('A[0]'))
    state.add_edge(producer, 'o', written, None, dace.Memlet('s[0]'))
    state.add_edge(written, None, ordered, None, dace.Memlet())
    state.add_edge(ordered, None, consumer, 'i', dace.Memlet('s[0]'))
    state.add_edge(consumer, 'o', out_write, None, dace.Memlet('out[0]'))
    return sdfg


def test_ordering_edge_is_neither_a_read_nor_a_write():
    sdfg = ordering_edge_sdfg()
    sdfg.validate()
    state = sdfg.states()[0]
    written, ordered = [n for n in state.data_nodes() if n.data == 's']

    reads, writes = FindAccessNodes().apply_pass(sdfg, {})[sdfg.cfg_id]['s'][state]

    assert ordered not in writes, 'a node reached only by an empty memlet is not written'
    assert ordered in reads, 'the ordering-only node still READS -- its outgoing memlet carries data'
    assert written in writes, 'the real write must stay a write'
    assert written not in reads, 'an empty OUTgoing memlet does not make the writer a read'


if __name__ == '__main__':
    test_ordering_edge_is_neither_a_read_nor_a_write()
