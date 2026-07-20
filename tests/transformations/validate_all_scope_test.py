# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``validate_all`` checks what the transformation that just applied could have broken.

A full ``sdfg.validate()`` after EVERY match costs O(whole SDFG) per application, which on a
many-state SDFG makes ``validate_all`` more expensive than the transformations it is
watching. A `SingleStateTransformation` rewrites one state, so only that state is checked;
anything that can move interstate edges, symbols or descriptors gets the full check, and the
end-of-pass ``validate`` stays the whole-SDFG net.

What must not regress is the *catching*: a transformation that corrupts the state it touched
still has to be caught, at the match that caused it.
"""

import warnings

import pytest

import dace
from dace import SDFG, InterstateEdge, Memlet, dtypes
from dace.sdfg import nodes as dnodes
from dace.sdfg.validation import InvalidSDFGError
from dace.transformation import transformation as xf
from dace.transformation.dataflow import MapFusionVertical
from dace.transformation.passes.pattern_matching import PatternMatchAndApply

M = 8


def two_map_state(sdfg: SDFG, index: int) -> dace.SDFGState:
    """One state holding ``T = A*2; B = T+1`` -- a vertically fusable Map pair."""
    for name in (f'A{index}', f'B{index}'):
        sdfg.add_array(name, [M], dtypes.float64)
    sdfg.add_transient(f'T{index}', [M], dtypes.float64)
    state = sdfg.add_state(f's{index}', is_start_block=(index == 0))

    ar, tw, bw = state.add_access(f'A{index}'), state.add_access(f'T{index}'), state.add_access(f'B{index}')
    entry1, exit1 = state.add_map(f'prod{index}', {'i': f'0:{M}'})
    t1 = state.add_tasklet('x2', {'_in'}, {'_out'}, '_out = _in * 2.0')
    state.add_memlet_path(ar, entry1, t1, dst_conn='_in', memlet=Memlet(f'A{index}[i]'))
    state.add_memlet_path(t1, exit1, tw, src_conn='_out', memlet=Memlet(f'T{index}[i]'))
    entry2, exit2 = state.add_map(f'cons{index}', {'i': f'0:{M}'})
    t2 = state.add_tasklet('p1', {'_in'}, {'_out'}, '_out = _in + 1.0')
    state.add_memlet_path(tw, entry2, t2, dst_conn='_in', memlet=Memlet(f'T{index}[i]'))
    state.add_memlet_path(t2, exit2, bw, src_conn='_out', memlet=Memlet(f'B{index}[i]'))
    return state


def chain_of_states(num_states: int) -> SDFG:
    sdfg = SDFG(f'validate_scope_chain{num_states}')
    previous = None
    for index in range(num_states):
        state = two_map_state(sdfg, index)
        if previous is not None:
            sdfg.add_edge(previous, state, InterstateEdge())
        previous = state
    sdfg.validate()
    return sdfg


def test_validate_all_still_fuses_a_valid_sdfg():
    """The scoped check must not reject anything a full validate accepts."""
    sdfg = chain_of_states(4)
    applied = sdfg.apply_transformations_repeated(MapFusionVertical, validate=True, validate_all=True)
    assert applied == 4, f'expected one fusion per state, got {applied}'
    sdfg.validate()


def test_validate_all_catches_a_transformation_that_breaks_its_own_state():
    """A `SingleStateTransformation` that corrupts the state it matched must still be caught
    at that match, not silently deferred -- that is the whole point of ``validate_all``."""

    class BreakTheState(xf.SingleStateTransformation):
        """Applies to any MapEntry and leaves a dangling connector behind."""
        map_entry = xf.PatternNode(dnodes.MapEntry)

        @classmethod
        def expressions(cls):
            return [dace.sdfg.utils.node_path_graph(cls.map_entry)]

        def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
            return True

        def apply(self, graph, sdfg):
            # An in-connector with no edge feeding it: state-local corruption, exactly the
            # class of damage a dataflow transformation causes.
            self.map_entry.add_in_connector('IN_dangling')

    sdfg = chain_of_states(3)
    with pytest.raises(InvalidSDFGError, match='Validation failed after applying'):
        sdfg.apply_transformations_repeated(BreakTheState, validate=False, validate_all=True)


def test_scoped_check_does_not_warn_about_transients_initialized_elsewhere():
    """A state-local check has no cross-state context: a transient another state initialized
    looks uninitialized. The scoped path seeds every descriptor as initialized, so a valid
    SDFG produces no spurious diagnostic (and no raise for a Reference)."""
    sdfg = SDFG('cross_state_transient')
    sdfg.add_array('A', [M], dtypes.float64)
    sdfg.add_array('B', [M], dtypes.float64)
    sdfg.add_transient('T', [M], dtypes.float64)

    producer = sdfg.add_state('produce', is_start_block=True)
    ar, tw = producer.add_access('A'), producer.add_access('T')
    entry, exit_ = producer.add_map('prod', {'i': f'0:{M}'})
    t = producer.add_tasklet('x2', {'_in'}, {'_out'}, '_out = _in * 2.0')
    producer.add_memlet_path(ar, entry, t, dst_conn='_in', memlet=Memlet('A[i]'))
    producer.add_memlet_path(t, exit_, tw, src_conn='_out', memlet=Memlet('T[i]'))

    # The consumer reads ``T`` without writing it: only the producer state initializes it.
    consumer = sdfg.add_state('consume')
    sdfg.add_edge(producer, consumer, InterstateEdge())
    tr, bw = consumer.add_access('T'), consumer.add_access('B')
    entry2, exit2 = consumer.add_map('cons', {'i': f'0:{M}'})
    t2 = consumer.add_tasklet('p1', {'_in'}, {'_out'}, '_out = _in + 1.0')
    consumer.add_memlet_path(tr, entry2, t2, dst_conn='_in', memlet=Memlet('T[i]'))
    consumer.add_memlet_path(t2, exit2, bw, src_conn='_out', memlet=Memlet('B[i]'))
    sdfg.validate()

    match = MapFusionVertical()
    match.state_id = sdfg.node_id(consumer)
    checker = PatternMatchAndApply([MapFusionVertical])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        checker.validate_after_match(match, consumer, sdfg)
    spurious = [str(w.message) for w in caught if 'uninitialized transient' in str(w.message)]
    assert not spurious, f'scoped validation warned about a transient another state initializes: {spurious}'


if __name__ == '__main__':
    test_validate_all_still_fuses_a_valid_sdfg()
    test_validate_all_catches_a_transformation_that_breaks_its_own_state()
    test_scoped_check_does_not_warn_about_transients_initialized_elsewhere()
