# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ForwardStoreToLoad``: a value stored this iteration is read from a register, not from memory.

The pass exists so a same-iteration round trip through a declared output stops looking like a
loop-carried dependence. Everything below is about telling that shape apart from a real carry:
``a[i]`` written then read is a store-to-load, ``a[i]`` written and ``a[i-1]`` read is a carry,
and the two differ only in the SUBSET -- never in the names -- so that is what the pass reads.
"""
import os

os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import copy

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize.forward_store_to_load import ForwardStoreToLoad

N = dace.symbol('N')

TOL = dict(rtol=1e-12, atol=1e-12)


@dace.program
def same_iteration_round_trip(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N],
                              e: dace.float64[N]):
    """TSVC s323: ``a[i]`` is stored and reloaded at the SAME ``i``."""
    for i in range(1, N):
        a[i] = b[i - 1] + c[i] * d[i]
        b[i] = a[i] + c[i] * e[i]


@dace.program
def genuine_carry(a: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    """``a[i-1]`` is the PREVIOUS iteration's value -- a carry no forwarding may touch."""
    for i in range(1, N):
        a[i] = a[i - 1] + c[i] * d[i]


@dace.program
def read_before_write(a: dace.float64[N], b: dace.float64[N]):
    """``a[i]`` is read BEFORE the store, so the reader wants the incoming value."""
    for i in range(N):
        b[i] = a[i] * 2.0
        a[i] = 3.0


@dace.program
def conditional_store(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """The store is guarded and the read is not, so the two are not co-executed."""
    for i in range(N):
        if c[i] > 0.0:
            a[i] = c[i] + 1.0
        b[i] = a[i] * 2.0


def prepared(program) -> dace.SDFG:
    """``program`` as an SDFG with the loop bodies folded into one state each.

    ``ForwardStoreToLoad`` reads one state's dataflow, so the frontend's per-statement states
    have to be fused first -- which is exactly where the pipeline runs it.
    """
    from dace.transformation.passes.lift_preprocess import LiftPreprocess
    from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
    from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended

    sdfg = program.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([StateFusionExtended()]).apply_pass(sdfg, {})
    LiftPreprocess().apply_pass(sdfg, {})
    return sdfg


def forwarding_transients(sdfg: dace.SDFG) -> list[str]:
    return sorted(name for name in sdfg.arrays if '_fwd' in name)


def in_iteration_reads(sdfg: dace.SDFG, name: str) -> int:
    """Out-edges of the access nodes for ``name`` -- the reads the pass is meant to reroute."""
    return sum(state.out_degree(node) for state in sdfg.states() for node in state.data_nodes() if node.data == name)


def stores(sdfg: dace.SDFG, name: str) -> int:
    return sum(state.in_degree(node) for state in sdfg.states() for node in state.data_nodes() if node.data == name)


def test_same_iteration_store_to_load_is_forwarded():
    """The reader stops reading ``a`` and the store to ``a`` stays: ``a`` becomes write-only."""
    sdfg = prepared(same_iteration_round_trip)
    assert in_iteration_reads(sdfg, 'a') == 1, 'the fixture must start with the round trip in place'

    assert ForwardStoreToLoad().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert forwarding_transients(sdfg), 'the forwarded value needs a transient of its own'
    assert in_iteration_reads(sdfg, 'a') == 0, '`a` must no longer be read inside the body'
    assert stores(sdfg, 'a') == 1, 'the store to the declared output `a` must survive'


def test_a_genuine_carry_is_left_alone():
    """``a[i-1]`` reads what an EARLIER iteration wrote; forwarding it would fabricate a value."""
    sdfg = prepared(genuine_carry)
    before = copy.deepcopy(sdfg.to_json())

    assert ForwardStoreToLoad().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before, 'a pass that does not apply must leave the SDFG untouched'


def test_a_read_of_the_incoming_value_is_left_alone():
    """The read is upstream of the store, so no access node carries both -- nothing to forward."""
    sdfg = prepared(read_before_write)
    before = copy.deepcopy(sdfg.to_json())

    assert ForwardStoreToLoad().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before, 'a pass that does not apply must leave the SDFG untouched'


def test_a_conditional_store_is_left_alone():
    """A guarded write and an unguarded read cannot share an access node, which is what refuses it.

    The pass reads ONE state, where every node runs whenever the state does. A conditional write
    lands in its own block, so the read never sees a node carrying both -- the co-execution the
    forwarding needs is structural here, not a condition anyone has to compare.
    """
    sdfg = prepared(conditional_store)
    before = copy.deepcopy(sdfg.to_json())

    assert ForwardStoreToLoad().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before, 'a pass that does not apply must leave the SDFG untouched'


def test_an_aliasing_output_is_left_alone():
    """``may_alias`` means another argument can name the same memory, so the store may not stand in.

    The reasoning the pass rests on is that an access node's out-edges observe what its in-edge
    wrote. A write through an alias is a write to a DIFFERENT container, which the state orders
    against nothing -- so the value in the register is no longer provably the value in ``a[i]``.
    """
    sdfg = prepared(same_iteration_round_trip)
    sdfg.arrays['a'].may_alias = True
    before = copy.deepcopy(sdfg.to_json())

    assert ForwardStoreToLoad().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before, 'a pass that does not apply must leave the SDFG untouched'


@pytest.mark.parametrize('seed', [0, 7])
def test_forwarding_preserves_values(seed: int):
    """The rewrite is a value identity, checked before any later pass gets to reassociate."""
    rng = np.random.default_rng(seed)
    arrays = {name: rng.random(48) + 1.0 for name in 'abcde'}

    want = {name: arr.copy() for name, arr in arrays.items()}
    for i in range(1, 48):
        want['a'][i] = want['b'][i - 1] + want['c'][i] * want['d'][i]
        want['b'][i] = want['a'][i] + want['c'][i] * want['e'][i]

    sdfg = prepared(same_iteration_round_trip)
    assert ForwardStoreToLoad().apply_pass(sdfg, {}) == 1
    got = {name: arr.copy() for name, arr in arrays.items()}
    sdfg.compile()(**got, N=48)
    for name in arrays:
        assert np.allclose(want[name], got[name], **TOL), f'forwarding changed {name}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
