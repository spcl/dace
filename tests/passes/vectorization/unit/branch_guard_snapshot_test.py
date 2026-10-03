# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``BranchNormalization`` must not re-test a guard its own arms have mutated.

A two-arm ``ConditionalBlock`` whose arms cannot share one ITE state is serialized into two
sequential single-arm blocks -- ``if c: A`` then ``if not c: B`` -- and each half lifts the guard
into its own state as an array read. That is value-preserving only while no arm writes data ``c``
reads. TSVC s2710 breaks it::

    if a[i] > b[i]:
        a[i] = a[i] + b[i] * d[i]     # <- mutates an operand of the guard
        if ...: c[i] = ...  else: c[i] = ...
    else:
        b[i] = a[i] + e[i] * e[i]
        if ...: c[i] = ...  else: c[i] = ...

The nested ``if`` in each arm makes the arms asymmetric, so they serialize. The negated half then
re-read the ALREADY-UPDATED ``a`` and evaluated ``a[i] <= b[i]`` against it, so every lane whose
update flipped the comparison executed BOTH arms: the else-arm's store to ``b`` landed on if-arm
lanes. The fix snapshots the guard once, into a bool transient in a state that dominates both
halves, and hands both halves a read of that snapshot.
"""
import json

import numpy as np
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.parallelize import parallelize
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import SameWriteSetIfElseToITECFG
from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')


@dace.program
def both_arms_nested(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N],
                     e: dace.float64[N]):
    for i in range(N):
        if a[i] > b[i]:
            a[i] = a[i] + b[i] * d[i]
            if d[i] > 1.0:
                c[i] = c[i] + d[i] * d[i]
            else:
                c[i] = d[i] * e[i] + 1.0
        else:
            b[i] = a[i] + e[i] * e[i]
            if d[i] > 0.0:
                c[i] = a[i] + d[i] * d[i]
            else:
                c[i] = c[i] + e[i] * e[i]


def reference(a, b, c, d, e):
    for i in range(len(a)):
        if a[i] > b[i]:
            a[i] = a[i] + b[i] * d[i]
            c[i] = (c[i] + d[i] * d[i]) if d[i] > 1.0 else (d[i] * e[i] + 1.0)
        else:
            b[i] = a[i] + e[i] * e[i]
            c[i] = (a[i] + d[i] * d[i]) if d[i] > 0.0 else (c[i] + e[i] * e[i])
    return a, b, c


def branch_lowered(tag):
    """SDFG after the vectorizer's branch-lowering front only (no tiling)."""
    sdfg = both_arms_nested.to_sdfg(simplify=True)
    sdfg.name = tag
    parallelize(sdfg, validate=True, validate_all=False)
    for cleaner in (SameWriteSetIfElseToITECFG(), BranchNormalization()):
        cleaner.apply_pass(sdfg, {})
    return sdfg


def test_guard_is_evaluated_once():
    """Only ONE tasklet may read the guard's operands; the serialized halves share its result.

    Two such tasklets means the guard was re-derived from the arrays after an arm had already
    stored into ``a`` -- exactly the stale re-test that miscompiled s2710.
    """
    sdfg = branch_lowered('branch_guard_struct')
    guard_producers = []
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if not isinstance(node, nd.Tasklet):
                    continue
                sources = {ed.src.data for ed in state.in_edges(node) if isinstance(ed.src, nd.AccessNode)}
                if {'a', 'b'} <= sources:
                    guard_producers.append(f'{state.label}/{node.label}')
    assert len(guard_producers) == 1, f'guard re-derived from a/b in several states: {guard_producers}'


def test_value_preserving():
    n = 61
    rng = np.random.default_rng(1)
    a, b = rng.random(n) - 0.5, rng.random(n) - 0.5
    c, d, e = rng.random(n), rng.random(n), rng.random(n)
    want_a, want_b, want_c = reference(a.copy(), b.copy(), c.copy(), d, e)

    sdfg = both_arms_nested.to_sdfg(simplify=True)
    sdfg.name = 'branch_guard_value'
    parallelize(sdfg, validate=True, validate_all=False)
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})
    sdfg.validate()

    got_a, got_b, got_c = a.copy(), b.copy(), c.copy()
    sdfg.compile()(a=got_a, b=got_b, c=got_c, d=d.copy(), e=e.copy(), N=n)
    assert np.allclose(got_a, want_a, rtol=1e-12, atol=1e-12)
    assert np.allclose(got_b, want_b, rtol=1e-12, atol=1e-12)
    assert np.allclose(got_c, want_c, rtol=1e-12, atol=1e-12)


def write_one(state: dace.SDFGState, name: str) -> None:
    tasklet = state.add_tasklet(f'set_{name}', {}, {'o'}, 'o = 1.0')
    state.add_edge(tasklet, 'o', state.add_write(name), None, dace.Memlet(f'{name}[0]'))


def arm_decrements_guard_symbol(name: str) -> dace.SDFG:
    """``if k > 0: {x = 1; k = k - 1; y = 1} else: {z = 1}`` with ``k`` an argument."""
    sdfg = dace.SDFG(name)
    for array in ('x', 'y', 'z'):
        sdfg.add_array(array, [1], dace.float64)
    sdfg.add_symbol('k', dace.int64)
    init = sdfg.add_state('init', is_start_block=True)
    cb = ConditionalBlock('cb', sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb)
    then_arm = ControlFlowRegion('then_arm', sdfg=sdfg)
    first = then_arm.add_state('first', is_start_block=True)
    write_one(first, 'x')
    second = then_arm.add_state('second')
    write_one(second, 'y')
    then_arm.add_edge(first, second, dace.InterstateEdge(assignments={'k': 'k - 1'}))
    else_arm = ControlFlowRegion('else_arm', sdfg=sdfg)
    write_one(else_arm.add_state('only', is_start_block=True), 'z')
    cb.add_branch(CodeBlock('k > 0'), then_arm)
    cb.add_branch(None, else_arm)
    sdfg.add_edge(init, cb, dace.InterstateEdge())
    return sdfg


def test_if_else_whose_arm_rebinds_a_guard_symbol_is_left_whole():
    """Serializing would re-test ``k > 0`` after the if-arm decremented ``k``, running the else-arm for ``k == 1``."""
    sdfg = arm_decrements_guard_symbol('arm_rebinds_guard_symbol')
    before = json.dumps(sdfg.to_json(), sort_keys=True)

    result = BranchNormalization().apply_pass(sdfg, {})

    assert result is None
    assert json.dumps(sdfg.to_json(), sort_keys=True) == before
    csdfg = sdfg.compile()
    rows = []
    for k in (0, 1, 2):
        x, y, z = np.zeros(1), np.zeros(1), np.zeros(1)
        csdfg(x=x, y=y, z=z, k=k)
        rows.append((x[0], y[0], z[0]))
    assert rows == [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
