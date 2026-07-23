# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SplitStatements(split_maps=True)``: fission a straight-line map that writes several GLOBAL outputs
into one FLAT map per output, recomputing a shared local (never materializing it to a buffer).

The hard invariant is value-preservation; on top of it we assert the granularity: one map per global
output, and a shared local temp stays a SCALAR in each split map rather than being promoted to a
size-N array (the anti-pattern of the max-fission MapFission). ``split_maps`` is OFF by default, so the
canonicalization pipeline is byte-identical -- the default-constructed pass must leave the map alone.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.split_statements import SplitStatements

N = dace.symbol('N')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _materialized_buffers(sdfg):
    """Transients that are NOT scalars (shape != (1,)) -- a shared local wrongly promoted to a buffer."""
    return {n for n, d in sdfg.arrays.items() if d.transient and tuple(d.shape) != (1, )}


@dace.program
def _single_out(x: dace.float64[N], A: dace.float64[N]):
    for i in dace.map[0:N]:
        t = x[i] * 2.0
        A[i] = t + 1.0


@dace.program
def _two_out(x: dace.float64[N], A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        t = x[i] * 2.0
        A[i] = t + 1.0
        B[i] = t * t


@dace.program
def _three_out(x: dace.float64[N], A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        t = x[i] * 2.0
        A[i] = t + 1.0
        B[i] = t * t
        C[i] = t - 3.0


@dace.program
def _dependent(x: dace.float64[N], A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        A[i] = x[i] + 1.0
        B[i] = A[i] * 2.0  # reads the just-written value -- the frontend wraps this in a NestedSDFG


def _value_preserved(prog, arrays, n=16):
    raw = prog.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in arrays.items()}
    raw.compile()(**ref, N=n)

    cand = prog.to_sdfg(simplify=True)
    SplitStatements(split_maps=True).apply_pass(cand, {})
    cand.validate()
    got = {k: v.copy() for k, v in arrays.items()}
    cand.compile()(**got, N=n)
    for k in arrays:
        assert np.allclose(ref[k], got[k], equal_nan=True), f"{prog.name}: '{k}' diverged"
    return cand


def _rand(n=16, seed=0):
    return np.random.default_rng(seed).random(n)


def test_two_global_outputs_split_into_one_map_each():
    cand = _value_preserved(_two_out, {'x': _rand(), 'A': np.zeros(16), 'B': np.zeros(16)})
    assert _nmaps(cand) == 2
    # the shared 't' is recomputed in each map as a scalar, not promoted to a size-N buffer.
    assert not _materialized_buffers(cand)


def test_three_global_outputs_split_into_three_maps():
    cand = _value_preserved(_three_out, {'x': _rand(), 'A': np.zeros(16), 'B': np.zeros(16), 'C': np.zeros(16)})
    assert _nmaps(cand) == 3
    assert not _materialized_buffers(cand)


def test_a_dependent_read_after_write_splits_by_recomputing_the_local():
    """``B[i]=A[i]*2`` reads a value written earlier in the SAME map; the frontend wraps it in a plain
    NestedSDFG. The split inlines that body and recomputes the shared local in each map -- so the two
    maps are independent, no cross-map buffer, value preserved."""
    cand = _value_preserved(_dependent, {'x': _rand(), 'A': np.zeros(16), 'B': np.zeros(16)})
    assert _nmaps(cand) == 2
    assert not _materialized_buffers(cand)


def test_a_single_output_map_is_left_alone():
    cand = _value_preserved(_single_out, {'x': _rand(), 'A': np.zeros(16)})
    assert _nmaps(cand) == 1


def test_off_by_default_leaves_the_map_unsplit():
    sdfg = _two_out.to_sdfg(simplify=True)
    before = _nmaps(sdfg)
    SplitStatements().apply_pass(sdfg, {})  # split_maps defaults False -> canon path unchanged
    assert _nmaps(sdfg) == before == 1


if __name__ == '__main__':
    test_two_global_outputs_split_into_one_map_each()
    test_three_global_outputs_split_into_three_maps()
    test_a_single_output_map_is_left_alone()
    test_off_by_default_leaves_the_map_unsplit()
    print("ok")
