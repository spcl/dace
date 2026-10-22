# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A slice update spelled as an assignment, ``A[S] = A[S] + e``, reaches the SDFG as ``A[S] -> map -> t -> A[S]``;
``EliminateCopyBack`` has the map write ``A[S]`` in place when each iteration reads only the element it writes."""

import copy

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.eliminate_copy_back import EliminateCopyBack

N, M, K = dace.symbol("N"), dace.symbol("M"), dace.symbol("K")


@dace.program
def deposit(A: dace.float64[M + 2, 2], w: dace.float64[M], c: dace.float64[3]):
    for p in range(M):
        A[p : p + 3, 1] = A[p : p + 3, 1] + w[p] * c


@dace.program
def shifted(A: dace.float64[N], c: dace.float64[3]):
    A[K : K + 3] = A[K + 1 : K + 4] + c


def transients_copied_back(sdfg: dace.SDFG) -> int:
    return sum(
        1
        for st in sdfg.states()
        for n in st.data_nodes()
        if sdfg.arrays[n.data].transient
        and st.out_degree(n) == 1
        and isinstance(st.out_edges(n)[0].dst, nodes.AccessNode)
        and st.out_edges(n)[0].dst.data == "A"
    )


def test_a_slice_update_writes_its_target_in_place():
    sdfg = deposit.to_sdfg(simplify=True)
    assert transients_copied_back(sdfg) == 1
    assert EliminateCopyBack().apply_pass(sdfg, {}) == 1
    assert transients_copied_back(sdfg) == 0
    sdfg.validate()
    m = 7
    rng = np.random.default_rng(0)
    w, c = rng.random(m), rng.random(3)
    A = rng.random((m + 2, 2))
    expected = A.copy()
    for p in range(m):
        expected[p : p + 3, 1] += w[p] * c
    sdfg(A=A, w=w, c=c, M=m)
    assert np.allclose(A, expected)


def test_a_read_of_the_neighbouring_element_keeps_the_copy_back():
    """Iteration ``i`` reads ``A[K + i + 1]``, which iteration ``i + 1`` writes: in place it would read the new value."""
    sdfg = shifted.to_sdfg(simplify=True)
    before = copy.deepcopy(sdfg)
    assert EliminateCopyBack().apply_pass(sdfg, {}) is None
    assert transients_copied_back(sdfg) == transients_copied_back(before)


if __name__ == "__main__":
    test_a_slice_update_writes_its_target_in_place()
    test_a_read_of_the_neighbouring_element_keeps_the_copy_back()
