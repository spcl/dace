# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the subset helpers shared by the redundant-array transformations."""
import numpy as np

import dace
from dace import subsets
from dace.transformation.dataflow import RedundantSecondArray
from dace.transformation.dataflow.redundant_array import (compose_and_push_back, find_dims_to_pop, find_dims_to_pop2,
                                                          pop_dims)


def test_find_dims_to_pop_is_descending() -> None:
    """``find_dims_to_pop()`` reports the dimensions to remove from last to first."""
    assert find_dims_to_pop([1, 3, 1, 4, 1], [3, 4]) == [4, 2, 0]


def test_find_dims_to_pop2_is_ascending() -> None:
    """``find_dims_to_pop2()`` reports them the other way round, from first to last."""
    assert find_dims_to_pop2([1, 3, 1, 4, 1], [3, 4]) == [0, 2, 4]


def test_pop_dims_on_indices() -> None:
    """``pop_dims()`` must also handle ``subsets.Indices`` and report the removed dimensions."""
    subset = subsets.Indices([7, 8, 9, 10])
    new_subset, popped = pop_dims(subset, [2, 0])

    assert isinstance(new_subset, subsets.Indices)
    assert new_subset.indices == [8, 10]
    assert [rb for (rb, _, _), _ in popped] == [9, 7]
    assert subset.indices == [7, 8, 9, 10], "`pop_dims()` must not modify its argument."


def test_pop_dims_indexes_the_original_subset() -> None:
    """``dims`` indexes the subset as it is passed in, so the order they come in is irrelevant."""
    subset = subsets.Range([(0, 0, 1), (0, 4, 1), (0, 0, 1)])
    ascending, _ = pop_dims(subset, [0, 2])
    descending, _ = pop_dims(subset, [2, 0])

    assert ascending.ranges == [(0, 4, 1)]
    assert descending.ranges == [(0, 4, 1)]


def test_compose_and_push_back_inverts_pop_dims() -> None:
    """The two helpers are exact inverses, so composing with the identity restores the subset."""
    subset = subsets.Range([(1, 1, 1), (0, 4, 1), (2, 2, 1)])
    for dims in ([0, 2], [2, 0]):
        reduced, popped = pop_dims(subset, dims)
        restored = compose_and_push_back(reduced, subsets.Range([(0, 4, 1)]), dims, popped)
        assert restored.ranges == subset.ranges


def test_redundant_second_array_pops_non_adjacent_dims() -> None:
    """Squeezing non-adjacent size-1 dimensions must not pop shifted indices."""
    sdfg = dace.SDFG("redundant_second_array_pop_non_adjacent_dims")
    sdfg.add_array("A", [1, 5, 1], dace.float64)
    sdfg.add_array("C", [5], dace.float64)
    sdfg.add_transient("B", [5], dace.float64)

    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", dict(i="0:5"))
    entry.add_scope_connectors("b")
    exit_.add_scope_connectors("c")
    read_b = state.add_access("B")
    tasklet = state.add_tasklet("add_one", {"inp"}, {"out"}, "out = inp + 1.0")
    state.add_edge(state.add_read("A"), None, read_b, None, dace.Memlet("A[0:1, 0:5, 0:1] -> [0:5]"))
    state.add_edge(read_b, None, entry, "IN_b", dace.Memlet("B[0:5]"))
    state.add_edge(entry, "OUT_b", tasklet, "inp", dace.Memlet("B[i]"))
    state.add_edge(tasklet, "out", exit_, "IN_c", dace.Memlet("C[i]"))
    state.add_edge(exit_, "OUT_c", state.add_write("C"), None, dace.Memlet("C[0:5]"))

    assert sdfg.apply_transformations_repeated(RedundantSecondArray) == 1

    A = np.arange(5, dtype=np.float64).reshape(1, 5, 1)
    C = np.zeros(5, dtype=np.float64)
    sdfg(A=A.copy(), C=C)
    assert np.allclose(C, A.reshape(5) + 1.0)


if __name__ == "__main__":
    test_find_dims_to_pop_is_descending()
    test_find_dims_to_pop2_is_ascending()
    test_pop_dims_on_indices()
    test_pop_dims_indexes_the_original_subset()
    test_compose_and_push_back_inverts_pop_dims()
    test_redundant_second_array_pops_non_adjacent_dims()
