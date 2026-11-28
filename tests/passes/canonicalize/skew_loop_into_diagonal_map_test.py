# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A sequential loop over a map whose dependences all lie on one diagonal becomes one map over the diagonals."""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.skew_loop_into_diagonal_map import SkewLoopIntoDiagonalMap

N = dace.symbol("N")


def diagonal_recurrence(name: str, reads: tuple[str, ...]) -> dace.SDFG:
    """``for i in 2..N-2: map j in 2..N-3: a[i, j] = sum(a[<read>]) + b[i, j]``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N, N], dace.float64)
    sdfg.add_array("b", [N, N], dace.float64)
    loop = LoopRegion("rows", "i < N - 1", "i", "i = 2", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state("body", is_start_block=True)
    inputs = {f"r{k}": dace.Memlet(f"a[{read}]") for k, read in enumerate(reads)}
    inputs["q"] = dace.Memlet("b[i, j]")
    state.add_mapped_tasklet(
        "step",
        {"j": "2:N-2"},
        inputs,
        "o = " + " + ".join(inputs),
        {"o": dace.Memlet("a[i, j]")},
        external_edges=True,
    )
    return sdfg


def reference(a: np.ndarray, b: np.ndarray, reads: tuple[tuple[int, int], ...]) -> np.ndarray:
    a = a.copy()
    n = a.shape[0]
    for i in range(2, n - 1):
        for j in range(2, n - 2):
            a[i, j] = sum(a[i + di, j + dj] for di, dj in reads) + b[i, j]
    return a


CASES = {
    "s119": (("i - 1, j - 1",), ((-1, -1),)),
    "anti": (("i - 1, j + 1",), ((-1, 1),)),
    "two": (("i - 1, j - 1", "i - 2, j - 2"), ((-1, -1), (-2, -2))),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_diagonal_recurrence_runs_as_one_map_over_diagonals(case: str):
    reads, offsets = CASES[case]
    sdfg = diagonal_recurrence(f"skew_diag_{case}", reads)
    assert SkewLoopIntoDiagonalMap().apply_pass(sdfg, {}) == 1
    assert not any(isinstance(block, LoopRegion) for block in sdfg.nodes()), "the loop must move into the map"
    maps = [n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert len(maps) == 1 and maps[0].map.params == ["j_diag"]
    sdfg.validate()
    rng = np.random.default_rng(3)
    a, b = rng.random((37, 37)), rng.random((37, 37))
    expected = reference(a, b, offsets)
    sdfg(a=a, b=b, N=37)
    np.testing.assert_allclose(a, expected, rtol=1e-13)


@pytest.mark.parametrize("reads", [("i - 1, j - 1", "i - 1, j + 1"), ("i, j - 1",), ("i - 1, j",)])
def test_dependences_off_one_diagonal_keep_the_loop(reads: tuple[str, ...]):
    """Two slopes cross diagonals, a same-row read is a dependence inside the map, slope 0 is plain interchange."""
    sdfg = diagonal_recurrence("skew_diag_refused", reads)
    assert SkewLoopIntoDiagonalMap().apply_pass(sdfg, {}) is None
    assert any(isinstance(block, LoopRegion) for block in sdfg.nodes())


if __name__ == "__main__":
    for name in sorted(CASES):
        test_a_diagonal_recurrence_runs_as_one_map_over_diagonals(name)
    test_dependences_off_one_diagonal_keep_the_loop(("i - 1, j - 1", "i - 1, j + 1"))
    test_dependences_off_one_diagonal_keep_the_loop(("i, j - 1",))
    test_dependences_off_one_diagonal_keep_the_loop(("i - 1, j",))
