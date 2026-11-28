# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import re

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion


def _make_sdfg_with_zero_sized_an_to_an_memlet() -> tuple[dace.SDFG, dace.SDFGState]:
    """Generates an SDFG that performs a copy that has a zero size."""
    sdfg = dace.SDFG("zero_size_copy_sdfg")
    state = sdfg.add_state(is_start_block=True)

    for name in "AB":
        sdfg.add_array(
            name=name,
            shape=(20, 20),
            dtype=dace.float64,
            transient=True,
        )

    state.add_nedge(
        state.add_access("A"),
        state.add_access("B"),
        dace.Memlet("A[2:17, 2:2] -> [2:18, 3:3]"),
    )

    return sdfg, state


def test_an_to_an_memlet_with_zero_size():
    sdfg, state = _make_sdfg_with_zero_sized_an_to_an_memlet()
    assert sdfg.number_of_nodes() == 1
    assert state.number_of_nodes() == 2

    sdfg.validate()

    # This zero sized copy should be considered valid.
    assert sdfg.is_valid()

    # The SDFG should be a no ops.
    ref = {
        "A": np.array(np.random.rand(20, 20), copy=True, order="C", dtype=np.float64),
        "B": np.array(np.random.rand(20, 20), copy=True, order="C", dtype=np.float64),
    }
    res = {k: np.array(v, order="C", copy=True) for k, v in ref.items()}

    csdfg = sdfg.compile()
    assert csdfg.sdfg.number_of_nodes() == 1
    assert next(iter(csdfg.sdfg.states())).number_of_nodes() == 2
    csdfg(**res)

    assert all(np.all(ref[k] == res[k]) for k in ref.keys())


def test_an_to_an_memlet_with_negative_size():
    """Tests if an AccessNode to AccessNode connection leads to an invalid SDFG."""
    sdfg = dace.SDFG("an_to_an_memlet_with_negative_size")
    state = sdfg.add_state(is_start_block=True)

    for name in "AB":
        sdfg.add_array(
            name=name,
            shape=(20, 20),
            dtype=dace.float64,
            transient=True,
        )

    state.add_nedge(
        state.add_access("A"),
        state.add_access("B"),
        dace.Memlet("A[2:17, 13:2] -> [2:18, 14:3]"),
    )

    with pytest.raises(
        expected_exception=dace.sdfg.InvalidSDFGEdgeError,
        match=re.escape(
            "`subset` of an AccessNode to AccessNode Memlet contains a negative size; the size was [15, -11]"
        ),
    ):
        sdfg.validate()


def test_copy_size_mismatch_at_a_code_node():
    """
    A memlet path that starts at a code node has no container to view, which the dimensionality check has to
    report as a mismatch rather than fail on.
    """
    sdfg = dace.SDFG("copy_size_mismatch_at_code_node")
    sdfg.add_array("A", [5], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("produce", {}, {"out"}, "out = 1")
    state.add_edge(tasklet, "out", state.add_write("A"), None, dace.Memlet(data="A", subset="0:2", other_subset="0:3"))

    with pytest.raises(dace.sdfg.InvalidSDFGEdgeError, match="Dimensionality mismatch"):
        sdfg.validate()


def test_strided_copy_with_a_symbolic_step_is_valid():
    """
    A strided slice with a symbolic step ``S`` (a max-pool tap, inside the loop over the window) holds
    ``ceiling((S * M + 1) / S)`` elements, which is ``M + 1`` once the loop over ``range(S)`` shows ``S`` is a positive
    integer; the check has to use the facts of the scope for that.
    """
    S, M = dace.symbol("S"), dace.symbol("M")
    sdfg = dace.SDFG("strided_copy_with_a_symbolic_step")
    sdfg.add_array("A", [S * M + 1], dace.float64)
    sdfg.add_array("B", [M + 1], dace.float64)
    loop = LoopRegion("window", "k < S", "k", "k = 0", "k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state(is_start_block=True)
    state.add_nedge(state.add_read("A"), state.add_write("B"), dace.Memlet("A[0:S*M + 1:S] -> [0:M + 1]"))

    sdfg.validate()


if __name__ == "__main__":
    test_an_to_an_memlet_with_zero_size()
    test_an_to_an_memlet_with_negative_size()
    test_copy_size_mismatch_at_a_code_node()
    test_strided_copy_with_a_symbolic_step_is_valid()
