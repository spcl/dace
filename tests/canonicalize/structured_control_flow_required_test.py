# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalization stops on control flow that ControlFlowRaising leaves unstructured."""
import pytest

import dace
from dace.sdfg import utils as sdutil
from dace.transformation.passes.canonicalize.pipeline import canonicalize


def branch_into_a_sibling_arm_sdfg() -> dace.SDFG:
    """A three-way branch on ``n`` whose first arm can jump into the third arm, which no ConditionalBlock expresses."""
    sdfg = dace.SDFG("branch_into_a_sibling_arm")
    sdfg.add_symbol("n", dace.int64)
    start = sdfg.add_state("start", is_start_block=True)
    first, second, third, end = (sdfg.add_state(label) for label in ("first", "second", "third", "end"))
    sdfg.add_edge(start, first, dace.InterstateEdge(condition="n > 1"))
    sdfg.add_edge(start, second, dace.InterstateEdge(condition="n == 1"))
    sdfg.add_edge(start, third, dace.InterstateEdge(condition="n < 1"))
    sdfg.add_edge(first, third, dace.InterstateEdge(condition="n > 5"))
    sdfg.add_edge(first, end, dace.InterstateEdge(condition="n <= 5"))
    sdfg.add_edge(second, end, dace.InterstateEdge())
    sdfg.add_edge(third, end, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def plain_branch_sdfg() -> dace.SDFG:
    """A two-way branch on ``n`` written as conditional interstate edges that meet in ``end``."""
    sdfg = dace.SDFG("plain_branch")
    sdfg.add_symbol("n", dace.int64)
    start = sdfg.add_state("start", is_start_block=True)
    big, small, end = sdfg.add_state("big"), sdfg.add_state("small"), sdfg.add_state("end")
    sdfg.add_edge(start, big, dace.InterstateEdge(condition="n > 0"))
    sdfg.add_edge(start, small, dace.InterstateEdge(condition="n <= 0"))
    sdfg.add_edge(big, end, dace.InterstateEdge())
    sdfg.add_edge(small, end, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def test_canonicalization_refuses_a_branch_that_jumps_into_a_sibling_arm() -> None:
    sut = branch_into_a_sibling_arm_sdfg()

    with pytest.raises(NotImplementedError, match=r"Canonicalization requires structured control flow.*'start'"):
        canonicalize(sut)


def test_canonicalization_accepts_a_branch_on_interstate_edges_and_leaves_it_structured() -> None:
    sut = plain_branch_sdfg()

    canonicalize(sut)

    assert list(sdutil.unstructured_control_flow(sut, recursive=True)) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
