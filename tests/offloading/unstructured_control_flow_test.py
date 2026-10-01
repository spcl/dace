# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Which blocks make an SDFG's control flow unstructured."""
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion


def branch_on_interstate_edges_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("branch_on_interstate_edges")
    sdfg.add_symbol("n", dace.int64)
    start = sdfg.add_state("start", is_start_block=True)
    guard = sdfg.add_state("guard")
    big, end = sdfg.add_state("big"), sdfg.add_state("end")
    sdfg.add_edge(start, guard, dace.InterstateEdge())
    sdfg.add_edge(guard, big, dace.InterstateEdge(condition="n > 0"))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition="n <= 0"))
    sdfg.add_edge(big, end, dace.InterstateEdge())
    return sdfg


def branch_in_a_conditional_block_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("branch_in_a_conditional_block")
    sdfg.add_symbol("n", dace.int64)
    start = sdfg.add_state("start", is_start_block=True)
    branch = ConditionalBlock("branch")
    sdfg.add_node(branch)
    body = ControlFlowRegion("big", sdfg=sdfg)
    body.add_state("big", is_start_block=True)
    branch.add_branch(CodeBlock("n > 0"), body)
    sdfg.add_edge(start, branch, dace.InterstateEdge())
    return sdfg


def test_a_state_leaving_through_conditional_edges_is_unstructured() -> None:
    sut = branch_on_interstate_edges_sdfg()

    found = [block.label for block in sdutil.unstructured_control_flow(sut)]

    assert found == ["guard"]


def test_a_branch_in_a_conditional_block_is_structured() -> None:
    sut = branch_in_a_conditional_block_sdfg()

    found = list(sdutil.unstructured_control_flow(sut, recursive=True))

    assert found == []


def test_requiring_structured_control_flow_names_the_consumer_and_the_offending_block() -> None:
    sut = branch_on_interstate_edges_sdfg()

    with pytest.raises(NotImplementedError, match=r"MyPass requires structured control flow.*\['guard'\]"):
        sdutil.require_structured_control_flow(sut, "MyPass")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
