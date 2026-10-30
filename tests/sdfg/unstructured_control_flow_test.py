# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Which blocks make an SDFG's control flow unstructured."""

import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion


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


def test_a_branch_in_a_conditional_block_is_structured() -> None:
    sut = branch_in_a_conditional_block_sdfg()

    found = list(sdutil.unstructured_control_flow(sut, recursive=True))

    assert found == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
