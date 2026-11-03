# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LowerInterstateConditionalAssignmentsToTasklets`` walks past a ``return`` block in a branch as it does a
``break`` (nussinov's early return raised "Unsupported node type")."""

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, ReturnBlock
from dace.transformation.passes.vectorization.lower_interstate_conditional_assignments_to_tasklets import (
    LowerInterstateConditionalAssignmentsToTasklets,
)


def test_return_block_in_a_branch_is_walked_past():
    """``if n == 2: return; B[0] = 1``."""
    sdfg = dace.SDFG("early_return_branch")
    sdfg.add_array("B", [1], dace.float64)
    sdfg.add_symbol("n", dace.int64)
    cb = ConditionalBlock("cb", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb, is_start_block=True)
    arm = ControlFlowRegion("arm", sdfg=sdfg, parent=cb)
    arm.add_node(ReturnBlock("leave"), is_start_block=True)
    cb.add_branch(CodeBlock("n == 2"), arm)
    write = sdfg.add_state("write")
    sdfg.add_edge(cb, write, dace.InterstateEdge())
    tasklet = write.add_tasklet("one", {}, {"b"}, "b = 1")
    write.add_edge(tasklet, "b", write.add_access("B"), None, dace.Memlet("B[0]"))

    LowerInterstateConditionalAssignmentsToTasklets().apply_pass(sdfg, {})

    sdfg.validate()
    for n, expected in ((2, 0), (3, 1)):
        b = np.zeros(1)
        sdfg(B=b, n=n)
        assert b[0] == expected, (n, b)


if __name__ == "__main__":
    test_return_block_in_a_branch_is_walked_past()
    print("ok")
