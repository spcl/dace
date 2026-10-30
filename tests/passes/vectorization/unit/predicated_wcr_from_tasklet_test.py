# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SameWriteSetIfElseToITECFG`` if-converts a predicated WCR write a tasklet produces (spgemm_hash's
``if c: C[0] += 2 * A[0]``) by staging its value through a scalar; it raised NotImplementedError."""

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import SameWriteSetIfElseToITECFG


def test_predicated_wcr_write_from_a_tasklet():
    sdfg = dace.SDFG("predicated_wcr_from_tasklet")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_symbol("c", dace.int64)
    head = sdfg.add_state("head", is_start_block=True)
    cb = ConditionalBlock("guard", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb)
    sdfg.add_edge(head, cb, dace.InterstateEdge())
    arm = ControlFlowRegion("arm", sdfg=sdfg, parent=cb)
    body = arm.add_state("acc", is_start_block=True)
    tasklet = body.add_tasklet("twice", {"a"}, {"o"}, "o = 2 * a")
    body.add_edge(body.add_access("A"), None, tasklet, "a", dace.Memlet("A[0]"))
    body.add_edge(tasklet, "o", body.add_access("C"), None, dace.Memlet("C[0]", wcr="lambda x, y: x + y"))
    cb.add_branch(CodeBlock("c > 0"), arm)

    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})

    sdfg.validate()
    assert not any(isinstance(block, ConditionalBlock) for block in sdfg.all_control_flow_blocks())
    for c, expected in ((1, 7.0), (0, 1.0)):
        out = np.ones(1)
        sdfg(A=np.full(1, 3.0), C=out, c=c)
        assert out[0] == expected, (c, out)


if __name__ == "__main__":
    test_predicated_wcr_write_from_a_tasklet()
    print("ok")
