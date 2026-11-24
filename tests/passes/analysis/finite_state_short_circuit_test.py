# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The short-circuit ``if`` before a ``while`` loop is recognized on the region that holds both."""

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.analysis.finite_state import short_circuit_guard


def test_the_guard_is_read_off_the_enclosing_region_not_the_branch():
    """kmp, gromacs_nbnxm: naming the branch body ``region`` shadowed the enclosing region, and the edge lookup
    raised ``KeyError: ConditionalBlock``."""
    sdfg = dace.SDFG("short_circuit")
    sdfg.add_symbol("t", dace.bool_)
    sdfg.add_symbol("a", dace.bool_)
    outer = LoopRegion("outer", "i < N", "i", "i = 0", "i = i + 1")
    sdfg.add_node(outer, is_start_block=True)
    start = outer.add_state("start", is_start_block=True)
    cond = ConditionalBlock("cond")
    body = ControlFlowRegion("cond_body")
    body.add_state("set_t", is_start_block=True)
    cond.add_branch(CodeBlock("a"), body)
    outer.add_node(cond)
    inner = LoopRegion("inner", "t")
    inner.add_state("work", is_start_block=True)
    outer.add_node(inner)
    outer.add_edge(start, cond, dace.InterstateEdge(assignments={"t": "a"}))
    outer.add_edge(cond, inner, dace.InterstateEdge())
    assert short_circuit_guard(outer, cond, inner) == "a"


if __name__ == "__main__":
    test_the_guard_is_read_off_the_enclosing_region_not_the_branch()
