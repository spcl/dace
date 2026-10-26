# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that ``replace_keys`` reaches interstate edges inside nested control flow regions."""

import dace
from dace.sdfg.state import ControlFlowRegion


def test_replace_keys_reaches_nested_region():
    sdfg = dace.SDFG("replace_keys_nested")
    sdfg.add_symbol("cur", dace.int64)

    region = ControlFlowRegion("region", sdfg=sdfg)
    sdfg.add_node(region)
    inner_a = region.add_state("inner_a", is_start_block=True)
    inner_b = region.add_state("inner_b")
    region.add_edge(inner_a, inner_b, dace.InterstateEdge(assignments={"cur": "cur + 1"}))

    start = sdfg.add_state("start", is_start_block=True)
    sdfg.add_edge(start, region, dace.InterstateEdge(assignments={"cur": "0"}))

    sdfg.replace_dict({"cur": "cur_new"}, replace_keys=True)

    # The nested assignment's target must be renamed alongside its uses, or the symbol splits in two.
    assert "cur_new" in list(sdfg.edges())[0].data.assignments
    assert "cur_new" in list(region.edges())[0].data.assignments
    assert "cur" not in list(region.edges())[0].data.assignments


def test_replacing_a_symbol_by_an_expression_keeps_its_assignment_target():
    sdfg = dace.SDFG("replace_keys_expression")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("M", dace.int64)
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state("second")
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={"N": "M", "k": "N + 1"}))

    sdfg.replace_dict({"N": "M + 1"}, replace_keys=True)

    assignments = sdfg.edges()[0].data.assignments
    # An expression is not a name: it replaces the use of N, while N stays the target of its assignment
    assert set(assignments) == {"N", "k"}
    assert dace.symbolic.pystr_to_symbolic(assignments["k"]) == dace.symbolic.pystr_to_symbolic("M + 2")


if __name__ == "__main__":
    test_replace_keys_reaches_nested_region()
    test_replacing_a_symbol_by_an_expression_keeps_its_assignment_target()
