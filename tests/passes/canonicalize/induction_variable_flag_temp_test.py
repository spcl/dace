# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A counter update that reads a comparison temporary is not a constant-step counter."""

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.induction_variable_substitution import sink_chained_iv_update


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_a_counter_bumped_by_a_comparison_flag_is_left_alone():
    """histogram_equalization: ``t = x < y; k = k + t`` substituted the comparison into the step and raised."""
    sdfg = dace.SDFG("flag_temp_counter")
    for name, dtype in (("k", dace.int64), ("t", dace.bool_), ("x", dace.int64), ("y", dace.int64)):
        sdfg.add_symbol(name, dtype)
    loop = LoopRegion("l", "i < N", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state("first", is_start_block=True)
    second = loop.add_state("second")
    third = loop.add_state("third")
    loop.add_edge(first, second, dace.InterstateEdge(assignments={"t": "x < y"}))
    loop.add_edge(second, third, dace.InterstateEdge(assignments={"k": "k + t"}))
    assert sink_chained_iv_update(loop, sdfg) is False
    assert loop.edges_between(second, third)[0].data.assignments == {"k": "k + t"}


if __name__ == "__main__":
    test_a_counter_bumped_by_a_comparison_flag_is_left_alone()
