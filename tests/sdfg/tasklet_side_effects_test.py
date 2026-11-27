# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Whether a tasklet's calls can have side effects, asked by state fusion of every tasklet."""

import dace


# Reproducer: QE h_psi's simplify crashed in StateFusionExtended with "TypeError: Invalid AST node type:
# <class 'ast.BinOp'>" on a tasklet ``_out = ((_in_a + (1j * 0.0)) - _in_b).real()``
def test_a_method_of_a_computed_value_has_no_side_effects():
    sdfg = dace.SDFG("method_of_a_value")
    sdfg.add_array("A", [1], dace.complex128)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("t", {"a"}, {"b"}, "b = ((a + (1j * 0.0)) - a).conjugate().real")
    state.add_edge(state.add_read("A"), None, tasklet, "a", dace.Memlet("A[0]"))
    state.add_edge(tasklet, "b", state.add_write("B"), None, dace.Memlet("B[0]"))
    assert not tasklet.has_side_effects(sdfg)


def test_a_call_of_a_symbol_may_have_side_effects():
    sdfg = dace.SDFG("callback_symbol")
    sdfg.add_symbol("cb", dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("t", {}, {}, "cb()")
    assert tasklet.has_side_effects(sdfg)


if __name__ == "__main__":
    test_a_method_of_a_computed_value_has_no_side_effects()
    test_a_call_of_a_symbol_may_have_side_effects()
