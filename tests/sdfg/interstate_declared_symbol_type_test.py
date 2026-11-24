# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An interstate assignment whose expression inference cannot type keeps the type its symbol is declared with."""

import dace


def test_an_untyped_assignment_keeps_the_declared_symbol_type():
    """jfnk_bratu: ``f0 = sqrt(__inl2_s)`` typed ``f0`` as None, and the CPF C rewrite failed on ``None.to_string()``."""
    sdfg = dace.SDFG("declared_symbol_type")
    sdfg.add_symbol("f0", dace.float64)
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state("second")
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={"f0": "sqrt(untyped_name)"}))
    tasklet = second.add_tasklet("t", {}, {}, "")
    assert second.symbols_defined_at(tasklet)["f0"] == dace.float64


if __name__ == "__main__":
    test_an_untyped_assignment_keeps_the_declared_symbol_type()
