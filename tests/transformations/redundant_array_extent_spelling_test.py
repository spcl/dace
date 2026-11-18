# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A redundant copy whose extent is spelled ``floor`` in the descriptor and ``int_floor`` in the memlet still folds."""

import dace
from dace.transformation.dataflow import RedundantArray, RedundantSecondArray

N = dace.symbol("N")
#: Python's ``//`` on an expression: sympy ``floor``, as a shape annotation spells it.
EXTENT = (N - 7) // 2 + 1
#: The parser's ``//``: ``int_floor``, as a memlet string spells it.
RANGE = "0:(N - 7) // 2 + 1"


def copy_sdfg(name: str, transient_first: bool) -> tuple[dace.SDFG, dace.SDFGState]:
    """``A -> map(2x) -> t -> B`` when ``transient_first``, else ``A -> t -> map(2x) -> B``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [EXTENT], dace.float64)
    sdfg.add_array("B", [EXTENT], dace.float64)
    sdfg.add_transient("t", [EXTENT], dace.float64)
    state = sdfg.add_state()
    a, t, b = state.add_read("A"), state.add_access("t"), state.add_write("B")
    src, dst = (a, t) if transient_first else (t, b)
    state.add_mapped_tasklet(
        "double",
        {"i": RANGE},
        {"x": dace.Memlet(f"{src.data}[i]")},
        "y = 2 * x",
        {"y": dace.Memlet(f"{dst.data}[i]")},
        external_edges=True,
        input_nodes={src.data: src},
        output_nodes={dst.data: dst},
    )
    if transient_first:
        state.add_nedge(t, b, dace.Memlet(f"t[{RANGE}] -> [{RANGE}]"))
    else:
        state.add_nedge(a, t, dace.Memlet(f"A[{RANGE}] -> [{RANGE}]"))
    sdfg.validate()
    return sdfg, state


def test_redundant_array_folds_a_copy_spelled_with_int_floor():
    sdfg, state = copy_sdfg("redundant_array_floor_spelling_first", transient_first=True)
    assert sdfg.apply_transformations(RedundantArray) == 1
    sdfg.validate()
    assert not any(n.data == "t" for n in state.data_nodes())


def test_redundant_second_array_folds_a_copy_spelled_with_int_floor():
    sdfg, state = copy_sdfg("redundant_array_floor_spelling_second", transient_first=False)
    assert sdfg.apply_transformations(RedundantSecondArray) == 1
    sdfg.validate()
    assert not any(n.data == "t" for n in state.data_nodes())


if __name__ == "__main__":
    test_redundant_array_folds_a_copy_spelled_with_int_floor()
    test_redundant_second_array_folds_a_copy_spelled_with_int_floor()
