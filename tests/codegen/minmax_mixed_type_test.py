# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that the C++ ``Min``/``Max`` templates reject mixing floating-point and
integer arguments.

Background: a symbolic ``Min(1, x)`` (where ``x`` is a double) used to compile to
a template whose return type was deduced from the first argument, silently
truncating the double result to an integer. We now make that a hard compile-time
error instead of producing wrong numbers, while still allowing all-integer and
all-floating-point ``Min``/``Max``.
"""

import numpy as np
import pytest

import dace
from dace.codegen.exceptions import CompilationError
from dace.sdfg import SDFG, InterstateEdge

# The static_assert messages, so a compile error is known to come from the guard
_MIXED_TYPE_ERROR = "{}: integer and floating-point arguments mixed"


def _build_minmax_interstate_sdfg(rhs: str) -> SDFG:
    """Build an SDFG whose only work is a ``zanew = <rhs>`` interstate assignment.

    The double symbol ``zanew`` is seeded to ``0.375`` (a cloud fraction in
    ``[0, 1]``), the assignment ``rhs`` is applied on the next edge, and the
    result is written to the single-element output array.

    :param rhs: the right-hand side of the ``zanew`` interstate assignment.
    :return: the constructed SDFG.
    """
    sdfg = SDFG("minmax_interstate")
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_symbol("zanew", dace.float64)
    init = sdfg.add_state("init", is_start_block=True)
    middle = sdfg.add_state("middle")
    final = sdfg.add_state("final")
    sdfg.add_edge(init, middle, InterstateEdge(assignments={"zanew": "0.375"}))
    sdfg.add_edge(middle, final, InterstateEdge(assignments={"zanew": rhs}))
    write = final.add_write("out")
    tasklet = final.add_tasklet("write_zanew", {}, {"o"}, "o = zanew")
    final.add_edge(tasklet, "o", write, None, dace.Memlet("out[0]"))
    return sdfg


def test_min_mixed_int_double_is_compile_error():
    """A ``Min`` mixing an integer literal with a double must fail to compile."""
    sdfg = _build_minmax_interstate_sdfg("Min(1, zanew)")
    out = np.zeros(1, dtype=np.float64)
    with pytest.raises(CompilationError) as excinfo:
        sdfg(out=out)
    assert _MIXED_TYPE_ERROR.format("Min") in str(excinfo.value)


def test_max_mixed_int_double_is_compile_error():
    """A ``Max`` mixing an integer literal with a double must fail to compile."""
    sdfg = _build_minmax_interstate_sdfg("Max(0, zanew)")
    out = np.zeros(1, dtype=np.float64)
    with pytest.raises(CompilationError) as excinfo:
        sdfg(out=out)
    assert _MIXED_TYPE_ERROR.format("Max") in str(excinfo.value)


if __name__ == "__main__":
    test_min_mixed_int_double_is_compile_error()
    test_max_mixed_int_double_is_compile_error()
