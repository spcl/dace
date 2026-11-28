# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``Subscript`` must stay a ``Subscript`` through serialization, and ``str`` must spell it ``a[i]``.

Inside a larger expression sympy's string printer printed a ``Subscript`` as a call, ``Max(j, Subscript(off, 0))``.
A pass that writes such an expression back as text -- an interstate assignment, as canonicalization does for QE
``vexx_bp_k_gpu``'s ``jblock_end = min(..., (ijt - 1) * jblock + all_start(wegrp))`` -- then left a call to an
undefined ``Subscript`` in the code: the CPU and CUDA builds failed with "'Subscript' was not declared in this
scope", and the CPF printer could not type the argument.

A serialized ``Subscript`` came back as an undefined function of that name, which prints the same way and counts
its container as a free symbol: every SDFG that went through JSON (a deep copy, a dump) carried the second form.
"""

import numpy as np

import dace
from dace import symbolic


def test_a_nested_subscript_prints_as_an_index():
    expr = symbolic.pystr_to_symbolic("i - Max(j, 2 * k + off[m - 1])")
    text = str(expr)
    assert "Subscript" not in text, text
    assert "off[m - 1]" in text, text
    assert symbolic.pystr_to_symbolic(text) == expr


def test_a_subscript_survives_serialization():
    expr = symbolic.pystr_to_symbolic("i - Max(j, off[m - 1])")
    loaded = symbolic.deserialize_symbolic(symbolic.serialize_symbolic(expr))
    assert loaded == expr, (loaded, expr)
    assert any(isinstance(atom, symbolic.Subscript) for atom in loaded.atoms(symbolic.sympy.Function)), loaded
    assert {str(s) for s in loaded.free_symbols} == {"i", "j", "m"}, loaded.free_symbols
    assert "Subscript" not in str(loaded), str(loaded)


def test_an_assignment_written_from_str_compiles_and_reads_the_array():
    """``k = Max(0, off[0]) + 1`` on an interstate edge, the text made with ``str``; ``B[0] = k``."""
    sdfg = dace.SDFG("symbolic_subscript_str")
    sdfg.add_array("off", [1], dace.int64)
    sdfg.add_array("B", [1], dace.int64)
    sdfg.add_symbol("k", dace.int64)
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state("second")
    value = symbolic.pystr_to_symbolic("Max(0, off[0]) + 1")
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={"k": str(value)}))
    tasklet = second.add_tasklet("store", {}, {"__out"}, "__out = k")
    second.add_edge(tasklet, "__out", second.add_write("B"), None, dace.Memlet("B[0]"))
    b = np.zeros(1, dtype=np.int64)
    sdfg(off=np.array([4], dtype=np.int64), B=b)
    assert b[0] == 5


if __name__ == "__main__":
    test_a_nested_subscript_prints_as_an_index()
    test_a_subscript_survives_serialization()
    test_an_assignment_written_from_str_compiles_and_reads_the_array()
