# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A negative integer power of an integer symbol is a float, as Python's ``int ** -k`` is."""

import numpy as np

import dace
from dace import cpf_lowering, symbolic


def test_negative_power_of_an_integer_symbol_runs_as_a_float():
    # A nested SDFG's symbol mapping prints through ``symstr``: ``reciprocal((N * N))`` truncated to 0
    inner = dace.SDFG("negative_power_inner")
    inner.add_symbol("x", dace.float64)
    inner.add_array("o", [1], dace.float64)
    state = inner.add_state()
    tasklet = state.add_tasklet("read", {}, {"a"}, "a = x")
    state.add_edge(tasklet, "a", state.add_write("o"), None, dace.Memlet("o[0]"))
    sdfg = dace.SDFG("negative_power_of_an_integer_symbol")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("out", [1], dace.float64)
    state = sdfg.add_state()
    nested = state.add_nested_sdfg(inner, set(), {"o"}, symbol_mapping={"x": symbolic.pystr_to_symbolic("N ** -2")})
    state.add_edge(nested, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
    out = np.zeros(1)
    sdfg(out=out, N=2)
    assert out[0] == 0.25


def test_negative_power_of_an_integer_symbol_prints_in_floating_point():
    count = symbolic.symbol("neg_sym_pow_n", dace.int64)
    expected = {
        cpf_lowering.Dialect.RUNTIME: "(reciprocal(((double(neg_sym_pow_n)) * (double(neg_sym_pow_n)))))",
        cpf_lowering.Dialect.STANDALONE: "((1 / (((double(neg_sym_pow_n)) * (double(neg_sym_pow_n))))))",
        cpf_lowering.Dialect.STANDALONE_C: ("((1 / (((((double)(neg_sym_pow_n))) * (((double)(neg_sym_pow_n)))))))"),
    }
    for dialect, printed in expected.items():
        with cpf_lowering.dialect_scope(dialect):
            assert symbolic.symstr(count**-2, cpp_mode=True) == printed


if __name__ == "__main__":
    test_negative_power_of_an_integer_symbol_runs_as_a_float()
    test_negative_power_of_an_integer_symbol_prints_in_floating_point()
