# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF folds a ``float64`` scalar into its reader without losing the conversion the store into it did."""

import numpy as np
import pytest

import dace
from dace.codegen.cpf import render
from tests.codegen.cpf.conftest import build_standalone, call_standalone


def widened_quotient_sdfg(name: str) -> dace.SDFG:
    """``t1 = n; t2 = m; out = t1 / t2`` with int64 ``n``, ``m`` and float64 ``t1``, ``t2``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("n", [1], dace.int64)
    sdfg.add_array("m", [1], dace.int64)
    sdfg.add_array("out", [1], dace.float64)
    state = sdfg.add_state()
    quotient = state.add_tasklet("quotient", {"a": None, "b": None}, {"o": None}, "o = a / b")
    for source, scalar, conn in (("n", "t1", "a"), ("m", "t2", "b")):
        sdfg.add_transient(scalar, [1], dace.float64)
        widen = state.add_tasklet(f"widen_{source}", {"inp": None}, {"o": None}, "o = inp")
        state.add_edge(state.add_read(source), None, widen, "inp", dace.Memlet(f"{source}[0]"))
        access = state.add_access(scalar)
        state.add_edge(widen, "o", access, None, dace.Memlet(f"{scalar}[0]"))
        state.add_edge(access, None, quotient, conn, dace.Memlet(f"{scalar}[0]"))
    state.add_edge(quotient, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


@pytest.mark.parametrize("language", ["c++", "c"])
def test_a_folded_integer_into_a_float64_scalar_divides_in_double(language):
    name = f"cpf_fold_widened_{'cpp' if language == 'c++' else 'c'}"
    result = render(widened_quotient_sdfg(name), language=language)
    args = {"n": np.array([7], dtype=np.int64), "m": np.array([2], dtype=np.int64), "out": np.zeros(1)}
    call_standalone(build_standalone(result.code, name, language=language), result.sdfg, args)
    np.testing.assert_array_equal(args["out"], np.array([7.0]) / np.array([2.0]))


if __name__ == "__main__":
    test_a_folded_integer_into_a_float64_scalar_divides_in_double("c++")
    test_a_folded_integer_into_a_float64_scalar_divides_in_double("c")
