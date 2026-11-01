# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Bitwise operators on symbols (e.g., map parameters) in programs: bit fields packed into one integer."""

import numpy as np

import dace
from dace import symbolic
from dace.codegen.targets.cpp import sym2cpp

N = dace.symbol("N")


@dace.program
def bit_fields(out: dace.int64[N, 5]):
    for i in dace.map[0:N]:
        high = i >> 3 << 3
        middle = (i >> 1) & 3
        either = i | 1
        toggled = i ^ 1
        inverted = ~i
        out[i, 0] = high
        out[i, 1] = middle
        out[i, 2] = either
        out[i, 3] = toggled
        out[i, 4] = inverted


def test_bitwise_operators_on_symbols():
    out = np.zeros((20, 5), dtype=np.int64)
    bit_fields(out=out, N=20)
    i = np.arange(20)
    assert np.array_equal(out, np.stack([i >> 3 << 3, (i >> 1) & 3, i | 1, i ^ 1, ~i], axis=1))


@dace.program
def field_into_tasklet(out: dace.int64[N]):
    for i in dace.map[0:N]:
        field = (i >> 1) & 3
        with dace.tasklet:
            f << field  # A symbol by then: substituted into the tasklet's code
            o >> out[i]
            o = f * 10


def test_bitwise_symbol_in_tasklet():
    out = np.zeros(20, dtype=np.int64)
    field_into_tasklet(out=out, N=20)
    assert np.array_equal(out, ((np.arange(20) >> 1) & 3) * 10)


def test_bitwise_functions_fold_and_print_as_operators():
    t = dace.symbol("t", dace.int64)
    field = symbolic.pystr_to_symbolic("(t >> 20) & 1048575")
    assert sym2cpp(field) == "((t >> 20) & 1048575)"
    assert field.subs(t, 5 << 20 | 7) == 5
    assert symbolic.pystr_to_symbolic("~t").subs(t, 5) == -6


if __name__ == "__main__":
    test_bitwise_operators_on_symbols()
    test_bitwise_symbol_in_tasklet()
    test_bitwise_functions_fold_and_print_as_operators()
