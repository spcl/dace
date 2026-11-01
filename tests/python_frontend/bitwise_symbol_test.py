# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Bitwise operators on symbols (e.g., map parameters) in programs: bit fields packed into one integer; integer
expressions of symbols and constants."""

import math

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


@dace.program
def first_part(out: dace.int64[N]):
    for i in range(N):
        if i < N // 4 * 2:  # A symbolic condition with an integer division
            out[i] = 1


def test_integer_division_in_symbolic_condition():
    out = np.zeros(10, dtype=np.int64)
    first_part(out=out, N=10)
    assert np.array_equal(out, (np.arange(10) < 10 // 4 * 2).astype(np.int64))


KIND = 7 << 40  # A tag above 32 bits


@dace.program
def tagged(values: dace.int32[N], out: dace.int64[N]):
    for i in dace.map[0:N]:
        v = values[i]
        out[i] = KIND | v << 4


def test_wide_constant_with_narrow_scalar():
    """A Python integer that does not fit the other operand's type widens the result (it is not cast down)."""
    values = np.arange(1, 9, dtype=np.int32)
    out = np.zeros(8, dtype=np.int64)
    tagged(values=values, out=out, N=8)
    assert np.array_equal(out, KIND | values.astype(np.int64) << 4)


@dace.program
def gelu_tanh(x: dace.float32[N], y: dace.float32[N]):
    for i in dace.map[0:N]:
        y[i] = math.tanh(x[i])


def test_math_tanh():
    x = np.linspace(-3, 3, 16, dtype=np.float32)
    y = np.zeros_like(x)
    gelu_tanh(x=x, y=y, N=16)
    assert np.allclose(y, np.tanh(x), atol=1e-6)


if __name__ == "__main__":
    test_bitwise_operators_on_symbols()
    test_bitwise_symbol_in_tasklet()
    test_bitwise_functions_fold_and_print_as_operators()
    test_integer_division_in_symbolic_condition()
    test_wide_constant_with_narrow_scalar()
    test_math_tanh()
