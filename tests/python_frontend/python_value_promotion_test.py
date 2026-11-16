# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A name bound to a Python value -- a Python number, a symbolic expression, or ``math`` and arithmetic over those
-- is a Python float or int in NumPy, which NEP 50 weighs as a weak operand: it takes the array's dtype. The
frontend stores such a value in a double Scalar; typed as a strong ``float64`` it widened every float32 operation it
met (``scale = 1.0 / math.sqrt(head_dim)`` in an attention kernel), where NumPy stays in float32. A NumPy scalar
(``np.sqrt(head_dim)``) stays strong, as in NumPy.
"""

import math

import numpy as np

import dace

N = dace.symbol("N", dtype=dace.int64, positive=True)
D = dace.symbol("D", dtype=dace.int64, positive=True)


def array_dtypes(program) -> set:
    """The dtypes of every array (not scalar) the program's SDFG holds."""
    sdfg = program.to_sdfg(simplify=False)
    return {desc.dtype for desc in sdfg.arrays.values() if not isinstance(desc, dace.data.Scalar)}


@dace.program
def python_value_scale(x: dace.float32[N, D], y: dace.float32[N, D]):
    scale = 1.0 / math.sqrt(D)
    half = 0.5
    inverse = 1.0 / D
    y[:] = x * scale + x * half + x * inverse + np.tanh(math.sqrt(2.0 / math.pi) * x)


@dace.program
def numpy_scalar_scale(x: dace.float32[N, D], y: dace.float32[N, D]):
    y[:] = x / np.sqrt(D)


@dace.program
def math_sqrt_of_an_integer(x: dace.float64[N], y: dace.float64[N]):
    y[:] = x * math.sqrt(D)


@dace.program
def declared_scalar_offset(x: dace.float32[N], y: dace.float32[N]):
    i = dace.define_local_scalar(dtype=dace.int32)
    i = 0
    y[:] = x + i


@dace.program
def real_part_of_complex(z: dace.complex128[N], out: dace.float64[N]):
    out[:] = np.real(z) + np.imag(z)


def test_a_python_value_keeps_the_array_float32():
    assert array_dtypes(python_value_scale) == {dace.float32}


def test_a_numpy_scalar_stays_strong():
    assert dace.float64 in array_dtypes(numpy_scalar_scale)


def test_a_declared_scalar_keeps_its_dtype():
    """``dace.define_local_scalar(dace.int32)`` is a NumPy ``int32``, not a Python int, even once assigned ``0``:
    ``float32 + int32`` is ``float64``."""
    assert dace.float64 in array_dtypes(declared_scalar_offset)


def test_a_complex_argument_of_a_real_result_is_not_cast():
    """Only an integer or boolean argument computes in the floating-point result type: ``np.real`` of a complex
    is real, and casting the complex to ``double`` does not compile."""
    sdfg = real_part_of_complex.to_sdfg(simplify=False)
    codes = [node.code.as_string for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.Tasklet)]
    assert codes and not any("float64" in code for code in codes), codes


def test_math_sqrt_of_an_integer_is_not_truncated():
    """``math.sqrt(D)`` was typed like ``D`` (int64), so sqrt(8) became 2."""
    x = np.arange(1.0, 5.0)
    y = np.zeros(4)
    math_sqrt_of_an_integer(x=x, y=y, N=4, D=8)
    assert np.allclose(y, x * math.sqrt(8)), y


def test_a_python_value_computes_what_numpy_computes():
    rng = np.random.default_rng(0)
    x = rng.random((3, 6)).astype(np.float32)
    y = np.zeros((3, 6), np.float32)
    python_value_scale(x=x, y=y, N=3, D=6)
    expected = x * (1.0 / math.sqrt(6)) + x * 0.5 + x * (1.0 / 6) + np.tanh(math.sqrt(2.0 / math.pi) * x)
    assert expected.dtype == np.float32
    assert np.allclose(y, expected, rtol=1e-6), y


if __name__ == "__main__":
    test_a_python_value_keeps_the_array_float32()
    test_a_numpy_scalar_stays_strong()
    test_a_declared_scalar_keeps_its_dtype()
    test_a_complex_argument_of_a_real_result_is_not_cast()
    test_math_sqrt_of_an_integer_is_not_truncated()
    test_a_python_value_computes_what_numpy_computes()
