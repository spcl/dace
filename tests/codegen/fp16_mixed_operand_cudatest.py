# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU tasklets mixing a ``dace.float16`` operand with a built-in one.

Each case is one mapped tasklet whose connectors carry the dtypes directly, with no casts in
between -- the shape a mixed-precision retyping of an fp64 program produces. Under CUDA every one
of them used to fail to compile, because ``dace::float16`` is ``__half`` and nvcc found the
built-in operator and ``__half``'s own equally good. All cases share one SDFG, so the test
compiles once; each result is checked against numpy, with the half promoted to the wider
operand's type.
"""

import numpy as np
import pytest

import dace
from dace import dtypes

pytestmark = pytest.mark.gpu

N = 8
#: Zero, negative zero, the smallest subnormal, and both signs.
HALVES = np.array([0.0, -0.0, 6e-8, -1.5, 0.5, 1.0, 2.5, 3.0], np.float16)
POSITIVE = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], np.float16)
FLOATS = np.array([-3.0, -1.0, 0.0, 0.1, 1.0, 2.5, 7.0, 1e3], np.float32)
DOUBLES = np.array([1e-20, 3.0] * (N // 2))
INTS = np.array([-2, -1, 0, 1, 2, 3, -3, 4], np.int32)

#: name -> (inputs, output dtype, tasklet code, numpy reference)
CASES = {
    "half_lt_double_literal": ({"a": HALVES}, np.bool_, "c = a < 1e-14", lambda a: a.astype(np.float64) < 1e-14),
    "float_div_half": ({"a": FLOATS, "b": POSITIVE}, np.float32, "c = a / b", lambda a, b: a / b.astype(np.float32)),
    # Half of the floats are not representable as the half they came from.
    "float_eq_half": (
        {"a": HALVES.astype(np.float32) + np.float32([0, 1e-3] * 4), "b": HALVES},
        np.bool_,
        "c = a == b",
        lambda a, b: a == b.astype(np.float32),
    ),
    "half_mul_int": ({"a": HALVES, "b": INTS}, np.float32, "c = a * b", lambda a, b: a.astype(np.float32) * b),
    "float_add_assign_half": (
        {"a": FLOATS, "b": HALVES},
        np.float32,
        "c = a\nc += b",
        lambda a, b: a + b.astype(np.float32),
    ),
    "abs_half": ({"a": HALVES}, np.float16, "c = abs(a)", lambda a: np.abs(a)),
    # A double below fp16's range must come back unchanged: min compares in double, not in half.
    "min_double_half": (
        {"a": DOUBLES, "b": POSITIVE},
        np.float64,
        "c = min(a, b)",
        lambda a, b: np.minimum(a, b.astype(np.float64)),
    ),
    "min_half_int": (
        {"a": HALVES, "b": INTS},
        np.float16,
        "c = min(a, b)",
        lambda a, b: np.minimum(a.astype(np.float32), b).astype(np.float16),
    ),
    "sin_half": (
        {"a": HALVES},
        np.float16,
        "c = math.sin(a)",
        lambda a: np.sin(a.astype(np.float32)).astype(np.float16),
    ),
    "fmod_half_double": (
        {"a": HALVES, "b": DOUBLES + 1.0},
        np.float64,
        "c = math.fmod(a, b)",
        lambda a, b: np.fmod(a.astype(np.float64), b),
    ),
    "pow_half_double": (
        {"a": POSITIVE, "b": np.array([-2.0, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])},
        np.float16,
        "c = a ** b",
        lambda a, b: (a.astype(np.float64) ** b).astype(np.float16),
    ),
    "pow_double_half": (
        {"a": DOUBLES + 1.0, "b": HALVES},
        np.float64,
        "c = a ** b",
        lambda a, b: a ** b.astype(np.float64),
    ),
    "pow_half_half": (
        {"a": POSITIVE, "b": HALVES},
        np.float16,
        "c = a ** b",
        lambda a, b: (a.astype(np.float32) ** b.astype(np.float32)).astype(np.float16),
    ),
    "pow_half_int": (
        {"a": POSITIVE, "b": INTS},
        np.float16,
        "c = a ** b",
        lambda a, b: (a.astype(np.float32) ** b).astype(np.float16),
    ),
}


def test_mixed_half_tasklets():
    sdfg = dace.SDFG("fp16_mixed_operands")
    state = sdfg.add_state()
    args = {}
    for name, (inputs, out_dtype, code, _) in CASES.items():
        arrays = {**inputs, "c": np.zeros(N, out_dtype)}
        for conn, value in arrays.items():
            sdfg.add_array(f"{name}_{conn}", [N], dtypes.dtype_to_typeclass(value.dtype.type))
            args[f"{name}_{conn}"] = value
        tasklet, _, _ = state.add_mapped_tasklet(
            name,
            {"i": f"0:{N}"},
            {conn: dace.Memlet(f"{name}_{conn}[i]") for conn in inputs},
            code,
            {"c": dace.Memlet(f"{name}_c[i]")},
            external_edges=True,
        )
        # Typed connectors, no casts in between: the half meets the other operand directly.
        for conn, value in arrays.items():
            connectors = tasklet.out_connectors if conn == "c" else tasklet.in_connectors
            connectors[conn] = dtypes.dtype_to_typeclass(value.dtype.type)
    sdfg.apply_gpu_transformations()
    sdfg(**args)

    for name, (inputs, _, _, reference) in CASES.items():
        np.testing.assert_allclose(
            args[f"{name}_c"].astype(np.float64), reference(**inputs).astype(np.float64), rtol=1e-3, err_msg=name
        )
