# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU tasklets mixing a ``dace.float16`` operand with a built-in one, end to end.

Each case is one mapped tasklet whose connectors carry the dtypes directly, with no casts in
between -- the shape a mixed-precision retyping of an fp64 program produces. Under CUDA every one
of them used to fail to compile, because ``dace::float16`` is ``__half`` and nvcc found the
built-in operator and ``__half``'s own equally good. Each case is compiled, run and checked
against numpy, with the half promoted to the wider operand's type.
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

#: name -> (inputs, output dtype, tasklet code, numpy reference)
CASES = {
    "half_lt_double_literal": (
        {
            "a": HALVES
        },
        np.bool_,
        "c = a < 1e-14",
        lambda a: a.astype(np.float64) < 1e-14,
    ),
    "float_div_half": (
        {
            "a": FLOATS,
            "b": POSITIVE
        },
        np.float32,
        "c = a / b",
        lambda a, b: a / b.astype(np.float32),
    ),
    # Half of the floats are not representable as the half they came from.
    "float_eq_half": (
        {
            "a": HALVES.astype(np.float32) + np.float32([0, 1e-3] * 4),
            "b": HALVES
        },
        np.bool_,
        "c = a == b",
        lambda a, b: a == b.astype(np.float32),
    ),
    "pow_half_double": (
        {
            "a": POSITIVE,
            "b": np.array([-2.0, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        },
        np.float16,
        "c = a ** b",
        lambda a, b: (a.astype(np.float64)**b).astype(np.float16),
    ),
    "pow_half_half": (
        {
            "a": POSITIVE,
            "b": HALVES
        },
        np.float16,
        "c = a ** b",
        lambda a, b: (a.astype(np.float32)**b.astype(np.float32)).astype(np.float16),
    ),
    "abs_half": ({
        "a": HALVES
    }, np.float16, "c = abs(a)", lambda a: np.abs(a)),
    # A double below fp16's range must come back unchanged: min compares in double, not in half.
    "min_double_half": (
        {
            "a": np.array([1e-20, 3.0] * (N // 2)),
            "b": POSITIVE
        },
        np.float64,
        "c = min(a, b)",
        lambda a, b: np.minimum(a, b.astype(np.float64)),
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_mixed_half_tasklet(name):
    inputs, out_dtype, code, reference = CASES[name]
    sdfg = dace.SDFG(f"fp16_mixed_{name}")
    state = sdfg.add_state()
    for conn, value in {**inputs, "c": np.zeros(N, out_dtype)}.items():
        sdfg.add_array(conn.upper(), [N], dtypes.dtype_to_typeclass(value.dtype.type))
    tasklet, _, _ = state.add_mapped_tasklet(
        "compute",
        {"i": f"0:{N}"},
        {conn: dace.Memlet(f"{conn.upper()}[i]")
         for conn in inputs},
        code,
        {"c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    # Typed connectors, no casts in between: the half meets the other operand directly.
    for conn, value in inputs.items():
        tasklet.in_connectors[conn] = dtypes.dtype_to_typeclass(value.dtype.type)
    tasklet.out_connectors["c"] = dtypes.dtype_to_typeclass(np.dtype(out_dtype).type)
    sdfg.apply_gpu_transformations()

    out = np.zeros(N, out_dtype)
    sdfg(**{conn.upper(): value for conn, value in inputs.items()}, C=out)
    np.testing.assert_allclose(out.astype(np.float64), reference(**inputs).astype(np.float64), rtol=1e-3)
