# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU tasklets mixing a ``dace.float16`` operand with a built-in one, end to end.

Each kernel is one mapped tasklet whose connectors carry the dtypes directly, with no casts in
between -- the shape a mixed-precision retyping of an fp64 program produces. Under CUDA every one
of them used to fail to compile, because ``dace::float16`` is ``__half`` and nvcc found the
built-in operator and ``__half``'s own equally good (see ``cuda_fp16_mixed_operand_test`` for the
header-level probes). These run the kernels and check the numbers against numpy, with the half
promoted to the wider operand's type as C++ (and ``dtypes.result_type_of``) promote it.
"""
from typing import Dict

import numpy as np
import pytest

import dace

#: The whole module is about float16, so it carries the marker the fp16 CI leg selects on.
pytestmark = [pytest.mark.gpu, pytest.mark.fp16]

N = 64
f16, f32, f64 = dace.float16, dace.float32, dace.float64


def _mixed_tasklet_sdfg(name: str, inputs: Dict[str, dace.typeclass], out_dtype: dace.typeclass,
                        code: str) -> dace.SDFG:
    """``C[i] = code(A[i], B[i], ...)`` on the GPU, the tasklet's connectors typed as given."""
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    for conn, dtype in {**inputs, "c": out_dtype}.items():
        sdfg.add_array(conn.upper(), [N], dtype)
    tasklet, _, _ = state.add_mapped_tasklet(
        "compute",
        {"i": f"0:{N}"},
        {conn: dace.Memlet(f"{conn.upper()}[i]")
         for conn in inputs},
        code,
        {"c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    for conn, dtype in inputs.items():
        tasklet.in_connectors[conn] = dtype
    tasklet.out_connectors["c"] = out_dtype
    sdfg.apply_gpu_transformations()
    return sdfg


def _run(sdfg: dace.SDFG, out_dtype, **inputs) -> np.ndarray:
    out = np.zeros(N, dtype=out_dtype)
    sdfg(**{k.upper(): v for k, v in inputs.items()}, C=out)
    return out


def _halves(rng: np.random.Generator, low: float = -4.0, high: float = 4.0) -> np.ndarray:
    return rng.uniform(low, high, N).astype(np.float16)


def test_half_compared_with_double_literal():
    rng = np.random.default_rng(0)
    a = _halves(rng)
    a[:4] = [0.0, -0.0, np.float16(6e-8), -1.0]  # around the literal: zero, the smallest subnormal
    sdfg = _mixed_tasklet_sdfg("fp16_mixed_lt_literal", {"a": f16}, dace.bool_, "c = a < 1e-14")
    out = _run(sdfg, np.bool_, a=a)
    np.testing.assert_array_equal(out, a.astype(np.float64) < 1e-14)


def test_float_divided_by_half():
    rng = np.random.default_rng(1)
    a = rng.uniform(-10, 10, N).astype(np.float32)
    b = _halves(rng, 0.5, 4.0)
    sdfg = _mixed_tasklet_sdfg("fp16_mixed_div", {"a": f32, "b": f16}, f32, "c = a / b")
    out = _run(sdfg, np.float32, a=a, b=b)
    np.testing.assert_array_equal(out, a / b.astype(np.float32))


def test_float_equal_to_half():
    rng = np.random.default_rng(2)
    b = _halves(rng)
    a = b.astype(np.float32)
    a[::2] += 1e-3  # half of them no longer representable as the half they came from
    sdfg = _mixed_tasklet_sdfg("fp16_mixed_eq", {"a": f32, "b": f16}, dace.bool_, "c = a == b")
    out = _run(sdfg, np.bool_, a=a, b=b)
    np.testing.assert_array_equal(out, a == b.astype(np.float32))


def test_half_power_of_double():
    rng = np.random.default_rng(3)
    a = _halves(rng, 0.5, 2.0)
    b = rng.uniform(-2, 2, N)
    sdfg = _mixed_tasklet_sdfg("fp16_mixed_pow_double", {"a": f16, "b": f64}, f64, "c = a ** b")
    out = _run(sdfg, np.float64, a=a, b=b)
    np.testing.assert_allclose(out, a.astype(np.float64)**b, rtol=1e-12)


def test_half_power_of_half():
    rng = np.random.default_rng(4)
    a = _halves(rng, 0.5, 2.0)
    b = _halves(rng, -2.0, 2.0)
    sdfg = _mixed_tasklet_sdfg("fp16_mixed_pow_half", {"a": f16, "b": f16}, f16, "c = a ** b")
    out = _run(sdfg, np.float16, a=a, b=b)
    expected = (a.astype(np.float32)**b.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(out.astype(np.float32), expected.astype(np.float32), rtol=1e-3)


def test_abs_of_half():
    rng = np.random.default_rng(5)
    a = _halves(rng)
    sdfg = _mixed_tasklet_sdfg("fp16_mixed_abs", {"a": f16}, f16, "c = abs(a)")
    out = _run(sdfg, np.float16, a=a)
    np.testing.assert_array_equal(out, np.abs(a))


def test_min_of_double_and_half_keeps_the_double():
    """A double below fp16's range must come back unchanged: the comparison runs in double, not
    in half, where ``1e-20`` would round to zero."""
    rng = np.random.default_rng(6)
    a = np.full(N, 1e-20)
    a[::2] = 3.0
    b = _halves(rng, 0.5, 2.0)
    sdfg = _mixed_tasklet_sdfg("fp16_mixed_min", {"a": f64, "b": f16}, f64, "c = min(a, b)")
    out = _run(sdfg, np.float64, a=a, b=b)
    np.testing.assert_array_equal(out, np.minimum(a, b.astype(np.float64)))


def test_float_add_assign_half():
    rng = np.random.default_rng(7)
    a = rng.uniform(-10, 10, N).astype(np.float32)
    b = _halves(rng)
    sdfg = _mixed_tasklet_sdfg("fp16_mixed_add_assign", {"a": f32, "b": f16}, f32, "c = a\nc += b")
    out = _run(sdfg, np.float32, a=a, b=b)
    np.testing.assert_array_equal(out, a + b.astype(np.float32))


def test_sin_and_fmod_of_half():
    rng = np.random.default_rng(8)
    a = _halves(rng)
    b = rng.uniform(1, 2, N)
    sdfg = _mixed_tasklet_sdfg("fp16_mixed_libm", {"a": f16, "b": f64}, f64, "c = math.sin(a) + math.fmod(a, b)")
    out = _run(sdfg, np.float64, a=a, b=b)
    expected = np.sin(a.astype(np.float32)).astype(np.float16).astype(np.float64) + np.fmod(a.astype(np.float64), b)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


if __name__ == "__main__":
    test_half_compared_with_double_literal()
    test_float_divided_by_half()
    test_float_equal_to_half()
    test_half_power_of_double()
    test_half_power_of_half()
    test_abs_of_half()
    test_min_of_double_and_half_keeps_the_double()
    test_float_add_assign_half()
    test_sin_and_fmod_of_half()
