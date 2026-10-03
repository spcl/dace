# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A CUDA ``dace::float16`` mixed with a built-in number, at the header level.

Under CUDA ``dace::float16`` IS ``__half``, which converts implicitly to every built-in arithmetic
type and is constructible from each of them. Any expression pairing a half with a ``float``,
``double`` or ``int`` therefore offers nvcc two equally good candidates -- the built-in operator
(converting the half up) and ``operator<op>(__half, __half)`` (converting the other side down) --
and nvcc rejects the translation unit:

    error: more than one operator "<" matches these operands:
                built-in operator "arithmetic < arithmetic"
                function "operator<(const __half &, const __half &)"
                operand types are: dace::float16 < double

The same tie hits every libm call (``sin``, ``floor``, ``fmod``, ``std::pow``, ``abs``: the
float/double/long double overloads are each one conversion away), compound assignment into a
built-in (``float x; x += h``), and every helper whose result type is ``std::common_type`` of a
half and a built-in (``IfExpr``, the variadic ``min``/``max``), which has no answer and so dropped
the helper from overload resolution. ``Min``/``Max`` compiled, but returned the first argument's
type, so ``Min(h, 1e-20)`` was rounded back to half.

``dace/math.h`` now answers each with an exact match that promotes the half to ``float`` and runs
the built-in operation, so the result follows C++ promotion (half op float -> float, half op
double -> double) and a half op half still uses CUDA's native half operators.

Every probe is one C++ tasklet in a GPU device map, compiled through ``SDFG.compile``.
"""

from typing import Optional

import pytest

import dace
from dace import dtypes
from dace.codegen.exceptions import CompilationError

pytestmark = pytest.mark.gpu

#: One operand of each kind, read from GPU memory so nothing folds at compile time.
INPUTS = {
    "h": dace.float16,
    "f": dace.float32,
    "d": dace.float64,
    "i": dace.int32,
    "b": dace.bool_,
}
#: Result connectors; a probe gets the ones its body assigns.
OUTPUTS = {
    "out_h": dace.float16,
    "out_f": dace.float32,
    "out_d": dace.float64,
    "out_i": dace.int32,
    "out_b": dace.bool_,
}


def _probe_sdfg(name: str, body: str) -> dace.SDFG:
    """A single-thread GPU map around one C++ tasklet running ``body`` on the ``INPUTS``."""
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    outputs = {conn: dtype for conn, dtype in OUTPUTS.items() if conn in body}
    for conn, dtype in {**INPUTS, **outputs}.items():
        sdfg.add_array(conn.upper(), [1], dtype, storage=dtypes.StorageType.GPU_Global)
    map_entry, map_exit = state.add_map(
        "probe", {"k": "0:1"}, schedule=dtypes.ScheduleType.GPU_Device
    )
    tasklet = state.add_tasklet(
        "probe", set(INPUTS), set(outputs), body + ";", language=dtypes.Language.CPP
    )
    for conn, dtype in INPUTS.items():
        tasklet.in_connectors[conn] = dtype
        state.add_memlet_path(
            state.add_read(conn.upper()),
            map_entry,
            tasklet,
            dst_conn=conn,
            memlet=dace.Memlet(f"{conn.upper()}[0]"),
        )
    for conn, dtype in outputs.items():
        tasklet.out_connectors[conn] = dtype
        state.add_memlet_path(
            tasklet,
            map_exit,
            state.add_write(conn.upper()),
            src_conn=conn,
            memlet=dace.Memlet(f"{conn.upper()}[0]"),
        )
    return sdfg


def _compile(name: str, body: str) -> Optional[str]:
    """Compile the probe; ``None`` on success, else the compiler output."""
    try:
        _probe_sdfg(f"fp16_mixed_{name}", body).compile()
        return None
    except CompilationError as exc:
        return str(exc)


#: Expressions that used to be rejected as ambiguous (or, for ``IfExpr`` and the variadic
#: ``min``/``max``, as having no viable candidate) and must now compile. The cases checked by
#: value are in ``fp16_mixed_operand_cudatest``.
MIXED_EXPRESSIONS = {
    # Arithmetic with an int (the comparison, float and double cases run in the GPU test).
    "half_mul_int": "out_f = h * i",
    # Compound assignment into a built-in.
    "float_add_assign_half": "f += h; out_f = f",
    # Result-type helpers: Min/Max with a float and with an int beside the half, and IfExpr.
    "Min_double_half": "out_d = Min(d, h)",
    "Min_half_int": "out_h = Min(h, i)",
    "IfExpr_half_float": "out_f = IfExpr(b, h, f)",
    # Lowercase min/max with an integral operand (the floating-point one runs in the GPU test).
    "min_half_int": "out_h = min(h, i)",
    # libm, one unary (half) and one binary (half with double), bare and through dace::math.
    "sin_half": "out_h = sin(h)",
    "fmod_half_double": "out_d = fmod(h, d)",
    "dace_math_sin_half": "out_h = dace::math::sin(h)",
    "dace_math_fmod_half_double": "out_d = dace::math::fmod(h, d)",
    # The pow overloads the GPU test does not reach: a half exponent, and an integral one (ipow).
    "pow_double_half": "out_d = dace::math::pow(d, h)",
    "pow_half_int": "out_h = dace::math::pow(h, i)",
}


@pytest.mark.parametrize(
    "name,body", MIXED_EXPRESSIONS.items(), ids=MIXED_EXPRESSIONS.keys()
)
def test_mixed_half_expression_compiles(name, body):
    error = _compile(name, body)
    assert error is None, f"`{body}` failed to compile:\n{error}"
