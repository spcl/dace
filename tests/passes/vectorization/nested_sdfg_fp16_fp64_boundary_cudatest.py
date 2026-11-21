# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An fp16 outer program calling an fp64 nested program, offloaded to the GPU.

The frontend now refuses the implicit call: an fp16 array handed to an fp64 parameter is a type error, not a
reinterpretation (``test_an_fp16_array_for_an_fp64_parameter_is_refused``). The program below converts at the call
the way the message asks -- ``reciprocal_fp64(a.astype(np.float64), wide)``, then ``b[:] = wide`` -- which is the
semantics the tests always meant: ``1.0 / a`` in double precision, rounded to fp16 once on the way out. The history
the rest of this docstring records is why the literal typing is still checked.

``nested`` is written entirely in ``float64`` and takes ``float64`` arrays; ``kernel`` is written in
``float16`` and calls ``nested`` on its own ``float16`` arrays. Simplification inlines ``nested``
into ``kernel``, and the connector dtypes are reconciled to the CONTAINER's dtype rather than the
callee's: the whole map ends up over ``dace::float16`` end to end, with no ``double`` array left
anywhere in the SDFG. That collapse is silent -- validation and ``apply_gpu_transformations`` both
accept it -- so the only place the mismatch still shows is the literal in the divide, which was
written for the ``float64`` body: ``1.0 / a`` reaches C++ as a bare, untyped ``1.0`` against a
``dace::float16`` operand. ``__half`` (what ``dace::float16`` is on the device) offers an implicit
conversion in both directions, so ``operator/`` cannot choose between ``double / double`` and
``__half / __half`` -- the exact ambiguity ``fp16_ite_literal_arm_cudatest`` documents for an
untyped ternary arm, here for a binary operator instead:

    error: more than one operator "/" matches these operands:
                built-in operator "arithmetic / arithmetic"
                function "operator/(const __half &, const __half &)"
    b_gpu[b_gpu_idx(__i0)] = (1.0 / a_gpu[a_gpu_idx(__i0)]);

The structural test below reproduces that at the source-text level without needing nvcc or a GPU:
the emitted divide carries no explicit type on either operand. The GPU tests attempt to compile and
run the same program and are expected to fail the same way until the literal (or the promotion
implied by the original ``float64`` body) carries an explicit dtype.
"""

import re

import numpy as np
import pytest

import dace

#: The whole module is about float16, so it carries the marker the fp16 CI leg selects on.
pytestmark = pytest.mark.fp16

N = dace.symbol("N")

#: One extent that divides a typical GPU block size, one that does not.
EXTENTS = (1024, 1023)


@dace.program
def reciprocal_fp64(a: dace.float64[N], b: dace.float64[N]):
    b[:] = 1.0 / a


@dace.program
def reciprocal_fp16_calls_fp64_nested(a: dace.float16[N], b: dace.float16[N]):
    wide = np.empty(N, dtype=np.float64)
    reciprocal_fp64(a.astype(np.float64), wide)
    b[:] = wide


@dace.program
def reciprocal_fp16_passes_fp16_to_fp64(a: dace.float16[N], b: dace.float16[N]):
    reciprocal_fp64(a, b)


def offloaded(n: int, name: str = None) -> dace.SDFG:
    """``reciprocal_fp16_calls_fp64_nested`` specialized to ``n`` and offloaded to the GPU.

    No vectorizer runs here: the bug under test is at the nested-SDFG dtype boundary, reproduced
    with ``apply_gpu_transformations`` alone, exactly as reported.

    :param n: the extent to specialize ``N`` to.
    :param name: optional distinct SDFG name, so repeated compiles do not share a stale build.
    :returns: the offloaded SDFG, ready to compile.
    """
    sdfg = reciprocal_fp16_calls_fp64_nested.to_sdfg(simplify=True)
    sdfg.specialize({"N": n})
    sdfg.apply_gpu_transformations()
    if name:
        sdfg.name = name
    return sdfg


def device_code(sdfg: dace.SDFG) -> str:
    return "\n".join(c.clean_code for c in sdfg.generate_code() if c.title == "CUDA")


def test_an_fp16_array_for_an_fp64_parameter_is_refused():
    """Handing float16 bytes to a float64 parameter reinterpreted them; the call must convert first."""
    with pytest.raises(dace.frontend.python.common.DaceSyntaxError, match="declared float64"):
        reciprocal_fp16_passes_fp16_to_fp64.to_sdfg(simplify=True)


# Structural: what reaches the C++ divide (no GPU device and no nvcc needed)
def test_boundary_division_operand_is_explicitly_typed():
    """The divide inherited from the fp64 nested program must not reach C++ with an untyped literal
    against a bare ``dace::float16`` operand -- that pairing is what nvcc rejects as ambiguous, and
    the arm that silently narrows to half precision instead would be worse than a compile error.
    """
    sdfg = offloaded(1024)
    code = device_code(sdfg)
    half = {name for sd in sdfg.all_sdfgs_recursive() for name, desc in sd.arrays.items() if desc.dtype == dace.float16}
    # The match window is the whole NUMERATOR of a divide by ``a_gpu[...]``, not just the ``1.0``:
    # a cast sits between the two, so a window that ended at the literal could only ever match the
    # spelling being ruled out -- and would then report "no division emitted" once it is gone.
    # The operand is ``a_gpu[...]`` once the tasklet connector is inlined, its connector otherwise.
    divides = re.findall(r"([^;\n=]*\b1\.0\b[^;\n=]*)/\s*([A-Za-z_]\w*)", code)
    assert divides, "no reciprocal division emitted; the test would prove nothing"
    for expr, operand in divides:
        if operand in half:
            assert "dace::float16(" in expr or "static_cast<double>" in expr or "double(" in expr, (
                f"a division by the float16 {operand} reached C++ with an untyped literal: {expr}"
            )


# GPU: compiling and running the same program
@pytest.mark.gpu
def test_generated_code_compiles():
    """nvcc is where the ambiguous ``operator/`` overload surfaces; the host compiler never sees it."""
    offloaded(1024, name="nested_fp16_fp64_boundary_compile").compile()


@pytest.mark.gpu
@pytest.mark.parametrize("n", EXTENTS)
def test_numeric_matches_fp64_then_fp16_cast(n):
    """The nested program's body is ``float64``: the intended result is ``1.0 / a`` computed in
    double precision and rounded to ``float16`` once on the way out, not a division carried out
    entirely in half precision. The tolerance is one fp16 ULP, not bit-exactness, so a correctly
    rounded but differently-derived device reciprocal is not penalized for reassociation the way a
    silent full-precision narrowing would be.
    """
    csr = offloaded(n, name=f"nested_fp16_fp64_boundary_numeric_{n}").compile()
    a = (np.arange(1, n + 1) % 15 + 1).astype(np.float16)  # nonzero, exactly representable
    b = np.zeros(n, dtype=np.float16)
    csr(a=a, b=b, N=n)
    expected = (1.0 / a.astype(np.float64)).astype(np.float16)
    fp16_ulp = float(np.finfo(np.float16).eps)
    assert np.allclose(b.astype(np.float64), expected.astype(np.float64), rtol=fp16_ulp, atol=0.0), (
        f"N={n}: {int((b != expected).sum())} of {n} lanes differ from the fp64-then-cast oracle"
    )


if __name__ == "__main__":
    test_an_fp16_array_for_an_fp64_parameter_is_refused()
    test_boundary_division_operand_is_explicitly_typed()
    test_generated_code_compiles()
    for n in EXTENTS:
        test_numeric_matches_fp64_then_fp16_cast(n)
