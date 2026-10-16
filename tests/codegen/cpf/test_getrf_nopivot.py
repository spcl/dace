# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF renders the no-pivot LU parallel on every target: C carries the elimination the CPU pipeline mapped (the
lifted node is expanded in the pipeline, so no node and no serial nest reach the renderer), HIP the rocSOLVER
call the GPU pipeline selected for the node it kept."""

import numpy as np
import pytest

import dace
from dace.codegen.cpf import render
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from tests.codegen.cpf.conftest import assert_standalone_units, build_standalone, call_standalone, render_gpu

N = dace.symbol("N")
SIZE = 19


@dace.program
def lu(A: dace.float64[N, N]):
    for k in range(N):
        A[k + 1 :, k] = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k + 1 :] = A[k + 1 :, k + 1 :] - A[k + 1 :, k][:, None] * A[k, k + 1 :][None, :]


def reference(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    for k in range(a.shape[0]):
        a[k + 1 :, k] /= a[k, k]
        a[k + 1 :, k + 1 :] -= np.outer(a[k + 1 :, k], a[k, k + 1 :])
    return a


@pytest.mark.parametrize("language", ["c", "c++"])
def test_the_cpu_rendering_is_the_parallel_elimination_and_factors(language):
    sdfg = lu.to_sdfg(simplify=False)
    sdfg.name = f"cpf_lu_nopivot_{'c' if language == 'c' else 'cpp'}"
    canonicalize(sdfg, validate=True, target="cpu")
    finalize_for_target(sdfg, "cpu", validate=True)
    rendering = render(sdfg, language=language)
    assert "#pragma omp parallel" in rendering.code and "#pragma omp for" in rendering.code, rendering.code
    assert "getrf" not in rendering.code
    library = build_standalone(rendering.code, sdfg.name, language=language)
    a = np.random.default_rng(1).random((SIZE, SIZE)) + SIZE * np.eye(SIZE)
    arguments = {"A": a.copy(), "N": SIZE}
    call_standalone(library, rendering.sdfg, arguments, rendering.arguments)
    assert np.allclose(arguments["A"], reference(a), rtol=1e-12, atol=1e-12)


def test_the_hip_rendering_calls_rocsolver_without_pivoting():
    rendering = render_gpu(lu, "cpf_hip_lu_nopivot")
    assert_standalone_units(rendering, "cpf_hip_lu_nopivot")
    assert "rocsolver_dgetrf_npvt(" in rendering.code, rendering.code
    assert rendering.code.count("rocblas_dgeam(") == 2, "the column-major solver needs the transposes around it"
    assert "rocsolver" in rendering.libraries, rendering.libraries


if __name__ == "__main__":
    for lang in ("c", "c++"):
        test_the_cpu_rendering_is_the_parallel_elimination_and_factors(lang)
    test_the_hip_rendering_calls_rocsolver_without_pivoting()
