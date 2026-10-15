# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF lowers BLAS, LAPACK and FFT library nodes to the one vendor library family of the architecture.

On the host that is OpenBLAS (CBLAS and LAPACKE) and FFTW3: a matrix product is ``cblas_dgemm``, a Cholesky
factorization ``LAPACKE_dpotrf``, a transform an FFTW plan -- not CPF loops. The unit stays self-contained in the
DaCe sense (no DaCe header, no init handshake): it includes the library's own system headers, and names the
library on :attr:`~dace.codegen.cpf.Rendering.link` and by family on :attr:`~dace.codegen.cpf.Rendering.libraries`.
Each case is built against that link line and computes what numpy does.
"""

import numpy as np
import pytest

import dace
from dace.codegen.cpf import render
from dace.libraries.blas.environments.openblas import OpenBLAS
from dace.libraries.fft.environments.fftw3 import FFTW3
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N")
SIZE = 24


@dace.program
def vendor_gemm(a: dace.float64[N, N], b: dace.float64[N, N], c: dace.float64[N, N]):
    c[:] = a @ b


@dace.program
def vendor_zgemm(a: dace.complex128[N, N], b: dace.complex128[N, N], c: dace.complex128[N, N]):
    c[:] = a @ b


@dace.program
def vendor_cholesky(a: dace.float64[N, N], low: dace.float64[N, N]):
    low[:] = np.linalg.cholesky(a)


@dace.program
def vendor_solve(a: dace.float64[N, N], b: dace.float64[N], x: dace.float64[N]):
    x[:] = np.linalg.solve(a, b)


@dace.program
def vendor_fft(a: dace.complex128[N], b: dace.complex128[N]):
    b[:] = np.fft.fft(a)


def spd(rng):
    m = rng.standard_normal((SIZE, SIZE))
    return m @ m.T + SIZE * np.eye(SIZE)


def complex_matrix(rng):
    return rng.standard_normal((SIZE, SIZE)) + 1j * rng.standard_normal((SIZE, SIZE))


#: label -> (program, the call the unit must make, its family, inputs, outputs from numpy)
CASES = {
    "gemm": (
        vendor_gemm,
        "cblas_dgemm",
        "blas",
        lambda rng: {"a": rng.standard_normal((SIZE, SIZE)), "b": rng.standard_normal((SIZE, SIZE))},
        lambda inputs: {"c": inputs["a"] @ inputs["b"]},
    ),
    "zgemm": (
        vendor_zgemm,
        "cblas_zgemm",
        "blas",
        lambda rng: {"a": complex_matrix(rng), "b": complex_matrix(rng)},
        lambda inputs: {"c": inputs["a"] @ inputs["b"]},
    ),
    "cholesky": (
        vendor_cholesky,
        "LAPACKE_dpotrf",
        "lapack",
        lambda rng: {"a": spd(rng)},
        lambda inputs: {"low": np.linalg.cholesky(inputs["a"])},
    ),
    "solve": (
        vendor_solve,
        "LAPACKE_dgetrf",
        "lapack",
        lambda rng: {"a": spd(rng), "b": rng.standard_normal(SIZE)},
        lambda inputs: {"x": np.linalg.solve(inputs["a"], inputs["b"])},
    ),
    "fft": (
        vendor_fft,
        "fftw_plan_guru64_dft",
        "fftw",
        lambda rng: {"a": rng.standard_normal(SIZE) + 1j * rng.standard_normal(SIZE)},
        lambda inputs: {"b": np.fft.fft(inputs["a"])},
    ),
}


@pytest.mark.parametrize("language", ["c++", "c"])
@pytest.mark.parametrize("label", sorted(CASES))
def test_a_library_node_renders_as_the_vendor_call_and_computes_what_numpy_does(label, language):
    program, call, family, make_inputs, expected = CASES[label]
    environment = FFTW3 if family == "fftw" else OpenBLAS
    if not environment.is_installed():
        pytest.skip(f"{environment.__name__} is not installed here")
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = f"cpf_vendor_{label}_{'c' if language == 'c' else 'cpp'}"
    rendering = render(sdfg, language=language)
    assert f"{call}(" in rendering.code, f"{label}: no {call} call in the rendering:\n{rendering.code}"
    assert family in rendering.libraries, rendering.libraries
    assert rendering.link, f"{label}: a unit that calls {family} must name it on its link line"

    # The harness adds this machine's install paths and the rendering's link line (conftest.vendor_build_flags).
    library = build_standalone(rendering.code, sdfg.name, language=language)
    inputs = make_inputs(np.random.default_rng(0))
    arguments = {name: value.copy() for name, value in inputs.items()}
    for name, want in expected(inputs).items():
        arguments[name] = np.zeros_like(want)
    arguments["N"] = SIZE
    call_standalone(library, rendering.sdfg, arguments, rendering.arguments)
    for name, want in expected(inputs).items():
        assert np.allclose(arguments[name], want, rtol=1e-10, atol=1e-10), (
            f"{label}/{name}: {arguments[name]} != {want}"
        )


def test_the_openblas_unit_keeps_a_pthreads_build_off_the_openmp_team():
    """DaCe's environment runs ``DeferOpenBlasThreadsToOpenMP`` at init; the unit has no init, so it carries the
    same step as a load-time constructor -- and no DaCe header for it."""
    code = render(vendor_gemm.to_sdfg(simplify=True), language="c", check_compiles=False).code
    assert "__attribute__((constructor)) static void cpf_defer_openblas_threads" in code
    assert "dace_openblas.h" not in code and "dace_blas.h" not in code


if __name__ == "__main__":
    for case in sorted(CASES):
        for lang in ("c++", "c"):
            test_a_library_node_renders_as_the_vendor_call_and_computes_what_numpy_does(case, lang)
    test_the_openblas_unit_keeps_a_pthreads_build_off_the_openmp_team()
