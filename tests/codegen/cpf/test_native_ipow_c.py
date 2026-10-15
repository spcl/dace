# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A library expansion formats its sizes with ``str`` of a dace expression, which spells a power ``ipow(N, 2)``.
C has no ``ipow``: the C rewrite of the body calls the typed helper for the base's type (ls3df_scf's GEMV over
``Lb**3`` rows did not compile)."""

import numpy as np
import pytest

import dace
from dace import cpf_lowering
from dace.codegen.cpf import render
from dace.libraries.blas.environments.openblas import OpenBLAS
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N")


def test_a_bare_and_a_qualified_ipow_become_the_typed_helper():
    site = cpf_lowering.NativeSite({"Lb": ("int64", False), "x": ("float64", False)}, "site")
    body = "if ((ipow(Lb, 3)) <= 0) { y = dace::math::ipow(x, 2) + ipow(ipow(Lb, 2), 2); }"
    code = cpf_lowering.rewrite_native_code(body, cpf_lowering.Dialect.STANDALONE_C, site)
    assert "cpf_ipow_int64(Lb, 3)" in code
    assert "cpf_ipow_float64(x, 2)" in code
    assert "cpf_ipow_int64(cpf_ipow_int64(Lb, 2), 2)" in code
    assert "ipow(" not in code.replace("cpf_ipow_", "")


def test_an_untyped_base_is_refused():
    with pytest.raises(NotImplementedError, match="ipow"):
        cpf_lowering.rewrite_native_code(
            "y = ipow(unknown, 2);", cpf_lowering.Dialect.STANDALONE_C, cpf_lowering.NativeSite({}, "site")
        )


@dace.program
def squared_rows_gemv(a: dace.float64[N**2, N], x: dace.float64[N], y: dace.float64[N**2]):
    y[:] = a @ x


def test_a_gemv_over_a_power_of_a_symbol_renders_and_runs_in_c():
    if not OpenBLAS.is_installed():
        pytest.skip("OpenBLAS is not installed here")
    sdfg = squared_rows_gemv.to_sdfg(simplify=True)
    sdfg.name = "cpf_native_ipow_gemv"
    rendering = render(sdfg, language="c")
    assert "cblas_dgemv(" in rendering.code
    library = build_standalone(rendering.code, sdfg.name, language="c")
    rng = np.random.default_rng(0)
    size = 5
    arguments = {"a": rng.standard_normal((size**2, size)), "x": rng.standard_normal(size), "y": np.zeros(size**2)}
    expected = arguments["a"] @ arguments["x"]
    arguments["N"] = size
    call_standalone(library, rendering.sdfg, arguments, rendering.arguments)
    assert np.allclose(arguments["y"], expected)


if __name__ == "__main__":
    test_a_bare_and_a_qualified_ipow_become_the_typed_helper()
    test_an_untyped_base_is_refused()
    test_a_gemv_over_a_power_of_a_symbol_renders_and_runs_in_c()
