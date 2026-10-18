# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``Persistent`` buffer of a nested SDFG a map calls is ONE buffer of the entry, as the state struct keeps it.

CPF has no state struct, and demoting the lifetime made the nested function allocate and free a private copy on
every call: cloudsc's column tiles each allocated 29 full-size buffers inside the ``omp parallel for`` and used
one tile of them. :func:`dace.codegen.cpf.lift_persistent_transients` moves the buffer to the entry. Without a
workspace the entry keeps it in a file-scope ``static`` pointer that lives across calls, only grows, and is freed when
the program exits or the library is unloaded; with one, the
workspace holds it (:func:`dace.codegen.cpf.plan_workspace`).
"""

import re

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.cpf import EntrySignature, render
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N")


@dace.program
def cpf_persistent_inner(x: dace.float64[N], y: dace.float64[N], i: dace.int64):
    buf = np.empty(N, dtype=np.float64)
    buf[i] = x[i] * 2.0
    y[i] = buf[i] + 1.0


@dace.program
def cpf_persistent_outer(x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        cpf_persistent_inner(x, y, i)


def persistent_sdfg(name: str) -> dace.SDFG:
    """The nested ``buf`` made ``Persistent``, as canonicalization leaves a nested SDFG's scratch."""
    sdfg = cpf_persistent_outer.to_sdfg(simplify=False)
    sdfg.name = name
    for nested in sdfg.all_sdfgs_recursive():
        if nested is not sdfg and "buf" in nested.arrays:
            nested.arrays["buf"].lifetime = dtypes.AllocationLifetime.Persistent
    return sdfg


def function_body(code: str, name: str) -> str:
    start = code.index(f"void {name}(")
    return code[code.index("{", start) : code.index("\n}\n", start)]


def nested_functions(code: str) -> list[str]:
    return re.findall(r"static inline void (\w+)\(", code)


@pytest.mark.parametrize("language", ["c++", "c"])
def test_without_a_workspace_the_buffer_is_one_static_allocation(language):
    name = f"cpf_persistent_static_{'cpp' if language == 'c++' else 'c'}"
    result = render(persistent_sdfg(name), language=language)
    for nested in nested_functions(result.code):
        body = function_body(result.code, nested)
        assert not re.search(r"aligned_alloc|= new\b|free\(|delete\[\]", body), body
    entry = function_body(result.code, name)
    # A file-scope pointer, released when the program exits or the library is unloaded: no leak
    assert "static double *__cpf_persistent_buf" in result.code and "__cpf_persistent_buf" in entry
    release = "__cpf_persistent_buf_release(void)" if language == "c" else "~__cpf_persistent_buf_release_t()"
    assert release in result.code
    assert "free(buf)" not in entry and "delete[](buf" not in entry
    library = build_standalone(result.code, name, language=language)
    # A larger call grows the buffer, a smaller one reuses it.
    for n in (50, 80, 30):
        x = np.random.default_rng(n).random(n)
        args = {"x": x, "y": np.zeros(n), "N": n}
        call_standalone(library, result.sdfg, args)
        np.testing.assert_allclose(args["y"], x * 2.0 + 1.0, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("language", ["c++", "c"])
def test_with_a_workspace_the_buffer_lives_in_it(language):
    name = f"cpf_persistent_ws_{'cpp' if language == 'c++' else 'c'}"
    order = ("x", "y", "N", "workspace", "workspace_size")
    result = render(persistent_sdfg(name), language=language, signature=EntrySignature(order, *order[3:]))
    assert "__cpf_persistent_" not in result.code
    assert re.search(r"buf = \(double \*\)\(workspace \+", result.code)
    n = 70
    needed = int(eval(result.workspace_bytes, {"N": n}))  # noqa: S307 -- an expression the renderer wrote
    assert needed >= 8 * n
    backing = np.zeros(needed + 64, dtype=np.uint8)
    start = (-backing.ctypes.data) % 64
    x = np.random.default_rng(1).random(n)
    args = {"x": x, "y": np.zeros(n), "N": n, "workspace": backing[start : start + needed], "workspace_size": needed}
    call_standalone(build_standalone(result.code, name, language=language), result.sdfg, args, order=result.arguments)
    np.testing.assert_allclose(args["y"], x * 2.0 + 1.0, rtol=1e-12, atol=0.0)


if __name__ == "__main__":
    test_without_a_workspace_the_buffer_is_one_static_allocation("c++")
    test_without_a_workspace_the_buffer_is_one_static_allocation("c")
    test_with_a_workspace_the_buffer_lives_in_it("c++")
    test_with_a_workspace_the_buffer_lives_in_it("c")
