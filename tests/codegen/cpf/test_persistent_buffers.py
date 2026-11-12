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
import subprocess
import warnings

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.cpf import EntrySignature, lift_persistent_transients, render
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


def lifted_nest(write: str, read: str, write_first: bool) -> tuple[dace.SDFG, dace.nodes.NestedSDFG]:
    """``t[write] = 2 * A`` and ``B[read] = t[read] + 1`` in one nested state, the write first or the read first."""
    inner = dace.SDFG("lift_inner")
    inner.add_array("A", [N], dace.float64)
    inner.add_array("B", [N], dace.float64)
    inner.add_array("t", [N], dace.float64, transient=True, lifetime=dtypes.AllocationLifetime.Persistent)
    first, second = inner.add_state("first"), inner.add_state("second")
    inner.add_edge(first, second, dace.InterstateEdge())
    write_state, read_state = (first, second) if write_first else (second, first)
    write_state.add_mapped_tasklet(
        "w", {"i": write}, {"a": dace.Memlet("A[i]")}, "o = 2 * a", {"o": dace.Memlet("t[i]")}, external_edges=True
    )
    read_state.add_mapped_tasklet(
        "r", {"i": read}, {"x": dace.Memlet("t[i]")}, "o = x + 1", {"o": dace.Memlet("B[i]")}, external_edges=True
    )
    sdfg = dace.SDFG("lift_outer")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    state = sdfg.add_state("call")
    node = state.add_nested_sdfg(inner, {"A": None}, {"B": None}, {"N": "N"})
    state.add_edge(state.add_read("A"), None, node, "A", dace.Memlet("A[0:N]"))
    state.add_edge(node, "B", state.add_write("B"), None, dace.Memlet("B[0:N]"))
    return sdfg, node


def test_a_buffer_written_before_every_read_is_no_input_of_the_nest():
    sdfg, node = lifted_nest("0:N", "0:N", write_first=True)
    assert lift_persistent_transients(sdfg) == 1
    assert "t" not in node.in_connectors and "t" in node.out_connectors
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sdfg.validate()
    assert not [w for w in caught if "uninitialized" in str(w.message)], [str(w.message) for w in caught]


@pytest.mark.parametrize("write, write_first", [("0:N", False), ("1:N", True)])
def test_a_buffer_the_nest_may_read_before_writing_stays_an_input(write, write_first):
    """Read first, or written only in part before a whole read: the caller's value of ``t`` is observed."""
    sdfg, node = lifted_nest(write, "0:N", write_first)
    assert lift_persistent_transients(sdfg) == 1
    assert "t" in node.in_connectors and "t" in node.out_connectors


if __name__ == "__main__":
    test_without_a_workspace_the_buffer_is_one_static_allocation("c++")
    test_without_a_workspace_the_buffer_is_one_static_allocation("c")
    test_with_a_workspace_the_buffer_lives_in_it("c++")
    test_with_a_workspace_the_buffer_lives_in_it("c")
    test_a_buffer_written_before_every_read_is_no_input_of_the_nest()
    test_a_buffer_the_nest_may_read_before_writing_stays_an_input("0:N", False)
    test_a_buffer_the_nest_may_read_before_writing_stays_an_input("1:N", True)


def test_two_units_of_one_library_each_release_their_own_buffer(tmp_path):
    """A GPU kernel's host and device units both hold ``__cpf_persistent_buf``. A named guard class gave its
    destructor external (weak) linkage, so the linker kept ONE for both guards and unloading freed one unit's buffer
    twice (``free(): double free``). Each unit's guard destructor must stay local to the unit."""
    units = []
    for name in ("cpf_persistent_host_unit", "cpf_persistent_device_unit"):
        unit = tmp_path / f"{name}.cpp"
        unit.write_text(render(persistent_sdfg(name), language="c++").code)
        units.append(str(unit))
    library = tmp_path / "libtwo_units.so"
    subprocess.run(["g++", "-std=c++20", "-fopenmp", "-fPIC", "-shared", *units, "-o", str(library)], check=True)

    symbols = subprocess.run(["nm", "-C", str(library)], capture_output=True, text=True, check=True).stdout

    destructors = [line for line in symbols.splitlines() if "__cpf_persistent_buf_release_t::~" in line]
    assert all(line.split()[1] == "t" for line in destructors), destructors
    assert len({line.split()[0] for line in destructors}) >= 2, "each unit keeps a destructor of its own"
