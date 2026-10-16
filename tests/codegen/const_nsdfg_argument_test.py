# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Const-qualifier detection for nested-SDFG (emitted function) parameters.

A nested SDFG's parameters must be ``const``-qualified exactly when the connector is read-only --
for array references as well as scalars. Codegen only, except for the pointer-typed scalars, whose
qualification is checked by compiling and running programs.
"""

import re

import numpy as np
import pytest

import dace
from dace.codegen.targets import cpp
from dace.codegen.targets.cpu import CPUCodeGen


def _signature(inner: dace.SDFG, in_conns, out_conns, wirings) -> str:
    """Build ``map -> inner`` and return the emitted parameter list of ``inner``.

    ``wirings``: list of ``(connector, outer_name, subset, is_input)``.
    """
    sdfg = dace.SDFG("outer_" + inner.name)
    for _conn, oname, _sub, _isin in wirings:
        if oname not in sdfg.arrays:
            sdfg.add_array(oname, (16, 8), dace.float64)
    state = sdfg.add_state("main")
    entry, exit_ = state.add_map("m", dict(i="0:16"))
    nsdfg = state.add_nested_sdfg(inner, dict.fromkeys(in_conns), dict.fromkeys(out_conns))
    for conn, oname, sub, is_input in wirings:
        access = state.add_access(oname)
        if is_input:
            state.add_memlet_path(access, entry, nsdfg, dst_conn=conn, memlet=dace.Memlet(data=oname, subset=sub))
        else:
            state.add_memlet_path(nsdfg, exit_, access, src_conn=conn, memlet=dace.Memlet(data=oname, subset=sub))
    nsdfg.integrate_into_parent()

    code = "\n".join(o.code for o in sdfg.generate_code())
    match = re.search(rf"void {re.escape(inner.name)}\w*\(([^)]*)\)", code)
    assert match is not None, f"no function emitted for {inner.name}"
    return match.group(1)


def _param(signature: str, connector: str) -> str:
    for part in signature.split(","):
        if re.search(rf"\b{re.escape(connector)}\b", part):
            return part.strip()
    raise AssertionError(f"connector {connector!r} not in signature: {signature!r}")


def _inner_copy(name: str) -> dace.SDFG:
    """Inner SDFG ``b[j] = a[j]``: ``a`` read-only, ``b`` written."""
    g = dace.SDFG(name)
    g.add_array("a", (8,), dace.float64)
    g.add_array("b", (8,), dace.float64)
    s = g.add_state("s")
    s.add_mapped_tasklet(
        "cp", dict(j="0:8"), {"x": dace.Memlet("a[j]")}, "y = x", {"y": dace.Memlet("b[j]")}, external_edges=True
    )
    return g


def _inner_with_view(name: str, write_through_view: bool) -> dace.SDFG:
    """Inner SDFG with a ``View`` ``av`` of ``a``, read-direction or write-direction."""
    g = dace.SDFG(name)
    g.add_array("a", (8,), dace.float64)
    g.add_array("b", (8,), dace.float64)
    g.add_view("av", (8,), dace.float64)
    s = g.add_state("s")
    a, b, av = s.add_access("a"), s.add_access("b"), s.add_access("av")
    t = s.add_tasklet("cp", {"x": None}, {"y": None}, "y = x")
    if write_through_view:
        s.add_edge(b, None, t, "x", dace.Memlet("b[0]"))
        s.add_edge(t, "y", av, None, dace.Memlet("av[0]"))
        s.add_edge(av, None, a, None, dace.Memlet("a[0:8]"))
    else:
        s.add_edge(a, None, av, None, dace.Memlet("a[0:8]"))
        s.add_edge(av, None, t, "x", dace.Memlet("av[0]"))
        s.add_edge(t, "y", b, None, dace.Memlet("b[0]"))
    return g


def test_readonly_input_is_const_and_written_output_is_not():
    sig = _signature(_inner_copy("roarr"), {"a"}, {"b"}, [("a", "A", "i,0:8", True), ("b", "B", "i,0:8", False)])
    assert _param(sig, "A").startswith("const "), _param(sig, "A")
    assert not _param(sig, "B").startswith("const "), _param(sig, "B")


def test_inout_array_is_not_const():
    g = dace.SDFG("inout")
    g.add_array("d", (8,), dace.float64)
    s = g.add_state("s")
    s.add_mapped_tasklet(
        "inc", dict(j="0:8"), {"v": dace.Memlet("d[j]")}, "w = v + 1.0", {"w": dace.Memlet("d[j]")}, external_edges=True
    )
    sig = _signature(g, {"d"}, {"d"}, [("d", "D", "i,0:8", True), ("d", "D", "i,0:8", False)])
    assert not _param(sig, "D").startswith("const "), _param(sig, "D")


def test_a_read_view_of_a_const_input_is_emitted_const():
    """A non-const view of const data is an illegal ``const T* -> T*`` conversion."""
    inner = _inner_with_view("rov", write_through_view=False)
    sdfg = dace.SDFG("outer_rov")
    for n in ("A", "B"):
        sdfg.add_array(n, (16, 8), dace.float64)
    state = sdfg.add_state("main")
    entry, exit_ = state.add_map("m", dict(i="0:16"))
    nsdfg = state.add_nested_sdfg(inner, {"a": None}, {"b": None})
    state.add_memlet_path(state.add_access("A"), entry, nsdfg, dst_conn="a", memlet=dace.Memlet("A[i, 0:8]"))
    state.add_memlet_path(nsdfg, exit_, state.add_access("B"), src_conn="b", memlet=dace.Memlet("B[i, 0:8]"))
    nsdfg.integrate_into_parent()
    code = "\n".join(o.code for o in sdfg.generate_code())

    # Integration leaves "a" a view of the const parameter "A", and "av" a view of that view.
    assert re.search(r"const\s+double\s*\*\s*a\s*;", code), 'view "a" is not pointer-to-const'
    assert re.search(r"const\s+double\s*\*\s*av\s*;", code), 'view "av" is not pointer-to-const'


def test_mutated_descriptors_follows_view_direction():
    """A read view leaves its parent const-qualifiable; a written view taints it."""
    assert "a" not in CPUCodeGen._mutated_descriptors(_inner_with_view("rov", write_through_view=False))
    assert "a" in CPUCodeGen._mutated_descriptors(_inner_with_view("wov", write_through_view=True))


def test_const_qualify_pointer_data():
    """
    Constant pointer-typed data is a constant pointer, not a pointer to constant data.
    """
    pointer = dace.pointer(dace.float64)
    assert cpp.const_qualify("double", dace.float64) == "const double"
    assert cpp.const_qualify("double*", dace.float64) == "const double*"
    assert cpp.const_qualify("double*", pointer) == "double* const"
    assert cpp.const_qualify("double**", pointer) == "double* const*"
    assert cpp.const_qualify("double* const*", pointer) == "double* const*"


def _pointer_program(gpu: bool) -> dace.SDFG:
    """
    A tasklet outputs a pointer to a table, which a nested SDFG in a map (a GPU kernel) reads through.
    """
    n = 16
    storage = dace.StorageType.GPU_Global if gpu else dace.StorageType.Default
    sdfg = dace.SDFG("pointer_scalar_" + ("gpu" if gpu else "cpu"))
    sdfg.add_array("y", [n], dace.float64, storage)
    sdfg.add_scalar("table", dace.pointer(dace.float64), transient=True)
    state = sdfg.add_state()
    if gpu:
        code = (
            f"static double* values = nullptr;\n"
            f"if (!values) {{\n"
            f"    double host[{n}];\n"
            f"    for (int i = 0; i < {n}; ++i) host[i] = i;\n"
            f"    cudaMalloc(&values, sizeof(host));\n"
            f"    cudaMemcpy(values, host, sizeof(host), cudaMemcpyHostToDevice);\n"
            f"}}\n"
            f"_table = values;"
        )
    else:
        code = f"static double values[{n}];\nfor (int i = 0; i < {n}; ++i) values[i] = i;\n_table = values;"
    tasklet = state.add_tasklet("table", {}, {"_table": dace.pointer(dace.float64)}, code, language=dace.Language.CPP)
    table = state.add_access("table")
    state.add_edge(tasklet, "_table", table, None, dace.Memlet("table[0]"))

    inner = dace.SDFG("read_table")
    inner.add_scalar("w", dace.pointer(dace.float64))
    inner.add_array("out", [n], dace.float64, storage)
    inner.add_symbol("i", dace.int64)
    inner_state = inner.add_state()
    read = inner_state.add_tasklet(
        "read", {"_w": dace.pointer(dace.float64)}, {"_y"}, "_y = 2 * _w[i];", language=dace.Language.CPP
    )
    inner_state.add_edge(inner_state.add_read("w"), None, read, "_w", dace.Memlet("w[0]"))
    inner_state.add_edge(read, "_y", inner_state.add_write("out"), None, dace.Memlet("out[i]"))

    schedule = dace.ScheduleType.GPU_Device if gpu else dace.ScheduleType.Default
    entry, exit_ = state.add_map("m", {"i": f"0:{n}"}, schedule)
    nsdfg = state.add_nested_sdfg(inner, {"w"}, {"out"}, {"i": "i"})
    state.add_memlet_path(table, entry, nsdfg, dst_conn="w", memlet=dace.Memlet("table[0]"))
    state.add_memlet_path(nsdfg, exit_, state.add_write("y"), src_conn="out", memlet=dace.Memlet("y[i]"))
    return sdfg


def test_pointer_scalar_into_nested_sdfg():
    y = np.zeros(16)
    _pointer_program(False)(y=y)
    assert np.allclose(y, 2 * np.arange(16))


@pytest.mark.gpu
def test_pointer_scalar_into_gpu_kernel():
    import cupy as cp  # CuPy is only installed on GPU machines

    y = cp.zeros(16)
    _pointer_program(True)(y=y)
    assert np.allclose(y.get(), 2 * np.arange(16))


if __name__ == "__main__":
    test_readonly_input_is_const_and_written_output_is_not()
    test_inout_array_is_not_const()
    test_a_read_view_of_a_const_input_is_emitted_const()
    test_mutated_descriptors_follows_view_direction()
    test_const_qualify_pointer_data()
    test_pointer_scalar_into_nested_sdfg()
    test_pointer_scalar_into_gpu_kernel()
