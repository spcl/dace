# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A nested SDFG in a ``GPU_Device`` map becomes a ``DACE_DFI`` device function whose scalar and array parameters
are ``const`` exactly when the connector is read-only (:func:`dace.codegen.targets.cpp.emit_memlet_reference`).
Codegen only."""

import re

import pytest

import dace
from dace import dtypes
from dace.codegen.targets.cpu import CPUCodeGen

GPU_GLOBAL = dtypes.StorageType.GPU_Global
GPU_DEVICE = dtypes.ScheduleType.GPU_Device


def _device_function_code(inner: dace.SDFG, in_conns, out_conns, wirings, impl="experimental") -> str:
    """The code of ``GPU_Device map -> inner``; ``wirings`` are ``(connector, outer_name, outer_shape, subset, is_input)``."""
    sdfg = dace.SDFG("outer_" + inner.name)
    declared = {}
    for _conn, oname, oshape, _sub, _isin in wirings:
        if oname not in declared:
            if oshape is None:
                sdfg.add_scalar(oname, dace.float64, storage=GPU_GLOBAL)
            else:
                sdfg.add_array(oname, oshape, dace.float64, storage=GPU_GLOBAL)
            declared[oname] = oshape
    state = sdfg.add_state("main")
    entry, exit_ = state.add_map("kmap", dict(i="0:16"), schedule=GPU_DEVICE)
    nsdfg = state.add_nested_sdfg(inner, dict.fromkeys(in_conns), dict.fromkeys(out_conns))
    for conn, oname, _oshape, sub, is_input in wirings:
        access = state.add_access(oname)
        if is_input:
            state.add_memlet_path(access, entry, nsdfg, dst_conn=conn, memlet=dace.Memlet(data=oname, subset=sub))
        else:
            state.add_memlet_path(nsdfg, exit_, access, src_conn=conn, memlet=dace.Memlet(data=oname, subset=sub))

    with dace.config.set_temporary("compiler", "cuda", "implementation", value=impl):
        return "\n".join(o.code for o in sdfg.generate_code())


def _device_function_signature(inner: dace.SDFG, in_conns, out_conns, wirings, impl="experimental") -> str:
    """Build ``GPU_Device map -> inner`` and return the emitted ``DACE_DFI`` parameter list."""
    code = _device_function_code(inner, in_conns, out_conns, wirings, impl=impl)
    match = re.search(r"DACE_DFI void %s\w*\(([^)]*)\)" % re.escape(inner.name), code)
    assert match is not None, f"no DACE_DFI device function emitted for {inner.name}\n{code[:2000]}"
    return match.group(1)


def _param(signature: str, connector: str) -> str:
    """The parameter substring for ``connector`` from a comma-separated signature."""
    for part in signature.split(","):
        if re.search(r"\b%s\b" % re.escape(connector), part):
            return part.strip()
    raise AssertionError(f"connector {connector!r} not found in signature: {signature!r}")


def _inner_copy(name: str, in_name: str, out_name: str) -> dace.SDFG:
    """Inner SDFG: elementwise ``out[i, j] = in[i, j]`` over row ``i`` of a 16x8 array (in read-only, out written)."""
    g = dace.SDFG(name)
    g.add_symbol("i", dace.int64)
    g.add_array(in_name, (16, 8), dace.float64)
    g.add_array(out_name, (16, 8), dace.float64)
    s = g.add_state("s")
    s.add_mapped_tasklet(
        "cp",
        dict(j="0:8"),
        {"x": dace.Memlet(f"{in_name}[i, j]")},
        "y = x",
        {"y": dace.Memlet(f"{out_name}[i, j]")},
        external_edges=True,
    )
    return g


@pytest.mark.parametrize("impl", ["experimental", "legacy"])
def test_readonly_array_input_is_const(impl):
    """A read-only array input becomes ``const T*`` (the previously-missing case)."""
    inner = _inner_copy("roarr", "a", "b")
    sig = _device_function_signature(
        inner, {"a"}, {"b"}, [("a", "A", (16, 8), "i,0:8", True), ("b", "B", (16, 8), "i,0:8", False)], impl=impl
    )
    assert _param(sig, "a").startswith("const "), _param(sig, "a")


@pytest.mark.parametrize("impl", ["experimental", "legacy"])
def test_written_array_output_is_not_const(impl):
    """A written array output must NOT be const (no over-qualification)."""
    inner = _inner_copy("woarr", "a", "b")
    sig = _device_function_signature(
        inner, {"a"}, {"b"}, [("a", "A", (16, 8), "i,0:8", True), ("b", "B", (16, 8), "i,0:8", False)], impl=impl
    )
    assert not _param(sig, "b").startswith("const "), _param(sig, "b")


@pytest.mark.parametrize("impl", ["experimental", "legacy"])
def test_readonly_scalar_input_is_const(impl):
    """A read-only scalar input is ``const`` (scalar reference or by-value)."""
    g = dace.SDFG("roscalar")
    g.add_symbol("i", dace.int64)
    g.add_array("a", (16, 8), dace.float64)
    g.add_array("b", (16, 8), dace.float64)
    g.add_scalar("c", dace.float64)
    s = g.add_state("s")
    s.add_mapped_tasklet(
        "cp",
        dict(j="0:8"),
        {"x": dace.Memlet("a[i, j]"), "sc": dace.Memlet("c[0]")},
        "y = x + sc",
        {"y": dace.Memlet("b[i, j]")},
        external_edges=True,
    )
    sig = _device_function_signature(
        g,
        {"a", "c"},
        {"b"},
        [("a", "A", (16, 8), "i,0:8", True), ("c", "C", None, "0", True), ("b", "B", (16, 8), "i,0:8", False)],
    )
    assert "const" in _param(sig, "c"), _param(sig, "c")


@pytest.mark.parametrize("impl", ["experimental", "legacy"])
def test_inout_array_is_not_const(impl):
    """An in/out (read-modify-write) array connector must NOT be const."""
    g = dace.SDFG("inout")
    g.add_symbol("i", dace.int64)
    g.add_array("d", (16, 8), dace.float64)
    s = g.add_state("s")
    s.add_mapped_tasklet(
        "inc",
        dict(j="0:8"),
        {"v": dace.Memlet("d[i, j]")},
        "w = v + 1.0",
        {"w": dace.Memlet("d[i, j]")},
        external_edges=True,
    )
    sig = _device_function_signature(
        g, {"d"}, {"d"}, [("d", "D", (16, 8), "i,0:8", True), ("d", "D", (16, 8), "i,0:8", False)], impl=impl
    )
    assert not _param(sig, "d").startswith("const "), _param(sig, "d")


# ---------------------------------------------------------------------------
# Const propagation through a View chain.
#
# C++ rule: ``const T* view = &parent[..]`` is valid whether or not the parent is const, but
# ``T* view = &const_parent[..]`` is an illegal ``const T* -> T*`` conversion. So const is monotonic
# parent -> view (a const parent forces a const view), and equivalently non-const propagates view ->
# parent (a *written* view forces a non-const parent). A read-only view never constrains its parent.
# This is the spmv regression: ``vals`` is a read-only input aliased by a read view ``_x``; the input
# must stay const AND the view must be emitted const so the chain compiles.
# ---------------------------------------------------------------------------


def _inner_with_view(name: str, write_through_view: bool) -> dace.SDFG:
    """Inner SDFG with a ``View`` ``av`` of input ``a``.

    * ``write_through_view=False``: ``av`` reads ``a`` (read-direction view), result to ``b``.
    * ``write_through_view=True``: ``a`` is written *through* ``av`` (write-direction view).
    """
    g = dace.SDFG(name)
    g.add_symbol("i", dace.int64)
    g.add_array("a", (16, 8), dace.float64)
    g.add_array("b", (16, 8), dace.float64)
    g.add_view("av", (8,), dace.float64)
    s = g.add_state("s")
    a, b, av = s.add_access("a"), s.add_access("b"), s.add_access("av")
    t = s.add_tasklet("cp", {"x": None}, {"y": None}, "y = x")
    if not write_through_view:
        s.add_edge(a, None, av, None, dace.Memlet("a[i, 0:8]"))  # read view: a -> av
        s.add_edge(av, None, t, "x", dace.Memlet("av[0]"))
        s.add_edge(t, "y", b, None, dace.Memlet("b[i, 0]"))
    else:
        s.add_edge(b, None, t, "x", dace.Memlet("b[i, 0]"))
        s.add_edge(t, "y", av, None, dace.Memlet("av[0]"))  # write view: av -> a
        s.add_edge(av, None, a, None, dace.Memlet("a[i, 0:8]"))
    return g


@pytest.mark.parametrize("impl", ["experimental", "legacy"])
def test_readonly_viewed_input_is_const_and_view_is_const(impl):
    """A read-only input aliased by a read view: the input stays ``const`` AND the view is emitted
    ``const`` (const propagates parent -> view -- the whole chain is const, the spmv regression)."""
    inner = _inner_with_view("rov", write_through_view=False)
    code = _device_function_code(
        inner, {"a"}, {"b"}, [("a", "A", (16, 8), "i,0:8", True), ("b", "B", (16, 8), "i,0:8", False)], impl=impl
    )
    sig = re.search(r"DACE_DFI void rov\w*\(([^)]*)\)", code).group(1)
    assert _param(sig, "a").startswith("const "), _param(sig, "a")
    # The view declaration must be pointer-to-const, else ``const T* -> T*`` fails to compile.
    assert re.search(r"const\s+double\s*\*\s*av\s*;", code), 'view "av" not declared pointer-to-const'
    assert not re.search(r"(?<!const )(?<!const)\bdouble\s*\*\s*av\s*;", code), 'view "av" declared non-const'


def test_mutated_descriptors_readonly_view_does_not_taint_parent():
    """``_mutated_descriptors`` must NOT include data aliased only by a *read* view (a const view of
    non-const data is valid, so the parent remains const-qualifiable)."""
    inner = _inner_with_view("rov", write_through_view=False)
    assert "a" not in CPUCodeGen._mutated_descriptors(inner)


def test_mutated_descriptors_written_view_taints_parent():
    """``_mutated_descriptors`` must include data written *through* a view (non-const propagates view
    -> parent: a written view exposes its parent through a non-const pointer)."""
    inner = _inner_with_view("wov", write_through_view=True)
    assert "a" in CPUCodeGen._mutated_descriptors(inner)


if __name__ == "__main__":
    for impl in ["experimental", "legacy"]:
        test_readonly_array_input_is_const(impl)
    for impl in ["experimental", "legacy"]:
        test_written_array_output_is_not_const(impl)
    for impl in ["experimental", "legacy"]:
        test_readonly_scalar_input_is_const(impl)
    for impl in ["experimental", "legacy"]:
        test_inout_array_is_not_const(impl)
    for impl in ["experimental", "legacy"]:
        test_readonly_viewed_input_is_const_and_view_is_const(impl)
    test_mutated_descriptors_readonly_view_does_not_taint_parent()
    test_mutated_descriptors_written_view_taints_parent()
