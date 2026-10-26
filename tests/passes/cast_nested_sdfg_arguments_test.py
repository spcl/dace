# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``CastNestedSDFGArguments``: a nested SDFG handed a container of another dtype gets a converting copy."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.sdfg import nodes
from dace.transformation.passes.cast_nested_sdfg_arguments import CastNestedSDFGArguments

N = 32


def build(scalar_dtype: dace.typeclass, array_dtype: dace.typeclass, gpu: bool = False) -> dace.SDFG:
    """``X[i] = X[i] * S + S`` in a nested SDFG that declares ``s`` and ``x`` as ``scalar_dtype`` / ``array_dtype``,
    called from a map over a float64 ``S`` and ``X`` (on the device when ``gpu``, as in spcl/dace#2402)."""
    inner = dace.SDFG("inner")
    inner.add_scalar("s", scalar_dtype)
    inner.add_array("x", (N,), array_dtype)
    inner.add_state().add_mapped_tasklet(
        "body",
        {"i": f"0:{N}"},
        {"a": dace.Memlet("s[0]"), "b": dace.Memlet("x[i]")},
        "c = b * a + a",
        {"c": dace.Memlet("x[i]")},
        external_edges=True,
    )

    sdfg = dace.SDFG(f"cast_nsdfg_args_{scalar_dtype.to_string()}_{array_dtype.to_string()}_{int(gpu)}")
    sdfg.add_scalar("S", dace.float64)
    sdfg.add_array("X", (N,), dace.float64)
    state = sdfg.add_state()
    data = "X"
    if gpu:
        data = "G"
        sdfg.add_array("G", (N,), dace.float64, storage=dace.StorageType.GPU_Global, transient=True)
        upload, download = sdfg.add_state_before(state), sdfg.add_state_after(state)
        upload.add_nedge(upload.add_read("X"), upload.add_write("G"), dace.Memlet(f"G[0:{N}]"))
        download.add_nedge(download.add_read("G"), download.add_write("X"), dace.Memlet(f"X[0:{N}]"))
    schedule = dace.ScheduleType.GPU_Device if gpu else dace.ScheduleType.Default
    entry, exit_ = state.add_map("outer", {"j": "0:1"}, schedule=schedule)
    call = state.add_nested_sdfg(inner, {"s": None, "x": None}, {"x": None})
    state.add_memlet_path(state.add_read("S"), entry, call, dst_conn="s", memlet=dace.Memlet("S[0]"))
    state.add_memlet_path(state.add_read(data), entry, call, dst_conn="x", memlet=dace.Memlet(f"{data}[0:{N}]"))
    state.add_memlet_path(call, exit_, state.add_write(data), src_conn="x", memlet=dace.Memlet(f"{data}[0:{N}]"))
    return sdfg


def casts(sdfg: dace.SDFG) -> list[str]:
    """``source -> destination`` of every converting copy the pass inserted."""
    found = []
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, CopyLibraryNode):
            src, dst = state.in_edges(node)[0].data.data, state.out_edges(node)[0].data.data
            found.append(f"{src}->{dst}")
    return sorted(found)


def expected(scalar_dtype: dace.typeclass, array_dtype: dace.typeclass, x: np.ndarray, s: float) -> np.ndarray:
    """What the nested SDFG computes on its own dtypes, cast back to float64."""
    st, at = scalar_dtype.type, array_dtype.type
    xs = x.astype(at)
    return (xs * at(st(s)) + at(st(s))).astype(np.float64)


def test_validator_names_the_pass_for_a_dtype_only_mismatch():
    """A dtype-only mismatch is still invalid, and the error points at the pass, not at a view."""
    sdfg = build(dace.float32, dace.float64)
    with pytest.raises(dace.sdfg.InvalidSDFGError, match="CastNestedSDFGArguments"):
        sdfg.validate()


def test_scalar_input_is_cast_with_a_warning():
    """The float64 scalar reaches the float32 connector through one copy; the bytes are never reinterpreted."""
    sdfg = build(dace.float32, dace.float64)
    with pytest.warns(UserWarning, match="declares 's' as float but is passed 'S' of double"):
        assert CastNestedSDFGArguments().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert casts(sdfg) == ["S->__cast_s"]
    transient = sdfg.arrays["__cast_s"]
    assert (type(transient), transient.dtype, transient.storage) == (
        dace.data.Scalar,
        dace.float32,
        dace.StorageType.Register,
    )

    code = "".join(obj.code for obj in sdfg.generate_code())
    assert "*(float *)(&" not in code
    rng = np.random.default_rng(0)
    x, s = rng.random(N), 1.0 / 3.0
    ref = expected(dace.float32, dace.float64, x, s)
    sdfg(S=s, X=x)
    np.testing.assert_allclose(x, ref, rtol=1e-12)


def test_inout_array_is_cast_in_and_back_through_one_transient():
    """An in-out argument is cast in before the call and back after it, through the same transient."""
    sdfg = build(dace.float64, dace.float32)
    with pytest.warns(UserWarning, match="'x' as float") as record:
        assert CastNestedSDFGArguments().apply_pass(sdfg, {}) == 2
    assert len(record) == 2
    sdfg.validate()
    assert casts(sdfg) == ["X->__cast_x", "__cast_x->X"]
    call = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG) and n.label == "inner")
    state = next(s for s in sdfg.states() if call in s.nodes())
    assert {e.data.data for e in state.in_edges(call) if e.dst_conn == "x"} == {"__cast_x"}
    assert {e.data.data for e in state.out_edges(call)} == {"__cast_x"}

    rng = np.random.default_rng(1)
    x, s = rng.random(N), 0.1
    ref = expected(dace.float64, dace.float32, x, s)
    sdfg(S=s, X=x)
    np.testing.assert_allclose(x, ref, rtol=1e-6)


def test_integer_mismatch_is_cast_too():
    """An int32 connector handed an int64 container is the same reinterpretation, so it is cast as well."""
    inner = dace.SDFG("inner_int")
    inner.add_array("x", (N,), dace.int32)
    inner.add_state().add_mapped_tasklet(
        "body",
        {"i": f"0:{N}"},
        {"b": dace.Memlet("x[i]")},
        "c = b + 1",
        {"c": dace.Memlet("x[i]")},
        external_edges=True,
    )
    sdfg = dace.SDFG("cast_nsdfg_args_int")
    sdfg.add_array("X", (N,), dace.int64)
    state = sdfg.add_state()
    call = state.add_nested_sdfg(inner, {"x": None}, {"x": None})
    state.add_edge(state.add_read("X"), None, call, "x", dace.Memlet(f"X[0:{N}]"))
    state.add_edge(call, "x", state.add_write("X"), None, dace.Memlet(f"X[0:{N}]"))
    with pytest.warns(UserWarning, match="as int but is passed 'X' of int64"):
        assert CastNestedSDFGArguments().apply_pass(sdfg, {}) == 2
    assert sdfg.arrays["__cast_x"].storage == dace.StorageType.Default  # unscoped: lives where X does
    x = np.arange(N, dtype=np.int64) + 2**31  # does not fit int32: the cast wraps, as the nested SDFG sees it
    ref = (x.astype(np.int32) + 1).astype(np.int64)
    sdfg(X=x)
    np.testing.assert_array_equal(x, ref)


def test_matching_dtypes_are_left_alone(recwarn):
    """Nothing to convert: no copy, no transient, no warning."""
    sdfg = build(dace.float64, dace.float64)
    arrays = set(sdfg.arrays)
    assert CastNestedSDFGArguments().apply_pass(sdfg, {}) is None
    assert casts(sdfg) == [] and set(sdfg.arrays) == arrays
    assert not [w for w in recwarn if "Nested SDFG" in str(w.message)]


@pytest.mark.gpu
def test_gpu_kernel_scalar_is_cast_per_thread():
    """spcl/dace#2402: a kernel's float64 scalar argument handed to a float32 nested SDFG is cast in the kernel,
    into a per-thread register, instead of reinterpreted."""
    sdfg = build(dace.float32, dace.float64, gpu=True)
    with pytest.warns(UserWarning, match="'s' as float"):
        assert CastNestedSDFGArguments().apply_pass(sdfg, {}) == 1
    code = "".join(obj.code for obj in sdfg.generate_code())
    assert "*(float *)(&" not in code
    assert "__shared__ float __cast_s" not in code and "float __cast_s;" in code
    rng = np.random.default_rng(2)
    x, s = rng.random(N), 1.0 / 3.0
    ref = expected(dace.float32, dace.float64, x, s)
    sdfg(S=s, X=x)
    np.testing.assert_allclose(x, ref, rtol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__])
