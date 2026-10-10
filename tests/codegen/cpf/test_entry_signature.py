# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A caller fixes the CPF entry point's signature: its parameter ORDER and a workspace pair.

CPF's own order is ``SDFG.arglist()``: arrays by name, then scalars by name. A calling convention
fixed elsewhere -- an ABI with a trailing ``uint8_t *workspace, int64_t workspace_size`` pair, a
pointer behind the scalars -- is an order no name sort reaches, so the caller states it as an
:class:`~dace.codegen.cpf.EntrySignature`. The order must land, the qualifiers must survive it, the
unit must BUILD and RUN when called that way, and a signature that does not fit is refused.
"""

import numpy as np
import pytest

import dace
from dace.codegen.cpf import EntrySignature, entry_parameter_name, render
from tests.codegen.cpf.conftest import assert_standalone, build_standalone, call_standalone, render_gpu

M = dace.symbol("M")
N = dace.symbol("N")

WORKSPACE = ("workspace", "workspace_size")


def with_workspace(*order: str) -> EntrySignature:
    return EntrySignature(order=(*order, *WORKSPACE), workspace=WORKSPACE[0], workspace_size=WORKSPACE[1])


@dace.program
def cpf_sig_scale(src: dace.float64[N], dst: dace.float64[N], factor: dace.float64):
    dst[:] = src * factor


@dace.program
def cpf_sig_rowsum(A: dace.float64[M, N], out: dace.float64[M]):
    for i in dace.map[0:M]:
        acc = 0.0
        for j in range(N):
            acc += A[i, j]
        out[i] = acc


@dace.program
def cpf_sig_axpy(x: dace.int64[N], y: dace.int64[N], a: dace.int64):
    y[:] = a * x + y


def rendered(program, name: str, language: str = "c++", signature: EntrySignature | None = None):
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    return render(sdfg, language=language, signature=signature)


def entry_parameters(code: str, name: str) -> list[str]:
    opened = code.index(f"void {name}(") + len(f"void {name}(")
    return [p.strip() for p in code[opened : code.index(")", opened)].split(",")]


def entry_names(code: str, name: str) -> tuple[str, ...]:
    return tuple(entry_parameter_name(p) for p in entry_parameters(code, name))


def test_no_signature_keeps_the_arglist_order():
    result = rendered(cpf_sig_scale, "cpf_sig_default")
    assert result.arguments == ("dst", "src", "N", "factor") == tuple(result.sdfg.arglist())
    assert entry_names(result.code, "cpf_sig_default") == result.arguments


def test_the_arglist_order_as_a_signature_is_byte_identical_to_none():
    default = rendered(cpf_sig_scale, "cpf_sig_same")
    asked = rendered(cpf_sig_scale, "cpf_sig_same", signature=EntrySignature(order=default.arguments))
    assert asked.code == default.code


@pytest.mark.parametrize("language", ["c++", "c"])
def test_the_order_lands_with_a_pointer_behind_the_scalars(language):
    order = ("dst", "N", "factor", "src")
    result = rendered(cpf_sig_scale, "cpf_sig_order", language, EntrySignature(order=order))
    assert result.arguments == order
    assert entry_names(result.code, "cpf_sig_order") == order
    assert_standalone(result.code, "cpf_sig_order", language=language)


@pytest.mark.parametrize("language", ["c++", "c"])
def test_the_workspace_pair_is_a_writable_byte_pointer_and_an_int64_size(language):
    result = rendered(cpf_sig_scale, "cpf_sig_ws", language, with_workspace("dst", "src", "N", "factor"))
    restrict = "restrict" if language == "c" else "__restrict__"
    assert entry_parameters(result.code, "cpf_sig_ws") == [
        f"double * {restrict} dst",
        f"const double * {restrict} src",
        "int N",
        "double factor",
        f"uint8_t * {restrict} workspace",
        "int64_t workspace_size",
    ]
    assert result.arguments == tuple(entry_names(result.code, "cpf_sig_ws"))
    assert set(result.sdfg.arglist()) == set(result.arguments)


CASES = {
    "scale": (cpf_sig_scale, with_workspace("dst", "N", "factor", "src")),
    "rowsum": (cpf_sig_rowsum, with_workspace("N", "out", "M", "A")),
    "axpy": (cpf_sig_axpy, EntrySignature(("a", "workspace", "y", "workspace_size", "N", "x"), *WORKSPACE)),
}


def arguments_for(case: str, rng: np.random.Generator) -> tuple[dict, dict]:
    """``(arguments, expected outputs)`` for one case, the workspace a real buffer the body ignores."""
    workspace = np.full(32, 0xA5, dtype=np.uint8)
    if case == "scale":
        src = rng.random(64)
        args = {"src": src, "dst": np.zeros(64), "N": 64, "factor": 2.5}
        expected = {"dst": src * 2.5}
    elif case == "rowsum":
        a = rng.random((7, 13))
        args = {"A": a, "out": np.zeros(7), "M": 7, "N": 13}
        expected = {"out": a.sum(axis=1)}
    else:
        x, y = rng.integers(-50, 50, 40), rng.integers(-50, 50, 40)
        args = {"x": x, "y": y.copy(), "a": -3, "N": 40}
        expected = {"y": -3 * x + y}
    args |= {"workspace": workspace, "workspace_size": workspace.size}
    return args, expected


@pytest.mark.parametrize("language", ["c++", "c"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_unit_called_in_the_asked_order_reproduces_the_numbers(case, language):
    program, signature = CASES[case]
    name = f"cpf_sig_run_{case}_{'cpp' if language == 'c++' else 'c'}"
    result = rendered(program, name, language, signature)
    assert entry_names(result.code, name) == tuple(signature.order)
    library = build_standalone(result.code, name, language=language)
    args, expected = arguments_for(case, np.random.default_rng(0))
    call_standalone(library, result.sdfg, args, order=result.arguments)
    for output, value in expected.items():
        np.testing.assert_allclose(args[output], value, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(args["workspace"], np.full(32, 0xA5, dtype=np.uint8))


def test_the_device_entry_takes_the_order_and_a_device_workspace():
    signature = with_workspace("dst", "N", "factor", "src")
    result = render_gpu(cpf_sig_scale, "cpf_sig_hip", signature=signature)
    assert entry_names(result.code, "cpf_sig_hip") == tuple(signature.order)
    assert result.sdfg.arrays["workspace"].storage is dace.StorageType.GPU_Global
    assert "uint8_t * __restrict__ workspace" in entry_parameters(result.code, "cpf_sig_hip")


def test_a_signature_that_does_not_fit_is_refused():
    with pytest.raises(ValueError, match="cpf_sig_short"):
        rendered(cpf_sig_scale, "cpf_sig_short", signature=EntrySignature(("dst", "src", "N")))
    with pytest.raises(ValueError, match="cpf_sig_extra"):
        rendered(cpf_sig_scale, "cpf_sig_extra", signature=EntrySignature(("dst", "src", "N", "factor", "nonesuch")))
    with pytest.raises(ValueError, match="cpf_sig_case"):
        rendered(cpf_sig_scale, "cpf_sig_case", signature=EntrySignature(("dst", "src", "N", "FACTOR")))
    with pytest.raises(ValueError, match="cpf_sig_twice"):
        rendered(cpf_sig_scale, "cpf_sig_twice", signature=EntrySignature(("dst", "src", "src", "N", "factor")))
    with pytest.raises(ValueError, match="cpf_sig_unnamed_ws"):
        rendered(
            cpf_sig_scale, "cpf_sig_unnamed_ws", signature=EntrySignature(("dst", "src", "N", "factor"), *WORKSPACE)
        )
    with pytest.raises(ValueError, match="together or neither"):
        EntrySignature(("dst", "src", "N", "factor", "workspace"), workspace="workspace")


def test_a_workspace_name_the_sdfg_already_uses_is_refused():
    with pytest.raises(ValueError, match=r"cpf_sig_taken: \['factor'\]"):
        rendered(
            cpf_sig_scale, "cpf_sig_taken", signature=EntrySignature(("dst", "src", "N", "factor"), "factor", "N2")
        )


if __name__ == "__main__":
    test_no_signature_keeps_the_arglist_order()
    test_the_arglist_order_as_a_signature_is_byte_identical_to_none()
    test_the_order_lands_with_a_pointer_behind_the_scalars("c++")
    test_the_order_lands_with_a_pointer_behind_the_scalars("c")
    test_the_workspace_pair_is_a_writable_byte_pointer_and_an_int64_size("c++")
    test_the_workspace_pair_is_a_writable_byte_pointer_and_an_int64_size("c")
    test_a_unit_called_in_the_asked_order_reproduces_the_numbers("axpy", "c++")
    test_a_unit_called_in_the_asked_order_reproduces_the_numbers("axpy", "c")
    test_a_unit_called_in_the_asked_order_reproduces_the_numbers("rowsum", "c++")
    test_a_unit_called_in_the_asked_order_reproduces_the_numbers("rowsum", "c")
    test_a_unit_called_in_the_asked_order_reproduces_the_numbers("scale", "c++")
    test_a_unit_called_in_the_asked_order_reproduces_the_numbers("scale", "c")
    test_the_device_entry_takes_the_order_and_a_device_workspace()
    test_a_signature_that_does_not_fit_is_refused()
    test_a_workspace_name_the_sdfg_already_uses_is_refused()
