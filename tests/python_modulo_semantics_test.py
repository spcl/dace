# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Modulo semantics of the Python frontend, tasklets, symbolic expressions and inter-state edges."""

import itertools

import numpy as np
import pytest
import sympy

import dace
from dace import symbolic

N = dace.symbol("N", dtype=dace.int64)
I = dace.symbol("I", dtype=dace.int64)

VALUES = (-32, -7, -3, -1, 1, 3, 7, 32)

DTYPES = (
    ("int32", dace.int32, np.int32),
    ("int64", dace.int64, np.int64),
    ("float32", dace.float32, np.float32),
    ("float64", dace.float64, np.float64),
)

TARGETS = (pytest.param("host"), pytest.param("device", marks=pytest.mark.gpu))


def modulo_program(dtype):
    """``out = a % b`` elementwise, as a map so that the GPU lowering is a real kernel."""

    @dace.program
    def prog(a: dtype[N], b: dtype[N], out: dtype[N]):
        for i in dace.map[0:N]:
            out[i] = a[i] % b[i]

    return prog


def operand_pairs(nptype):
    pairs = list(itertools.product(VALUES, VALUES))
    return (np.array([pair[0] for pair in pairs], dtype=nptype), np.array([pair[1] for pair in pairs], dtype=nptype))


@pytest.mark.parametrize("name,dtype,nptype", DTYPES)
@pytest.mark.parametrize("target", TARGETS)
def test_modulo_agrees_with_numpy(name, dtype, nptype, target):
    """``-32 % 7 == 3``, where C gives ``-4``."""
    a, b = operand_pairs(nptype)
    expected = a % b

    sdfg = modulo_program(dtype).to_sdfg()
    sdfg.name = f"division_mod_{name}_{target}"
    if target == "device":
        sdfg.apply_gpu_transformations()
    out = np.zeros(a.shape, dtype=nptype)
    sdfg(a=a, b=b, out=out, N=a.shape[0])

    disagreements = [(a[i], b[i], out[i], expected[i]) for i in range(a.shape[0]) if out[i] != expected[i]]
    assert not disagreements, f"% on {name}: {disagreements[:4]}"


def test_python_modulo_is_not_emitted_infix():
    """An infix ``%`` on floats does not compile, so the numeric table cannot catch it."""
    sdfg = modulo_program(dace.int64).to_sdfg()
    sdfg.name = "division_code_mod"
    code = sdfg.generate_code()[0].clean_code
    assert "py_mod(" in code, "% lowered without py_mod(, so it lowered infix"


@pytest.mark.parametrize(
    "text,value",
    (
        ("(-7) % 3", -1),
        ("7 % (-3)", 1),
        ("CMod(-7, 3)", -1),
        ("FtnMod(-7, 3)", -1),
        ("Mod(-7, 3)", 2),
        ("PyMod(-7, 3)", 2),
        ("PyMod(7, -3)", -2),
        ("FtnModulo(-7, 3)", 2),
    ),
)
def test_a_constant_modulo_folds_with_the_rounding_its_spelling_names(text, value):
    """``%``, ``CMod`` and ``FtnMod`` truncate; ``Mod``, ``PyMod`` and ``FtnModulo`` floor."""
    assert symbolic.pystr_to_symbolic(text) == value


def test_a_symbolic_modulo_reads_back_from_its_string_with_its_own_rounding():
    """A saved SDFG must load with the rounding it was saved with."""
    floored = symbolic.pystr_to_symbolic("PyMod(I - 2, N)")
    truncating = symbolic.pystr_to_symbolic("(I - 2) % N")

    floored_back = symbolic.pystr_to_symbolic(str(floored))
    truncating_back = symbolic.pystr_to_symbolic(str(truncating))

    assert floored_back == floored and truncating_back == truncating
    assert floored_back.subs({I: 0, N: 5}) == 3
    assert truncating_back.subs({I: 0, N: 5}) == -2
    assert "py_mod(" in symbolic.symstr(floored, cpp_mode=True)
    # C's modulo prints as ``%`` on operands proven integer and as the ``c_mod`` helper otherwise (C has no ``%`` on
    # floating point); either way it truncates.
    printed = symbolic.symstr(truncating, cpp_mode=True)
    assert "%" in printed or "c_mod(" in printed


def test_the_c_modulo_of_a_nonnegative_dividend_and_a_positive_divisor_is_sympy_mod():
    """The two roundings agree there, so simplification sees a single form."""
    # SymPy's own assumptions; a dace.symbol's sign facts live in the symbol registry, not on the sympy symbol.
    k = sympy.Symbol("k", integer=True, nonnegative=True)
    m = sympy.Symbol("m", integer=True, positive=True)

    assert isinstance(symbolic.pystr_to_symbolic("CMod(k, m)", symbol_map={"k": k, "m": m}), sympy.Mod)


def elementwise_tasklet_sdfg(name: str, code: str, dtype) -> dace.SDFG:
    """``z = <code>`` over ``a`` and ``b``, built with the SDFG API rather than a frontend."""
    sdfg = dace.SDFG(name)
    for array in ("a", "b", "out"):
        sdfg.add_array(array, [N], dtype)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("elements", dict(i="0:N"))
    tasklet = state.add_tasklet("remainder", {"x": dtype, "y": dtype}, {"z": dtype}, code)
    state.add_memlet_path(state.add_read("a"), entry, tasklet, dst_conn="x", memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(state.add_read("b"), entry, tasklet, dst_conn="y", memlet=dace.Memlet("b[i]"))
    state.add_memlet_path(tasklet, exit_node, state.add_write("out"), src_conn="z", memlet=dace.Memlet("out[i]"))
    return sdfg


INT64 = ("int64", dace.int64, np.int64)
FLOAT64 = ("float64", dace.float64, np.float64)
TASKLET_CASES = (
    ("percent", "z = x % y", INT64, np.fmod),
    ("cmod", "z = CMod(x, y)", INT64, np.fmod),
    ("cmod", "z = CMod(x, y)", FLOAT64, np.fmod),
    ("ftnmod", "z = FtnMod(x, y)", INT64, np.fmod),
    ("ftnmod", "z = FtnMod(x, y)", FLOAT64, np.fmod),
    ("pymod", "z = PyMod(x, y)", INT64, np.mod),
    ("pymod", "z = PyMod(x, y)", FLOAT64, np.mod),
    ("ftnmodulo", "z = FtnModulo(x, y)", INT64, np.mod),
    ("ftnmodulo", "z = FtnModulo(x, y)", FLOAT64, np.mod),
)


@pytest.mark.parametrize("label,code,types,reference", TASKLET_CASES)
def test_a_tasklet_modulo_computes_what_its_spelling_names(label, code, types, reference):
    """A tasklet's ``%`` is C's, whatever language its code is written in; the named functions pick the others."""
    name, dtype, nptype = types
    a, b = operand_pairs(nptype)
    sdfg = elementwise_tasklet_sdfg(f"tasklet_modulo_{label}_{name}", code, dtype)

    out = np.zeros(a.shape, dtype=nptype)
    sdfg(a=a, b=b, out=out, N=a.shape[0])

    assert np.array_equal(out, reference(a, b))


def interstate_modulo_sdfg(name: str, assignment: str) -> dace.SDFG:
    """Assigns ``k = <assignment>`` on an inter-state edge and stores ``k``."""
    sdfg = dace.SDFG(name)
    sdfg.add_symbol("I", dace.int64)
    sdfg.add_array("out", [1], dace.int64)
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state("second")
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={"k": assignment}))
    tasklet = second.add_tasklet("store", {}, {"o": dace.int64}, "o = k")
    second.add_edge(tasklet, "o", second.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


@pytest.mark.parametrize("label,assignment,value", (("percent", "(I - 7) % 3", -1), ("pymod", "PyMod(I - 7, 3)", 2)))
def test_an_interstate_edge_modulo_computes_what_its_spelling_names(label, assignment, value):
    """A symbolic ``%`` is C's too; ``PyMod`` floors."""
    sdfg = interstate_modulo_sdfg(f"interstate_modulo_{label}", assignment)

    out = np.zeros(1, dtype=np.int64)
    sdfg(out=out, I=0)

    assert out[0] == value


@dace.program
def floored_subscript(a: dace.int64[N], out: dace.int64[1]):
    out[0] = a[(I - 3) % N]


@dace.program
def floored_condition(out: dace.int64[1]):
    if (I - 3) % 4 == 1:
        out[0] = 1
    else:
        out[0] = 0


def test_a_python_program_modulo_in_a_subscript_floors():
    """``(0 - 3) % 8`` is ``5`` in Python and ``-3`` in C, an out-of-bounds read."""
    a = np.arange(8, dtype=np.int64)
    out = np.zeros(1, dtype=np.int64)

    floored_subscript(a=a, out=out, I=0, N=8)

    assert out[0] == 5


@dace.program
def called_modulos(
    a: dace.int64[N], b: dace.int64[N], python: dace.int64[N], truncated: dace.int64[N], floored: dace.int64[N]
):
    python[:] = PyMod(a, b)
    truncated[:] = FtnMod(a, b)
    floored[:] = FtnModulo(a, b)


def test_a_python_program_calls_each_modulo_spelling_like_a_builtin():
    """Fortran's ``MOD(-7, 3)`` is ``-1`` (as ``np.fmod``), ``MODULO(-7, 3)`` is ``2`` (as ``np.mod``)."""
    a, b = operand_pairs(np.int64)
    python, truncated, floored = (np.zeros(a.shape, dtype=np.int64) for unused in range(3))
    sdfg = called_modulos.to_sdfg()

    code = sdfg.generate_code()[0].clean_code
    sdfg(a=a, b=b, python=python, truncated=truncated, floored=floored, N=a.shape[0])

    assert all(f"{function}(" in code for function in ("py_mod", "ftn_mod", "ftn_modulo"))
    assert np.array_equal(python, np.mod(a, b))
    assert np.array_equal(truncated, np.fmod(a, b))
    assert np.array_equal(floored, np.mod(a, b))


@dace.program
def clamped_subscript(a: dace.int64[N], out: dace.int64[1]):
    out[0] = a[min(I, N - 1)]


def test_a_call_in_a_subscript_indexes_with_its_result():
    a = np.arange(8, dtype=np.int64)
    out = np.zeros(1, dtype=np.int64)

    clamped_subscript(a=a, out=out, I=20, N=8)

    assert out[0] == 7


def test_a_python_program_modulo_in_a_condition_floors():
    """``(0 - 3) % 4`` is ``1`` in Python and ``-3`` in C."""
    out = np.zeros(1, dtype=np.int64)

    floored_condition(out=out, I=0)

    assert out[0] == 1


@dace.program
def cast_of_a_symbolic_modulo(a: dace.float64[N], S: dace.int64):
    for i in range(N):
        a[i] = float((i - S) * 7 % N)


def test_a_cast_of_a_symbolic_modulo_floors():
    """``float(...)`` of a symbolic ``%`` takes the expression, not the call's result list, and keeps Python's
    floored rounding for a negative dividend."""
    a = np.zeros(8)

    cast_of_a_symbolic_modulo(a=a, S=5, N=8)

    np.testing.assert_array_equal(a, [float((i - 5) * 7 % 8) for i in range(8)])


if __name__ == "__main__":
    for name, dtype, nptype in DTYPES:
        test_modulo_agrees_with_numpy(name, dtype, nptype, "host")
    test_python_modulo_is_not_emitted_infix()
    for text, value in (
        ("(-7) % 3", -1),
        ("7 % (-3)", 1),
        ("CMod(-7, 3)", -1),
        ("FtnMod(-7, 3)", -1),
        ("Mod(-7, 3)", 2),
        ("PyMod(-7, 3)", 2),
        ("PyMod(7, -3)", -2),
        ("FtnModulo(-7, 3)", 2),
    ):
        test_a_constant_modulo_folds_with_the_rounding_its_spelling_names(text, value)
    test_a_symbolic_modulo_reads_back_from_its_string_with_its_own_rounding()
    for label, code, types, reference in TASKLET_CASES:
        test_a_tasklet_modulo_computes_what_its_spelling_names(label, code, types, reference)
    for label, assignment, value in (("percent", "(I - 7) % 3", -1), ("pymod", "PyMod(I - 7, 3)", 2)):
        test_an_interstate_edge_modulo_computes_what_its_spelling_names(label, assignment, value)
    test_a_python_program_modulo_in_a_subscript_floors()
    test_a_python_program_modulo_in_a_condition_floors()
    test_a_python_program_calls_each_modulo_spelling_like_a_builtin()
    test_a_call_in_a_subscript_indexes_with_its_result()
    test_a_cast_of_a_symbolic_modulo_floors()
