# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scal, Ger, Symv, Symm and Syrk against numpy, in C and Fortran layout. The triangle of a symmetric operand that
the node must not read holds junk."""

import numpy as np
import pytest

import dace
from dace.codegen.common import get_gpu_backend
from dace.libraries.blas.nodes import Ger, Scal, Symm, Symv, Syrk

IMPLEMENTATIONS = [
    "pure",
    pytest.param("OpenBLAS", marks=pytest.mark.lapack),
    # The vendor library of this build's GPU backend, resolved only when a GPU test runs.
    pytest.param("gpu", marks=pytest.mark.gpu),
]
rng = np.random.default_rng(42)


def array(sdfg: dace.SDFG, name: str, value: np.ndarray, storage: dace.StorageType):
    """Adds ``value``'s container to ``sdfg`` with ``value``'s layout and data type."""
    strides = [s // value.itemsize for s in value.strides]
    sdfg.add_array(name, list(value.shape), dace.dtype_to_typeclass(value.dtype.type), storage, strides=strides)


def run(
    name: str,
    node,
    implementation: str,
    inputs: dict[str, tuple[str, np.ndarray]],
    outputs: dict[str, tuple[str, np.ndarray]],
):
    """Connects ``node`` to one array per connector and runs it, on the GPU for a GPU implementation; writes the
    results back into the ``outputs`` arrays."""
    if implementation == "gpu":
        implementation = "hipBLAS" if get_gpu_backend() == "hip" else "cuBLAS"
    sdfg = build(name, node, implementation, inputs, outputs)
    if implementation not in GPU_IMPLEMENTATIONS:
        sdfg(**dict({**inputs, **outputs}.values()))
        return
    cupy = pytest.importorskip("cupy")
    device = {
        array_name: cupy.asarray(value, order="F" if value.flags.f_contiguous and value.ndim > 1 else "C")
        for array_name, value in {**inputs, **outputs}.values()
    }
    sdfg(**device)
    for array_name, value in outputs.values():
        value[...] = device[array_name].get()


GPU_IMPLEMENTATIONS = ("cuBLAS", "hipBLAS")


def build(
    name: str,
    node,
    implementation: str,
    inputs: dict[str, tuple[str, np.ndarray]],
    outputs: dict[str, tuple[str, np.ndarray]],
) -> dace.SDFG:
    """An SDFG with ``node`` connected to one array per connector, in GPU memory for a GPU implementation."""
    sdfg = dace.SDFG(f"{name}_{implementation}")
    state = sdfg.add_state()
    node.implementation = implementation
    storage = dace.StorageType.GPU_Global if implementation in GPU_IMPLEMENTATIONS else dace.StorageType.Default
    for connector, (array_name, value) in {**inputs, **outputs}.items():
        if array_name not in sdfg.arrays:
            array(sdfg, array_name, value, storage)
    for connector, (array_name, _) in inputs.items():
        state.add_edge(
            state.add_read(array_name),
            None,
            node,
            connector,
            dace.Memlet.from_array(array_name, sdfg.arrays[array_name]),
        )
    for connector, (array_name, _) in outputs.items():
        state.add_edge(
            node,
            connector,
            state.add_write(array_name),
            None,
            dace.Memlet.from_array(array_name, sdfg.arrays[array_name]),
        )
    return sdfg


def matrix(shape, order: str) -> np.ndarray:
    return np.asarray(rng.standard_normal(shape), order=order)


def symmetric_from(a: np.ndarray, upper: bool) -> np.ndarray:
    """The symmetric matrix of which ``a`` holds the upper or the lower triangle."""
    triangle = np.triu(a) if upper else np.tril(a)
    return triangle + triangle.T - np.diag(np.diag(a))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_scal(implementation):
    x, res = rng.standard_normal(7), np.zeros(7)
    run("scal", Scal("scal", a=2.5), implementation, {"_x": ("x", x)}, {"_res": ("res", res)})
    np.testing.assert_allclose(res, 2.5 * x)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("order", ["C", "F"])
def test_ger(implementation, order):
    a, x, y = matrix((5, 4), order), rng.standard_normal(5), rng.standard_normal(4)
    res = np.zeros((5, 4), order=order)
    run(
        f"ger_{order}",
        Ger("ger", m=5, n=4, alpha=1.5),
        implementation,
        {"_A": ("A", a), "_x": ("x", x), "_y": ("y", y)},
        {"_res": ("res", res)},
    )
    np.testing.assert_allclose(res, a + 1.5 * np.outer(x, y))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("upper", [False, True])
def test_symv(implementation, order, upper):
    a, x, y = matrix((6, 6), order), rng.standard_normal(6), rng.standard_normal(6)
    out = np.zeros(6)
    node = Symv("symv", uplo=upper, alpha=2.0, beta=0.5)
    run(
        f"symv_{order}_{upper}",
        node,
        implementation,
        {"_A": ("A", a), "_x": ("x", x), "_yin": ("y", y)},
        {"_yout": ("out", out)},
    )
    np.testing.assert_allclose(out, 2.0 * symmetric_from(a, upper) @ x + 0.5 * y)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("right,upper", [(False, False), (True, True)])
def test_symm(implementation, order, right, upper):
    m, n = 5, 4
    a, b, c = matrix((n, n) if right else (m, m), order), matrix((m, n), order), matrix((m, n), order)
    out = np.zeros((m, n), order=order)
    node = Symm("symm", side=right, uplo=upper, alpha=2.0, beta=0.5)
    run(
        f"symm_{order}_{right}_{upper}",
        node,
        implementation,
        {"_A": ("A", a), "_B": ("B", b), "_Cin": ("C", c)},
        {"_Cout": ("out", out)},
    )
    s = symmetric_from(a, upper)
    np.testing.assert_allclose(out, 2.0 * (b @ s if right else s @ b) + 0.5 * c)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_beta_zero_does_not_read_the_accumulator(implementation):
    """As in BLAS, beta == 0 leaves the input accumulator unread: NaN there does not reach the result."""
    a, x = matrix((4, 4), "C"), rng.standard_normal(4)
    y, out = np.full(4, np.nan), np.zeros(4)
    run(
        "symv_beta0",
        Symv("symv", alpha=2.0, beta=0),
        implementation,
        {"_A": ("A", a), "_x": ("x", x), "_yin": ("y", y)},
        {"_yout": ("out", out)},
    )
    np.testing.assert_allclose(out, 2.0 * symmetric_from(a, False) @ x)

    b, c, cout = matrix((4, 3), "C"), np.full((4, 3), np.nan), np.zeros((4, 3))
    run(
        "symm_beta0",
        Symm("symm", alpha=2.0, beta=0),
        implementation,
        {"_A": ("A", a), "_B": ("B", b), "_Cin": ("C", c)},
        {"_Cout": ("out", cout)},
    )
    np.testing.assert_allclose(cout, 2.0 * symmetric_from(a, False) @ b)

    k = matrix((4, 2), "C")
    c, cout = np.full((4, 4), np.nan), np.zeros((4, 4))
    run(
        "syrk_beta0",
        Syrk("syrk", alpha=2.0, beta=0),
        implementation,
        {"_A": ("A", k), "_Cin": ("C", c)},
        {"_Cout": ("out", cout)},
    )
    lower = np.tril(np.ones((4, 4), bool))
    np.testing.assert_allclose(cout[lower], (2.0 * k @ k.T)[lower])


@pytest.mark.parametrize("implementation", ["pure", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_complex_symv(implementation):
    """Complex symv is symmetric, not Hermitian: the mirrored element is not conjugated. CBLAS has none, the GPU
    libraries do."""
    a = matrix((4, 4), "C") + 1j * matrix((4, 4), "C")
    x, y = rng.standard_normal(4) + 1j * rng.standard_normal(4), rng.standard_normal(4) + 1j * rng.standard_normal(4)
    out = np.zeros(4, dtype=np.complex128)
    run(
        "symv_complex",
        Symv("symv", alpha=2.0, beta=0.5),
        implementation,
        {"_A": ("A", a), "_x": ("x", x), "_yin": ("y", y)},
        {"_yout": ("out", out)},
    )
    np.testing.assert_allclose(out, 2.0 * symmetric_from(a, False) @ x + 0.5 * y)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("transposed,upper", [(False, False), (True, True)])
def test_syrk(implementation, order, transposed, upper):
    n, k = 5, 3
    a, c = matrix((k, n) if transposed else (n, k), order), matrix((n, n), order)
    out = np.zeros((n, n), order=order)
    node = Syrk("syrk", uplo=upper, transA=transposed, alpha=2.0, beta=0.5)
    run(
        f"syrk_{order}_{transposed}_{upper}",
        node,
        implementation,
        {"_A": ("A", a), "_Cin": ("C", c)},
        {"_Cout": ("out", out)},
    )
    full = 2.0 * (a.T @ a if transposed else a @ a.T) + 0.5 * c
    referenced = np.triu(np.ones((n, n), bool)) if upper else np.tril(np.ones((n, n), bool))
    np.testing.assert_allclose(out, np.where(referenced, full, c))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("order", ["C", "F"])
def test_complex_ger_does_not_conjugate(implementation, order):
    """``Ger`` is ``A + alpha x y^T``: the complex routine is ``?geru``, not the conjugating ``?gerc``."""
    a = np.asarray(matrix((5, 4), order) + 1j * matrix((5, 4), order), order=order)
    x, y = rng.standard_normal(5) + 1j * rng.standard_normal(5), rng.standard_normal(4) + 1j * rng.standard_normal(4)
    res = np.zeros((5, 4), complex, order=order)
    run(
        f"ger_complex_{order}",
        Ger("ger", m=5, n=4, alpha=1.5),
        implementation,
        {"_A": ("A", a), "_x": ("x", x), "_y": ("y", y)},
        {"_res": ("res", res)},
    )
    np.testing.assert_allclose(res, a + 1.5 * np.outer(x, y))


@pytest.mark.parametrize(
    "node,inputs,routine,arguments",
    [
        (
            Symm("symm", side=False, uplo=False),
            {"_A": (4, 4), "_B": (4, 3), "_Cin": (4, 3)},
            "hipblasDsymm",
            "HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_UPPER, 3, 4",
        ),
        (
            Syrk("syrk", uplo=False, transA=False),
            {"_A": (4, 2), "_Cin": (4, 4)},
            "hipblasDsyrk",
            "HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_T, 4, 2",
        ),
        (
            Symv("symv", uplo=True),
            {"_A": (4, 4), "_x": (4,), "_yin": (4,)},
            "hipblasDsymv",
            "HIPBLAS_FILL_MODE_LOWER, 4",
        ),
        (Ger("ger", m=4, n=3), {"_A": (4, 3), "_x": (4,), "_y": (3,)}, "hipblasDger", "3, 4"),
    ],
)
def test_hipblas_call_of_a_row_major_operand(node, inputs, routine, arguments):
    """hipBLAS is column-major, so a row-major operand reaches it transposed: the side, the triangle and the
    transposition flip and the dimensions swap. No ROCm device is needed to read the generated call."""
    output = {Symm: "_Cout", Syrk: "_Cout", Symv: "_yout", Ger: "_res"}[type(node)]
    output_shape = inputs.get("_Cin", inputs.get("_yin", inputs.get("_A")))
    values = {c: (c.strip("_").lower(), np.zeros(shape)) for c, shape in inputs.items()}
    sdfg = build(type(node).__name__, node, "hipBLAS", values, {output: ("out", np.zeros(output_shape))})
    sdfg.expand_library_nodes()
    code = "".join(t.code.as_string for t, _ in sdfg.all_nodes_recursive() if isinstance(t, dace.nodes.Tasklet))
    call = code[code.index(f"{routine}(") :]
    assert call.startswith(f"{routine}(__dace_hipblas_handle, {arguments}"), call[:120]


VENDOR_NODES = {
    "scal": (lambda: Scal("scal"), {"_x": (4,)}, ("_res", (4,))),
    "ger": (lambda: Ger("ger", m=4, n=3), {"_A": (4, 3), "_x": (4,), "_y": (3,)}, ("_res", (4, 3))),
    "symv": (lambda: Symv("symv"), {"_A": (4, 4), "_x": (4,), "_yin": (4,)}, ("_yout", (4,))),
    "symm": (lambda: Symm("symm"), {"_A": (4, 4), "_B": (4, 3), "_Cin": (4, 3)}, ("_Cout", (4, 3))),
    "syrk": (lambda: Syrk("syrk"), {"_A": (4, 2), "_Cin": (4, 4)}, ("_Cout", (4, 4))),
}
VENDOR_ROUTINES = {"OpenBLAS": "cblas_d{}(", "MKL": "cblas_d{}(", "cuBLAS": "cublasD{}(", "hipBLAS": "hipblasD{}("}


@pytest.mark.parametrize("implementation", list(VENDOR_ROUTINES))
@pytest.mark.parametrize("name", list(VENDOR_NODES))
def test_vendor_expansion_calls_the_library(implementation, name):
    """A vendor expansion calls its library rather than falling back to the pure loops. Expanding needs neither the
    library nor a device."""
    make, inputs, (output, shape) = VENDOR_NODES[name]
    values = {c: (c.strip("_").lower(), np.zeros(s)) for c, s in inputs.items()}
    sdfg = build(f"{name}_vendor", make(), implementation, values, {output: ("out", np.zeros(shape))})
    sdfg.expand_library_nodes()
    code = "".join(t.code.as_string for t, _ in sdfg.all_nodes_recursive() if isinstance(t, dace.nodes.Tasklet))
    assert VENDOR_ROUTINES[implementation].format(name) in code
    assert not any(isinstance(n, dace.nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())


@pytest.mark.parametrize("implementation", list(VENDOR_ROUTINES))
def test_vendor_call_prints_a_symbolic_extent_as_cpp(implementation):
    """The routine call is C++: an ``N**2`` extent prints as ``(N * N)``, not as the Python power."""
    n = dace.symbol("N")
    sdfg = dace.SDFG(f"scal_symbolic_{implementation}")
    storage = dace.StorageType.GPU_Global if implementation in GPU_IMPLEMENTATIONS else dace.StorageType.Default
    sdfg.add_array("x", [n**2], dace.float64, storage)
    sdfg.add_array("res", [n**2], dace.float64, storage)
    state = sdfg.add_state()
    node = Scal("scal", a=2.0)
    node.implementation = implementation
    state.add_edge(state.add_read("x"), None, node, "_x", dace.Memlet("x"))
    state.add_edge(node, "_res", state.add_write("res"), None, dace.Memlet("res"))
    sdfg.expand_library_nodes()
    code = "".join(t.code.as_string for t, _ in sdfg.all_nodes_recursive() if isinstance(t, dace.nodes.Tasklet))
    call = code[code.index(VENDOR_ROUTINES[implementation].format("scal")) :]
    assert "(N * N), " in call and "**" not in call, call


def copy_states(sdfg: dace.SDFG) -> list[str]:
    return [
        n.label for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.SDFGState) and n.label.endswith("_copy")
    ]


@pytest.mark.parametrize("implementation", list(VENDOR_ROUTINES))
def test_in_place_update_does_not_copy_the_operand(implementation):
    """``A = A + x y^T`` and ``x = a x`` hand the vendor routine the operand itself; an out-of-place call still copies
    it first. Expanding needs neither the library nor a device."""
    a, x, y = np.zeros((4, 3)), np.zeros(4), np.zeros(3)
    ger_in = {"_A": ("A", a), "_x": ("x", x), "_y": ("y", y)}
    in_place = build("ger_in_place", Ger("ger"), implementation, ger_in, {"_res": ("A", a)})
    out_of_place = build("ger_out_of_place", Ger("ger"), implementation, ger_in, {"_res": ("B", a)})
    scal = build("scal_in_place", Scal("scal", a=2.0), implementation, {"_x": ("x", x)}, {"_res": ("x", x)})
    for sdfg in (in_place, out_of_place, scal):
        sdfg.expand_library_nodes()
    assert copy_states(in_place) == [] and copy_states(scal) == []
    assert copy_states(out_of_place) == ["ger_copy"]


@pytest.mark.parametrize("implementation", list(VENDOR_ROUTINES))
@pytest.mark.parametrize("name", ["symv", "symm"])
def test_beta_zero_does_not_copy_the_accumulator(implementation, name):
    """With beta == 0 the vendor routine overwrites the accumulator, so it is not copied first; any other beta reads
    it. Expanding needs neither the library nor a device."""
    make, inputs, (output, shape) = VENDOR_NODES[name]
    labels = {}
    for beta in (0, 0.5):
        node = make()
        node.beta = beta
        values = {c: (c.strip("_").lower(), np.zeros(s)) for c, s in inputs.items()}
        sdfg = build(f"{name}_beta", node, implementation, values, {output: ("out", np.zeros(shape))})
        sdfg.expand_library_nodes()
        labels[beta] = copy_states(sdfg)
    assert labels == {0: [], 0.5: [f"{name}_copy"]}


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_in_place_ger_and_scal(implementation):
    a, x, y = matrix((5, 3), "C"), rng.standard_normal(5), rng.standard_normal(3)
    expected = a + 2.0 * np.outer(x, y)
    run(
        "ger_alias",
        Ger("ger", alpha=2.0),
        implementation,
        {"_A": ("A", a), "_x": ("x", x), "_y": ("y", y)},
        {"_res": ("A", a)},
    )
    np.testing.assert_allclose(a, expected)
    expected = 2.5 * x
    run("scal_alias", Scal("scal", a=2.5), implementation, {"_x": ("x", x)}, {"_res": ("x", x)})
    np.testing.assert_allclose(x, expected)


@pytest.mark.parametrize("implementation", GPU_IMPLEMENTATIONS)
def test_gpu_library_rejects_a_matrix_without_a_contiguous_dimension(implementation):
    """The vendor libraries take row- or column-major matrices only; a matrix strided in both dimensions is an error
    at expansion, not a call that reads the wrong elements. Expanding needs no device."""
    strided = np.zeros((8, 8))[::2, ::2]
    inputs = {"_A": ("a", strided), "_B": ("b", np.zeros((4, 3))), "_Cin": ("cin", np.zeros((4, 3)))}
    sdfg = build("symm_strided", Symm("symm"), implementation, inputs, {"_Cout": ("cout", np.zeros((4, 3)))})
    with pytest.raises(ValueError, match="row- or column-major"):
        sdfg.expand_library_nodes()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("shape", [(4, 1), (1, 4)])
def test_ger_of_a_single_row_or_column(implementation, order, shape):
    """Squeezing a genuine ``(N, 1)`` matrix rejected a valid BLAS call; a one-element vector has no stride either."""
    a, x, y = matrix(shape, order), rng.standard_normal(shape[0]), rng.standard_normal(shape[1])
    res = np.zeros(shape, order=order)
    connectors = {"_A": ("A", a), "_x": ("x", x), "_y": ("y", y)}
    run(f"ger_unit_{shape[0]}_{order}", Ger("ger", alpha=1.5), implementation, connectors, {"_res": ("res", res)})
    np.testing.assert_allclose(res, a + 1.5 * np.outer(x, y))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("order", ["C", "F"])
def test_symm_of_a_single_column_next_to_a_full_matrix(implementation, order):
    """A packed column's stride is free: as the leading dimension of a column-major call it must span the rows."""
    a = symmetric_from(matrix((4, 4), order), upper=False)
    a = np.asarray(a, order=order)
    b, cin = rng.standard_normal((4, 1)), rng.standard_normal((4, 1))
    cout = np.zeros((4, 1))
    connectors = {"_A": ("A", a), "_B": ("B", b), "_Cin": ("Cin", cin)}
    node = Symm("symm", alpha=1.5, beta=0.5)
    run(f"symm_unit_{order}", node, implementation, connectors, {"_Cout": ("Cout", cout)})
    np.testing.assert_allclose(cout, 1.5 * a @ b + 0.5 * cin)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_syrk_of_a_single_column(implementation):
    a, cin = rng.standard_normal((4, 1)), rng.standard_normal((4, 4))
    cout = np.zeros((4, 4))
    connectors = {"_A": ("A", a), "_Cin": ("Cin", cin)}
    run("syrk_unit", Syrk("syrk", alpha=1.5, beta=0.5), implementation, connectors, {"_Cout": ("Cout", cout)})
    np.testing.assert_allclose(np.tril(cout), np.tril(1.5 * a @ a.T + 0.5 * cin))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_symv_of_one_element(implementation):
    a, x, yin, yout = np.array([[3.0]]), np.array([2.0]), np.array([1.0]), np.zeros(1)
    connectors = {"_A": ("A", a), "_x": ("x", x), "_yin": ("yin", yin)}
    node = Symv("symv", alpha=1.5, beta=0.5)
    run("symv_unit", node, implementation, connectors, {"_yout": ("yout", yout)})
    np.testing.assert_allclose(yout, 1.5 * a @ x + 0.5 * yin)


SCALE = dace.symbol("scale", dace.float64)
SYMBOLIC_SCALES = {
    "scal": (lambda: Scal("scal", a=SCALE), {"_x": (7,)}, "_res", (7,)),
    "ger": (lambda: Ger("ger", alpha=SCALE), {"_A": (4, 3), "_x": (4,), "_y": (3,)}, "_res", (4, 3)),
    "symv": (lambda: Symv("symv", alpha=SCALE), {"_A": (4, 4), "_x": (4,), "_yin": (4,)}, "_yout", (4,)),
    "symm": (lambda: Symm("symm", alpha=SCALE), {"_A": (4, 4), "_B": (4, 3), "_Cin": (4, 3)}, "_Cout", (4, 3)),
    "syrk": (lambda: Syrk("syrk", alpha=SCALE), {"_A": (4, 3), "_Cin": (4, 4)}, "_Cout", (4, 4)),
}


@pytest.mark.parametrize("implementation", ["pure", pytest.param("OpenBLAS", marks=pytest.mark.lapack)])
@pytest.mark.parametrize("name", list(SYMBOLIC_SCALES))
def test_a_symbolic_scale_reaches_the_nested_call(implementation, name):
    """The vendor scalars are C++ text in a nested SDFG, whose code no symbol inference reads."""
    make, inputs, output, shape = SYMBOLIC_SCALES[name]
    arrays = {connector: (connector[1:], rng.standard_normal(extents)) for connector, extents in inputs.items()}
    result = np.zeros(shape)
    sdfg = build(f"symbolic_scale_{name}", make(), implementation, arrays, {output: (output[1:], result)})
    sdfg.add_symbol("scale", dace.float64)

    sdfg(scale=0.0, **dict([*arrays.values(), (output[1:], result)]))
    baseline = result.copy()
    sdfg(scale=2.0, **dict([*arrays.values(), (output[1:], result)]))

    assert not np.allclose(result, baseline)


def symbolic_symv(name: str, rows, cols, length) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    sdfg.add_array("A", [rows, cols], dace.float64)
    for vector in ("x", "yin", "yout"):
        sdfg.add_array(vector, [length], dace.float64)
    node = Symv("symv", alpha=1.0, beta=0.0)
    node.implementation = "pure"
    for connector, array_name in (("_A", "A"), ("_x", "x"), ("_yin", "yin")):
        memlet = dace.Memlet.from_array(array_name, sdfg.arrays[array_name])
        state.add_edge(state.add_read(array_name), None, node, connector, memlet)
    state.add_edge(node, "_yout", state.add_write("yout"), None, dace.Memlet("yout"))
    return sdfg


def test_extents_equal_only_at_run_time_warn():
    """``a(:,:)`` and ``x(:)`` differ as symbols and agree at run time, which Gemm already allows."""
    sdfg = symbolic_symv("unproved_extents", "R", "C", "L")
    with pytest.warns(UserWarning, match="may not match"):
        sdfg.validate()
    a, x = symmetric_from(matrix((4, 4), "C"), upper=False), rng.standard_normal(4)
    yout = np.zeros(4)
    sdfg(A=a, x=x, yin=np.zeros(4), yout=yout, R=4, C=4, L=4)
    np.testing.assert_allclose(yout, a @ x)


def test_extents_proved_different_are_rejected():
    with pytest.raises(dace.sdfg.InvalidSDFGNodeError, match="square matrix"):
        symbolic_symv("different_extents", 4, 5, 4).validate()


@pytest.mark.parametrize("order", ["C", "F"])
def test_pure_ger_updates_only_its_subset(order):
    """``dger`` with M below LDA lowers to a subset, which the pure expansion refused."""
    a, x, y = matrix((10, 4), order), rng.standard_normal(3), rng.standard_normal(2)
    res = a.copy(order="K")
    connectors = {"_A": ("A", a), "_x": ("x", x), "_y": ("y", y)}
    sdfg = build(f"ger_subset_{order}", Ger("ger", alpha=2.0), "pure", connectors, {"_res": ("res", res)})
    for edge in sdfg.start_block.edges():
        if edge.data.data in ("A", "res"):
            edge.data = dace.Memlet(f"{edge.data.data}[0:3, 0:2]")

    sdfg(A=a, x=x, y=y, res=res)

    expected = a.copy()
    expected[:3, :2] += 2.0 * np.outer(x, y)
    np.testing.assert_allclose(res, expected)


if __name__ == "__main__":
    for implementation in ("pure", "OpenBLAS"):
        test_scal(implementation)
        for order in ("C", "F"):
            test_ger(implementation, order)
            for upper in (False, True):
                test_symv(implementation, order, upper)
            for flag, upper in ((False, False), (True, True)):
                test_symm(implementation, order, flag, upper)
                test_syrk(implementation, order, flag, upper)
            test_ger_of_a_single_row_or_column(implementation, order, (4, 1))
            test_ger_of_a_single_row_or_column(implementation, order, (1, 4))
            test_symm_of_a_single_column_next_to_a_full_matrix(implementation, order)
        test_syrk_of_a_single_column(implementation)
        test_symv_of_one_element(implementation)
        for name in SYMBOLIC_SCALES:
            test_a_symbolic_scale_reaches_the_nested_call(implementation, name)
    test_extents_equal_only_at_run_time_warn()
    test_extents_proved_different_are_rejected()
    test_pure_ger_updates_only_its_subset("C")
    test_pure_ger_updates_only_its_subset("F")
    for implementation in VENDOR_ROUTINES:
        for name in VENDOR_NODES:
            test_vendor_expansion_calls_the_library(implementation, name)
        test_vendor_call_prints_a_symbolic_extent_as_cpp(implementation)
    for implementation in GPU_IMPLEMENTATIONS:
        test_gpu_library_rejects_a_matrix_without_a_contiguous_dimension(implementation)
