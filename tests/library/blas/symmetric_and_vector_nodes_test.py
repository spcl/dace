# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scal, Ger, Symv, Symm and Syrk against numpy, in C and Fortran layout. The triangle of a symmetric operand that
the node must not read holds junk."""

import numpy as np
import pytest

import dace
from dace.codegen.common import get_gpu_backend
from dace.libraries.blas.nodes import Ger, Scal, Symm, Symv, Syr2k, Syrk

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
@pytest.mark.parametrize("transposed,upper", [(False, False), (True, True)])
def test_syr2k(implementation, order, transposed, upper):
    n, k = 5, 3
    shape = (k, n) if transposed else (n, k)
    a, b, c = matrix(shape, order), matrix(shape, order), matrix((n, n), order)
    out = np.zeros((n, n), order=order)
    node = Syr2k("syr2k", uplo=upper, transA=transposed, alpha=2.0, beta=0.5)
    run(
        f"syr2k_{order}_{transposed}_{upper}",
        node,
        implementation,
        {"_A": ("A", a), "_B": ("B", b), "_Cin": ("C", c)},
        {"_Cout": ("out", out)},
    )
    full = 2.0 * (a.T @ b + b.T @ a if transposed else a @ b.T + b @ a.T) + 0.5 * c
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
    code = "".join(t.code.as_string for t, parent in sdfg.all_nodes_recursive() if isinstance(t, dace.nodes.Tasklet))
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
    code = "".join(t.code.as_string for t, parent in sdfg.all_nodes_recursive() if isinstance(t, dace.nodes.Tasklet))
    assert VENDOR_ROUTINES[implementation].format(name) in code
    assert not any(isinstance(n, dace.nodes.MapEntry) for n, parent in sdfg.all_nodes_recursive())


@pytest.mark.parametrize("implementation", GPU_IMPLEMENTATIONS)
def test_gpu_library_rejects_a_matrix_without_a_contiguous_dimension(implementation):
    """The vendor libraries take row- or column-major matrices only; a matrix strided in both dimensions is an error
    at expansion, not a call that reads the wrong elements. Expanding needs no device."""
    strided = np.zeros((8, 8))[::2, ::2]
    inputs = {"_A": ("a", strided), "_B": ("b", np.zeros((4, 3))), "_Cin": ("cin", np.zeros((4, 3)))}
    sdfg = build("symm_strided", Symm("symm"), implementation, inputs, {"_Cout": ("cout", np.zeros((4, 3)))})
    with pytest.raises(ValueError, match="row- or column-major"):
        sdfg.expand_library_nodes()


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
    for implementation in VENDOR_ROUTINES:
        for name in VENDOR_NODES:
            test_vendor_expansion_calls_the_library(implementation, name)
    for implementation in GPU_IMPLEMENTATIONS:
        test_gpu_library_rejects_a_matrix_without_a_contiguous_dimension(implementation)
