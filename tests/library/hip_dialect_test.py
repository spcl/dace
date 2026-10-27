# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The HIP expansions are the CUDA ones renamed: every library call, enum and type takes its HIP name, and no CUDA
library name is left. Expanding needs neither ROCm nor a device."""

import re

import pytest

import dace
from dace.libraries.blas import Dot, Gemm, Gemv
from dace.libraries.lapack import Getrs, Potrf
from dace.libraries.linalg import Cholesky, TensorDot, Transpose

CUDA_NAMES = re.compile(
    r"\b(cublas|cusolver|cutensor|CUBLAS_|CUSOLVER_|CUTENSOR_|cuComplex|cuDoubleComplex|CUDA_R_|cudaMalloc|cudaFree)"
)


def code_of(sdfg: dace.SDFG) -> str:
    sdfg.expand_library_nodes()
    return "\n".join(t.code.as_string for t, parent in sdfg.all_nodes_recursive() if isinstance(t, dace.nodes.Tasklet))


def environments_of(sdfg: dace.SDFG) -> set[str]:
    return {
        env
        for t, parent in sdfg.all_nodes_recursive()
        if isinstance(t, (dace.nodes.CodeNode, dace.nodes.LibraryNode))
        for env in t.environments or ()
    }


def connect(node, inputs: dict, outputs: dict, implementation: str) -> dace.SDFG:
    sdfg = dace.SDFG(f"hip_{type(node).__name__.lower()}")
    state = sdfg.add_state()
    for conn, (name, shape, dtype) in {**inputs, **outputs}.items():
        if name not in sdfg.arrays:
            sdfg.add_array(name, shape, dtype, dace.StorageType.GPU_Global)
    node.implementation = implementation
    state.add_node(node)
    for conn, (name, shape, value) in inputs.items():
        state.add_edge(state.add_read(name), None, node, conn, dace.Memlet.from_array(name, sdfg.arrays[name]))
    for conn, (name, shape, value) in outputs.items():
        state.add_edge(node, conn, state.add_write(name), None, dace.Memlet.from_array(name, sdfg.arrays[name]))
    return sdfg


F64 = dace.float64
CASES = [
    (
        Gemm("gemm"),
        {"_a": ("A", [4, 4], F64), "_b": ("B", [4, 4], F64)},
        {"_c": ("C", [4, 4], F64)},
        "hipBLAS",
        ["hipblasDgemm(__dace_hipblas_handle", "HIPBLAS_OP_N"],
    ),
    (
        Gemv("gemv"),
        {"_A": ("A", [4, 4], F64), "_x": ("x", [4], F64)},
        {"_y": ("y", [4], F64)},
        "hipBLAS",
        ["hipblasDgemv(__dace_hipblas_handle"],
    ),
    (
        Dot("dot"),
        {"_x": ("x", [4], F64), "_y": ("y", [4], F64)},
        {"_result": ("r", [1], F64)},
        "hipBLAS",
        ["hipblasDdot(__dace_hipblas_handle"],
    ),
    (
        Transpose("t", dtype=F64),
        {"_inp": ("A", [4, 3], F64)},
        {"_out": ("B", [3, 4], F64)},
        "hipBLAS",
        ["hipblasDgeam(", "HIPBLAS_OP_T"],
    ),
    (
        Potrf("potrf"),
        {"_xin": ("A", [4, 4], F64)},
        {"_xout": ("A", [4, 4], F64), "_res": ("info", [1], dace.int32)},
        "hipSolverDn",
        ["hipsolverDnDpotrf(", "HIPSOLVER_FILL_MODE_", "gpuMalloc"],
    ),
    # A complex operand is handed over as the solver's own complex type
    (
        Potrf("potrf"),
        {"_xin": ("A", [4, 4], dace.complex128)},
        {"_xout": ("A", [4, 4], dace.complex128), "_res": ("info", [1], dace.int32)},
        "hipSolverDn",
        ["hipsolverDnZpotrf(", "(hipDoubleComplex*)_xin"],
    ),
    (
        Getrs("getrs"),
        {"_a": ("A", [4, 4], F64), "_rhs_in": ("B", [4, 2], F64), "_ipiv": ("p", [4], dace.int32)},
        {"_rhs_out": ("B", [4, 2], F64), "_res": ("info", [1], dace.int32)},
        "hipSolverDn",
        ["hipsolverDnDgetrs(", "HIPSOLVER_OP_N"],
    ),
    # The solver wrapper picks the HIP BLAS for the transposes around the factorization
    (
        Cholesky("cholesky"),
        {"_a": ("A", [4, 4], F64)},
        {"_b": ("B", [4, 4], F64)},
        "hipSolverDn",
        ["hipsolverDnDpotrf(", "hipblasDgeam("],
    ),
    (
        TensorDot("tdot", left_axes=[1], right_axes=[0]),
        {"_left_tensor": ("A", [3, 4], F64), "_right_tensor": ("B", [4, 5], F64)},
        {"_out_tensor": ("C", [3, 5], F64)},
        "hipTENSOR",
        ["hiptensorContract(", "HIPTENSOR_R_64F"],
    ),
]


@pytest.mark.parametrize("node,inputs,outputs,implementation,expected", CASES, ids=lambda c: getattr(c, "label", None))
def test_hip_expansion_speaks_hip(node, inputs, outputs, implementation, expected):
    sdfg = connect(node, inputs, outputs, implementation)
    code = code_of(sdfg)
    for name in expected:
        assert name in code, (name, code[:400])
    assert not CUDA_NAMES.search(code), CUDA_NAMES.search(code).group(0)
    # A CUDA environment would link the CUDA library (``-lcublas`` on a ROCm host)
    linked = environments_of(sdfg)
    assert not any(".cublas." in env or ".cusolverdn." in env or ".cutensor." in env for env in linked), linked
