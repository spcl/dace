# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Library nodes read a slice of an array through the slice's memlet, not through a View of it.

A View in front of a library node blocks fusion and widens the memlets around it to the whole array. The nodes find
the offset, leading dimension and step in the memlet subset, so the frontend wires a sliced operand to the array it
slices, and ``LoopToTranspose`` writes its window's subsets directly. A View stays where the slice is no subset of
the node's operand: a unit dimension the matrix nodes would squeeze away.
"""

import numpy as np
import pytest

import dace
from dace import data as dt
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

M, N, K, L = 5, 6, 7, 3


@dace.program
def gemm_offset(A: dace.float64[2 * M, K], B: dace.float64[K, N], C: dace.float64[M, 3 * N]):
    C[:, 2 : 2 + N] = A[3 : 3 + M, :] @ B


@dace.program
def gemm_strided_rows(A: dace.float64[2 * M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
    C[:] = A[::2, :] @ B


@dace.program
def gemm_transposed_slice(A: dace.float64[L, K, M], B: dace.float64[K, N], C: dace.float64[M, N]):
    C[:] = A[1].T @ B


@dace.program
def gemm_named_slice(A: dace.float64[L, M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
    rows = A[L - 1]
    C[:] = rows @ B


@dace.program
def gemv_slices(A: dace.float64[L, M, K], x: dace.float64[2 * K], y: dace.float64[M]):
    y[:] = A[2] @ x[K:]


@dace.program
def dot_strided(x: dace.float64[L, K], z: dace.float64[2 * K], out: dace.float64[1]):
    out[0] = x[1] @ z[0 : 2 * K : 2]


@dace.program
def transpose_strided(A: dace.float64[2 * M, K], C: dace.float64[K, M]):
    C[:] = A[1::2, :].T


@dace.program
def tensor_transpose_window(A: dace.float64[L + 1, 2 * M, K], C: dace.float64[K, L, M]):
    C[:] = np.transpose(A[1:, 0 : 2 * M : 2, :], (2, 0, 1))


@dace.program
def tensordot_strided(A: dace.float64[L, 2 * M, K], B: dace.float64[K, N], C: dace.float64[L, M, N]):
    C[:] = np.tensordot(A[:, 1::2, :], B, axes=1)


@dace.program
def inv_strided(A: dace.float64[2 * M, 2 * M], C: dace.float64[M, M]):
    C[:] = np.linalg.inv(A[::2, 1::2])


@dace.program
def solve_slices(A: dace.float64[L, M, M], b: dace.float64[L, M, N], C: dace.float64[M, N]):
    C[:] = np.linalg.solve(A[1], b[2])


@dace.program
def gemm_unit_row(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[1, N]):
    C[:] = A[2:3, :] @ B


SUBSET_PROGRAMS = [
    gemm_offset,
    gemm_strided_rows,
    gemm_transposed_slice,
    gemm_named_slice,
    gemv_slices,
    dot_strided,
    transpose_strided,
    tensor_transpose_window,
    tensordot_strided,
    inv_strided,
    solve_slices,
]


def views(sdfg: dace.SDFG) -> list[str]:
    return [
        node.data
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.AccessNode) and isinstance(node.desc(state.sdfg), dt.View)
    ]


def arguments(sdfg: dace.SDFG) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    args = {name: rng.random(desc.shape) for name, desc in sdfg.arglist().items()}
    if "A" in args and args["A"].shape[-1] == args["A"].shape[-2]:
        args["A"] += np.eye(args["A"].shape[-1]) * args["A"].shape[-1]  # well conditioned for inv and solve
    return args


def check(prog, sdfg: dace.SDFG, device: bool = False) -> None:
    args = arguments(sdfg)
    expected = {k: v.copy() for k, v in args.items()}
    prog.f(**expected)
    if device:
        import cupy

        given = {k: cupy.asarray(v) for k, v in args.items()}
        sdfg(**given)
        args = {k: cupy.asnumpy(v) for k, v in given.items()}
    else:
        sdfg(**args)
    for name in args:
        assert np.allclose(args[name], expected[name]), name


@pytest.mark.parametrize("prog", SUBSET_PROGRAMS, ids=lambda p: p.name)
def test_a_library_node_reads_the_slice_without_a_view(prog):
    """The parsed program has no View, and the default expansions read the subset right."""
    sdfg = prog.to_sdfg(simplify=True)
    assert not views(sdfg)
    assert any(isinstance(node, nodes.LibraryNode) for node, _ in sdfg.all_nodes_recursive())
    check(prog, sdfg)


@pytest.mark.parametrize("prog", SUBSET_PROGRAMS, ids=lambda p: p.name)
def test_the_canonical_form_reads_the_slice_without_a_view(prog):
    """Canonicalization adds no View, and the finalized CPU program (BLAS, LAPACK, HPTT) reads the subset right."""
    sdfg = prog.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    assert not views(sdfg)
    finalize_for_target(sdfg, "cpu", validate=True)
    check(prog, sdfg)


def test_a_unit_row_keeps_its_view():
    """``A[2:3, :]`` is a 1 x K matrix; read as a subset, MatMul would squeeze it into a vector. The View stays."""
    sdfg = gemm_unit_row.to_sdfg(simplify=True)
    assert views(sdfg)
    check(gemm_unit_row, sdfg)


@pytest.mark.gpu
@pytest.mark.parametrize("prog", [gemm_offset, gemm_strided_rows, gemv_slices, transpose_strided, tensordot_strided])
def test_the_gpu_canonical_form_reads_the_slice(prog):
    """cuBLAS / hipBLAS and the vendor tensor library read the subset right."""
    from dace.transformation.passes.canonicalize.finalize import offload_to_gpu

    sdfg = prog.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target="gpu")
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu", validate=True)
    check(prog, sdfg, device=True)
