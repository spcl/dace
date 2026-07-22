# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Library-node lowering under the experimental "readable" CPU generator.

Verifies that expanded library nodes produce numerically-equivalent results under
``compiler.cpu.implementation = experimental`` vs ``legacy``:

* CPU: BLAS ``matmul`` / ``gemm`` / ``dot`` via the ``pure`` expansion (no external
  BLAS dependency), so the readable generator lowers the expanded tasklets/maps.
* GPU (skip without a device): cuBLAS ``matmul``, and the ``CopyLibraryNode`` /
  ``MemsetLibraryNode`` standard library nodes on ``GPU_Global`` data.

For each case: build the SDFG, ``expand_library_nodes()`` under the chosen BLAS
implementation, compile + run under both generators, and compare. Same
:func:`experimental_available` gate as the corpus suite, so this file skips green
today and activates when the readable generator lands.

This is a working skeleton with concrete cases; extend ``CPU_BLAS_CASES`` (and add
``gemv`` / ``syrk`` / batched forms) as coverage grows.
"""
import numpy as np
import pytest

import dace
import dace.libraries.blas as blas
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode
from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, run_isolated,
                                             use_implementation)

N = dace.symbol("N")
#: Small square/vector size -- tiny SDFGs, fast to build and compile.
SIZE = 16


@dace.program
def blas_matmul(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    C[:] = A @ B


@dace.program
def blas_gemm(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N], D: dace.float64[N, N]):
    D[:] = A @ B + C


@dace.program
def blas_dot(x: dace.float64[N], y: dace.float64[N], out: dace.float64[1]):
    out[0] = x @ y


def matmul_inputs():
    rng = np.random.default_rng(0)
    arrays = {"A": rng.random((SIZE, SIZE)), "B": rng.random((SIZE, SIZE)), "C": np.zeros((SIZE, SIZE))}
    return arrays, {"N": SIZE}


def gemm_inputs():
    rng = np.random.default_rng(1)
    arrays = {
        "A": rng.random((SIZE, SIZE)),
        "B": rng.random((SIZE, SIZE)),
        "C": rng.random((SIZE, SIZE)),
        "D": np.zeros((SIZE, SIZE)),
    }
    return arrays, {"N": SIZE}


def dot_inputs():
    rng = np.random.default_rng(2)
    arrays = {"x": rng.random(SIZE), "y": rng.random(SIZE), "out": np.zeros(1)}
    return arrays, {"N": SIZE}


CPU_BLAS_CASES = [
    ("matmul", blas_matmul, matmul_inputs),
    ("gemm", blas_gemm, gemm_inputs),
    ("dot", blas_dot, dot_inputs),
]


def expand_and_run(program, inputs, symbols, implementation, blas_implementation, gpu):
    """Expand ``program``'s library nodes and run it under ``implementation``.

    ``blas_implementation`` selects the library expansion (``pure`` / ``cuBLAS``).
    Forks on CPU (a crashing kernel must not kill pytest); runs in-process on GPU
    (CUDA and ``os.fork`` are incompatible).
    """

    def work():
        sdfg = program.to_sdfg(simplify=True)
        if gpu:
            sdfg.apply_gpu_transformations()
        blas.default_implementation = blas_implementation
        try:
            sdfg.expand_library_nodes()
        finally:
            blas.default_implementation = None
        sdfg.name = f"{sdfg.name}_{implementation}_{'gpu' if gpu else 'cpu'}"
        work_arrays = {name: value.copy() for name, value in inputs.items()}
        sdfg.compile()(**work_arrays, **symbols)
        return work_arrays

    with use_implementation(implementation):
        if gpu:
            return work()
        return run_isolated(work)


@pytest.mark.parametrize(("label", "program", "inputs_fn"), CPU_BLAS_CASES, ids=[case[0] for case in CPU_BLAS_CASES])
def test_blas_cpu_lowering(label, program, inputs_fn, require_experimental):
    """Pure-BLAS ``matmul`` / ``gemm`` / ``dot`` lower identically under both generators."""
    inputs, symbols = inputs_fn()
    legacy = expand_and_run(program, inputs, symbols, LEGACY, "pure", gpu=False)
    experimental = expand_and_run(program, inputs, symbols, EXPERIMENTAL, "pure", gpu=False)
    assert_outputs_equivalent(legacy, experimental, "cpu", label=f"blas/{label}")


@pytest.mark.gpu
def test_blas_gpu_cublas_lowering(require_experimental, require_gpu):
    """cuBLAS ``matmul`` lowering matches between the two generators."""
    inputs, symbols = matmul_inputs()
    legacy = expand_and_run(blas_matmul, inputs, symbols, LEGACY, "cuBLAS", gpu=True)
    experimental = expand_and_run(blas_matmul, inputs, symbols, EXPERIMENTAL, "cuBLAS", gpu=True)
    assert_outputs_equivalent(legacy, experimental, "gpu", label="blas/cublas_matmul")


def build_gpu_copy_sdfg(name):
    """One-state SDFG copying a slice of ``src`` -> ``dst`` on GPU via a CopyLibraryNode."""
    sdfg = dace.SDFG(name)
    for array in ("src", "dst"):
        sdfg.add_array(array, [64], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("main")
    read, write = state.add_read("src"), state.add_write("dst")
    node = CopyLibraryNode(name="cp")
    state.add_edge(read, None, node, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("src[0:64]"))
    state.add_edge(node, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, write, None, dace.memlet.Memlet("dst[0:64]"))
    return sdfg


def build_gpu_memset_sdfg(name):
    """One-state SDFG zeroing a slice of ``B`` on GPU via a MemsetLibraryNode."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("B", [200], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("main")
    out = state.add_access("B")
    node = MemsetLibraryNode(name="mset")
    state.add_edge(node, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, out, None, dace.memlet.Memlet("B[50:100]"))
    return sdfg


def expand_and_run_gpu_libnode(build_sdfg, inputs, implementation):
    """Expand + run a directly-built GPU library-node SDFG (in-process; cupy inputs)."""
    with use_implementation(implementation):
        sdfg = build_sdfg(f"readable_libnode_{implementation}")
        sdfg.expand_library_nodes()
        work_arrays = {name: value.copy() for name, value in inputs.items()}
        sdfg.compile()(**work_arrays)
        return work_arrays


@pytest.mark.gpu
def test_copy_libnode_gpu_lowering(require_experimental, require_gpu):
    """``CopyLibraryNode`` on GPU lowers identically under both generators."""
    import cupy as cp
    inputs = {"src": cp.asarray(np.random.default_rng(0).random(64)), "dst": cp.zeros(64, dtype=cp.float64)}
    legacy = expand_and_run_gpu_libnode(build_gpu_copy_sdfg, inputs, LEGACY)
    experimental = expand_and_run_gpu_libnode(build_gpu_copy_sdfg, inputs, EXPERIMENTAL)
    assert_outputs_equivalent(legacy, experimental, "gpu", label="copy_libnode")


@pytest.mark.gpu
def test_memset_libnode_gpu_lowering(require_experimental, require_gpu):
    """``MemsetLibraryNode`` on GPU lowers identically under both generators."""
    import cupy as cp
    inputs = {"B": cp.ones(200, dtype=cp.float64)}
    legacy = expand_and_run_gpu_libnode(build_gpu_memset_sdfg, inputs, LEGACY)
    experimental = expand_and_run_gpu_libnode(build_gpu_memset_sdfg, inputs, EXPERIMENTAL)
    assert_outputs_equivalent(legacy, experimental, "gpu", label="memset_libnode")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
