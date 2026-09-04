# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU (cuSOLVER / cuBLAS) lowering of the dense-linear-algebra library nodes.

These exercise the two fixes that make ``Cholesky`` (POTRF) and ``Solve`` (GETRF/GETRS) compile and
run on the GPU after ``apply_gpu_transformations``:

1. The cuSOLVER ``devInfo`` status scalar (``_info``) is placed in host-accessible pinned memory
   instead of inheriting the input matrix's ``GPU_Global`` storage. cuSOLVER writes ``devInfo``
   through a raw (unified-address) pointer, while the status is a host-checkable value -- so a
   GPU_Global ``_info`` both fails SDFG validation ("accessed on host") and is not host-readable.
2. On the GPU the frontend-generated library nodes (``implementation=None``, whose library default is
   the CPU OpenBLAS/MKL expansion) are steered to the cuSOLVER expansion through the standard
   ``set_fast_implementations`` mechanism. Without it the node lowers to ``LAPACKE_*`` host calls over
   device pointers and segfaults.

Every kernel is additionally checked under both the legacy and the experimental (readable) CPU/host
code generator, which are meant to be orthogonal to the GPU library lowering.
"""
import copy

import numpy as np
import pytest

import dace
from dace import Memlet
from dace.config import set_temporary
from dace.libraries.linalg import Cholesky, Solve
from dace.transformation.auto.auto_optimize import set_fast_implementations
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap

from tests.codegen.readable.conftest import gpu_available
from tests.corpus.npbench import npbench
from tests.corpus.polybench import polybench

pytestmark = pytest.mark.gpu

CODEGENS = ["legacy", "experimental"]


def _require_gpu():
    if not gpu_available():
        pytest.skip("No CUDA device available")


def _cpu_pipeline(sdfg):
    sdfg.simplify()
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.simplify()


def _gpu_pipeline(sdfg):
    _cpu_pipeline(sdfg)
    sdfg.apply_gpu_transformations()
    # Standard mechanism: prefer cuBLAS/cuSOLVER on GPU. CPU builds never call this, so the CPU
    # OpenBLAS/MKL/pure defaults are untouched.
    set_fast_implementations(sdfg, dace.DeviceType.GPU)


def _worst_diff(reference, candidate):
    worst = 0.0
    for key, ref in reference.items():
        cand = candidate.get(key)
        if isinstance(ref, np.ndarray) and isinstance(cand, np.ndarray):
            worst = max(worst, float(np.max(np.abs(ref.astype(np.complex128) - cand.astype(np.complex128)))))
    return worst


###############################################################################
# Direct library-node coverage: exercises the pinned ``_info`` storage fix and
# the cuSOLVER expansion selected explicitly on the node.
###############################################################################


def _spd_matrix(size, dtype):
    rng = np.random.default_rng(42)
    a = rng.random((size, size)).astype(dtype)
    return (0.5 * a @ a.T).copy()


def _make_cholesky_sdfg(dtype):
    n = dace.symbol("n", dace.int64)
    sdfg = dace.SDFG(f"cusolver_cholesky_{dtype.to_string()}")
    state = sdfg.add_state("dataflow")
    inp = sdfg.add_array("xin", [n, n], dtype)
    out = sdfg.add_array("xout", [n, n], dtype)
    node = Cholesky("cholesky", lower=True)
    node.implementation = "cuSolverDn"
    state.add_memlet_path(state.add_read("xin"), node, dst_conn="_a", memlet=Memlet.from_array(*inp))
    state.add_memlet_path(node, state.add_write("xout"), src_conn="_b", memlet=Memlet.from_array(*out))
    return sdfg


@pytest.mark.parametrize("dtype", [dace.float32, dace.float64])
def test_cholesky_libnode_gpu(dtype):
    _require_gpu()
    sdfg = _make_cholesky_sdfg(dtype)
    sdfg.apply_gpu_transformations()
    sdfg.simplify()
    compiled = sdfg.compile()

    np_dtype = getattr(np, dtype.to_string())
    size = 8
    a = _spd_matrix(size, np_dtype)
    b = np.zeros([size, size], dtype=np_dtype)
    compiled(xin=a.copy(), xout=b, n=size)

    reference = np.linalg.cholesky(a)
    rtol = 1e-6 if dtype == dace.float32 else 1e-12
    assert np.linalg.norm(reference - b) / np.linalg.norm(reference) < rtol


def _make_solve_sdfg(dtype):
    n = dace.symbol("n", dace.int64)
    sdfg = dace.SDFG(f"cusolver_solve_{dtype.to_string()}")
    sdfg.add_symbol("n", dace.int64)
    state = sdfg.add_state("dataflow")
    sdfg.add_array("ain", [n, n], dtype)
    sdfg.add_array("bin", [n, n], dtype)
    sdfg.add_array("bout", [n, n], dtype)
    node = Solve("solve")
    node.implementation = "cuSolverDn"
    state.add_memlet_path(state.add_read("ain"), node, dst_conn="_ain", memlet=Memlet.simple("ain", "0:n, 0:n"))
    state.add_memlet_path(state.add_read("bin"), node, dst_conn="_bin", memlet=Memlet.simple("bin", "0:n, 0:n"))
    state.add_memlet_path(node, state.add_write("bout"), src_conn="_bout", memlet=Memlet.simple("bout", "0:n, 0:n"))
    return sdfg


@pytest.mark.parametrize("dtype", [dace.float32, dace.float64])
def test_solve_libnode_gpu(dtype):
    _require_gpu()
    sdfg = _make_solve_sdfg(dtype)
    sdfg.apply_gpu_transformations()
    sdfg.simplify()
    compiled = sdfg.compile()

    np_dtype = getattr(np, dtype.to_string())
    size = 8
    rng = np.random.default_rng(0)
    a = rng.random((size, size)).astype(np_dtype)
    a = (a @ a.T + size * np.eye(size)).astype(np_dtype)
    b = rng.random((size, size)).astype(np_dtype)
    out = np.zeros((size, size), dtype=np_dtype)
    compiled(ain=a.copy(), bin=b.copy(), bout=out, n=size)

    reference = np.linalg.solve(a, b)
    rtol = 1e-5 if dtype == dace.float32 else 1e-11
    assert np.linalg.norm(reference - out) / np.linalg.norm(reference) < rtol


###############################################################################
# Corpus coverage: the frontend lowers ``np.linalg.cholesky`` to the Cholesky
# library node with ``implementation=None``. Verifies both the selection and the
# storage fix end-to-end, and that legacy/experimental host codegen agree with
# the CPU legacy result on identical inputs.
###############################################################################


@pytest.mark.parametrize("codegen", CODEGENS)
def test_polybench_cholesky_gpu(codegen):
    _require_gpu()
    kernel = polybench.collect("cholesky")[0]
    arrays, psize = polybench.make_inputs(kernel)

    cpu = polybench.fresh_sdfg(kernel)
    _cpu_pipeline(cpu)
    cpu_out = copy.deepcopy(arrays)
    cpu.compile()(**cpu_out, **psize)

    with set_temporary("compiler", "cpu", "implementation", value=codegen):
        gpu = polybench.fresh_sdfg(kernel)
        gpu.name = f"{gpu.name}_gpu_{codegen}"
        _gpu_pipeline(gpu)
        gpu_out = copy.deepcopy(arrays)
        gpu.compile()(**gpu_out, **psize)

    assert _worst_diff(cpu_out, gpu_out) < 1e-10


@pytest.mark.parametrize("codegen", CODEGENS)
def test_npbench_cholesky2_gpu(codegen):
    _require_gpu()
    descriptor = npbench.collect("cholesky2")[0]
    arrays, params = npbench.make_inputs(descriptor)

    cpu = npbench.fresh_sdfg(descriptor)
    _cpu_pipeline(cpu)
    cpu_out = npbench.run_outputs(descriptor, cpu, copy.deepcopy(arrays), params)

    with set_temporary("compiler", "cpu", "implementation", value=codegen):
        gpu = npbench.fresh_sdfg(descriptor)
        gpu.name = f"{gpu.name}_gpu_{codegen}"
        _gpu_pipeline(gpu)
        gpu_out = npbench.run_outputs(descriptor, gpu, copy.deepcopy(arrays), params)

    assert _worst_diff(cpu_out, gpu_out) < 1e-9


###############################################################################

if __name__ == "__main__":
    test_cholesky_libnode_gpu(dace.float64)
    test_solve_libnode_gpu(dace.float64)
    for cg in CODEGENS:
        test_polybench_cholesky_gpu(cg)
        test_npbench_cholesky2_gpu(cg)
    print("All cuSOLVER GPU tests passed.")
