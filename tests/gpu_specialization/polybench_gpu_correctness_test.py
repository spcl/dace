# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU-offloading correctness tests for npbench polybench kernels: CPU SDFG vs GPU-transformed SDFG
compared element-wise at small sizes (kernels imported from ``tests/npbench/polybench``)."""
import importlib.util
import os
from typing import Callable, Dict

import numpy as np
import pytest

pytestmark = pytest.mark.new_gpu_codegen_only

_POLYBENCH_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "npbench", "polybench")


def _kernel_module(name):
    """Load an npbench polybench kernel-test module by path (no ``sys.path`` mutation)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_POLYBENCH_DIR, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adi_test = _kernel_module("adi_test")
atax_test = _kernel_module("atax_test")
bicg_test = _kernel_module("bicg_test")
correlation_test = _kernel_module("correlation_test")
covariance_test = _kernel_module("covariance_test")
deriche_test = _kernel_module("deriche_test")
doitgen_test = _kernel_module("doitgen_test")
durbin_test = _kernel_module("durbin_test")
fdtd_2d_test = _kernel_module("fdtd_2d_test")
floyd_warshall_test = _kernel_module("floyd_warshall_test")
gemm_npbench_test = _kernel_module("gemm_npbench_test")
gemver_test = _kernel_module("gemver_test")
gesummv_test = _kernel_module("gesummv_test")
gramschmidt_test = _kernel_module("gramschmidt_test")
heat_3d_test = _kernel_module("heat_3d_test")
jacobi_1d_test = _kernel_module("jacobi_1d_test")
jacobi_2d_test = _kernel_module("jacobi_2d_test")
k2mm_test = _kernel_module("k2mm_test")
k3mm_test = _kernel_module("k3mm_test")
lu_test = _kernel_module("lu_test")
ludcmp_test = _kernel_module("ludcmp_test")
mvt_test = _kernel_module("mvt_test")
nussinov_test = _kernel_module("nussinov_test")
seidel_2d_test = _kernel_module("seidel_2d_test")
symm_test = _kernel_module("symm_test")
syr2k_test = _kernel_module("syr2k_test")
syrk_test = _kernel_module("syrk_test")
trisolv_test = _kernel_module("trisolv_test")
trmm_test = _kernel_module("trmm_test")


def _compare_arrays(cpu_args: Dict[str, np.ndarray], gpu_args: Dict[str, np.ndarray], rtol: float, atol: float):
    for name, cpu_val in cpu_args.items():
        if not isinstance(cpu_val, np.ndarray):
            continue
        np.testing.assert_allclose(gpu_args[name], cpu_val, rtol=rtol, atol=atol, err_msg=f'arg "{name}" mismatch')


def _compare_returns(cpu_ret, gpu_ret, rtol: float, atol: float):
    if cpu_ret is None:
        return
    if isinstance(cpu_ret, tuple):
        for i, (c, g) in enumerate(zip(cpu_ret, gpu_ret)):
            np.testing.assert_allclose(g, c, rtol=rtol, atol=atol, err_msg=f'return[{i}] mismatch')
    else:
        np.testing.assert_allclose(gpu_ret, cpu_ret, rtol=rtol, atol=atol, err_msg='return mismatch')


def _run_gpu_vs_cpu(kernel,
                    build_args: Callable[[], Dict[str, np.ndarray]],
                    symbols: Dict[str, int],
                    *,
                    rtol: float = 1e-10,
                    atol: float = 1e-12):
    """Run ``kernel`` on a CPU SDFG and a GPU-transformed SDFG and assert the outputs match."""
    cpu_sdfg = kernel.to_sdfg(simplify=True)
    cpu_args = build_args()
    cpu_ret = cpu_sdfg(**cpu_args, **symbols)

    gpu_sdfg = kernel.to_sdfg(simplify=True)
    gpu_sdfg.apply_gpu_transformations()
    gpu_args = build_args()
    gpu_ret = gpu_sdfg(**gpu_args, **symbols)

    _compare_arrays(cpu_args, gpu_args, rtol, atol)
    _compare_returns(cpu_ret, gpu_ret, rtol, atol)


_TSTEPS_SMALL = 3


@pytest.mark.gpu
def test_atax_gpu_matches_cpu():
    M, N = 12, 16
    A, x, _y = atax_test.init_data(M, N)
    _run_gpu_vs_cpu(atax_test.kernel, lambda: dict(A=A.copy(), x=x.copy()), dict(M=M, N=N), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_bicg_gpu_matches_cpu():
    M, N = 12, 16
    A, p, r = bicg_test.initialize(M, N)
    _run_gpu_vs_cpu(bicg_test.bicg_kernel, lambda: dict(A=A.copy(), p=p.copy(), r=r.copy()), dict(M=M, N=N))


@pytest.mark.gpu
def test_gemm_gpu_matches_cpu():
    NI, NJ, NK = 12, 14, 16
    alpha, beta, C, A, B = gemm_npbench_test.initialize(NI, NJ, NK)
    _run_gpu_vs_cpu(gemm_npbench_test.gemm_kernel,
                    lambda: dict(alpha=alpha, beta=beta, C=C.copy(), A=A.copy(), B=B.copy()), dict(NI=NI, NJ=NJ, NK=NK))


@pytest.mark.gpu
def test_k2mm_gpu_matches_cpu():
    NI, NJ, NK, NL = 8, 10, 12, 14
    alpha, beta, A, B, C, D = k2mm_test.initialize(NI, NJ, NK, NL)
    _run_gpu_vs_cpu(k2mm_test.k2mm_kernel,
                    lambda: dict(alpha=alpha, beta=beta, A=A.copy(), B=B.copy(), C=C.copy(), D=D.copy()),
                    dict(NI=NI, NJ=NJ, NK=NK, NL=NL))


@pytest.mark.gpu
def test_k3mm_gpu_matches_cpu():
    NI, NJ, NK, NL, NM = 6, 8, 10, 12, 14
    A, B, C, D = k3mm_test.initialize(NI, NJ, NK, NL, NM)
    _run_gpu_vs_cpu(k3mm_test.k3mm_kernel, lambda: dict(A=A.copy(), B=B.copy(), C=C.copy(), D=D.copy()),
                    dict(NI=NI, NJ=NJ, NK=NK, NL=NL, NM=NM))


@pytest.mark.gpu
def test_mvt_gpu_matches_cpu():
    N = 16
    x1, x2, y_1, y_2, A = mvt_test.initialize(N)
    _run_gpu_vs_cpu(mvt_test.mvt_kernel,
                    lambda: dict(x1=x1.copy(), x2=x2.copy(), y_1=y_1.copy(), y_2=y_2.copy(), A=A.copy()), dict(N=N))


@pytest.mark.gpu
def test_gesummv_gpu_matches_cpu():
    N = 16
    alpha, beta, A, B, x = gesummv_test.initialize(N)
    _run_gpu_vs_cpu(gesummv_test.gesummv_kernel,
                    lambda: dict(alpha=alpha, beta=beta, A=A.copy(), B=B.copy(), x=x.copy()), dict(N=N))


@pytest.mark.gpu
def test_gemver_gpu_matches_cpu():
    N = 16
    alpha, beta, A, u1, v1, u2, v2, w, x, y, z = gemver_test.initialize(N)
    _run_gpu_vs_cpu(
        gemver_test.gemver_kernel, lambda: dict(alpha=alpha,
                                                beta=beta,
                                                A=A.copy(),
                                                u1=u1.copy(),
                                                v1=v1.copy(),
                                                u2=u2.copy(),
                                                v2=v2.copy(),
                                                w=w.copy(),
                                                x=x.copy(),
                                                y=y.copy(),
                                                z=z.copy()), dict(N=N))


@pytest.mark.gpu
def test_syrk_gpu_matches_cpu():
    N, M = 12, 16
    alpha, beta, C, A = syrk_test.init_data(N, M)
    _run_gpu_vs_cpu(syrk_test.kernel,
                    lambda: dict(alpha=alpha, beta=beta, C=C.copy(), A=A.copy()),
                    dict(M=M, N=N),
                    rtol=1e-5,
                    atol=1e-6)


@pytest.mark.gpu
def test_syr2k_gpu_matches_cpu():
    N, M = 12, 16
    alpha, beta, C, A, B = syr2k_test.initialize(N, M)
    _run_gpu_vs_cpu(syr2k_test.syr2k_kernel, lambda: dict(alpha=alpha, beta=beta, C=C.copy(), A=A.copy(), B=B.copy()),
                    dict(M=M, N=N))


@pytest.mark.gpu
def test_symm_gpu_matches_cpu():
    M, N = 12, 16
    alpha, beta, C, A, B = symm_test.initialize(M, N)
    _run_gpu_vs_cpu(symm_test.symm_kernel, lambda: dict(alpha=alpha, beta=beta, C=C.copy(), A=A.copy(), B=B.copy()),
                    dict(M=M, N=N))


@pytest.mark.gpu
def test_trmm_gpu_matches_cpu():
    M, N = 12, 16
    alpha, A, B = trmm_test.initialize(M, N)
    _run_gpu_vs_cpu(trmm_test.trmm_kernel, lambda: dict(alpha=alpha, A=A.copy(), B=B.copy()), dict(M=M, N=N))


@pytest.mark.gpu
def test_trisolv_gpu_matches_cpu():
    N = 16
    L, x, b = trisolv_test.initialize(N)
    _run_gpu_vs_cpu(trisolv_test.trisolv_kernel, lambda: dict(L=L.copy(), x=x.copy(), b=b.copy()), dict(N=N))


@pytest.mark.gpu
def test_durbin_gpu_matches_cpu():
    N = 16
    r = durbin_test.initialize(N)
    _run_gpu_vs_cpu(durbin_test.durbin_kernel, lambda: dict(r=r.copy()), dict(N=N))


@pytest.mark.gpu
def test_lu_gpu_matches_cpu():
    N = 16
    A = lu_test.init_data(N)
    _run_gpu_vs_cpu(lu_test.lu_kernel, lambda: dict(A=A.copy()), dict(N=N), rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
def test_ludcmp_gpu_matches_cpu():
    N = 16
    A, b = ludcmp_test.initialize(N)
    _run_gpu_vs_cpu(ludcmp_test.ludcmp_kernel, lambda: dict(A=A.copy(), b=b.copy()), dict(N=N))


@pytest.mark.gpu
def test_correlation_gpu_matches_cpu():
    M, N = 12, 16
    float_n, data = correlation_test.initialize(M, N)
    _run_gpu_vs_cpu(correlation_test.correlation_kernel, lambda: dict(float_n=float_n, data=data.copy()), dict(M=M,
                                                                                                               N=N))


@pytest.mark.gpu
def test_covariance_gpu_matches_cpu():
    M, N = 12, 16
    float_n, data = covariance_test.init_data(M, N)
    _run_gpu_vs_cpu(covariance_test.covariance_kernel,
                    lambda: dict(float_n=float_n, data=data.copy()),
                    dict(M=M, N=N),
                    rtol=1e-4,
                    atol=1e-5)


@pytest.mark.gpu
def test_gramschmidt_gpu_matches_cpu():
    M, N = 14, 10
    A = gramschmidt_test.initialize(M, N)
    _run_gpu_vs_cpu(gramschmidt_test.gramschmidt_kernel, lambda: dict(A=A.copy()), dict(M=M, N=N), rtol=1e-6, atol=1e-8)


@pytest.mark.gpu
def test_doitgen_gpu_matches_cpu():
    NR, NQ, NP = 4, 6, 8
    A, C4 = doitgen_test.initialize(NR, NQ, NP)
    _run_gpu_vs_cpu(doitgen_test.doitgen_kernel, lambda: dict(A=A.copy(), C4=C4.copy()), dict(NR=NR, NQ=NQ, NP=NP))


@pytest.mark.gpu
def test_deriche_gpu_matches_cpu():
    W, H = 16, 20
    alpha, imgIn = deriche_test.initialize(W, H)
    _run_gpu_vs_cpu(deriche_test.deriche_kernel, lambda: dict(alpha=alpha, imgIn=imgIn.copy()), dict(W=W, H=H))


@pytest.mark.gpu
def test_floyd_warshall_gpu_matches_cpu():
    N = 16
    path = floyd_warshall_test.init_data(N)
    _run_gpu_vs_cpu(floyd_warshall_test.kernel, lambda: dict(path=path.copy()), dict(N=N))


@pytest.mark.gpu
def test_nussinov_gpu_matches_cpu():
    N = 16
    seq, _table = nussinov_test.init_data(N)
    _run_gpu_vs_cpu(nussinov_test.kernel, lambda: dict(seq=seq.copy()), dict(N=N))


@pytest.mark.gpu
def test_jacobi_1d_gpu_matches_cpu():
    N = 16
    A, B = jacobi_1d_test.initialize(N)
    _run_gpu_vs_cpu(jacobi_1d_test.jacobi_1d_kernel, lambda: dict(TSTEPS=_TSTEPS_SMALL, A=A.copy(), B=B.copy()),
                    dict(N=N))


@pytest.mark.gpu
def test_jacobi_2d_gpu_matches_cpu():
    N = 16
    A, B = jacobi_2d_test.init_data(N)
    _run_gpu_vs_cpu(jacobi_2d_test.kernel,
                    lambda: dict(TSTEPS=_TSTEPS_SMALL, A=A.copy(), B=B.copy()),
                    dict(N=N),
                    rtol=1e-5,
                    atol=1e-6)


@pytest.mark.gpu
def test_seidel_2d_gpu_matches_cpu():
    N = 16
    A = seidel_2d_test.initialize(N)
    _run_gpu_vs_cpu(seidel_2d_test.seidel_2d_kernel, lambda: dict(TSTEPS=_TSTEPS_SMALL, A=A.copy()), dict(N=N))


@pytest.mark.gpu
def test_heat_3d_gpu_matches_cpu():
    N = 10
    A, B = heat_3d_test.initialize(N)
    _run_gpu_vs_cpu(heat_3d_test.heat_3d_kernel, lambda: dict(TSTEPS=_TSTEPS_SMALL, A=A.copy(), B=B.copy()), dict(N=N))


@pytest.mark.gpu
def test_adi_gpu_matches_cpu():
    N = 16
    u = adi_test.initialize(N)
    _run_gpu_vs_cpu(adi_test.adi_kernel, lambda: dict(TSTEPS=_TSTEPS_SMALL, u=u.copy()), dict(N=N))


@pytest.mark.gpu
def test_fdtd_2d_gpu_matches_cpu():
    NX, NY = 12, 16
    TMAX = _TSTEPS_SMALL
    ex, ey, hz, _fict_ = fdtd_2d_test.init_data(TMAX, NX, NY)
    _run_gpu_vs_cpu(fdtd_2d_test.kernel,
                    lambda: dict(ex=ex.copy(), ey=ey.copy(), hz=hz.copy(), _fict_=_fict_.copy()),
                    dict(TMAX=TMAX, NX=NX, NY=NY),
                    rtol=1e-5,
                    atol=1e-6)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
